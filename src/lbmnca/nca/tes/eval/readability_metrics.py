"""RQ4／C5 重組可讀性的**純量測函式**（與資料格式無關）。

三個指標（清單 C5 的操作定義）：

  M1 起源局部性 `m1_locality_index`
      **最早啟動的 20% 像素**，其平均「距編輯環 E 的距離」除以畫布對角線。
      ★**愈低愈局部**（正規化距離，不是比例）。0＝最早動的像素全在編輯環上。
      同時報 `m1_spearman`＝啟動時間對距編輯距離的 Spearman 相關，
      正相關表示遠處晚啟動。
  M2 擴散有界比 `m2_within_bound_frac`
      影響半徑隨步數的成長是否落在架構上界 `_RHO_PX_PER_STEP`＝20 px/step 之內
      （＝transport 的 `disp_max_px`）。另報下游／上游擴張比。
  M3 末段殘差比 `m3_tail_residual_frac`
      最後 20% 步數（`a_t[int(0.8*n):]`）的平均逐步變化量除以全程峰值。
      →0 表示乾淨收斂；等速 crossfade ≈ 1。

`_MAX_STEPS`＝500、`_CTU_PER_STEP` 取自 `protocol.NOMINAL_DELTA_CTU_PER_STEP`。
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from lbmnca.nca.tes.inference import protocol

_CTU_PER_STEP = protocol.NOMINAL_DELTA_CTU_PER_STEP


_MAX_STEPS = 500                          # 量測 horizon（步）；CTU 標籤 = 步數 × _CTU_PER_STEP


_RHO_PX_PER_STEP = 20.0                   # 指標二架構上界斜率 px/step；預設=transport disp_max_px


def _edit_region(child_sdf: np.ndarray, parent_sdf: np.ndarray, sdf_thr: float) -> np.ndarray:
    """E = XOR(child_solid, parent_solid)，bool [H,W]。solid = SDF_norm <= sdf_thr。

    注意 E 多落於固體內（建物），流體擾動不從此起源；量測參考請用 _edit_fluid_ring。
    """
    child_solid = child_sdf <= sdf_thr
    parent_solid = parent_sdf <= sdf_thr
    return np.logical_xor(child_solid, parent_solid)


def _edit_fluid_ring(edit_solid: np.ndarray, fluid: np.ndarray, dilate_px: int) -> np.ndarray:
    """擾動源參考 = edit 幾何膨脹進流體的環（geometry 錨定、非循環）。

    物理：設計編輯改變的是建物表面；流場擾動自該表面之流體側邊界發生、向下游擴散。
    以 edit_solid 膨脹 dilate_px 後與 fluid 交集，即擾動的合法起源帶。
    """
    ring = ndimage.binary_dilation(edit_solid, iterations=int(dilate_px)) & fluid
    return ring


def _vel_mag(converter, phys: torch.Tensor) -> np.ndarray:
    """physics(norm)[1,9,H,W] → vel_mag [H,W] np（batch=1 squeeze 至 2D）。"""
    vphy = converter.toVphy(phys.float(), is_norm=True)
    arr = np.asarray(vphy["vel_mag"].detach().cpu().numpy())
    return np.squeeze(arr)   # [1,1,H,W]/[1,H,W] → [H,W]


def measure_sequence(vel_seq_iter, edit_region: np.ndarray, fluid: np.ndarray,
                     edit_centroid_x: float) -> dict:
    """吃 vel_mag 序列 iterator（含 v0），串流算三指標。

    Args:
        vel_seq_iter: yield vel_mag[H,W]（第一個為 v0=IC）。
        edit_region: bool [H,W] 編輯區 E。
        fluid: bool [H,W] 流體區。
        edit_centroid_x: E 質心 x（分上下游用）。

    Returns:
        dict：三指標曲線與純量。
    """
    from scipy.stats import spearmanr
    dist_to_E = ndimage.distance_transform_edt(~edit_region)     # [H,W] 每點到 E 距離
    H, W = fluid.shape
    diag = float(np.hypot(H, W))
    xx = np.broadcast_to(np.arange(W)[None, :], (H, W)).astype(np.float32)

    # log-spaced snapshot 步數（含末步）→ 存 Δ_t=|v_t-v0| 供啟動時間分析（省 RAM）
    n_steps = _MAX_STEPS
    snap_steps = np.unique(np.round(np.geomspace(1, n_steps, num=min(48, n_steps))).astype(int))
    snap_set = set(int(s) for s in snap_steps)

    v0 = None
    v_prev = None
    a_t = []                                   # 指標三：逐步變化幅度 ||v_t-v_{t-1}||
    snaps: list[np.ndarray] = []               # Δ_t 於 snapshot 步（[K][H,W]）
    snaps_at: list[int] = []
    v_last = None

    for t, vel in enumerate(vel_seq_iter):
        if v0 is None:
            v0 = vel; v_prev = vel; v_last = vel
            continue
        step_delta = np.abs(vel - v_prev)[fluid]
        a_t.append(float(np.sqrt((step_delta ** 2).mean())) if step_delta.size else 0.0)
        if t in snap_set:
            snaps.append(np.abs(vel - v0).astype(np.float32))
            snaps_at.append(t)
        v_prev = vel; v_last = vel

    a_t = np.asarray(a_t)
    n = len(a_t)
    # 確保末步 Δ 在 snaps（最終變化基準）
    final_change = np.abs(v_last - v0).astype(np.float32)
    if not snaps or snaps_at[-1] != n:
        snaps.append(final_change); snaps_at.append(n)
    S = np.stack(snaps, axis=0)                 # [K,H,W]
    snaps_at = np.asarray(snaps_at, dtype=np.float32)

    # ── 啟動時間：每像素首次達最終變化 50% 之 snapshot 步 ──
    eps = max(1e-9, 0.05 * float(final_change[fluid].max()) if fluid.any() else 1e-9)
    thr_map = 0.5 * final_change                        # 半程門檻
    reached = S >= thr_map[None]                        # [K,H,W]
    has_reached = reached.any(axis=0)
    first_k = reached.argmax(axis=0)                    # 首個達標 snapshot index
    activation_step = snaps_at[first_k]                # [H,W]（步）
    valid = fluid & has_reached & (final_change > eps)  # 有意義變化像素

    if valid.sum() < 10:                                # 退化保護
        m1_locality = m1_spearman = float("nan")
        within_bound = first_r = downstream_asym = float("nan")
        r_curve = []
    else:
        act = activation_step[valid]
        dE = dist_to_E[valid]
        xs_v = xx[valid]
        # 指標一：啟動時間 vs 距編輯距離 正相關 → 遠處晚啟動 → 局部起源；
        #   locality_index = 最早啟動 20% 像素之平均距編輯 / diag（越小越局部）
        m1_spearman = float(spearmanr(act, dE).correlation)
        early_q = np.quantile(act, 0.2)
        early_mask = act <= early_q
        m1_locality = float(dE[early_mask].mean() / diag) if early_mask.any() else float("nan")
        # 指標二：r(t)=啟動於 t 前之像素最大距編輯；成長曲線 vs 上界 ρt
        r_curve = []
        for k, tk in enumerate(snaps_at):
            act_by_t = valid & (activation_step <= tk)
            r_curve.append(float(dist_to_E[act_by_t].max()) if act_by_t.any() else 0.0)
        r_curve = np.asarray(r_curve)
        bound = _RHO_PX_PER_STEP * snaps_at
        within_bound = float(np.mean(r_curve <= bound))
        first_r = float(r_curve[0] / diag)              # 首 snapshot r/對角（≈1→全場即啟動）
        dn = valid & (xx > edit_centroid_x)             # 下游 +x
        up = valid & (xx < edit_centroid_x)
        dn_reach = float(dist_to_E[dn].max()) if dn.any() else 0.0
        up_reach = float(dist_to_E[up].max()) if up.any() else 0.0
        downstream_asym = (dn_reach / up_reach) if up_reach > 1e-6 else float("nan")

    # ── 指標三：末段殘餘（主）+ 單調衰減 spearman（輔，長序列不穩）──
    peak = float(a_t.max()) if n else float("nan")
    tail = a_t[int(0.8 * n):] if n else np.array([])
    tail_residual = float(tail.mean() / peak) if (peak and peak > 1e-9 and tail.size) else float("nan")
    mono = float(spearmanr(np.arange(n), a_t).correlation) if n >= 3 else float("nan")

    return {
        "metric1_origin_locality": {
            "locality_index": m1_locality, "activation_dist_spearman": m1_spearman,
            "note": "spearman>0（遠處晚啟動）+ locality_index 小 → 局部起源；crossfade 全場同時→spearman≈0",
        },
        "metric2_diffusion_boundedness": {
            "r_curve": (r_curve.tolist() if len(r_curve) else []),
            "snap_steps": snaps_at.tolist(), "bound_rho_px_per_step": _RHO_PX_PER_STEP,
            "within_bound_frac": within_bound, "first_snap_r_over_diag": first_r,
            "downstream_upstream_ratio": downstream_asym,
            "note": "r 由小漸長且守界 + ratio>1 → 定向擴散；first_snap≈1 → 全場即啟動（無結構）",
        },
        "metric3_convergence_cleanliness": {
            "a_t": a_t.tolist(), "tail_residual_frac": tail_residual, "monotonic_spearman": mono,
            "note": "tail_residual→0 → 安定收斂（主判準）；crossfade 等速→tail_residual≈1",
        },
    }


def _aggregate(results: list[dict]) -> dict:
    """跨 case 平均三指標純量（model vs crossfade 對照）。"""
    def _collect(key, sub, field):
        vals = [r[key][sub][field] for r in results
                if isinstance(r[key][sub].get(field), (int, float))
                and np.isfinite(r[key][sub][field])]
        return float(np.mean(vals)) if vals else float("nan")
    if not results:
        return {}
    agg = {}
    for label, key in [("model", "model"), ("crossfade", "baseline_crossfade")]:
        agg[label] = {
            "m1_locality_index": _collect(key, "metric1_origin_locality", "locality_index"),
            "m1_activation_dist_spearman": _collect(key, "metric1_origin_locality", "activation_dist_spearman"),
            "m2_within_bound_frac": _collect(key, "metric2_diffusion_boundedness", "within_bound_frac"),
            "m2_first_snap_r_over_diag": _collect(key, "metric2_diffusion_boundedness", "first_snap_r_over_diag"),
            "m2_downstream_upstream_ratio": _collect(key, "metric2_diffusion_boundedness", "downstream_upstream_ratio"),
            "m3_tail_residual_frac": _collect(key, "metric3_convergence_cleanliness", "tail_residual_frac"),
            "m3_monotonic_spearman": _collect(key, "metric3_convergence_cleanliness", "monotonic_spearman"),
        }
    return agg
