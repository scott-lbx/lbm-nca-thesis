"""Tier 3 流場預測效力驗證（Phase G4: 22-d, native PNG, FULL/BBOX split）。

對既有 lbm_L1_geo100_case300：
    1. 對每個 mask 計算 22 維 G_vec（E1, E2 用該 case 的 Re/U_inlet）
    2. 對每個 case 從 HDF5 抽取 12 個流場 metrics（8 全域 + 4 空間）
    3. 單變量 Pearson ρ(f_i, y_j)
    4. 多變量 OLS ablation（含 Re_aux baseline）

通過判定（每個 feature）：
    max_j |ρ(f_i, y_j)| > 0.30  或
    max_j importance_i_j > 0.05

產出（cache 檔名依 --lbm-dataset-dir 命名空間化，見 tier3_cache_paths()）：
    1. data/g_vec/tier3_features_<dataset>.npy   — [N_cases, 22] G_vec 矩陣
    2. data/g_vec/tier3_targets_<dataset>.npy    — [N_cases, K] flow metrics
    3. data/g_vec/tier3_metadata_<dataset>.json  — case 對應 (mask_id, Re, h5_path)
    4. docs/g_vec_tier3_report.md                — 綜合報告

使用方式：
    python -m lbmnca.morphospace.physics_proxy.validate_tier3 \\
        --lbm-dataset-dir dataset/CEDS_dataset/lbm_L1_geo100_case300
    python -m lbmnca.morphospace.physics_proxy.validate_tier3 \\
        --lbm-dataset-dir dataset/CEDS_dataset/lbm_L1_geo100_case300 --force
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from lbmnca.paths import REPO_ROOT

from .features import FEATURE_DIM, FEATURE_NAMES, SimConfig, compute_all
from .validate_tier1 import CACHE_DIR, dataset_cache_key, load_mask_native

# ---------------------------------------------------------------------------
# 資料路徑
# ---------------------------------------------------------------------------

# 無預設 LBM dataset：由呼叫端（CLI --lbm-dataset-dir，或 main() 的
# lbm_dataset_dir 參數）指定，避免靜默套用某台機器上的特定資料集。

# E1 = Re·L/B1, E2 = U/(1-C2) 用的 sim_cfg 預設
DEFAULT_L_CHAR_PX: float = 6.0
DEFAULT_U_INLET_LU: float = 0.04

# Re 正規化的固定 scale（讓 Re=300 → 0.3）；用於 auxiliary regressor
RE_NORM_SCALE: float = 1000.0

REPORT_PATH: Path = REPO_ROOT / "docs" / "g_vec_tier3_report.md"


def tier3_cache_paths(lbm_dataset_dir: Path) -> tuple[Path, Path, Path]:
    """依 ``--lbm-dataset-dir`` 命名空間化的 Tier 3 cache 檔案路徑。

    避免先後對兩個不同 LBM dataset 跑本腳本時，第二次靜默讀到第一次的
    stale features/targets/meta cache（見 dataset_cache_key()）。

    Returns:
        (cache_features, cache_targets, cache_meta)
    """
    key = dataset_cache_key(lbm_dataset_dir)
    return (
        CACHE_DIR / f"tier3_features_{key}.npy",
        CACHE_DIR / f"tier3_targets_{key}.npy",
        CACHE_DIR / f"tier3_metadata_{key}.json",
    )

# 通過門檻
TIER3_RHO_THRESHOLD: float = 0.30
TIER3_IMPORTANCE_THRESHOLD: float = 0.05

# Flow metric names（對應 extract_flow_metrics 的 keys，順序固定）
# 前 8 個為「全域聚合 metric」（mass-conservation dominated），後 4 個為「空間 metric」
# 用於檢驗 C4 等非對稱性 features 的真正預測力（Option B verdict）。
METRIC_NAMES: list[str] = [
    # ── 全域聚合（global aggregates）─────────────────────────────────────────
    "mean_u_x",  # 平均 x 速度 → 平均通量
    "std_u_x",  # x 速度空間標準差 → 流場異質性
    "mean_speed",  # 平均速度模 → 整體流速
    "max_speed",  # 最大速度模 → throat 加速尖值
    "p99_speed",  # p99 速度模 → 高速區強度
    "reverse_flow_frac",  # u_x < 0 的流體比例 → 回流區面積
    "mean_KE",  # 平均動能 → wake 強度代理
    "pressure_drop",  # 入口 ρ 平均 - 出口 ρ 平均 → ΔP 代理
    # ── 空間 metric（spatial pattern）— 給 C4 等對稱性 feature 公平 test ──
    "speed_asymmetry_lr",  # log(speed_left/speed_right)：C4 的 mirror metric
    "pressure_drop_imbalance",  # (前段 ΔP - 後段 ΔP) / total ΔP：壓降偏哪段
    "KE_centroid_x",  # KE 加權 x-centroid (∈[0,1])：動能集中位置
    "reverse_flow_centroid_x",  # 回流像素 x 中心 (∈[0,1])：recirculation 位置
]
# 全域聚合 metric 的索引範圍（用於報告分組）
N_GLOBAL_METRICS: int = 8


# ---------------------------------------------------------------------------
# 流場 metric 抽取
# ---------------------------------------------------------------------------


def extract_flow_metrics(h5_path: Path) -> dict[str, float]:
    """從一個 LBM HDF5 抽取 8 個流場 scalar metrics。

    9 channel MRT moments (raw lattice units)：
        ch 0=ρ, ch 1=e, ch 2=ε, ch 3=j_x, ch 5=j_y
        u_x = j_x / ρ, u_y = j_y / ρ
    static_mask[0] = binary occupancy (1=solid)。

    Args:
        h5_path: HDF5 檔案路徑。

    Returns:
        dict[metric_name → scalar] with keys = METRIC_NAMES。
    """
    with h5py.File(h5_path, "r") as f:
        mom = f["mean_moments_field"][:]  # (9, H, W) float32
        sm = f["static_mask"][:]  # (2, H, W) float32

    rho = mom[0].astype(np.float64)
    j_x = mom[3].astype(np.float64)
    j_y = mom[5].astype(np.float64)
    u_x = j_x / np.maximum(rho, 1e-8)
    u_y = j_y / np.maximum(rho, 1e-8)
    speed = np.sqrt(u_x**2 + u_y**2)

    # static_mask[0]：1=solid，0=fluid
    fluid_mask = sm[0] < 0.5
    if not fluid_mask.any():
        raise RuntimeError(f"No fluid pixels in {h5_path.name}")

    fluid_u_x = u_x[fluid_mask]
    fluid_u_y = u_y[fluid_mask]
    fluid_speed = speed[fluid_mask]

    H, W = rho.shape
    eps = 1e-8

    # ── 全域聚合 metrics ─────────────────────────────────────────────────────
    inlet_strip = rho[:, : max(20, W // 40)]
    outlet_strip = rho[:, -max(20, W // 40) :]
    pressure_drop = float(inlet_strip.mean() - outlet_strip.mean())

    # ── 空間 metrics（左右切半，匹配 C4 結構）────────────────────────────────
    mid = W // 2
    left_fluid = fluid_mask[:, :mid]
    right_fluid = fluid_mask[:, mid:]
    speed_left = speed[:, :mid]
    speed_right = speed[:, mid:]
    rho_left = rho[:, :mid]
    rho_right = rho[:, mid:]

    speed_l_mean = float(speed_left[left_fluid].mean()) if left_fluid.any() else 0.0
    speed_r_mean = float(speed_right[right_fluid].mean()) if right_fluid.any() else 0.0
    speed_asymmetry_lr = float(np.log((speed_l_mean + eps) / (speed_r_mean + eps)))

    # 壓降偏哪段：把 x 軸切 4 段 (Q1, Q2, Q3, Q4)，比較前半 ΔP vs 後半 ΔP
    Q = W // 4
    rho_q1 = float(rho[:, :Q].mean())
    rho_q2 = float(rho[:, Q : 2 * Q].mean())
    rho_q3 = float(rho[:, 2 * Q : 3 * Q].mean())
    rho_q4 = float(rho[:, 3 * Q :].mean())
    dp_first = rho_q1 - rho_q2  # 前段（Q1→Q2）ΔP
    dp_second = rho_q3 - rho_q4  # 後段（Q3→Q4）ΔP
    total_dp = abs(dp_first) + abs(dp_second) + eps
    pressure_drop_imbalance = (dp_first - dp_second) / total_dp

    # KE centroid x（normalized to [0, 1]）
    KE_field = 0.5 * (u_x**2 + u_y**2) * fluid_mask  # 非流體=0
    x_coords = np.arange(W, dtype=np.float64)[None, :]
    KE_total = float(KE_field.sum())
    if KE_total > eps:
        KE_x_weighted = float((KE_field * x_coords).sum())
        KE_centroid_x = (KE_x_weighted / KE_total) / (W - 1)
    else:
        KE_centroid_x = 0.5

    # 回流像素 x 中心（reverse flow centroid）
    backflow = (u_x < -1e-3) & fluid_mask
    if backflow.any():
        x_indices = np.argwhere(backflow)[:, 1].astype(np.float64)
        reverse_flow_centroid_x = float(x_indices.mean()) / (W - 1)
    else:
        reverse_flow_centroid_x = 0.5  # 無 backflow 時放 placeholder

    return {
        "mean_u_x": float(fluid_u_x.mean()),
        "std_u_x": float(fluid_u_x.std()),
        "mean_speed": float(fluid_speed.mean()),
        "max_speed": float(fluid_speed.max()),
        "p99_speed": float(np.percentile(fluid_speed, 99)),
        "reverse_flow_frac": float((fluid_u_x < -1e-3).mean()),
        "mean_KE": float(0.5 * (fluid_u_x**2 + fluid_u_y**2).mean()),
        "pressure_drop": pressure_drop,
        "speed_asymmetry_lr": speed_asymmetry_lr,
        "pressure_drop_imbalance": pressure_drop_imbalance,
        "KE_centroid_x": KE_centroid_x,
        "reverse_flow_centroid_x": reverse_flow_centroid_x,
    }


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


@dataclass
class CaseInfo:
    """單一 LBM case 的描述。"""

    mask_id: str
    file_name: str  # h5 檔名 base（不含副檔名）
    h5_path: Path
    Re: float
    nu: float


def discover_cases(manifest_path: Path, raw_dir: Path) -> list[CaseInfo]:
    """從 manifest + 實際 raw/ 目錄找出所有完成的 case。"""
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    cases: list[CaseInfo] = []
    for entry in manifest["cases"]:
        mask_id = entry["mask_id"]
        Re = float(entry["Re"])
        # 找對應的 h5（檔名格式：diverse_NNNN_cfg_ReTNNNN_NuN-NNNN.h5）
        prefix = f"{mask_id}_cfg_ReT{int(Re):04d}_"
        matches = list(raw_dir.glob(f"{prefix}*.h5"))
        if not matches:
            continue
        h5_path = matches[0]
        # 從檔名抽取 nu
        stem = h5_path.stem
        nu_str = stem.split("_Nu")[1] if "_Nu" in stem else "0"
        nu = float(nu_str.replace("-", "."))
        cases.append(
            CaseInfo(
                mask_id=mask_id,
                file_name=h5_path.stem,
                h5_path=h5_path,
                Re=Re,
                nu=nu,
            )
        )
    return cases


# ---------------------------------------------------------------------------
# G_vec 計算（per case，因為 E1 隨 Re 變）
# ---------------------------------------------------------------------------


def compute_case_features(
    cases: list[CaseInfo], lbm_masks_dir: Path
) -> tuple[np.ndarray, list[str]]:
    """對每個 case 計算 22 維 G_vec（直接讀 native PNG，內部 BBOX）。

    E1 = Re·L_char/B1, E2 = U_inlet/(1-C2) 隨 case 的 Re 變化，
    其他 20 個 features 對同一 mask 重複。

    Args:
        cases: discover_cases() 找到的 case 列表。
        lbm_masks_dir: case mask PNG 所在目錄。

    Returns:
        (G [N_cases, 22], mask_ids [N_cases])
    """
    print(f"[features] 處理 {len(cases)} cases...")
    unique_masks = sorted({c.mask_id for c in cases})
    print(f"[features] {len(unique_masks)} 個 unique masks")

    # 每個 unique mask 算一次 base G（用 placeholder Re=1, U=DEFAULT_U_INLET_LU）
    base_g_per_mask: dict[str, np.ndarray] = {}
    for mid in unique_masks:
        mask_path = lbm_masks_dir / f"{mid}.png"
        binary = load_mask_native(mask_path)
        base_cfg: SimConfig = {
            "Re": 1.0,
            "U_inlet_lu": DEFAULT_U_INLET_LU,
            "L_char_px": DEFAULT_L_CHAR_PX,
        }
        base_g_per_mask[mid] = compute_all(binary, base_cfg)

    # E1, E2 indexing
    e1_idx = FEATURE_NAMES.index("E1_re_eff")
    e2_idx = FEATURE_NAMES.index("E2_blockage_corrected_velocity")
    b1_idx = FEATURE_NAMES.index("B1_throat_width_px")
    c2_idx = FEATURE_NAMES.index("C2_blockage_ratio")

    G = np.zeros((len(cases), FEATURE_DIM), dtype=np.float32)
    mask_ids: list[str] = []
    for i, c in enumerate(cases):
        g = base_g_per_mask[c.mask_id].copy()
        b1 = float(g[b1_idx])
        c2 = float(g[c2_idx])
        # 用 case 的 Re 重算 E1
        g[e1_idx] = c.Re * DEFAULT_L_CHAR_PX / (b1 + 1e-8)
        # E2 = U_inlet/(1-c2)；U_inlet 在這裡用 DEFAULT（其實隨 Re 變但簡化）
        g[e2_idx] = DEFAULT_U_INLET_LU / (1.0 - c2 + 1e-8)
        G[i] = g
        mask_ids.append(c.mask_id)
    return G, mask_ids


def build_re_aux(cases: list[CaseInfo]) -> np.ndarray:
    """建立 Re auxiliary regressor 向量（已 normalize）。

    Returns:
        Re_aux [N_cases, 1] float32
    """
    return np.array([[c.Re / RE_NORM_SCALE] for c in cases], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tier 3：Pearson + Ablation
# ---------------------------------------------------------------------------


def pearson_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """計算每對 (X[:, i], Y[:, j]) 的 Pearson ρ。

    Returns:
        rho [D_X, D_Y]
    """
    Dx = X.shape[1]
    Dy = Y.shape[1]
    out = np.zeros((Dx, Dy), dtype=np.float64)
    for i in range(Dx):
        for j in range(Dy):
            x = X[:, i]
            y = Y[:, j]
            sx = x.std()
            sy = y.std()
            if sx < 1e-10 or sy < 1e-10:
                out[i, j] = 0.0
                continue
            out[i, j] = float(np.corrcoef(x, y)[0, 1])
    return out


def ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    """OLS regression y ~ X (含 intercept)，回傳 R²。"""
    n = X.shape[0]
    X_aug = np.hstack([X, np.ones((n, 1))])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    y_hat = X_aug @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def ablation_importance(
    G_norm: np.ndarray,
    Y: np.ndarray,
    aux: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """多變量 ablation：對每個 G_vec feature i，計算 R²_full - R²_drop_i。

    aux 若提供（如 Re_aux），則 always 包含於 baseline 與 ablated model 中——
    亦即 importance 測量「在 aux 已知下，drop 該 G_vec feature 的代價」。

    Args:
        G_norm: [N, D] normalized G_vec features。
        Y: [N, K] targets。
        aux: [N, A] auxiliary regressor (e.g., Re_aux normalized)，or None。

    Returns:
        (importance [D, K], r2_full [K], r2_aux_only [K])
        r2_aux_only：只用 aux 沒有 G_vec 的 R²，作為「baseline / Re alone」基準。
    """
    D = G_norm.shape[1]
    K = Y.shape[1]
    importance = np.zeros((D, K), dtype=np.float64)
    r2_full = np.zeros(K, dtype=np.float64)
    r2_aux_only = np.zeros(K, dtype=np.float64)

    if aux is not None:
        X_full = np.hstack([G_norm, aux])
    else:
        X_full = G_norm

    for j in range(K):
        y = Y[:, j]
        r2_full[j] = ols_r2(X_full, y)
        if aux is not None:
            r2_aux_only[j] = ols_r2(aux, y)
        for i in range(D):
            keep_g = [d for d in range(D) if d != i]
            X_drop = G_norm[:, keep_g]
            if aux is not None:
                X_drop = np.hstack([X_drop, aux])
            r2_drop = ols_r2(X_drop, y)
            importance[i, j] = r2_full[j] - r2_drop
    return importance, r2_full, r2_aux_only


# ---------------------------------------------------------------------------
# 報告
# ---------------------------------------------------------------------------


def write_report(
    rho: np.ndarray,  # [12, K]
    importance: np.ndarray,  # [12, K]
    r2_full: np.ndarray,  # [K]
    r2_aux_only: np.ndarray,  # [K] — Re_aux 單獨 R²
    n_cases: int,
    n_unique_masks: int,
    out_path: Path,
) -> None:
    """寫 Tier 3 報告。"""
    K = len(METRIC_NAMES)
    lines: list[str] = []
    lines.append("# G_vec Tier 3 流場預測效力報告\n")
    lines.append(
        f"**資料來源**：lbm_L1_geo100_case300（{n_cases} cases，{n_unique_masks} unique masks）"
    )
    lines.append(f"**Re pool**：150 / 400 / 500（每 mask 3 個 Re）")
    lines.append(
        f"**通過門檻**：max_j |ρ(f_i, y_j)| > {TIER3_RHO_THRESHOLD} 或 max_j importance > {TIER3_IMPORTANCE_THRESHOLD}\n"
    )

    # 全模型 R²（每個 metric）+ adjusted R² + aux-only baseline
    n = n_cases
    k = FEATURE_DIM + 1  # +1 for Re_aux
    lines.append(
        f"## 一、Full-model R²（{FEATURE_DIM} 維 G_vec + Re_aux → 流場 metric）\n"
    )
    lines.append(
        f"註：N={n}, k={FEATURE_DIM} G_vec features + 1 Re_aux = {k}。"
        f"In-sample R² 會被高估，附 adjusted R² 作為較保守估計。"
    )
    lines.append("`adj R² = 1 − (1 − R²)(N−1)/(N−k−1)`")
    lines.append("`R²_aux_only` = 只用 Re_aux 不含 G_vec 的 R²（baseline）")
    lines.append("`gain` = R²_full − R²_aux_only（G_vec 在 Re 已知下的邊際解釋力）\n")
    lines.append("| Metric | R²_aux_only | R²_full | adj R² | gain |")
    lines.append("|--------|-------------|---------|--------|------|")
    for j, name in enumerate(METRIC_NAMES):
        r2 = r2_full[j]
        r2_aux = r2_aux_only[j]
        adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)
        gain = r2 - r2_aux
        lines.append(
            f"| `{name}` | {r2_aux:.3f} | {r2:.3f} | {adj:.3f} | **{gain:+.3f}** |"
        )
    lines.append("")

    # Pearson correlation table
    lines.append("## 二、單變量 Pearson ρ\n")
    header = "| Feature | " + " | ".join(METRIC_NAMES) + " | max\\|ρ\\| |"
    sep = "|" + "---|" * (K + 2)
    lines.append(header)
    lines.append(sep)
    max_abs_rho = np.abs(rho).max(axis=1)
    for i in range(FEATURE_DIM):
        cells = [f"{rho[i, j]:+.3f}" for j in range(K)]
        # 標粗顯著項（|ρ| > 0.30）
        cells_disp = [
            f"**{c}**" if abs(float(c)) > TIER3_RHO_THRESHOLD else c for c in cells
        ]
        lines.append(
            f"| `{FEATURE_NAMES[i]}` | "
            + " | ".join(cells_disp)
            + f" | {max_abs_rho[i]:+.3f} |"
        )
    lines.append("")

    # Ablation importance
    lines.append("## 三、多變量 Ablation importance（R²_full − R²_drop_i）\n")
    header = "| Feature | " + " | ".join(METRIC_NAMES) + " | max imp |"
    sep = "|" + "---|" * (K + 2)
    lines.append(header)
    lines.append(sep)
    max_imp = importance.max(axis=1)
    for i in range(FEATURE_DIM):
        cells = [f"{importance[i, j]:+.3f}" for j in range(K)]
        cells_disp = [
            f"**{c}**" if float(c) > TIER3_IMPORTANCE_THRESHOLD else c for c in cells
        ]
        lines.append(
            f"| `{FEATURE_NAMES[i]}` | "
            + " | ".join(cells_disp)
            + f" | {max_imp[i]:+.3f} |"
        )
    lines.append("")

    # Per-feature verdict
    lines.append("## 四、Per-feature 判定\n")
    lines.append("| Feature | max\\|ρ\\| | max imp | 通過？ | 備註 |")
    lines.append("|---------|----------|---------|--------|------|")
    pass_count = 0
    for i in range(FEATURE_DIM):
        rho_ok = max_abs_rho[i] > TIER3_RHO_THRESHOLD
        imp_ok = max_imp[i] > TIER3_IMPORTANCE_THRESHOLD
        passed = rho_ok or imp_ok
        if passed:
            pass_count += 1
        flag = "[PASS]" if passed else "[FAIL]"
        # 找 best metric for ρ
        best_j = int(np.argmax(np.abs(rho[i])))
        best_metric = METRIC_NAMES[best_j]
        note = f"strongest: `{best_metric}` (ρ={rho[i, best_j]:+.3f})"
        lines.append(
            f"| `{FEATURE_NAMES[i]}` | {max_abs_rho[i]:+.3f} | {max_imp[i]:+.3f} | {flag} | {note} |"
        )
    lines.append("")
    lines.append(f"**通過數**：{pass_count} / {FEATURE_DIM}\n")

    # D 群現狀
    lines.append("## 五、D 群冗餘狀況（22-d schema）\n")
    if "D1_mean_wall_distance_px" in FEATURE_NAMES:
        d1_idx = FEATURE_NAMES.index("D1_mean_wall_distance_px")
        d2_idx = FEATURE_NAMES.index("D2_std_wall_distance_px")
        d3_idx = FEATURE_NAMES.index("D3_hydraulic_diameter_est_px")
        lines.append(
            f"- D1：max\\|ρ\\|={max_abs_rho[d1_idx]:+.3f}, max imp={max_imp[d1_idx]:+.3f}"
        )
        lines.append(
            f"- D2：max\\|ρ\\|={max_abs_rho[d2_idx]:+.3f}, max imp={max_imp[d2_idx]:+.3f}"
        )
        lines.append(
            f"- D3：max\\|ρ\\|={max_abs_rho[d3_idx]:+.3f}, max imp={max_imp[d3_idx]:+.3f}"
        )
    else:
        d2_idx = FEATURE_NAMES.index("D2_std_wall_distance_px")
        d3_idx = FEATURE_NAMES.index("D3_hydraulic_diameter_est_px")
        lines.append(
            f"- D2：max\\|ρ\\|={max_abs_rho[d2_idx]:+.3f}, max imp={max_imp[d2_idx]:+.3f}"
        )
        lines.append(
            f"- D3：max\\|ρ\\|={max_abs_rho[d3_idx]:+.3f}, max imp={max_imp[d3_idx]:+.3f}"
        )
    lines.append("")

    # C4 spatial verdict — Option B 專屬段落
    lines.append("## 六、C4 在空間 metric 上的 verdict（Option B）\n")
    lines.append(
        "全域聚合 metric（mean/std）受質量守恆主導，對「**阻塞在哪**」不敏感。"
        "用 4 個 spatial metric 給 C4 公平 test：\n"
    )
    spatial_metric_names = [
        "speed_asymmetry_lr",
        "pressure_drop_imbalance",
        "KE_centroid_x",
        "reverse_flow_centroid_x",
    ]
    spatial_indices = [METRIC_NAMES.index(n) for n in spatial_metric_names]
    c4_idx = FEATURE_NAMES.index("C4_ud_bcr_ratio")

    lines.append(
        "| Spatial metric | Pearson ρ(C4, ·) | 全 22 維 R² | drop C4 R² | Δ R² |"
    )
    lines.append("|---|---|---|---|---|")
    spatial_max_imp_for_c4 = 0.0
    spatial_max_rho_for_c4 = 0.0
    for j in spatial_indices:
        rho_c4 = rho[c4_idx, j]
        r2_full_j = r2_full[j]
        imp_c4_j = importance[c4_idx, j]
        r2_drop = r2_full_j - imp_c4_j
        if abs(rho_c4) > abs(spatial_max_rho_for_c4):
            spatial_max_rho_for_c4 = float(rho_c4)
        if imp_c4_j > spatial_max_imp_for_c4:
            spatial_max_imp_for_c4 = float(imp_c4_j)
        rho_str = (
            f"**{rho_c4:+.3f}**"
            if abs(rho_c4) > TIER3_RHO_THRESHOLD
            else f"{rho_c4:+.3f}"
        )
        imp_str = (
            f"**{imp_c4_j:+.3f}**"
            if imp_c4_j > TIER3_IMPORTANCE_THRESHOLD
            else f"{imp_c4_j:+.3f}"
        )
        lines.append(
            f"| `{METRIC_NAMES[j]}` | {rho_str} | {r2_full_j:.3f} | {r2_drop:.3f} | {imp_str} |"
        )
    lines.append("")
    lines.append(
        f"**C4 在空間 metric 的 max\\|ρ\\| = {abs(spatial_max_rho_for_c4):+.3f}**"
    )
    lines.append(f"**C4 在空間 metric 的 max imp = {spatial_max_imp_for_c4:+.3f}**\n")

    spatial_pass = (
        abs(spatial_max_rho_for_c4) > TIER3_RHO_THRESHOLD
        or spatial_max_imp_for_c4 > TIER3_IMPORTANCE_THRESHOLD
    )
    if spatial_pass:
        lines.append(
            "→ **C4 在空間 metric 上通過 Tier 3**。確認 C4 在不對稱性 metric 上"
            "確實有預測力，全域聚合 metric 失敗是 metric 選擇問題不是 feature 問題。"
            "**保留 C4**。"
        )
    else:
        lines.append(
            "→ **C4 即使在空間 metric 上也失敗**。flat_diverse2000 + Re∈{150,400,500} "
            "範圍內，建物前後分布差異**不會**產生可預測的流場非對稱模式。**建議刪除 C4**。"
        )
    lines.append("")

    lines.append("---\n")
    lines.append("*由 `src/physics_proxy/validate_tier3.py` 產生。*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] 已寫入 {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(lbm_dataset_dir: Path, force_recompute: bool = False) -> None:
    """跑完整 Tier 3 驗證。

    Args:
        lbm_dataset_dir: LBM dataset 目錄（含 lbm_manifest_L1.json、raw/、masks/）。
        force_recompute: 忽略既有 cache 強制重算。
    """
    manifest_path = lbm_dataset_dir / "lbm_manifest_L1.json"
    raw_dir = lbm_dataset_dir / "raw"
    lbm_masks_dir = lbm_dataset_dir / "masks"

    print(f"[main] 解析 manifest: {manifest_path}")
    cases = discover_cases(manifest_path, raw_dir)
    print(f"[main] 找到 {len(cases)} 個完成的 cases")

    unique_masks = sorted({c.mask_id for c in cases})
    print(f"[main] {len(unique_masks)} 個 unique masks")

    cache_features, cache_targets, cache_meta = tier3_cache_paths(lbm_dataset_dir)

    if (
        not force_recompute
        and cache_features.exists()
        and cache_targets.exists()
        and cache_meta.exists()
    ):
        print("[cache] 載入 features + targets cache")
        G = np.load(cache_features)
        Y = np.load(cache_targets)
        with cache_meta.open(encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[cache] G shape={G.shape}, Y shape={Y.shape}")
    else:
        # 計算 G_vec
        print("[main] 計算 G_vec...")
        G, _ = compute_case_features(cases, lbm_masks_dir)

        # 抽取流場 metrics
        print(f"[main] 抽取流場 metrics（{len(cases)} cases）...")
        Y_rows: list[list[float]] = []
        for i, c in enumerate(cases):
            metrics = extract_flow_metrics(c.h5_path)
            Y_rows.append([metrics[k] for k in METRIC_NAMES])
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(cases)} 完成")
        Y = np.array(Y_rows, dtype=np.float64)

        # cache
        cache_features.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_features, G)
        np.save(cache_targets, Y)
        meta = [
            {
                "mask_id": c.mask_id,
                "file_name": c.file_name,
                "Re": c.Re,
                "nu": c.nu,
            }
            for c in cases
        ]
        with cache_meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[cache] 已儲存到 {CACHE_DIR}")

    # 正規化（z-score；T3 regression 對 scale 不敏感但仍正規化方便比較）
    g_mean = G.mean(axis=0)
    g_std = G.std(axis=0)
    g_std = np.where(g_std < 1e-8, 1.0, g_std)
    G_norm = (G - g_mean) / g_std

    # Pearson
    print("[main] 跑 Pearson ρ...")
    rho = pearson_matrix(G_norm, Y)

    # 建立 Re auxiliary regressor
    Re_aux = build_re_aux(cases)
    print(f"[main] Re_aux shape={Re_aux.shape}")

    # Ablation
    print("[main] 跑 ablation regression（含 Re_aux baseline）...")
    importance, r2_full, r2_aux_only = ablation_importance(G_norm, Y, aux=Re_aux)

    # 報告
    write_report(
        rho,
        importance,
        r2_full,
        r2_aux_only,
        len(cases),
        len(unique_masks),
        REPORT_PATH,
    )

    print("\n=== Tier 3 summary ===")
    print(f"R2 per metric (aux_only / full / gain):")
    for j, name in enumerate(METRIC_NAMES):
        gain = r2_full[j] - r2_aux_only[j]
        print(
            f"  {name:25s}  aux={r2_aux_only[j]:.3f}  full={r2_full[j]:.3f}  gain={gain:+.3f}"
        )
    print()
    max_abs_rho = np.abs(rho).max(axis=1)
    max_imp = importance.max(axis=1)
    pass_count = 0
    for i in range(FEATURE_DIM):
        rho_ok = max_abs_rho[i] > TIER3_RHO_THRESHOLD
        imp_ok = max_imp[i] > TIER3_IMPORTANCE_THRESHOLD
        passed = rho_ok or imp_ok
        if passed:
            pass_count += 1
        flag = "[PASS]" if passed else "[FAIL]"
        print(
            f"  {flag} {FEATURE_NAMES[i]:45s} max|ρ|={max_abs_rho[i]:+.3f}  max_imp={max_imp[i]:+.3f}"
        )
    print(f"\n通過 {pass_count}/{FEATURE_DIM}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--lbm-dataset-dir", type=Path, required=True,
        help="LBM dataset 目錄（含 lbm_manifest_L1.json、raw/、masks/，"
             "e.g. dataset/CEDS_dataset/lbm_L1_geo100_case300）",
    )
    p.add_argument(
        "--force", action="store_true", help="強制重算 features + targets cache"
    )
    args = p.parse_args()
    main(lbm_dataset_dir=args.lbm_dataset_dir, force_recompute=args.force)
