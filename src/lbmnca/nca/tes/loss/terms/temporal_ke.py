"""時序動能穩定性損失（長程 rollout 衰減懲罰）。

提取自 loss_terms_v2.py（§ domain_mean_ke / two_window_ke_loss / temporal_ke_2ctu /
jacobian_amp_loss）。

命名規則：
  domain_mean_ke       — 單幀 domain-mean KE（供 trainer 增量累積）
  two_window_ke_loss   — 兩窗 KE 穩定性（Dy-C/E）
  temporal_ke_2ctu     — 一次性 API（已 stack 序列）
  jacobian_amp_loss    — 單步 Jacobian 放大率懲罰（Dy-V8/V9）

所有函式回傳 scalar Tensor（可微，支援 backward）；
domain_mean_ke 回傳 [B]。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import ddx as _ddx, ddy as _ddy  # noqa: F401  （JAWS 用）

# ---------------------------------------------------------------------------
# 模組層級常數
# ---------------------------------------------------------------------------

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

_RHO_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# 內部輔助：局部複雜度（供 JAWS 加權）
# ---------------------------------------------------------------------------


def _local_complexity(phys_norm: Tensor) -> Tensor:
    """局部複雜度 [B,H,W]∈[0,1]：physics 空間梯度幅值，per-sample p95 正規化（robust）。

    高複雜度 = 渦核/剪切層（細節區）；低 = 自由流/遠場（boring 區）。供 JAWS 加權用。

    Args:
        phys_norm: ``[B, pd, H, W]`` 正規化物理場（任何 channel 數）。

    Returns:
        ``[B, H, W]`` float，局部複雜度 ∈[0,1]。
    """
    gx = _ddx(phys_norm)                                    # [B, pd, H, W]
    gy = _ddy(phys_norm)
    mag = (gx.square() + gy.square()).sum(dim=1).sqrt()     # [B, H, W]
    flat = mag.flatten(1)                                   # [B, H*W]
    p95 = torch.quantile(flat, 0.95, dim=1, keepdim=True).clamp_min(1e-8)
    return (flat / p95).clamp(0.0, 1.0).view_as(mag)        # [B, H, W]


# ---------------------------------------------------------------------------
# 公開介面
# ---------------------------------------------------------------------------


def domain_mean_ke(
    phys_norm: Tensor,      # [B, 9, H, W] normalized（單幀）
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """單幀 domain-mean KE = mean_fluid(u² + v²)（denorm，u=jx/rho, v=jy/rho）。

    回傳 [B]。逐幀計算（記憶體輕，供長 rollout 增量累積 KE 曲線用，避免 stack
    整段 [B,R,9,H,W]）。grad 隨 phys_norm 流（grad 幀帶梯度、detached 幀為常數）。

    Args:
        phys_norm: ``[B, 9, H, W]`` 正規化單幀物理場。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        ``[B]`` Tensor，各 batch 元素的 domain-mean KE。
    """
    raw = normalizer.denorm_mphy(phys_norm)        # [B, 9, H, W]
    rho = raw[:, RHO_CH].clamp_min(rho_eps)
    vel_x = raw[:, JX_CH] / rho
    vel_y = raw[:, JY_CH] / rho
    ke_map = vel_x.square() + vel_y.square()       # [B, H, W]
    m = fluid_mask.squeeze(1)                      # [B, H, W]
    return (ke_map * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp_min(1.0)  # [B]


def two_window_ke_loss(
    ke_t: Tensor,           # [B, R] per-step domain-mean KE
    ctu_seq: Tensor,        # [B, R] 每步累積 CTU（raw，與 ke_t 對齊）
    window_ctu: float = 5.0,
    eps: float = 0.05,
) -> Tensor:
    """solver 2CTU two-window relu loss（從 per-step KE 序列）。

    取最近 2×window_ctu 分兩半比窗均，``ratio = |mean(KE_late)−mean(KE_early)| / mean(KE)``，
    ``loss = relu(ratio − eps)``（只在比 solver 容忍度更不穩時罰 → 不過度凍結，不變 smooth-blob）。
    兩窗 mask 由 ``ctu_seq`` 決定（對 grad 常數）；窗內樣本 <2 的 batch 元素跳過（回 0）。
    grad 僅經 ke_t 中帶梯度的元素（rollout 尾段）回流。

    Args:
        ke_t: ``[B, R]`` per-step domain-mean KE。
        ctu_seq: ``[B, R]`` 每步累積 CTU（raw）。
        window_ctu: 半窗大小（CTU）。
        eps: 收斂門檻（|Δmean|/mean）。

    Returns:
        scalar Tensor（valid batch 元素的 relu loss 平均）。
    """
    t_end = ctu_seq[:, -1:]                                 # [B, 1]
    t_mid = t_end - window_ctu
    t_start = t_end - 2.0 * window_ctu
    w1 = ((ctu_seq > t_start) & (ctu_seq <= t_mid)).float()  # [B, R]
    w2 = ((ctu_seq > t_mid) & (ctu_seq <= t_end)).float()
    n1 = w1.sum(dim=1)
    n2 = w2.sum(dim=1)                                       # [B]
    valid = (n1 >= 2) & (n2 >= 2) & (t_start[:, 0] >= 0.0)
    m1 = (ke_t * w1).sum(dim=1) / n1.clamp_min(1.0)
    m2 = (ke_t * w2).sum(dim=1) / n2.clamp_min(1.0)
    mb = 0.5 * (m1 + m2)
    ratio = (m2 - m1).abs() / (mb.abs() + 1e-10)            # [B]
    loss_b = F.relu(ratio - eps) * valid.float()
    return loss_b.sum() / valid.float().sum().clamp_min(1.0)


def spectral_temporal_loss(
    ke_t: Tensor,           # [B, R] per-step domain-mean KE
    ctu_seq: Tensor,        # [B, R] 每步累積 CTU（均勻間距）
    f_ref: float = 0.20,    # St 帶中心 CTU⁻¹（T=5.0 CTU）
    f_bw: float = 0.04,     # 半帶寬（覆蓋 Re=150–500 St 範圍）
    band_weight: float = 0.5,
    min_R: int = 16,
) -> Tensor:
    """LC-1 peak-sharpening spectral loss（semi-self-supervised，T_GT-independent）。

    主軸：罰 rollout KE 功率分散、獎勵窄峰 → 拉高 prominence/ac_peak。
    次軸：軟性把 FFT 主峰偏向 St 帶 [f_ref-f_bw, f_ref+f_bw]（T=4.0–6.7 CTU）。

    fp32：``with torch.amp.autocast("cuda", enabled=False)`` + ke.float()。
    grad 僅流經 ke_t 帶梯度尾段；preroll detached 幀為常數（semi-self-supervised）。

    ★merge 自 main（LC-1 idea，原綁舊 trainer 結構）。函式已移入 LC terms/ 正確位置，
    供 turb 階段評估接線；接線需 spectral_temporal_weight>0（config 已留欄位）。

    Args:
        ke_t:        [B, R] per-step domain-mean KE。
        ctu_seq:     [B, R] 每步累積 CTU（均勻間距）。
        f_ref:       St 帶中心 CTU⁻¹。
        f_bw:        St 帶半帶寬 CTU⁻¹。
        band_weight: 次軸損失相對權重（軟約束，非硬鎖）。
        min_R:       序列最小長度（< min_R 回零維持 graph）。

    Returns:
        scalar Tensor（fp32）。
    """
    ke = ke_t.float()                                           # [B, R] fp32
    R = ke.shape[1]
    if R < min_R:
        return ke.sum() * 0.0                                   # 維持 graph，值=0

    with torch.amp.autocast("cuda", enabled=False):
        # 均勻 delta CTU（detach 避免 delta 帶梯度至頻率 bins）
        delta = (ctu_seq[:, 1:] - ctu_seq[:, :-1]).float().mean().detach().clamp_min(1e-6)

        # 去均值 detrend（去 DC，focus 振盪分量）
        ke_dt = ke - ke.mean(dim=1, keepdim=True)               # [B, R]

        # rfft → power（跳過 DC bin=index 0）
        spec = torch.fft.rfft(ke_dt, dim=1)                     # [B, n_bin] complex
        power = spec.real.square() + spec.imag.square()         # [B, n_bin] real
        power = power[:, 1:]                                     # 去 DC；[B, n_bin-1]
        n_freqs = power.shape[1]
        if n_freqs < 2:
            return ke.sum() * 0.0

        freqs = torch.fft.rfftfreq(R, d=float(delta), device=ke.device)[1:]  # [n_freqs] CTU⁻¹

        # ── 主軸：concentration loss = 1 − peak/total ────────────────────────
        total_pow = power.sum(dim=1).clamp_min(1e-10)           # [B]
        peak_pow, peak_idx = power.max(dim=1)                   # [B]
        concentration = peak_pow / total_pow                     # [B] ∈ (0, 1]
        loss_main = (1.0 - concentration).mean()                 # → 0 = 完美窄峰

        # ── 次軸：軟 St 帶（只罰 |f_peak − f_ref| > f_bw）──────────────────
        f_peak = freqs[peak_idx]                                 # [B]
        loss_band = torch.relu((f_peak - f_ref).abs() - f_bw).mean()

    return loss_main + band_weight * loss_band


def temporal_ke_2ctu(
    pred_norm: Tensor,      # [B, R, 9, H, W] normalized（preroll detached + grad tail 串接）
    ctu_seq: Tensor,        # [B, R] 每步累積 CTU（raw）
    fluid_mask: Tensor,     # [B, 1, H, W] float
    window_ctu: float = 5.0,
    eps: float = 0.05,
    rho_eps: float = _RHO_EPS,
    *,
    normalizer: FieldNormalizer,
) -> Tensor:
    """L_temporal：solver 2CTU two-window KE 統計穩態 loss（Dy plan §2.4）。

    與真實 LBM solver 同款收斂判準（``convergence_monitor.check_ke_convergence_twowindow``）。
    逐幀算 domain-mean KE → ``two_window_ke_loss``。一次性 API（unit test / 已 stack 的序列）；
    trainer 長 rollout 走 ``domain_mean_ke`` + ``two_window_ke_loss`` 增量版（省記憶體）。

    Args:
        pred_norm: ``[B, R, 9, H, W]`` 長 rollout 序列（normalized Mphy）。
        ctu_seq: ``[B, R]`` 每步累積 CTU（raw，與 pred_norm 對齊）。
        fluid_mask: ``[B, 1, H, W]`` 流體=1 固體=0。
        window_ctu: 半窗大小（CTU），每半窗應 ≥1 脫落週期（~5 CTU）。
        eps: |Δmean|/mean 收斂門檻（~0.05，與 solver 同）。
        normalizer: FieldNormalizer（反正規化至 raw Mphy）。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor，valid batch 元素的 relu(ratio−eps) 平均。
    """
    R = pred_norm.shape[1]
    ke_t = torch.stack(
        [domain_mean_ke(pred_norm[:, i], fluid_mask, normalizer, rho_eps) for i in range(R)],
        dim=1,
    )  # [B, R]
    return two_window_ke_loss(ke_t, ctu_seq, window_ctu=window_ctu, eps=eps)


def jacobian_amp_loss(
    f_x: Tensor,            # [B, pd, H, W] 單步輸出（norm physics）
    f_xe: Tensor,           # [B, pd, H, W] 擾動後單步輸出
    eps: Tensor,            # [B, pd, H, W] 擾動量（同 norm space）
    fluid_mask: Tensor,     # [B, 1, H, W]
    gamma: float = 1.0,
    gamma_low: float = 0.0,    # W0-1 bug-B：two-sided 下界（防死場免費）；0=向後相容單側
    jaws_enabled: bool = False,
    jaws_lambda_c: float = 1.0,
) -> Tensor:
    """Dy-V8/V9：單步 Jacobian 放大率懲罰（真收縮 + 死場下界；plan §3.5/§3.6 + W0-1 bug-B）。

    局部放大率 ``ratio = ‖f(x+ε) − f(x)‖ / ‖ε‖``（fluid 區）。誤差傳播 e_{t+1}≈J·e_t，
    ratio>1 ⟺ 放大 ⟺ long-roll 發散。
    **two-sided**：``loss = relu(ratio − γ) + relu(γ_low − ratio)``。
      - 上界 relu(ratio−γ)：罰放大（γ≤1 強制真收縮）。
      - 下界 relu(γ_low−ratio)：罰 ``ratio<γ_low`` ⟺ 對擾動無響應 ⟺ **死場/凍結**（視覺 D 殺手）。
        ``γ_low=0`` 時 relu(−ratio)=0（ratio≥0），退回原單側行為（向後相容）。
    短訓練即壓長 rollout（逐步壓放大率，同時禁止塌成 dead-field）。

    - V8 全域（jaws_enabled=False）：per-sample ratio → ``relu(ratio−γ).mean()``。
    - V9 JAWS（jaws_enabled=True）：per-pixel ratio，權重 ``w(x)=1/(1+λ_c·complexity(x))``
      （complexity=局部 |∇f_x|，detach）→ 渦核放鬆、平滑區強收縮 → 收縮但不 smooth-blob。

    Args:
        f_x: ``[B, pd, H, W]`` 單步輸出（norm physics）。
        f_xe: ``[B, pd, H, W]`` 擾動後單步輸出。
        eps: ``[B, pd, H, W]`` 擾動量（norm space）。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        gamma: 上界放大率門檻（≤1 強制收縮）。
        gamma_low: 下界放大率門檻（0=關閉死場檢測）。
        jaws_enabled: True=JAWS per-pixel 加權模式。
        jaws_lambda_c: JAWS 複雜度懲罰強度。

    Returns:
        scalar Tensor。
    """
    m = fluid_mask                                          # [B, 1, H, W]
    diff = (f_xe - f_x) * m
    eps_m = eps * m
    if not jaws_enabled:
        num = diff.flatten(1).norm(dim=1)                   # [B]
        den = eps_m.flatten(1).norm(dim=1).clamp_min(1e-8)
        ratio = num / den
        return (F.relu(ratio - gamma) + F.relu(gamma_low - ratio)).mean()
    # JAWS：per-pixel ratio + complexity 加權（complexity detach，防模型 gaming 權重）
    num_px = diff.norm(dim=1)                               # [B, H, W]（physics ch L2）
    den_px = eps_m.norm(dim=1).clamp_min(1e-8)              # [B, H, W]
    ratio_px = num_px / den_px
    w = 1.0 / (1.0 + jaws_lambda_c * _local_complexity(f_x.detach()))  # [B, H, W]
    mask2d = m.squeeze(1)                                   # [B, H, W]
    pen = (F.relu(ratio_px - gamma) + F.relu(gamma_low - ratio_px)) * w * mask2d
    return pen.sum() / mask2d.sum().clamp_min(1.0)
