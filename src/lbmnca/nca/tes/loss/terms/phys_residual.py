"""物理守恆殘差損失函式 — 從 loss_terms_v2 提取。

命名規則（繼承 loss_terms_v2 語義化命名）：
  PhysMass_*   — 質量守恆 proxy（raw/denorm space，無成對 GT）
  PhysNeq_*    — 非均衡應力殘差（raw/denorm space）

``__all__`` 的 **4 個公開函式**（`phys_mass_divj_absmean`／`phys_mass_divvel_absmean`／
`mean_ke_log_anchor`／`fneq_match_loss`）皆回傳 scalar Tensor（可微，支援 backward）。
物理項需傳入 FieldNormalizer 以反正規化至 raw Mphy space。

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  JX_CH  = 3  (m3: x 動量)
  JY_CH  = 5  (m5: y 動量)
  PXX_CH = 7  (m7: 偏差應力矩，fneq 死場哨兵)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import ddx as _ddx, ddy as _ddy

# ---------------------------------------------------------------------------
# Channel index 常數（與 loss_terms_v2 保持同步）
# ---------------------------------------------------------------------------

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5
PXX_CH: int = 7   # m7 = Pxx（偏差應力矩）

_RHO_EPS: float = 1e-6

__all__ = [
    "phys_mass_divj_absmean",
    "phys_mass_divvel_absmean",
    "mean_ke_log_anchor",
    "fneq_match_loss",
]


# ---------------------------------------------------------------------------
# 質量守恆殘差（raw/denorm space，無 GT）
# ---------------------------------------------------------------------------


def phys_mass_divj_absmean(
    pred_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """PhysMass_DivJ_AbsMean：raw momentum 散度的均絕對殘差（fluid 區域）。

    在反正規化後計算 div(j_raw)，不在 norm space 計算。
    LBM floor：即使是精確解，raw momentum 散度在有限差分下仍有非零的數值底噪，
    其量級隨「場的種類」（mean 或 snapshot，相差約 14×）與「網格解析度」（近線性
    變化）而動，不可當成普適常數，引用時須連同場種類與解析度一起說明。
    （分母 = n_fluid × T。）

    ★量測慣例：workspace 內有**數個**名字都叫 divj／質量守恆殘差、
    但空間／差分核／遮罩／量綱互不等價的實作，**讀數不可互比**；**清單不保證窮盡**。
      本函式 = **raw/denorm 空間** + ``spatial_ops.ddx/ddy`` 中央差分 + **完整 fluid_mask**
      + **絕對量**（不除 |∇j|）。其餘已知版本（含
      ``tes.eval.physics_gate.divj_over_lbm_floor`` 的三個 region 變體、以及
      ``tes.eval.metrics.mass_conservation_residual`` 的 Sobel/8 版）逐條對照表見
      ``tes.eval.metrics.mass_conservation_residual`` docstring 的「量測慣例」段，
      不在此重複維護第二份清單。

    ★算子敏感度限制：中央差分核 ``[-0.5, 0, 0.5]`` 的頻率響應 ``i·sin(k)`` 在 2Δ 波長
    增益恰為 0、λ=4 px 低估約 36% → 本項對 band/grain 這類 grid-scale 結構**結構性免疫**，
    即使數值下降也不代表 grid-scale 守恆改善。grid-scale 敏感的對照算子見
    ``tes.utils.spatial_ops.divergence_flux``（通量式，2Δ 增益 2、界面上有定義；
    ★兩家族只有在 ``spatial_ops.interior_core_mask`` 核心區才可相除比較，開放域的
    域邊界/壁面讀數含 O(|j|) 邊界處理人工項）。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器（denorm 至 raw Mphy space）。
        rho_eps: 密度下限夾值（此函式未直接用，保留介面一致性）。

    Returns:
        scalar Tensor（momentum 散度均絕對值）。
    """
    B, K, C, H, W = pred_norm.shape
    flat = pred_norm.reshape(B * K, C, H, W)
    raw = normalizer.denorm_mphy(flat)          # [B*K, 9, H, W]
    jx = raw[:, JX_CH]
    jy = raw[:, JY_CH]
    div_j = _ddx(jx) + _ddy(jy)               # [B*K, H, W]

    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return (div_j.abs() * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)


def phys_mass_divvel_absmean(
    pred_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """PhysMass_DivVel_AbsMean：raw velocity 散度的均絕對殘差（fluid 區域）。

    ux = jx_raw / rho_raw，uy = jy_raw / rho_raw。
    LBM floor：即使是精確解，raw velocity 散度在有限差分下仍有非零的數值底噪，
    其量級隨「場的種類」（mean 或 snapshot，相差約 14×）與「網格解析度」（近線性
    變化）而動，不可當成普適常數，引用時須連同場種類與解析度一起說明。
    （與 DivJ 接近，因 rho ≈ 1 在 LBM 場成立。）

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器（denorm 至 raw Mphy space）。
        rho_eps: 密度下限夾值，避免除以 0。

    Returns:
        scalar Tensor（velocity 散度均絕對值）。
    """
    B, K, C, H, W = pred_norm.shape
    flat = pred_norm.reshape(B * K, C, H, W)
    raw = normalizer.denorm_mphy(flat)          # [B*K, 9, H, W]
    rho = raw[:, RHO_CH].clamp_min(rho_eps)
    jx = raw[:, JX_CH]
    jy = raw[:, JY_CH]
    ux = jx / rho
    uy = jy / rho
    div_vel = _ddx(ux) + _ddy(uy)             # [B*K, H, W]

    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return (div_vel.abs() * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)


def mean_ke_log_anchor(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    eps: float = 1e-8,
) -> Tensor:
    """PhysEnergy_KE_LogAnchor：momentum-KE 的雙向 log-ratio 錨（chain energy-anchor）。

    動機：chain 訓練使 momentum 場結構貼 GT，但**全域能量**
    無監督 → warm hand-off 可能逐 stage 泵能，個別鏈亦可能反塌縮。此項逐 stage 錨
    pred KE 對 GT KE，訂「每 edge 單位能量增益」契約。

    定義（raw/denorm space，per-sample 純量）：
        KE(x) = 0.5 · mean_fluid(jx_raw² + jy_raw²)
        l     = mean_B[ (log(KE_pred + eps) − log(KE_gt + eps))² ]

    設計選擇：
    - **momentum-KE（jx²+jy²）而非 velocity-KE（(j/ρ)²）**：後者除以可學 ρ 會開作弊梯度
      （灌大 ρ 壓 u、j 照爆）；LBM ρ≈1 兩者差二階小量；ρ 漂移另由 moment ch0 + divj 看管。
    - **two-sided log-ratio**：失效雙向（多數泵 10×、個別塌 0.05×）→ 平方 log-ratio 兩向皆罰；
      log 型式跨 Re 尺度不變、梯度有界（KE 爆 10× 梯度不跟爆）、ratio≈2 處軟牆。
    - GT 端 ``no_grad``。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 或 ``[B, 9, H, W]`` 正規化預測。
        target_norm: 同形狀正規化 GT。
        fluid_mask: ``[B, 1, H, W]`` float（fluid=1）。
        normalizer: 反正規化器（denorm 至 raw 後算 momentum-KE）。
        eps: log 保護。

    Returns:
        scalar loss（fp32）。
    """
    if pred_norm.dim() == 4:
        pred_norm = pred_norm.unsqueeze(1)
        target_norm = target_norm.unsqueeze(1)
    B, K, C, H, W = pred_norm.shape
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    denom = mask_bk.sum(dim=(1, 2)).clamp_min(1.0)

    def _ke(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W)).float()  # fp16-safe（二階矩）
        e = 0.5 * (raw[:, JX_CH].square() + raw[:, JY_CH].square())          # [B*K, H, W]
        return (e * mask_bk).sum(dim=(1, 2)) / denom                          # [B*K] per-sample KE

    ke_p = _ke(pred_norm)
    with torch.no_grad():
        ke_t = _ke(target_norm)
    return ((ke_p.add(eps).log() - ke_t.add(eps).log()) ** 2).mean()


# ---------------------------------------------------------------------------
# 非均衡應力殘差（raw/denorm space，自監督死場哨兵）
# ---------------------------------------------------------------------------


def fneq_match_loss(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    under_weight: float = 1.0,
    over_weight: float = 0.25,
    eps: float = 1e-8,
) -> Tensor:
    """PhysNeq_FneqMatch：非均衡應力能量 ⟨f_neq²⟩ 匹配（WFC 死場哨兵）。

    動機：cross-all-CTU 的 no-GT 步無 pixel 監督，需一個
    **自監督**信號偵測「場死掉/塌成均衡」。LBM 非均衡矩 ``f_neq = Pxx − Pxx_eq`` 量測
    場偏離**局部平衡**的程度——活躍湍流 f_neq≠0，場死/塌成平滑層流 → f_neq→0。故罰
    pred 的 ⟨f_neq²⟩ **低於**參考（asymmetric deficit，鏡射 ``enstrophy_match_loss``）
    = 死場哨兵：場一塌，deficit 暴。

    f_neq 定義（正確位置為
    ``src/lbmnca/lbm/lbm_mrt_les/core/LBM2D_MRT_LES.py::LBM2D_MRT_LES.get_meq``，
    其回傳向量第 8 個分量即 ``rho * (u * u - v * v)``）：
        Pxx_eq = m^eq[7] = rho·(u² − v²) = (jx² − jy²)/rho     （u=jx/rho, v=jy/rho）
        f_neq  = ch7(Pxx_raw) − Pxx_eq

    **自監督**：值只由各自場算（pred / target 各自的 f_neq）→ no-GT 步 target 可換
    detached 參考（last-landing pred / GT），landing 步 target=GT frame。

    ★fp16-safe：raw 先 ``.float()``（jx²/Pxx 為二階矩，fp16 易溢位；對齊 SSIM
    fp16 修正教訓）→ 過 F=AMP grad-check。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測物理場。
        target_norm: ``[B, k, 9, H, W]`` 正規化目標物理場。
        fluid_mask: ``[B, 1, H, W]`` float（fluid=1）。
        normalizer: 反正規化器（denorm 至 raw 後算 f_neq）。
        rho_eps: 密度下限夾值。
        under_weight: pred f_neq 能量不足的相對缺額權重（主，死場罰）。
        over_weight: 過量輕罰。
        eps: 相對化分母保護。

    Returns:
        scalar loss（fp32），≈ 相對 f_neq 能量缺額（0=pred 非均衡應力≥參考）。
    """
    B, K, C, H, W = pred_norm.shape

    def _fneq_sq(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W)).float()  # fp16-safe
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        jx = raw[:, JX_CH]
        jy = raw[:, JY_CH]
        pxx = raw[:, PXX_CH]
        pxx_eq = (jx * jx - jy * jy) / rho        # = rho·(u²−v²)，solver m^eq[7]
        fneq = pxx - pxx_eq                        # [B*K, H, W] 非均衡偏差應力
        return fneq.square()

    sp = _fneq_sq(pred_norm)
    st = _fneq_sq(target_norm)
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    denom = mask_bk.sum(dim=(1, 2)).clamp_min(1.0)             # per-sample fluid count
    lp = (sp * mask_bk).sum(dim=(1, 2)) / denom                # [B*K] pred ⟨f_neq²⟩
    lt = (st * mask_bk).sum(dim=(1, 2)) / denom
    deficit = (lt - lp).clamp_min(0.0) / (lt + eps)           # pred 非均衡應力不足（主＝死場）
    excess = (lp - lt).clamp_min(0.0) / (lt + eps)            # 過量（輕罰）
    return (under_weight * deficit + over_weight * excess).mean()
