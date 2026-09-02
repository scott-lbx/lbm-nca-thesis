"""速度場結構損失（需 denorm 至 raw space）。

包含以下損失函式（依讀取原始碼順序）：
  - motion_twoframe_loss        — 相鄰幀速度增量的 moment matching（相位不變）
  - obs_vel_vorticity_mae       — 速度場渦量差 MAE
  - obs_vel_vorticity_bl_mae    — band-limited 渦量差 MAE
  - enstrophy_match_loss        — 渦量能量匹配（非對稱 deficit）
  - vel_field_decoupled_loss    — 方向 + 幅度解耦速度 loss
  - ke_magnitude_match_loss     — 總動能匹配（非對稱 deficit）
  - obs_vel_gradient_mae        — 速度梯度幅值差 MAE（逐點，無界，慎用大權重）
  - grad_energy_ratio_loss      — 速度梯度能量的有界相對匹配（抗過度平滑）

所有函式回傳 scalar Tensor（可微，支援 backward）。
物理項需傳入 FieldNormalizer（ctx.normalizer）進行 denorm。

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  JX_CH  = 3  (m3: x 動量)
  JY_CH  = 5  (m5: y 動量)

mask 慣例：
  fluid_mask [B,1,H,W] float，流體=1 固體=0
"""
from __future__ import annotations

import torch
from torch import Tensor

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import (
    ddx as _ddx,
    ddy as _ddy,
    gaussian_lowpass as _gaussian_lowpass,
)

# ---------------------------------------------------------------------------
# Channel 索引常數
# ---------------------------------------------------------------------------

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

_RHO_EPS: float = 1e-6
#: 梯度幅值 sqrt 的保護項。sqrt(0) 的導數是無限大，而固體區梯度恰為 0。
_GRAD_EPS: float = 1e-12

__all__ = [
    "motion_twoframe_loss",
    "obs_vel_vorticity_mae",
    "obs_vel_vorticity_bl_mae",
    "enstrophy_match_loss",
    "vel_field_decoupled_loss",
    "ke_magnitude_match_loss",
    "obs_vel_gradient_mae",
    "grad_energy_ratio_loss",
]


# ---------------------------------------------------------------------------
# 1. motion_twoframe_loss
# ---------------------------------------------------------------------------


def motion_twoframe_loss(
    pred: Tensor,                   # [B, k, 9, H, W] normalized
    target: Tensor,                 # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,             # [B, 1, H, W] float
    normalizer: FieldNormalizer,    # raw-space 反正規化橋接
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """Motion_TwoFrame：相鄰幀速度增量分布的 moment matching（相位不變運動 loss）。

    語意：直接用 jx/jy 算速度場兩幀運動統計，**禁用光流網路**。
    對 pred 與 target 各算「相鄰幀速度增量」::

        vel_t   = (jx_raw / rho_raw, jy_raw / rho_raw)   # denorm 後 fluid 區
        Δvel_t  = vel_{t+1} − vel_t                       # t = 0..k-2

    再以 moment matching 比較 pred 與 target 的 Δvel 分布（x/y 分量分別取
    fluid 區 mean 與 std）::

        L = |mean_p^x − mean_t^x| + |mean_p^y − mean_t^y|
          + |std_p^x  − std_t^x|  + |std_p^y  − std_t^y|

    只比對運動「分布統計」而非逐點對齊，因此對整體相位偏移不敏感（相位不變），
    懲罰的是 pred 與 target 在「相鄰幀運動量級/方向散布」上的差異。

    pred 幀數 < 2 時無相鄰幀 → 回 0（保留 graph/device/dtype，仿 ``stab_improve_loss``）。

    Args:
        pred: ``[B, k, 9, H, W]`` NCA 預測軌跡（normalized Mphy）。
        target: ``[B, k, 9, H, W]`` LBM 目標軌跡（normalized Mphy）。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: ``FieldNormalizer``，反正規化至 raw Mphy（取 jx/jy/rho）。
        rho_eps: rho 反正規化後的下限夾值，避免除以 0。

    Returns:
        scalar ``Tensor``（可微，支援 backward）。
    """
    if pred.shape[1] < 2:
        return (pred * 0.0).sum()

    B, K, C, H, W = pred.shape

    def _delta_vel(seq: Tensor) -> tuple[Tensor, Tensor]:
        """回傳 (Δvx, Δvy)，shape 皆 [B, K-1, H, W]（raw 速度增量）。"""
        raw = normalizer.denorm_mphy(seq.reshape(B * K, C, H, W))  # [B*K, 9, H, W]
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vx = (raw[:, JX_CH] / rho).reshape(B, K, H, W)
        vy = (raw[:, JY_CH] / rho).reshape(B, K, H, W)
        dvx = vx[:, 1:] - vx[:, :-1]      # [B, K-1, H, W]
        dvy = vy[:, 1:] - vy[:, :-1]
        return dvx, dvy

    # fluid 區 broadcast mask（與 Δvel 同形）：[B, 1, H, W] → [B, K-1, H, W]
    m = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K - 1, H, W)
    denom = m.sum().clamp_min(1.0)

    def _masked_mean_std(x: Tensor) -> tuple[Tensor, Tensor]:
        """fluid 區（mask 加權）的 mean 與 std（population std）。"""
        mean = (x * m).sum() / denom
        var = (((x - mean) ** 2) * m).sum() / denom
        std = (var + _RHO_EPS).sqrt()
        return mean, std

    p_dvx, p_dvy = _delta_vel(pred)
    t_dvx, t_dvy = _delta_vel(target)

    p_mx, p_sx = _masked_mean_std(p_dvx)
    p_my, p_sy = _masked_mean_std(p_dvy)
    t_mx, t_sx = _masked_mean_std(t_dvx)
    t_my, t_sy = _masked_mean_std(t_dvy)

    return (
        (p_mx - t_mx).abs()
        + (p_my - t_my).abs()
        + (p_sx - t_sx).abs()
        + (p_sy - t_sy).abs()
    )


# ---------------------------------------------------------------------------
# 2. obs_vel_vorticity_mae
# ---------------------------------------------------------------------------


def obs_vel_vorticity_mae(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    target_norm: Tensor,    # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """ObsVel_Vorticity_Mae：預測/目標速度場渦量差的 MAE（fluid 區域）。

    渦量 ω = dv/dx - du/dy（中央差分），u=jx/rho，v=jy/rho。
    需反正規化至 raw Mphy space 後計算。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（fluid 區渦量 MAE）。
    """
    B, K, C, H, W = pred_norm.shape

    def _vorticity(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))  # [B*K, 9, H, W]
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        return _ddx(vel_y) - _ddy(vel_x)           # [B*K, H, W]  ω = dv/dx - du/dy

    vort_pred = _vorticity(pred_norm)
    vort_tgt = _vorticity(target_norm)

    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return ((vort_pred - vort_tgt).abs() * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# 3. obs_vel_vorticity_bl_mae
# ---------------------------------------------------------------------------


def obs_vel_vorticity_bl_mae(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    target_norm: Tensor,    # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    band_limit_sigma: float = 2.0,
) -> Tensor:
    """ObsVel_VorticityBL_Mae：**band-limited** 渦量差 MAE。

    渦量 ω 算法同 ``obs_vel_vorticity_mae``，但 pred/tgt 各先過高斯低通（sigma≈2px，
    cutoff ~4-6px）**再** MAE。動機：plain vort-MAE 在 grid-scale 會獎勵匹配高頻渦度，
    使 model 灌入 grid 噪。band-limit 擋掉 2-3px 的 grid 噪頻段、放行渦核尺度
    （~數十 px），使 vort-MAE 可升高權重（~5）**找回定位/銳度而不放大 grid 噪**。

    視覺優先：純 checkerboard hack 經低通後→≈0，與低通後的 GT 真渦度結構差大
    → MAE **HIGH（不被 checker 騙）**，優於 plain vort-MAE（GT 尺度下漏，0.004）。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target_norm: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。
        band_limit_sigma: 高斯低通 sigma（px）。0=退化為 plain vort-MAE。預設 2.0；
            cutoff 卡在此值：太高 grid 噪回來、太低渦核糊掉。

    Returns:
        scalar Tensor（band-limited 渦量 MAE）。
    """
    B, K, C, H, W = pred_norm.shape

    def _vorticity(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        return _ddx(vel_y) - _ddy(vel_x)           # [B*K, H, W]

    vort_pred = _gaussian_lowpass(_vorticity(pred_norm), band_limit_sigma)
    vort_tgt = _gaussian_lowpass(_vorticity(target_norm), band_limit_sigma)

    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return ((vort_pred - vort_tgt).abs() * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# 4. enstrophy_match_loss
# ---------------------------------------------------------------------------


def enstrophy_match_loss(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    target_norm: Tensor,    # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    under_weight: float = 1.0,
    over_weight: float = 0.75,   # 加重 excess 罰（grid-噪/checkerboard）
    eps: float = 1e-8,
) -> Tensor:
    """StructVort_EnstrophyMatch：渦量能量（enstrophy = mean_fluid ω²）匹配，**非逐點**。

    動機：MAE/L1 對「整場變平滑」滿意（smooth blob 的 ω²≈0 也低 loss）→ 獎勵時間平均、
    殺死脫落。本 loss 直接罰 pred 的渦量**總能量低於** GT（asymmetric：缺渦旋罰重、過量
    輕罰），逼模型保留脫落渦旋強度。per-sample 相對化（除 GT enstrophy）→ 尺度無關。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測物理場。
        target_norm: ``[B, k, 9, H, W]`` 正規化目標物理場。
        fluid_mask: ``[B, 1, H, W]`` float（fluid=1）。
        normalizer: 反正規化器（denorm 至 raw 後算 ω）。
        rho_eps: 密度下限夾值。
        under_weight: pred 渦量能量低於 GT 的相對缺額權重（主項）。
        over_weight: 過量的輕罰權重。
        eps: 相對化分母保護。

    Returns:
        scalar loss（fp32），≈ 相對 enstrophy 缺額（0=pred 渦量能量≥GT）。
    """
    B, K, C, H, W = pred_norm.shape

    def _vorticity(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        return _ddx(vel_y) - _ddy(vel_x)           # [B*K, H, W]

    vp = _vorticity(pred_norm)
    vt = _vorticity(target_norm)
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    denom = mask_bk.sum(dim=(1, 2)).clamp_min(1.0)              # per-sample fluid count
    ens_p = (vp.square() * mask_bk).sum(dim=(1, 2)) / denom     # [B*K]
    ens_t = (vt.square() * mask_bk).sum(dim=(1, 2)) / denom
    # 分母用 `max(ens_t, ens_p)` 而非單純 `ens_t + eps`：若用後者，`deficit`
    #   因分子 ≤ ens_t 而天然有界於 1，但 `excess` 完全無界——近均勻流的
    #   case（ens_t 遠小於 eps 想擋的量級）會讓它衝到 1e5–1e9。改用
    #   `max(ens_t, ens_p)` 當分母：語義仍是「相對缺額／過量」，
    #   但兩項都有界於 [0, 1]，不需要人為 cap，也不必猜一個 eps 門檻。
    scale = torch.maximum(ens_t, ens_p).clamp_min(eps)
    deficit = (ens_t - ens_p).clamp_min(0.0) / scale   # pred 渦量能量不足（主），∈[0,1]
    excess = (ens_p - ens_t).clamp_min(0.0) / scale    # 過量（輕罰），∈[0,1]
    return (under_weight * deficit + over_weight * excess).mean()


# ---------------------------------------------------------------------------
# 5. vel_field_decoupled_loss
# ---------------------------------------------------------------------------


def vel_field_decoupled_loss(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    target_norm: Tensor,    # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    eps: float = 1e-6,
    dir_weight: float = 1.0,
    mag_weight: float = 1.0,
    mag_deficit_only: bool = True,
    mag_over_weight: float = 0.0,
    mag_over_thresh: float = 1.3,
) -> Tensor:
    """StructVel_DirMagDecoupled：DyNCA-style 方向+強度解耦 velocity loss。

    動機：cross-CTU 的 no-GT 步驟中，方向對齊尚可、但強度明顯衰減——這正是
    DyNCA target vector field 方法要處理的「方向/強度」兩軸解耦失效模式。
    **★關鍵簡化**：本 state 已有顯性速度（jx/jy）→ 直接從 denorm 算
    v=(vx,vy)，**省 DyNCA 的預訓練 optical flow**。

    兩軸解耦：
        - **方向** dir = 1 − cos_sim(v_pred, v_gt)（0=同向）。**以 GT 速度大小加權**（near-zero
          GT 速度區方向噪自然降權＝DyNCA 方向 gating）。
        - **強度** mag：相對強度差。``mag_deficit_only=True`` → 只罰 ‖v_pred‖<‖v_gt‖（補幅，
          warm 幅值太低需 boost；deficit-only 自限於 ratio=1 不過壓）。

    ★fp16-safe：raw ``.float()``、sqrt 內 +eps、所有分母 clamp_min（近零速度除零防護，對齊
    fneq/SSIM overflow 教訓 → 過 F=AMP grad finite）。

    Args:
        pred_norm: ``[B,k,9,H,W]`` 正規化預測物理場。
        target_norm: ``[B,k,9,H,W]`` 正規化目標物理場。
        fluid_mask: ``[B,1,H,W]`` float。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。
        eps: 向量長度最小值（防除零）。
        dir_weight: 方向項權重。
        mag_weight: 強度項權重。
        mag_deficit_only: True=只罰強度不足（補幅）；False=雙向 |Δ‖v‖|。
        mag_over_weight: 過衝輕罰權重（0=關閉）。
        mag_over_thresh: 過衝門檻倍率（‖v_pred‖ > thresh·‖v_gt‖ 才罰）。

    Returns:
        scalar loss（fp32）。
    """
    B, K, C, H, W = pred_norm.shape

    def _vel(x_norm: Tensor) -> tuple[Tensor, Tensor]:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W)).float()  # fp16-safe
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        return raw[:, JX_CH] / rho, raw[:, JY_CH] / rho        # [B*K, H, W]

    vpx, vpy = _vel(pred_norm)
    vgx, vgy = _vel(target_norm)
    mask = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)

    mp = (vpx * vpx + vpy * vpy + eps).sqrt()                  # ‖v_pred‖
    mg = (vgx * vgx + vgy * vgy + eps).sqrt()                  # ‖v_gt‖
    # 方向：cos_sim，以 GT 速度大小加權（方向 gating）
    cos = (vpx * vgx + vpy * vgy) / (mp * mg).clamp_min(eps)
    dir_err = (1.0 - cos.clamp(-1.0, 1.0))                     # [0,2]
    w_dir = mg * mask
    l_dir = (dir_err * w_dir).sum() / w_dir.sum().clamp_min(eps)
    # 強度：相對 deficit（補幅）或雙向
    if mag_deficit_only:
        mag_rel = (mg - mp).clamp_min(0.0) / mg.clamp_min(eps)
    else:
        mag_rel = (mp - mg).abs() / mg.clamp_min(eps)
    l_mag = (mag_rel * mask).sum() / mask.sum().clamp_min(eps)
    # over-amp clamp（防止複利過衝；僅 ‖v_pred‖ > over_thresh·‖v_gt‖ 才輕罰，
    # deficit 端不受影響＝補幅照常、只擋過衝）
    if mag_over_weight > 0.0:
        over_rel = (mp - mag_over_thresh * mg).clamp_min(0.0) / mg.clamp_min(eps)
        l_over = (over_rel * mask).sum() / mask.sum().clamp_min(eps)
        l_mag = l_mag + mag_over_weight * l_over
    return dir_weight * l_dir + mag_weight * l_mag


# ---------------------------------------------------------------------------
# 6. ke_magnitude_match_loss
# ---------------------------------------------------------------------------


def ke_magnitude_match_loss(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    target_norm: Tensor,    # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    under_weight: float = 1.0,
    over_weight: float = 0.75,   # 加重 excess 罰（grid-噪/checkerboard）
    eps: float = 1e-8,
) -> Tensor:
    """StructVel_KEMagMatch：總動能 KE = mean_fluid(jx²+jy²) 匹配，**非逐點**。

    動機：`enstrophy_match_loss` 控的是渦量能量 ω²（旋轉結構），**不含
    速度 magnitude**——渦旋強度提升時 ke_ratio 可能反而略低（兩者解耦）。
    本 loss 補總動能：罰 pred KE 低於 GT（asymmetric），把 ke_ratio→1。與 longroll
    `_ke_norm`（jx²+jy² 動量 KE proxy）同定義 → 訓練/eval 對齊。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測物理場。
        target_norm: ``[B, k, 9, H, W]`` 正規化目標物理場。
        fluid_mask: ``[B, 1, H, W]`` float（fluid=1）。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。
        under_weight: pred KE 低於 GT 的相對缺額權重（主）。
        over_weight: 過量輕罰。
        eps: 相對化分母保護。

    Returns:
        scalar loss（fp32），≈ 相對 KE 缺額（0=pred KE≥GT）。
    """
    B, K, C, H, W = pred_norm.shape

    def _ke(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        return raw[:, JX_CH].square() + raw[:, JY_CH].square()     # [B*K, H, W] 動量 KE proxy

    kp = _ke(pred_norm)
    kt = _ke(target_norm)
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    denom = mask_bk.sum(dim=(1, 2)).clamp_min(1.0)
    ke_p = (kp * mask_bk).sum(dim=(1, 2)) / denom                  # [B*K]
    ke_t = (kt * mask_bk).sum(dim=(1, 2)) / denom
    deficit = (ke_t - ke_p).clamp_min(0.0) / (ke_t + eps)         # pred 總動能不足（主）
    excess = (ke_p - ke_t).clamp_min(0.0) / (ke_t + eps)
    return (under_weight * deficit + over_weight * excess).mean()


# ---------------------------------------------------------------------------
# 7. obs_vel_gradient_mae
# ---------------------------------------------------------------------------


def obs_vel_gradient_mae(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    target_norm: Tensor,    # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """ObsVel_Gradient_Mae：速度梯度幅值差的 MAE（fluid 區域）。

    梯度幅值 |∇u|² = (du/dx)² + (du/dy)² + (dv/dx)² + (dv/dy)²，取 sqrt。
    需反正規化至 raw Mphy space 後計算。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測物理場。
        target_norm: ``[B, k, 9, H, W]`` 正規化目標物理場。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。

    Returns:
        scalar Tensor（速度梯度幅值差 MAE）。
    """
    B, K, C, H, W = pred_norm.shape

    def _grad_mag(x_norm: Tensor) -> Tensor:
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))  # [B*K, 9, H, W]
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        dudx = _ddx(vel_x); dudy = _ddy(vel_x)
        dvdx = _ddx(vel_y); dvdy = _ddy(vel_y)
        # sqrt 內必須加 eps。`sqrt(0)` 的導數是無限大，而固體區與均勻區的
        #   梯度幅值**恰好**是 0 → backward 直接產生 NaN。
        return (dudx.square() + dudy.square() + dvdx.square() + dvdy.square()
                + _GRAD_EPS).sqrt()

    gm_pred = _grad_mag(pred_norm)
    gm_tgt = _grad_mag(target_norm)

    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return ((gm_pred - gm_tgt).abs() * mask_bk).sum() / mask_bk.sum().clamp_min(1.0)

# ---------------------------------------------------------------------------
# 8. grad_energy_ratio_loss
# ---------------------------------------------------------------------------


def grad_energy_ratio_loss(
    pred_norm: Tensor,      # [B, k, 9, H, W] normalized
    target_norm: Tensor,    # [B, k, 9, H, W] normalized
    fluid_mask: Tensor,     # [B, 1, H, W] float
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
    under_weight: float = 1.0,
    over_weight: float = 0.75,
    eps: float = 1e-12,
    per_direction: bool = True,
) -> Tensor:
    """StructVel_GradEnergyRatio：速度梯度能量 ⟨|∇u|²⟩ 的**有界**相對匹配。

    為什麼需要這一項（DT 保留集實測）：

    - `enstrophy` 比 0.959、高渦度面積比 0.982 → 渦度**總量與面積都沒問題**
    - 速度梯度能量比 **0.780** → **模型比 GT 平滑 22%**，這才是視覺缺陷

    而 SSIM 在**任何值域下**對模糊都幾乎全盲：σ=3.0 的高斯模糊只被罰
    0.0057（R=0.04）～0.0129（R=0.006），乘上 0.3 權重後佔總 loss 的 0.03–0.17%。
    同一個模糊，本項罰 **0.4136**——敏感度差 **73 倍**。

    為什麼不用既有的 `obs_vel_gradient_mae`：它是 `|∇u|` 的**逐點 MAE**，
    在 raw lattice 單位下數值極小（收斂模型約 2.65e-4），要有作用得配權重 ~400，
    但它**完全無界**——實測 pred 幅值過量 100× 時值達 5.3e5，×400 → 2.1e8，
    訓練直接發散（初次啟動即 loss=5e9）。

    本項改用**純量能量比**並以 ``max(E_t, E_p)`` 正規化，使 deficit 與 excess
    **都有界於 [0, 1]**（與 `enstrophy_match_loss` 同一形式），
    不需要人為 cap，也不必猜 eps 門檻。非對稱：缺梯度（過平滑）罰重、過量輕罰。

    ★**為什麼要逐方向**（``per_direction=True``）：
    若把 x／y 兩個方向的梯度能量**加總**成一個純量再匹配，會留下一條鑽漏洞的
    路徑——**水平條紋**（沿 x 均勻、沿 y 變化）只抬高
    ``⟨(∂u/∂y)²⟩`` 而完全不動 ``⟨(∂u/∂x)²⟩``，卻能把總和衝上去，
    而且不必碰尾流結構，是最便宜的滿足方式。

    逐方向匹配讓兩個分量**各自**要對，加條紋就不再有幫助，漏洞從構造上消失。
    ``per_direction=False`` 保留舊行為（僅加總，不逐方向）。

    ★本項仍是**全域純量**，不指出「哪裡」該更銳利。防線是既有的 anti-grain 項
    （``band``／``vgg``／``colored huber``）與新接上的 ``horiz_band_fft_loss``，
    且每次訓練後必須看視覺圖確認沒有 grain 或條紋。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測物理場。
        target_norm: ``[B, k, 9, H, W]`` 正規化目標物理場。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        normalizer: 反正規化器。
        rho_eps: 密度下限夾值。
        under_weight: 梯度能量**低於** GT 的相對缺額權重（主項＝抗過度平滑）。
        over_weight: 過量的輕罰權重。
        eps: 正規化分母下限。

    Returns:
        scalar Tensor，∈ ``[0, under_weight + over_weight]``。
    """
    B, K, C, H, W = pred_norm.shape

    def _grad_energy(x_norm: Tensor) -> tuple[Tensor, Tensor]:
        """回傳 ``(x 方向能量, y 方向能量)``，各為 ``[B*K]``。"""
        raw = normalizer.denorm_mphy(x_norm.reshape(B * K, C, H, W))
        rho = raw[:, RHO_CH].clamp_min(rho_eps)
        vel_x = raw[:, JX_CH] / rho
        vel_y = raw[:, JY_CH] / rho
        gx = _ddx(vel_x).square() + _ddx(vel_y).square()          # ∂/∂x 分量
        gy = _ddy(vel_x).square() + _ddy(vel_y).square()          # ∂/∂y 分量
        m = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
        den = m.sum(dim=(1, 2)).clamp_min(1.0)
        return (gx * m).sum(dim=(1, 2)) / den, (gy * m).sum(dim=(1, 2)) / den

    def _ratio_penalty(e_p: Tensor, e_t: Tensor) -> Tensor:
        scale = torch.maximum(e_t, e_p).clamp_min(eps)
        deficit = (e_t - e_p).clamp_min(0.0) / scale
        excess = (e_p - e_t).clamp_min(0.0) / scale
        return under_weight * deficit + over_weight * excess

    px, py = _grad_energy(pred_norm)
    tx, ty = _grad_energy(target_norm)
    if per_direction:
        # ★逐方向匹配（預設）。見 docstring 的「為什麼要逐方向」。
        return (0.5 * (_ratio_penalty(px, tx) + _ratio_penalty(py, ty))).mean()
    return _ratio_penalty(px + py, tx + ty).mean()
