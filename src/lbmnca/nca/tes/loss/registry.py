"""registry.py — Loss term 註冊表。

本模組從 semantic_loss.py 拆出，集中管理所有 loss term 的建立與查詢。
``_build_registry()`` 一次性建立所有 ``_LossTerm`` 並以語義名稱為鍵回傳字典；
模組層級常數 ``_REGISTRY`` / ``_SUPPORTED_TERMS`` / ``_TERM_KEY`` 供外部直接引用。

新增 loss term 步驟：
1. 在對應的 ``loss_terms_*.py`` 實作函式。
2. 在本模組頂部新增 import。
3. 在 ``_build_registry()`` 末尾附加 ``_LossTerm(...)`` 條目。
4. 不需修改 ``semantic_loss.py``（自動從 _REGISTRY 路由）。
"""
from __future__ import annotations

from typing import Callable

from torch import Tensor

from .context import _LossContext, _LossTerm
from .terms.struct_banding import struct_column_banding_loss

# ---------------------------------------------------------------------------
# loss_terms_v2 匯入
# ---------------------------------------------------------------------------
from .loss_terms_v2 import (
    bnd_wall_normalflux_absmean,
    compute_wall_band,          # noqa: F401（_LossContext 依賴，此處留作文件化）
    enstrophy_match_loss,
    fneq_match_loss,
    horiz_band_fft_loss,
    ke_magnitude_match_loss,
    motion_twoframe_loss,
    obs_mphy_delta_mae,
    obs_mphy_final_mae,
    obs_mphy_traj_mae,
    obs_vel_gradient_mae,
    obs_vel_vorticity_bl_mae,
    obs_vel_vorticity_mae,
    phys_mass_divj_absmean,
    phys_mass_divvel_absmean,
    stab_improve_loss,
    struct_vel_ssim_loss,
    struct_vort_signed_huber,
    struct_vort_ssim_loss,
    struct_wake_vel_mae,
    struct_wake_vort_mae,
    vel_field_decoupled_loss,
)

# ---------------------------------------------------------------------------
# loss_terms_m1a 匯入
# ---------------------------------------------------------------------------
from .loss_terms_m1a import (
    obs_state_rho_mae,
    obs_state_stress_anchor,
    obs_state_stress_mae,
    obs_vphy_delta_mae,
    obs_vphy_vel_mae,
    obs_vphy_wakevert_mae,
)

# ---------------------------------------------------------------------------
# 其他依賴
# ---------------------------------------------------------------------------
from ..eval.pod_metrics import load_pod_basis, pod_coeff_loss  # noqa: F401
from .inflow_prior import inflow_weight_map_seq, obs_mphy_traj_mae_prior  # noqa: F401


# ---------------------------------------------------------------------------
# _build_registry
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, _LossTerm]:
    """建立 loss term 註冊表，回傳 ``name → _LossTerm`` 對應。

    所有 loss term 在此集中定義；lambda 封裝 ``_LossContext`` 存取，
    確保各 term 的引數皆從 context 取得而非散落各處。

    Returns:
        以語義名稱為鍵、_LossTerm 為值的字典。
    """
    terms: list[_LossTerm] = [
        _LossTerm(
            "ObsMphy_Traj_Mae",
            lambda c: obs_mphy_traj_mae(c.pred, c.target, c.fluid_mask),
        ),
        _LossTerm(
            # HF-1c：inflow-prior region-weighted trajectory MAE（loss-mask only，
            # 不改 forward）。weight map=None 時 == ObsMphy_Traj_Mae（identity）。
            "ObsMphy_TrajPrior_Mae",
            lambda c: obs_mphy_traj_mae_prior(
                c.pred, c.target, c.fluid_mask, c.inflow_weight_map
            ),
        ),
        _LossTerm(
            "ObsMphy_Delta_Mae",
            lambda c: obs_mphy_delta_mae(c.pred, c.target, c.ic_physics, c.fluid_mask),
        ),
        _LossTerm(
            "ObsMphy_Final_Mae",
            lambda c: obs_mphy_final_mae(c.pred, c.target, c.fluid_mask),
        ),
        _LossTerm(
            # TV2 §1.4：non-worsening / L_improve（懲罰誤差逐步變大；preds-only，無 model ref）。
            "Stab_Improve",
            lambda c: stab_improve_loss(c.pred, c.target, c.fluid_mask),
        ),
        _LossTerm(
            # Dy2 §4：velocity-field 兩幀運動統計（jx/jy 直接算，禁光流網路）。
            # 比對 pred vs target 的 Δvel 分布（mean/std，x/y 分量）→ 相位不變運動 loss。
            "Motion_TwoFrame",
            lambda c: motion_twoframe_loss(
                c.pred, c.target, c.fluid_mask, c.normalizer
            ),
            needs_normalizer=True,
        ),
        _LossTerm(
            "ObsVel_Vorticity_Mae",
            lambda c: obs_vel_vorticity_mae(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            # WF-W1b：band-limited 渦量 MAE（高斯低通 sigma~2px 後 MAE）→ 可升高權找回
            # 定位/銳度而不放大 grid 噪（W1 大審：plain vort-MAE 5→2 削太狠致瀰漫+broadband 噪）。
            "ObsVel_VorticityBL_Mae",
            lambda c: obs_vel_vorticity_bl_mae(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            # Dy2-S3a §16：渦量能量（enstrophy）匹配，非逐點 → 罰 pred 渦量能量低於 GT，
            # 逼保留脫落渦旋強度（解 MAE 獎勵平均/殺脫落根因）。
            "StructVort_EnstrophyMatch",
            lambda c: enstrophy_match_loss(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            # WFC §4 缺塊2：非均衡應力能量 ⟨f_neq²⟩ 匹配（死場哨兵）。f_neq=Pxx−(jx²−jy²)/rho；
            # 場塌成均衡→f_neq→0→deficit 暴。自監督（no-GT 步 target 可換 detached 參考）。
            "PhysNeq_FneqMatch",
            lambda c: fneq_match_loss(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            # WFC spike-1c（[A28] 用戶 A）：DyNCA-style 方向(1−cos,GT-mag 加權)+強度(相對
            # deficit 補幅) 解耦 velocity loss。state 顯性 jx/jy → 免 optical flow。
            "StructVel_DirMagDecoupled",
            lambda c: vel_field_decoupled_loss(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            # Dy2-S3b4 §17：總動能 KE=mean(jx²+jy²) 匹配，補 enstrophy 不管的速度
            # magnitude（warm 上 enstrophy 強渦旋但 ke_ratio 略低=解耦）→ ke_ratio→1。
            "StructVel_KEMagMatch",
            lambda c: ke_magnitude_match_loss(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "ObsVel_Gradient_Mae",
            lambda c: obs_vel_gradient_mae(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "PhysMass_DivJ_AbsMean",
            lambda c: phys_mass_divj_absmean(c.pred, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "PhysMass_DivVel_AbsMean",
            lambda c: phys_mass_divvel_absmean(c.pred, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "BndWall_NormalFlux_AbsMean",
            lambda c: bnd_wall_normalflux_absmean(
                c.pred, c.sdf_raw, c.fluid_mask, c.wall_band, c.normalizer
            ),
            needs_normalizer=True,
        ),
        _LossTerm(
            "StructWake_VelMae",
            lambda c: struct_wake_vel_mae(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "StructWake_VortMae",
            lambda c: struct_wake_vort_mae(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "StructVort_SignedHuber",
            lambda c: struct_vort_signed_huber(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "StructVel_SsimLoss",
            lambda c: struct_vel_ssim_loss(
                c.pred, c.target, c.fluid_mask, c.normalizer,
                range_from_stats=c.ssim_range_from_stats,
            ),
            needs_normalizer=True,
        ),
        _LossTerm(
            "StructVort_SsimLoss",
            lambda c: struct_vort_ssim_loss(
                c.pred, c.target, c.fluid_mask, c.normalizer,
                range_from_stats=c.ssim_range_from_stats,
            ),
            needs_normalizer=True,
        ),
        _LossTerm(
            "StructHBandFFT_Loss",
            lambda c: horiz_band_fft_loss(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "StructColumnBanding",
            lambda c: struct_column_banding_loss(c.pred, c.target, c.fluid_mask),
        ),
        _LossTerm(
            "StructVel_PODCoeffLoss",
            lambda c: pod_coeff_loss(
                c.pred, c.target, c.fluid_mask, c.normalizer, c.pod_basis, c.pod_modes
            ),
            needs_normalizer=True,
        ),
        _LossTerm(
            "ObsVphy_Vel_Mae",
            lambda c: obs_vphy_vel_mae(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "ObsVphy_Delta_Mae",
            lambda c: obs_vphy_delta_mae(
                c.pred, c.target, c.ic_physics, c.fluid_mask, c.normalizer
            ),
            needs_normalizer=True,
        ),
        _LossTerm(
            "ObsVphy_WakeVort_Mae",
            lambda c: obs_vphy_wakevert_mae(c.pred, c.target, c.fluid_mask, c.normalizer),
            needs_normalizer=True,
        ),
        _LossTerm(
            "ObsState_Rho_Mae",
            lambda c: obs_state_rho_mae(c.pred, c.target, c.fluid_mask),
        ),
        _LossTerm(
            "ObsState_Stress_Mae",
            lambda c: obs_state_stress_mae(c.pred, c.target, c.fluid_mask),
        ),
        _LossTerm(
            # W0-1 bug-C：純 stress moment (ch7=pxx, ch8=pxy) 弱約束。LBM-native 偏差應力，
            # 免 curl（不放大 checkerboard）→ L-Cal §4.3 enstrophy 替換候選前置。
            "ObsState_Stress_Anchor",
            lambda c: obs_state_stress_anchor(c.pred, c.target, c.fluid_mask),
        ),
    ]
    return {t.name: t for t in terms}


# ---------------------------------------------------------------------------
# 模組層級常數（供 semantic_loss.py 及其他模組引用）
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, _LossTerm] = _build_registry()
"""所有已知 loss term 的 name → _LossTerm 對應表。"""

_SUPPORTED_TERMS: frozenset[str] = frozenset(_REGISTRY)
"""已支援的 loss term 語義名稱集合（用於 config 驗證）。"""

_TERM_KEY: dict[str, str] = {name: name.lower() for name in _SUPPORTED_TERMS}
"""語義名稱 → 小寫 metrics key 的對應（與 trainer CSV 相容）。"""


__all__ = ["_build_registry", "_REGISTRY", "_SUPPORTED_TERMS", "_TERM_KEY"]
