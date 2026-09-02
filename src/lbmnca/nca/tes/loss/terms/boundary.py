"""邊界軟性約束損失函式 — 從 loss_terms_v2 提取。

命名規則（繼承 loss_terms_v2 語義化命名）：
  BndWall_*    — 近壁邊界軟性約束（raw/denorm space）

所有函式回傳 scalar Tensor（可微，支援 backward）。
物理項需傳入 FieldNormalizer 以反正規化至 raw Mphy space。

mask 慣例：
  fluid_mask [B,1,H,W] float，流體=1 固體=0
  wall_band  [B,1,H,W] float，近壁流體格點=1

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  JX_CH  = 3  (m3: x 動量)
  JY_CH  = 5  (m5: y 動量)
"""
from __future__ import annotations

from typing import Optional

from torch import Tensor

from ....utils.FieldNormalizer import FieldNormalizer
from ...utils.spatial_ops import (
    compute_wall_band,
    ddx as _ddx,
    ddy as _ddy,
)

# ---------------------------------------------------------------------------
# Channel index 常數（與 loss_terms_v2 保持同步）
# ---------------------------------------------------------------------------

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

_RHO_EPS: float = 1e-6

__all__ = [
    "bnd_wall_normalflux_absmean",
    "compute_wall_band",
]


# ---------------------------------------------------------------------------
# 近壁邊界軟性約束
# ---------------------------------------------------------------------------


def bnd_wall_normalflux_absmean(
    pred_norm: Tensor,
    sdf_raw: Tensor,
    fluid_mask: Tensor,
    wall_band: Optional[Tensor],
    normalizer: FieldNormalizer,
    rho_eps: float = _RHO_EPS,
) -> Tensor:
    """BndWall_NormalFlux_AbsMean：近壁法向流量的均絕對值。

    LBM floor: ~7e-4 to 1.9e-3（S0/S1 audit，已修正）。先做診斷；不作硬約束。

    法向量由 SDF 梯度計算（中央差分），nx = ∂SDF/∂x / |∇SDF|，ny 同理。
    近壁法向流量 = ux·nx + uy·ny，在 wall_band 遮罩內取均絕對值。

    Args:
        pred_norm: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        sdf_raw: ``[B, H, W]`` raw SDF（反正規化後）。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        wall_band: ``[B, 1, H, W]`` 近壁帶遮罩；None 時自動由 fluid_mask 計算。
        normalizer: 反正規化器（denorm 至 raw Mphy space）。
        rho_eps: 密度下限夾值，避免除以 0。

    Returns:
        scalar Tensor（近壁法向流量均絕對值）。
    """
    if wall_band is None:
        wall_band = compute_wall_band(fluid_mask)

    B, K, C, H, W = pred_norm.shape
    flat = pred_norm.reshape(B * K, C, H, W)
    raw = normalizer.denorm_mphy(flat)          # [B*K, 9, H, W]
    rho = raw[:, RHO_CH].clamp_min(rho_eps)
    ux = raw[:, JX_CH] / rho
    uy = raw[:, JY_CH] / rho

    sdf_x = _ddx(sdf_raw)                      # [B, H, W]
    sdf_y = _ddy(sdf_raw)
    grad_norm = (sdf_x.square() + sdf_y.square()).sqrt().clamp_min(rho_eps)
    nx = (sdf_x / grad_norm)                    # [B, H, W] unit normal x
    ny = (sdf_y / grad_norm)

    nx_bk = nx[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    ny_bk = ny[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    normal_flux = ux * nx_bk + uy * ny_bk      # [B*K, H, W]

    wb_bk = wall_band.squeeze(1)[:, None, :, :].expand(B, K, H, W).reshape(B * K, H, W)
    return (normal_flux.abs() * wb_bk).sum() / wb_bk.sum().clamp_min(1.0)
