"""確定性入流先驗 loss（HF-1c/1d）。

由 current_ctu + 幾何 metadata 推得 inflow front 的 active / transition / inactive
soft masks，**完全不用 target**（無洩漏）。座標慣例以 `ARCHITECTURE_CURRENT.md §6`
的 position embedding 為準：

    X_01 = (col + 0.5) / W   ∈ [0,1]，0 = inlet（左），1 = outlet（右）；inflow_dir = +x

front 沿 inflow 方向以 convective 速度前進：1 CTU 行進 L_char（物理），
故在 X_01 單位：

    dX01_per_ctu = L_char_px / ROI_W          # ROI 原始解析度寬（L_char_px 為 ROI scale）
    x_front_01   = x_inlet_01 + current_ctu * dX01_per_ctu
    active(ctu)  = sigmoid((x_front_01 - X_01) / w_front)   # 已抵達區
    transition   = active * (1 - active)                    # 鋒面帶（峰值 0.25 在 front）
    inactive     = 1 - active

時序鐵律（HF_plan §0.9 / §2.3）：prior 一律用步首 `current_ctu`，不用 target_ctu，
鋒面前進交給 STREAM（HF-1d composition），避免 double-advection。

x_inlet_01 預設 0（ROI 左邊界）；solver crop 造成的 ~1.5 L_char 偏移被 w_front 吸收，
且 w_front / x_inlet 列入 HPO-1，不在此硬擬合 target。
"""
from __future__ import annotations

import torch
from torch import Tensor

# ── HF-1c 預設常數（w_front 等列入 HPO-1；此處為固定 probe 值）───────────────
ROI_W_PX: float = 1920.0          # ROI 原始解析度寬（L_char_px 的 scale，見 §6 降採樣 pipeline）
X_INLET_01: float = 0.0           # 均勻入口在 ROI 左邊界（X_01=0）
DEFAULT_W_FRONT: float = 0.05     # sigmoid 鋒面寬（X_01 單位，≈ 1 L_char 量級）
DEFAULT_W_ACTIVE: float = 1.0     # 已抵達 active 區 loss 權重
DEFAULT_W_TRANSITION: float = 1.5 # 鋒面帶 loss 權重（最強）
DEFAULT_W_INACTIVE: float = 0.2   # 尚未抵達區 loss 權重（target≈0，下調）
_X01_CH: int = 2                  # static channel index：[SDF, upstream, X_01, Y_01]


def compute_inflow_masks(
    current_ctu: Tensor,
    l_char_px: Tensor,
    x01: Tensor,
    fluid_mask: Tensor,
    w_front: float = DEFAULT_W_FRONT,
    roi_w_px: float = ROI_W_PX,
    x_inlet_01: float = X_INLET_01,
) -> tuple[Tensor, Tensor, Tensor]:
    """計算入流鋒面的三種空間 soft mask。

    回傳 (active, transition, inactive)，皆 [B, 1, H, W]。
    active 與 inactive 已乘 fluid_mask；transition = active*(1-active)
    不另乘 fluid_mask（後續 weight map 會乘）。

    Args:
        current_ctu: [B] 步首 CTU（raw，非 normalized）。
        l_char_px: [B] ROI scale 特徵長度（= p_vec[:,2]）。
        x01: [B, 1, H, W] static X_01 channel ∈[0,1]。
        fluid_mask: [B, 1, H, W] float，fluid=1 solid=0。
        w_front: sigmoid 鋒面寬（X_01 單位）。
        roi_w_px: ROI 原始解析度寬（像素）。
        x_inlet_01: 入口 X_01 座標（預設 0 = 左邊界）。

    Returns:
        (active, transition, inactive)：各為 [B, 1, H, W] soft mask。
    """
    b = current_ctu.shape[0]
    dx01_per_ctu = (l_char_px.float() / roi_w_px).view(b, 1, 1, 1)        # [B,1,1,1]
    x_front_01 = x_inlet_01 + current_ctu.float().view(b, 1, 1, 1) * dx01_per_ctu
    active = torch.sigmoid((x_front_01 - x01.float()) / float(w_front))    # [B,1,H,W]
    transition = active * (1.0 - active)
    active = active * fluid_mask
    inactive = (1.0 - active) * fluid_mask
    return active, transition, inactive


def inflow_weight_map_seq(
    current_ctu_seq: Tensor,
    l_char_px: Tensor,
    static: Tensor,
    fluid_mask: Tensor,
    w_front: float = DEFAULT_W_FRONT,
    w_active: float = DEFAULT_W_ACTIVE,
    w_transition: float = DEFAULT_W_TRANSITION,
    w_inactive: float = DEFAULT_W_INACTIVE,
    roi_w_px: float = ROI_W_PX,
    x_inlet_01: float = X_INLET_01,
) -> Tensor:
    """為 trajectory 每步生成 region-weight map，供 weighted obs MAE 使用。

    weight = w_inactive + (w_active - w_inactive)*active + (w_transition - 1)*(4*transition)
    （4*transition 把鋒面帶峰值正規化到 1.0）。
    當 w_active=w_transition=w_inactive=1 → weight = fluid_mask（identity，退化為標準 fluid MAE）。

    Args:
        current_ctu_seq: [B, k] 每步步首 CTU（raw）。
        l_char_px: [B] ROI scale 特徵長度。
        static: [B, 4, H, W] normalized static（取 ch2 = X_01）。
        fluid_mask: [B, 1, H, W]。
        w_front: sigmoid 鋒面寬（X_01 單位）。
        w_active: 已抵達 active 區 loss 權重。
        w_transition: 鋒面帶 loss 權重。
        w_inactive: 尚未抵達區 loss 權重。
        roi_w_px: ROI 原始解析度寬（像素）。
        x_inlet_01: 入口 X_01 座標。

    Returns:
        weight maps：[B, k, 1, H, W]。
    """
    x01 = static[:, _X01_CH:_X01_CH + 1]    # [B,1,H,W]
    k = current_ctu_seq.shape[1]
    maps = []
    for i in range(k):
        active, transition, _ = compute_inflow_masks(
            current_ctu_seq[:, i], l_char_px, x01, fluid_mask,
            w_front=w_front, roi_w_px=roi_w_px, x_inlet_01=x_inlet_01,
        )
        base = w_inactive + (w_active - w_inactive) * active        # [B,1,H,W]
        front_emph = (w_transition - 1.0) * (4.0 * transition)
        wmap = (base + front_emph) * fluid_mask
        maps.append(wmap)
    return torch.stack(maps, dim=1)         # [B,k,1,H,W]


def obs_mphy_traj_mae_prior(
    pred: Tensor,
    target: Tensor,
    fluid_mask: Tensor,
    weight_map: Tensor | None,
) -> Tensor:
    """region-weighted trajectory MAE（HF-1c loss-mask）。

    weight_map=None 時等價於 ObsMphy_Traj_Mae（identity，標準 fluid MAE）。

    Args:
        pred: [B, k, 9, H, W] normalized Mphy prediction trajectory。
        target: [B, k, 9, H, W] normalized Mphy target trajectory。
        fluid_mask: [B, 1, H, W] float，流體=1 固體=0。
        weight_map: [B, k, 1, H, W] region weight map；None 時退化為標準 fluid MAE。

    Returns:
        scalar: region-weighted trajectory MAE。
    """
    diff = (pred - target).abs()            # [B,k,9,H,W]
    if weight_map is None:
        mask_bc = fluid_mask.unsqueeze(1).expand_as(diff)
        return (diff * mask_bc).sum() / mask_bc.sum().clamp_min(1.0)
    w = weight_map.expand_as(diff)          # [B,k,1,H,W] → [B,k,9,H,W]
    return (diff * w).sum() / w.sum().clamp_min(1.0)
