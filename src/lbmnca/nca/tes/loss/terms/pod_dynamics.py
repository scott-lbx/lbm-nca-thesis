"""pod_dynamics.py — 相空間切線/法線分解 loss。

turb 學極限環 → 在**固定 GT POD basis**（signed jx/jy fluct）投影 pred/GT 係數軌跡：
- **切線 loss**：pred 軌跡切向 dâ/dt 與 GT 切向 da/dt 對齊（1−cos）= 沿環流向動態對。
- **法線 loss**：pred 偏離 GT 環的**法向分量**（去掉切向相位 lag）→ 強制拉回環（橫向穩定）。

★固定 GT basis 是關鍵：pred 投影到 GT 渦街模態 → 對角條紋 artifact 係數≠GT → 同時罰結構+動力學
（治 diagonal-band）。norm-space fluct（pred 本就 norm fluct，與 phase-portrait 一致）。
"""
from __future__ import annotations

import torch
from torch import Tensor

_JX, _JY = 3, 5


def pod_tangent_normal_loss(
    pred_norm: Tensor,      # [B, T, 9, H, W] norm fluct（model 輸出軌跡）
    target_norm: Tensor,    # [B, T, 9, H, W] norm fluct（GT）
    basis: dict,            # {"modes":[m,2hw], "mean":[2hw], "stride":s}
    w_tangent: float = 1.0,
    w_normal: float = 1.0,
    eps: float = 1e-8,
) -> tuple[Tensor, float, float]:
    """回 (loss, l_tangent_val, l_normal_val)。"""
    B, T, C, H, W = pred_norm.shape
    s = int(basis["stride"])
    modes = basis["modes"]
    mean = basis["mean"]

    def _coeff_traj(x: Tensor) -> Tensor:
        jx = x[:, :, _JX, ::s, ::s].reshape(B * T, -1)
        jy = x[:, :, _JY, ::s, ::s].reshape(B * T, -1)
        vec = torch.cat([jx, jy], dim=1).float()            # [BT, 2hw]
        c = (vec - mean.to(vec)) @ modes.to(vec).t()         # [BT, m]
        return c.reshape(B, T, -1)

    cp = _coeff_traj(pred_norm)
    ct = _coeff_traj(target_norm)
    if T < 2:
        z = pred_norm.new_zeros(())
        return z, 0.0, 0.0

    # 切線：軌跡前向差分（相空間流向）
    tp = cp[:, 1:] - cp[:, :-1]      # [B,T-1,m] pred tangent
    tt = ct[:, 1:] - ct[:, :-1]      # GT tangent
    cos = (tp * tt).sum(-1) / (tp.norm(dim=-1) * tt.norm(dim=-1) + eps)
    l_tan = (1.0 - cos).mean()        # 0=完全對齊流向

    # 法線：偏離 GT 環、去掉切向相位 lag 後的法向分量 → 強制拉回環
    dev = cp[:, 1:] - ct[:, 1:]                              # [B,T-1,m] off-target
    ttu = tt / (tt.norm(dim=-1, keepdim=True) + eps)         # GT 切向單位向量
    dev_nor = dev - (dev * ttu).sum(-1, keepdim=True) * ttu  # 法向分量（橫向偏離）
    # 除以 GT 係數 norm：使 l_nor 無量綱（偏離量/GT 軌跡振幅），避免 POD 係數
    # O(24-88) 的原始量級壓制其他所有 loss term。
    gt_scale = ct[:, 1:].norm(dim=-1)                         # [B, T-1]
    l_nor = (dev_nor.norm(dim=-1) / (gt_scale + eps)).mean()

    loss = w_tangent * l_tan + w_normal * l_nor
    return loss, float(l_tan.detach()), float(l_nor.detach())
