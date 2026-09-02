"""KE proxy 與兩窗收斂判準（mean gate / rollout 共用）。

與資料格式無關的純函式，算的是「一條 KE 序列收斂了沒」，不綁在特定 loader 上。

語義與 LBM solver 端 twowindow 判準對齊。
"""
from __future__ import annotations

import numpy as np
import torch

#: D2Q9 moment 向量中的動量通道（``FieldConverter`` MPHY_*_IDX：
#: ``[rho, e, eps, jx, qx, jy, qy, pxx, pxy]``）。
JX_CH: int = 3
JY_CH: int = 5


def ke_norm(phys: torch.Tensor, fluid_mask: torch.Tensor) -> float:
    """norm-space momentum KE proxy：fluid 區 ``jx² + jy²`` 平均。

    ★是 proxy 不是物理 KE：在正規化空間直接取動量平方和，未除密度、未反正規化。
    只用於「收斂了沒」這種相對比較，不可當物理量引用。

    Args:
        phys: 物理場張量，shape ``[B, 9, H, W]``（norm space）。
        fluid_mask: fluid mask，shape ``[B, 1, H, W]``，fluid=1。

    Returns:
        fluid 區 KE proxy 平均值。
    """
    ke = phys[:, JX_CH].square() + phys[:, JY_CH].square()  # [B,H,W]
    fm2 = fluid_mask.squeeze(1)
    denom = fm2.sum().clamp_min(1.0)
    return float((ke * fm2).sum().item() / denom.item())


def two_window_ke(
    ke: list[float] | np.ndarray,
    ctu: list[float] | np.ndarray,
    *,
    window_ctu: float = 5.0,
    eps: float = 0.05,
) -> tuple[float, float]:
    """KE 曲線的 solver 式兩窗收斂判準（對齊 LBM solver 的 twowindow）。

    取最近 ``2×window_ctu`` 的 CTU 區間，依 CTU 軸分前後兩半窗，各取窗均
    ``m1``/``m2``，``ratio = |m2 − m1| / mean(m1, m2)``；``ratio < eps`` →
    converged。對 vortex-shedding 振盪 robust（窗均抵銷振幅，只留慢漂移）。

    Args:
        ke: 逐步 KE 值序列（與 ``ctu`` 等長平行）。
        ctu: 每步對應的累積 CTU 值（單調遞增）。
        window_ctu: 半窗大小（CTU）；每半窗應 ≥1 個脫落週期（典型 ~5）。
        eps: 宣告收斂的 ``|Δmean|/mean`` 門檻（典型 0.05）。

    Returns:
        ``(converged, ratio)``：``converged`` 為 0/1 float（資料不足回 0.0），
        ``ratio`` 為兩窗均相對差（資料不足回 nan）。
    """
    n = min(len(ke), len(ctu))
    if n < 4:
        return 0.0, float("nan")
    ctu_arr = np.asarray(ctu[:n], dtype=np.float64)
    ke_arr = np.asarray(ke[:n], dtype=np.float64)
    finite = np.isfinite(ctu_arr) & np.isfinite(ke_arr)
    ctu_arr, ke_arr = ctu_arr[finite], ke_arr[finite]
    if ctu_arr.size < 4:
        return 0.0, float("nan")
    t_end = float(ctu_arr[-1])
    t_mid = t_end - window_ctu
    t_start = t_end - 2.0 * window_ctu
    if t_start < float(ctu_arr[0]):
        return 0.0, float("nan")
    mask_w1 = (ctu_arr > t_start) & (ctu_arr <= t_mid)
    mask_w2 = (ctu_arr > t_mid) & (ctu_arr <= t_end)
    if int(mask_w1.sum()) < 2 or int(mask_w2.sum()) < 2:
        return 0.0, float("nan")
    m1 = float(np.mean(ke_arr[mask_w1]))
    m2 = float(np.mean(ke_arr[mask_w2]))
    mb = 0.5 * (m1 + m2)
    ratio = abs(m2 - m1) / (abs(mb) + 1e-10)
    return (1.0 if ratio < eps else 0.0), float(ratio)
