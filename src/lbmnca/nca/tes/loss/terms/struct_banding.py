"""Column-banding 結構損失——直接懲罰 03 部署稽核發現的垂直條紋（vertical stripe）偽影。

背景：SFA mean rollout 在 ρ/moments 產生**週期性垂直 column 條紋**（pressure=ρ/3、
vorticity 尤明顯；GT 無此高頻）。moment MSE 因 ρ 常數主導而遮蔽此缺陷（raw9 rel_l2 0.05
卻視覺 banding）。本項對 pred 沿 x（W 方向）的**高波數(週期≤40px)頻譜能量佔比**，懲罰其
**超出 target(GT)** 的量——只罰多出來的條紋、不動 GT 本有的合法 x 結構。

量化 gate（視覺缺陷檢測的 `banding_excess`）
的可微訓練版。作用於 ρ(ch0)+jx(ch3)+jy(ch5)。norm 空間即可（tanh 單調，banding 在 norm
亦存在，免 denorm）。

回傳 scalar Tensor（可微，支援 backward）。
mask 慣例：fluid_mask [B,1,H,W] float，流體=1 固體=0。
"""
from __future__ import annotations

import torch
from torch import Tensor

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5

# 條紋波數門檻：週期 ≤ HF_PERIOD_PX 視為條紋高頻（W=640 → 週期40px ≈ kx index 16）。
_HF_PERIOD_PX: float = 40.0
_EPS: float = 1e-8


def _column_hf_ratio(F: Tensor, fluid: Tensor) -> Tensor:
    """沿 x 高波數(column 條紋)頻譜能量佔比。

    Args:
        F: ``[B, H, W]`` 單通道場（norm 空間）。
        fluid: ``[B, H, W]`` 流體遮罩 float（流體=1）。

    Returns:
        ``[B]`` 每樣本高波數能量 / 總能量（∈[0,1]）。
    """
    W = F.shape[-1]
    Ff = F * fluid                                  # solid 區歸零
    Ff = Ff - Ff.mean(dim=-1, keepdim=True)         # 去每列均值，留 x 方向結構
    spec = torch.fft.rfft(Ff, dim=-1).abs() ** 2    # [B, H, Wf]
    P = spec.mean(dim=1)                            # 沿列平均 → [B, Wf]
    kc = max(1, int(W / _HF_PERIOD_PX))             # 高波數起始 index
    hf = P[..., kc:].sum(dim=-1)                    # 條紋高頻能量
    tot = P[..., 1:].sum(dim=-1) + _EPS             # 總能量（去 DC）
    return hf / tot


def struct_column_banding_loss(
    pred: Tensor, target: Tensor, fluid_mask: Tensor, normalizer=None
) -> Tensor:
    """pred 相對 target 之 column 條紋(垂直高頻)超出量，對 ρ/jx/jy 平均。

    Args:
        pred: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` 流體遮罩 float。
        normalizer: 未使用（norm 空間直接計算；保留簽名一致）。

    Returns:
        scalar Tensor：Σ_ch mean_bk relu(hf_ratio(pred) − hf_ratio(target)) / n。
    """
    B, k = pred.shape[0], pred.shape[1]
    fluid = fluid_mask[:, 0]                        # [B, H, W]
    total = pred.new_zeros(())
    n = 0
    for ch in (RHO_CH, JX_CH, JY_CH):
        for t in range(k):
            pb = _column_hf_ratio(pred[:, t, ch], fluid)     # [B]
            tb = _column_hf_ratio(target[:, t, ch], fluid)   # [B]
            total = total + torch.relu(pb - tb).mean()
            n += 1
    return total / max(1, n)


import torch.nn.functional as F  # noqa: E402


def _dir_band_map(f: Tensor, blur_k: int = 15) -> Tensor:
    """方向性 band 圖：x-梯度能量 − y-梯度能量（局部平均）。

    band（垂直條紋）= 沿 x 高頻振盪 → ∂/∂x 能量遠大於 ∂/∂y → dirmap>0。
    smooth 自由流 → 兩者皆小 → dirmap≈0。建物銳邊由 free-stream mask 排除（另處理）。

    Args:
        f: ``[B, H, W]`` 單通道場（norm 空間）。
        blur_k: 局部聚合窗（px），涵蓋 band 波長尺度（13–40px）。

    Returns:
        ``[B, H, W]`` 方向性能量圖（>0=x-梯度主導=條紋傾向）。
    """
    gx = F.pad(f[..., 1:] - f[..., :-1], (0, 1))          # ∂/∂x（沿 W）
    gy = F.pad(f[:, 1:, :] - f[:, :-1, :], (0, 0, 0, 1))  # ∂/∂y（沿 H）
    dirmap = gx ** 2 - gy ** 2                            # >0 → 垂直條紋傾向
    pad = blur_k // 2
    return F.avg_pool2d(dirmap[:, None], blur_k, stride=1, padding=pad)[:, 0]  # 局部聚合 coherence


def struct_directional_band_loss(
    pred: Tensor, target: Tensor, fluid_mask: Tensor, normalizer=None,
    freestream_erode: int = 6, blur_k: int = 15, eps: float = _EPS,
) -> Tensor:
    """★SFA2-2x（Idea A，NCA 文獻挖掘）：空間局域方向性 band loss。

    頻域三連敗根因＝band 與銳邊同 kx 不可分。改用**空間局域**軸：band 是**自由流區
    大面積 x-梯度主導**（全域相干），建物銳邊是**貼牆局部**（被 free-stream mask 排除）。
    在自由流區（fluid 且遠離 solid ``freestream_erode`` px）罰 pred 的 x-梯度主導**超出 GT**。
    作用於 jx(3)/jy(5)。norm 空間（banding 於 norm 亦存，tanh 單調）。

    Args:
        pred: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        fluid_mask: ``[B, 1, H, W]`` 流體遮罩 float。
        normalizer: 未使用（norm 空間直接計算；保留簽名一致）。
        freestream_erode: 自由流侵蝕半徑（px）；排除建物邊界近牆區。
        blur_k: 方向性圖局部聚合窗。

    Returns:
        scalar Tensor：自由流區 relu(dir_pred − dir_gt) 均值，對 jx/jy×k 平均。
    """
    B, k = pred.shape[0], pred.shape[1]
    fluid = fluid_mask[:, 0]                                          # [B,H,W]
    ks = 2 * freestream_erode + 1
    solid_dil = F.max_pool2d((1.0 - fluid)[:, None], ks, stride=1, padding=freestream_erode)[:, 0]
    freestream = (fluid * (1.0 - solid_dil).clamp(0.0, 1.0))         # fluid 且遠離 solid
    total = pred.new_zeros(())
    n = 0
    for ch in (JX_CH, JY_CH):
        for t in range(k):
            dp = _dir_band_map(pred[:, t, ch], blur_k)
            dg = _dir_band_map(target[:, t, ch], blur_k)
            excess = (torch.relu(dp - dg) * freestream).sum() / (freestream.sum() + eps)
            total = total + excess
            n += 1
    return total / max(1, n)


__all__ = ["struct_column_banding_loss", "struct_directional_band_loss"]
