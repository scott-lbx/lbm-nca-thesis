"""多尺度操作工具集（下採樣/上採樣/限制/延拓/自適應塊選擇）。

所有操作均為純 PyTorch 實作，保留梯度計算圖。
Mask 降採樣使用 area-pool + threshold 策略，避免雙線性模糊在粗格網邊界
將流體滲入固體格（solid cell）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """深度可分離 Gaussian blur（anti-alias/anti-image）。

    可分離 1D kernel × 2（省算），replicate padding，depthwise（groups=C）。dtype/device 跟隨 x。

    Args:
        x: ``[B, C, H, W]``。
        sigma: Gaussian 標準差（px）。

    Returns:
        ``[B, C, H, W]`` 模糊後張量（尺寸不變）。
    """
    r = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
    g = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    g = (g / g.sum()).to(x.dtype)
    C = x.shape[1]
    kx = g.view(1, 1, 1, -1).expand(C, 1, 1, 2 * r + 1)
    ky = g.view(1, 1, -1, 1).expand(C, 1, 2 * r + 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), ky, groups=C)
    return x


def downsample_state(
    state: torch.Tensor,
    ratio: int,
    mode: str = "area",
) -> torch.Tensor:
    """將狀態張量以整數倍率進行空間降採樣。

    支援兩種模式：
    - ``"area"``：使用 avg_pool2d，等同於 area-average 降採樣，適合
      保留物理量平均值（如速度場、壓力場）。
    - ``"bilinear"``：使用雙線性插值，適合需要平滑過渡的場景。

    Args:
        state: 輸入張量，形狀 [B, C, H, W]。
        ratio: 正整數降採樣倍率。
        mode: ``"area"``（avg_pool）或 ``"bilinear"``。

    Returns:
        降採樣後張量，形狀 [B, C, H//ratio, W//ratio]。

    Raises:
        ValueError: 若 mode 不在支援清單中。
    """
    if ratio == 1:
        return state
    if mode == "area":
        return F.avg_pool2d(state, kernel_size=ratio, stride=ratio)
    if mode == "blur":   # BlurPool：小固定 anti-alias Gaussian 再 subsample（治下採 folding）
        # σ cap 3.0（kernel≤19×19）：ratio/2 對大 ratio(32→σ16→97×97)爆記憶體/算力；小 blur 已除高頻混疊
        blurred = _gaussian_blur(state, sigma=min(max(0.5, ratio / 2.0), 3.0))
        return F.avg_pool2d(blurred, kernel_size=ratio, stride=ratio)
    if mode == "bilinear":
        h, w = state.shape[-2:]
        return F.interpolate(
            state,
            size=(max(1, h // ratio), max(1, w // ratio)),
            mode="bilinear",
            align_corners=False,
        )
    raise ValueError(f"downsample_state: unknown mode '{mode}'")


def upsample_state(
    state: torch.Tensor,
    target_hw: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """將狀態張量上採樣至目標空間大小。

    支援兩種模式：
    - ``"bilinear"``：雙線性插值，梯度平滑，適合多重網格延伸（prolongation）。
    - ``"nearest"``：最近鄰插值，保留邊界清晰，適合 mask 類別場上採樣。

    若輸入張量已符合目標大小，直接回傳原張量（節省記憶體）。

    Args:
        state: 輸入張量，形狀 [B, C, h, w]。
        target_hw: 目標空間大小 (H, W)。
        mode: ``"bilinear"`` 或 ``"nearest"``。

    Returns:
        上採樣後張量，形狀 [B, C, H, W]；若已符合目標大小則直接回傳原張量。

    Raises:
        ValueError: 若 mode 不在支援清單中。
    """
    h, w = state.shape[-2:]
    tgt_h, tgt_w = target_hw
    if h == tgt_h and w == tgt_w:
        return state
    if mode == "bilinear":
        return F.interpolate(
            state, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False
        )
    if mode == "nearest":
        return F.interpolate(state, size=(tgt_h, tgt_w), mode="nearest")
    if mode == "bicubic":
        # C1 連續：導數平滑，治渦度(curl)在 coarse cell 邊界的 ratio-px 格紋。
        return F.interpolate(
            state, size=(tgt_h, tgt_w), mode="bicubic", align_corners=False
        )
    if mode == "blur":   # blur-upsample：bilinear 後小 anti-image Gaussian（消 imaging replica → curl 不放大格紋）
        up = F.interpolate(state, size=(tgt_h, tgt_w), mode="bilinear", align_corners=False)
        _ratio = max(tgt_h / max(1, h), tgt_w / max(1, w))
        return _gaussian_blur(up, sigma=min(max(0.5, _ratio / 2.0), 3.0))   # σ cap 3.0（避大 ratio 巨核）
    raise ValueError(f"upsample_state: unknown mode '{mode}'")


def downsample_mask(
    mask: torch.Tensor,
    ratio: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """對流體 mask 進行降採樣（area-pool + threshold）。

    使用 avg_pool + threshold 而非雙線性插值，確保粗格網 mask
    維持乾淨的 {0, 1} 二值場，避免流固邊界滲漏。

    多重網格法中，若粗格網 mask 含有中間值（0~1），會導致邊界格
    被當作部分流體計算，引入物理不一致性。此函式以 threshold 轉換
    確保粗格網的嚴格二值性。

    Args:
        mask: 流體 mask 張量，形狀任意（最後兩維為空間維）。
        ratio: 正整數降採樣倍率。
        threshold: 判定為流體格的均值閾值，預設 0.5。

    Returns:
        降採樣後的二值 float mask，dtype=float32。
    """
    if ratio == 1:
        return mask.float()
    pooled = F.avg_pool2d(mask.float(), kernel_size=ratio, stride=ratio)
    return (pooled > threshold).float()


def make_block_mask(
    score: torch.Tensor,
    block_size: int,
    active_ratio: float,
) -> torch.Tensor:
    """依平均 score 選出 top-k block，產生 block 解析度 mask。

    自適應塊選擇邏輯：
    1. 以 avg_pool2d 計算每個 block 的平均 score（如殘差幅度）。
    2. 用 topk 選出 score 最高的 k 個 block（k = active_ratio × 總 block 數）。
    3. 輸出 block 解析度二值 mask，供後續 halo 膨脹與細網格 upsampling 使用。

    此機制允許模型在殘差大的區域（如邊界層、渦流區）分配更多計算資源，
    同時跳過遠場穩態區域，提升推論效率。

    Args:
        score: 非負 score 張量，形狀 [B, 1, H, W]（如殘差幅度）。
        block_size: 每個 block 的像素邊長。
        active_ratio: 標記為 active 的 block 比例，範圍 (0, 1]。

    Returns:
        block 解析度 mask，形狀 [B, 1, H//block_size, W//block_size]，
        top-k block 為 1，其餘為 0。

    Raises:
        ValueError: 若 score 形狀不符或 active_ratio 超出範圍。
    """
    if score.dim() != 4 or score.shape[1] != 1:
        raise ValueError(f"make_block_mask expects [B,1,H,W], got {tuple(score.shape)}")
    if not 0.0 < active_ratio <= 1.0:
        raise ValueError(f"make_block_mask: active_ratio must be in (0,1], got {active_ratio}")
    b, _, h, w = score.shape
    block_score = F.avg_pool2d(score.float(), kernel_size=block_size, stride=block_size)
    hb, wb = block_score.shape[-2:]
    flat = block_score.flatten(2)
    n_blocks = flat.shape[-1]
    k = max(1, min(n_blocks, int(round(active_ratio * n_blocks))))
    _, idx = torch.topk(flat, k=k, dim=-1)
    out = torch.zeros_like(flat)
    out.scatter_(-1, idx, 1.0)
    return out.view(b, 1, hb, wb)


def expand_block_halo(
    block_mask: torch.Tensor,
    halo_blocks: int = 1,
) -> torch.Tensor:
    """透過 max-pool 膨脹 block mask，擴展至鄰近 halo 範圍。

    自適應塊選擇後，活躍 block 的鄰近 block 也需要參與計算（因 NCA
    kernel 有感受野），否則邊界格的 NCA 更新將缺少鄰格資訊。
    此函式以 max_pool2d 實現形態學膨脹，將活躍區域向外擴展 halo_blocks 格。

    Args:
        block_mask: block 解析度二值 mask，形狀 [B, 1, Hb, Wb]。
        halo_blocks: 向外膨脹的 block 數，預設 1。

    Returns:
        膨脹後的二值 float mask，形狀與輸入相同。
    """
    if halo_blocks <= 0:
        return block_mask
    kernel = 2 * halo_blocks + 1
    pooled = F.max_pool2d(
        block_mask.float(), kernel_size=kernel, stride=1, padding=halo_blocks
    )
    return (pooled > 0).float()


def restrict_fine_to_coarse(
    fine_state: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """多重網格限制算子：fine → coarse（area-pool）。

    限制（restriction）是多重網格法的核心操作之一，將細格網殘差或狀態
    轉移至粗格網，使粗格網能夠修正低頻誤差分量。
    此實作採用 area-pool（等同平均降採樣），確保物理量守恆性。

    Args:
        fine_state: 細格網狀態張量，形狀 [B, C, H, W]。
        ratio: 粗細格網的空間倍率。

    Returns:
        粗格網狀態張量，形狀 [B, C, H//ratio, W//ratio]。
    """
    return downsample_state(fine_state, ratio, mode="area")


def prolong_coarse_to_fine(
    coarse_state: torch.Tensor,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """多重網格延伸算子：coarse → fine（雙線性插值）。

    延伸（prolongation）是多重網格法的另一核心操作，將粗格網修正量
    插值回細格網，供細格網做最終收斂修正。
    此實作採用雙線性插值，確保梯度平滑傳遞，避免最近鄰插值產生的
    階梯狀偽影干擾梯度計算圖。

    Args:
        coarse_state: 粗格網狀態張量，形狀 [B, C, h, w]。
        target_hw: 細格網目標空間大小 (H, W)。

    Returns:
        細格網狀態張量，形狀 [B, C, H, W]。
    """
    return upsample_state(coarse_state, target_hw, mode="bilinear")


__all__ = [
    "downsample_state",
    "upsample_state",
    "downsample_mask",
    "make_block_mask",
    "expand_block_halo",
    "restrict_fine_to_coarse",
    "prolong_coarse_to_fine",
]
