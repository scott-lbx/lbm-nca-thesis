"""Dilated-ResNet backbone（Stachenfeld et al. 2021「Learned Coarse Models」風格）。

提供 CFD 代理 baseline 的 dilated residual stack：以一連串「擴張卷積（dilated
conv）+ GroupNorm + SiLU」殘差塊取代下採樣，藉由逐塊循環的 dilation 讓感受野
（receptive field）先擴張再收縮，從而在**完全不改變空間尺寸**的前提下捕捉長程
依賴。此設計對任意 (H, W)（含 360×640 與不可整除的 70×130）皆成立。

協定（baseline `00_PROTOCOL.md`）：backbone 自身只做純映射
``[B, in_ch, H, W] → [B, out_ch, H, W]``；最後的 1×1 conv 採 **zero-init**，因為
外層 stepper 會以殘差（``out_residual``）方式組合輸出，近零初始可避免訓練初期
數值爆炸。

僅依賴 PyTorch（``torch`` / ``torch.nn`` / ``torch.nn.functional``），無外部相依，
import 時不做任何 print。
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DilatedResNetBackbone"]

# dilation 循環序列：感受野先擴張（1→8）再收縮（8→1），跨塊重複取用。
_DILATION_CYCLE: tuple[int, ...] = (1, 2, 4, 8, 4, 2, 1)


def _num_groups(channels: int) -> int:
    """為 GroupNorm 選一個能整除 ``channels`` 的合理組數。

    優先嘗試 32（GroupNorm 常見預設），逐步退讓到能整除的最大值，最差情況
    退回 1（等價 LayerNorm-over-channels），確保任意 width 都能建構。

    Args:
        channels: 該層的通道數。

    Returns:
        能整除 ``channels`` 的組數。
    """
    for g in (32, 16, 8, 4, 2):
        if channels % g == 0:
            return g
    return 1


class _DilatedResidualBlock(nn.Module):
    """單一擴張殘差塊：2×(Conv3x3[dilation=d] + GroupNorm + SiLU) + 殘差相加。

    透過設定 ``padding = dilation`` 保證輸出空間尺寸與輸入一致（SAME），因此不會
    有任何下採樣或尺寸漂移。

    Attributes:
        conv1: 第一個擴張 3×3 卷積。
        conv2: 第二個擴張 3×3 卷積。
        norm1: 第一個 GroupNorm。
        norm2: 第二個 GroupNorm。
        act: SiLU 激活（兩處共用，無參數）。
    """

    def __init__(self, width: int, dilation: int) -> None:
        """初始化擴張殘差塊。

        Args:
            width: 隱藏通道寬度（輸入與輸出通道數相同）。
            dilation: 此塊兩個卷積使用的擴張率；padding 同步設為此值以保尺寸。
        """
        super().__init__()
        groups = _num_groups(width)
        # padding=dilation → 'SAME' 空間尺寸（不分 H/W 是否可整除）。
        self.conv1 = nn.Conv2d(
            width, width, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(groups, width)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播。

        Args:
            x: 輸入張量 ``[B, width, H, W]``。

        Returns:
            ``[B, width, H, W]``，與輸入同空間尺寸的殘差輸出。
        """
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        return out + residual


class DilatedResNetBackbone(nn.Module):
    """Dilated-ResNet backbone（state→state CFD 代理）。

    結構：

        1. encoder 1×1 conv：``in_ch → drn_width``。
        2. ``drn_blocks`` 個 :class:`_DilatedResidualBlock`，dilation 依
           :data:`_DILATION_CYCLE` 循環取用。
        3. tail：GroupNorm + SiLU。
        4. head 1×1 conv：``drn_width → out_ch``，**zero-init weight 與 bias**。

    全程無下採樣，輸出空間尺寸與輸入完全一致；支援任意 (H, W)。

    Attributes:
        encoder: 進入隱藏空間的 1×1 卷積。
        blocks: 擴張殘差塊序列。
        tail_norm: 末端 GroupNorm。
        tail_act: 末端 SiLU。
        head: 輸出 1×1 卷積（zero-init）。
    """

    def __init__(self, in_ch: int, out_ch: int, bcfg: object) -> None:
        """初始化 backbone。

        Args:
            in_ch: 輸入通道數（physics + static + condition channels）。
            out_ch: 輸出通道數（通常為 physics_dim）。
            bcfg: BaselineConfig dataclass，需含整數屬性 ``drn_width``（隱藏寬度）
                與 ``drn_blocks``（殘差塊數）。
        """
        super().__init__()
        width = int(bcfg.drn_width)
        n_blocks = int(bcfg.drn_blocks)

        # encoder：1×1 conv 映射至隱藏寬度（不改空間尺寸）。
        self.encoder = nn.Conv2d(in_ch, width, kernel_size=1)

        # dilated residual stack：dilation 跨塊循環，感受野先擴後縮。
        blocks: list[nn.Module] = []
        for i in range(n_blocks):
            dilation = _DILATION_CYCLE[i % len(_DILATION_CYCLE)]
            blocks.append(_DilatedResidualBlock(width, dilation))
        self.blocks: nn.ModuleList = nn.ModuleList(blocks)

        # tail：先正規化+激活再投影。
        self.tail_norm = nn.GroupNorm(_num_groups(width), width)
        self.tail_act = nn.SiLU(inplace=True)

        # head：1×1 conv 投影回 out_ch；zero-init（殘差由外層 stepper 加總）。
        self.head = nn.Conv2d(width, out_ch, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播。

        Args:
            x: 輸入張量 ``[B, in_ch, H, W]``。

        Returns:
            ``[B, out_ch, H, W]``，與輸入同空間尺寸的輸出。
        """
        h = self.encoder(x)
        for block in self.blocks:
            h = block(h)
        h = self.tail_act(self.tail_norm(h))
        return self.head(h)
