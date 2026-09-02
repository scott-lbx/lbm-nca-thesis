"""現代 U-Net backbone（CFD surrogate baseline，P_new 自回歸用）。

提供 `ModernUNetBackbone`：純 [B, in_ch, H, W] → [B, out_ch, H, W] 的映射，
H、W **保持不變**。殘差組合（`out = phys_in + backbone(x)`）由外層 stepper
負責（見 `baseline_stepper.py`），本 backbone 自身只做場到場的轉換。

設計重點：
    * 現代 U-Net 結構：encoder 以 stride-2 conv 下採樣 `unet_depth` 層，channel
      依 base、2*base、4*base… 倍增；bottleneck；decoder 以 transpose-conv 上採樣
      並與 encoder skip concat。每個 block = 2×(Conv3x3 + GroupNorm + SiLU)。
    * **任意 H、W**：U-Net 下採樣要求邊長能被 ``2**unet_depth`` 整除。本實作在
      forward 入口用 ``F.pad``（replicate）把輸入補到該倍數，跑完 U-Net 後再
      crop 回原始 H、W。因此 360×640（可整除 8）與 70×130（不可整除）皆正確。
    * **末層 1×1 conv 採 zero-init**（weight、bias 皆 0）：stepper 以殘差加回輸出，
      近零初始化可避免訓練初期數值爆炸。

僅依賴 torch，無外部相依。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ModernUNetBackbone"]


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """建立通道數安全的 GroupNorm。

    取不超過 8 且能整除 ``num_channels`` 的最大群數，確保任意 base channel
    （含非 8 倍數）都合法。

    Args:
        num_channels: 欲正規化的通道數。

    Returns:
        對應的 ``nn.GroupNorm`` 模組。
    """
    num_groups = min(8, num_channels)
    # 往下找能整除的群數（GroupNorm 要求 num_channels % num_groups == 0）
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class _DoubleConv(nn.Module):
    """雙層卷積區塊：2×(Conv3x3 + GroupNorm + SiLU)。

    U-Net 的基本運算單元，encoder/decoder/bottleneck 共用。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """初始化雙卷積 block。

        Args:
            in_ch: 輸入通道數。
            out_ch: 輸出通道數。
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """套用雙卷積。

        Args:
            x: 形狀 ``[B, in_ch, H, W]`` 的輸入。

        Returns:
            形狀 ``[B, out_ch, H, W]`` 的輸出。
        """
        return self.block(x)


class ModernUNetBackbone(nn.Module):
    """現代 U-Net backbone，輸出空間尺寸與輸入相同（任意 H、W）。

    encoder 以 stride-2 conv 下採樣 ``bcfg.unet_depth`` 層，bottleneck 後以
    transpose-conv 上採樣並接 skip。forward 入口先 pad 到 ``2**unet_depth``
    的倍數、跑完再 crop 回原尺寸，故支援非整除邊長（如 70×130）。

    Attributes:
        depth: 下採樣/上採樣層數（``bcfg.unet_depth``）。
        in_ch: 輸入通道數。
        out_ch: 輸出通道數。
    """

    def __init__(self, in_ch: int, out_ch: int, bcfg: object) -> None:
        """建構 U-Net。

        Args:
            in_ch: 輸入通道數（physics + static + cond）。
            out_ch: 輸出通道數（physics_dim）。
            bcfg: ``BaselineConfig``，需含整數屬性 ``unet_base``、``unet_depth``。

        Raises:
            ValueError: ``unet_base`` 或 ``unet_depth`` 不為正整數時拋出。
        """
        super().__init__()
        base = int(bcfg.unet_base)
        depth = int(bcfg.unet_depth)
        if base <= 0 or depth < 1:
            raise ValueError(f"unet_base、unet_depth 必須為正整數，得到 base={base}, depth={depth}")

        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.depth = depth

        # 各層通道數：base、2*base、4*base...（encoder 由淺到深）
        enc_channels = [base * (2 ** i) for i in range(depth)]

        # ── Encoder：stem(DoubleConv) + 每層 stride-2 conv 下採樣 + DoubleConv ──
        self.stem = _DoubleConv(self.in_ch, enc_channels[0])
        self.down_samplers: nn.ModuleList = nn.ModuleList()
        self.down_convs: nn.ModuleList = nn.ModuleList()
        for i in range(depth):
            ch = enc_channels[i]
            nxt = enc_channels[i + 1] if i + 1 < depth else ch * 2
            # stride-2 conv 下採樣（取代 maxpool，保留可學習性）
            self.down_samplers.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
            self.down_convs.append(_DoubleConv(ch, nxt))

        # ── Bottleneck 通道數 = 最深 encoder 輸出 ──
        bottleneck_ch = enc_channels[-1] * 2

        # ── Decoder：每層 transpose-conv 上採樣 + skip concat + DoubleConv ──
        self.up_samplers: nn.ModuleList = nn.ModuleList()
        self.up_convs: nn.ModuleList = nn.ModuleList()
        cur = bottleneck_ch
        for i in reversed(range(depth)):
            skip = enc_channels[i]
            # 上採樣到 skip 的通道數，再與 skip concat（→ 2*skip）後 DoubleConv 回 skip
            self.up_samplers.append(
                nn.ConvTranspose2d(cur, skip, kernel_size=2, stride=2)
            )
            self.up_convs.append(_DoubleConv(skip * 2, skip))
            cur = skip

        # ── 末層 1×1 conv：zero-init（殘差加回前近零，避免初期爆炸）──
        self.head = nn.Conv2d(cur, self.out_ch, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播，輸出空間尺寸等於輸入。

        Args:
            x: 形狀 ``[B, in_ch, H, W]`` 的輸入場。

        Returns:
            形狀 ``[B, out_ch, H, W]`` 的輸出（H、W 與輸入相同）。
        """
        orig_h, orig_w = x.shape[-2], x.shape[-1]

        # ── 補邊到 2**depth 的倍數（replicate 邊界，避免引入虛假 0 邊界）──
        multiple = 2 ** self.depth
        pad_h = (multiple - orig_h % multiple) % multiple
        pad_w = (multiple - orig_w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            # F.pad 順序為 (left, right, top, bottom)；補在右側與下側
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        # ── Encoder：保存各層 skip ──
        skips: list[torch.Tensor] = []
        h = self.stem(x)
        for i in range(self.depth):
            skips.append(h)                       # 下採樣前的特徵當 skip
            h = self.down_samplers[i](h)          # stride-2 下採樣
            h = self.down_convs[i](h)

        # ── Decoder：上採樣 → 與對應 skip concat → DoubleConv ──
        for j in range(self.depth):
            skip = skips[self.depth - 1 - j]      # 由深到淺取 skip
            h = self.up_samplers[j](h)
            # transpose-conv 後尺寸恆等於 skip（因 pad 已對齊到 2**depth 倍數），
            # 仍保險裁切以容忍極端情況
            if h.shape[-2:] != skip.shape[-2:]:
                h = h[..., : skip.shape[-2], : skip.shape[-1]]
            h = torch.cat([skip, h], dim=1)
            h = self.up_convs[j](h)

        out = self.head(h)

        # ── crop 回原始 H、W ──
        if pad_h > 0 or pad_w > 0:
            out = out[..., :orig_h, :orig_w]
        return out
