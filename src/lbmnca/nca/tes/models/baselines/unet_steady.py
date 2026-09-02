"""穩態 U-Net backbone（Thuerey et al. 2020 "Deep-Flow-Prediction" 範式）。

本檔實作 P_old（geometry→final-field）範式的 baseline backbone：一次性
（one-shot，無時間推進）把幾何 + 條件 channel 直接映射成已發展完成的穩態流場。

與自回歸 baseline（FNO / modern U-Net / Dilated-ResNet = P_new，state→state）
的關鍵差異：

* P_old **結構上無法**接受「前一時刻流場」當輸入 —— 它只吃 geometry+cond，
  直接輸出 developed field。因此這是論文 CH1 的「舊範式」對照組，用來凸顯
  其無法續推（warm-init 無效）的本質限制。
* 包覆的 adapter（``SteadyStateUNetNCA``）**不加殘差**，backbone 必須從零
  產生完整流場 → 故末層採一般初始化（**不 zero-init**），有別於自回歸
  baseline 的殘差 + 末層 zero-init 慣例。

架構為 Thuerey 式對稱 U-Net：``unet_depth`` 層 strided-conv 編碼器
（channel base→2·base→4·base…）、bottleneck、對稱解碼器（transpose-conv
上採樣 + skip concat）。編碼 block 用 LeakyReLU、解碼 block 用 ReLU，
皆配 BatchNorm，末層 1×1 conv 投影到 ``out_ch``。

為支援任意 H、W（含非 ``2**unet_depth`` 倍數，如 70×130），forward 內先把
輸入 F.pad 到該倍數，跑完網路後再 crop 回原始 H、W。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SteadyUNetBackbone"]


class _DownBlock(nn.Module):
    """編碼器下採樣 block：strided 3×3 conv → BatchNorm → LeakyReLU。

    空間尺寸減半（stride=2），channel 由 ``in_ch`` 變 ``out_ch``。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """初始化下採樣 block。

        Args:
            in_ch: 輸入通道數。
            out_ch: 輸出通道數。
        """
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播。

        Args:
            x: 形狀 ``[B, in_ch, H, W]`` 的輸入。

        Returns:
            形狀 ``[B, out_ch, H/2, W/2]`` 的輸出（空間尺寸減半）。
        """
        return self.act(self.norm(self.conv(x)))


class _UpBlock(nn.Module):
    """解碼器上採樣 block：transpose-conv 上採樣 → skip concat → 3×3 conv 融合。

    先以 stride=2 transpose-conv 將空間尺寸加倍，與對應編碼層 skip 串接後，
    用一般 3×3 conv → BatchNorm → ReLU 融合。
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        """初始化上採樣 block。

        Args:
            in_ch: 上採樣前輸入的通道數。
            skip_ch: 對應 encoder skip connection 的通道數。
            out_ch: 融合後輸出的通道數。
        """
        super().__init__()
        # 上採樣（空間 ×2），channel in_ch → out_ch
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # skip concat 後融合（out_ch + skip_ch → out_ch）
        self.conv = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """前向傳播。

        Args:
            x: 形狀 ``[B, in_ch, H, W]`` 的深層特徵輸入。
            skip: 形狀 ``[B, skip_ch, H*2, W*2]`` 的 encoder skip 特徵。

        Returns:
            形狀 ``[B, out_ch, H*2, W*2]`` 的融合輸出。
        """
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)   # skip concatenation
        return self.act(self.norm(self.conv(x)))


class SteadyUNetBackbone(nn.Module):
    """穩態 geometry→field 對稱 U-Net backbone（Thuerey 範式，P_old）。

    一次性把 ``[B, in_ch, H, W]`` 幾何 + 條件 channel 映射成 ``[B, out_ch, H, W]``
    的已發展穩態流場（輸出空間尺寸與輸入相同）。結構上不接受前次流場，
    為論文 CH1「舊範式」對照組。

    Attributes:
        stem: 入口投影卷積 block（in_ch → base）。
        downs: 編碼器下採樣 block 序列。
        bottleneck: 最深層特徵融合 block。
        ups: 解碼器上採樣 block 序列。
        head: 末層 1×1 投影卷積（一般初始化，非 zero-init）。
    """

    def __init__(self, in_ch: int, out_ch: int, bcfg: object) -> None:
        """初始化穩態 U-Net backbone。

        Args:
            in_ch: 輸入 channel 數（static 幾何 + broadcast 條件）。
            out_ch: 輸出 channel 數（physics_dim）。
            bcfg: ``BaselineConfig`` dataclass，需含整數屬性 ``unet_base``
                （base channel 數）與 ``unet_depth``（下/上採樣層數）。

        Raises:
            ValueError: ``unet_depth`` 小於 1 時拋出。
        """
        super().__init__()
        base: int = int(bcfg.unet_base)
        depth: int = int(bcfg.unet_depth)
        if depth < 1:
            raise ValueError(f"unet_depth 必須 >=1，得到 {depth}")
        self._depth = depth
        # 對齊需求：輸入須 pad 到 2**depth 的倍數
        self._multiple = 2 ** depth

        # 入口投影：in_ch → base（保持空間尺寸）
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 各層 channel 規劃：base, 2·base, 4·base, ...
        enc_chs = [base * (2 ** i) for i in range(depth + 1)]  # 含 bottleneck

        # 編碼器：depth 個 strided 下採樣 block
        self.downs: nn.ModuleList = nn.ModuleList(
            [_DownBlock(enc_chs[i], enc_chs[i + 1]) for i in range(depth)]
        )

        # bottleneck：最深層特徵融合（不改空間尺寸）
        bott_ch = enc_chs[depth]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bott_ch, bott_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bott_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 解碼器：對稱 depth 個上採樣 block，skip 取對應編碼層輸出
        # 第 j 個 up（j 從深到淺）：輸入 enc_chs[depth-j]，skip enc_chs[depth-1-j]，
        # 輸出 enc_chs[depth-1-j]
        self.ups: nn.ModuleList = nn.ModuleList(
            [
                _UpBlock(
                    in_ch=enc_chs[depth - j],
                    skip_ch=enc_chs[depth - 1 - j],
                    out_ch=enc_chs[depth - 1 - j],
                )
                for j in range(depth)
            ]
        )

        # 末層 1×1 conv 投影到 out_ch。
        # 注意：此為 P_old 直接預測器（adapter 不加殘差），故 **不 zero-init**，
        # 採 Conv2d 預設一般初始化，使其能從零產生完整流場。
        self.head = nn.Conv2d(base, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播，輸出空間尺寸等於輸入。

        為支援任意 H、W（含非 ``2**depth`` 倍數），先把輸入右/下 pad 到該倍數，
        跑完 U-Net 後再 crop 回原始 H、W。

        Args:
            x: 形狀 ``[B, in_ch, H, W]`` 的幾何 + 條件輸入。

        Returns:
            形狀 ``[B, out_ch, H, W]`` 的穩態流場輸出（H、W 與輸入相同）。
        """
        h, w = x.shape[-2], x.shape[-1]
        m = self._multiple
        # 計算 pad 量（補到 2**depth 的倍數；reflect pad 不適用小尺寸 → 用 replicate）
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m
        if pad_h or pad_w:
            # F.pad 順序：(left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        # 編碼：stem → 逐層下採樣，保留各層 skip
        x = self.stem(x)
        skips: list[torch.Tensor] = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # bottleneck（最深層 skips[-1]）
        x = self.bottleneck(x)

        # 解碼：逐層上採樣 + skip concat（skip 取較淺一層的編碼輸出）
        for j, up in enumerate(self.ups):
            skip = skips[self._depth - 1 - j]
            x = up(x, skip)

        # 末層投影
        x = self.head(x)

        # crop 回原始尺寸
        if pad_h or pad_w:
            x = x[..., :h, :w]
        return x
