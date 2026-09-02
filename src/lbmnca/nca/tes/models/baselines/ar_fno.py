"""2D Fourier Neural Operator（FNO2d）backbone — CFD surrogate baseline。

提供標準 FNO2d 主幹（Li et al. 2021, *Fourier Neural Operator for Parametric
PDEs*）作為自回歸 baseline 的 backbone。契約上只做純空間映射：
`[B, in_ch, H, W] -> [B, out_ch, H, W]`，且 **嚴格保持 H, W 不變**，由外層
`BaselineStepperNCA` 以殘差方式組合（`new_phys = phys_in + backbone(x)`）。

設計要點：
    1. lift（1×1 conv）：`in_ch -> fno_width`。
    2. `fno_layers` 個 spectral-conv block，每個 block =
       `SpectralConv2d`（rfft2 截斷至 `fno_modes` → 複數權重 → irfft2）
       與並聯的 1×1 pointwise conv 殘差相加，再過 GELU。
    3. project（兩層 1×1 conv）：`fno_width -> out_ch`；末層 zero-init。
    4. AMP-safe：頻域運算強制 float32（`torch.autocast(..., enabled=False)`），
       複數權重以 float32 儲存；block 輸出再 cast 回原 dtype。
    5. mode 截斷上限隨輸入尺寸 clamp（`<= H`、`<= W//2+1`），任意尺寸（例如
       360×640 或非 2 次冪）皆可**執行**；但 `modes1 >= H` 時 `SpectralConv2d`
       的 `weight1` 會被 `weight2` 覆寫而失效（見該類別 docstring 的「★已知限制」），
       此情況下「任意尺寸皆可運作」僅指不會報錯，不代表所有權重都有作用。

baseline 公平性靠 param budget；backbone 自身不含條件注入，條件由 stepper 以
channel concat 提供。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FNO2DBackbone"]


@runtime_checkable
class _BaselineConfigLike(Protocol):
    """BaselineConfig 的結構型介面（供 type checker 使用，不影響執行期行為）。

    定義 FNO2DBackbone 實際存取的三個屬性，讓 type checker 在傳入非相容物件時
    能發出警告，同時維持對既有 dataclass 的 duck-typing 相容性。
    """

    fno_modes: int
    fno_width: int
    fno_layers: int


class SpectralConv2d(nn.Module):
    """2D 頻域卷積層（FNO 核心算子）。

    對輸入做 `rfft2` 取得頻譜，僅保留低頻 `modes` 個 mode（兩個空間維），與
    可學複數權重做逐 mode 的 channel 混合（einsum），再以 `irfft2` 還原至原始
    空間尺寸 `(H, W)`。頻域運算全程以 float32 執行以確保 AMP 數值穩定。

    複數權重以「實部/虛部分離的 float32 tensor」儲存（避免部分後端對複數參數
    與 AMP 的相容問題），於 forward 內組裝成 complex64 使用。

    ★已修：
        **舊行為**：forward 先寫 `out_ft[:, :, :m1, :m2]`（weight1）再寫
        `out_ft[:, :, -m1:, :m2]`（weight2），其中 `m1 = min(modes1, H)`。
        當 `modes1 >= H` 時 `m1 == H`，後者切片涵蓋全部列 → weight1 的輸出被完全
        覆寫、且拿不到梯度，該解析度下等於只有 weight2 在作用。實測顯示：當
        coarse cell 的空間解析度小於設定的 mode 數上限時，weight1 張量會因此
        完全零梯度、恆為零，占 spectral 參數相當比例。

        **新行為**：兩段各自 clamp——`m1_lo = min(modes1, (H+1)//2)` 取低正頻、
        `m1_hi = min(modes1, H//2)` 取高負頻，兩者相加恆 ``<= H`` ⇒ **切片永不重疊**，
        weight1 與 weight2 都拿得到梯度。這是標準 FNO 的語義（正負頻各一組權重）。

        **相容性**：`modes1 < H/2` 的模型逐位元行為不變（兩段的 clamp 值都等於
        `modes1`，與舊碼相同）；只有原本會重疊者（`modes1 >= H/2`）行為改變，
        這些 ckpt 需重訓。
        行為測試見 `tests/nca/tes/test_spectral_conv_weight1_overwrite.py`。

    Attributes:
        in_ch: 輸入通道數。
        out_ch: 輸出通道數。
        modes1: 第一空間維（H 方向）保留的 Fourier mode 上限。
        modes2: 第二空間維（W 方向，rfft 半譜）保留的 Fourier mode 上限。
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        """初始化頻域卷積層。

        Args:
            in_ch: 輸入通道數。
            out_ch: 輸出通道數。
            modes1: H 方向保留的 mode 數（會在 forward 中再隨輸入尺寸 clamp）。
            modes2: W 方向保留的 mode 數（同上）。
        """
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        # 縮放因子（標準 FNO 初始化），避免深層頻譜權重過大。
        scale = 1.0 / (in_ch * out_ch)
        # rfft2 後左上/左下兩個角落各需一組權重（共軛對稱僅覆蓋半譜的右半）。
        # 權重以 float32 實部/虛部分離儲存，shape [in, out, modes1, modes2]。
        self.weight1_real = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.float32)
        )
        self.weight1_imag = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.float32)
        )
        self.weight2_real = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.float32)
        )
        self.weight2_imag = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.float32)
        )

    @staticmethod
    def _complex_mul2d(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """逐 mode 的 channel 混合（complex einsum）。

        Args:
            inp: 頻譜切片，complex tensor，shape [B, in_ch, m1, m2]。
            weight: 複數權重，shape [in_ch, out_ch, m1, m2]。

        Returns:
            混合後頻譜，complex tensor，shape [B, out_ch, m1, m2]。
        """
        return torch.einsum("bixy,ioxy->boxy", inp, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """執行頻域卷積，輸出空間尺寸與輸入相同。

        Args:
            x: 輸入特徵圖（已為 float32），shape [B, in_ch, H, W]。

        Returns:
            float32 特徵圖，shape [B, out_ch, H, W]。
        """
        batch, _, height, width = x.shape
        # rfft 半譜寬度。
        w_half = width // 2 + 1
        m2 = min(self.modes2, w_half)
        # H 方向是完整 FFT，低正頻在前段、高負頻在
        #   尾段。原碼兩段都用 `m1 = min(modes1, height)`，當 modes1 >= height 時
        #   `:m1` 與 `-m1:` 同為全部列 → weight2 完全覆寫 weight1、weight1 拿不到
        #   梯度。
        #   改為兩段各自 clamp，保證 m1_lo + m1_hi <= height ⇒ **兩塊永不重疊**：
        #     height=45, modes1=48 → 23 + 22 = 45（剛好覆蓋、不重疊）
        #     height=11, modes1=48 → 6 + 5 = 11
        #     height=100, modes1=48 → 48 + 48 = 96 < 100（原本就不重疊，行為不變）
        #     height=1 → 1 + 0（無負頻，只寫 weight1）
        #   ★modes1 < height/2 的模型逐位元行為不變；只有原本會重疊者改變 → 需重訓。
        m1_lo = min(self.modes1, (height + 1) // 2)   # 低正頻列數（含 DC）
        m1_hi = min(self.modes1, height // 2)         # 高負頻列數（不含 DC）

        # 頻域變換（強制 float32，輸入已於外層 cast）。
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

        # 組裝 complex64 權重（各取用該段實際用到的低頻子區塊）。
        weight1 = torch.complex(
            self.weight1_real[:, :, :m1_lo, :m2], self.weight1_imag[:, :, :m1_lo, :m2]
        )

        out_ft = torch.zeros(
            batch, self.out_ch, height, w_half, dtype=torch.cfloat, device=x.device
        )
        # 左上角（低正頻 H × 低頻 W）——weight1。
        out_ft[:, :, :m1_lo, :m2] = self._complex_mul2d(
            x_ft[:, :, :m1_lo, :m2], weight1
        )
        # 左下角（高負頻 H，fft 尾端 m1_hi 列 × 低頻 W）——weight2。
        # m1_hi == 0 只在 height == 1 時發生（無負頻列），此時不寫第二塊。
        if m1_hi > 0:
            weight2 = torch.complex(
                self.weight2_real[:, :, :m1_hi, :m2],
                self.weight2_imag[:, :, :m1_hi, :m2],
            )
            out_ft[:, :, -m1_hi:, :m2] = self._complex_mul2d(
                x_ft[:, :, -m1_hi:, :m2], weight2
            )

        # 逆變換回原始空間尺寸（s 明確指定以保持 H, W）。
        out = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1), norm="ortho")
        return out


class _FNOBlock(nn.Module):
    """單一 FNO block：spectral conv ‖ 1×1 pointwise conv 殘差 → GELU。

    頻域分支（`SpectralConv2d`）與空間 pointwise 分支（1×1 conv）並聯相加後過
    GELU 非線性。整個 block 的頻域運算在 forward 中以 float32 執行（AMP-safe）。
    """

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        """初始化 FNO block。

        Args:
            width: 通道寬度（in == out == width）。
            modes1: H 方向保留 mode 數。
            modes2: W 方向保留 mode 數。
        """
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """執行 block 前向。

        AMP-safe：在 `torch.autocast(..., enabled=False)` 下將輸入 cast 至
        float32 計算頻域與 pointwise 分支，最後將結果 cast 回原 dtype。

        Args:
            x: 輸入特徵圖，shape [B, width, H, W]，可為 fp16/bf16/fp32。

        Returns:
            與輸入同 dtype、同 H, W 的特徵圖，shape [B, width, H, W]。
        """
        in_dtype = x.dtype
        # 頻域 / 複數運算強制 float32（關閉 autocast）。
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            spec = self.spectral(x32)
            point = self.pointwise(x32)
            out = F.gelu(spec + point)
        # cast 回呼叫端 dtype，融入外層 AMP 流程。
        return out.to(in_dtype)


class FNO2DBackbone(nn.Module):
    """標準 2D FNO backbone（純空間映射，保持 H, W）。

    結構：lift(1×1) → `fno_layers` 個 `_FNOBlock` → project(兩層 1×1)。輸出
    空間尺寸恆等於輸入，供外層 stepper 以殘差組合。末層 project conv 以
    zero-initialization 初始化（weight 與 bias 皆為 0），確保訓練初期 backbone
    輸出近 0、殘差不爆炸。

    Args:
        in_ch: 輸入通道數（physics + static + cond，由 stepper 計算）。
        out_ch: 輸出通道數（physics_dim）。
        bcfg: BaselineConfig dataclass，需含整數屬性 `fno_modes`、`fno_width`、
            `fno_layers`。實際型別為 `Any` 以相容不同版本的 BaselineConfig
            dataclass；執行期透過 duck-typing 存取屬性。
    """

    def __init__(self, in_ch: int, out_ch: int, bcfg: Any) -> None:
        """初始化 FNO2d backbone。

        Args:
            in_ch: 輸入通道數。
            out_ch: 輸出通道數。
            bcfg: BaselineConfig，需含 `fno_modes`、`fno_width`、`fno_layers`
                  三個整數屬性。型別標為 Any 以相容不同版本的 BaselineConfig
                  dataclass（實際介面見 `_BaselineConfigLike` Protocol）。
        """
        super().__init__()
        modes = int(bcfg.fno_modes)
        width = int(bcfg.fno_width)
        n_layers = int(bcfg.fno_layers)

        # lift：in_ch -> width（1×1 conv）。
        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)

        # fno_layers 個 spectral-conv block（兩維皆截斷至 modes）。
        self.blocks = nn.ModuleList(
            [_FNOBlock(width, modes, modes) for _ in range(n_layers)]
        )

        # project：width -> width -> out_ch（兩層 1×1 conv，中間接 GELU）。
        proj_hidden = width * 2
        self.proj1 = nn.Conv2d(width, proj_hidden, kernel_size=1)
        self.proj2 = nn.Conv2d(proj_hidden, out_ch, kernel_size=1)

        # 末層 zero-init（殘差組合下避免初期 blow-up）。
        nn.init.zeros_(self.proj2.weight)
        if self.proj2.bias is not None:
            nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FNO2d 前向，輸出空間尺寸與輸入相同。

        Args:
            x: 輸入張量，shape [B, in_ch, H, W]。

        Returns:
            輸出張量，shape [B, out_ch, H, W]（H, W 與輸入一致）。
        """
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x
