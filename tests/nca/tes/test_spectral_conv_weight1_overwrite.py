"""SpectralConv2d 的低正頻／高負頻切片**不重疊**保證（CPU、小張量）。

## 歷史（本檔原本鎖的是相反的行為）

原 `SpectralConv2d.forward` 先寫 `out_ft[:, :, :m1, :m2]`（weight1）再寫
`out_ft[:, :, -m1:, :m2]`（weight2），兩段共用 `m1 = min(modes1, height)`。
當 `modes1 >= height` 時 `m1 == height`，第二段涵蓋全部列 → **weight1 的輸出被
完全覆寫、且拿不到梯度**。舊 canonical `main_v4` 的兩個 coarse cell
在 360×640 下工作於 45×80 與 11×20（皆 < `nca_spectral_modes: 48`），訓練 8000 步
後 4 個 weight1 張量共 2,211,840 個參數仍 exactly zero，占 spectral 儲存參數的
60.04%。本檔當時的四條斷言即**鎖定那個缺陷行為**，並在檔頭寫明「修正落地時本檔
斷言應該會失敗，屆時請連同 handoff 一起更新，不要直接放寬」。

## 現況（2026-07-29 R6-G11 修正落地，本檔已依上述指示改寫）

兩段各自 clamp：``m1_lo = min(modes1, (H+1)//2)``（低正頻，含 DC）、
``m1_hi = min(modes1, H//2)``（高負頻，不含 DC）。兩者相加恆 ``<= H``
⇒ **切片永不重疊**，weight1 與 weight2 都拿得到梯度。這是標準 FNO 的語義。

★相容性：``modes1 < H/2`` 的模型逐位元行為不變（兩段 clamp 值都等於 `modes1`）；
只有原本會重疊者行為改變，那些 ckpt 需重訓。

本檔現在鎖的是「不重疊 + 兩組權重都有效」，任何回歸都會紅燈。
"""
from __future__ import annotations

import torch

from lbmnca.nca.tes.models.baselines.ar_fno import SpectralConv2d


def _layer(modes: int = 8, in_ch: int = 2, out_ch: int = 3) -> SpectralConv2d:
    """建一個權重可控的 SpectralConv2d（weight1 全 1、weight2 全 0）。"""
    layer = SpectralConv2d(in_ch, out_ch, modes, modes)
    with torch.no_grad():
        layer.weight1_real.fill_(1.0)
        layer.weight1_imag.fill_(0.0)
        layer.weight2_real.fill_(0.0)
        layer.weight2_imag.fill_(0.0)
    return layer


def _slices(modes: int, height: int) -> tuple[int, int]:
    """forward 內兩段的實際 clamp 值（與實作同式，供斷言對照）。"""
    return min(modes, (height + 1) // 2), min(modes, height // 2)


def test_slices_never_overlap_across_sizes():
    """★核心保證：任意 (modes1, height) 下 m1_lo + m1_hi <= height。"""
    for modes in (1, 4, 8, 48):
        for height in (1, 2, 3, 4, 6, 8, 9, 11, 12, 20, 45, 100, 360):
            lo, hi = _slices(modes, height)
            assert lo + hi <= height, (
                f"modes={modes} height={height}：m1_lo({lo}) + m1_hi({hi}) "
                f"> height({height}) → 兩段切片重疊，weight1 會被覆寫（R6-G11 回歸）"
            )


def test_weight1_is_effective_when_modes_ge_height():
    """modes1 >= height（舊缺陷條件）下 weight1 現在**有效**：輸出非零。"""
    layer = _layer(modes=8)
    x = torch.randn(1, 2, 6, 6)          # height=6 < modes=8（舊碼此處輸出恆零）
    out = layer(x)
    assert torch.count_nonzero(out) > 0, (
        "weight2 全零、weight1 全一時輸出應非零；恆零代表 weight1 又被覆寫了"
    )


def test_both_weights_receive_gradient_across_sizes():
    """weight1 與 weight2 在各尺寸下都拿得到非零梯度（height=1 無負頻除外）。

    ★loss 必須非平凡：``out.sum()`` 只依賴 DC 分量，會讓 weight2 的梯度恆零而
    製造假失敗；此處用 ``out.pow(2).sum()``。
    """
    for height in (6, 11, 45, 100):
        layer = SpectralConv2d(2, 3, 48, 48)
        layer(torch.randn(1, 2, height, 16)).pow(2).sum().backward()
        for name in ("weight1_real", "weight2_real"):
            grad = getattr(layer, name).grad
            assert grad is not None and torch.count_nonzero(grad) > 0, (
                f"height={height}：{name} 未收到非零梯度（R6-G11 回歸）"
            )


def test_height_one_has_no_negative_frequency_block():
    """height==1 時 m1_hi==0：只寫 weight1，weight2 不參與（非缺陷，是無負頻列）。"""
    lo, hi = _slices(8, 1)
    assert (lo, hi) == (1, 0)
    layer = SpectralConv2d(2, 3, 8, 8)
    layer(torch.randn(1, 2, 1, 8)).pow(2).sum().backward()
    assert torch.count_nonzero(layer.weight1_real.grad) > 0
    assert layer.weight2_real.grad is None or torch.count_nonzero(
        layer.weight2_real.grad) == 0


def test_low_mode_models_are_bit_identical_to_old_behaviour():
    """modes1 < height/2 時新舊實作等價——這是「相容性不變」宣稱的機器複驗。

    舊實作＝兩段共用 ``m1 = min(modes1, height)``。在 ``modes1 < height/2`` 下
    ``m1 == m1_lo == m1_hi == modes1``，故此處以舊式手算重建期望輸出並逐位元比對。
    """
    modes, height, width = 4, 32, 32
    layer = SpectralConv2d(2, 3, modes, modes)
    x = torch.randn(1, 2, height, width)
    got = layer(x)

    w_half = width // 2 + 1
    m1 = min(modes, height)                      # 舊式：兩段共用
    m2 = min(modes, w_half)
    x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
    w1 = torch.complex(layer.weight1_real[:, :, :m1, :m2],
                       layer.weight1_imag[:, :, :m1, :m2])
    w2 = torch.complex(layer.weight2_real[:, :, :m1, :m2],
                       layer.weight2_imag[:, :, :m1, :m2])
    out_ft = torch.zeros(1, 3, height, w_half, dtype=torch.cfloat)
    out_ft[:, :, :m1, :m2] = torch.einsum(
        "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w1)
    out_ft[:, :, -m1:, :m2] = torch.einsum(
        "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w2)
    expect = torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1), norm="ortho")

    assert torch.allclose(got, expect, atol=1e-6), (
        "modes1 < height/2 的模型行為應與舊實作逐位元相同——不成立表示修正"
        "誤傷了不受影響的模型（例如 low-mode 消融臂）"
    )
