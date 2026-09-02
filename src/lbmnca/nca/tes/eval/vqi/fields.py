"""刻度所需的參照場。**這些場由該案的真值當場造出，不取自任何母體。**

VQI 的兩個端點與否決門檻皆為逐案量，其定義只用到該案的真值與流體遮罩

| 場 | 用途 | 構造 |
|---|---|---|
| 真值本身 | 上端點 $+1$ | 距離為零 |
| `zero_field` | 下端點 $-1$ | 動量兩通道歸零 |
| `veto_reference_fields` | 否決門檻 | 四類偽影的輕級 |

## 為什麼擾動只加在動量通道

第一版加在全部九個通道（含密度 `m0`），反正規化後 ``vel = j / rho`` 在密度
過零處爆掉，實測條紋場的 ``vel_mag`` 標準差達 541 而真值是 0.0235，差兩萬倍。
那已經不是條紋而是一個壞掉的場，用它訂門檻等於拿壞尺量東西。

## 零場即正規化空間的全零

``zero_field`` 把九個通道全部設為零。正規化採 tanh 模式，反正規化為
``atanh(x)·s + m``，故正規化空間的零反正規化後即為逐通道的均值
``m = [1, -2, 1, 0, …]``，其物理意義為**密度均勻為 1、速度逐點為零的靜止平衡場**。
這正是求解器冷啟動的初始條件，論文全書以「零場」指稱它。

**刻度的下端點與冷啟動的起點是同一個場。** 分數 $-1$ 因而可讀作
「與冷啟動開始之前的狀態一樣，完全不含任何流場資訊」。

★ 與「均勻場」不同。後者逐通道取流體區空間平均後廣播回全場，
空間變化盡失而**動量守住**，仍是一股均勻流。
"""
from __future__ import annotations

import torch

from .. import visual_metrics as VM

__all__ = ["MOMENTUM_CHANNELS", "zero_field", "veto_reference_fields",
           "perturb", "VETO_REF_AMPLITUDE"]

#: 動量通道。MRT 矩空間的 ``m3`` 與 ``m5``。與 `terms` 同源，不另行宣告常數。
MOMENTUM_CHANNELS: tuple[int, int] = (VM._JX_CH, VM._JY_CH)

#: 否決參照的擾動振幅，以真值動量標準差為單位。
#: 取輕級而非重級，語義為「這一類偽影裡最輕、但仍必須被擋下的那一級」。
VETO_REF_AMPLITUDE: float = 0.10


def zero_field(target: torch.Tensor) -> torch.Tensor:
    """零場，即正規化空間的全零。刻度的下端點。

    反正規化後為密度均勻為 1、速度逐點為零的靜止平衡場，
    亦即求解器冷啟動的初始條件。

    ★本場**不依賴 `target` 的內容**，只借它的形狀與裝置。刻度的下端點因而是
    全案共通的同一個物理狀態，而各案的刻度單位仍逐案不同——單位由該案真值與
    本場之間的距離決定。

    Args:
        target: ``[N,9,H,W]`` 正規化真值場。只用其形狀、dtype 與 device。

    Returns:
        與 `target` 同形狀的全零張量。
    """
    return torch.zeros_like(target)


def perturb(target: torch.Tensor, fluid_mask: torch.Tensor,
            delta: torch.Tensor) -> torch.Tensor:
    """在動量通道上加擾動，並乘流體遮罩。

    ★**必須乘流體遮罩。** 固體內反正規化後的密度趨近零，往那裡加動量會讓
    ``vel = j / rho`` 爆掉，實測 ``vel_mag`` 標準差自真值的 0.024 衝到 649。

    Args:
        target: ``[N,9,H,W]`` 正規化真值場。
        fluid_mask: ``[N,1,H,W]``，流體為 1。
        delta: 要加在動量通道上的擾動，形狀須可廣播至 ``[N,2,H,W]``。

    Returns:
        擾動後的場。
    """
    y = target.clone()
    y[:, list(MOMENTUM_CHANNELS)] = (
        y[:, list(MOMENTUM_CHANNELS)] + delta * fluid_mask)
    return y


def veto_reference_fields(target: torch.Tensor, fluid_mask: torch.Tensor,
                          ) -> dict[str, torch.Tensor]:
    """四類偽影的輕級，供逐案導出否決門檻。

    四類分別為棋盤格（網格尺度振盪，流體力學裡的 odd-even decoupling）、
    白雜訊（顆粒）、水平條紋（``kx=0`` 偽影）與斜向條紋。
    四類全部列入是因為精簡組只有單一頻譜項負責它們，門檻取四者中**最高**的
    子分數，因而落在最容易漏掉的那一類上。

    Args:
        target: ``[N,9,H,W]`` 正規化真值場。
        fluid_mask: ``[N,1,H,W]``，流體為 1。

    Returns:
        類別名 → 場。
    """
    dev, dtype = target.device, target.dtype
    h, w = target.shape[-2], target.shape[-1]
    ys = torch.arange(h, device=dev, dtype=dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=dev, dtype=dtype).view(1, 1, 1, w)
    mom = target[:, list(MOMENTUM_CHANNELS)]
    amp = VETO_REF_AMPLITUDE * float(mom.std().item())
    # ★雜訊的種子固定。門檻若隨呼叫次數改變，同一份讀數重跑會得到不同否決率。
    gen = torch.Generator(device=dev).manual_seed(0)
    return {
        "棋盤格": perturb(target, fluid_mask, amp * (-1.0) ** (ys + xs)),
        "白雜訊": perturb(target, fluid_mask,
                        amp * torch.randn(mom.shape, generator=gen,
                                          device=dev, dtype=dtype)),
        "水平條紋": perturb(target, fluid_mask, amp * torch.sin(ys * 0.9)),
        "斜向條紋": perturb(target, fluid_mask, amp * torch.sin((ys + xs) * 0.9)),
    }
