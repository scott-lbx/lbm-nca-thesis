"""狀態通道弱約束損失（norm space，M1A 延伸）。

在 normalized Mphy space 直接計算 MAE，無需 denorm。
用於防止 Vphy loss 主導下各通道發生漂移。

函式：
  obs_state_rho_mae      — density ch0 MAE（防 ch0 drift）
  obs_state_stress_mae   — ch5-ch8 MAE（legacy 弱約束，含 jy/qy）
  obs_state_stress_anchor — ch7-ch8 純 stress MAE（W0-1 bug-C 修正版）

MRT channel indices（9ch physics tensor）：
  RHO_CH = 0  (m0: 密度)
  ch5    = 5  (jy: y 動量)
  ch6    = 6  (qy: y 熱通量)
  ch7    = 7  (pxx: 偏差應力 xx)
  ch8    = 8  (pxy: 偏差應力 xy)
"""
from __future__ import annotations

from torch import Tensor

RHO_CH: int = 0


def obs_state_rho_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
) -> Tensor:
    """ObsState_Rho_Mae：density channel (ch0) MAE in norm space。

    弱約束防止 Vphy loss 主導下密度通道漂移。M1A_0_base 觀察到 ch0_mae
    從 1.3 漸增至 1.8（warm/cold 均如此），Vphy 不約束 ch0 magnitude。

    Args:
        pred_norm: [B, K, 9, H, W] normalized Mphy prediction。
        target_norm: [B, K, 9, H, W] normalized Mphy target。
        fluid_mask: [B, 1, H, W] float，流體=1 固體=0。

    Returns:
        scalar: ch0 MAE in norm space，fluid only。
    """
    B, K, _, H, W = pred_norm.shape
    pred_ch0 = pred_norm[:, :, RHO_CH]   # [B, K, H, W]
    tgt_ch0 = target_norm[:, :, RHO_CH]
    diff = (pred_ch0 - tgt_ch0).abs()    # [B, K, H, W]
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W)
    n = mask_bk.sum().clamp_min(1.0)
    return (diff * mask_bk).sum() / n


def obs_state_stress_mae(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
) -> Tensor:
    """ObsState_Stress_Mae：ch5..ch8 MAE in norm space（legacy 弱約束）。

    Vphy loss 只看 vel（ch3=jx, ch5=jy via rho）。本項約束 ch5..8 = jy/qy/pxx/pxy
    （注意：含動量 jy + 熱通量 qy，**非純 stress**）。純 stress-moment anchor（pxx,pxy）
    見 ``obs_state_stress_anchor``（W0-1 bug-C 新增）。保留本項供既有 config（la4）相容。

    Args:
        pred_norm: [B, K, 9, H, W] normalized Mphy prediction。
        target_norm: [B, K, 9, H, W] normalized Mphy target。
        fluid_mask: [B, 1, H, W] float，流體=1 固體=0。

    Returns:
        scalar: ch5-ch8 平均 MAE in norm space，fluid only。
    """
    B, K, _, H, W = pred_norm.shape
    pred_st = pred_norm[:, :, 5:9]   # [B, K, 4, H, W]（jy,qy,pxx,pxy）
    tgt_st = target_norm[:, :, 5:9]
    diff = (pred_st - tgt_st).abs().mean(dim=2)   # avg over 4 channels → [B, K, H, W]
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W)
    n = mask_bk.sum().clamp_min(1.0)
    return (diff * mask_bk).sum() / n


def obs_state_stress_anchor(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
) -> Tensor:
    """ObsState_Stress_Anchor：純 stress moment (ch7=pxx, ch8=pxy) MAE in norm space。

    W0-1 bug-C：原 ``obs_state_stress_mae`` 切 ``[5:9]`` 誤含 jy(ch5)/qy(ch6)；本 anchor
    切 ``[7:9]`` = 純 stress moment（pxx,pxy）。LBM-native 偏差應力，**不經 curl → 免
    checkerboard 放大**（L-Cal §4.3：enstrophy 替換候選的前置）。弱約束維持 stress 一致性。

    Args:
        pred_norm: [B, K, 9, H, W] normalized Mphy prediction。
        target_norm: [B, K, 9, H, W] normalized Mphy target。
        fluid_mask: [B, 1, H, W] float，流體=1 固體=0。

    Returns:
        scalar: ch7-ch8 平均 MAE in norm space，fluid only。
    """
    B, K, _, H, W = pred_norm.shape
    pred_st = pred_norm[:, :, 7:9]   # [B, K, 2, H, W]（pxx,pxy）
    tgt_st = target_norm[:, :, 7:9]
    diff = (pred_st - tgt_st).abs().mean(dim=2)   # avg over 2 stress channels → [B, K, H, W]
    mask_bk = fluid_mask.squeeze(1)[:, None, :, :].expand(B, K, H, W)
    n = mask_bk.sum().clamp_min(1.0)
    return (diff * mask_bk).sum() / n
