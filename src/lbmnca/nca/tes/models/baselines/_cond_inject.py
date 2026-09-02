"""Baseline 條件注入：把 cond 向量 + per-step Δ CTU broadcast 成空間 channel。

全 baseline（FNO/UNet/DRN/steady）共用，使條件資訊對齊 T-NCA（協定 cond19）：
取 Re_norm、L_char_norm（已 z-score）+ Δ CTU（raw→log1p normalize）。
G_vec 幾何資訊由 static SDF channel 提供，不另 broadcast（gvec_mode=none）。

cond 向量 layout（cond17 / cond19 皆 g_vec 在前 14；見 data/normalizer.py）：

    [0:14]  G_vec_norm
    14      Re_norm
    15      U_inlet_norm
    16      L_char_norm
    (17     delta_ctu_norm)   # cond19 才有；baseline 改用 per-step delta_ctu_seq
    (18     cumul_ctu_norm)
"""
from __future__ import annotations

import math

import torch

# 與 TESCondNormalizer.normalize_delta_ctu 一致：log1p(delta)/log1p(5)
_DELTA_CTU_LOG_MAX: float = math.log1p(5.0)
_RE_IDX: int = 14
_LCHAR_IDX: int = 16

_COND_CH: dict[str, int] = {"re_lchar_delta": 3, "re_lchar": 2, "none": 0}


def cond_channel_count(mode: str) -> int:
    """回傳指定模式下的條件 broadcast channel 數。

    Args:
        mode: 條件注入模式，``re_lchar_delta`` | ``re_lchar`` | ``none``。

    Returns:
        對應的 channel 數量（0、2 或 3）。

    Raises:
        KeyError: 若 mode 不在支援清單中。
    """
    return _COND_CH[mode]


def _norm_delta(delta_raw: torch.Tensor) -> torch.Tensor:
    """將 raw Δ CTU 正規化，對齊 TESCondNormalizer。

    Args:
        delta_raw: shape [B]，原始 Δ CTU 值（非負）。

    Returns:
        shape [B]，log1p 正規化後的值，範圍約 [0, 1]。
    """
    return torch.log1p(delta_raw.float().clamp_min(0.0)) / _DELTA_CTU_LOG_MAX


def build_cond_channels(
    cond_norm: torch.Tensor,
    delta_raw: torch.Tensor | None,
    h: int,
    w: int,
    mode: str,
) -> torch.Tensor:
    """組空間條件 channel tensor，shape 為 [B, c, H, W]（c = cond_channel_count(mode)）。

    Args:
        cond_norm: shape [B, cond_dim]，z-score 條件向量（Re@14, L_char@16）。
        delta_raw: shape [B]，本步 raw Δ CTU；當 mode 含 delta 但此參數為 None
            時以全零填補。
        h: 空間高度。
        w: 空間寬度。
        mode: 條件注入模式，``re_lchar_delta`` | ``re_lchar`` | ``none``。

    Returns:
        shape [B, c, H, W] 的空間條件 channel tensor；
        mode 為 ``none`` 時 c=0，回傳零長度 channel tensor。
    """
    b = cond_norm.shape[0]
    if mode == "none":
        return cond_norm.new_zeros(b, 0, h, w)
    feats = [cond_norm[:, _RE_IDX:_RE_IDX + 1], cond_norm[:, _LCHAR_IDX:_LCHAR_IDX + 1]]
    if mode == "re_lchar_delta":
        dt = _norm_delta(delta_raw) if delta_raw is not None else cond_norm.new_zeros(b)
        feats.append(dt.view(b, 1).to(cond_norm.dtype))
    vec = torch.cat(feats, dim=1)                       # [B, c]
    return vec[:, :, None, None].expand(b, vec.shape[1], h, w)
