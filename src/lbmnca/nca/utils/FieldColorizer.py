"""
FieldColorizer — GPU-tensor 視覺化上色器
=========================================
將 Vphy_lattice_raw 場量映射為 RGB tensor，全程在 GPU 上運算。

視覺化固定範圍（vel_max / vor_range / p_center / p_range，lattice 單位）為
**colormap SSOT**，來源 = stats.json `vis_range`（與 tanh norm 正交）。建構時
注入（``from_stats`` / ``from_normalizer``），確保 sim-viz / 訓練 SSIM metric /
03 部署三階段同一固定尺度 → 不閃、跨 run/ckpt 可比、部署一致。

- ``use_fixed_range=True``：用注入的 vis_range（未注入則報錯，禁 fallback）。
- ``use_fixed_range=False``：自適應當前場量 min/max（僅 debug，會閃爍）。
- ``_colorize(field, vmin, vmax, ...)`` 直接帶顯式範圍：不需注入（per-sample 診斷圖用）。

Input : [B, 1, H, W] or [1, H, W]  — Vphy_lattice_raw 單通道
Output: [B, 3, H, W] or [3, H, W]  — RGB float32, [0, 1]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


# ── LUT 建構工具 ─────────────────────────────────────────────────────────────


def _build_lut(cmap, n: int = 256) -> torch.Tensor:
    """從 matplotlib colormap 建立 [n, 3] float32 LUT tensor。"""
    t = np.linspace(0, 1, n)
    rgba = cmap(t)[:, :3]
    return torch.from_numpy(rgba.astype(np.float32))  # [n, 3]


def _apply_lut(
    data_01: torch.Tensor,  # [B, 1, H, W], values in [0, 1]
    lut: torch.Tensor,  # [n, 3], on same device
) -> torch.Tensor:  # [B, 3, H, W]
    """GPU 線性插值 LUT 映射。"""
    B, _, H, W = data_01.shape
    n = lut.shape[0] - 1  # max valid index

    d = data_01[:, 0].clamp(0.0, 1.0)  # [B, H, W]
    idx_f = d * n  # [B, H, W] float
    idx_lo = idx_f.long().clamp(0, n - 1)  # [B, H, W]
    idx_hi = (idx_lo + 1).clamp(0, n)  # [B, H, W]
    frac = idx_f - idx_lo.float()  # [B, H, W]

    flat_lo = idx_lo.reshape(B, -1)  # [B, H*W]
    flat_hi = idx_hi.reshape(B, -1)

    c_lo = lut[flat_lo]  # [B, H*W, 3]
    c_hi = lut[flat_hi]
    frac_3 = frac.reshape(B, H * W, 1)  # [B, H*W, 1]

    color = c_lo + frac_3 * (c_hi - c_lo)  # [B, H*W, 3]
    return color.permute(0, 2, 1).reshape(B, 3, H, W)  # [B, 3, H, W]


# ── FieldColorizer ────────────────────────────────────────────────────────────


class FieldColorizer(nn.Module):
    """
    GPU-tensor 場量 → RGB 上色器。

    視覺範圍 (lattice units, D2Q9 MRT) 由 stats.json `vis_range` 注入（SSOT）：
        vel_mag   : [0, vel_max]
        vorticity : [-vor_range, +vor_range]
        pressure  : [p_center - p_range, p_center + p_range]
        obstacle  : 0.5 灰色
    """

    OBSTACLE_GRAY: float = 0.5
    LUT_SIZE: int = 256

    def __init__(
        self,
        vel_max: Optional[float] = None,
        vor_range: Optional[float] = None,
        p_center: Optional[float] = None,
        p_range: Optional[float] = None,
    ) -> None:
        """建構上色器。

        Args:
            vel_max / vor_range / p_center / p_range: vis_range 固定範圍（lattice
                units）。省略 → ``use_fixed_range=True`` 會報錯；``_colorize`` 帶顯式
                範圍仍可用（per-sample 診斷圖）。一般用 ``from_stats`` /
                ``from_normalizer`` 由 stats.json 注入。
        """
        super().__init__()

        self._vel_max = vel_max
        self._vor_range = vor_range
        self._p_center = p_center
        self._p_range = p_range

        # Velocity: plasma
        vel_lut = _build_lut(cm.get_cmap("plasma"), self.LUT_SIZE)
        self.register_buffer("_vel_lut", vel_lut, persistent=False)

        # Vorticity: 黃-橘-黑-綠-青 (identical to reference)
        vor_cmap = LinearSegmentedColormap.from_list(
            "vorticity_cmap",
            [
                (1, 1, 0),
                (0.953, 0.490, 0.016),
                (0, 0, 0),
                (0.176, 0.976, 0.529),
                (0, 1, 1),
            ],
        )
        vor_lut = _build_lut(vor_cmap, self.LUT_SIZE)
        self.register_buffer("_vor_lut", vor_lut, persistent=False)

        # Pressure: RdBu_r
        prs_lut = _build_lut(cm.get_cmap("RdBu_r"), self.LUT_SIZE)
        self.register_buffer("_prs_lut", prs_lut, persistent=False)

    # ── 工廠（由 stats.json vis_range 注入 SSOT 範圍）─────────────────────────

    @classmethod
    def from_normalizer(cls, normalizer: nn.Module) -> "FieldColorizer":
        """從 FieldNormalizer 取 vis_range 建構（共用同一 stats.json SSOT）。"""
        return cls(
            vel_max=normalizer.vel_vis_max,
            vor_range=normalizer.vor_vis_range,
            p_center=normalizer.p_vis_center,
            p_range=normalizer.p_vis_range,
        )

    @classmethod
    def from_stats(cls, stats: Union[str, Path, dict]) -> "FieldColorizer":
        """從 stats.json 路徑或 dict 的 `vis_range` 建構（strict，缺鍵報錯）。"""
        if isinstance(stats, (str, Path)):
            with open(str(stats), "r", encoding="utf-8") as f:
                stats = json.load(f)
        vr = stats["vis_range"]
        return cls(
            vel_max=float(vr["vel_max"]),
            vor_range=float(vr["vor_range"]),
            p_center=float(vr["p_center"]),
            p_range=float(vr["p_range"]),
        )

    # ── 內部輔助 ──────────────────────────────────────────────────────────

    def _colorize(
        self,
        field: torch.Tensor,
        vmin: float,
        vmax: float,
        lut: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. 統一將輸入轉為 4D: [B, 1, H, W]
        is_3d = field.ndim == 3
        if is_3d:
            field = field.unsqueeze(0)

        # 確保 mask 也是 4D (獨立檢查，避免 field 是 4D 但 mask 是 3D 的情況)
        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(0)

        # 2. 準備 LUT 與正規化
        lut = lut.to(field.device)
        # 使用 clamp 確保數值在 0~1 之間，避免 LUT 索引越界
        d01 = ((field - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
        rgb = _apply_lut(d01, lut)  # 預期輸出 [B, 3, H, W]

        # 3. 處理遮罩
        if mask is not None:
            # mask 為 1 是流體，0 是固體
            # 這裡利用 torch.where：如果 mask > 0.5 則取 rgb，否則取灰色
            # 灰色需要擴展維度以匹配 [B, 3, H, W]
            rgb = torch.where(mask > 0.5, rgb, torch.full_like(rgb, self.OBSTACLE_GRAY))

        # 4. 還原維度並回傳
        if is_3d:
            rgb = rgb.squeeze(0)

        return rgb.clamp(0.0, 1.0)

    # ── 公開 API ──────────────────────────────────────────────────────────

    def getVelColor(
        self,
        vel_mag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_fixed_range: bool = False,
    ) -> torch.Tensor:
        """速度量值 → RGB。"""
        if use_fixed_range:
            if self._vel_max is None:
                raise ValueError(
                    "FieldColorizer 未注入 vis_range，無法 use_fixed_range；"
                    "請用 FieldColorizer.from_stats(...) 或 from_normalizer(...) 建構。"
                )
            vmin, vmax = 0.0, self._vel_max
        else:
            vmin, vmax = vel_mag.min().item(), vel_mag.max().item()

        return self._colorize(vel_mag, vmin, vmax, self._vel_lut, mask)

    def getVorColor(
        self,
        vorticity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_fixed_range: bool = False,
    ) -> torch.Tensor:
        """渦度場 → RGB。"""
        if use_fixed_range:
            if self._vor_range is None:
                raise ValueError(
                    "FieldColorizer 未注入 vis_range，無法 use_fixed_range；"
                    "請用 FieldColorizer.from_stats(...) 或 from_normalizer(...) 建構。"
                )
            vmin, vmax = -self._vor_range, self._vor_range
        else:
            vmin, vmax = vorticity.min().item(), vorticity.max().item()

        return self._colorize(vorticity, vmin, vmax, self._vor_lut, mask)

    def getPressureColor(
        self,
        pressure: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_fixed_range: bool = False,
    ) -> torch.Tensor:
        """壓力場 → RGB。"""
        if use_fixed_range:
            if self._p_center is None or self._p_range is None:
                raise ValueError(
                    "FieldColorizer 未注入 vis_range，無法 use_fixed_range；"
                    "請用 FieldColorizer.from_stats(...) 或 from_normalizer(...) 建構。"
                )
            vmin, vmax = self._p_center - self._p_range, self._p_center + self._p_range
        else:
            vmin, vmax = pressure.min().item(), pressure.max().item()

        return self._colorize(pressure, vmin, vmax, self._prs_lut, mask)

    def forward(
        self,
        vel_mag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_fixed_range: bool = False,
    ) -> torch.Tensor:
        """預設 forward 等同 getVelColor。"""
        return self.getVelColor(vel_mag, mask, use_fixed_range)
