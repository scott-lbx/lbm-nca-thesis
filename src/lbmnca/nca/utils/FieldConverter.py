"""
FieldConverter.py - v13.0
======================================================
GPU-tensor 物理場轉換器。全面整合 FieldConverter 的 getter 方法，
移除所有 hard-coded 通道索引，並透過 region 標記與 docstrings 提升可讀性。
現在，所有通道索引都由文件頂部的全局變量定義，進一步提升了可維護性。

主要職責:
  - 狀態張量切片: 從完整的 29-channel state 張量中安全地提取 9-channel
    物理矩 (Mphy) 或 3-channel 宏觀物理量 (MacroPhy)。
  - 物理量計算: 從 Mphy 計算衍伸的物理量，如速度 (velocity)、渦度 (vorticity)。
  - 平衡態計算: 封裝 LBM 的平衡態分布函數 (f_eq) 計算邏輯。
  - (反)正規化: 整合 FieldNormalizer，支援在物理量與 Z-score 空間之間轉換。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer

# ==========================================
# region: Global Channel Indices (Mphy)
# ==========================================
MPHY_RHO_IDX = 0
MPHY_E_IDX = 1
MPHY_EPS_IDX = 2
MPHY_JX_IDX = 3
MPHY_QX_IDX = 4
MPHY_JY_IDX = 5
MPHY_QY_IDX = 6
MPHY_PXX_IDX = 7
MPHY_PXY_IDX = 8

# ==========================================
# region: Global Channel Indices (MacroPhy - from Mphy_raw[[0,3,5]])
# ==========================================
MACRO_RHO_IDX = 0  # 對應 MPHY_RHO_IDX
MACRO_JX_IDX = 1  # 對應 MPHY_JX_IDX
MACRO_JY_IDX = 2  # 對應 MPHY_JY_IDX


class FieldConverter(nn.Module):
    """
    物理場轉換器，負責所有與物理通道相關的提取、計算與轉換。
    """

    # ==========================================
    # region: Constants & Initialization
    # ==========================================

    VEL_CLAMP: float = 1.0  # 速度截斷值 (晶格單位)

    def __init__(self, normalizer: FieldNormalizer) -> None:
        """
        初始化 FieldConverter。

        Args:
            normalizer (FieldNormalizer): 一個已初始化的 FieldNormalizer 實例。
        """
        super().__init__()
        self.normalizer = normalizer

        # Sobel kernels 用於計算渦度
        kx = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            / 8.0
        )
        ky = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            / 8.0
        )
        self.register_buffer("_sobel_kx", kx)
        self.register_buffer("_sobel_ky", ky)

        # LBM 平衡態計算所需參數
        self.register_buffer(
            "w_vec",
            torch.tensor(
                [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
            ).view(1, 9, 1, 1),
        )
        self.register_buffer(
            "cx_vec", torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1]).view(1, 9, 1, 1)
        )
        self.register_buffer(
            "cy_vec", torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1]).view(1, 9, 1, 1)
        )

    # ==========================================
    # region: Factory Method
    # ==========================================

    @classmethod
    def from_file(cls, stats_path: Union[str, Path]) -> "FieldConverter":
        """
        從 global_stats.json 檔案路徑建立 FieldConverter 實例。

        Args:
            stats_path (Union[str, Path]): global_stats.json 的路徑。

        Returns:
            FieldConverter: 一個新的 FieldConverter 實例。
        """
        return cls(FieldNormalizer.from_file(stats_path))

    # ==========================================
    # region: Main Conversion Methods
    # ==========================================

    def toMphy(
        self,
        x: torch.Tensor,
        from_state: bool = False,
        denorm: bool = False,
    ) -> torch.Tensor:
        """
        從輸入張量中提取 9-channel 的物理矩 (Mphy)。

        Args:
            x (torch.Tensor): 輸入張量，可以是完整的 state ([B,C>=13,H,W]) 或 Mphy ([B,9,H,W])。
            from_state (bool): 若為 True，則從輸入 state 張量中提取物理通道 (ch 4:13)。
            denorm (bool): 若為 True，對提取出的 Mphy 進行反正規化。

        Returns:
            torch.Tensor: 9-channel 的 Mphy 張量。
        """
        if from_state:
            mphy = x[:, 4:13] if x.ndim == 4 else x[4:13]
        else:
            mphy = x

        if denorm:
            mphy = self.normalizer.denorm_mphy(mphy)

        return mphy

    def toVphy(
        self,
        mphy: torch.Tensor,
        is_norm: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        從 Mphy 計算衍伸的物理量 (Vphy)，如速度、渦度等。

        Args:
            mphy (torch.Tensor): 9-channel Mphy 張量。
            is_norm (bool): 若為 True，則輸入的 mphy 會先被反正規化再進行計算。
                            **僅適用於 mean_field_norm**（反正規化使用 S_combined）。
                            若要從 u_prime_norm 計算衍伸量，需先手動呼叫
                            ``normalizer.denorm_uprime()`` 取得 fluctuation，
                            再加回 ``denorm_mphy(mean_norm)`` 得到 m_turb_raw，
                            最後以 ``is_norm=False`` 呼叫本方法。

        Returns:
            Dict[str, torch.Tensor]: 一個包含各種物理量的字典。
        """
        squeeze = mphy.ndim == 3
        if squeeze:
            mphy = mphy.unsqueeze(0)

        if is_norm:
            mphy = self.normalizer.denorm_mphy(mphy)

        with torch.amp.autocast("cuda", enabled=False):
            m = mphy.float()
            rho = self.get_mphy_rho(m).clamp(min=1e-3)
            jx = self.get_mphy_jx(m)
            jy = self.get_mphy_jy(m)

            vel_x = (jx / rho).clamp(-self.VEL_CLAMP, self.VEL_CLAMP)
            vel_y = (jy / rho).clamp(-self.VEL_CLAMP, self.VEL_CLAMP)
            vel_mag = (
                vel_x**2 + vel_y**2 + 1e-8
            ).sqrt()  # +eps: gradient-safe at zero velocity (0/0→NaN otherwise)
            vorticity = self._sobel_curl(vel_x, vel_y)
            pressure = rho / 3.0

        result = {
            "vel_x": vel_x,
            "vel_y": vel_y,
            "vel_mag": vel_mag,
            "vorticity": vorticity,
            "pressure": pressure,
        }

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result

    def toMacroPhy(self, mphy_raw: torch.Tensor) -> torch.Tensor:
        """
        從 9-channel 原始物理矩中提取 3-channel 宏觀物理量 (rho, jx, jy)。

        Args:
            mphy_raw (torch.Tensor): 9-channel 的 Mphy_raw 張量。

        Returns:
            torch.Tensor: 3-channel 的宏觀物理量張量。
        """
        if mphy_raw.shape[1] != 9:
            raise ValueError(
                f"toMacroPhy 需要 9 個物理矩通道，但輸入有 {mphy_raw.shape[1]} 個。"
            )
        return (
            mphy_raw[:, [MPHY_RHO_IDX, MPHY_JX_IDX, MPHY_JY_IDX]]
            if mphy_raw.ndim == 4
            else mphy_raw[[MPHY_RHO_IDX, MPHY_JX_IDX, MPHY_JY_IDX]]
        )

    # ==========================================
    # region: Masking Methods
    # ==========================================

    def get_mask(self, sdf_norm: torch.Tensor) -> torch.Tensor:
        """
        從正規化後的 SDF 張量計算流體遮罩 (1=Fluid, 0=Solid)。
        SDF 定義: 外部(Fluid) > 0, 內部(Solid) < 0。

        Args:
            sdf_norm (torch.Tensor): 已正規化的 SDF 通道。

        Returns:
            torch.Tensor: 1.0 代表流體(Fluid)，0.0 代表固體(Solid)。
        """
        # 1. 轉回原始物理數值的 SDF
        sdf_raw = self.normalizer.denorm_sdf(sdf_norm)

        # 2. 邏輯修正：SDF > 0 的地方是 Fluid，設為 1
        # 使用 > 0.0 或是加上一個極小的 epsilon 避免數值誤差 (例如 > 1e-5)
        return (sdf_raw > 0.0).float()

    # ==========================================
    # region: Equilibrium Calculation
    # ==========================================

    def get_f_eq(
        self, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 LBM 平衡態分布函數 f_eq。

        Args:
            rho (torch.Tensor): 密度張量。
            ux (torch.Tensor): x 方向速度張量。
            uy (torch.Tensor): y 方向速度張量。

        Returns:
            torch.Tensor: 9-channel 的平衡態分布函數 f_eq。
        """
        w_vec = self.w_vec.to(rho.device)
        cx_vec = self.cx_vec.to(rho.device)
        cy_vec = self.cy_vec.to(rho.device)

        cu = cx_vec * ux + cy_vec * uy
        u2 = ux * ux + uy * uy

        f_eq = w_vec * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
        return f_eq

    # ==========================================
    # region: Single-Channel Getters (Mphy)
    # ==========================================

    def _get_mphy_channel(self, mphy: torch.Tensor, index: int) -> torch.Tensor:
        """內部輔助方法：從 9-channel 物理矩中安全地獲取單一通道。"""
        if mphy.shape[1] != 9:
            raise ValueError(f"需要 9 個物理矩通道，但輸入有 {mphy.shape[1]} 個。")
        return mphy[:, index : index + 1] if mphy.ndim == 4 else mphy[index : index + 1]

    def get_mphy_rho(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_RHO_IDX)

    def get_mphy_e(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_E_IDX)

    def get_mphy_eps(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_EPS_IDX)

    def get_mphy_jx(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_JX_IDX)

    def get_mphy_qx(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_QX_IDX)

    def get_mphy_jy(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_JY_IDX)

    def get_mphy_qy(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_QY_IDX)

    def get_mphy_pxx(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_PXX_IDX)

    def get_mphy_pxy(self, mphy: torch.Tensor) -> torch.Tensor:
        return self._get_mphy_channel(mphy, MPHY_PXY_IDX)

    # ==========================================
    # region: Single-Channel Getters (MacroPhy)
    # ==========================================

    def _get_macro_raw_channel(
        self, macro_raw: torch.Tensor, index: int
    ) -> torch.Tensor:
        """內部輔助方法：從 3-channel 宏觀物理量中安全地獲取單一通道。"""
        if macro_raw.shape[1] != 3:
            raise ValueError(
                f"需要 3 個宏觀物理量通道 (rho, jx, jy)，但輸入有 {macro_raw.shape[1]} 個。"
            )
        return (
            macro_raw[:, index : index + 1]
            if macro_raw.ndim == 4
            else macro_raw[index : index + 1]
        )

    def get_macro_rho(self, macro_raw: torch.Tensor) -> torch.Tensor:
        return self._get_macro_raw_channel(macro_raw, MACRO_RHO_IDX)

    def get_macro_jx(self, macro_raw: torch.Tensor) -> torch.Tensor:
        return self._get_macro_raw_channel(macro_raw, MACRO_JX_IDX)

    def get_macro_jy(self, macro_raw: torch.Tensor) -> torch.Tensor:
        return self._get_macro_raw_channel(macro_raw, MACRO_JY_IDX)

    # ==========================================
    # region: Internal Helpers & Forward
    # ==========================================

    def _sobel_curl(self, vel_x: torch.Tensor, vel_y: torch.Tensor) -> torch.Tensor:
        """內部輔助方法：使用 Sobel kernel 計算渦度。"""
        kx = self._sobel_kx.to(vel_x.device, dtype=vel_x.dtype)
        ky = self._sobel_ky.to(vel_x.device, dtype=vel_x.dtype)
        dvy_dx = F.conv2d(F.pad(vel_y, (1, 1, 1, 1), mode="replicate"), kx)
        dvx_dy = F.conv2d(F.pad(vel_x, (1, 1, 1, 1), mode="replicate"), ky)
        return dvy_dx - dvx_dy

    def forward(
        self, mphy: torch.Tensor, is_norm: bool = False
    ) -> Dict[str, torch.Tensor]:
        """預設 forward 等同 toVphy。"""
        return self.toVphy(mphy, is_norm=is_norm)
