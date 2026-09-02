"""TES_MRP — TES-NCA 的多受體感知模組（Multi-Receptor Perception）。

5 種受體並聯感知，嵌入 LBM 物理先驗作為固定 kernel 歸納偏置：
  TopoR  : Fixed Sobel_x + Sobel_y + Laplacian（空間梯度，作用於 physics channels）
  StreamK: Fixed D2Q9 streaming kernels（LBM 平流方向，作用於 physics channels）
  MTransK: Fixed MRT M-matrix（矩空間投影，作用於 physics channels）
  AdapR  : Learnable 3×3 conv（資料驅動，作用於完整 state）
  HormB  : Global avg pool + expand（全域 context，作用於完整 state）

所有固定 kernel 以 register_buffer 儲存（隨模型 save/load，不參與 backward）。
enabled 受體的輸出 concat 後回傳，供上層 fusion 1×1 conv 降至 nca_channels。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import MRPConfig


class TES_MRP(nn.Module):
    """TES-NCA 的多受體感知模組（Multi-Receptor Perception）。

    並聯五種感知受體，將固定 LBM 物理先驗與可學習感知融合為統一表示：

    - **TopoR**：固定 Sobel_x、Sobel_y、Laplacian kernel（depthwise），對 physics channels
      執行空間梯度計算，捕捉局部拓撲與曲率資訊。
    - **StreamK**：固定 D2Q9 方向平移 kernel（depthwise），對 physics channels
      執行 LBM 平流方向投影，嵌入格子 Boltzmann 流動先驗。
    - **MTransK**：固定 MRT M-matrix 1×1 conv，對 physics channels 執行矩空間投影，
      將分布函數轉換至 [ρ, e, ε, jx, qx, jy, qy, pxx, pxy] 矩空間。
    - **AdapR**：可學習 3×3 conv，對完整 state（全通道）進行資料驅動局部感知。
    - **HormB**：Global avg pool → 1×1 conv → 空間擴展，提供全域 context（hormone signal）。
      ``hormone_dim=0`` 時完全停用（對應 M1_mrp_nohorm 消融實驗）。

    所有固定 kernel 使用 ``register_buffer`` 儲存，隨模型 checkpoint 儲存與載入，
    不參與梯度計算。

    Args:
        state_dim: 完整 state 通道數（static + physics + hidden），預設 29。
        physics_dim: 物理通道數（state 中 C4–C12），預設 9。
        cfg: MRPConfig，控制各受體啟用與超參數；傳入 ``None`` 時使用預設值。

    Attributes:
        combined_ch: 所有啟用受體輸出拼接後的通道總數，供外部 fusion conv 參考。
            預設全開時：27（TopoR）+ 9（StreamK）+ 9（MTransK）+ 32（AdapR）+ 8（HormB）= 85。
    """

    # D2Q9 streaming 方向 (row, col) 索引，對應 MRT moment 通道順序
    # [ρ, e, ε, jx, qx, jy, qy, pxx, pxy] → 對應 rest, E, N, W, S, NE, NW, SW, SE
    _D2Q9_DIRECTIONS: list[tuple[int, int]] = [
        (1, 1),  # ch0: rest / center
        (1, 0),  # ch1: East  (jx 方向)
        (2, 1),  # ch2: North (jy 方向)
        (1, 2),  # ch3: West
        (0, 1),  # ch4: South
        (2, 0),  # ch5: NE
        (2, 2),  # ch6: NW
        (0, 2),  # ch7: SW
        (0, 0),  # ch8: SE
    ]

    def __init__(
        self,
        state_dim: int = 29,
        physics_dim: int = 9,
        cfg: MRPConfig | None = None,
    ) -> None:
        """初始化多受體感知模組，依 cfg 建立固定 buffer 與可學習子模組。

        Args:
            state_dim: 完整 state 通道數（static + physics + hidden），預設 29。
            physics_dim: 物理通道數（state 中 C4–C12），預設 9。
            cfg: MRPConfig，控制各受體啟用與超參數；傳入 ``None`` 時使用預設值。
        """
        super().__init__()
        if cfg is None:
            cfg = MRPConfig()
        self._cfg = cfg
        self._state_dim = state_dim
        self._physics_dim = physics_dim

        combined_ch = 0

        # ── TopoR：Sobel_x + Sobel_y + Laplacian（depthwise，作用於 physics channels）
        if cfg.use_topo:
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            )
            sobel_y = sobel_x.T.contiguous()
            laplacian = torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]
            )
            base = torch.stack([sobel_x, sobel_y, laplacian]).unsqueeze(1)  # [3, 1, 3, 3]
            topo_filters = base.repeat(physics_dim, 1, 1, 1)                # [physics_dim*3, 1, 3, 3]
            self.register_buffer("topo_filters", topo_filters)
            self._topo_ch = physics_dim * 3  # 27
            if cfg.use_groupnorm:
                self.norm_topo = nn.GroupNorm(1, self._topo_ch, eps=1e-3)
            combined_ch += self._topo_ch

        # ── StreamK：D2Q9 方向平移（depthwise，作用於 physics channels）
        if cfg.use_stream:
            sk = torch.zeros(physics_dim, 1, 3, 3, dtype=torch.float32)
            for i, (r, c) in enumerate(self._D2Q9_DIRECTIONS[:physics_dim]):
                sk[i, 0, r, c] = 1.0
            self.register_buffer("streaming_kernels", sk)  # [9, 1, 3, 3]
            self._stream_ch = physics_dim  # 9
            if cfg.use_groupnorm:
                self.norm_stream = nn.GroupNorm(1, self._stream_ch, eps=1e-3)
            combined_ch += self._stream_ch

        # ── MTransK：MRT M-matrix（1×1 conv，作用於 physics channels）
        if cfg.use_trans:
            M = torch.tensor([
                [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
                [-4, -1, -1, -1, -1,  2,  2,  2,  2],
                [ 4, -2, -2, -2, -2,  1,  1,  1,  1],
                [ 0,  1,  0, -1,  0,  1, -1, -1,  1],
                [ 0, -2,  0,  2,  0,  1, -1, -1,  1],
                [ 0,  0,  1,  0, -1,  1,  1, -1, -1],
                [ 0,  0, -2,  0,  2,  1,  1, -1, -1],
                [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  1, -1,  1, -1],
            ], dtype=torch.float32)
            self.register_buffer("lbm_m_matrix", M.view(9, 9, 1, 1))  # [9, 9, 1, 1]
            self._trans_ch = physics_dim  # 9（M 是方陣）
            if cfg.use_groupnorm:
                self.norm_trans = nn.GroupNorm(1, self._trans_ch, eps=1e-3)
            combined_ch += self._trans_ch

        # ── AdapR：可學習 3×3 conv（作用於完整 state）
        self.adapt_conv = nn.Conv2d(
            state_dim, cfg.num_learnable, kernel_size=3, padding=1, bias=True
        )
        self._adapt_ch = cfg.num_learnable  # 32
        combined_ch += self._adapt_ch

        # ── HormB：Global avg pool → 1×1 conv → expand（作用於完整 state）
        # hormone_dim=0 → 完全停用（M1_mrp_nohorm ablation 用）
        self._horm_ch = cfg.hormone_dim
        if cfg.hormone_dim > 0:
            self.hormone_proj = nn.Sequential(
                nn.Conv2d(state_dim, cfg.hormone_dim, kernel_size=1, bias=True),
                nn.Tanh(),
            )
            combined_ch += self._horm_ch
        else:
            self.hormone_proj = None

        self.combined_ch = combined_ch  # 預設全開：27+9+9+32+8 = 85

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """感知前向傳播：並聯各受體後 concat 輸出。

        依序執行各啟用受體的感知計算，並將所有輸出沿通道維拼接後回傳。
        各受體說明：

        - TopoR：對 physics channels 施加 Sobel/Laplacian，padding=replicate 保邊界。
        - StreamK：對 physics channels 施加 D2Q9 方向 kernel，padding=replicate。
        - MTransK：對 physics channels 施加 MRT M-matrix 1×1 conv（無需 padding）。
        - AdapR：對完整 state 施加可學習 3×3 conv。
        - HormB：對完整 state 做 global avg pool → 1×1 proj → expand 至原始空間尺寸。

        Args:
            state: 完整 NCA 狀態張量，形狀 ``[B, state_dim, H, W]``。

        Returns:
            perception: 各受體輸出拼接結果，形狀 ``[B, combined_ch, H, W]``，
                通道順序依受體啟用順序：TopoR → StreamK → MTransK → AdapR → HormB。
        """
        _, _, H, W = state.shape
        # physics channels: static_dim=4，固定從 C4 開始
        physics = state[:, 4: 4 + self._physics_dim]
        parts: list[torch.Tensor] = []

        # TopoR
        if self._cfg.use_topo:
            pad = F.pad(physics, (1, 1, 1, 1), mode="replicate")
            topo = F.conv2d(pad, self.topo_filters, groups=self._physics_dim)
            if self._cfg.use_groupnorm:
                topo = self.norm_topo(topo)
            parts.append(topo)

        # StreamK
        if self._cfg.use_stream:
            pad = F.pad(physics, (1, 1, 1, 1), mode="replicate")
            stream = F.conv2d(pad, self.streaming_kernels, groups=self._physics_dim)
            if self._cfg.use_groupnorm:
                stream = self.norm_stream(stream)
            parts.append(stream)

        # MTransK（1×1 conv，無需 padding）
        if self._cfg.use_trans:
            trans = F.conv2d(physics, self.lbm_m_matrix)
            if self._cfg.use_groupnorm:
                trans = self.norm_trans(trans)
            parts.append(trans)

        # AdapR
        parts.append(self.adapt_conv(state))

        # HormB（hormone_dim=0 時跳過）
        if self.hormone_proj is not None:
            hormone = self.hormone_proj(
                F.adaptive_avg_pool2d(state, (1, 1))
            ).expand(-1, -1, H, W)
            parts.append(hormone)

        return torch.cat(parts, dim=1)  # [B, combined_ch, H, W]


__all__ = ["TES_MRP"]
