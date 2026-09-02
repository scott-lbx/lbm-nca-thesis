"""context.py — _LossContext 與 _LossTerm dataclass 定義。

本模組從 semantic_loss.py 拆出，集中放置 loss 執行時共用的資料容器。
``_LossContext`` 以 lazy @property 封裝常用的中間張量（SDF、wall_band、
inflow_weight_map、POD 基底），避免不使用的 term 觸發不必要的計算。
``_LossTerm`` 為 registry 的單一條目，包含語義名稱、呼叫函式及正規化旗標。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .loss_terms_v2 import compute_wall_band
from .inflow_prior import inflow_weight_map_seq


# ---------------------------------------------------------------------------
# _LossContext
# ---------------------------------------------------------------------------


@dataclass
class _LossContext:
    """執行 loss term 函式所需的張量集合（含 lazy 計算欄位）。

    Attributes:
        pred: ``[B, k, 9, H, W]`` 正規化預測軌跡。
        target: ``[B, k, 9, H, W]`` 正規化目標軌跡。
        ic_physics: ``[B, 9, H, W]`` 正規化物理初值（t=0 Mphy_norm）。
        static: ``[B, 4, H, W]`` 正規化靜態通道，ch0 = SDF_norm。
        fluid_mask: ``[B, 1, H, W]`` float；流體格=1，固體格=0。
        _normalizer_getter: 回傳 FieldNormalizer 實例的 callable；lazy-load 用。
        _pod_basis_getter: 回傳 POD 基底字典的 callable；未設定時為 None。
        pod_modes: POD 投影保留模態數，預設 16。
        current_ctu_seq: ``[B, k]`` 各步步首 CTU（HF-1c inflow prior 用）。
        l_char_px: ``[B]`` ROI scale 特徵長度（px）。
        _sdf_raw: 內部快取；請用 :attr:`sdf_raw` property 存取。
        _wall_band: 內部快取；請用 :attr:`wall_band` property 存取。
        _inflow_wmap: 內部快取；請用 :attr:`inflow_weight_map` property 存取。
        _inflow_wmap_done: 標記 inflow_weight_map 是否已嘗試計算，避免重複呼叫。
    """

    pred: Tensor                                          # [B, k, 9, H, W] normalized
    target: Tensor                                        # [B, k, 9, H, W] normalized
    ic_physics: Tensor                                    # [B, 9, H, W] normalized
    static: Tensor                                        # [B, 4, H, W] normalized
    fluid_mask: Tensor                                    # [B, 1, H, W]
    _normalizer_getter: Callable[[], object]
    _pod_basis_getter: Optional[Callable[[], dict[str, Tensor]]] = None
    pod_modes: Optional[int] = None
    # HF-1c：deterministic inflow prior loss-mask（current_ctu，無 target 洩漏）
    current_ctu_seq: Optional[Tensor] = None             # [B, k] raw 步首 CTU
    l_char_px: Optional[Tensor] = None                   # [B] ROI scale 特徵長度
    # SSIM 值域是否讀 stats.json 的 vis_range。mean_step 的直呼點與 registry
    #   路徑（StructVel/Vort_SsimLoss，共 8 個呼叫端）皆已接上這個開關。
    ssim_range_from_stats: bool = False
    _sdf_raw: Optional[Tensor] = None
    _wall_band: Optional[Tensor] = None
    _inflow_wmap: Optional[Tensor] = None
    _inflow_wmap_done: bool = False

    @property
    def normalizer(self) -> object:
        """回傳 lazy-load 的 FieldNormalizer 實例。

        Returns:
            已初始化的 FieldNormalizer 物件。
        """
        return self._normalizer_getter()

    @property
    def inflow_weight_map(self) -> Optional[Tensor]:
        """``[B, k, 1, H, W]`` region weight map。

        若 metadata 不全（current_ctu_seq 或 l_char_px 為 None）或 k 不符，
        回傳 None（退化為標準 MAE，等同 identity weight）。

        Returns:
            inflow 區域加權 map，或 None（退化）。
        """
        if not self._inflow_wmap_done:
            self._inflow_wmap_done = True
            cc, lc = self.current_ctu_seq, self.l_char_px
            if cc is not None and lc is not None and cc.shape[1] == self.pred.shape[1]:
                self._inflow_wmap = inflow_weight_map_seq(
                    cc, lc, self.static, self.fluid_mask
                )
        return self._inflow_wmap

    @property
    def sdf_raw(self) -> Tensor:
        """反正規化後的 SDF 場（lazy 計算）。

        Returns:
            ``[B, H, W]`` 原始尺度 SDF 值。
        """
        if self._sdf_raw is None:
            self._sdf_raw = self.normalizer.denorm_sdf(self.static[:, 0])
        return self._sdf_raw

    @property
    def wall_band(self) -> Tensor:
        """壁面附近 band mask（lazy 計算）。

        Returns:
            ``[B, 1, H, W]`` float；壁面鄰近格=1，其餘=0。
        """
        if self._wall_band is None:
            self._wall_band = compute_wall_band(self.fluid_mask)
        return self._wall_band

    @property
    def pod_basis(self) -> dict[str, Tensor]:
        """POD 基底張量字典；未設定路徑時拋出 ValueError。

        Returns:
            以場名為鍵、POD 基底 Tensor 為值的字典。

        Raises:
            ValueError: 未透過 ``_pod_basis_getter`` 設定 POD 基底路徑。
        """
        if self._pod_basis_getter is None:
            raise ValueError("POD basis is required but no pod_basis_path was configured")
        return self._pod_basis_getter()


# ---------------------------------------------------------------------------
# _LossTerm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LossTerm:
    """單一 loss term 註冊條目。

    Attributes:
        name: YAML 中使用的語義名稱（PascalCase_with_underscores）。
        call: 接受 ``_LossContext`` 並回傳 scalar ``Tensor`` 的函式。
        needs_normalizer: 若 True，``__init__`` 會將此 term 計入「需要
            normalizer」集合，供 trainer 提前實例化 FieldNormalizer。
    """

    name: str
    call: Callable[[_LossContext], Tensor]
    needs_normalizer: bool = False


__all__ = ["_LossContext", "_LossTerm"]
