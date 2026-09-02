"""
mask_rasterizer.py
GPU winding number 光柵化引擎，解析度 360×640（H×W）。

重寫自 v1.0 src/middleware/mask_engine.py（ComputeEngine）。
主要差異：
- 解析度由 256×256 改為 360×640（H, W 非正方形）
- mask 慣例：0.0=solid（建物），1.0=fluid（自由流）
  （v1.0 慣例相反：winding≠0→1.0=solid，此處統一改正）
- 移除 GUI canvas 和 _upsample kernel（職責分離）
- 改用 shape=(max_edges, 2, 2) 的 ti.field（與 polygons_to_edges_np 輸出對齊）
"""

import numpy as np
import taichi as ti


@ti.data_oriented
class MaskRasterizer:
    """GPU winding number 光柵化，產生 (H, W) 二元遮罩。

    mask 慣例（對應 ARCHITECTURE_CURRENT.md）：
        0.0 = solid（建物內部）
        1.0 = fluid（自由流通區域）
    """

    def __init__(self, H: int = 360, W: int = 640, max_edges: int = 2000):
        """初始化 Taichi field。

        Args:
            H: 遮罩高度（像素），對應 y 軸。
            W: 遮罩寬度（像素），對應 x 軸。
            max_edges: Taichi edges field 最大邊緣容量。
        """
        self.H = H
        self.W = W
        self.MAX_EDGES = max_edges

        # edges[e, 0] = v0 (x, y)，edges[e, 1] = v1 (x, y)
        # shape 對應 polygons_to_edges_np 輸出的 (max_edges, 2, 2)
        self.edges = ti.field(dtype=ti.f32, shape=(max_edges, 2, 2))

        # mask[i, j]：i ∈ [0, H)，j ∈ [0, W)
        self.mask = ti.field(dtype=ti.f32, shape=(H, W))

        # 實際有效邊緣數
        self.edge_count = ti.field(dtype=ti.i32, shape=())

    # ------------------------------------------------------------------
    # 私有 Taichi kernels
    # ------------------------------------------------------------------

    @ti.kernel
    def _clear_mask(self):
        """[GPU] 將 mask field 清零（全部填 0.0 = solid）。"""
        for i, j in self.mask:
            self.mask[i, j] = 0.0

    @ti.kernel
    def _fill_mask(self):
        """[GPU] 使用 winding number 演算法光柵化多邊形至 mask field。

        座標慣例：
            j（列索引）→ x，正規化為 j / W
            i（行索引）→ y，正規化為 i / H

        遮罩邏輯（0.0=solid, 1.0=fluid）：
            winding ≠ 0  →  pixel 在多邊形內  →  0.0 (solid)
            winding == 0 →  pixel 在多邊形外  →  1.0 (fluid)

        邊緣格式：edges[e, 0, 0/1] = v0.x / v0.y
                  edges[e, 1, 0/1] = v1.x / v1.y
        """
        n_edges = self.edge_count[None]
        H = self.H
        W = self.W

        for i, j in self.mask:
            # 正規化像素中心座標
            px = (j + 0.5) / W
            py = (i + 0.5) / H

            winding = 0

            for e in range(n_edges):
                v0x = self.edges[e, 0, 0]
                v0y = self.edges[e, 0, 1]
                v1x = self.edges[e, 1, 0]
                v1y = self.edges[e, 1, 1]

                is_upward   = v0y <= py < v1y
                is_downward = v1y <= py < v0y

                if is_upward or is_downward:
                    denom = v1y - v0y
                    if ti.abs(denom) > 1e-12:
                        t = (py - v0y) / denom
                        intersect_x = v0x + t * (v1x - v0x)

                        if px < intersect_x:
                            if is_upward:
                                winding += 1
                            else:
                                winding -= 1

            # winding ≠ 0 → solid (0.0)，winding == 0 → fluid (1.0)
            self.mask[i, j] = 0.0 if winding != 0 else 1.0

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def rasterize(self, edges_np: np.ndarray, n_edges: int):
        """光柵化多邊形邊緣至 GPU mask field。

        Args:
            edges_np: shape (max_edges, 2, 2) float32，歸一化 [0, 1] 座標。
                      來自 geometry_preprocess.polygons_to_edges_np 的輸出。
            n_edges:  實際有效邊緣數（≤ max_edges）。
        """
        self._clear_mask()

        if n_edges <= 0:
            # 無邊緣 → 全部 fluid
            self.mask.fill(1.0)
            return

        self.edges.from_numpy(edges_np)
        self.edge_count[None] = n_edges
        self._fill_mask()

    def get_mask_numpy(self) -> np.ndarray:
        """將 GPU mask field 複製回 CPU（偵錯用）。

        Returns:
            shape (H, W) float32 numpy array。
            0.0=solid，1.0=fluid。
        """
        return self.mask.to_numpy()
