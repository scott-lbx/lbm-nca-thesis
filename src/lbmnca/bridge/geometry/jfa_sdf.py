"""Taichi JFA SDF — GPU 有符號距離場，替代 scipy.ndimage.distance_transform_edt。

JFA (Jump Flooding Algorithm)：O(log N) GPU passes，每 pass 3×3 Gather。
精度：sub-pixel 誤差 < 0.5px（與 scipy EDT 接近）。
速度：360×640 ~5ms vs scipy ~100ms（含 CPU→GPU→CPU 搬移）。

前提：ti.init() 已由 main.py 呼叫（GGUI 啟動時）。
未初始化時 compute_sdf_taichi() 回 None，呼叫端退回 scipy。
"""
import math
from typing import Optional

import numpy as np

try:
    import taichi as ti  # type: ignore[import]

    @ti.data_oriented
    class JFASdfSolver:
        """JFA 有符號距離場求解器。需在 ti.init() 後建立實例。"""

        def __init__(self, H: int, W: int) -> None:
            self.H = H
            self.W = W
            # ping-pong seed coordinate buffers (explicit attrs for @ti.data_oriented)
            self._sx0 = ti.field(ti.i32, shape=(H, W))
            self._sy0 = ti.field(ti.i32, shape=(H, W))
            self._sx1 = ti.field(ti.i32, shape=(H, W))
            self._sy1 = ti.field(ti.i32, shape=(H, W))
            self._mask    = ti.field(ti.f32, shape=(H, W))
            self._d_solid = ti.field(ti.f32, shape=(H, W))
            self._d_fluid = ti.field(ti.f32, shape=(H, W))
            self._sdf     = ti.field(ti.f32, shape=(H, W))

        @ti.kernel
        def _seed_solid(self):
            for y, x in ti.ndrange(self.H, self.W):
                if self._mask[y, x] < 0.5:
                    self._sx0[y, x] = x
                    self._sy0[y, x] = y
                else:
                    self._sx0[y, x] = -1
                    self._sy0[y, x] = -1

        @ti.kernel
        def _seed_fluid(self):
            for y, x in ti.ndrange(self.H, self.W):
                if self._mask[y, x] >= 0.5:
                    self._sx0[y, x] = x
                    self._sy0[y, x] = y
                else:
                    self._sx0[y, x] = -1
                    self._sy0[y, x] = -1

        @ti.kernel
        def _jfa_step(
            self,
            step: int,
            src_x: ti.template(),
            src_y: ti.template(),
            dst_x: ti.template(),
            dst_y: ti.template(),
        ):
            for y, x in ti.ndrange(self.H, self.W):
                bx = src_x[y, x]
                by = src_y[y, x]
                bd = 1 << 30
                if bx >= 0:
                    bd = (x - bx) * (x - bx) + (y - by) * (y - by)
                for ky in ti.static(range(-1, 2)):
                    for kx in ti.static(range(-1, 2)):
                        ny = y + ky * step
                        nx = x + kx * step
                        if 0 <= ny < self.H and 0 <= nx < self.W:
                            sx2 = src_x[ny, nx]
                            sy2 = src_y[ny, nx]
                            if sx2 >= 0:
                                d2 = (x - sx2) * (x - sx2) + (y - sy2) * (y - sy2)
                                if d2 < bd:
                                    bd = d2
                                    bx = sx2
                                    by = sy2
                dst_x[y, x] = bx
                dst_y[y, x] = by

        @ti.kernel
        def _dist_from(
            self,
            sx: ti.template(),
            sy: ti.template(),
            out: ti.template(),
        ):
            for y, x in ti.ndrange(self.H, self.W):
                bx = sx[y, x]
                by = sy[y, x]
                if bx < 0:
                    out[y, x] = 0.0
                else:
                    ddx = ti.cast(x - bx, ti.f32)
                    ddy = ti.cast(y - by, ti.f32)
                    out[y, x] = ti.sqrt(ddx * ddx + ddy * ddy)

        @ti.kernel
        def _build_sdf(self):
            for y, x in ti.ndrange(self.H, self.W):
                if self._mask[y, x] >= 0.5:
                    self._sdf[y, x] = self._d_solid[y, x]
                else:
                    self._sdf[y, x] = -self._d_fluid[y, x]

        def _run_jfa(self) -> int:
            """Run JFA starting from buf0. Returns buf index with final result."""
            n_rounds = math.ceil(math.log2(max(self.H, self.W))) + 1
            step = 1 << (n_rounds - 1)
            cur = 0
            while step >= 1:
                if cur == 0:
                    self._jfa_step(step, self._sx0, self._sy0, self._sx1, self._sy1)
                else:
                    self._jfa_step(step, self._sx1, self._sy1, self._sx0, self._sy0)
                cur = 1 - cur
                step >>= 1
            return cur

        def _write_dist(self, buf: int, out_field) -> None:
            if buf == 0:
                self._dist_from(self._sx0, self._sy0, out_field)
            else:
                self._dist_from(self._sx1, self._sy1, out_field)

        def compute(self, mask_np: np.ndarray) -> np.ndarray:
            """Signed distance field.

            Args:
                mask_np: [H, W] float32, 0=solid, 1=fluid.

            Returns:
                [H, W] float32 SDF（fluid > 0, solid < 0, px units）.
            """
            self._mask.from_numpy(mask_np.astype(np.float32))

            self._seed_solid()
            self._write_dist(self._run_jfa(), self._d_solid)

            self._seed_fluid()
            self._write_dist(self._run_jfa(), self._d_fluid)

            self._build_sdf()
            return self._sdf.to_numpy()

    _SOLVER_AVAILABLE = True

except ImportError:
    _SOLVER_AVAILABLE = False

_solver = None  # JFASdfSolver | None


def compute_sdf_taichi(mask_np: np.ndarray) -> Optional[np.ndarray]:
    """GPU JFA SDF。Returns [H,W] float32 or None（Taichi 未安裝 / 未 init）。"""
    if not _SOLVER_AVAILABLE:
        return None
    global _solver
    H, W = mask_np.shape
    try:
        if _solver is None or _solver.H != H or _solver.W != W:
            _solver = JFASdfSolver(H, W)  # type: ignore[name-defined]
        return _solver.compute(mask_np)
    except Exception as e:
        print(f"[jfa_sdf] WARN JFA failed ({e}), fallback to scipy")
        return None
