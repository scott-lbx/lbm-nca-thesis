"""單層 IBFV（Image Based Flow Visualization）pass，Taichi GPU 實作。

IBFV（van Wijk 2002）= 沿驅動速度場平流一張噪聲紋理，並每幀混入新 animated
噪聲，產生連續流動觀感。本模組實作**單層** ping-pong pass；雙層（mean / turb）
由上層各持一個 `IBFVPass` 實例（綁不同 ping-pong field）達成。

核心更新（每 render frame，對齊 SSOT_SPEC_dual_layer_ibfv.md §1/§6）：

    F_curr(x) = (1 - α) · F_prev( x - v(x)·dt )  +  α · noise(x, t)
    swap(F_prev, F_curr)

座標與取樣約定（與 `streamlines.py` / `field_ops.py` 完全一致）：
    - field 形狀 (H, W)，索引 `field[i, j]`，i = row = y、j = col = x。
    - 歸一化座標 px = x ∈ [0, 1] 對應 W；py = y ∈ [0, 1] 對應 H。
    - bilinear 取樣 `xi = px·W − 0.5`（像素中心對齊），與 `bilinear_vel` 同式。
    - mask：< 0.5 = solid、>= 0.5 = fluid；solid 區紋理歸零。

設計原則：
    - 不自行呼叫 `ti.init()`；所有 Taichi field 由 `GPUFields` 提供（建構時注入）。
    - config 存取一律 `cfg.field`（`IBFVParams` dataclass，無 getattr fallback）。
"""
import taichi as ti

from lbmnca.bridge.render.ibfv_params import IBFVParams


@ti.func
def _hash21(p: ti.types.vector(2, ti.f32)) -> ti.f32:
    """2D → 1D 雜湊（GPU 端 value-noise 用），回傳 [0, 1)。

    Args:
        p: 二維輸入座標。

    Returns:
        擬隨機純量，落於 [0, 1)。
    """
    h = p.dot(ti.Vector([127.1, 311.7]))
    return ti.math.fract(ti.sin(h) * 43758.5453123)


@ti.func
def _value_noise(x: ti.f32, y: ti.f32, phase: ti.f32) -> ti.f32:
    """animated value noise，雙線性平滑 + 時間相位調變，回傳 [0, 1]。

    Args:
        x: 已乘上空間頻率的 x 座標。
        y: 已乘上空間頻率的 y 座標。
        phase: 時間相位（rad），令噪聲隨時間流動。

    Returns:
        平滑噪聲值，落於 [0, 1]。
    """
    xi = ti.floor(x)
    yi = ti.floor(y)
    fx = x - xi
    fy = y - yi
    # smoothstep 權重
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    n00 = _hash21(ti.Vector([xi, yi]))
    n10 = _hash21(ti.Vector([xi + 1.0, yi]))
    n01 = _hash21(ti.Vector([xi, yi + 1.0]))
    n11 = _hash21(ti.Vector([xi + 1.0, yi + 1.0]))

    nx0 = n00 * (1.0 - ux) + n10 * ux
    nx1 = n01 * (1.0 - ux) + n11 * ux
    base = nx0 * (1.0 - uy) + nx1 * uy
    # 時間相位調變（animated）：相位平移後重映射至 [0, 1]
    return 0.5 + 0.5 * ti.sin(base * 6.2831853 + phase)


@ti.kernel
def seed_noise(
    tex: ti.template(),
    mask: ti.template(),
    freq: ti.f32,
    H: int,
    W: int,
):
    """以靜態噪聲初始化紋理場（首幀避免全黑）。solid 區歸零。"""
    for i, j in tex:
        if mask[i, j] < 0.5:
            tex[i, j] = 0.0
        else:
            px = (ti.cast(j, ti.f32) + 0.5) / ti.cast(W, ti.f32)
            py = (ti.cast(i, ti.f32) + 0.5) / ti.cast(H, ti.f32)
            tex[i, j] = _value_noise(px * freq, py * freq, 0.0)


@ti.kernel
def advect_blend(
    prev: ti.template(),
    curr: ti.template(),
    vel_x: ti.template(),
    vel_y: ti.template(),
    mask: ti.template(),
    alpha: ti.f32,
    dt_advect: ti.f32,
    noise_freq: ti.f32,
    noise_phase: ti.f32,
    injection_floor: ti.f32,
    H: int,
    W: int,
):
    """單步 IBFV 更新：backward-warp 平流 prev + α 混入 animated 噪聲 → curr。

    對每個輸出像素 (i, j)，沿驅動速度往回取樣 `x − v·dt` 處的 prev 紋理值，
    再依 α 混入該位置的 animated 噪聲。solid 區強制歸零。

    Args:
        prev: 上一幀紋理 [H, W]（F_prev，唯讀）。
        curr: 本幀輸出紋理 [H, W]（F_curr，寫入）。
        vel_x: 驅動速度 x 分量 [H, W]（LBM 格子速度）。
        vel_y: 驅動速度 y 分量 [H, W]。
        mask: 幾何遮罩 [H, W]，< 0.5 = solid。
        alpha: 時間混合係數 α。
        dt_advect: 平流步長（px / frame，作用於格子速度）。
        noise_freq: 噪聲空間頻率。
        noise_phase: 噪聲時間相位（rad）。
        injection_floor: 噪聲注入下限（保底，避免長期不更新）。
        H: 場高。
        W: 場寬。
    """
    for i, j in curr:
        if mask[i, j] < 0.5:
            curr[i, j] = 0.0
        else:
            # 本像素中心的歸一化座標
            px = (ti.cast(j, ti.f32) + 0.5) / ti.cast(W, ti.f32)
            py = (ti.cast(i, ti.f32) + 0.5) / ti.cast(H, ti.f32)

            # backward warp：往回走 -v·dt（v 取本像素速度，單位 px/frame）
            u = vel_x[i, j]
            v = vel_y[i, j]
            src_jf = ti.cast(j, ti.f32) + 0.5 - u * dt_advect
            src_if = ti.cast(i, ti.f32) + 0.5 - v * dt_advect

            # bilinear 取樣 prev（像素中心對齊，clamp 邊界，與 bilinear_vel 同式）
            xi = src_jf - 0.5
            yi = src_if - 0.5
            x0 = ti.cast(ti.floor(xi), ti.i32)
            y0 = ti.cast(ti.floor(yi), ti.i32)
            x1 = ti.min(x0 + 1, W - 1)
            y1 = ti.min(y0 + 1, H - 1)
            x0 = ti.max(x0, 0)
            y0 = ti.max(y0, 0)
            fx = ti.math.clamp(xi - ti.cast(x0, ti.f32), 0.0, 1.0)
            fy = ti.math.clamp(yi - ti.cast(y0, ti.f32), 0.0, 1.0)

            advected = (prev[y0, x0] * (1.0 - fx) * (1.0 - fy)
                      + prev[y0, x1] * fx * (1.0 - fy)
                      + prev[y1, x0] * (1.0 - fx) * fy
                      + prev[y1, x1] * fx * fy)

            noise = _value_noise(px * noise_freq, py * noise_freq, noise_phase)
            blend_a = ti.max(alpha, injection_floor)
            curr[i, j] = (1.0 - blend_a) * advected + blend_a * noise


class IBFVPass:
    """單層 IBFV ping-pong pass。

    持有一組 (a, b) 紋理 field 與一個驅動速度的 advect-blend 循環。每次
    `render()` 推進一個 IBFV step 並交換 ping-pong，回傳當前紋理 field。
    Taichi field 由 `GPUFields` 提供（建構時注入），本類不配置 field、
    不呼叫 `ti.init()`。

    雙層用法：mean / turb 各建一個 `IBFVPass`，分別綁
    `(gpu.ibfv_mean_a, gpu.ibfv_mean_b)` 與 `(gpu.ibfv_turb_a, gpu.ibfv_turb_b)`。
    """

    def __init__(
        self,
        tex_a: ti.Field,
        tex_b: ti.Field,
        H: int,
        W: int,
    ) -> None:
        """初始化單層 IBFV pass。

        Args:
            tex_a: ping-pong 紋理 field A [H, W]（來自 GPUFields）。
            tex_b: ping-pong 紋理 field B [H, W]（來自 GPUFields）。
            H: 場高。
            W: 場寬。
        """
        self.tex_a = tex_a
        self.tex_b = tex_b
        self.H = H
        self.W = W
        self._frame = 0
        self._prev_is_a = True  # True: prev=a, curr=b；交換後反轉
        self._seeded = False

    def init(self, mask: ti.Field, cfg: IBFVParams) -> None:
        """以靜態噪聲填充兩組 ping-pong field（首幀避免全黑）。

        Args:
            mask: 幾何遮罩 [H, W]，solid 區歸零。
            cfg: 本層 IBFV 參數。
        """
        seed_noise(self.tex_a, mask, cfg.noise_freq, self.H, self.W)
        seed_noise(self.tex_b, mask, cfg.noise_freq, self.H, self.W)
        self._prev_is_a = True
        self._frame = 0
        self._seeded = True

    def render(
        self,
        vel_x: ti.Field,
        vel_y: ti.Field,
        mask: ti.Field,
        cfg: IBFVParams,
    ) -> ti.Field:
        """推進一個 IBFV step 並回傳當前紋理 field。

        驅動速度場以兩個 scalar field `(vel_x, vel_y)` 表示（單位 = LBM 格子
        速度，由 `field_ops.derive_fields` 自 moments 推得）。雙層時：
        mean 層傳 derive(mean_moments)；turb 層傳 derive(mean)+derive(fluct)
        相加後的合成速度（見 dual_layer SPEC §3.3）。

        Args:
            vel_x: 驅動速度 x 分量 [H, W]。
            vel_y: 驅動速度 y 分量 [H, W]。
            mask: 幾何遮罩 [H, W]。
            cfg: 本層 IBFV 參數（dataclass，bracket-free 直接存取）。

        Returns:
            本幀的當前紋理 field [H, W]（灰階，[0, 1]），供著色 / 合成使用。
        """
        if not self._seeded:
            self.init(mask, cfg)

        prev = self.tex_a if self._prev_is_a else self.tex_b
        curr = self.tex_b if self._prev_is_a else self.tex_a

        phase = self._frame * cfg.noise_time_rate
        advect_blend(
            prev, curr, vel_x, vel_y, mask,
            cfg.alpha, cfg.dt_advect, cfg.noise_freq, phase,
            cfg.injection_floor, self.H, self.W,
        )

        self._prev_is_a = not self._prev_is_a
        self._frame += 1
        return curr
