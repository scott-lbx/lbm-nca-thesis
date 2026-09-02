"""
GPU field 集中宣告模組。
所有 Taichi field 在此單例初始化，其他模組直接引用。

必須在 ti.init() 之後、任何 @ti.kernel 呼叫之前完成初始化。
"""
import numpy as np
import taichi as ti


class GPUFields:
    """持有所有 Taichi GPU field 的容器。

    初始化後 field 即在 GPU 上分配，其他模組透過此物件引用。
    LUT 初始化是唯一允許的 CPU 操作（一次性，init 時執行）。
    """

    def __init__(self, H: int = 360, W: int = 640) -> None:
        self.H = H
        self.W = W

        # ── 物理場（從 NCA DLPack 接收）──────────────────────────────
        # physics[c, i, j]: 9ch MRT moments，c=channel, i=row, j=col
        self.physics   = ti.field(ti.f32, shape=(9, H, W))
        self.physics_b = ti.field(ti.f32, shape=(9, H, W))  # 插值用第二幀

        # ── 衍生場（從 physics 計算）─────────────────────────────────
        self.vel_x     = ti.field(ti.f32, shape=(H, W))
        self.vel_y     = ti.field(ti.f32, shape=(H, W))
        self.vel_mag   = ti.field(ti.f32, shape=(H, W))
        self.vorticity = ti.field(ti.f32, shape=(H, W))
        self.pressure  = ti.field(ti.f32, shape=(H, W))

        # ── 幾何場──────────────────────────────────────────────────
        # mask: 0.0=solid(建物), 1.0=fluid(流體)
        self.mask      = ti.field(ti.f32, shape=(H, W))

        # ── 收斂場（convergence channel, 1=active 0=settled；無則全 0）──
        self.convergence = ti.field(ti.f32, shape=(H, W))

        # ── 流線場（N_SEEDS/MAX_STEPS SSOT 見 src/render/streamlines.py）──
        self.seed_pos   = ti.Vector.field(2, ti.f32, shape=100)
        self.line_verts = ti.Vector.field(2, ti.f32, shape=100 * 300)

        # ── IBFV ping-pong + noise 場（單層；雙層時 mean/turb 各持一組）──
        # ibfv_*_a / ibfv_*_b: 灰階紋理 [H, W]，ping-pong 交換（F_prev / F_curr）。
        # ibfv_noise: 每幀重生的 animated 噪聲暫存 [H, W]。
        # 命名前綴 mean/turb 對應雙層語意（Phase A 僅用 mean）。
        self.ibfv_mean_a  = ti.field(ti.f32, shape=(H, W))
        self.ibfv_mean_b  = ti.field(ti.f32, shape=(H, W))
        self.ibfv_turb_a  = ti.field(ti.f32, shape=(H, W))
        self.ibfv_turb_b  = ti.field(ti.f32, shape=(H, W))
        self.ibfv_noise   = ti.field(ti.f32, shape=(H, W))

        # ── 時間軸 UI（預配置一次，禁止每幀 ti.Vector.field！）────────
        self.tl_bg  = ti.Vector.field(2, ti.f32, shape=200)  # 背景點（靜態）
        self.tl_ph  = ti.Vector.field(2, ti.f32, shape=1)    # playhead
        self.tl_gen = ti.Vector.field(2, ti.f32, shape=1)    # generation head
        self._init_timeline_bg()

        # ── 輸出場（送至 canvas.set_image，GGUI 自動 upscale）────────
        # 固定 NCA 解析度，不需要 display_scale
        # rgb 為運算慣例 (H, W)（i=row=y, j=col=x）；
        # 但 Taichi canvas.set_image 期望 (W, H) x-major → 需轉置到 rgb_disp 再送，
        # 否則畫面被壓縮/變形（W:H=640:360 landscape 比例跑掉）。
        self.rgb = ti.Vector.field(3, ti.f32, shape=(H, W))
        self.rgb_disp = ti.Vector.field(3, ti.f32, shape=(W, H))

        # ── Colormap LUT（一次性 CPU 烘入）──────────────────────────
        # 格式：lut[idx] → ti.Vector[3]，idx ∈ [0, 255]
        self.lut_plasma    = ti.Vector.field(3, ti.f32, shape=256)
        self.lut_rdbu      = ti.Vector.field(3, ti.f32, shape=256)
        self.lut_vorticity = ti.Vector.field(3, ti.f32, shape=256)
        self.lut_comfort   = ti.Vector.field(3, ti.f32, shape=256)

        # vmax scalar field（Taichi parallel reduction 用）
        self._vmax = ti.field(ti.f32, shape=())

        self._init_luts()

    def _init_timeline_bg(self) -> None:
        """背景點位置靜態，一次性烘入（取代每幀 field 配置）。"""
        x0, x1, y = 0.05, 0.95, 0.93
        n = 200
        bg = np.zeros((n, 2), dtype=np.float32)
        bg[:, 0] = np.linspace(x0, x1, n)
        bg[:, 1] = y
        self.tl_bg.from_numpy(bg)

    def _init_luts(self) -> None:
        """從 matplotlib 烘入 colormap LUT。唯一允許的 CPU 初始化操作。"""
        import matplotlib.cm as cm

        xs = np.linspace(0.0, 1.0, 256)

        # plasma
        arr = cm.get_cmap("plasma")(xs)[:, :3].astype(np.float32)
        self.lut_plasma.from_numpy(arr)

        # RdBu_r（高壓=藍，低壓=紅）
        arr = cm.get_cmap("RdBu_r")(xs)[:, :3].astype(np.float32)
        self.lut_rdbu.from_numpy(arr)

        # 渦度：與 02 utils/FieldColorizer.py 的 vor_cmap 完全一致（5 個等距色點的線性
        # 分段插值：黃→橘→黑（中心）→綠→青），確保 reference mp4（用 FieldColorizer
        # 上色）與 live GGUI 的渦度顏色一致，不各自漂移。
        vort_stops = np.array([
            [1.0,   1.0,   0.0],
            [0.953, 0.490, 0.016],
            [0.0,   0.0,   0.0],
            [0.176, 0.976, 0.529],
            [0.0,   1.0,   1.0],
        ], dtype=np.float32)
        n_stops = vort_stops.shape[0] - 1
        seg = np.clip((xs * n_stops).astype(np.int32), 0, n_stops - 1)
        frac = (xs * n_stops) - seg
        vort = vort_stops[seg] * (1.0 - frac[:, None]) + vort_stops[seg + 1] * frac[:, None]
        self.lut_vorticity.from_numpy(vort.astype(np.float32))

        # 舒適度：綠(BF0-3) → 黃(BF4-5) → 橘(BF6-7) → 紅(BF8+)
        # 對應速度 0→5.5→10.8→17.2→inf m/s，映射至 [0,1] 用 v/20 clip
        comfort = np.zeros((256, 3), dtype=np.float32)
        for i, x in enumerate(xs):
            if x < 0.275:   # BF 0-3: 綠
                comfort[i] = [0.2, 0.8, 0.2]
            elif x < 0.54:  # BF 4-5: 黃
                comfort[i] = [1.0, 0.9, 0.0]
            elif x < 0.86:  # BF 6-7: 橘
                comfort[i] = [1.0, 0.5, 0.0]
            else:            # BF 8+: 紅
                comfort[i] = [0.9, 0.1, 0.1]
        self.lut_comfort.from_numpy(comfort)
