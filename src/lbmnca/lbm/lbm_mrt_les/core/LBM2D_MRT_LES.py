import numpy as np
import taichi as ti
import taichi.math as tm
import math as _math


@ti.func
def mrt_moments_from_f(f_val):
    """由單點分布函數 f 投影出 D2Q9 MRT 9 通道矩（Lallemand & Luo 順序）。

    ★ 唯一真相：`compute_moments_for_output`（snapshot 路徑）與
    `core.time_mean_accumulator`（求解迴圈內逐步時均累加路徑）**共用**此函式，
    避免兩條路徑的矩定義漂移（DEEP_AUDIT_2026-07 Stage 1.15）。

    Args:
        f_val: 長度 9 的分布函數向量（單一格點）。

    Returns:
        長度 9 的矩向量 `[rho, e, eps, jx, qx, jy, qy, pxx, pxy]`。
    """
    # --- 0. 密度 rho ---
    rho = f_val.sum()

    # --- 1. 能量 e (-4, -1, -1, -1, -1, 2, 2, 2, 2) ---
    e = (
        -4.0 * f_val[0]
        - (f_val[1] + f_val[2] + f_val[3] + f_val[4])
        + 2.0 * (f_val[5] + f_val[6] + f_val[7] + f_val[8])
    )

    # --- 2. 能量平方 epsilon (4, -2, -2, -2, -2, 1, 1, 1, 1) ---
    eps = (
        4.0 * f_val[0]
        - 2.0 * (f_val[1] + f_val[2] + f_val[3] + f_val[4])
        + (f_val[5] + f_val[6] + f_val[7] + f_val[8])
    )

    # --- 3. 動量 jx (0, 1, 0, -1, 0, 1, -1, -1, 1) ---
    jx = f_val[1] - f_val[3] + f_val[5] - f_val[6] - f_val[7] + f_val[8]

    # --- 4. 熱通流 qx (0, -2, 0, 2, 0, 1, -1, -1, 1) ---
    qx = -2.0 * f_val[1] + 2.0 * f_val[3] + f_val[5] - f_val[6] - f_val[7] + f_val[8]

    # --- 5. 動量 jy (0, 0, 1, 0, -1, 1, 1, -1, -1) ---
    jy = f_val[2] - f_val[4] + f_val[5] + f_val[6] - f_val[7] - f_val[8]

    # --- 6. 熱通流 qy (0, 0, -2, 0, 2, 1, 1, -1, -1) ---
    qy = -2.0 * f_val[2] + 2.0 * f_val[4] + f_val[5] + f_val[6] - f_val[7] - f_val[8]

    # --- 7. 正應力 pxx (0, 1, -1, 1, -1, 0, 0, 0, 0) ---
    pxx = f_val[1] - f_val[2] + f_val[3] - f_val[4]

    # --- 8. 切應力 pxy (0, 0, 0, 0, 0, 1, -1, 1, -1) ---
    pxy = f_val[5] - f_val[6] + f_val[7] - f_val[8]

    return ti.Vector([rho, e, eps, jx, qx, jy, qy, pxx, pxy])


@ti.data_oriented
class LBM2D_MRT_LES:
    # ------------------------------------------------
    # region init
    def __init__(self, config, mask_data=None):
        """
        LBM Solver 初始化入口
        """
        self.config = config

        # 求解迴圈內時均累加器（DEEP_AUDIT_2026-07 Stage 1.15）。
        # None = 未掛載（既有行為，零額外成本）。由
        # `attach_time_mean_accumulator` 掛上後，`run_step` 每步呼叫一次
        # `accumulate()`，把時均從「事後平均 snapshot」改成「逐步累加」。
        self._time_mean_accum = None

        # 1. 讀取模擬與物理參數
        self._init_params()

        # 2. 配置 Taichi 記憶體 (Fields)
        self._init_fields(mask_data)

        # 3. 定義 LBM/MRT 常數與矩陣 (優化重點)
        self._init_constants()

        # # 4. 初始化場數值 (由 0 開始或讀取設定)
        # self.init_sim()

    #  init 子函式: 參數讀取
    def _init_params(self):
        sim_cfg = self.config["simulation"]

        # 基礎幾何與時間
        self.name = sim_cfg["name"]
        self.nx = sim_cfg["nx"]
        self.ny = sim_cfg["ny"]
        self.steps_per_frame = sim_cfg["compute_step_size"]
        self.warmup_steps = sim_cfg["warmup_steps"]

        # 物理參數
        self.nu = sim_cfg["nu"]
        self.tau_0 = 3.0 * self.nu + 0.5
        # ★2026-07-29（P2-B 建議 2）：tau_0 <= 0.5 使 s_nu = 1/tau_0 >= 2，
        #   MRT 鬆弛越過穩定上限 → 靜默發散（不是報錯，是數值爆掉後才被 NaN
        #   守衛抓到，浪費整個 case 的 wall time）。原本唯一的把關在
        #   gen_c_series_configs 的 check_feasibility，而它是 warn-only 且
        #   不涵蓋「直接餵 config 給 solver」的路徑。此處硬擋，禁 fallback。
        if not self.tau_0 > 0.5:
            raise ValueError(
                f"tau_0 = 3*nu + 0.5 = {self.tau_0:.6f} 未大於 0.5"
                f"（nu = {self.nu:.8g}）。s_nu = 1/tau_0 >= 2 會使 MRT 鬆弛"
                "越過穩定上限而發散。請提高 nu（降 Re 或增大 L_char）。"
            )

        # ==========================================
        # [新增] 供外部呼叫的物理屬性 (Re, U_inlet)
        # ==========================================
        # 1. 特徵長度
        self.characteristic_length = sim_cfg["characteristic_length"]

        # 2. 速度入口 + 壓力出口
        # bracket 存取、禁 fallback（CLAUDE.md / audit B2）：config 須明寫 u_inlet，
        # 否則 nu(由 U_INLET 反推) 與 solver inlet 脫鉤會靜默污染實際 Re。
        self.u_inlet_target = float(sim_cfg["u_inlet"])
        # ★N1：矩的匯出時間點。"pre_collision"=post-streaming（物理正確，預設）、
        #   "post_collision"=舊行為（非守恆通道帶 (1-s) 因子）。
        # ★ping-pong 資料流（2026-07-30）。True 時每步只跑 2 個 kernel
        #   （collide+stream 直寫 rho/vel → apply_bc），省掉 update_macro_var
        #   的整場 f 複製（265 MB/步）。缺鍵時預設啟用；設 False 可切回原路徑。
        self.pingpong = bool(sim_cfg["pingpong"] if "pingpong" in sim_cfg else True)
        self._pp_parity = 0

        self.moments_export_stage = str(
            sim_cfg["moments_export_stage"]
            if "moments_export_stage" in sim_cfg
            else "pre_collision"
        )
        if self.pingpong and self.moments_export_stage == "post_collision":
            raise ValueError(
                "pingpong=True 與 moments_export_stage='post_collision' 不相容："
                "ping-pong 下最新的分布場在 f_old/f_new 之間交替，固定讀 f_new "
                "會有一半的步讀到上一步的舊場。"
                "pre_collision（預設）不受影響——它的矩在 collide 內寫入 "
                "moments_pre_field，與哪個場當 dst 無關。"
                "要用 post_collision 對照請同時設 pingpong=False。"
            )
        if self.moments_export_stage not in ("pre_collision", "post_collision"):
            raise ValueError(
                f"moments_export_stage 只接受 'pre_collision' / 'post_collision'，"
                f"實得 {self.moments_export_stage!r}"
            )
        # ⚠★2026-07-29 L3 驗收結論：`bc_type = 3` **不可用於生產**。
        #   它在無障礙均勻流上確實把出口影響帶壓掉 14 倍（4.0e-2 → 1.2e-4），
        #   但在**含建物的真實尾流**下會破壞質量守恆——六個代表性 case 無一例外，
        #   T3.2（漂移/振盪）從 bc=1 的 0.03-0.37 惡化到 2.0-16.2（最高 506 倍）。
        #   實作保留供未來研究，生產一律用 `bc_type = 1`。
        #   詳見 docs/reviews/DEEP_AUDIT_2026-07/P3_L3_FINDINGS_2026-07-29.md §2。
        #
        # ★convective outflow（bc_type=3）的對流速度。物理上聲波以 u + c_s 傳播
        #   （順流方向的相速度），用純流速 u 會讓聲波幾乎不被對流出去——實測
        #   cfl=u=0.05 時 λ≈320px 的反射只從 R=1.045 降到 0.906。
        #   c_s = 1/sqrt(3) ≈ 0.5774；u + c_s ≈ 0.627 仍滿足 CFL ≤ 1。
        #   為明參（禁 fallback）：缺鍵即用 u + c_s，該預設寫在 config 生成端。
        self.outlet_conv_speed = float(
            sim_cfg["outlet_conv_speed"]
            if "outlet_conv_speed" in sim_cfg
            else self.u_inlet_target + 1.0 / np.sqrt(3.0)
        )
        self.rho_out_target = sim_cfg["rho_out"]

        # Re 直接由 u_inlet 計算（精確）
        if self.nu > 0:
            self.Re = (self.u_inlet_target * self.characteristic_length) / self.nu
        else:
            self.Re = float("inf")

        print(
            f"[Solver] Velocity-inlet: u_inlet={self.u_inlet_target:.5f}, "
            f"rho_out={self.rho_out_target}, Re={self.Re:.1f}"
        )
        # ==========================================

        # LES (大渦模擬) 參數
        self.C_smag = sim_cfg["smagorinsky_constant"]
        self.Cs_sq_factor = 18.0 * (self.C_smag**2)

        # MRT 鬆弛參數
        self.S_other = sim_cfg["ghost_moments_s"]
        # 視覺化參數。★這是專案「禁 fallback config 存取」規則的**明確例外**：
        #   `outputs.gui` 是純顯示設定，不進入任何物理計算或輸出資料，TES pipeline
        #   的 per-case YAML 本來就不含該區塊。缺鍵用預設不會讓「實際跑的參數 !=
        #   config 所寫」，因此不適用該規則的立論。物理參數一律直接索引。
        _gui_cfg = self.config.get("outputs", {}).get("gui", {})
        self.viz_sigma = _gui_cfg.get("gaussian_sigma", 1.0)

        # ==========================================
        # Sponge Layer Parameters from Config
        # ==========================================
        zones = self.config["domain_zones"]
        # ★2026-07-29 更正：本行原註「inlet sponge = 0：壓差 BC 驅動」——兩點都已過時。
        #   (1) 壓差驅動早已廢棄，現行是 Zou-He velocity inlet（見 `apply_bc_core` type 0）。
        #   (2) 生產 `sponge_in = 80`（非 0），見 `gen_c_series_configs.SPONGE_IN`。
        self.sponge_w_in = int(zones["sponge_in"])
        self.sponge_w_out = max(1, zones["sponge_out"])
        self.sponge_w_top = max(1, zones["sponge_top"])
        self.sponge_w_bot = max(1, zones["sponge_bot"])
        self.sponge_strength = zones["sponge_strength"]

        # Phase 4: Sponge macro relaxation 參數
        # ★禁 fallback（專案規則）：原用 `zones.get(key, default)`，缺鍵時會靜默
        #   採用預設值，使實際跑的參數 != config 所寫。三個鍵在生產 config 生成器
        #   （`gen_c_series_configs.py` domain_zones）皆明寫，改為直接索引，缺鍵即 KeyError。
        self.sponge_relax_strength = float(zones["sponge_relax_strength"])
        self.sponge_ref_u_factor = float(zones["sponge_reference_u_factor"])
        self.sponge_ref_v = float(zones["sponge_reference_v_target"])

        # 參考入口速度（velocity-inlet 模式直接使用 prescribed 值）
        self.u_inlet_est = self.u_inlet_target

    #  init 子函式: 記憶體配置 (Fields)
    def _init_fields(self, mask_data):
        # 巨觀量 (Macro-scopic)
        self.rho = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.ny))

        # 分布函數 (Micro-scopic, f_old / f_new)
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))

        # 遮罩 (Mask)
        self.mask = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        if mask_data is not None:
            self.mask.from_numpy(mask_data.astype(np.float32))
        else:
            self.mask.fill(0.0)

        # 邊界條件 (Boundary Conditions)
        bc_cfg = self.config["boundary_condition"]
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.Vector.field(2, dtype=ti.f32, shape=4)

        self.bc_type.from_numpy(np.array(bc_cfg["type"], dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_cfg["value"], dtype=np.float32))

        # 統計與計數
        self.frame_count = ti.field(dtype=ti.i32, shape=())

        # 用於儲存障礙物受力的累加器 (0D Field)
        self.force_sum = ti.Vector.field(2, dtype=ti.f32, shape=())

        # 用於儲存 MRT 矩空間數據 (9個通道)，供數據集輸出使用
        self.moments_field = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))
        # ★N1 修正（2026-07-29）：碰撞**前**（= post-streaming）的矩。
        #   物理狀態的矩定義在 post-streaming 分布上；`f_new` 是碰撞**後**的分布，
        #   其非守恆通道（e, eps, qx, qy, pxx, pxy）帶因子 (1 - s)：
        #       m*_i = m_eq,i + (1 - s_i)·(m_i - m_eq,i)
        #   剪切通道的 s = 1/tau_eff，生產 tau 範圍 0.5167-1.1471 使
        #   (1 - s) 從 -0.935 掃到 +0.128 並在 tau_eff = 1 時**歸零**——
        #   同一物理狀態的應力會隨 Re 變號甚至消失。守恆通道（rho, jx, jy）
        #   因碰撞守恆而不受影響，故此缺陷在只看速度場時完全隱形。
        self.moments_pre_field = ti.Vector.field(
            9, dtype=ti.f32, shape=(self.nx, self.ny)
        )

    #  init 子函式: 常數與矩陣 (Constants)
    def _init_constants(self):
        # D2Q9 權重
        self.w = ti.types.vector(9, ti.f32)(
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        )

        # D2Q9 離散速度向量
        self.e = ti.types.matrix(9, 2, ti.i32)(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ]
        )

        # --- MRT 轉換矩陣 (核心優化) ---
        # 這裡將 M 定義為 Taichi Matrix 而非 Field，
        # 定義 9x9 的場來儲存矩陣
        self.M_field = ti.field(dtype=ti.f32, shape=(9, 9))
        self.invM_field = ti.field(dtype=ti.f32, shape=(9, 9))

        # 定義 numpy 數據
        M_np = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                [4, -2, -2, -2, -2, 1, 1, 1, 1],
                [0, 1, 0, -1, 0, 1, -1, -1, 1],
                [0, -2, 0, 2, 0, 1, -1, -1, 1],
                [0, 0, 1, 0, -1, 1, 1, -1, -1],
                [0, 0, -2, 0, 2, 1, 1, -1, -1],
                [0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 1, -1],
            ],
            dtype=np.float32,
        )

        invM_np = np.linalg.inv(M_np).astype(np.float32)

        # 將數據寫入 Field (這一步會在 Python 端執行一次)
        self.M_field.from_numpy(M_np)
        self.invM_field.from_numpy(invM_np)

        # MRT 鬆弛對角矩陣 (S vector)
        # 用於碰撞步驟: m* = m - S * (m - m_eq)
        # S_other 是 Ghost Moments 的鬆弛率
        self.S_base = ti.types.vector(9, ti.f32)(
            0.0,  # density (conserved)
            self.S_other,  # energy
            self.S_other,  # epsilon
            0.0,  # jx (conserved)
            self.S_other,  # qx
            0.0,  # jy (conserved)
            self.S_other,  # qy
            # ★pxx / pxy **不是**守恆量（原註解「conserved in standard LBM」有誤）。
            #   D2Q9 MRT 的守恆矩只有 rho、jx、jy 三個。此處填 0.0 僅為佔位——
            #   兩者的實際鬆弛率在 collide 內被 `S_local[7] = S_local[8] = s_eff`
            #   （= 1/tau_eff，含 LES 與 sponge）逐格覆寫。
            0.0,  # pxx 佔位（實際 s 由 s_eff 動態覆寫）
            0.0,  # pxy 佔位（同上）
        )

    # endregion

    # ------------------------------------------------
    # region LBM MRT-LES Kernels and Functions
    def get_physical_fields(self):
        """
        將 GPU 上的速度場與遮罩數據導出為 NumPy 數組
        """
        # .to_numpy() 會自動處理數據同步與拷貝
        return self.vel.to_numpy(), self.mask.to_numpy()

    @ti.func
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.func
    def get_meq(self, rho, u, v):
        u2 = u * u + v * v
        return ti.types.vector(9, float)(
            rho,
            rho * (-2.0 + 3.0 * u2),
            rho * (1.0 - 3.0 * u2),
            rho * u,
            -rho * u,
            rho * v,
            -rho * v,
            rho * (u * u - v * v),
            rho * u * v,
        )

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.frame_count[None] = 0
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def _init_from_rho_vel(self):
        """Initialize f_old = f_new = f_eq(ρ, u) from pre-loaded rho/vel fields."""
        self.frame_count[None] = 0
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    def init_warm_start(self, rho_np: np.ndarray, vel_np: np.ndarray) -> None:
        """Initialize solver from a warm-start state using equilibrium distributions.

        Loads the given density and velocity fields into GPU memory, then sets
        f_old = f_new = f_eq(ρ, u) for every lattice node.

        Args:
            rho_np: Density field [nx, ny] float32.
            vel_np: Velocity field [nx, ny, 2] float32 (lattice units).
        """
        self.rho.from_numpy(rho_np.astype(np.float32))
        self.vel.from_numpy(vel_np.astype(np.float32))
        self._init_from_rho_vel()

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):

            # 固體節點不參與碰撞與串流，保持 f 不變
            if self.mask[i, j] == 1.0:
                self.f_new[i, j] = self.f_old[i, j]
                continue

            # ============================================================
            # 1. Streaming + Half-way Bounce-Back (Phase 4A)
            # ------------------------------------------------------------
            # Pull-scheme: 從鄰居節點拉取分布函數
            # HWBB: 若鄰居為固體，取自身反方向的值（壁面在半格位置，二階精度）
            # D2Q9 opposite: 0→0, 1↔3, 2↔4, 5↔7, 6↔8
            # ============================================================
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                if self.mask[ip, jp] == 1.0:
                    # HWBB: 粒子在半格壁面反彈，取自身反方向
                    k_opp = ti.static([0, 3, 4, 1, 2, 7, 8, 5, 6][k])
                    f_temp[k] = self.f_old[i, j][k_opp]
                else:
                    f_temp[k] = self.f_old[ip, jp][k]

            # ============================================================
            # 2. MRT Moment Transformation (矩陣轉換：速度空間 → 矩空間)
            # ------------------------------------------------------------
            # m = M · f
            # M 為 9×9 的 D2Q9 MRT 轉換矩陣 (Lallemand & Luo, 2000)
            # 矩空間順序：[rho, e, eps, jx, qx, jy, qy, pxx, pxy]
            # ============================================================
            m = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.M_field[r, c] * f_temp[c]
                m[r] = val

            # ★N1：碰撞前的矩即為此處的 m（由 post-streaming 的 f_temp 投影）。
            self.moments_pre_field[i, j] = m

            # ============================================================
            # 3. Macroscopic Variables (計算巨觀量：密度與速度)
            # ------------------------------------------------------------
            # 守恆量直接從矩空間讀取：
            #   rho = m[0] = Σ f_k
            #   j_x = m[3] = Σ e_kx · f_k  →  u = jx / rho
            #   j_y = m[5] = Σ e_ky · f_k  →  v = jy / rho
            # ============================================================
            rho_l = m[0]
            u_l, v_l = 0.0, 0.0
            if rho_l > 0:
                u_l, v_l = m[3] / rho_l, m[5] / rho_l

            # ============================================================
            # 4. Equilibrium Moments (計算平衡矩)
            # ------------------------------------------------------------
            # m_eq 的完整定義 (見 get_meq)：
            #   m_eq[0] = rho
            #   m_eq[1] = rho(-2 + 3u²)      [能量 e]
            #   m_eq[2] = rho( 1 - 3u²)      [能量平方 eps]
            #   m_eq[3] = rho·u              [動量 jx]
            #   m_eq[4] = -rho·u             [熱通流 qx]
            #   m_eq[5] = rho·v              [動量 jy]
            #   m_eq[6] = -rho·v             [熱通流 qy]
            #   m_eq[7] = rho(u² - v²)       [正應力差 pxx]
            #   m_eq[8] = rho·u·v            [切應力 pxy]
            # ============================================================
            m_eq = self.get_meq(rho_l, u_l, v_l)

            # ============================================================
            # 5. LES Smagorinsky 渦黏修正 (大渦模擬)
            # ------------------------------------------------------------
            #  ★2026-06-29 修正 LES 範數（見 docs/LBM_SOLVER_PHYSICS_AUDIT_2026-06-24.md §2.1-2.3）
            #
            #  由非平衡動量通量張量回復濾波應變率（Yu, Girimaji & Luo 2005；Krafczyk 2003）。
            #  Lallemand–Luo D2Q9 基底中（M row7=[0,1,-1,1,-1,0,0,0,0]、m_eq[7]=rho(u²-v²)）：
            #     m[7] = Π_xx − Π_yy （正應力「差」，非 Π_xx 本身）⇒ Π_xx^neq = m7^neq/2、Π_yy^neq = −m7^neq/2
            #     m[8] = Π_xy                                      ⇒ Π_xy^neq = m8^neq
            #     Π:Π = Π_xx² + Π_yy² + 2·Π_xy² = m7²/2 + 2·m8²
            #  應變強度所需範數 Q = sqrt(2·Π:Π) = sqrt(m7² + 4·m8²)（√2 已摺入 Q）。
            #  標準閉式（c_s²=1/3, Δ=1）：tau_eddy = 0.5·[sqrt(tau_0² + 18·Cs²·Q/rho) − tau_0]，
            #     Cs_sq_factor = 18·Cs² ⇒ term_inside = tau_0² + Cs_sq_factor·Q/rho（見下，term_inside 不變）。
            #  舊式誤把 m7 當 Π_xx（norm=sqrt(2m7²+2m8²)）→ 正應變區 ν_t 高估 **√2×**、
            #  剪切區低估 √2×（等混合 1.118×）。★倍率更正 2026-07-29：本行原寫「高估 2×」，
            #  commit 0e2e72f 的 message 同誤；`tests/test_les_strain_norm.py:89` 自己印出的
            #  1.414 才是對的。
            # ============================================================
            neq_7 = m[7] - m_eq[7]  # (Π_xx − Π_yy) 非平衡分量
            neq_8 = m[8] - m_eq[8]  # Π_xy 非平衡分量

            # Q = sqrt(2·Π:Π) = sqrt(neq_7² + 4·neq_8²)
            neq_tensor_norm = tm.sqrt(neq_7 * neq_7 + 4.0 * neq_8 * neq_8)

            tau_eff = self.tau_0
            if self.C_smag > 0.001:
                # Smagorinsky 動態 tau 修正：
                #   tau_eddy = 0.5 · [sqrt(tau_0² + Cs_sq_factor · |Π^neq|_F / rho) - tau_0]
                #   tau_eff  = tau_0 + tau_eddy
                # 其中 Cs_sq_factor = 18 · Cs²（見 _init_params：self.Cs_sq_factor = 18.0 * C_smag**2）
                term_inside = self.tau_0**2 + (
                    self.Cs_sq_factor * neq_tensor_norm
                ) / tm.max(
                    rho_l, 1e-6
                )  # rho floor 防 rho_l<=0（發散徵兆）時 LES 項 div-by-zero→NaN（audit C2）
                tau_eddy = 0.5 * (tm.sqrt(term_inside) - self.tau_0)
                tau_eff = self.tau_0 + tau_eddy

            # ============================================================
            # 6. Omni-directional Sponge Layer (全方位阻尼層)
            # ------------------------------------------------------------
            # 在邊界緩衝區內，額外增大有效 tau 來抑制波動反射。
            # 阻尼強度以二次函數從邊界向內遞減：
            #   coord ∈ [0, 1]，0 = 場域邊緣，1 = 阻尼層內緣
            #   damping = sponge_strength · coord²
            # X 與 Y 方向阻尼取最大值疊加到 tau_eff
            # ============================================================
            # A. X 方向阻尼 (Outlet & Inlet)
            # 四個邊界各自獨立
            damping_x = 0.0
            if i > (self.nx - self.sponge_w_out):
                coord = (i - (self.nx - self.sponge_w_out)) / self.sponge_w_out
                damping_x = self.sponge_strength * (coord * coord)
            elif i < self.sponge_w_in:
                coord = (self.sponge_w_in - i) / tm.max(self.sponge_w_in, 1)
                damping_x = self.sponge_strength * (coord * coord)

            damping_y = 0.0
            if j < self.sponge_w_bot:
                coord = (self.sponge_w_bot - j) / self.sponge_w_bot
                damping_y = self.sponge_strength * (coord * coord)
            elif j > (self.ny - self.sponge_w_top):
                coord = (j - (self.ny - self.sponge_w_top)) / self.sponge_w_top
                damping_y = self.sponge_strength * (coord * coord)

            damping_max = tm.max(damping_x, damping_y)
            tau_eff += damping_max

            # ============================================================
            # 6b. Sponge Macro Relaxation (Phase 4B)
            # ------------------------------------------------------------
            # 只在 outlet sponge 區域內將巨觀量混合到參考值。
            # 入口與側向 sponge 僅用 tau 增大（避免壓低入口驅動速度）。
            #
            # ★2026-07-29 現況記錄：本區塊由 `sponge_relax_strength > 0.001`
            #   守門，而**全部** config（生產、archive、所有 bench 腳本）都明寫
            #   `0.0` → 此路徑至今從未執行過。保留不刪，因為它是完整實作的
            #   可選機制；但任何「sponge 會把巨觀量鬆弛到參考值」的敘述，
            #   在現行設定下都**不成立**——實際生效的只有上方 6a 的 tau 增大。
            # ============================================================
            damping_out = 0.0
            if i > (self.nx - self.sponge_w_out):
                coord_out = (i - (self.nx - self.sponge_w_out)) / self.sponge_w_out
                damping_out = self.sponge_strength * (coord_out * coord_out)

            if self.sponge_relax_strength > 0.001 and damping_out > 0.001:
                damping_ratio = damping_out / self.sponge_strength
                alpha = (
                    self.sponge_relax_strength * damping_ratio / (damping_ratio + 1.0)
                )
                u_ref = self.u_inlet_est * self.sponge_ref_u_factor
                v_ref = self.sponge_ref_v
                rho_ref = self.rho_out_target
                u_l = u_l * (1.0 - alpha) + u_ref * alpha
                v_l = v_l * (1.0 - alpha) + v_ref * alpha
                rho_l = rho_l * (1.0 - alpha) + rho_ref * alpha
                m_eq = self.get_meq(rho_l, u_l, v_l)

            # ============================================================
            # 7. MRT Relaxation (矩空間鬆弛運算)
            # ------------------------------------------------------------
            # m* = m - S · (m - m_eq)
            # S 為對角鬆弛矩陣：
            #   S[0] = 0     (rho，守恆量，不鬆弛)
            #   S[1] = S_other (能量 e)
            #   S[2] = S_other (能量平方 eps)
            #   S[3] = 0     (jx，守恆量)
            #   S[4] = S_other (熱通流 qx)
            #   S[5] = 0     (jy，守恆量)
            #   S[6] = S_other (熱通流 qy)
            #   S[7] = s_eff  ← 由 LES + Sponge 動態決定的剪切鬆弛率
            #   S[8] = s_eff  ← 同上
            # s_eff = 1 / tau_eff
            # ============================================================
            s_eff = 1.0 / tau_eff
            S_local = self.S_base

            # S[7], S[8] 控制剪切黏滯 (pxx, pxy)，使用動態 s_eff
            S_local[7] = s_eff
            S_local[8] = s_eff

            m_star = m - S_local * (m - m_eq)

            # ============================================================
            # 8. Inverse MRT (逆轉換：矩空間 → 速度空間)
            # ------------------------------------------------------------
            # f_new = M⁻¹ · m*
            # 得到碰撞後的分布函數，寫入 f_new 供下一步使用
            # ============================================================
            f_new_val = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.invM_field[r, c] * m_star[c]
                f_new_val[r] = val

            self.f_new[i, j] = f_new_val

    @ti.kernel
    def update_macro_var(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            local_rho = 0.0
            local_vel = tm.vec2(0.0, 0.0)
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                local_rho += self.f_new[i, j][k]
                local_vel += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            self.rho[i, j] = local_rho
            if local_rho > 0:
                self.vel[i, j] = local_vel / local_rho
            else:
                self.vel[i, j] = tm.vec2(0, 0)

    @ti.kernel
    def apply_bc(self):
        self.frame_count[None] += 1
        # 緩啟動 (Smoother Cosine Ramp)；warmup_steps=0 時 ramp=1（顯式分支，
        # 不依賴除零→inf 行為，audit C7）。
        ramp = 1.0
        if self.warmup_steps > 0:
            progress = tm.min(1.0, float(self.frame_count[None]) / self.warmup_steps)
            ramp = 1.0 - tm.cos(0.5 * 3.14159265 * progress)

        for j in range(1, self.ny - 1):
            self.apply_bc_core(1, 0, 0, j, 1, j, ramp)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j, ramp)
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2, ramp)
            self.apply_bc_core(1, 3, i, 0, i, 1, ramp)

        # Phase 4A: HWBB 在 streaming 時已處理壁面反射，
        # 不再重置固體節點的分布函數為平衡態。
        # 僅設定 vel=0 供輸出/視覺化使用。
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                self.vel[i, j] = 0.0, 0.0

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb, ramp: float):
        if outer == 1:
            if self.bc_type[dr] == 0:  # Velocity Inlet (Zou-He)
                if ibc == 0:
                    # Zou-He Velocity Boundary at West
                    # 已知 ux（prescribed），推導 rho
                    ux = self.u_inlet_target * ramp
                    uy = 0.0

                    f0 = self.f_old[inb, jnb][0]
                    f2 = self.f_old[inb, jnb][2]
                    f3 = self.f_old[inb, jnb][3]
                    f4 = self.f_old[inb, jnb][4]
                    f6 = self.f_old[inb, jnb][6]
                    f7 = self.f_old[inb, jnb][7]

                    rho_local = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)

                    f1 = f3 + (2.0 / 3.0) * rho_local * ux
                    f5 = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_local * ux
                    f8 = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_local * ux

                    self.rho[ibc, jbc] = rho_local
                    self.vel[ibc, jbc] = tm.vec2(ux, uy)

                    self.f_old[ibc, jbc] = self.f_eq(ibc, jbc)
                    self.f_old[ibc, jbc][1] = f1
                    self.f_old[ibc, jbc][5] = f5
                    self.f_old[ibc, jbc][8] = f8
                else:
                    self.vel[ibc, jbc] = self.bc_value[dr] * ramp
                    self.rho[ibc, jbc] = self.rho[inb, jnb]
                    self.f_old[ibc, jbc] = (
                        self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
                    )

            elif self.bc_type[dr] == 1:  # Outlet (Zou-He Pressure)
                if ibc == self.nx - 1:
                    rho_out = self.rho_out_target
                    f0 = self.f_old[inb, jnb][0]
                    f1 = self.f_old[inb, jnb][1]
                    f2 = self.f_old[inb, jnb][2]
                    f4 = self.f_old[inb, jnb][4]
                    f5 = self.f_old[inb, jnb][5]
                    f8 = self.f_old[inb, jnb][8]

                    ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out
                    uy = 0.0

                    # [FIX] Backflow 防護
                    if ux < 0.0:
                        # 改用零梯度外推，防止非物理分布函數積累
                        self.vel[ibc, jbc] = self.vel[inb, jnb]
                        self.rho[ibc, jbc] = rho_out
                        self.f_old[ibc, jbc] = (
                            self.f_eq(ibc, jbc)
                            - self.f_eq(inb, jnb)
                            + self.f_old[inb, jnb]
                        )
                    else:
                        f3 = f1 - (2.0 / 3.0) * rho_out * ux
                        f6 = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
                        f7 = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux

                        self.rho[ibc, jbc] = rho_out
                        self.vel[ibc, jbc] = tm.vec2(ux, uy)
                        self.f_old[ibc, jbc] = self.f_eq(ibc, jbc)
                        self.f_old[ibc, jbc][3] = f3
                        self.f_old[ibc, jbc][6] = f6
                        self.f_old[ibc, jbc][7] = f7

            elif self.bc_type[dr] == 2:  # Free-Slip (Symmetry / Specular)
                # 邏輯：切向速度照抄鄰居 (Free)，法向速度設為 0 (Wall)

                # 判斷牆壁方向：
                # 如果 Ghost Node 和 Neighbor Node 的 X 座標相同 (ibc == inb)，代表是上下牆壁
                if ibc == inb:
                    self.vel[ibc, jbc][0] = self.vel[inb, jnb][
                        0
                    ]  # u_x (切向): 保留流體速度
                    self.vel[ibc, jbc][1] = 0.0  # u_y (法向): 撞牆歸零

                # 否則就是左右牆壁 (jbc == jnb)
                else:
                    self.vel[ibc, jbc][0] = 0.0  # u_x (法向): 撞牆歸零
                    self.vel[ibc, jbc][1] = self.vel[inb, jnb][
                        1
                    ]  # u_y (切向): 保留流體速度

                self.rho[ibc, jbc] = self.rho[inb, jnb]
                self.f_old[ibc, jbc] = (
                    self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
                )

            elif self.bc_type[dr] == 3:  # Convective Outflow（非反射出口）
                # ★2026-07-29 新增（P2-D 必修項）。動機：type=1 的 Zou-He 固定壓力
                #   出口 `rho_out = const` 對聲波而言是一面硬牆。實測反射係數
                #   （高斯密度脈衝、生產尺寸 nx=2560、sponge_out=240、strength=3.0）：
                #       λ≈100 px → R = 0.626      λ≈400 px（尾流尺度）→ R = 0.943
                #   sponge 只把回波壓掉 2–14%。這正是「密度場帶 8.7% 域尺度偏移且
                #   600 CTU 後仍不衰減」的根因（兩份獨立複核從不同角度撞到同一處）。
                #
                # 作法：對流出口條件 ∂φ/∂t + U_c ∂φ/∂x = 0，以一階迎風離散
                #       φ_ghost^{n+1} = φ_ghost^n − C (φ_ghost^n − φ_inner^n)
                #   其中 C = U_c·Δt/Δx = U_c（格子單位）。擾動被「對流出去」而不是
                #   被反射回來。C ∈ (0,1] 才穩定（CFL）；U_c 取 u_inlet_target
                #   （均勻來流的相速度尺度），生產值 0.05 遠小於 1，穩定裕度充足。
                #
                # ★只作用在**東側**（ibc == nx-1）；其餘方向落到零梯度外推，
                #   與 type=1 的 backflow 分支同一手法，不引入新語義。
                if ibc == self.nx - 1:
                    cfl = self.outlet_conv_speed
                    for k in ti.static(range(9)):
                        self.f_old[ibc, jbc][k] = self.f_old[ibc, jbc][k] - cfl * (
                            self.f_old[ibc, jbc][k] - self.f_old[inb, jnb][k]
                        )
                    # 巨觀量由更新後的 f 重算，保持 f 與 (rho, vel) 自洽
                    rho_c = 0.0
                    mx_c = 0.0
                    my_c = 0.0
                    for k in ti.static(range(9)):
                        fk = self.f_old[ibc, jbc][k]
                        rho_c += fk
                        mx_c += fk * self.e[k, 0]
                        my_c += fk * self.e[k, 1]
                    self.rho[ibc, jbc] = rho_c
                    self.vel[ibc, jbc] = tm.vec2(mx_c / rho_c, my_c / rho_c)
                else:
                    self.vel[ibc, jbc] = self.vel[inb, jnb]
                    self.rho[ibc, jbc] = self.rho[inb, jnb]
                    self.f_old[ibc, jbc] = (
                        self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
                    )

    def attach_time_mean_accumulator(self, accumulator) -> None:
        """掛載求解迴圈內時均累加器（DEEP_AUDIT_2026-07 Stage 1.15）。

        掛載後 `run_step` / `run_step_batch` 每推進一個 LBM 步就呼叫一次
        `accumulator.accumulate()`，在 GPU 上以 float64 累加 MRT 矩。

        Args:
            accumulator: :class:`core.time_mean_accumulator.TimeMeanAccumulator`
                實例，或 None（解除掛載）。
        """
        self._time_mean_accum = accumulator

    def run_step(self, steps=1):
        """執行 LBM 的物理時間步推進。

        每步循環：collide_and_stream → update_macro_var → apply_bc
        （→ 時均累加，若已掛載累加器）。
        多步在 GPU 端連續執行，不中斷回 CPU，減少 sync 開銷。
        """
        if self.pingpong:
            for _ in range(steps):
                # 這一步是否有人要讀矩？只有時均累加器會在迴圈內要，
                # 且它自己有 stride——問它，不要每步都寫。
                acc = self._time_mean_accum
                need_m = acc is not None and acc.will_accumulate_next()
                if self._pp_parity == 0:
                    self._collide_stream_pp(self.f_old, self.f_new, need_m)
                    self._apply_bc_pp(self.f_new)
                else:
                    self._collide_stream_pp(self.f_new, self.f_old, need_m)
                    self._apply_bc_pp(self.f_old)
                self._pp_parity ^= 1
                if acc is not None:
                    acc.accumulate()
            return
        for _ in range(steps):
            self.collide_and_stream()
            self.update_macro_var()
            self.apply_bc()
            if self._time_mean_accum is not None:
                self._time_mean_accum.accumulate()

    def run_step_batch(self, steps: int, sync_interval: int = 0):
        """批次執行多步，減少 GPU-CPU 同步次數。

        與 run_step 物理結果相同，但在 sync_interval 步內不回到 Python。
        若 sync_interval=0（預設），等同 run_step(steps)。

        Args:
            steps: 總步數。
            sync_interval: 每多少步做一次 ti.sync()。0 = 不額外同步。
        """
        if sync_interval <= 0:
            for _ in range(steps):
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()
                if self._time_mean_accum is not None:
                    self._time_mean_accum.accumulate()
            return

        done = 0
        while done < steps:
            batch = min(sync_interval, steps - done)
            for _ in range(batch):
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()
                if self._time_mean_accum is not None:
                    self._time_mean_accum.accumulate()
            ti.sync()
            done += batch

    # endregion
    # ------------------------------------------------

    # ------------------------------------------------
    # region DFG 驗證專用函式
    @ti.func
    def get_parabolic_inlet_velocity(self, j: int, u_max: float, ny: int) -> float:
        # y 座標 (0 到 ny-1)
        y = float(j)
        h = float(ny - 1)
        # 拋物線公式
        return 4.0 * u_max * y * (h - y) / (h * h)

    @ti.kernel
    def compute_force_on_obstacle(self):
        # 1. 歸零累加器
        self.force_sum[None] = tm.vec2(0.0, 0.0)

        # [終極解法] 整合查表法 (Lookup Table)
        # 我們將所有需要的資訊打包在一起，只用 k (靜態整數) 來索引
        # 格式: (鄰居方向X, 鄰居方向Y, 反向索引ID, 受力方向X, 受力方向Y)
        # 受力方向其實就是鄰居方向的相反 (-x, -y)
        lookup_data = (
            (0, 0, 0, 0, 0),  # k=0
            (1, 0, 3, -1, 0),  # k=1 (E)  -> inv=3 (W)  -> Force=(-1, 0)
            (0, 1, 4, 0, -1),  # k=2 (N)  -> inv=4 (S)  -> Force=(0, -1)
            (-1, 0, 1, 1, 0),  # k=3 (W)  -> inv=1 (E)  -> Force=(1, 0)
            (0, -1, 2, 0, 1),  # k=4 (S)  -> inv=2 (N)  -> Force=(0, 1)
            (1, 1, 7, -1, -1),  # k=5 (NE) -> inv=7 (SW) -> Force=(-1, -1)
            (-1, 1, 8, 1, -1),  # k=6 (NW) -> inv=8 (SE) -> Force=(1, -1)
            (-1, -1, 5, 1, 1),  # k=7 (SW) -> inv=5 (NE) -> Force=(1, 1)
            (1, -1, 6, -1, 1),  # k=8 (SE) -> inv=6 (NW) -> Force=(-1, 1)
        )

        # 遍歷整個場
        for i, j in self.mask:
            if self.mask[i, j] == 1:  # 固體

                # 使用 ti.static 展開，確保 k 是純 Python 整數
                for k in ti.static(range(9)):

                    # [關鍵] 因為 k 是 static，這裡是在編譯期直接解包 Python Tuple
                    # 完全不涉及 Taichi 運行時計算，絕對不會報錯
                    vals = lookup_data[k]

                    dir_x = vals[0]
                    dir_y = vals[1]
                    inv_k = vals[2]
                    force_x = vals[3]
                    force_y = vals[4]

                    # 尋找鄰居
                    ni = i + dir_x
                    nj = j + dir_y

                    # 邊界與流體檢查
                    if ni >= 0 and ni < self.nx and nj >= 0 and nj < self.ny:
                        if self.mask[ni, nj] == 0:

                            # 讀取分布函數 (inv_k 這裡是整數，Taichi Field 接受整數或 Expr 索引，所以沒問題)
                            f_val = self.f_new[ni, nj][inv_k]

                            # 計算受力
                            force_vec = tm.vec2(force_x, force_y)

                            # 累加 (動量交換公式: 2 * f_in * c)
                            self.force_sum[None] += 2.0 * f_val * force_vec

    # Python 端呼叫用的 Helper
    def get_force(self):
        self.compute_force_on_obstacle()
        return self.force_sum[None].to_numpy()

    @ti.kernel
    def _get_max_velocity_kernel(self) -> ti.f32:
        max_v = 0.0
        for i, j in self.vel:
            v_mag = self.vel[i, j].norm()
            ti.atomic_max(max_v, v_mag)
        return max_v

    def get_max_velocity(self):
        """
        獲取全場最大流速 (LBM 單位)
        """
        return self._get_max_velocity_kernel()

    # endregion
    # ------------------------------------------------

    # ══════════════════════════════════════════════════════════════════
    # ping-pong 路徑（2026-07-30 效能優化）
    # ------------------------------------------------------------------
    # 原路徑每步跑三個 kernel：collide_and_stream → update_macro_var → apply_bc。
    # 其中 update_macro_var 做的是**整場 9 通道複製** `f_old <- f_new`：
    # 2560×1440×9×4 B = 132 MB 讀 + 132 MB 寫 = 265 MB/步的多餘流量，
    # 讓總記憶體流量整整翻倍。實測 RTX 5080 只用到 237 GB/s（理論約 960），
    # 即 25% 頻寬，這個複製是主因。
    #
    # ping-pong 是 LBM 的標準解法：兩個 f 場交替當 src/dst，完全不複製。
    # rho/vel 改在 collide 內直接寫入（rho 與 j 是碰撞守恆矩，post-collision
    # 與 pre-collision 相同，用 collide 內已算好的值即可，無需再讀一次整場 f）。
    #
    # ★原路徑完整保留，`pingpong: false` 可切回，供逐項對照驗證。
    # ══════════════════════════════════════════════════════════════════

    @ti.kernel
    def _collide_stream_pp(
        self, src: ti.template(), dst: ti.template(), write_moments: ti.template()
    ):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):

            # 固體節點不參與碰撞與串流，保持 f 不變
            if self.mask[i, j] == 1.0:
                dst[i, j] = src[i, j]
                continue

            # ============================================================
            # 1. Streaming + Half-way Bounce-Back (Phase 4A)
            # ------------------------------------------------------------
            # Pull-scheme: 從鄰居節點拉取分布函數
            # HWBB: 若鄰居為固體，取自身反方向的值（壁面在半格位置，二階精度）
            # D2Q9 opposite: 0→0, 1↔3, 2↔4, 5↔7, 6↔8
            # ============================================================
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                if self.mask[ip, jp] == 1.0:
                    # HWBB: 粒子在半格壁面反彈，取自身反方向
                    k_opp = ti.static([0, 3, 4, 1, 2, 7, 8, 5, 6][k])
                    f_temp[k] = src[i, j][k_opp]
                else:
                    f_temp[k] = src[ip, jp][k]

            # ============================================================
            # 2. MRT Moment Transformation (矩陣轉換：速度空間 → 矩空間)
            # ------------------------------------------------------------
            # m = M · f
            # M 為 9×9 的 D2Q9 MRT 轉換矩陣 (Lallemand & Luo, 2000)
            # 矩空間順序：[rho, e, eps, jx, qx, jy, qy, pxx, pxy]
            # ============================================================
            m = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.M_field[r, c] * f_temp[c]
                m[r] = val

            # ★N1：碰撞前的矩即為此處的 m（由 post-streaming 的 f_temp 投影）。
            #   ★2026-07-30：改為編譯期開關。每步寫整場 9 通道要 132 MB/步
            #   （2560×1440×9×4 B），佔 ping-pong 後總流量的四成，但實際只有
            #   snapshot 與 mean_accum 累加的那些步用得到。
            #   不寫的步由 `_recompute_moments_pre` 在需要時重建。
            if ti.static(write_moments):
                self.moments_pre_field[i, j] = m

            # ============================================================
            # 3. Macroscopic Variables (計算巨觀量：密度與速度)
            # ------------------------------------------------------------
            # 守恆量直接從矩空間讀取：
            #   rho = m[0] = Σ f_k
            #   j_x = m[3] = Σ e_kx · f_k  →  u = jx / rho
            #   j_y = m[5] = Σ e_ky · f_k  →  v = jy / rho
            # ============================================================
            rho_l = m[0]
            u_l, v_l = 0.0, 0.0
            if rho_l > 0:
                u_l, v_l = m[3] / rho_l, m[5] / rho_l

            # ============================================================
            # 4. Equilibrium Moments (計算平衡矩)
            # ------------------------------------------------------------
            # m_eq 的完整定義 (見 get_meq)：
            #   m_eq[0] = rho
            #   m_eq[1] = rho(-2 + 3u²)      [能量 e]
            #   m_eq[2] = rho( 1 - 3u²)      [能量平方 eps]
            #   m_eq[3] = rho·u              [動量 jx]
            #   m_eq[4] = -rho·u             [熱通流 qx]
            #   m_eq[5] = rho·v              [動量 jy]
            #   m_eq[6] = -rho·v             [熱通流 qy]
            #   m_eq[7] = rho(u² - v²)       [正應力差 pxx]
            #   m_eq[8] = rho·u·v            [切應力 pxy]
            # ============================================================
            m_eq = self.get_meq(rho_l, u_l, v_l)

            # ============================================================
            # 5. LES Smagorinsky 渦黏修正 (大渦模擬)
            # ------------------------------------------------------------
            #  ★2026-06-29 修正 LES 範數（見 docs/LBM_SOLVER_PHYSICS_AUDIT_2026-06-24.md §2.1-2.3）
            #
            #  由非平衡動量通量張量回復濾波應變率（Yu, Girimaji & Luo 2005；Krafczyk 2003）。
            #  Lallemand–Luo D2Q9 基底中（M row7=[0,1,-1,1,-1,0,0,0,0]、m_eq[7]=rho(u²-v²)）：
            #     m[7] = Π_xx − Π_yy （正應力「差」，非 Π_xx 本身）⇒ Π_xx^neq = m7^neq/2、Π_yy^neq = −m7^neq/2
            #     m[8] = Π_xy                                      ⇒ Π_xy^neq = m8^neq
            #     Π:Π = Π_xx² + Π_yy² + 2·Π_xy² = m7²/2 + 2·m8²
            #  應變強度所需範數 Q = sqrt(2·Π:Π) = sqrt(m7² + 4·m8²)（√2 已摺入 Q）。
            #  標準閉式（c_s²=1/3, Δ=1）：tau_eddy = 0.5·[sqrt(tau_0² + 18·Cs²·Q/rho) − tau_0]，
            #     Cs_sq_factor = 18·Cs² ⇒ term_inside = tau_0² + Cs_sq_factor·Q/rho（見下，term_inside 不變）。
            #  舊式誤把 m7 當 Π_xx（norm=sqrt(2m7²+2m8²)）→ 正應變區 ν_t 高估 **√2×**、
            #  剪切區低估 √2×（等混合 1.118×）。★倍率更正 2026-07-29：本行原寫「高估 2×」，
            #  commit 0e2e72f 的 message 同誤；`tests/test_les_strain_norm.py:89` 自己印出的
            #  1.414 才是對的。
            # ============================================================
            neq_7 = m[7] - m_eq[7]  # (Π_xx − Π_yy) 非平衡分量
            neq_8 = m[8] - m_eq[8]  # Π_xy 非平衡分量

            # Q = sqrt(2·Π:Π) = sqrt(neq_7² + 4·neq_8²)
            neq_tensor_norm = tm.sqrt(neq_7 * neq_7 + 4.0 * neq_8 * neq_8)

            tau_eff = self.tau_0
            if self.C_smag > 0.001:
                # Smagorinsky 動態 tau 修正：
                #   tau_eddy = 0.5 · [sqrt(tau_0² + Cs_sq_factor · |Π^neq|_F / rho) - tau_0]
                #   tau_eff  = tau_0 + tau_eddy
                # 其中 Cs_sq_factor = 18 · Cs²（見 _init_params：self.Cs_sq_factor = 18.0 * C_smag**2）
                term_inside = self.tau_0**2 + (
                    self.Cs_sq_factor * neq_tensor_norm
                ) / tm.max(
                    rho_l, 1e-6
                )  # rho floor 防 rho_l<=0（發散徵兆）時 LES 項 div-by-zero→NaN（audit C2）
                tau_eddy = 0.5 * (tm.sqrt(term_inside) - self.tau_0)
                tau_eff = self.tau_0 + tau_eddy

            # ============================================================
            # 6. Omni-directional Sponge Layer (全方位阻尼層)
            # ------------------------------------------------------------
            # 在邊界緩衝區內，額外增大有效 tau 來抑制波動反射。
            # 阻尼強度以二次函數從邊界向內遞減：
            #   coord ∈ [0, 1]，0 = 場域邊緣，1 = 阻尼層內緣
            #   damping = sponge_strength · coord²
            # X 與 Y 方向阻尼取最大值疊加到 tau_eff
            # ============================================================
            # A. X 方向阻尼 (Outlet & Inlet)
            # 四個邊界各自獨立
            damping_x = 0.0
            if i > (self.nx - self.sponge_w_out):
                coord = (i - (self.nx - self.sponge_w_out)) / self.sponge_w_out
                damping_x = self.sponge_strength * (coord * coord)
            elif i < self.sponge_w_in:
                coord = (self.sponge_w_in - i) / tm.max(self.sponge_w_in, 1)
                damping_x = self.sponge_strength * (coord * coord)

            damping_y = 0.0
            if j < self.sponge_w_bot:
                coord = (self.sponge_w_bot - j) / self.sponge_w_bot
                damping_y = self.sponge_strength * (coord * coord)
            elif j > (self.ny - self.sponge_w_top):
                coord = (j - (self.ny - self.sponge_w_top)) / self.sponge_w_top
                damping_y = self.sponge_strength * (coord * coord)

            damping_max = tm.max(damping_x, damping_y)
            tau_eff += damping_max

            # ============================================================
            # 6b. Sponge Macro Relaxation (Phase 4B)
            # ------------------------------------------------------------
            # 只在 outlet sponge 區域內將巨觀量混合到參考值。
            # 入口與側向 sponge 僅用 tau 增大（避免壓低入口驅動速度）。
            #
            # ★2026-07-29 現況記錄：本區塊由 `sponge_relax_strength > 0.001`
            #   守門，而**全部** config（生產、archive、所有 bench 腳本）都明寫
            #   `0.0` → 此路徑至今從未執行過。保留不刪，因為它是完整實作的
            #   可選機制；但任何「sponge 會把巨觀量鬆弛到參考值」的敘述，
            #   在現行設定下都**不成立**——實際生效的只有上方 6a 的 tau 增大。
            # ============================================================
            damping_out = 0.0
            if i > (self.nx - self.sponge_w_out):
                coord_out = (i - (self.nx - self.sponge_w_out)) / self.sponge_w_out
                damping_out = self.sponge_strength * (coord_out * coord_out)

            if self.sponge_relax_strength > 0.001 and damping_out > 0.001:
                damping_ratio = damping_out / self.sponge_strength
                alpha = (
                    self.sponge_relax_strength * damping_ratio / (damping_ratio + 1.0)
                )
                u_ref = self.u_inlet_est * self.sponge_ref_u_factor
                v_ref = self.sponge_ref_v
                rho_ref = self.rho_out_target
                u_l = u_l * (1.0 - alpha) + u_ref * alpha
                v_l = v_l * (1.0 - alpha) + v_ref * alpha
                rho_l = rho_l * (1.0 - alpha) + rho_ref * alpha
                m_eq = self.get_meq(rho_l, u_l, v_l)

            # ============================================================
            # 7. MRT Relaxation (矩空間鬆弛運算)
            # ------------------------------------------------------------
            # m* = m - S · (m - m_eq)
            # S 為對角鬆弛矩陣：
            #   S[0] = 0     (rho，守恆量，不鬆弛)
            #   S[1] = S_other (能量 e)
            #   S[2] = S_other (能量平方 eps)
            #   S[3] = 0     (jx，守恆量)
            #   S[4] = S_other (熱通流 qx)
            #   S[5] = 0     (jy，守恆量)
            #   S[6] = S_other (熱通流 qy)
            #   S[7] = s_eff  ← 由 LES + Sponge 動態決定的剪切鬆弛率
            #   S[8] = s_eff  ← 同上
            # s_eff = 1 / tau_eff
            # ============================================================
            s_eff = 1.0 / tau_eff
            S_local = self.S_base

            # S[7], S[8] 控制剪切黏滯 (pxx, pxy)，使用動態 s_eff
            S_local[7] = s_eff
            S_local[8] = s_eff

            m_star = m - S_local * (m - m_eq)

            # ============================================================
            # 8. Inverse MRT (逆轉換：矩空間 → 速度空間)
            # ------------------------------------------------------------
            # f_new = M⁻¹ · m*
            # 得到碰撞後的分布函數，寫入 f_new 供下一步使用
            # ============================================================
            f_new_val = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.invM_field[r, c] * m_star[c]
                f_new_val[r] = val

            dst[i, j] = f_new_val

            # ★ping-pong：rho/vel 直接在此寫入，取代 update_macro_var。
            #   rho 與 j 是碰撞守恆矩，post-collision 與 pre-collision 相同，
            #   故可直接用上面算好的 rho_l/u_l/v_l——省掉整場 f 的再讀一次。
            self.rho[i, j] = rho_l
            self.vel[i, j] = tm.vec2(u_l, v_l)

    @ti.func
    def _apply_bc_core_pp(
        self, fld: ti.template(), outer, dr, ibc, jbc, inb, jnb, ramp: float
    ):
        if outer == 1:
            if self.bc_type[dr] == 0:  # Velocity Inlet (Zou-He)
                if ibc == 0:
                    # Zou-He Velocity Boundary at West
                    # 已知 ux（prescribed），推導 rho
                    ux = self.u_inlet_target * ramp
                    uy = 0.0

                    f0 = fld[inb, jnb][0]
                    f2 = fld[inb, jnb][2]
                    f3 = fld[inb, jnb][3]
                    f4 = fld[inb, jnb][4]
                    f6 = fld[inb, jnb][6]
                    f7 = fld[inb, jnb][7]

                    rho_local = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux)

                    f1 = f3 + (2.0 / 3.0) * rho_local * ux
                    f5 = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_local * ux
                    f8 = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho_local * ux

                    self.rho[ibc, jbc] = rho_local
                    self.vel[ibc, jbc] = tm.vec2(ux, uy)

                    fld[ibc, jbc] = self.f_eq(ibc, jbc)
                    fld[ibc, jbc][1] = f1
                    fld[ibc, jbc][5] = f5
                    fld[ibc, jbc][8] = f8
                else:
                    self.vel[ibc, jbc] = self.bc_value[dr] * ramp
                    self.rho[ibc, jbc] = self.rho[inb, jnb]
                    fld[ibc, jbc] = (
                        self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + fld[inb, jnb]
                    )

            elif self.bc_type[dr] == 1:  # Outlet (Zou-He Pressure)
                if ibc == self.nx - 1:
                    rho_out = self.rho_out_target
                    f0 = fld[inb, jnb][0]
                    f1 = fld[inb, jnb][1]
                    f2 = fld[inb, jnb][2]
                    f4 = fld[inb, jnb][4]
                    f5 = fld[inb, jnb][5]
                    f8 = fld[inb, jnb][8]

                    ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_out
                    uy = 0.0

                    # [FIX] Backflow 防護
                    if ux < 0.0:
                        # 改用零梯度外推，防止非物理分布函數積累
                        self.vel[ibc, jbc] = self.vel[inb, jnb]
                        self.rho[ibc, jbc] = rho_out
                        fld[ibc, jbc] = (
                            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + fld[inb, jnb]
                        )
                    else:
                        f3 = f1 - (2.0 / 3.0) * rho_out * ux
                        f6 = f8 - 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux
                        f7 = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho_out * ux

                        self.rho[ibc, jbc] = rho_out
                        self.vel[ibc, jbc] = tm.vec2(ux, uy)
                        fld[ibc, jbc] = self.f_eq(ibc, jbc)
                        fld[ibc, jbc][3] = f3
                        fld[ibc, jbc][6] = f6
                        fld[ibc, jbc][7] = f7

            elif self.bc_type[dr] == 2:  # Free-Slip (Symmetry / Specular)
                # 邏輯：切向速度照抄鄰居 (Free)，法向速度設為 0 (Wall)

                # 判斷牆壁方向：
                # 如果 Ghost Node 和 Neighbor Node 的 X 座標相同 (ibc == inb)，代表是上下牆壁
                if ibc == inb:
                    self.vel[ibc, jbc][0] = self.vel[inb, jnb][
                        0
                    ]  # u_x (切向): 保留流體速度
                    self.vel[ibc, jbc][1] = 0.0  # u_y (法向): 撞牆歸零

                # 否則就是左右牆壁 (jbc == jnb)
                else:
                    self.vel[ibc, jbc][0] = 0.0  # u_x (法向): 撞牆歸零
                    self.vel[ibc, jbc][1] = self.vel[inb, jnb][
                        1
                    ]  # u_y (切向): 保留流體速度

                self.rho[ibc, jbc] = self.rho[inb, jnb]
                fld[ibc, jbc] = (
                    self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + fld[inb, jnb]
                )

            elif self.bc_type[dr] == 3:  # Convective Outflow（非反射出口）
                # ★2026-07-29 新增（P2-D 必修項）。動機：type=1 的 Zou-He 固定壓力
                #   出口 `rho_out = const` 對聲波而言是一面硬牆。實測反射係數
                #   （高斯密度脈衝、生產尺寸 nx=2560、sponge_out=240、strength=3.0）：
                #       λ≈100 px → R = 0.626      λ≈400 px（尾流尺度）→ R = 0.943
                #   sponge 只把回波壓掉 2–14%。這正是「密度場帶 8.7% 域尺度偏移且
                #   600 CTU 後仍不衰減」的根因（兩份獨立複核從不同角度撞到同一處）。
                #
                # 作法：對流出口條件 ∂φ/∂t + U_c ∂φ/∂x = 0，以一階迎風離散
                #       φ_ghost^{n+1} = φ_ghost^n − C (φ_ghost^n − φ_inner^n)
                #   其中 C = U_c·Δt/Δx = U_c（格子單位）。擾動被「對流出去」而不是
                #   被反射回來。C ∈ (0,1] 才穩定（CFL）；U_c 取 u_inlet_target
                #   （均勻來流的相速度尺度），生產值 0.05 遠小於 1，穩定裕度充足。
                #
                # ★只作用在**東側**（ibc == nx-1）；其餘方向落到零梯度外推，
                #   與 type=1 的 backflow 分支同一手法，不引入新語義。
                if ibc == self.nx - 1:
                    cfl = self.outlet_conv_speed
                    for k in ti.static(range(9)):
                        fld[ibc, jbc][k] = fld[ibc, jbc][k] - cfl * (
                            fld[ibc, jbc][k] - fld[inb, jnb][k]
                        )
                    # 巨觀量由更新後的 f 重算，保持 f 與 (rho, vel) 自洽
                    rho_c = 0.0
                    mx_c = 0.0
                    my_c = 0.0
                    for k in ti.static(range(9)):
                        fk = fld[ibc, jbc][k]
                        rho_c += fk
                        mx_c += fk * self.e[k, 0]
                        my_c += fk * self.e[k, 1]
                    self.rho[ibc, jbc] = rho_c
                    self.vel[ibc, jbc] = tm.vec2(mx_c / rho_c, my_c / rho_c)
                else:
                    self.vel[ibc, jbc] = self.vel[inb, jnb]
                    self.rho[ibc, jbc] = self.rho[inb, jnb]
                    fld[ibc, jbc] = (
                        self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + fld[inb, jnb]
                    )

    @ti.kernel
    def _apply_bc_pp(self, fld: ti.template()):
        self.frame_count[None] += 1
        # 緩啟動 (Smoother Cosine Ramp)；warmup_steps=0 時 ramp=1（顯式分支，
        # 不依賴除零→inf 行為，audit C7）。
        ramp = 1.0
        if self.warmup_steps > 0:
            progress = tm.min(1.0, float(self.frame_count[None]) / self.warmup_steps)
            ramp = 1.0 - tm.cos(0.5 * 3.14159265 * progress)

        for j in range(1, self.ny - 1):
            self._apply_bc_core_pp(fld, 1, 0, 0, j, 1, j, ramp)
            self._apply_bc_core_pp(fld, 1, 2, self.nx - 1, j, self.nx - 2, j, ramp)
        for i in range(self.nx):
            self._apply_bc_core_pp(fld, 1, 1, i, self.ny - 1, i, self.ny - 2, ramp)
            self._apply_bc_core_pp(fld, 1, 3, i, 0, i, 1, ramp)

        # Phase 4A: HWBB 在 streaming 時已處理壁面反射，
        # 不再重置固體節點的分布函數為平衡態。
        # 僅設定 vel=0 供輸出/視覺化使用。
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                self.vel[i, j] = 0.0, 0.0

    @ti.kernel
    def _recompute_moments_pre(self, src: ti.template()):
        """由當前分布場重建 post-streaming（碰撞前）的 9 通道矩。

        ping-pong 路徑下 `_collide_stream_pp` 只在「有人要讀」的步才寫
        `moments_pre_field`（每步寫整場要 132 MB）。snapshot 這種不定時的讀取
        就靠本 kernel 按需重建：重做一次 streaming（含 HWBB）並投影出矩。

        成本是一次讀 f + 一次寫矩，只在真正需要時付；相對於每步都寫划算得多。

        ★streaming 與矩投影必須與 `_collide_stream_pp` **逐行相同**，否則
        snapshot 的矩與時均累加的矩會來自兩種不同定義。本 kernel 由該函式的
        前半段程式化生成（見 git 紀錄），不是手抄。
        """
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):

            # 固體節點不參與碰撞與串流；本 kernel 不寫 dst，直接跳過
            if self.mask[i, j] == 1.0:
                continue

            # ============================================================
            # 1. Streaming + Half-way Bounce-Back (Phase 4A)
            # ------------------------------------------------------------
            # Pull-scheme: 從鄰居節點拉取分布函數
            # HWBB: 若鄰居為固體，取自身反方向的值（壁面在半格位置，二階精度）
            # D2Q9 opposite: 0→0, 1↔3, 2↔4, 5↔7, 6↔8
            # ============================================================
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                if self.mask[ip, jp] == 1.0:
                    # HWBB: 粒子在半格壁面反彈，取自身反方向
                    k_opp = ti.static([0, 3, 4, 1, 2, 7, 8, 5, 6][k])
                    f_temp[k] = src[i, j][k_opp]
                else:
                    f_temp[k] = src[ip, jp][k]

            # ============================================================
            # 2. MRT Moment Transformation (矩陣轉換：速度空間 → 矩空間)
            # ------------------------------------------------------------
            # m = M · f
            # M 為 9×9 的 D2Q9 MRT 轉換矩陣 (Lallemand & Luo, 2000)
            # 矩空間順序：[rho, e, eps, jx, qx, jy, qy, pxx, pxy]
            # ============================================================
            m = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.M_field[r, c] * f_temp[c]
                m[r] = val
            self.moments_pre_field[i, j] = m

    # ------------------------------------------------
    # region moments for outputs
    @ti.kernel
    def compute_moments_for_output(self):
        """
        將 f 分布函數投影到 MRT 矩空間，準備輸出給 AI 訓練
        順序對應標準 D2Q9 MRT (Lallemand & Luo):
        0: rho (密度)
        1: e   (能量 - 與壓力相關)
        2: eps (能量平方 - 高階修正)
        3: jx  (動量 x)
        4: qx  (熱通流 x)
        5: jy  (動量 y)
        6: qy  (熱通流 y)
        7: pxx (正應力 - 黏性相關)
        8: pxy (切應力 - 渦旋相關)

        ★ 矩公式已抽出為模組層級 `mrt_moments_from_f`（同檔案上方），
        與時均累加器共用同一份定義（DEEP_AUDIT_2026-07 Stage 1.15）。
        """
        for i, j in self.f_new:
            self.moments_field[i, j] = mrt_moments_from_f(self.f_new[i, j])

    @ti.kernel
    def _fill_moments_pre_boundary(self):
        """補 domain 最外一圈的 pre-collision 矩。

        `collide_and_stream` 的 ndrange 是 `((1, nx-1), (1, ny-1))`，最外圈不被
        寫入。該圈是 ghost/BC 層，其物理狀態由 `apply_bc` 寫進 `f_old`，故直接
        由 `f_old` 投影——對邊界而言這比 `f_new`（未經 BC 修正）更貼近真值。
        """
        for i, j in self.f_old:
            if i == 0 or j == 0 or i == self.nx - 1 or j == self.ny - 1:
                self.moments_pre_field[i, j] = mrt_moments_from_f(self.f_old[i, j])

    def get_moments_numpy(self):
        """回傳 9 通道 MRT 矩 `(nx, ny, 9)`。

        來源由 `moments_export_stage` 決定：`"pre_collision"`（預設、物理正確）
        或 `"post_collision"`（2026-07-29 前的舊行為，僅供對照復現）。
        """
        if self.moments_export_stage == "pre_collision":
            if self.pingpong:
                # ★ping-pong 下矩只在需要的步才寫，故此處按需重建。
                #   要重建的是**剛結束那一步**的 pre-collision 矩，所以要讀
                #   該步 collide 的 src——不是最新的 dst（那會超前一步）。
                #   parity 追蹤：進入第 N 步時 parity=p
                #     p=0 → collide(f_old→f_new)、bc(f_new)、parity→1
                #     p=1 → collide(f_new→f_old)、bc(f_old)、parity→0
                #   故「上一步的 src」= parity==1 ? f_old : f_new。
                #   該場在該步之後未被寫入，內容仍是 collide 當時讀的值。
                #   ★用錯邊會差一步：實測 mom 相對 L2 從 4.6e-8 惡化到 2.6e-4。
                if self._pp_parity == 1:
                    self._recompute_moments_pre(self.f_old)
                else:
                    self._recompute_moments_pre(self.f_new)
            self._fill_moments_pre_boundary()
            return self.moments_pre_field.to_numpy()
        self.compute_moments_for_output()
        return self.moments_field.to_numpy()

    def get_mrt_moments_np(self) -> np.ndarray:
        """回傳當前 9 通道 MRT moments，shape `[9, ny, nx]`，dtype float32。

        為 CEDS Phase 2 convergence trajectory dump 提供 channel-first 介面。
        既有 `get_moments_numpy()` 回傳 `(nx, ny, 9)`（Taichi `(i, j) = (x, y)`
        慣例），這裡轉置為 `(channels, ny, nx)` 以對齊 LBM_Writer 既有 H/W
        儲存慣例（H=ny, W=nx）。

        Returns:
            np.ndarray，shape `[9, ny, nx]`，dtype `np.float32`。
        """
        moments_xyc = self.get_moments_numpy()  # (nx, ny, 9)
        # (nx, ny, 9) -> (9, ny, nx)：channel-first + (H=ny, W=nx)
        return moments_xyc.transpose(2, 1, 0).astype(np.float32, copy=False)

    # endregion
    # ------------------------------------------------
