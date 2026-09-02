"""求解迴圈內時均累加器（DEEP_AUDIT_2026-07 Stage 1.15）。

問題
----
生產 GT 的 ``mean_flow_post_conv``（``src/post_process/tes_h5_to_zarr.py``）
是對 H5 內 post-conv **snapshot** 取算術平均。每 case 只有 26-29 幀、
post-conv 間距約 1-2 CTU，而脫落週期同量級 → 該「時均」其實是**極稀疏抽樣
的相位殘留**：

* 監督者實測 c0137 / c0162 的 ``SE_full``（同一條軌跡、前半 vs 後半分半差 /2）
  = 0.167 / 0.066，達論文所報模型誤差 0.229 的 29-73%。
* 更糟的是**混疊**：c0137 的 rho/e/eps 通道，偶數幀均值與奇數幀均值的 RMS 差
  達 0.99（tanh 值域上限 1.0），全窗均值只有 0.06 —— 靠正負抵消得到小均值，
  幀數為奇數時抵消不完全，殘留偏差 ~= 0.99/29 ~= 0.034。

解法
----
把時均改成 **求解迴圈內逐步累加**（每個 LBM 步都進 sum），一次消除抽樣雜訊
與混疊。GPU 成本 = 每步一次 element-wise 加法（實測見 Stage 1.15 報告）。

設計選擇（★ 與監督者原提案的差異與理由）
------------------------------------------
監督者原提案：全程累加 ``S(t)``，在收斂閂鎖當下另存 ``S_conv``/``n_conv``，
事後以 ``(S_end - S_conv)/(n_end - n_conv)`` 取 post-conv 時均。

本模組改採 **分段和（segment sums）**：把整條軌跡切成固定 CTU 長度的段，
每段獨立累加、段末讀出並歸零，存**每段的段時均**與該段步數。理由：

1. **不把正確性綁在收斂偵測上。** 監督者 1.13 已證 KE 主導模態週期 ~11.1 CTU、
   收斂判準的兩個半窗合計只有 10 CTU，「收斂閂鎖時刻」原理上不可靠。單一
   ``S_conv`` 快照等於把起點永久釘死在一個不可信的時刻上；分段和則讓**任何**
   起點都能事後重算（解析度 = 段長），不必預先猜、也不必重跑模擬。
2. **分半自我檢查是自然副產物。** 任意窗都能拆成前半段集合 / 後半段集合，
   直接算 ``Δ_self(前半|後半)``，不需要事先預測模擬何時停、也不需要
   snapshot（驗收依據）。
3. **數值條件更好。** 段時均直接是 ``S_seg/n_seg``，重組窗時均是**正權重加權
   平均**，不含 ``S_end - S_conv`` 這種大數相減；以 float32 存段時均不會有
   災難性抵消（相對誤差 ~1e-7 量級）。
4. 全域累加量可由各段完全重建（``Σ n_i m_i / Σ n_i``），資訊無損。

代價：每段一份 [9,360,640] float32（8.3 MB）。段長由 config 控制，見
``segment_ctu``。

數值精度
--------
* GPU 累加器 = **float64**（``ti.f64``）。40 萬步 float32 累加的相對誤差
  可達 ~1e-2（隨機遊走 ~ sqrt(N)*eps）；float64 為 ~1e-11，完全不需要
  Kahan 補償求和。成本：每步 9×360×640 = 2.07M 次 f64 加法，相對於
  collide_and_stream 可忽略（實測見報告）。
* 段時均以 float32 存檔（值域 O(1)，相對精度 6e-8），僅用於存檔，不參與累加。

網格對齊（★ 必須與 snapshot 路徑同一表示，否則新舊兩法不可比）
------------------------------------------------------------
``io/tes_writer.TESCaseWriter._crop_downsample_fp16`` 的作法：

1. ``moments[slice_x, slice_y, :]`` 裁掉 sponge+buffer → (crop_w, crop_h, 9)
   = ROI = 1920×1080
2. ``transpose(1,0,2)`` → (crop_h, crop_w, 9)，row = y、col = x
3. 逐通道 ``cv2.resize(..., (W_ds, H_ds), INTER_AREA)`` 3× 降採樣 → 640×360

★ 已數值驗證：整數倍縮放時 ``cv2.INTER_AREA`` **等於** 3×3 方塊平均
（1080×1920 隨機場實測最大絕對差 2.4e-7 = float32 捨入）。因此本累加器
直接在 GPU 上做「裁切 + 3×3 方塊平均」，得到與 snapshot **逐位元同一種
表示**的 [9,360,640] 場，兩法可直接相減比較。

唯一差異：snapshot 路徑存 fp16（相對量化 ~5e-4，且逐幀獨立→平均後抵消），
累加路徑全程 float64。此差異使累加法**更準**，不影響可比性。

時間點：矩由 ``f_new`` 投影（與 ``compute_moments_for_output`` 同一個
``mrt_moments_from_f``），在每步 ``collide_and_stream → update_macro_var →
apply_bc`` 之後累加，與 snapshot 取樣時機一致（``apply_bc`` 只改 ``f_old``
的 ghost node，不動 ``f_new``）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import taichi as ti

from .LBM2D_MRT_LES import LBM2D_MRT_LES, mrt_moments_from_f

ACCUM_VERSION: str = "running_accum_v2"
"""H5 ``/mean_accum`` 群組的 schema 版本標籤。

v2（Stage 1.16 修 3）：新增以**步數**定義段長（``segment_steps``），
並記錄縱向聲學共振週期，使段界可與該週期公度。"""

NUM_CHANNELS: int = 9

CS_LU: float = 1.0 / np.sqrt(3.0)
"""LBM D2Q9 晶格音速（lattice unit）。"""


def acoustic_period_steps(nx: int, u_inlet_lu: float) -> float:
    """求解域縱向聲學**四分之一波共振**週期（LBM 步）。

    ★ DEEP_AUDIT_2026-07 Stage 1.17 的發現，1.16 修 3 落地。

    主導 GT 時均不確定度的**不是渦脫**，而是求解域本身的縱向聲學駐波：
    入口 Zou-He 指定速度 = 剛性端（速度節點），出口指定壓力 = 開放端
    （壓力節點）→ 基模波長 ``lambda = 4*nx``，週期

        ``T = lambda / c_s = 4*nx/c_s``，再乘平均流造成的 ``1/(1 - M^2)`` 修正。

    這一條同時解釋了先前四個互相獨立的異常：

    * 1.13 的 KE 主峰 ``T ~= 10.9 CTU``（不是脫落週期，是這個）；
    * 1.11 步驟 5「``ny`` 加倍後主峰完全不變」—— 縱向模態只依賴 ``nx``；
    * 1.12 的 rho/e/eps 混疊 —— 聲學就是密度/壓力波，擾動能量集中在這三通道；
    * 1.14 的「KE 與 enstrophy 選中同一個錯 bin」—— 同一個相干模態。

    ★ **在步數單位下這是生產母體常數**：監督者掃全部 814 個生產 config，
    ``nx=2560`` 814/814、``u_inlet=0.05`` 814/814。但換算成 CTU 後**逐 case 不同**
    （``L_char in [25,220] px`` -> ``T in [4.06, 35.74] CTU``），
    所以以 CTU 定義段長**無法**表達其整數倍 —— 這正是要改用步數的原因。

    Args:
        nx: 求解域 x 方向格點數（含 sponge/buffer，即 solver 全域寬度）。
        u_inlet_lu: 入口速度（lattice unit）。

    Returns:
        聲學週期（LBM 步，float）。

    Raises:
        ValueError: ``nx <= 0``、``u_inlet_lu < 0``，或 Mach >= 1。
    """
    if nx <= 0:
        raise ValueError(f"nx 必須 > 0，得到 {nx}")
    if u_inlet_lu < 0.0:
        raise ValueError(f"u_inlet_lu 不可為負，得到 {u_inlet_lu}")
    mach = u_inlet_lu / CS_LU
    if mach >= 1.0:
        raise ValueError(f"Mach = {mach} >= 1，聲學週期公式不適用")
    return (4.0 * nx / CS_LU) / (1.0 - mach ** 2)


@ti.data_oriented
class TimeMeanAccumulator:
    """在求解迴圈內逐步累加 MRT 矩，產生分段時均。

    Attributes:
        h_ds: 降採樣後高度（= crop_h / downsample）。
        w_ds: 降採樣後寬度（= crop_w / downsample）。
        segments: 已封存的段清單，每項為 dict（見 :meth:`close_segment`）。
        n_total: 已累加的 LBM 步數（全程）。
    """

    def __init__(
        self,
        solver: LBM2D_MRT_LES,
        slice_x: slice,
        slice_y: slice,
        downsample: int,
        h_ds: int,
        w_ds: int,
        steps_per_ctu: float,
        segment_ctu: float | None = None,
        segment_steps: int | None = None,
        stride: int = 1,
        acoustic_period_steps_value: float | None = None,
    ) -> None:
        """建立累加器並配置 GPU float64 累加場。

        Args:
            solver: 已 ``init()`` 的 LBM solver（需其 ``f_new`` 場）。
            slice_x: solver x 方向裁切範圍（與 TESCaseWriter 同一份）。
            slice_y: solver y 方向裁切範圍。
            downsample: 降採樣倍率（TESCaseWriter.DOWNSAMPLE_FACTOR，目前 3）。
            h_ds: 降採樣後高度（TESCaseWriter.H_ds）。
            w_ds: 降採樣後寬度（TESCaseWriter.W_ds）。
            steps_per_ctu: ``l_char_px / u_inlet_lu``，步數 → CTU 換算。
            segment_ctu: 段長（CTU）。與 ``segment_steps`` **二擇一**，不可同時給。
                ⚠ 以 CTU 定義的段界**無法與聲學週期公度**（該週期在 CTU 下逐
                case 不同，見 :func:`acoustic_period_steps`），會白白放棄
                「窗長取週期整數倍 -> 相位殘差歸零」這個 lever（1.17 實測窗長差
                2.5% 可讓 SE 差 15.6 倍）。**新資料一律建議用 segment_steps。**
            segment_steps: 段長（LBM 步）。與 ``segment_ctu`` 二擇一。
                建議取 :func:`acoustic_period_steps` 的整數倍。
            acoustic_period_steps_value: 本 case 的聲學週期（步），僅供記錄
                （寫進 H5 attrs 讓下游可判斷段界是否可公度）。
                ⛔ **不參與任何門檻或過濾** —— 這是最佳化不是正確性判準。
            stride: 每幾個 LBM 步累加一次（1 = 每步，預設）。>1 時
                段時均仍為無偏估計，但抽樣密度下降；僅在實測成本不可接受時使用。

        Raises:
            ValueError: 裁切尺寸無法被 ``downsample`` 整除（此時
                ``cv2.INTER_AREA`` 不等於方塊平均，兩法將不可比），或參數非法，
                或 ``segment_ctu`` / ``segment_steps`` 未恰好給一個。
        """
        crop_w = slice_x.stop - slice_x.start
        crop_h = slice_y.stop - slice_y.start
        if crop_w % downsample != 0 or crop_h % downsample != 0:
            raise ValueError(
                f"[TimeMeanAccum] crop {crop_w}x{crop_h} 無法被 downsample="
                f"{downsample} 整除 -> INTER_AREA 不等於方塊平均，累加法與 "
                "snapshot 法將不可比。請檢查 domain_zones。"
            )
        if crop_h // downsample != h_ds or crop_w // downsample != w_ds:
            raise ValueError(
                f"[TimeMeanAccum] 降採樣尺寸不一致：crop {crop_h}x{crop_w} / "
                f"{downsample} != ({h_ds}, {w_ds})"
            )
        if (segment_ctu is None) == (segment_steps is None):
            raise ValueError(
                "[TimeMeanAccum] segment_ctu 與 segment_steps 必須**恰好給一個**"
                f"（得到 segment_ctu={segment_ctu}, segment_steps={segment_steps}）"
            )
        if segment_ctu is not None and segment_ctu <= 0.0:
            raise ValueError(f"[TimeMeanAccum] segment_ctu 必須 > 0，得到 {segment_ctu}")
        if segment_steps is not None and segment_steps < 1:
            raise ValueError(f"[TimeMeanAccum] segment_steps 必須 >= 1，得到 {segment_steps}")
        if stride < 1:
            raise ValueError(f"[TimeMeanAccum] stride 必須 >= 1，得到 {stride}")
        if steps_per_ctu <= 0.0:
            raise ValueError(f"[TimeMeanAccum] steps_per_ctu 必須 > 0，得到 {steps_per_ctu}")

        # ★N1（2026-07-29）：矩的來源跟隨 solver 的 `moments_export_stage`。
        #   "pre_collision" 時直接讀 solver 已在 collide 內存好的 post-streaming
        #   矩場（免再投影一次，且與 snapshot 路徑同一時間點）；
        #   "post_collision" 為舊行為，由 `f_new` 現場投影。
        self.export_stage = str(solver.moments_export_stage)
        self.f_new = solver.f_new           # Taichi field，kernel 內直接索引
        self.moments_pre = solver.moments_pre_field
        self.x0 = int(slice_x.start)
        self.y0 = int(slice_y.start)
        self.ds = int(downsample)
        self.block_n = int(downsample) ** 2   # 方塊平均的除數（kernel 內為編譯期整數常數）
        self.h_ds = int(h_ds)
        self.w_ds = int(w_ds)
        self.steps_per_ctu = float(steps_per_ctu)
        # 段長一律以**步**為內部真相；segment_ctu 只是換算入口。
        if segment_steps is not None:
            self.segment_steps = int(segment_steps)
            self.segment_basis = "steps"
        else:
            self.segment_steps = int(round(float(segment_ctu) * self.steps_per_ctu))
            self.segment_basis = "ctu"
            if self.segment_steps < 1:
                raise ValueError(
                    f"[TimeMeanAccum] segment_ctu={segment_ctu} 換算後 < 1 步")
        self.segment_ctu = self.segment_steps / self.steps_per_ctu
        self.stride = int(stride)
        self.acoustic_period_steps = (
            float(acoustic_period_steps_value)
            if acoustic_period_steps_value is not None else -1.0
        )

        # ── GPU float64 累加場 [9, H_ds, W_ds] ─────────────────────────────
        self._sum = ti.field(dtype=ti.f64, shape=(NUM_CHANNELS, self.h_ds, self.w_ds))
        self._sum.fill(0.0)

        # ── 段狀態 ────────────────────────────────────────────────────────
        self.n_total: int = 0        # 全程已推進的 LBM 步數
        self._n_seg: int = 0         # 本段已累加的樣本數（stride>1 時 < 步數）
        self._seg_step0: int = 0     # 本段起始步（不含，即上一段結束步）
        self.segments: list[dict[str, Any]] = []

        mib = NUM_CHANNELS * self.h_ds * self.w_ds * 8 / 1024 ** 2
        print(
            f"[TimeMeanAccum] 啟用：grid=[{NUM_CHANNELS},{self.h_ds},{self.w_ds}] "
            f"f64 累加場 {mib:.1f} MiB | segment={self.segment_steps} steps "
            f"(={self.segment_ctu:.4f} CTU, basis={self.segment_basis}) | "
            f"stride={self.stride}"
        )
        # 段界與聲學週期的可公度性：**只報告，不設門檻**（1.16 修 3 的明確要求）。
        if self.acoustic_period_steps > 0.0:
            k = self.segment_steps / self.acoustic_period_steps
            eps = abs(k - round(k)) / max(round(k), 1.0)
            print(
                f"[TimeMeanAccum] 聲學週期 = {self.acoustic_period_steps:.1f} 步 "
                f"(={self.acoustic_period_steps / self.steps_per_ctu:.4f} CTU) | "
                f"段長 = {k:.4f} 個週期 | 失配 eps = {eps * 100:.3f}% "
                f"(1.17: eps<=3.2% 達 10% 目標, <=0.3% 逼近地板；僅供參考，非門檻)"
            )

    # ── 累加 ──────────────────────────────────────────────────────────────

    @ti.kernel
    def _accumulate_kernel_pre(self):
        """累加 solver 的 post-streaming 矩場（N1 修正後的預設路徑）。"""
        for r, c in ti.ndrange(self.h_ds, self.w_ds):
            acc = ti.Vector.zero(ti.f64, NUM_CHANNELS)
            for dy in ti.static(range(self.ds)):
                for dx in ti.static(range(self.ds)):
                    i = self.x0 + c * self.ds + dx
                    j = self.y0 + r * self.ds + dy
                    m = self.moments_pre[i, j]
                    for k in ti.static(range(NUM_CHANNELS)):
                        acc[k] += ti.cast(m[k], ti.f64)
            for k in ti.static(range(NUM_CHANNELS)):
                self._sum[k, r, c] += acc[k] / self.block_n

    @ti.kernel
    def _accumulate_kernel(self):
        """把當前 ``f_new`` 的 MRT 矩以 3×3 方塊平均累加進 float64 sum 場。"""
        for r, c in ti.ndrange(self.h_ds, self.w_ds):
            acc = ti.Vector.zero(ti.f64, NUM_CHANNELS)
            for dy in ti.static(range(self.ds)):
                for dx in ti.static(range(self.ds)):
                    i = self.x0 + c * self.ds + dx      # solver x（= 列方向 col）
                    j = self.y0 + r * self.ds + dy      # solver y（= 行方向 row）
                    m = mrt_moments_from_f(self.f_new[i, j])
                    for k in ti.static(range(NUM_CHANNELS)):
                        acc[k] += ti.cast(m[k], ti.f64)
            # 除以編譯期整數常數（不用 f32 浮點倒數，避免 6e-8 的系統性尺度誤差）
            for k in ti.static(range(NUM_CHANNELS)):
                self._sum[k, r, c] += acc[k] / self.block_n

    def will_accumulate_next(self) -> bool:
        """下一次 :meth:`accumulate` 會不會真的累加（供 solver 省工用）。

        solver 在 ping-pong 路徑下需要事先知道「這一步的矩有沒有人要」，
        才能決定是否寫 `moments_pre_field`（每步 132 MB）。這裡回報的是
        **下一個** `accumulate()` 的判斷結果，與其內部的 stride 邏輯一致。
        """
        if self.stride <= 1:
            return True
        return ((self.n_total + 1) % self.stride) == 0

    def accumulate(self) -> None:
        """推進一個 LBM 步後呼叫；依 ``stride`` 決定是否真的累加。"""
        self.n_total += 1
        if self.stride > 1 and (self.n_total % self.stride) != 0:
            return
        if self.export_stage == "pre_collision":
            self._accumulate_kernel_pre()
        else:
            self._accumulate_kernel()
        self._n_seg += 1

    # ── 分段 ──────────────────────────────────────────────────────────────

    @property
    def next_boundary_step(self) -> int:
        """下一個段邊界對應的全程步數（★ 1.16 修 3：以**步**定義，非 CTU）。"""
        return (len(self.segments) + 1) * self.segment_steps

    def maybe_close_segment(self) -> bool:
        """若已跨過段邊界則封存（一次呼叫可能跨過多個邊界）。

        Returns:
            True 表示本次呼叫至少封存了一段。
        """
        closed = False
        while self.n_total >= self.next_boundary_step:
            if not self.close_segment():   # 本段無樣本（stride > batch）→ 停，避免空轉
                break
            closed = True
        return closed

    def close_segment(self) -> bool:
        """封存目前這一段：讀出 sum、除以樣本數、歸零累加場。

        Returns:
            True 表示成功封存；False 表示本段沒有樣本（不產生空段）。
        """
        if self._n_seg <= 0:
            return False
        seg_sum = self._sum.to_numpy()                      # [9,H,W] float64
        seg_mean = (seg_sum / float(self._n_seg)).astype(np.float32)
        step0, step1 = self._seg_step0, self.n_total
        self.segments.append(
            {
                "mean": seg_mean,
                "n": int(self._n_seg),
                "step0": int(step0),
                "step1": int(step1),
                "ctu0": float(step0) / self.steps_per_ctu,
                "ctu1": float(step1) / self.steps_per_ctu,
            }
        )
        self._sum.fill(0.0)
        self._n_seg = 0
        self._seg_step0 = self.n_total
        return True

    # ── 匯出 ──────────────────────────────────────────────────────────────

    def to_h5_payload(self) -> dict[str, Any] | None:
        """組出寫入 H5 ``/mean_accum`` 的資料（呼叫前應先 close_segment）。

        Returns:
            dict 或 None（無任何段時）。
        """
        if not self.segments:
            return None
        return {
            "segment_mean": np.stack([s["mean"] for s in self.segments], axis=0),
            "segment_n": np.array([s["n"] for s in self.segments], dtype=np.int64),
            "segment_step0": np.array([s["step0"] for s in self.segments], dtype=np.int64),
            "segment_step1": np.array([s["step1"] for s in self.segments], dtype=np.int64),
            "segment_ctu0": np.array([s["ctu0"] for s in self.segments], dtype=np.float64),
            "segment_ctu1": np.array([s["ctu1"] for s in self.segments], dtype=np.float64),
            "attrs": {
                "accum_version": ACCUM_VERSION,
                "accum_dtype": "float64",
                "storage_dtype": "float32",
                "stride": self.stride,
                "segment_steps_target": self.segment_steps,
                "segment_basis": self.segment_basis,
                "segment_ctu_target": self.segment_ctu,
                "acoustic_period_steps": self.acoustic_period_steps,
                "segment_acoustic_periods": (
                    self.segment_steps / self.acoustic_period_steps
                    if self.acoustic_period_steps > 0.0 else -1.0
                ),
                "steps_per_ctu": self.steps_per_ctu,
                "n_steps_total": self.n_total,
                "n_segments": len(self.segments),
                "downsample_mode": f"block_mean_{self.ds}x",
                "moments_source": ("moments_pre_field"
                                   if self.export_stage == "pre_collision"
                                   else "f_new"),
                "note": (
                    "段時均（raw moment 空間，未正規化）。任意窗時均 = "
                    "sum(n_i * mean_i) / sum(n_i)（權重為各段步數）。"
                ),
            },
        }


# 讀取端工具（H5 → 窗時均、分半自我檢查）在
# `src/post_process/mean_accum_io.py`（無 taichi 依賴，供轉檔與分析腳本共用）。
