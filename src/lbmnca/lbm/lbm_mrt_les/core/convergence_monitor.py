"""
收斂監測器：在模擬主迴圈中定期計算並記錄收斂指標。

供 simulation_ops.py 在每個 diagnostic interval 呼叫，
最終結果寫入 run_one_case metadata → sim_results.json。

量測指標：
  1. domain-integrated kinetic energy (KE) 滑動平均
  2. inlet/outlet mass flux balance
  3. max_v 歷史窗口
"""

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


class ConvergenceMonitor:
    """每 N 步計算收斂指標的監測器。

    Args:
        window_size: 滑動窗口大小（幾筆 snapshot）。
        steps_per_ctu: 每個 CTU 的步數，用於換算 CTU 時間。
    """

    def __init__(self, window_size: int = 10, steps_per_ctu: int = 1):
        self._window_size = max(3, window_size)
        self._steps_per_ctu = max(1, steps_per_ctu)

        self._ke_history: deque = deque(maxlen=self._window_size)
        self._max_v_history: deque = deque(maxlen=self._window_size)
        self._mass_flux_history: deque = deque(maxlen=self._window_size)
        self._step_history: deque = deque(maxlen=self._window_size)

        self._snapshot_count = 0
        self._latest: Dict[str, Any] = {}

        # CEDS Phase 2: 完整 snapshot history（無 maxlen），供 post-hoc
        # progress curve 反推使用。每 entry 為 update / simulation_ops inline
        # 寫入的 snapshot dict，至少包含 "step" 與 "ke_proxy"。
        self.history: List[Dict[str, Any]] = []

    def update(self, solver: Any, step: int) -> Dict[str, Any]:
        """採集一次收斂 snapshot。

        Args:
            solver: LBM2D_MRT_LES solver 實例。
            step: 目前的模擬步數。

        Returns:
            本次 snapshot 的指標字典。
        """
        self._snapshot_count += 1

        # 1. max_v（直接從 solver 取，不重複計算）
        max_v = float(solver.get_max_velocity())
        self._max_v_history.append(max_v)

        # 2. Domain-integrated KE (簡化版：用 max_v^2 近似趨勢)
        #    完整版需要 sum(rho * |u|^2)，但那需要從 GPU 拉全場資料
        #    用 max_v 的趨勢足以判斷收斂
        ke_proxy = max_v * max_v
        self._ke_history.append(ke_proxy)

        # 3. Mass flux balance（inlet vs outlet plane）
        #    完整版需讀取邊界 plane 的速度 → 昂貴
        #    用 solver 的 force 作為 proxy：收斂時 force 趨於穩定
        forces = solver.get_force()
        force_mag = float(np.sqrt(forces[0] ** 2 + forces[1] ** 2))
        self._mass_flux_history.append(force_mag)

        self._step_history.append(step)

        # 計算滑動窗口統計
        snapshot = {
            "step": step,
            "ctu": round(step / self._steps_per_ctu, 2),
            "max_v": round(max_v, 6),
            "ke_proxy": round(ke_proxy, 8),
            "force_magnitude": round(force_mag, 6),
        }

        if len(self._ke_history) >= 3:
            ke_arr = np.array(self._ke_history)
            ke_mean = float(np.mean(ke_arr))
            ke_std = float(np.std(ke_arr))
            snapshot["ke_window_mean"] = round(ke_mean, 8)
            snapshot["ke_window_rel_drift"] = (
                round(ke_std / max(ke_mean, 1e-12), 6)
            )

            fmag_arr = np.array(self._mass_flux_history)
            fmag_mean = float(np.mean(fmag_arr))
            fmag_std = float(np.std(fmag_arr))
            snapshot["force_window_mean"] = round(fmag_mean, 6)
            snapshot["force_window_rel_drift"] = (
                round(fmag_std / max(fmag_mean, 1e-12), 6)
            )
        else:
            snapshot["ke_window_mean"] = None
            snapshot["ke_window_rel_drift"] = None
            snapshot["force_window_mean"] = None
            snapshot["force_window_rel_drift"] = None

        self._latest = snapshot
        # CEDS Phase 2: 同步寫入完整 history（供 compute_progress_curve_post_hoc）
        self.history.append(snapshot)
        return snapshot

    def is_converged(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """判斷是否已收斂。

        Args:
            thresholds: 收斂門檻字典。預設使用保守值。
                可用鍵值：ke_window_rel_drift, force_window_rel_drift。

        Returns:
            True 表示流場已收斂。
        """
        if len(self._ke_history) < self._window_size:
            return False

        if thresholds is None:
            thresholds = {
                "ke_window_rel_drift": 0.02,
                "force_window_rel_drift": 0.03,
            }

        ke_drift = self._latest.get("ke_window_rel_drift")
        force_drift = self._latest.get("force_window_rel_drift")

        if ke_drift is None or force_drift is None:
            return False

        return (
            ke_drift < thresholds.get("ke_window_rel_drift", 0.02)
            and force_drift < thresholds.get("force_window_rel_drift", 0.03)
        )

    def get_summary(self) -> Dict[str, Any]:
        """回傳收斂監測摘要，供寫入 metadata。

        Returns:
            包含最終收斂指標的字典。
        """
        return {
            "total_snapshots": self._snapshot_count,
            "window_size": self._window_size,
            "final_snapshot": self._latest,
            "converged": self.is_converged(),
            "max_v_history_last5": [
                round(v, 6) for v in list(self._max_v_history)[-5:]
            ],
        }

    @staticmethod
    def check_ke_convergence(
        ke_history: list,
        threshold: float = 1e-3,
        min_samples: int = 5,
    ) -> tuple[bool, float]:
        """Check KE variance ratio over last min_samples snapshots.

        Args:
            ke_history: List of KE values accumulated by TESCaseWriter._ke_list.
            threshold: std/mean ratio below which convergence is declared.
            min_samples: Minimum history length required.

        Returns:
            (converged, ratio): bool and computed std/mean ratio.
        """
        if len(ke_history) < min_samples:
            return False, 1.0
        recent = ke_history[-min_samples:]
        arr = np.array(recent, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val  = float(np.std(arr))
        ratio    = float(std_val / max(abs(mean_val), 1e-10))
        return ratio < threshold, ratio

    @staticmethod
    def check_ke_convergence_twowindow(
        ke_history: list,
        ctu_history: list,
        window_ctu: float = 5.0,
        eps: float = 0.05,
    ) -> tuple[bool, float]:
        """Two-window mean comparison for unsteady oscillating flow.

        Splits the most recent 2×window_ctu into two halves and compares means.

        ⚠ **已知限制（2026-07-28，DEEP_AUDIT_2026-07 Stage 1.16 更正）**：
        本函式原 docstring 宣稱「每個半窗涵蓋 >=1 個脫落週期，故振盪被平均掉，
        只剩慢漂移」。**該宣稱在全域積分 KE 上不成立**：

          - 1.13 實測（Re=100 cylinder，600 幀）：全域 KE 中真實脫落諧波
            比 FFT 主峰**弱 4744 倍** → KE 幾乎不含脫落訊號，
            「涵蓋一個脫落週期」對這條訊號沒有意義。
          - 1.11 步驟 4 實測：KE 上真正存在的主導模態週期約 **11.1 CTU**、
            調變深度 113%；生產設定 ``window_ctu=5.0`` 的兩個半窗合計 10 CTU
            **短於該週期** → 兩窗均值差主要反映**相位差**而非漂移。

        後果（生產規模）：母體 74.6% 的 ``convergence_ctu`` 卡在最早允許檢查的
        floor；1.14 量到 31.1% 的 post-conv 池 ``|漂移| > 2×SEM_eff``。

        → 本函式的輸出應視為**粗略提示**，不得當作「已達統計穩態」的證明。
        GT 時均的正確性已於 Stage 1.15/1.16 改由分段累加器 + 協定固定窗承擔，
        不再依賴本判準。

        Args:
            ke_history: List of KE values (parallel to ctu_history).
            ctu_history: List of CTU values for each snapshot.
            window_ctu: Half-window size in CTU（生產值 5.0）。
                ★ 沒有可靠依據可重訂此值 —— 唯一能訂它的量是被監測訊號的
                主導模態週期，而 1.14 已撤除週期偵測（全域 KE 上不可偵測），
                生產端亦無真值可驗收任何偵測器。
            eps: |Δmean|/mean threshold for declaring convergence (~0.05).

        Returns:
            (converged, ratio): bool and computed mean-difference ratio.
        """
        n = min(len(ke_history), len(ctu_history))
        if n < 4:
            return False, 1.0
        ctu_arr = np.asarray(ctu_history[:n], dtype=np.float64)
        ke_arr  = np.asarray(ke_history[:n],  dtype=np.float64)
        t_end   = float(ctu_arr[-1])
        t_mid   = t_end - window_ctu
        t_start = t_end - 2.0 * window_ctu
        if t_start < 0:
            return False, 1.0
        mask_w1 = (ctu_arr > t_start) & (ctu_arr <= t_mid)
        mask_w2 = (ctu_arr > t_mid)   & (ctu_arr <= t_end)
        if int(mask_w1.sum()) < 2 or int(mask_w2.sum()) < 2:
            return False, 1.0
        m1 = float(np.mean(ke_arr[mask_w1]))
        m2 = float(np.mean(ke_arr[mask_w2]))
        mb = 0.5 * (m1 + m2)
        ratio = abs(m2 - m1) / (abs(mb) + 1e-10)
        return ratio < eps, ratio

    @staticmethod
    def check_trend_drift(
        ke_history: list,
        ctu_history: list,
        span_ctu: float,
        eps: float = 0.05,
    ) -> tuple[bool, float]:
        """Trend-line 累積漂移閘門。

        對最近 ``span_ctu`` 的 KE 做線性迴歸，要求該區間的**預測總變化量**
        相對均值小於 eps：``|slope| · span / mean < eps``。

        為什麼需要：two-window 判準只約束「相鄰兩窗的局部漂移率」，
        從未約束累積漂移。TIER0 實測（`TIER0_結果與機制定案.md` §1.4）
        宣告收斂後 KE 仍中位再漲 +82%、最高 +2134%，即此閘門要擋的情形。

        依據：Scandurra, Alexias & de Villiers (2025), arXiv:2511.22618 §4-5
        （在 N_eff 修正的信賴區間之外，另加一道 ``n·|∂x̄/∂n| ≤ ε`` 的趨勢檢查）。

        Args:
            ke_history: KE 值序列（與 ctu_history 平行）。
            ctu_history: 各 snapshot 的 CTU。
            span_ctu: 回看區間長度（CTU）。
            eps: 相對總漂移門檻。

        Returns:
            (passed, rel_drift)：是否通過，以及相對總漂移量。
        """
        n = min(len(ke_history), len(ctu_history))
        if n < 4:
            return False, 1.0
        ctu_arr = np.asarray(ctu_history[:n], dtype=np.float64)
        ke_arr = np.asarray(ke_history[:n], dtype=np.float64)
        t_end = float(ctu_arr[-1])
        m = ctu_arr >= (t_end - span_ctu)
        if int(m.sum()) < 4:
            return False, 1.0
        t, y = ctu_arr[m], ke_arr[m]
        mean_val = float(np.mean(y))
        if abs(mean_val) < 1e-30:
            return False, 1.0
        # 一次多項式擬合：slope 單位 = KE / CTU
        slope = float(np.polyfit(t, y, 1)[0])
        actual_span = float(t[-1] - t[0])
        rel_drift = abs(slope) * actual_span / abs(mean_val)
        return rel_drift < eps, rel_drift


    def compute_progress_curve_post_hoc(self) -> np.ndarray:
        """從整段 history 反推 progress %（已知整段曲線後可正確處理 non-monotonic）。

        LBM impulse start 階段 KE 會先升後降，online `-log10(ke_ratio)` 會出現
        progress 反向跳變。改為 post-hoc：用 `cummax(reversed)` 把每個點的
        baseline 設為「該點往後最大 KE」，平滑 transient 上升段，再以
        `-log10` 換算為 0–100% 單調曲線。

        Numerical detail（與規格初版的微調）：規格 normalize 用 `progress[-1]`，
        但 `ke_baseline[-1] == ke_final` 會讓最後一個點的 numer 退化為 epsilon
        → progress[-1] 被 epsilon 拉到一個與其他點不同尺度的大值，造成倒數
        第二點以後出現劇烈跳變。改用 `progress[-2]` 作為 100% 的標準（避免
        epsilon 主導），並把 last entry 顯式設為 100%。

        Returns:
            shape `[N]` 的 float ndarray，對應 `self.history[i]` 的 progress %，
            範圍 [0, 100] 且單調遞增。若 history 不足 2 點則回傳 `[0.0]`。
        """
        if len(self.history) < 2:
            return np.array([0.0], dtype=np.float64)

        ke_series = np.array(
            [float(h["ke_proxy"]) for h in self.history], dtype=np.float64
        )

        # cummax(從後往前)：每個點的 baseline 設為「該點往後的最大 KE」，
        # 移除 impulse start 的 transient 上升段
        ke_baseline = np.flip(np.maximum.accumulate(np.flip(ke_series)))
        ke_final = ke_series[-1]

        # progress = -log10((baseline - final) / (baseline_initial - final))
        denom = ke_baseline[0] - ke_final + 1e-12
        numer = ke_baseline - ke_final + 1e-12
        progress = -np.log10(np.maximum(1e-12, numer / denom))

        # Normalize：用倒數第二點作 100% 標準，避免 last entry 的 epsilon
        # （ke_baseline[-1] == ke_final → numer = 1e-12）主導 normalize。
        # last entry 顯式設為 100，前面的點按 progress[-2] 的尺度線性對映。
        if len(progress) >= 2:
            scale = max(progress[-2], 1e-9)
            progress = 100.0 * progress / scale
            progress[-1] = 100.0
        else:
            progress = 100.0 * progress / max(progress[-1], 1e-9)
        return np.clip(progress, 0.0, 100.0)


class SustainedConvergenceTracker:
    """持續性收斂追蹤器 — 取代「首次命中即閂鎖」的舊判定方式。

    ★ 2026-07-26 新增（DEEP_AUDIT_2026-07 / `TIER0_結果與機制定案.md`）。

    舊行為的問題
    ------------
    ``run_one_case_tes.py`` 原本以 ``convergence_ctu < 0.0`` 作為 guard，
    使 two-window 判準成為**單次閂鎖**：命中一次即永久成立，不再複查。
    對一個振盪中的漂移率訊號而言，這等於量測「首次隨機低於門檻的時刻」，
    而非任何持續狀態。

    TIER0 實測證據（102 case / 670 node）：
      - 母體 **74.6%** 的 convergence_ctu 卡在 floor（允許檢查的最早時刻）
      - 宣告收斂後，**89%** 的 case 若重新套用同一判準會 FAIL
        （ratio 中位回到門檻的 4.7 倍）
      - 宣告收斂後 KE 仍中位再漲 +82%

    對照 OpenLB ``util::ValueTracer::hasConverged()``：每步呼叫、要求**當下**
    窗滿足 σ/μ < ε，退化即不再算收斂。本類別採同樣的「必須持續成立」語義。

    新判定
    ------
    同時滿足才宣告收斂：
      1. two-window ratio < eps **連續 k_required 次**（消除隨機夾點）
      2. trend-line 累積漂移 < cum_eps（消除持續單向漂移）

    收斂時刻取**連續成立的起點**（streak 首次成立的 CTU），而非第 k 次的時刻——
    因為那才是統計穩態實際開始的時間，用它當時均窗起點才正確。
    """

    def __init__(
        self,
        window_ctu: float = 5.0,
        eps: float = 0.05,
        k_required: int = 3,
        cum_eps: float = 0.05,
        enable_trend_gate: bool = True,
    ) -> None:
        """初始化。

        Args:
            window_ctu: two-window 判準的半窗長度（CTU），限制見
                :meth:`ConvergenceMonitor.check_ke_convergence_twowindow`。
            eps: two-window 的 |Δmean|/mean 門檻。
            k_required: 需連續成立幾次。3 次涵蓋約 15 CTU。
                ⚠ 原註解「以 TIER0 實測 T_shed≈8.3 CTU 計，約 1.8 個脫落週期」
                **已於 2026-07-28（Stage 1.16）撤除**：該 T_shed 來自 1.13 證偽的
                週期偵測器（全域 KE 上高估真值 3.59 倍），不可作為任何依據。
                現行說法只剩「連續 3 次 ~= 15 CTU」這個算術事實，
                不宣稱它對應任何物理週期數。
            cum_eps: trend-line 累積漂移門檻。
            enable_trend_gate: 是否啟用累積漂移閘門。
        """
        self.window_ctu = float(window_ctu)
        self.eps = float(eps)
        self.k_required = int(k_required)
        self.cum_eps = float(cum_eps)
        self.enable_trend_gate = bool(enable_trend_gate)

        self.streak: int = 0
        self.streak_start_ctu: float = -1.0
        self.last_ratio: float = 1.0
        self.last_drift: float = 1.0
        self.converged_ctu: float = -1.0
        self.n_rejected_by_trend: int = 0

    def update(self, ke_history: list, ctu_history: list,
               current_ctu: float) -> bool:
        """餵入當前歷史，回傳是否**此刻**判定收斂。

        Args:
            ke_history: KE 序列。
            ctu_history: CTU 序列（與 ke_history 平行）。
            current_ctu: 當前 CTU（用於記錄 streak 起點）。

        Returns:
            True 表示本次呼叫達成收斂條件（僅在首次達成時回傳 True）。
        """
        if self.converged_ctu >= 0.0:
            return False       # 已收斂，不重複宣告

        ok, ratio = ConvergenceMonitor.check_ke_convergence_twowindow(
            ke_history, ctu_history,
            window_ctu=self.window_ctu, eps=self.eps,
        )
        self.last_ratio = ratio

        if not ok:
            self.streak = 0
            self.streak_start_ctu = -1.0
            return False

        if self.streak == 0:
            self.streak_start_ctu = float(current_ctu)
        self.streak += 1

        if self.streak < self.k_required:
            return False

        if self.enable_trend_gate:
            span = self.k_required * self.window_ctu
            passed, drift = ConvergenceMonitor.check_trend_drift(
                ke_history, ctu_history, span_ctu=span, eps=self.cum_eps,
            )
            self.last_drift = drift
            if not passed:
                # 局部平穩但仍持續單向漂移 → 不算收斂，重新累積
                self.n_rejected_by_trend += 1
                self.streak = 0
                self.streak_start_ctu = -1.0
                return False

        # 收斂時刻 = 連續成立的起點（統計穩態實際開始處）
        self.converged_ctu = self.streak_start_ctu
        return True

    def diagnostics(self) -> dict:
        """供寫入 H5 attrs 的診斷欄位。"""
        return {
            "conv_k_required": self.k_required,
            "conv_eps": self.eps,
            "conv_window_ctu": self.window_ctu,
            "conv_cum_eps": self.cum_eps,
            "conv_trend_gate": self.enable_trend_gate,
            "conv_final_ratio": float(self.last_ratio),
            "conv_final_trend_drift": float(self.last_drift),
            "conv_n_rejected_by_trend": self.n_rejected_by_trend,
        }

