"""DecomposeBackend — LC-decompose 雙模組推論 handoff。

Phase MEAN：mean_module rollout，KE 兩窗判準收斂 → 凍結 mean。
Phase TURB：turb_module 以 fluct 為 state，凍結 mean 經空間 FiLM 注入，無限 rollout。
phy_field() 回 denorm(mean + fluct)（Reynolds 組合）。

config.yaml（model 段）：
    backend_type: "decompose"
    delta_ctu: 1.0
    mean: {tes_run_config: ..., checkpoint: ...}
    turb: {tes_run_config: ..., checkpoint: ...}
    decompose:
      mean_max_steps: 200
      mean_min_steps: 8
      ke_window_ctu: 2.0
      ke_eps: 0.05
      fluct_seed_amp: 1.0e-3
      fluct_seed: 42
      stop_metric: "ke_two_window"        # 見下；預設＝現行行為
      consec_l2_eps: 1.0e-3               # consec_l2_phy 判準門檻
      stop_evidence_full_run: false       # 見下；預設＝現行行為

上列 ``decompose.*`` 鍵中，``stop_metric`` / ``consec_l2_eps`` /
``stop_evidence_full_run`` 為後加的可選開關，程式以 ``dict.get`` 搭配上面這份樣板的
預設值讀取，好讓既有部署 config（沒有這些鍵）維持原行為。前端的
``config.yaml`` 已顯式寫入這些鍵，該側一律直接存取、不留 fallback。

★停步判準：現行判準 ``_two_window_ke`` 與 ``ke_anchor`` 是同一個泛函（KE），
修正器一開就同時動到「被判的量」與「判準」。``decompose.stop_metric`` 提供第二種
與 KE 正交的判準 ``consec_l2_phy``（相鄰兩步 physics 相對 L2）。
**預設維持 ``ke_two_window``**，避免動到既有部署數字。

★並排記錄的實際能力：兩判準每步都算，但 active 判準一收斂就 handoff 並
結束迴圈，因此**預設只觀察得到「另一判準是否比 active 更早收斂」**；較晚者的步數
在 :attr:`DecomposeBackend.stop_evidence` 中恆為 ``None``。要拿到完整並排資料，需開
``decompose.stop_evidence_full_run: true``（續跑到兩判準都宣告過或到 max_steps）。
"""
from __future__ import annotations

import numpy as np
import torch

from lbmnca.nca.inference.backends.tes_backend import TESBackend
from lbmnca.nca.inference.geometry_types import GeometryInputs
from lbmnca.nca.tes.inference import protocol

STOP_METRIC_KE_TWO_WINDOW = "ke_two_window"
STOP_METRIC_CONSEC_L2 = "consec_l2_phy"

PHASE_MEAN = "mean"
PHASE_TURB = "turb"


def _two_window_ke(
    ke: list[float], ctu: list[float], *, window_ctu: float, eps: float
) -> tuple[float, float]:
    """KE 兩窗收斂判準（複刻自 tes 評估模組的兩窗邏輯）。"""
    n = min(len(ke), len(ctu))
    if n < 4:
        return 0.0, float("nan")
    ctu_arr = np.asarray(ctu[:n], dtype=np.float64)
    ke_arr = np.asarray(ke[:n], dtype=np.float64)
    finite = np.isfinite(ctu_arr) & np.isfinite(ke_arr)
    ctu_arr, ke_arr = ctu_arr[finite], ke_arr[finite]
    if ctu_arr.size < 4:
        return 0.0, float("nan")
    t_end = float(ctu_arr[-1])
    t_mid = t_end - window_ctu
    t_start = t_end - 2.0 * window_ctu
    if t_start < float(ctu_arr[0]):
        return 0.0, float("nan")
    m1_mask = (ctu_arr > t_start) & (ctu_arr <= t_mid)
    m2_mask = (ctu_arr > t_mid) & (ctu_arr <= t_end)
    if int(m1_mask.sum()) < 2 or int(m2_mask.sum()) < 2:
        return 0.0, float("nan")
    m1 = float(np.mean(ke_arr[m1_mask]))
    m2 = float(np.mean(ke_arr[m2_mask]))
    ratio = abs(m2 - m1) / (abs(0.5 * (m1 + m2)) + 1e-10)
    return (1.0 if ratio < eps else 0.0), float(ratio)


class DecomposeBackend:
    """InferenceBackend：mean_module ⊕ turb_module 兩階段推論。"""

    def __init__(self, H: int, W: int, config: dict, device: str = "cuda") -> None:
        self.H = H
        self.W = W
        self.phy_channels = 9
        self.has_convergence = True
        self.device = torch.device(device)
        self._config = config

        dcfg = config["model"].get("decompose", {})
        self._mean_max_steps = int(dcfg.get("mean_max_steps", 200))
        self._mean_min_steps = int(dcfg.get("mean_min_steps", 8))
        self._ke_window_ctu = float(dcfg.get("ke_window_ctu", 2.0))
        self._ke_eps = float(dcfg.get("ke_eps", 0.05))
        self._seed_amp = float(dcfg.get("fluct_seed_amp", 1e-3))
        self._seed = int(dcfg.get("fluct_seed", 42))
        # step<->CTU 換算採直接存取 + 與協定硬核對（禁 fallback）。
        self._delta_ctu = protocol.assert_delta_ctu(
            config["model"]["delta_ctu"], kind="mean",
            context="config.model.delta_ctu",
        )
        # 停步判準開關。預設 ke_two_window＝現行行為，不動既有數字。
        # `.get` 例外理由：本 backend 的 config 由呼叫端（src/lbmnca/nca/inference/engine.py）
        # 任意組出，沒有一份必然含此鍵的 config 檔；這三個鍵是後加的可選開關，
        # 既有 config（未含此鍵）必須維持原行為不中止。鍵與預設值同時寫在本檔模組
        # docstring 的 config 樣板中。前端側因 config.yaml 已顯式寫入這些鍵，
        # 該側一律直接存取、不留 fallback。
        self._stop_metric = str(dcfg.get("stop_metric", STOP_METRIC_KE_TWO_WINDOW))
        if self._stop_metric not in (STOP_METRIC_KE_TWO_WINDOW, STOP_METRIC_CONSEC_L2):
            raise ValueError(
                f"未知的 decompose.stop_metric={self._stop_metric!r}；僅支援 "
                f"{STOP_METRIC_KE_TWO_WINDOW!r} 與 {STOP_METRIC_CONSEC_L2!r}。"
            )
        self._consec_l2_eps = float(dcfg.get("consec_l2_eps", 1e-3))
        # active 判準一收斂即 handoff 並結束迴圈，故預設只能看到
        # 「另一判準是否比 active 早收斂」。True = 續跑到兩判準都宣告過（或 max_steps），
        # 才有完整並排資料。預設 False＝現行行為不變。
        self._stop_evidence_full_run = bool(dcfg.get("stop_evidence_full_run", False))

        self._mean_ad = TESBackend(H, W, self._sub_config("mean"), device=device)
        self._turb_ad = TESBackend(H, W, self._sub_config("turb"), device=device)

        self.phase: str = PHASE_MEAN
        self._mean_steps = 0
        self._ke_curve: list[float] = []
        self._consec_l2_curve: list[float] = []
        self._ke_ratio = float("nan")
        # 兩判準各自「首次宣告收斂」的步數。
        # ★None 的意思是「迴圈結束前該判準未宣告過」——在預設
        # stop_evidence_full_run=False 下，晚於 active 判準者恆為 None，因為迴圈在
        # active 收斂當步就結束了。要看「哪個比較晚」必須開 full_run。
        self.stop_evidence: dict[str, object] = {
            "stop_metric": self._stop_metric,
            "full_run": self._stop_evidence_full_run,
            "ke_two_window_step": None,
            "consec_l2_phy_step": None,
            "consec_l2_eps": self._consec_l2_eps,
            "mean_steps_at_stop": None,
        }
        self._mean_norm: torch.Tensor | None = None
        self._prev_phys: torch.Tensor | None = None
        self._activity: torch.Tensor | None = None
        self._steady_mean: torch.Tensor | None = None

    def _sub_config(self, key: str) -> dict:
        sub = dict(self._config["model"].get(key) or {})
        if not sub:
            raise FileNotFoundError(
                f"backend_type=decompose 需要 config.model.{key}"
                "（tes_run_config + checkpoint）"
            )
        return {
            "model": {**sub, "delta_ctu": self._config["model"]["delta_ctu"]},
        }

    def prepare(self, geo: GeometryInputs, warm: bool) -> None:
        if warm and self._steady_mean is not None:
            self._mean_ad._steady = self._steady_mean
        self._mean_ad.prepare(geo, warm=warm and self._steady_mean is not None)
        self._turb_ad._mask = self._mean_ad._mask
        self._turb_ad._build_cond(geo)
        self._turb_static = self._turb_ad._normalize_static(geo)

        self.phase = PHASE_MEAN
        self._mean_steps = 0
        self._ke_curve = []
        self._consec_l2_curve = []                 # 每次 prepare 重置
        self.stop_evidence["ke_two_window_step"] = None
        self.stop_evidence["consec_l2_phy_step"] = None
        self._ke_ratio = float("nan")
        self._mean_norm = None
        self._prev_phys = None
        self._activity = torch.ones(1, self.H, self.W, device=self.device)

    def _mean_phys_norm(self) -> torch.Tensor:
        ad = self._mean_ad
        return ad._state[:, ad._sd : ad._sd + ad._pd]

    def _ke_norm(self, phys: torch.Tensor) -> float:
        fm = self._mean_ad._mask
        ke = phys[:, 3].square() + phys[:, 5].square()
        denom = fm.squeeze(1).sum().clamp_min(1.0)
        return float((ke * fm.squeeze(1)).sum().item() / denom.item())

    def _print_stop_evidence(self) -> None:
        """把 :attr:`stop_evidence` 印出一行，供事後查核判準結果。

        ``full_run=False``（預設）時晚於 active 判準者恆為 ``None``——代表迴圈在它
        宣告前就結束了，不是「它不會收斂」。
        """
        self.stop_evidence["mean_steps_at_stop"] = self._mean_steps
        print(f"[decompose] stop_evidence={self.stop_evidence}")

    def _handoff_to_turb(self) -> None:
        self._print_stop_evidence()
        self._mean_norm = self._mean_phys_norm().detach().clone()
        gen = torch.Generator(device="cpu").manual_seed(self._seed)
        fluct0 = (
            torch.randn(self._mean_norm.shape, generator=gen, dtype=torch.float32)
            .to(self.device) * self._seed_amp
        )
        fluct0 = fluct0 * self._mean_ad._mask
        self._turb_ad._state = self._turb_ad.model.init_state(self._turb_static, fluct0)
        self.phase = PHASE_TURB
        print(f"[decompose] mean 收斂 handoff → TURB_LIVE"
              f"（steps={self._mean_steps}, ke_ratio={self._ke_ratio:.4f}）")

    def step(self, n: int = 1) -> None:
        for _ in range(max(1, n)):
            self._step_one()

    def _step_one(self) -> None:
        if self.phase == PHASE_MEAN:
            prev = self._mean_phys_norm().detach().clone()
            self._mean_ad.step(1)
            phys = self._mean_phys_norm()
            self._ke_curve.append(self._ke_norm(phys.float()))
            self._mean_steps += 1
            with torch.inference_mode():
                delta = (phys - prev).abs().mean(dim=1)
                thr = delta.mean() * 0.5 + 1e-8
                self._activity = (delta > thr).float()
            self._prev_phys = phys
            # consec_l2_phy = 相鄰兩步 physics 相對 L2（與 ke_anchor 正交）。
            # 兩判準每步都算，各自記下首次宣告收斂的步數。
            l2 = protocol.consec_rel_l2(phys, prev, self._mean_ad._mask)
            self._consec_l2_curve.append(l2)
            if self._mean_steps >= self._mean_min_steps:
                ctu_axis = [i * self._delta_ctu for i in range(len(self._ke_curve))]
                conv, self._ke_ratio = _two_window_ke(
                    self._ke_curve, ctu_axis,
                    window_ctu=self._ke_window_ctu, eps=self._ke_eps,
                )
                conv_l2 = 1.0 if l2 < self._consec_l2_eps else 0.0
                if conv >= 1.0 and self.stop_evidence["ke_two_window_step"] is None:
                    self.stop_evidence["ke_two_window_step"] = self._mean_steps
                if conv_l2 >= 1.0 and self.stop_evidence["consec_l2_phy_step"] is None:
                    self.stop_evidence["consec_l2_phy_step"] = self._mean_steps
                active = (conv if self._stop_metric == STOP_METRIC_KE_TWO_WINDOW
                          else conv_l2)
                both_declared = (
                    self.stop_evidence["ke_two_window_step"] is not None
                    and self.stop_evidence["consec_l2_phy_step"] is not None
                )
                if active >= 1.0 and (not self._stop_evidence_full_run or both_declared):
                    self._handoff_to_turb()
                    return
            if self._mean_steps >= self._mean_max_steps:
                print(f"[decompose] ⚠ mean 未收斂達上限 {self._mean_max_steps} 步"
                      f"（ke_ratio={self._ke_ratio:.4f}）→ 仍切 TURB（容錯）")
                self._handoff_to_turb()
        else:
            ad = self._turb_ad
            # 本步**不傳 ke_anchor**：DecomposeBackend.prepare() 從不呼叫
            # self._turb_ad.prepare()（只手動設 _mask / _build_cond / _turb_static），
            # 而 _ke_anchor 只在 TESBackend.prepare() 內被賦值，此處傳了也恆為 {}。
            # 已於 tests/nca/tes/test_forward_ke_anchor_coverage.py 的 ALLOWLIST 具名豁免；
            # 理由同前端側：turb arch（turb_perception_a）的 forward 不收 ke_anchor，
            # 且 turb config 為 ke_anchor_mode='off'。日後 turb 線要接三態協定，
            # 請改為呼叫 self._turb_ad.prepare() 取真錨點。
            with torch.inference_mode():
                ad._state = ad.model(
                    ad._state, ad._cond, ad._mask, 1,
                    return_trajectory=False, g_vec_norm=ad._g_vec_norm,
                    mean_flow_norm=self._mean_norm,
                )

    def phy_field(self) -> torch.Tensor:
        if self.phase == PHASE_MEAN or self._mean_norm is None:
            return self._mean_ad.phy_field()
        ad = self._turb_ad
        fluct = ad._state[:, ad._sd : ad._sd + ad._pd]
        composed = (self._mean_norm + fluct)[0]
        return ad.field_norm.denorm_mphy(composed).contiguous()

    def mean_field(self) -> torch.Tensor | None:
        """回傳 [9,H,W] denorm mean moments（ū 來源，雙層 IBFV 的 mean 層）。

        MEAN 期回當前 mean module 的 moments（即時推進中）；
        TURB 期回凍結的 self._mean_norm 經 denorm_mphy（Reynolds 分解的時間平均項）。

        注意：回傳的是 MRT moments（9ch，含 rho/j 等），非 velocity。
        velocity（u = j / rho）由前端的 field_ops.derive_fields 計算（DRY）。

        Returns:
            [9, H, W] CUDA float32 denorm moments；prepare() 前 mean state 未建則 None。
        """
        if self._mean_norm is not None:
            # TURB 期：凍結 mean（norm，[1,9,H,W]）→ 取 [9,H,W] denorm。
            return self._turb_ad.field_norm.denorm_mphy(self._mean_norm[0]).contiguous()
        if self._mean_ad._state is None:
            return None
        # MEAN 期：當前 mean module moments（已 denorm，[9,H,W]）。
        return self._mean_ad.phy_field()

    def fluct_field(self) -> torch.Tensor | None:
        """回傳 [9,H,W] denorm fluct moments（u' 來源，雙層 IBFV 的 turb 層）。

        TODO Phase B：denorm 為非線性（atanh 反變換），故 u' 物理量必須以
        ``denorm(mean + fluct) - denorm(mean)`` 計算，不可直接 ``denorm(fluct)``。
        Phase A 先回 None（stub）。

        Returns:
            None（Phase A stub）；Phase B 改回 [9, H, W] CUDA float32 denorm fluct moments。
        """
        return None

    def convergence_field(self) -> torch.Tensor | None:
        if self.phase == PHASE_MEAN and self._activity is not None:
            return self._activity
        return None

    def is_diverged(self) -> bool:
        if self.phase == PHASE_MEAN:
            return self._mean_ad.is_diverged()
        ad = self._turb_ad
        if ad._state is None:
            return False
        fluct = ad._state[0, ad._sd : ad._sd + ad._pd]
        return bool(torch.isnan(fluct).any() or fluct.abs().max() > 1e4)

    def dynamic_dim(self) -> int:
        return self._turb_ad.dynamic_dim()

    def save_steady(self) -> None:
        if self._mean_ad._state is not None:
            self._steady_mean = self._mean_ad._state.clone()


__all__ = ["DecomposeBackend"]
