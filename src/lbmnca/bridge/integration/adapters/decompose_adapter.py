"""DecomposeAdapter — LC-decompose 雙模組推理 handoff（UI-02/UI-03 修）。

``docs/research_plan/LC_decompose/archive/SSOT_PLAN_lc_decompose_MASTER.md`` §5
（「推理 handoff（應用）」）的前端落地。★三輪更正路徑：原文只寫檔名
``PLAN_lc_decompose_MASTER``，該檔早已移進 ``LC_decompose/archive/`` 並改名為
``SSOT_`` 前綴，照原文搜尋找不到（實測 workspace 內無同名檔）。該文件現為**歷史歸檔**，
與現行程式有出入時以程式為準。

    Phase MEAN（MEAN_CONVERGING）：mean_module rollout，逐步累積 norm-空間
        KE，solver 式兩窗判準（複刻 02 long_rollout._two_window_ke 語意）
        收斂 → 凍結 mean。
    Phase TURB（TURB_LIVE）：turb_module 以 fluct 為 state、凍結 mean 經
        空間 FiLM 注入（forward(..., mean_flow_norm=mean)），無限 rollout；
        `phy_field()` 回 denorm(mean + fluct)（Reynolds 組合）。

對外仍實作 ModelBackend Protocol → render 層零改動。
`has_convergence=True`：MEAN 期提供 per-cell activity 場（既有 convergence
overlay 顯示「均場收斂中」）；TURB 期回 None。

fluct 起點 = 固定 seed 小擾動（非全 0——殘差累加 cell 無內生振盪器，
0→0 永不起振；對齊 02 decompose_inference.seed_fluct，診斷 M7-1）。

config.yaml（model 段）：
    backend: "decompose"
    delta_ctu: 1.0
    mean: {tes_run_config: ..., checkpoint: ...}
    turb: {tes_run_config: ..., checkpoint: ...}
    decompose:
      mean_max_steps: 200
      mean_min_steps: 8
      ke_window_ctu: 2.0      # pseudo-CTU 半窗（step × delta_ctu 軸）
      ke_eps: 0.05            # 過嚴放寬 0.08（上述 MASTER 檔 §9 第 3 點）
      stop_metric: "ke_two_window"   # 或 "consec_l2_phy"（R3-G4，見下）
      consec_l2_eps: 1.0e-3
      stop_evidence_full_run: false  # R3-G4b，見下；預設 false = 現行部署行為
      fluct_seed_amp: 1.0e-3
      fluct_seed: 42

★上列 ``stop_metric`` / ``consec_l2_eps`` / ``stop_evidence_full_run`` 三鍵在本檔是
**直接存取**（``dcfg["..."]``，禁 fallback），缺鍵即 ``KeyError``——照這份樣板抄 config
時三個都要寫進去。本 repo 的 ``config.yaml`` 已顯式寫入。

★DEEP_AUDIT R3-G4（停步判準）：``_two_window_ke`` 與 ``ke_anchor`` 是同一個泛函（KE），
修正器一開就同時動到「被判的量」與「判準」，且 ``mean_min_steps=4`` +
``ke_window_ctu=2.0 / delta_ctu=1.0``（半窗 2 步）使理論最早收斂宣告落在第 5 步。
``stop_metric`` 提供與 KE 正交的替代判準 ``consec_l2_phy``（相鄰兩步 physics 相對 L2，
實作在 ``tes.inference.protocol.consec_rel_l2``——02 與 03 兩端都呼叫該函式，本檔不另
複製一份）。**預設維持 ke_two_window**，切換與否屬作者裁決（D5，會移動論文「部署自動
收斂步數」與加速比分子）。

★R3-G4b（並排記錄的實際能力）：兩判準每步都算，各自記下**首次宣告收斂**的步數於
:attr:`DecomposeAdapter.stop_evidence`。但預設 ``stop_evidence_full_run=False`` 下，
active 判準一收斂就 ``_freeze_mean()`` 並結束迴圈，**晚於 active 者的步數恆為 None**
（意思是「迴圈在它宣告前就結束」，不是「它不收斂」）。要拿到完整並排資料須開
``stop_evidence_full_run: true``——該開關會改變部署輸出的 mean 步數，僅供量測。
"""
from __future__ import annotations

import numpy as np
import torch

from lbmnca.bridge.integration.adapters.tes_adapter import TESAdapter
from lbmnca.bridge.integration.model_backend import GeometryInputs
from lbmnca.nca.tes.inference import protocol as tes_protocol

PHASE_MEAN = "mean"
PHASE_TURB = "turb"

# ★R3-G4：停步判準識別字（與 02 decompose_backend 同名，兩端一致）。
STOP_METRIC_KE_TWO_WINDOW = "ke_two_window"
STOP_METRIC_CONSEC_L2 = "consec_l2_phy"


def _two_window_ke(
    ke: list[float], ctu: list[float], *, window_ctu: float, eps: float
) -> tuple[float, float]:
    """KE 兩窗收斂判準——複刻 02 `eval/long_rollout._two_window_ke`。

    ⚠ 與 02 版為同語意複本（前端避免拉入 02 eval 重依賴鏈）；
    修改 02 版時同步此處。
    """
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


class DecomposeAdapter:
    """ModelBackend：mean_module ⊕ turb_module 兩階段推理。"""

    def __init__(self, H: int, W: int, config: dict, device: str = "cuda") -> None:
        self.H = H
        self.W = W
        self.phy_channels = 9
        self.has_convergence = True          # MEAN 期供 activity overlay
        self.device = torch.device(device)
        self._config = config

        dcfg = config["model"].get("decompose", {})
        self._mean_max_steps = int(dcfg.get("mean_max_steps", 200))
        self._mean_min_steps = int(dcfg.get("mean_min_steps", 8))
        self._ke_window_ctu = float(dcfg.get("ke_window_ctu", 2.0))
        self._ke_eps = float(dcfg.get("ke_eps", 0.05))
        self._seed_amp = float(dcfg.get("fluct_seed_amp", 1e-3))
        self._seed = int(dcfg.get("fluct_seed", 42))
        self._enable_turb = bool(dcfg.get("enable_turb", True))  # False → mean-only，收斂後凍結不切 TURB
        # step<->CTU 換算的單一真相在 tes.inference.protocol.NOMINAL_DELTA_CTU_PER_STEP。
        # 直接存取 config（禁 fallback）並與協定值硬核對——不一致即代表部署端的 CTU 軸
        # 與訓練/評估用的是兩套刻度，KE 兩窗收斂判準會在不同的窗長上做判定。
        self._delta_ctu = float(config["model"]["delta_ctu"])
        if abs(self._delta_ctu - tes_protocol.NOMINAL_DELTA_CTU_PER_STEP) > 1e-9:
            raise ValueError(
                f"config.model.delta_ctu={self._delta_ctu} 與三態協定的 "
                f"NOMINAL_DELTA_CTU_PER_STEP={tes_protocol.NOMINAL_DELTA_CTU_PER_STEP} 不一致；"
                "step<->CTU 換算只能有一套。"
            )

        # 子 adapter（各自 ckpt-config 驅動）。turb 僅在 enable_turb 時載入
        # （mean-only 不依賴 turb config/ckpt，避免缺檔崩潰）。
        self._mean_ad = TESAdapter(H, W, self._sub_config("mean"), device=device)
        self._turb_ad = (
            TESAdapter(H, W, self._sub_config("turb"), device=device)
            if self._enable_turb else None
        )

        # cold-start IC 生成器：直接映射 steady baseline（幾何→mean 單次 forward）。
        # 無 config.model.ic_generator → None（回 physics=0 冷啟）。
        icg = config["model"].get("ic_generator")
        # ★R3-G3（二輪）：delta_ctu 改直接存取（缺欄位即 KeyError）。
        # 上方已用 config["model"]["delta_ctu"] 讀過同一鍵並與協定核對，此處的
        # `.get(..., 1.0)` 是不可能被用到的死 fallback，字面上仍會誤導維護者。
        self._ic_gen = (
            TESAdapter(
                H, W,
                {"model": {**icg, "delta_ctu": config["model"]["delta_ctu"]}},
                device=device,
            )
            if icg else None
        )
        # IC 生成器要跑幾步（直接存取，禁 fallback：步數直接決定 IC 品質，
        # 靜默代預設值＝部署跑的 IC 與量測過的那一個不同）。
        self._ic_steps = int(icg["ic_steps"]) if icg else 0
        if self._ic_gen is not None:
            print(f"[decompose] IC 生成器已載入：cold-start 先跑 {self._ic_steps} 步產生 mean IC")

        # ★R3-G4：停步判準開關。現行 _two_window_ke 與 ke_anchor 是同一個泛函（KE），
        # 修正器一開就同時動到「被判的量」與「判準」。consec_l2_phy 是與 KE 正交的量。
        # **預設維持 ke_two_window**（不動既有部署數字）；是否切換屬作者裁決
        # （DEEP_AUDIT D5）。
        # 直接存取（禁 fallback）：本 repo 的 config.yaml 已顯式寫入這三個鍵，
        # 缺欄位就該報錯而不是靜默走預設。
        self._stop_metric = str(dcfg["stop_metric"])
        if self._stop_metric not in (STOP_METRIC_KE_TWO_WINDOW, STOP_METRIC_CONSEC_L2):
            raise ValueError(
                f"未知的 decompose.stop_metric={self._stop_metric!r}；僅支援 "
                f"{STOP_METRIC_KE_TWO_WINDOW!r} 與 {STOP_METRIC_CONSEC_L2!r}。"
            )
        self._consec_l2_eps = float(dcfg["consec_l2_eps"])
        # ★R3-G4b（二輪修正）：預設路徑下 active 判準一收斂就 freeze / handoff 並結束
        # 迴圈，因此**只能**觀察到「另一判準是否比 active 更早收斂」；若另一判準較晚，
        # 其 step 永遠是 None。要取得完整並排資料須把 mean 迴圈跑到兩判準都宣告過
        # （或 mean_max_steps），由本開關控制。預設 False = 現行部署行為完全不變。
        self._stop_evidence_full_run = bool(dcfg["stop_evidence_full_run"])
        # ★auto_stop（free-run 實驗開關）：false = 判準照算但不凍結，模型持續推進。
        # 直接存取（禁 fallback）：靜默代預設會讓「跑的行為」與 config 所寫的不同。
        self._auto_stop = bool(dcfg["auto_stop"])
        # free-run 下判準首次宣告收斂只印一次、只存一次 steady。
        self._free_run_declared = False
        # 三態協定模組（consec_rel_l2 等）；mean 與 decompose 兩線共用同一份實作。
        self._protocol = tes_protocol

        self.phase: str = PHASE_MEAN
        self._mean_steps = 0
        self._mean_done = False                         # mean-only：收斂後 True → 停步凍結
        self._ke_curve: list[float] = []
        self._consec_l2_curve: list[float] = []
        # 兩判準各自「首次宣告收斂」的步數；None = 迴圈結束前未宣告過。
        # ★預設（stop_evidence_full_run=False）下，晚於 active 判準者恆為 None——
        # 迴圈在 active 收斂當步就結束了，不是「該判準永不收斂」。
        self.stop_evidence: dict[str, object] = {
            "stop_metric": self._stop_metric,
            "full_run": self._stop_evidence_full_run,
            "ke_two_window_step": None,
            "consec_l2_phy_step": None,
            "consec_l2_eps": self._consec_l2_eps,
            "mean_steps_at_stop": None,
        }
        self._ke_ratio = float("nan")
        self._mean_norm: torch.Tensor | None = None    # 凍結 mean [1,9,H,W] norm
        self._prev_phys: torch.Tensor | None = None    # activity 場用
        self._activity: torch.Tensor | None = None
        self._steady_mean: torch.Tensor | None = None  # warm chain：前次收斂 mean state（root 由 zarr 注入）

    # ------------------------------------------------------------------
    def _sub_config(self, key: str) -> dict:
        sub = dict(self._config["model"].get(key) or {})
        if not sub:
            raise FileNotFoundError(
                f"backend=decompose 需要 config.model.{key}"
                "（tes_run_config + checkpoint）"
            )
        return {
            "model": {**sub, "delta_ctu": self._config["model"]["delta_ctu"]},
        }

    # ------------------------------------------------------------------
    def _gen_ic(self, geo: GeometryInputs) -> torch.Tensor:
        """從 physics=0 跑 IC 生成器 ``ic_steps`` 步 → norm 9ch mean，作 cold-start IC。

        ``ic_steps`` 由 ``config.model.ic_generator.ic_steps`` 指定（直接存取，缺鍵即
        ``KeyError``）。steady 直接映射型 baseline 用 1（忽略 n_steps，單次前向即完整場）；
        AR stepper 型（``baseline_unet_modern``）需多步才推得到發展完成的場——現行部署為
        後者，見 ``config.yaml`` 該欄註解的量測數字。

        逐步呼叫 ``g.step(1)`` 而非單次 ``g.step(ic_steps)``：兩者對 AR baseline 語意相同，
        但逐步版是實際量測 ic_steps 曲線時走的路徑，保持一致以免日後 arch 換代時語意漂移。

        Returns:
            ``[9, H, W]`` norm physics（與 mean 模組同一組 normalizer）。
        """
        g = self._ic_gen
        g.prepare(geo, warm=False)   # cold init（physics=0）+ build cond
        for _ in range(self._ic_steps):
            g.step(1)
        ic = g._state[:, g._sd : g._sd + g._pd].detach().clone()   # [1,9,H,W] norm physics
        return ic[0]                 # [9,H,W]

    def prepare(self, geo: GeometryInputs, warm: bool,
                ic_field: torch.Tensor | None = None) -> None:
        """ic_field：[9,H,W] norm 9ch moments，作 warm IC（root 由 zarr mean_flow 注入）。
        給定時建成完整 mean state → 取代「前次收斂態」當起點（cold-start 用真實場、非零場）。"""
        # fresh cold-start（非 warm、無外部 IC）且有 IC 生成器 → 用直接映射 baseline
        # 生成 mean IC（取代 physics=0），SFA mean 再 warm-refine。★僅在 config 有設
        # model.ic_generator 時成立（self._ic_gen is None 就退回 physics=0 冷啟）。
        if (not warm) and ic_field is None and self._ic_gen is not None:
            ic_field = self._gen_ic(geo)
            warm = True
            self._mean_ad._state = None   # fresh：確保用生成 IC 而非前案 live state
        if ic_field is not None:
            static_n = self._mean_ad._normalize_static(geo)
            ic = ic_field.to(self.device).float()
            if ic.dim() == 3:
                ic = ic.unsqueeze(0)                                   # [1,9,H,W]
            # ★R3（三輪）：warm-IC 的 stage 0 改走三態協定入口（原為裸
            # model.init_state，完全不套 apply_ic_mask）——這是部署端最後一條繞過
            # IC 政策的路徑（DEEP_AUDIT W0.6）。遮罩用**當前幾何**的 mask：此刻
            # self._mean_ad._mask 還是上一 stage 的幾何，不能拿來遮。
            fluid_new = self._mean_ad.fluid_mask_of(static_n)
            self._steady_mean = self._mean_ad._protocol.initial_state(
                self._mean_ad.model, self._mean_ad.run_cfg, static_n, ic, fluid_new,
            )
            warm = True
        # warm chain：在連續拖曳/編輯 (adapt) 時，優先從上一幀的當前狀態 (self._mean_ad._state) 溫啟動；
        # 否則從前次收斂態（或注入的 zarr 場）出發。
        if warm:
            if self._mean_ad._state is not None:
                # 真實跨幾何 handoff（承接自己上一 stage 的收斂輸出）：self._mean_ad._mask 此刻
                # 仍是上一 stage 的幾何（下一行 self._mean_ad.prepare() 才會覆寫）→ 是有效的舊幾何。
                self._mean_ad._steady = self._mean_ad._state.clone()
                self._mean_ad._steady_mask = (
                    self._mean_ad._mask.clone() if self._mean_ad._mask is not None else None
                )
            elif self._steady_mean is not None:
                # ic_field 路徑建出的全新 state（見上）已對「當前幾何」正確初始化，不存在真實
                # 「舊幾何」——steady_mask 明確清 None，關閉 ghost-IC 擴大清零（避免拿無關的
                # 前一鏈殘留幾何來擴大重置區，2026-07-15 v20_ghosthandoff pilot 回歸的根因）。
                self._mean_ad._steady = self._steady_mean
                self._mean_ad._steady_mask = None
        self._mean_ad.prepare(geo, warm=warm and self._mean_ad._steady is not None)
        # turb：cond/mask 先備好，state 等 mean 收斂後才 seed（mean-only 時跳過）
        if self._turb_ad is not None:
            self._turb_ad._mask = self._mean_ad._mask
            self._turb_ad._build_cond(geo)
            self._turb_static = self._turb_ad._normalize_static(geo)

        self.phase = PHASE_MEAN
        self._mean_steps = 0
        self._mean_done = False
        self._ke_curve = []
        self._consec_l2_curve = []                 # ★R3-G4：每次 prepare 重置
        self.stop_evidence["ke_two_window_step"] = None
        self.stop_evidence["consec_l2_phy_step"] = None
        self._free_run_declared = False
        self._ke_ratio = float("nan")
        self._mean_norm = None
        self._prev_phys = None
        self._activity = torch.ones(1, self.H, self.W, device=self.device)

    # ------------------------------------------------------------------
    def _mean_phys_norm(self) -> torch.Tensor:
        ad = self._mean_ad
        return ad._state[:, ad._sd : ad._sd + ad._pd]

    def _ke_norm(self, phys: torch.Tensor) -> float:
        """norm-空間 momentum KE proxy（jx²+jy²，fluid 均值；對齊 02 _ke_norm）。"""
        fm = self._mean_ad._mask
        ke = phys[:, 3].square() + phys[:, 5].square()
        denom = fm.squeeze(1).sum().clamp_min(1.0)
        return float((ke * fm.squeeze(1)).sum().item() / denom.item())

    def _handoff_to_turb(self) -> None:
        """mean 收斂（或達上限）→ 凍結 mean、seed fluct、切 TURB_LIVE。"""
        self._mean_norm = self._mean_phys_norm().detach().clone()
        gen = torch.Generator(device="cpu").manual_seed(self._seed)
        fluct0 = (
            torch.randn(self._mean_norm.shape, generator=gen, dtype=torch.float32)
            .to(self.device) * self._seed_amp
        )
        fluct0 = fluct0 * self._mean_ad._mask                  # solid 區歸零
        self._turb_ad._state = self._turb_ad.model.init_state(self._turb_static, fluct0)
        self.phase = PHASE_TURB
        print(f"[decompose] mean 收斂 handoff → TURB_LIVE"
              f"（steps={self._mean_steps}, ke_ratio={self._ke_ratio:.4f}）")

    def set_auto_stop(self, enabled: bool) -> None:
        """執行期切換停步守門。

        關閉時若已凍結，解除凍結讓模型自收斂態繼續推進（free-run 觀察用）；
        重新開啟時不追溯已跑的步數，下一次判準宣告即凍結。

        Args:
            enabled: True = 判準宣告收斂即停（部署行為）；False = 持續推進。
        """
        self._auto_stop = bool(enabled)
        if self._auto_stop:
            print(f"[decompose] auto_stop=True（判準停步，steps={self._mean_steps}）")
            return
        if self._mean_done:
            self._mean_done = False        # 解凍：自收斂態繼續推進
            self._free_run_declared = True  # 已凍結過＝判準早已宣告，不重複印
        print(f"[decompose] auto_stop=False → free-run 持續推進（steps={self._mean_steps}）")

    def step(self, n: int = 1) -> None:
        for _ in range(max(1, n)):
            self._step_one()

    def _print_stop_evidence(self) -> None:
        """把 :attr:`stop_evidence` 印出一行（R3-G4，D5 裁決依據）。

        原本此屬性只存在於記憶體，沒有任何 print / json 落地，量測完就消失；
        必須有人手動去讀屬性才看得到。此處在 mean 停步時印一行留下痕跡。
        ★注意 ``full_run=False``（預設）時晚於 active 判準者恆為 ``None``——
        代表「迴圈在它宣告前就結束了」，不是「它不會收斂」。
        """
        self.stop_evidence["mean_steps_at_stop"] = self._mean_steps
        print(f"[decompose] stop_evidence={self.stop_evidence}")

    def _freeze_mean(self) -> None:
        """mean 收斂（或達上限）：存 steady 供 warm chain。
        enable_turb → handoff 切 TURB；否則 mean-only 凍結停留 MEAN_CONVERGING。"""
        self._print_stop_evidence()
        self.save_steady()
        if self._enable_turb:
            self._handoff_to_turb()
        else:
            self._mean_done = True
            print(f"[decompose] mean 收斂（turb 停用）→ 凍結停留 MEAN"
                  f"（steps={self._mean_steps}, ke_ratio={self._ke_ratio:.4f}）")

    def _step_one(self) -> None:
        if self.phase == PHASE_MEAN:
            if self._mean_done:        # mean-only 收斂後凍結，不再推進
                return
            prev = self._mean_phys_norm().detach().clone()
            self._mean_ad.step(1)
            phys = self._mean_phys_norm()
            self._ke_curve.append(self._ke_norm(phys.float()))
            self._mean_steps += 1
            # activity 場：per-cell |Δphys| 9ch 均值 → 1=仍在變、0=已定
            with torch.inference_mode():
                delta = (phys - prev).abs().mean(dim=1)        # [1,H,W]
                thr = delta.mean() * 0.5 + 1e-8
                self._activity = (delta > thr).float()
            self._prev_phys = phys
            # ★R3-G4：consec_l2_phy = 相鄰兩步 physics 相對 L2（與 ke_anchor 正交）。
            # 兩判準每步都算，各自記下首次宣告收斂的步數（D5 裁決依據）。
            l2 = self._protocol.consec_rel_l2(phys, prev, self._mean_ad._mask)
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
                # full_run=False（預設）：active 一收斂即停＝現行部署行為。
                # full_run=True：續跑到兩判準都宣告過（或下方 mean_max_steps 上限），
                # 才拿得到「哪個判準比較晚」的那一半資料。
                declared = active >= 1.0 and (
                    not self._stop_evidence_full_run or both_declared
                )
                if declared and self._auto_stop:
                    self._freeze_mean()
                    return
                # free-run 遙測：判準宣告之後每 25 步印一次場的量級，供觀察漂移/發散
                # （KE 兩窗判準對結構性瓦解不敏感，只看 ke_ratio 會漏掉）。
                if (not self._auto_stop and self._free_run_declared
                        and self._mean_steps % 25 == 0):
                    with torch.inference_mode():
                        _f = phys.float()
                        _rms = float(_f.pow(2).mean().sqrt().item())
                        _amax = float(_f.abs().max().item())
                        _finite = bool(torch.isfinite(_f).all().item())
                    print("[decompose] free-run step=%d ke_ratio=%.6f rms=%.4f max|phys|=%.4f"
                          " finite=%s" % (self._mean_steps, self._ke_ratio, _rms, _amax, _finite))
                if declared and not self._free_run_declared:
                    # free-run：不凍結，但把「判準本來會停在哪一步」留成證據，
                    # 並存一次 steady，讓下一次 warm analyze 仍自收斂態接續。
                    self._free_run_declared = True
                    self._print_stop_evidence()
                    self.save_steady()
                    print(f"[decompose] free-run：判準於第 {self._mean_steps} 步宣告收斂"
                          f"（ke_ratio={self._ke_ratio:.6f}），auto_stop=false → 繼續推進")
            if self._auto_stop and self._mean_steps >= self._mean_max_steps:
                print(f"[decompose] WARN mean 未收斂達上限 {self._mean_max_steps} 步"
                      f"（ke_ratio={self._ke_ratio:.4f}）→ 凍結")
                self._freeze_mean()
        else:
            ad = self._turb_ad
            # ★R3（三輪）：本步**不傳 ke_anchor**，理由如下（二輪曾放一段
            # `anchor_kwargs = {} if ad._ke_anchor is None else {...}`，但
            # DecomposeAdapter.prepare() 從不呼叫 self._turb_ad.prepare()——只手動設
            # _mask / _build_cond / _turb_static——而 _ke_anchor 只在 TESAdapter.prepare()
            # 內被賦值，因此該展開恆為 {}，是永遠不會生效的死碼，且會讓 ke_anchor 掃描器
            # 把這個呼叫點誤判成「已帶錨點」。現改為不寫，並在
            # tests/tes/test_forward_ke_anchor_coverage.py 的 ALLOWLIST 具名豁免）：
            #   1. turb arch（turb_perception_a）的 forward 簽章不收 ke_anchor；
            #   2. turb config 的 ke_anchor_mode 為 'off'，錨點語意本就不適用 turb 線。
            # 日後 turb 線要接三態協定，請改成呼叫 self._turb_ad.prepare() 取得真錨點，
            # 而不是把這段死碼放回來。
            with torch.inference_mode():
                ad._state = ad.model(
                    ad._state, ad._cond, ad._mask, 1,
                    return_trajectory=False, g_vec_norm=ad._g_vec_norm,
                    mean_flow_norm=self._mean_norm,
                )

    # ------------------------------------------------------------------
    def phy_field(self) -> torch.Tensor:
        if self.phase == PHASE_MEAN or self._mean_norm is None:
            return self._mean_ad.phy_field()
        ad = self._turb_ad
        fluct = ad._state[:, ad._sd : ad._sd + ad._pd]
        # 嚴格對齊 lbmnca.nca 規格：先反正規化回真實物理空間
        mean_raw = ad.field_norm.denorm_mphy(self._mean_norm[0].float())
        fluct_raw = ad.field_norm.denorm_mphy(fluct[0].float())
        # 在真實物理空間進行雷諾相加 (u = ū + u')，避免非線性 tanh 疊加發散
        composed_raw = mean_raw + fluct_raw
        return composed_raw.contiguous()

    # ------------------------------------------------------------------
    # 分層輸出（雙層 IBFV）——對齊 02 InferenceEngine.field_layers 契約。
    # 回傳 denorm MRT moments（9ch，含 rho/jx/jy）「非」velocity；velocity 由
    # 03 render/field_ops.derive_fields 計算（DRY）。
    # ★三輪：原註解指向「EXECUTION_PLAN §2.3 決策 A」，但本 repo 內唯一的
    # EXECUTION_PLAN（the deployment plan.md）§2 標題是
    # 「補 compute_gvec_from_mask」，沒有這個決策；該指路已失效，故移除路徑、
    # 把結論留在此處。
    # ★四輪把範圍講明（原寫「workspace 內唯一的 03 EXECUTION_PLAN」易被讀成全域唯一）：
    # 2026-07-28 實跑 `find . -name "EXECUTION_PLAN*.md"` 於 workspace 根得 3 個檔——
    # 本 repo 這一份，另兩份在 docs/reviews/DEEP_AUDIT_2026-07/archive/（不同文件族）。
    # ------------------------------------------------------------------
    def mean_field(self) -> torch.Tensor | None:
        """回傳 mean 層 denorm MRT moments（ū 來源）。

        MEAN 期回當前推進中的 mean；TURB 期回凍結的時間平均 mean。
        03 端對其 derive_fields 得 mean 層驅動速度。

        Returns:
            [9, H, W] CUDA float32 denorm MRT moments，或 None（prepare 前）。
        """
        if self._mean_norm is not None:
            # TURB 期：凍結 mean（norm [1,9,H,W]）→ denorm [9,H,W]
            return self._turb_ad.field_norm.denorm_mphy(self._mean_norm[0]).contiguous()
        if self._mean_ad._state is None:
            return None
        # MEAN 期：mean_module 即時 denorm physics（已是 [9,H,W]）
        return self._mean_ad.phy_field()

    def fluct_field(self) -> torch.Tensor | None:
        """回傳 fluct 層 denorm MRT moments（u' 來源）。Phase A stub，恆回 None。

        TODO Phase B：denorm 為非線性 atanh 反變換，故 u' 物理量必須以
        ``denorm(mean + fluct) - denorm(mean)`` 計算，不可直接 ``denorm(fluct)``
        （★三輪：原文指向「EXECUTION_PLAN §2.1 denorm 非線性陷阱」，該節在
        ``docs/reports/EXECUTION_PLAN.md`` 不存在，指路已失效——理由本身就寫在上一句，
        不需要外部文件佐證。）

        Returns:
            None（Phase A 恆 None；Phase B 才回 [9, H, W]）。
        """
        return None

    def field_layers(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """回傳 (mean_moments, fluct_moments)，對齊 02 InferenceEngine.field_layers。

        回傳的是 denorm MRT moments「非」velocity；velocity 由 03 端 derive_fields
        計算（mean 層 = derive(mean)，turb 層 = derive(mean) + derive(fluct)）。

        Returns:
            (mean_moments, fluct_moments)：
              - mean_moments：[9, H, W] CUDA float32 denorm，或 None。
              - fluct_moments：None（Phase A 恆 None）。
        """
        return self.mean_field(), self.fluct_field()

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
        ad = self._turb_ad if self._turb_ad is not None else self._mean_ad
        return ad.dynamic_dim()

    def save_steady(self) -> None:
        """adapt 模式素材：存收斂 mean 的完整 state（warm-transfer 推論）。"""
        if self._mean_ad._state is not None:
            self._steady_mean = self._mean_ad._state.clone()
