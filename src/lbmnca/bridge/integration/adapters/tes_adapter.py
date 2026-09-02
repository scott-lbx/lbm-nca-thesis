"""TESAdapter — lbmnca.nca 真 TES NCA 接入 ModelBackend（UI-01/UI-07 修）。

ckpt-config 驅動：模型結構（arch/state_dim/hidden/cond_dim/gvec_mode）全部
由 02 run 的 yaml config 決定，03 端不寫死通道常數——`phy_field()` 切片
`state[:, sd:sd+pd]` 動態計算（解 UI-07：13ch baseline 與 29ch LC run 通吃）。

config.yaml（model 段）需求：
    backend: "tes"
    tes_run_config: "${REPO_ROOT}/configs/nca/main.yaml"
    checkpoint: "${CHECKPOINT_ROOT}/<series>/<arm>.pth"
    delta_ctu: 1.0        # 推論步距（CTU；C-3 post-conv 慣例）

static 正規化（v3，對齊 01 tes_h5_to_zarr Pass2）：
    C0 SDF      → per-case tanh(SDF_px / L_char_h5px)，L_char_h5px = L_char_simpx / 3
    C1 upstream → 全域 z-score（stats json）
    C2/C3 pos   → raw [0,1] identity
"""
from __future__ import annotations

from pathlib import Path

import torch

from lbmnca.bridge.integration.model_backend import GeometryInputs
from lbmnca.nca.tes import config as tes_config
from lbmnca.nca.tes.data import normalizer as tes_normalizer
from lbmnca.nca.tes.data import runtime_global_stats as tes_runtime_global_stats
from lbmnca.nca.tes.inference import protocol as tes_protocol
from lbmnca.nca.tes.models.model_factory import build_tes_model
from lbmnca.nca.tes.training.checkpoint import identity as tes_identity
from lbmnca.nca.tes.utils import spatial_ops as tes_spatial_ops
from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer
from lbmnca.paths import REPO_ROOT, resolve_placeholders

# 01 tes_h5_to_zarr：ROI 1920×1080 → H5 640×360 = 1/3；p_vec L_char 為 sim px
# SDF tanh 尺度 = L_char_sim / _SIM_TO_H5_SCALE。
# ★ SSOT = lbmnca.morphospace.physics_proxy.lchar_api::H5_DOWNSAMPLE（同值 3）。
# 若 H5 降採樣倍率改變，此處、lbmnca.nca tes_backend、lbmnca.lbm norm_api 三處必須同步。
_SIM_TO_H5_SCALE = 3.0


class TESAdapter:
    """ModelBackend for lbmnca.nca TES NCA（單模型；decompose 雙模組另見 DecomposeAdapter）。"""

    def __init__(self, H: int, W: int, config: dict, device: str = "cuda") -> None:
        self.H = H
        self.W = W
        self.phy_channels = 9
        self.has_convergence = False
        self.device = torch.device(device)
        self._config = config

        self._state: torch.Tensor | None = None
        self._cond: torch.Tensor | None = None
        self._g_vec_norm: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None
        # ★DEEP_AUDIT 3.6：舊版此處為跨幾何跨鏈**永不重置**的 running-min，輸出因此取決於
        # 本進程先前評過哪些幾何——連「輸入的函式」都不是，G3-6 不可能成立。現改為每次
        # prepare() 由 02 端 tes.inference.protocol.resolve_ke_anchor 從本次 rollout 起始
        # state 重新產生（訓練 / 評估 / 部署共用同一實作）。
        self._ke_anchor: torch.Tensor | None = None
        self._steady: torch.Tensor | None = None
        # ★ghost-IC handoff：與 _steady 綁定的「該 steady 存下當下」的 fluid mask，非「上一次
        # prepare() 剛好留下的 self._mask」（兩者在 DecomposeAdapter fresh ic_field 分支下不同一
        # 回事——None 表示無有效舊幾何可比對，見 GHOST_IC_HANDOFF_FORENSIC_2026-07-15.md 修正記錄）。
        self._steady_mask: torch.Tensor | None = None
        # inlet-BC injection（deploy 每步鉗入口 K 欄動量=第 K 欄，消 boundary undershoot band）。
        # env-var gated，預設 0=off（零 frontend 行為變更）；實驗用 SFA_INLET_BC_COLS 開。
        import os as _os
        self._inlet_bc_cols = int(_os.environ.get("SFA_INLET_BC_COLS", "0"))
        self._load(REPO_ROOT)

    # ------------------------------------------------------------------
    def _resolve_path(self, rel: str, base: Path) -> Path:
        """解析 config 路徑，展開 ``lbmnca.paths`` 佔位符（``${CHECKPOINT_ROOT}`` 等）。

        Args:
            rel: config 中的路徑字串，可含 ``${CHECKPOINT_ROOT}`` / ``${REPO_ROOT}``
                等前綴（見 ``lbmnca.paths.resolve_placeholders``）。
            base: repo 根目錄（無佔位符時，相對路徑的基準）。

        Returns:
            解析後的絕對路徑。
        """
        rel = resolve_placeholders(rel)
        p = Path(rel)
        return p if p.is_absolute() else (base / p).resolve()

    # ------------------------------------------------------------------
    def _load(self, base: Path) -> None:
        mcfg = self._config["model"]
        run_cfg_rel = mcfg.get("tes_run_config")
        ckpt_rel = mcfg.get("checkpoint")
        if not run_cfg_rel or not ckpt_rel:
            raise FileNotFoundError(
                "backend=tes 需要 config.model.tes_run_config（02 run 的 yaml）"
                "與 config.model.checkpoint；開發期請改 backend=dummy。"
            )
        run_cfg_path = self._resolve_path(run_cfg_rel, base)
        ckpt_path = self._resolve_path(ckpt_rel, base)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TES checkpoint 不存在：{ckpt_path}")

        # 三態推論協定（IC mask / chain 交接 / KE 錨點 / step<->CTU）統一由
        # tes.inference.protocol 提供，本 adapter 不自行實作。
        self._protocol = tes_protocol
        self.run_cfg = tes_config.load_tes_config(str(run_cfg_path))
        m = self.run_cfg.model
        self._sd, self._pd, self._hd = int(m.static_dim), int(m.physics_dim), int(m.hidden_dim)
        self._cond_dim = int(m.cond_dim)
        # ★R3-G3（2026-07-28 二輪）：本檔是**實際部署路徑**（backend="tes" 單模型，
        # 以及 DecomposeAdapter 的 mean/turb 子 adapter），先前只有 02 的離線鏡像
        # src/inference/backends/tes_backend.py 做了硬核對，此處仍是 .get(...,1.0)：
        # config 少欄位或填成 0.2 都會靜默通過，部署端的 CTU 軸因此可能與訓練/評估
        # 用的是兩套刻度（KE 兩窗這類以 CTU 定窗長的判準會在不同窗長上判定）。
        # 改為直接存取（缺欄位即 KeyError）＋ protocol 硬核對。
        self._delta_ctu = self._protocol.assert_delta_ctu(
            mcfg["delta_ctu"], kind="mean", context="config.model.<sub>.delta_ctu",
        )

        self.model = build_tes_model(m).to(self.device).eval()
        ck = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"[tes_adapter] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[tes_adapter] loaded ckpt step={ck.get('step')} ({ckpt_path.name}), "
              f"arch={m.arch} state_dim={self._sd + self._pd + self._hd}")

        self._build_normalizers()
        self._verify_ckpt_identity(ck, ckpt_path, run_cfg_path)
        self._warn_protocol_version(ck, ckpt_path)

    def _warn_protocol_version(self, ck: dict, ckpt_path: Path) -> None:
        """讀 ckpt 的協定語意版本欄位並印警告，**只警告不中止**（R3-G6）。

        與 02 端 ``src/inference/backends/tes_backend.py::_warn_protocol_version``
        同一套判斷（兩端共用 ``tes.inference.protocol.PROTOCOL_VERSION``）。

        ★現況（2026-07-28 四輪覆核，**推翻二／三輪的相反記載**）：存檔端
        已接線——02 的 ``training/checkpoint/identity.py::collect_inference_hparams``
        末尾 ``out.update(protocol_version_record())``。版本比對分支對接線後新存的
        ckpt 是可達的。

        仍然成立的部分：接線之前存下的 ckpt（含現行部署用的
        ``main_v4``）無此欄位，落在「未記錄」分支且無法區分 v0/v1。

        Args:
            ck: ``torch.load`` 得到的 checkpoint dict。
            ckpt_path: ckpt 路徑（僅供訊息）。
        """
        hp = ((ck.get("identity") or {}).get("inference_hparams") or {})
        recorded = hp.get("protocol_version")
        current = self._protocol.PROTOCOL_VERSION
        if recorded is None:
            print(f"[tes_adapter] WARN ckpt 未記錄 protocol_version（存檔端接線前存下的"
                  f"舊 ckpt → 無法辨識 v0/v1），當前協定 {current}；"
                  f"訓練與推論的 ke_anchor 語意可能不同 → {ckpt_path.name}")
        elif str(recorded) != current:
            print(f"[tes_adapter] WARN ckpt protocol_version={recorded} 與當前協定 "
                  f"{current} 不符 → {ckpt_path.name}")

    def _verify_ckpt_identity(self, ck: dict, ckpt_path: Path, run_cfg_path: Path) -> None:
        """比對 ckpt 內嵌身分與本次載入的 ``tes_run_config``；不一致大聲報警。

        ★這是 psi_lowpass 0.8/0.9 事故（``docs/SSOT/MODEL_CKPT_SSOT.md`` §1.2.4）
        的程式層防線：同一顆 v28 ckpt 曾被以 ``configs/LBM_NCA-8/main_v4.yaml``
        （``psi_lowpass_sigma=0.9``，訓練值 0.8）re-gate，得到錯誤的 band_s9 1.315
        並一度被標為 canonical。``psi_lowpass_sigma`` 是 inference-time 算子參數，
        不在 ``state_dict`` 內，strict=False 也攔不到。

        行為：
          * ckpt 無身分資訊（舊 ckpt）→ 印出明確警告，**不中斷**（向後相容）。
          * 有身分且不一致 → 逐欄印出「訓練值 vs 本次載入值」。
            環境變數 ``SFA_CKPT_IDENTITY_STRICT=1`` 時改為 ``ValueError`` 中止。

        Args:
            ck: 已 ``torch.load`` 的 checkpoint dict。
            ckpt_path: checkpoint 路徑。
            run_cfg_path: 本次使用的 02 run yaml 路徑。
        """
        import os as _os

        strict = _os.environ.get("SFA_CKPT_IDENTITY_STRICT", "0") == "1"
        if tes_identity.extract_identity(ck) is None:
            print(
                "\033[93m[tes_adapter] WARN 此 checkpoint **無身分資訊**"
                f"（pre-identity ckpt：{ckpt_path.name}）——無法自動確認 "
                f"{run_cfg_path.name} 就是訓練它的 config。"
                "推論敏感超參數（psi_lowpass_sigma / ke_anchor_mode / apply_ic_mask）"
                "請人工對照 SSOT。\033[0m"
            )
            return

        report = tes_identity.verify_identity(
            ck,
            self.run_cfg,
            field_normalizer=self.field_norm,
            ckpt_path=ckpt_path,
            on_mismatch="warn",
            context="tes_adapter",
        )
        if report["mismatches"] or report["denorm"].get("status") == "mismatch":
            print(f"\033[91m{report['summary']}\033[0m")
            print(
                f"\033[91m[tes_adapter] 本次使用的 arch config = {run_cfg_path}\033[0m"
            )
            if strict:
                raise ValueError(report["summary"])
            print(
                "\033[91m[tes_adapter] （設 SFA_CKPT_IDENTITY_STRICT=1 可讓此情況直接中止）\033[0m"
            )
        else:
            print(f"[tes_adapter] {report['summary']}")

    def _build_normalizers(self) -> None:
        """global/per-zarr stats → cond + field normalizers（對齊 tes eval CLI 慣例）。"""
        cfg = self.run_cfg
        # stats_override：明確 optional（03 config.yaml 顯式指定才生效）——
        # 當 run config 的 stats/zarr 依賴已被 dataset regen 刪除時，載入預先
        # 合併好的 global stats json（tools/build_stats_proxy.py 產出）代替。
        # ── SSOT 烘檔（優先）：換機部署路徑 ────────────────────────────
        # `stats_policy: dataset_global_runtime` 的下方分支會呼叫
        # `build_runtime_global_stats(..., zarr_paths=...)`，而該函式在有
        # zarr_paths 時要 `_case_count(zp)` **開 zarr** 數 case 數做加權合併。
        # 沒有訓練資料集的機器做不到；而改傳 zarr_paths=None 會退成等權重，
        # 得到與訓練時**不同**的正規化常數（已實測兩者不等）。
        # 故改讀在有 zarr 的機器上算好、逐位元相同的合併結果。
        # 這與下方 stats_override 的差別是：那個是 proxy 統計（近似），
        # 這個是同一函式的真實輸出，只是預先算好。
        gstats_rel = self._config["model"].get("global_stats")
        if gstats_rel:
            import json as _json
            gp = self._resolve_path(gstats_rel, REPO_ROOT)
            if not gp.is_file():
                raise FileNotFoundError(f"global_stats 不存在：{gp}")
            with open(gp, encoding="utf-8") as f:
                gstats = _json.load(f)
            baked = gstats.get("_baked") or {}
            print(f"[tes_adapter] global_stats 烘檔：{gp.name}"
                  f"（{len(baked.get('stats_paths') or [])} 份 stats 合併，不需 zarr）")
            self.cond_normalizer = tes_normalizer.TESCondNormalizer.from_stats_dict(gstats)
            self.field_norm = FieldNormalizer.from_stats_dict(gstats).to(self.device)
            return

        override_rel = self._config["model"].get("stats_override")
        if override_rel:
            import json as _json
            p = Path(override_rel)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            with open(p, encoding="utf-8") as f:
                gstats = _json.load(f)
            print(f"\033[93m[tes_adapter] WARN stats_override 生效：{p.name}"
                  f"（proxy 統計，非訓練原始 stats——見檔內 proxy_note）\033[0m")
            self.cond_normalizer = tes_normalizer.TESCondNormalizer.from_stats_dict(gstats)
            self.field_norm = FieldNormalizer.from_stats_dict(gstats).to(self.device)
            return
        if cfg.stats_policy == "dataset_global_runtime":
            zarr_paths = [str(z) for z in cfg.zarr_paths] if cfg.zarr_paths else [str(cfg.zarr_path)]
            sp = tes_runtime_global_stats.resolve_stats_paths(zarr_paths, cfg.stats_paths)
            gstats = tes_runtime_global_stats.build_runtime_global_stats(
                sp, zarr_paths=zarr_paths, dataset_names=cfg.dataset_names or None
            )
            self.cond_normalizer = tes_normalizer.TESCondNormalizer.from_stats_dict(gstats)
            self.field_norm = FieldNormalizer.from_stats_dict(gstats).to(self.device)
        else:
            self.cond_normalizer = tes_normalizer.TESCondNormalizer.from_stats_json(cfg.stats_path)
            self.field_norm = FieldNormalizer.from_file(cfg.stats_path).to(self.device)

    # ------------------------------------------------------------------
    def _normalize_static(self, geo: GeometryInputs) -> torch.Tensor:
        """raw static（SDF px / upstream 0-1 / pos 0-1）→ v3 norm static [1,4,H,W]。"""
        st = geo.static_raw.unsqueeze(0).to(self.device).float()
        # ★ 禁 fallback（2026-07-26）：原為 `.get("L_char") or 64.0`，L_char 缺失或為 0 時
        # 會靜默改用 64 —— SDF tanh 尺度與 Re 條件皆依賴它，靜默代入＝無聲送錯條件。
        if "L_char" not in geo.cond_raw or not geo.cond_raw["L_char"]:
            raise KeyError(
                "cond_raw 缺少有效的 'L_char'；SDF tanh 尺度與 Re 條件皆依賴它，"
                "不得以預設值代替（見 DEEP_AUDIT_2026-07 / FIX_B_Lchar統一.md）"
            )
        l_char_sim = float(geo.cond_raw["L_char"])
        l_char_h5 = max(1.0, l_char_sim / _SIM_TO_H5_SCALE)
        out = st.clone()
        out[:, 0] = torch.tanh(st[:, 0] / l_char_h5)           # v3 tanh_lchar（per-case）
        out[:, 1] = self.field_norm.norm_upstream(st[:, 1])    # 全域 z-score
        # C2/C3 pos：raw_01 → identity（FieldNormalizer norm_pos_* 即 passthrough）
        return out

    def _build_cond(self, geo: GeometryInputs) -> None:
        cr = geo.cond_raw
        g_vec = torch.tensor(cr["g_vec"], dtype=torch.float32, device=self.device).unsqueeze(0)
        p_vec = torch.tensor(cr["p_vec"], dtype=torch.float32, device=self.device).unsqueeze(0)
        # ★2026-08-05 起 02 端 build_cond 只支援 17D（cond_17D = G_vec_norm(14) + P_vec_norm(3)）。
        # 原先多傳的兩個 None 是 delta_ctu / cumulative_ctu，該兩維已隨「mean flow 是時均場，
        # 時間維語義上不存在」的裁決移除，函式簽章縮為 (cond_dim, g_vec, p_vec)。
        self._cond = self.cond_normalizer.build_cond(self._cond_dim, g_vec, p_vec)
        m = self.run_cfg.model
        if m.gvec_mode in ("additive", "gated") and m.g_vec_dim > 0:
            self._g_vec_norm = self.cond_normalizer.normalize_g(g_vec)
        else:
            self._g_vec_norm = None

    # ------------------------------------------------------------------
    def fluid_mask_of(self, static_n: torch.Tensor) -> torch.Tensor:
        """由 norm static 算流體遮罩，供本 adapter 與 DecomposeAdapter 共用。

        v3 ``tanh_lchar`` 正規化下 SDF=0 即障礙物邊界，故門檻恆為 0.0。抽成方法是為了讓
        DecomposeAdapter 的 warm-IC 路徑能在 :meth:`prepare` 尚未覆寫 ``self._mask``
        之前，就拿到**當前幾何**的遮罩去套 ``apply_ic_mask``（見該檔 ``prepare()``）。

        Args:
            static_n: 已正規化的 static ``[1, 4, H, W]``。

        Returns:
            流體遮罩 ``[1, 1, H, W]``，1=流體。
        """
        return (static_n[:, 0:1] > 0.0).float()

    def prepare(self, geo: GeometryInputs, warm: bool) -> None:
        static = self._normalize_static(geo)
        self._mask = self.fluid_mask_of(static)
        self._build_cond(geo)

        if warm and self._steady is not None:
            # ★ghost-IC handoff 修復（docs/audit/0714/GHOST_IC_HANDOFF_FORENSIC_2026-07-15.md）：
            # 重置區從「僅新幾何固體」擴大為「新固體∪舊固體∪變動邊界膨脹」，鏡射訓練端
            # mean_step.py::run_chain（spatial_ops.handoff_reset_mask）。dilate_px=0 時零行為變更。
            # ★用 self._steady_mask（與 _steady 綁定存下的舊幾何遮罩），非「上次 prepare() 剛好
            # 留下的 self._mask」——DecomposeAdapter 的 fresh ic_field 分支下兩者不同一回事，
            # 誤用會把「全新 GT 注入」誤判成「跨幾何 warm handoff」，拿無關的前一鏈殘留幾何來
            # 擴大清零範圍，污染本應乾淨的 IC（2026-07-15 v20_ghosthandoff pilot 回歸，已修正）。
            reset = tes_spatial_ops.handoff_reset_mask(
                self._mask, self._steady_mask, int(self.run_cfg.mean_chain_handoff_dilate_px)
            )
            state = self._protocol.chain_handoff(
                self.model, self.run_cfg, self._steady.clone(), static,
                self._mask, self._steady_mask,
            )
            self._state = state
        else:
            self._state = self._protocol.initial_state(
                self.model, self.run_cfg, static, None, self._mask)   # cold：physics=0
        # KE ceiling 錨點：走 02 端 protocol.resolve_ke_anchor（mean 線三態共用的那一份），
        # 每次 prepare 重算（不跨 rollout 累積）。
        # ke_anchor_mode='off' 時回 None → arch ceiling 分支完全 bypass。
        self._ke_anchor = self._protocol.resolve_ke_anchor(
            self.run_cfg, self._state, self._mask
        )

    def _ke_of(self, state: torch.Tensor) -> torch.Tensor:
        """fluid-masked momentum-KE [B]（norm space，同 arch ceiling 定義）。"""
        phys = state[:, self._sd : self._sd + self._pd].float()
        fm = self._mask.float()
        denom = fm.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return ((phys[:, 3:4] ** 2 + phys[:, 5:6] ** 2) * fm).sum(dim=(1, 2, 3)) / denom

    def step(self, n: int = 1) -> None:
        assert self._state is not None, "call prepare() first"
        with torch.inference_mode():
            self._state = self.model(
                self._state, self._cond, self._mask, max(1, n),
                return_trajectory=False, g_vec_norm=self._g_vec_norm,
                ke_anchor=self._ke_anchor,
            )
            # inlet-BC injection：x<K 欄動量 = 第 K 欄（零-gradient inflow BC；消 boundary undershoot band）
            K = self._inlet_bc_cols
            if K > 0:
                jx, jy = self._sd + 3, self._sd + 5
                self._state[:, jx, :, :K] = self._state[:, jx, :, K:K + 1]
                self._state[:, jy, :, :K] = self._state[:, jy, :, K:K + 1]

    def phy_field(self) -> torch.Tensor:
        phy_norm = self._state[0, self._sd : self._sd + self._pd]
        return self.field_norm.denorm_mphy(phy_norm).contiguous()

    def convergence_field(self) -> torch.Tensor | None:
        return None

    def is_diverged(self) -> bool:
        if self._state is None:
            return False
        phy = self._state[0, self._sd : self._sd + self._pd]
        return bool(torch.isnan(phy).any() or phy.abs().max() > 1e4)

    def dynamic_dim(self) -> int:
        return self._pd + self._hd

    def save_steady(self) -> None:
        if self._state is not None:
            self._steady = self._state.clone()
            self._steady_mask = self._mask.clone() if self._mask is not None else None
