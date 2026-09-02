"""TESBackend — TES NCA 推論 backend。

ckpt-config 驅動：模型結構由訓練 run 的 yaml config 決定。
推論端不需寫死通道常數——phy_field() 切片 state[:, sd:sd+pd] 動態計算。

config.yaml（model 段）需求：
    backend_type: "tes"
    tes_run_config: "<run_yaml_path>"
    checkpoint: "<ckpt_path>"
    delta_ctu: 1.0

static 正規化（v3）：
    C0 SDF      → per-case tanh(SDF_px / L_char_h5px)
    C1 upstream → 全域 z-score
    C2/C3 pos   → raw [0,1] identity
"""
from __future__ import annotations

from pathlib import Path

import torch

from lbmnca.nca.inference.geometry_types import GeometryInputs
from lbmnca.nca.tes.config import load_tes_config
from lbmnca.nca.tes.data import runtime_global_stats
from lbmnca.nca.tes.data.normalizer import TESCondNormalizer
from lbmnca.nca.tes.inference import protocol
from lbmnca.nca.tes.models.model_factory import build_tes_model
from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer
from lbmnca.paths import REPO_ROOT

# SDF tanh 尺度 = L_char_sim / _SIM_TO_H5_SCALE。
# ★ SSOT = src/lbmnca/morphospace/physics_proxy/lchar_api.py::H5_DOWNSAMPLE（同值 3）。
# 此處保留本地常數以避免對 morphospace 產生 import 依賴（此模組目前完全不依賴
# morphospace）；但若 H5 降採樣倍率改變，這裡、bridge 端的 tes_adapter、
# src/lbmnca/lbm/post_process/norm_api.py 三處必須同步。
_SIM_TO_H5_SCALE = 3.0


class TESBackend:
    """InferenceBackend for TES NCA（單模型）。"""

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
        self._steady: torch.Tensor | None = None
        # 與 _steady 綁定的舊幾何遮罩（ghost-IC handoff 用），
        # 對齊 bridge 端 TESAdapter；None = 無有效舊幾何。
        self._steady_mask: torch.Tensor | None = None
        self._ke_anchor: torch.Tensor | None = None
        self._load()

    def _load(self) -> None:
        mcfg = self._config["model"]
        run_cfg_rel = mcfg.get("tes_run_config")
        ckpt_rel = mcfg.get("checkpoint")
        if not run_cfg_rel or not ckpt_rel:
            raise FileNotFoundError(
                "backend_type=tes 需要 config.model.tes_run_config 與 checkpoint；"
                "開發期請改 backend_type=dummy。"
            )
        # 相對路徑一律以 repo 根目錄為基準（見 lbmnca.paths）。
        run_cfg_path = Path(run_cfg_rel) if Path(run_cfg_rel).is_absolute() else (
            REPO_ROOT / run_cfg_rel
        ).resolve()
        ckpt_path = Path(ckpt_rel) if Path(ckpt_rel).is_absolute() else (
            REPO_ROOT / ckpt_rel
        ).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TES checkpoint 不存在：{ckpt_path}")

        self.run_cfg = load_tes_config(str(run_cfg_path))
        m = self.run_cfg.model
        self._sd, self._pd, self._hd = int(m.static_dim), int(m.physics_dim), int(m.hidden_dim)
        self._cond_dim = int(m.cond_dim)
        # step<->CTU 換算的單一真相 = lbmnca.nca.tes.inference.protocol。直接存取
        # （缺欄位即 KeyError）＋硬核對：帶 fallback 的取值會讓 config 少欄位或填錯值
        # 都靜默通過，部署端的 CTU 軸因此可能與訓練/評估用的是兩套刻度。
        self._delta_ctu = protocol.assert_delta_ctu(
            mcfg["delta_ctu"], kind="mean", context="config.model.delta_ctu",
        )

        self.model = build_tes_model(m).to(self.device).eval()
        ck = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"[tes_backend] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        print(f"[tes_backend] loaded ckpt step={ck.get('step')} ({ckpt_path.name}), "
              f"arch={m.arch} state_dim={self._sd + self._pd + self._hd}")
        self._warn_protocol_version(ck, ckpt_path)

        self._build_normalizers()

    @staticmethod
    def _warn_protocol_version(ck: dict, ckpt_path: Path) -> None:
        """讀 ckpt 的協定語意版本欄位並印警告，**只警告不中止**。

        ``ke_anchor`` 的來源語意（監督目標 GT mean 的 KE ／ rollout 起始 state 的 KE）
        會決定推論結果，但 ckpt 的身分欄位只記 ``ke_anchor_mode`` / ``gmin`` / ``margin``，
        記不到這個**語意版本**。存檔端由
        ``training/checkpoint/identity.py::collect_inference_hparams`` 寫入
        ``protocol_version``，載入端據以比對。

        欄位不存在的 ckpt 落在 ``recorded is None`` 分支，該分支只 warn 不 raise ——
        無此欄位者無從判定其語意版本，直接中止會讓所有既有 ckpt 都不可載入。
        要讓訓練與推論確定同語意，需以現行協定重訓。

        Args:
            ck: ``torch.load`` 得到的 checkpoint dict。
            ckpt_path: ckpt 路徑（僅供訊息）。
        """
        hp = ((ck.get("identity") or {}).get("inference_hparams") or {})
        recorded = hp.get("protocol_version")
        current = protocol.PROTOCOL_VERSION
        if recorded is None:
            print(f"[tes_backend] WARN ckpt 未記錄 protocol_version（無法辨識其協定版本），"
                  f"當前協定 {current}；"
                  f"訓練與推論的錨點語意可能不同 → {ckpt_path.name}")
        elif str(recorded) != current:
            print(f"[tes_backend] WARN ckpt protocol_version={recorded} 與當前協定 "
                  f"{current} 不符 → {ckpt_path.name}")

    def _build_normalizers(self) -> None:
        cfg = self.run_cfg
        if cfg.stats_policy == "dataset_global_runtime":
            zarr_paths = [str(z) for z in cfg.zarr_paths] if cfg.zarr_paths else [str(cfg.zarr_path)]
            sp = runtime_global_stats.resolve_stats_paths(zarr_paths, cfg.stats_paths)
            gstats = runtime_global_stats.build_runtime_global_stats(
                sp, zarr_paths=zarr_paths, dataset_names=cfg.dataset_names or None
            )
            self.cond_normalizer = TESCondNormalizer.from_stats_dict(gstats)
            self.field_norm = FieldNormalizer.from_stats_dict(gstats).to(self.device)
        else:
            self.cond_normalizer = TESCondNormalizer.from_stats_json(cfg.stats_path)
            self.field_norm = FieldNormalizer.from_file(cfg.stats_path).to(self.device)

    def _normalize_static(self, geo: GeometryInputs) -> torch.Tensor:
        st = geo.static_raw.unsqueeze(0).to(self.device).float()
        # 禁 fallback：SDF 的 tanh 尺度與 p_vec 的 Re 條件都建立在 L_char 上，
        # 缺失或為 0 時靜默代入預設值等於無聲送出錯誤條件，故一律直接報錯。
        if "L_char" not in geo.cond_raw or not geo.cond_raw["L_char"]:
            raise KeyError(
                "cond_raw 缺少有效的 'L_char'；SDF tanh 尺度與 Re 條件皆依賴它，"
                "不得以預設值代替。"
            )
        l_char_sim = float(geo.cond_raw["L_char"])
        l_char_h5 = max(1.0, l_char_sim / _SIM_TO_H5_SCALE)
        out = st.clone()
        out[:, 0] = torch.tanh(st[:, 0] / l_char_h5)
        out[:, 1] = self.field_norm.norm_upstream(st[:, 1])
        return out

    def _build_cond(self, geo: GeometryInputs) -> None:
        cr = geo.cond_raw
        g_vec = torch.tensor(cr["g_vec"], dtype=torch.float32, device=self.device).unsqueeze(0)
        p_vec = torch.tensor(cr["p_vec"], dtype=torch.float32, device=self.device).unsqueeze(0)
        self._cond = self.cond_normalizer.build_cond(self._cond_dim, g_vec, p_vec)
        m = self.run_cfg.model
        if m.gvec_mode in ("additive", "gated") and m.g_vec_dim > 0:
            self._g_vec_norm = self.cond_normalizer.normalize_g(g_vec)
        else:
            self._g_vec_norm = None

    def prepare(self, geo: GeometryInputs, warm: bool) -> None:
        static = self._normalize_static(geo)
        self._mask = (static[:, 0:1] > 0.0).float()
        self._build_cond(geo)

        # IC / 交接 / KE 錨點走 mean 線三態共用實作（tes.inference.protocol；
        # 該模組自承不是全 workspace 唯一入口，覆蓋界線見其模組 docstring 的
        # 「誠實界線」段——turb 線與訓練層都不在其中）。
        if warm and self._steady is not None:
            self._state = protocol.chain_handoff(
                self.model, self.run_cfg, self._steady.clone(), static,
                self._mask, self._steady_mask,
            )
        else:
            self._state = protocol.initial_state(
                self.model, self.run_cfg, static, None, self._mask)
        self._ke_anchor = protocol.resolve_ke_anchor(
            self.run_cfg, self._state, self._mask)

    def step(self, n: int = 1) -> None:
        assert self._state is not None, "call prepare() first"
        with torch.inference_mode():
            self._state = self.model(
                self._state, self._cond, self._mask, max(1, n),
                return_trajectory=False, g_vec_norm=self._g_vec_norm,
                ke_anchor=self._ke_anchor,
            )

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


__all__ = ["TESBackend"]
