"""InferenceEngine — model-agnostic 推論引擎（推論的唯一入口）。

from_config(config_path) 讀 yaml → 自動 dispatch backend（tes/decompose/dummy）。

Config yaml 結構：
    model:
      backend_type: "tes" | "decompose" | "dummy"
      tes_run_config: "<path>"        # tes 用
      checkpoint: "<path>"           # tes 用
      mean: {tes_run_config, checkpoint}  # decompose 用
      turb: {tes_run_config, checkpoint}  # decompose 用
      delta_ctu: 1.0
      decompose: {mean_max_steps, ...}    # decompose 可選
      hidden_channels: 16                # dummy 可選
    resolution:
      H: 360
      W: 640
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from lbmnca.nca.inference.geometry_types import GeometryInputs
from lbmnca.paths import REPO_ROOT, resolve_placeholders

if TYPE_CHECKING:
    from lbmnca.nca.inference.backends.base import InferenceBackend

_PHASE_MEAN = "mean"
_PHASE_TURB = "turb"


def _expand(node):
    """遞迴展開 config 樹中所有字串的 ``${...}`` 路徑佔位符。

    Args:
        node: yaml 解析結果的任一節點（dict / list / str / 其他純量）。

    Returns:
        同結構的節點，其中所有字串已展開；非字串純量原樣回傳。
    """
    if isinstance(node, str):
        return resolve_placeholders(node)
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return node


class InferenceEngine:
    """Model-agnostic inference session，推論的唯一入口。"""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._backend: "InferenceBackend" = self._build_backend(config)
        self._prepared = False

    @classmethod
    def from_config(cls, config_path: str | Path) -> "InferenceEngine":
        """讀 yaml config → 建立 InferenceEngine（自動 dispatch backend）。

        config 內的 ``${DATA_ROOT}`` / ``${ZARR_ROOT}`` / ``${CHECKPOINT_ROOT}`` /
        ``${REPO_ROOT}`` 佔位符由 ``lbmnca.paths`` 展開，故 yaml 不含機器相關路徑。

        Args:
            config_path: yaml 設定檔路徑（絕對，或相對於 repo 根目錄）。
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML 必要：pip install pyyaml")

        path = Path(config_path)
        if not path.is_absolute():
            path = (REPO_ROOT / config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"config 不存在：{path}")

        with open(path, encoding="utf-8") as f:
            config = _expand(yaml.safe_load(f))
        return cls(config)

    @staticmethod
    def _build_backend(config: dict) -> "InferenceBackend":
        backend_type = config["model"]["backend_type"]
        H = config["resolution"]["H"]
        W = config["resolution"]["W"]
        device = config["model"].get("device", "cuda")

        if backend_type == "dummy":
            from lbmnca.nca.inference.backends.dummy_backend import DummyBackend
            return DummyBackend(
                H, W, device=device,
                hidden_channels=int(config["model"].get("hidden_channels", 16)),
            )
        if backend_type == "tes":
            from lbmnca.nca.inference.backends.tes_backend import TESBackend
            return TESBackend(H, W, config, device=device)
        if backend_type == "decompose":
            from lbmnca.nca.inference.backends.decompose_backend import DecomposeBackend
            return DecomposeBackend(H, W, config, device=device)
        raise ValueError(f"Unknown backend_type: {backend_type!r}")

    # ------------------------------------------------------------------
    @property
    def sim_phase(self) -> str:
        """"IDLE" | "MEAN_CONVERGING" | "TURB_LIVE" | "COMPLETED"。"""
        if not self._prepared:
            return "IDLE"
        phase = getattr(self._backend, "phase", None)
        if phase is None:
            return "TURB_LIVE"
        if phase == _PHASE_MEAN:
            return "MEAN_CONVERGING"
        if phase == _PHASE_TURB:
            return "TURB_LIVE"
        return "IDLE"

    def prepare(self, geo: GeometryInputs, warm: bool = False) -> None:
        """幾何/條件改變時呼叫。warm=True：從前次收斂態出發（adapt 模式）。"""
        self._backend.prepare(geo, warm)
        self._prepared = True

    def step(self, n: int = 1) -> None:
        """推進 n 步，狀態留 GPU。decompose 模式內部管理狀態機切換。"""
        self._backend.step(n)

    def phy_field(self) -> torch.Tensor:
        """[9, H, W] CUDA float32，denorm 物理單位（MRT moments）。"""
        return self._backend.phy_field()

    def field_layers(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """分離的 (mean_moments, fluct_moments)，供前端雙層 IBFV 渲染。

        回傳的是 denorm MRT moments（9ch，含 rho/j 等）「非」velocity；
        velocity（u = j / rho）由前端的 render/field_ops.derive_fields 計算，
        mean 層 vel = derive(mean)，turb 層 vel = derive(mean) + derive(fluct)。

        僅 DecomposeBackend 提供分層；其餘 backend（tes/dummy）無分離概念，
        回 (None, None)。Phase A 的 fluct 一律 None（見 DecomposeBackend.fluct_field）。

        Returns:
            (mean_moments, fluct_moments)：
              - mean_moments：[9, H, W] CUDA float32 denorm，或 None。
              - fluct_moments：[9, H, W] CUDA float32 denorm，或 None（Phase A 恆 None）。
        """
        mean_fn = getattr(self._backend, "mean_field", None)
        fluct_fn = getattr(self._backend, "fluct_field", None)
        mean = mean_fn() if callable(mean_fn) else None
        fluct = fluct_fn() if callable(fluct_fn) else None
        return mean, fluct

    def convergence_field(self) -> torch.Tensor | None:
        """[1, H, W] MEAN_CONVERGING 期有效；其他 None。"""
        return self._backend.convergence_field()

    def is_diverged(self) -> bool:
        """NaN / 爆炸偵測。"""
        return self._backend.is_diverged()

    def dynamic_dim(self) -> int:
        """phy + hidden 通道數，用於 warm-start IC masking。"""
        return self._backend.dynamic_dim()

    # backward compat: expose has_convergence for overlay check
    @property
    def has_convergence(self) -> bool:
        return self._backend.has_convergence

    @property
    def H(self) -> int:
        return self._backend.H

    @property
    def W(self) -> int:
        return self._backend.W


__all__ = ["InferenceEngine"]
