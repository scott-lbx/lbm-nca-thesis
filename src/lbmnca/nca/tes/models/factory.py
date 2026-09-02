"""TES 模型架構工廠（新路徑）：依 config.arch 字串建構對應模型實例。

本模組為 model_factory.py 的後繼版本，import 路由已更新為新組裝層目錄：
  - 核心架構：models/assembly/
  - H-variants：models/assembly/h_variants.py
  - Transport variants：models/assembly/transport_variants.py
  - Baselines：models/baselines/（維持原路徑）

舊路徑 ``models/model_factory.py`` 已改為向後相容 shim，重匯出本模組符號。
"""
from __future__ import annotations

import logging

import torch.nn as nn

from ..config import TESModelConfig

logger = logging.getLogger(__name__)


def build_tes_model(cfg: TESModelConfig) -> nn.Module:
    """依 ``cfg.arch`` 建構 TES 模型，並套用 Dy-C update_clip（若 >0）。

    Args:
        cfg: TESModelConfig，包含架構名稱與所有超參數。

    Returns:
        已初始化的 nn.Module 模型實例。
    """
    model: nn.Module = _build_arch(cfg)
    clip: float = float(cfg.update_clip or 0.0)
    if clip > 0.0:
        from .update.markov_cell import MarkovNCACell

        n: int = 0
        for m in model.modules():
            if isinstance(m, MarkovNCACell):
                m.update_clip = clip
                n += 1
        logger.info("[Dy-C] update_clip=%.3f 套用至 %d 個 MarkovNCACell", clip, n)
    # ★stochastic update（Growing NCA）：train-only per-cell Bernoulli(rate)；1.0=全同步（現行）。
    stoch: float = float(cfg.stoch_update_rate)
    if stoch < 1.0:
        from .update.markov_cell import MarkovNCACell

        ns: int = 0
        for m in model.modules():
            if isinstance(m, MarkovNCACell):
                m.stoch_update_rate = stoch
                ns += 1
        logger.info("[StochUpdate] stoch_update_rate=%.3f 套用至 %d 個 MarkovNCACell（train-only）", stoch, ns)
    return model


def _build_arch(cfg: TESModelConfig) -> nn.Module:
    """依 ``cfg.arch`` 字串路由至對應架構的建構式。

    路由優先順序：
      flat / h_* 系列 → models/assembly/ 及 models/assembly/h_variants.py
      h3t_* / operator_gated_* → models/assembly/transport_variants.py
      baseline_* → models/baselines/（路徑維持不變）

    Args:
        cfg: TESModelConfig，arch 欄位決定路由目標。

    Returns:
        已初始化的 nn.Module 模型實例。

    Raises:
        ValueError: 當 ``cfg.arch`` 不符合任何已知架構名稱時。
    """
    # ── 核心平架構 ──────────────────────────────────────────────────────────
    if cfg.arch == "flat":
        from .assembly import TESNCA

        return TESNCA(cfg)

    # ── H-variants（多尺度階層架構）────────────────────────────────────────
    if cfg.arch == "h_shared_scale":
        from .assembly.h_variants import HSharedScaleNCA

        return HSharedScaleNCA(cfg)
    if cfg.arch == "h_lapres":
        from .assembly.h_variants import HLapResNCA

        return HLapResNCA(cfg)
    if cfg.arch == "h_parent_leaf":
        from .assembly.h_variants import HParentLeafNCA

        return HParentLeafNCA(cfg)
    if cfg.arch == "h_multirate":
        from .assembly.h_variants import HMultirateNCA

        return HMultirateNCA(cfg)
    if cfg.arch == "h_block_refine":
        from .assembly.h_variants import HBlockRefineNCA

        return HBlockRefineNCA(cfg)
    if cfg.arch == "h_pod_modal":
        from .assembly.h_variants import HPODModalNCA

        return HPODModalNCA(cfg)

    # ── Transport variants（LBM 物理先驗算子）──────────────────────────────
    if cfg.arch == "h3t_warpresidual":
        from .assembly.transport_variants import H3TWarpResidualNCA

        return H3TWarpResidualNCA(cfg)
    if cfg.arch == "h3t_streamcollide":
        from .assembly.transport_variants import H3TStreamCollideNCA

        return H3TStreamCollideNCA(cfg)
    if cfg.arch == "h3t_warpresidual_serial":
        from .assembly.transport_variants import H3TWarpResidualSerialNCA

        return H3TWarpResidualSerialNCA(cfg)
    if cfg.arch == "operator_gated_streamcollide":
        from .assembly.transport_variants import OperatorGatedStreamCollideNCA

        return OperatorGatedStreamCollideNCA(cfg)

    # ── TURB-NCA 重新設計（clean-slate, spec §5/§9/§10）─────────────────────

    # ── Baselines（維持原路徑）──────────────────────────────────────────────
    if cfg.arch == "baseline_ar_fno":
        from .baselines.baseline_stepper import BaselineStepperNCA

        return BaselineStepperNCA(cfg, backbone="fno")
    if cfg.arch == "baseline_unet_modern":
        from .baselines.baseline_stepper import BaselineStepperNCA

        return BaselineStepperNCA(cfg, backbone="unet")
    if cfg.arch == "baseline_dilated_resnet":
        from .baselines.baseline_stepper import BaselineStepperNCA

        return BaselineStepperNCA(cfg, backbone="drn")
    if cfg.arch == "baseline_unet_steady":
        from .baselines.baseline_stepper import SteadyStateUNetNCA

        return SteadyStateUNetNCA(cfg, backbone="unet_steady")
    if cfg.arch == "baseline_fno_steady":
        from .baselines.baseline_stepper import SteadyStateUNetNCA

        return SteadyStateUNetNCA(cfg, backbone="fno")

    raise ValueError(f"Unknown TES model arch: {cfg.arch!r}")


__all__ = ["build_tes_model"]
