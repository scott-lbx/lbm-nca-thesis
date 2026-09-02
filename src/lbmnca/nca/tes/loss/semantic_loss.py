"""TESSemanticLoss — 語義化 loss 路由，使用 loss_terms_v2 / loss_terms_m1a 函式。

Config 格式：
    loss_terms:
      ObsMphy_Traj_Mae: 1.0
      ObsMphy_Delta_Mae: 0.5
      PhysMass_DivJ_AbsMean: 10.0
      BndWall_NormalFlux_AbsMean: 0.05

未知 loss term 一律在 __init__ 立即 ValueError；新增 loss 須註冊到
`_REGISTRY`（定義於 :mod:`registry`，dataclass-based）。

本模組僅保留 :class:`TESSemanticLoss` 路由邏輯；``_LossContext`` /
``_LossTerm`` dataclass 已移至 :mod:`context`，``_REGISTRY`` /
``_SUPPORTED_TERMS`` / ``_TERM_KEY`` / ``_build_registry`` 已移至
:mod:`registry`，兩者由本檔 import 後沿用，行為與單體版完全一致。
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .context import _LossContext, _LossTerm  # noqa: F401（向後相容 re-export）
from .registry import _REGISTRY, _SUPPORTED_TERMS, _TERM_KEY, _build_registry  # noqa: F401
from ..eval.pod_metrics import load_pod_basis

_logger = logging.getLogger(__name__)


class TESSemanticLoss(nn.Module):
    """語義化複合損失，由 ``_REGISTRY`` 路由至 loss term 函式。

    Args:
        loss_terms: 語義名稱→權重 dict，例如 ``{"ObsMphy_Traj_Mae": 1.0}``。
        stats_path: ``global_stats.json`` 路徑，供物理項反正規化 lazy-load。
        field_normalizer: 預先建好的 runtime-global ``FieldNormalizer``（可選）。
        pod_basis_path: POD 基底檔案路徑；啟用 StructVel_PODCoeffLoss 時必填。
        pod_modes: POD 投影保留模態數，預設 16。

    Raises:
        ValueError: ``loss_terms`` 含未註冊名稱，或啟用 PODCoeffLoss 但未提供路徑。
    """

    def __init__(
        self,
        loss_terms: dict[str, float],
        stats_path: str,
        field_normalizer: Optional[object] = None,
        pod_basis_path: str = "",
        pod_modes: int = 16,
        ssim_range_from_stats: bool = False,
    ) -> None:
        super().__init__()
        self._ssim_range_from_stats = bool(ssim_range_from_stats)
        unknown = set(loss_terms.keys()) - _SUPPORTED_TERMS
        if unknown:
            raise ValueError(
                f"TESSemanticLoss: 不支援的 loss term：{sorted(unknown)}。"
                f"支援項：{sorted(_SUPPORTED_TERMS)}"
            )
        self._terms: dict[str, float] = {k: float(v) for k, v in loss_terms.items()}
        self._stats_path = stats_path
        self._normalizer: Optional[object] = field_normalizer
        self._pod_basis_path = str(pod_basis_path or "")
        self._pod_modes = int(pod_modes)
        self._pod_basis: Optional[dict[str, Tensor]] = None
        if "StructVel_PODCoeffLoss" in self._terms and not self._pod_basis_path:
            raise ValueError("StructVel_PODCoeffLoss requires config.pod_basis_path")
        self._needs_normalizer: bool = any(
            _REGISTRY[name].needs_normalizer for name in self._terms
        )

    def _get_normalizer(self) -> object:
        """Lazy-load FieldNormalizer；已存在則直接回傳。

        Returns:
            FieldNormalizer 實例。
        """
        if self._normalizer is None:
            from ...utils.FieldNormalizer import FieldNormalizer
            self._normalizer = FieldNormalizer.from_file(self._stats_path)
        return self._normalizer

    def _get_pod_basis(self) -> dict[str, Tensor]:
        """Lazy-load POD 基底；已存在則直接回傳。

        Returns:
            以場名為鍵、POD 基底 Tensor 為值的字典。
        """
        if self._pod_basis is None:
            self._pod_basis = load_pod_basis(self._pod_basis_path)
        return self._pod_basis

    def forward(
        self,
        pred_trajectory: list[Tensor],
        target_traj: Tensor,
        ic_physics: Tensor,
        static: Tensor,
        fluid_mask: Tensor,
        loss_version: str,
        k: int,
        p_vec: Optional[Tensor] = None,
        collect_metrics: bool = True,
        current_ctu_seq: Optional[Tensor] = None,
        active_terms: Optional[set[str]] = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """計算語義複合損失。

        Args:
            pred_trajectory: ``List[k × Tensor[B, 9, H, W]]``，NCA 預測軌跡各步。
            target_traj: ``[B, k, 9, H, W]``，LBM 目標軌跡。
            ic_physics: ``[B, 9, H, W]``，物理初值（t=0 Mphy_norm）。
            static: ``[B, 4, H, W]``，靜態通道（ch0 = SDF_norm）。
            fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
            loss_version: 忽略（語義模式不使用 curriculum 版本切換）。
            k: rollout 步數（用於 delta_mae 維度確認）。
            p_vec: 選用；shape ``[B, D]``，D>=3 時 p_vec[:,2] 作為 l_char_px。
            collect_metrics: 是否收集各項 scalar 指標（訓練中每 N 步才開啟）。
            current_ctu_seq: ``[B, k]`` 各步步首 CTU，供 inflow prior 計算。
            active_terms: 若指定，只計算該子集 term（省記憶體模式）。

        Returns:
            ``(total_loss, metrics)``：``total_loss`` 為 scalar Tensor（可微）；
            ``metrics`` 含每項原值、加權值、9 通道 MAE、final-step vel_mag_mae。
        """
        pred = torch.stack(pred_trajectory, dim=1)
        device = pred.device
        dtype = pred.dtype
        total = torch.tensor(0.0, device=device, dtype=dtype)
        metrics: dict[str, float] = {}

        _l_char = p_vec[:, 2] if p_vec is not None and p_vec.shape[-1] >= 3 else None
        ctx = _LossContext(
            pred=pred,
            target=target_traj,
            ic_physics=ic_physics,
            static=static,
            fluid_mask=fluid_mask,
            _normalizer_getter=self._get_normalizer,
            _pod_basis_getter=self._get_pod_basis if self._pod_basis_path else None,
            pod_modes=self._pod_modes,
            current_ctu_seq=current_ctu_seq,
            l_char_px=_l_char,
            ssim_range_from_stats=self._ssim_range_from_stats,
        )

        for term, weight in self._terms.items():
            if active_terms is not None and term not in active_terms:
                continue  # loss-cut：只算指定 term 子集（如 SSIM 只在末 N landing 省記憶體）
            val = _REGISTRY[term].call(ctx)
            weighted = weight * val
            total = total + weighted
            if collect_metrics:
                key = _TERM_KEY[term]
                metrics[key] = val.item()
                metrics[f"w_{key}"] = weighted.item()

        if collect_metrics and not torch.isfinite(total):
            nan_keys = [k for k, v in metrics.items() if not math.isfinite(v)]
            _logger.warning("NaN in semantic loss: %s", nan_keys)
            metrics["nan_keys"] = nan_keys

        if not collect_metrics:
            return total, metrics

        metrics["total"] = total.item()

        # Per-channel monitoring（inference_mode，對 trainer CSV 相容）
        with torch.inference_mode():
            mask4d = fluid_mask.unsqueeze(1)
            ch_err = (pred - target_traj).abs() * mask4d
            ch_denom = mask4d.expand(
                pred.shape[0], pred.shape[1], 1, pred.shape[-2], pred.shape[-1]
            ).sum().clamp_min(1.0)
            for ch in range(9):
                metrics[f"ch{ch}_mae"] = (ch_err[:, :, ch].sum() / ch_denom).item()
            metrics["rho_mae"] = metrics["ch0_mae"]
            metrics["jx_mae"] = metrics["ch3_mae"]
            metrics["jy_mae"] = metrics["ch5_mae"]
            norm = self._get_normalizer()
            raw_pred = norm.denorm_mphy(pred[:, -1].float())
            raw_tgt = norm.denorm_mphy(target_traj[:, -1].float())
            rho_pred = raw_pred[:, 0].clamp_min(1e-6)
            rho_tgt = raw_tgt[:, 0].clamp_min(1e-6)
            vm_pred = torch.sqrt(
                (raw_pred[:, 3] / rho_pred).square()
                + (raw_pred[:, 5] / rho_pred).square()
                + 1e-8
            )
            vm_tgt = torch.sqrt(
                (raw_tgt[:, 3] / rho_tgt).square()
                + (raw_tgt[:, 5] / rho_tgt).square()
                + 1e-8
            )
            fm = fluid_mask.squeeze(1)
            metrics["vel_mag_mae"] = (
                ((vm_pred - vm_tgt).abs() * fm).sum() / fm.sum().clamp_min(1.0)
            ).item()

        return total, metrics


__all__ = ["TESSemanticLoss"]
