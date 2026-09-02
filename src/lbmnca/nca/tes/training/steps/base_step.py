"""training/steps/base_step.py — 標準單步訓練（forward→loss→backward→step），無特殊課程。

從 TESTrainer._train_step 提取：forward rollout → semantic loss → backward →
optimizer step → metrics。所有 curriculum 排程（k、loss_version、tf_ratio、
p_teacher、RP、SS）均在此封裝。

設計原則：
- 只依賴 TrainerContext；不直接存取 TESTrainer self。
- ctx.step 每次 run() 呼叫結束後遞增（與原 trainer 行為一致）。
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.nn as nn

from ..rng_streams import stream

if TYPE_CHECKING:
    from ..context import TrainerContext

_LOG_INTERVAL: int = 100  # metrics.json 寫入頻率（每 N 步）

logger = logging.getLogger(__name__)


class BaseStep:
    """標準單步訓練（forward→loss→backward→step），無特殊課程。

    封裝 TESTrainer._train_step 的完整邏輯：k curriculum、RP（Rolling Pool）
    預熱 rollout、IC masking、Teacher Forcing / Scheduled Sampling、temporal KE
    正則、Jacobian 正則、NaN guard、AMP grad scaler、gradient clip，以及
    per-channel MAE / cold-warm split MAE metrics 收集。

    Args:
        ctx: TESTrainer 共享狀態容器。
        data_prep: 資料前處理回呼（signature 與 TESTrainer._renorm_source_to_global 等
            輔助方法一致），為包含以下 callable 的 dict：
            - "renorm_source_to_global": (static, ic, traj, batch) -> tuple
            - "renorm_ic_mphy": (ic_raw, batch) -> Tensor
            - "get_fluid_mask": (static) -> Tensor
            - "build_cond_norm": (g_vec, p_vec, delta_ctu, cumul_ctu) -> Tensor
            - "build_g_vec_norm": (g_vec) -> Tensor
            - "maybe_train_noise": (ic, fluid_mask) -> Tensor
            - "uses_per_step_delta_ctu": () -> bool
            - "get_prior_vel_norm": () -> Tensor
        regularizer: 額外正則項回呼 dict（可為空 dict）：
            - "temporal_ke_loss": (state, cond, mask, g_norm, cumul_ctu, l_char) -> Tensor
            - "jacobian_reg_loss": (state, cond, mask, g_norm, dt_first, prior_ctx) -> Tensor
    """

    def __init__(
        self,
        ctx: "TrainerContext",
        data_prep: dict[str, Callable[..., Any]],
        regularizer: dict[str, Callable[..., Any]],
    ) -> None:
        self.ctx = ctx
        self._data_prep = data_prep
        self._regularizer = regularizer
        self._nan_consecutive: int = 0

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    def run(self, batch: dict) -> dict[str, float]:
        """執行一步梯度更新。

        Args:
            batch: DataLoader 回傳的 batch dict（需含 static / trajectory /
                g_vec / p_vec 等標準欄位）。

        Returns:
            metrics dict（loss 各項 + k / loss_version + 訓練動態指標）。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        sched = self._get_schedule_vals()
        k: int = sched["k"]
        loss_version: str = sched["loss_version"]
        tf_ratio: float = sched["tf_ratio"]
        p_teacher: float = sched["p_teacher"]
        use_ss: bool = sched["use_ss"]
        if use_ss:
            tf_ratio = p_teacher

        # ── t_start CPU 端先決定，只傳需要的幀到 GPU ──────────────────
        _traj_cpu: torch.Tensor = batch["trajectory"]  # [B, T, 9, H, W] CPU tensor
        B, T, _, H, W = _traj_cpu.shape

        # ── RP 模式判斷 ──────────────────────────────────────────────
        rp_cfg = cfg.rp
        # pool 觸發與 t_start 各走具名串流（isolation 關閉時＝全域 random，行為不變）
        use_rp: bool = (
            rp_cfg.enabled
            and stream("pool").random() < rp_cfg.pool_prob
            and T >= rp_cfg.r_steps + rp_cfg.k_rp_steps + 1
        )
        if use_rp:
            r_rp: int = rp_cfg.r_steps
            k = rp_cfg.k_rp_steps
        else:
            r_rp = 0
            if T < k + 1:
                k = max(1, T - 1)

        t_start: int = stream("rollout").randint(0, max(0, T - r_rp - k - 1))
        dataset_t0: int = int(
            batch.get("t0", torch.zeros(1, dtype=torch.long))[0].item()
        )
        global_t_start: int = dataset_t0 + t_start + r_rp

        # ── 資料移至 device ──────────────────────────────────────────
        static: torch.Tensor = batch["static"].to(ctx.device, non_blocking=True)
        ic_physics: torch.Tensor = (
            _traj_cpu[:, t_start].to(ctx.device, non_blocking=True).float()
        )
        target_traj: torch.Tensor = (
            _traj_cpu[:, t_start + r_rp + 1 : t_start + r_rp + k + 1]
            .to(ctx.device, non_blocking=True)
            .float()
        )

        # RP delta-loss 參考幀
        ic_for_delta_raw: Optional[torch.Tensor] = None
        if use_rp and r_rp > 0:
            ic_for_delta_raw = (
                _traj_cpu[:, t_start + r_rp]
                .to(ctx.device, non_blocking=True)
                .float()
            )

        g_vec: torch.Tensor = batch["g_vec"].to(ctx.device, non_blocking=True)
        p_vec: torch.Tensor = batch["p_vec"].to(ctx.device, non_blocking=True)

        delta_ctu_seq: Optional[torch.Tensor] = None
        delta_ctu_pre: Optional[torch.Tensor] = None
        if self._data_prep["uses_per_step_delta_ctu"]() and "delta_ctu" in batch:
            _full_dt = batch["delta_ctu"][
                :, t_start : t_start + r_rp + k
            ].to(ctx.device, non_blocking=True)
            if use_rp and r_rp > 0:
                delta_ctu_pre = _full_dt[:, :r_rp]
                delta_ctu_seq = _full_dt[:, r_rp:]
            else:
                delta_ctu_seq = _full_dt

        # renorm source-local → global
        static, ic_physics, target_traj = self._data_prep["renorm_source_to_global"](
            static, ic_physics, target_traj, batch
        )
        ic_for_delta: Optional[torch.Tensor]
        if ic_for_delta_raw is not None:
            ic_for_delta = self._data_prep["renorm_ic_mphy"](ic_for_delta_raw, batch)
        else:
            ic_for_delta = None

        fluid_mask: torch.Tensor = self._data_prep["get_fluid_mask"](static)

        # ── IC masking ───────────────────────────────────────────────
        if cfg.apply_ic_mask and t_start == 0:
            solid_mask = fluid_mask <= 0.5
            ic_physics = ic_physics.masked_fill(solid_mask.expand_as(ic_physics), 0.0)

        ic_physics = self._data_prep["maybe_train_noise"](ic_physics, fluid_mask)

        # ── 建立初始狀態 ─────────────────────────────────────────────
        state: torch.Tensor = ctx.model.init_state(static, ic_physics)

        # ── 條件向量（cond_5D / cond_19D 含 cumulative_ctu）────────
        _cumul_ctu: Optional[torch.Tensor] = None
        _delta_ctu_cond: Optional[torch.Tensor] = None
        _cumul_ctu_norm_seq: Optional[torch.Tensor] = None
        _ctu_seq: Optional[torch.Tensor] = None
        if cfg.model.cond_dim in (5, 19) and "ctu_values" in batch:
            ctu_vals: torch.Tensor = batch["ctu_values"].to(ctx.device, non_blocking=True)
            _t_sup = min(t_start + r_rp, ctu_vals.shape[1] - 1)
            _cumul_ctu = ctu_vals[:, _t_sup]
            if "delta_ctu" in batch:
                _dc = batch["delta_ctu"].to(ctx.device, non_blocking=True)
                _t_dc = min(t_start + r_rp, _dc.shape[1] - 1)
                _delta_ctu_cond = _dc[:, _t_dc]
            _ctu_end = min(t_start + r_rp + k + 1, ctu_vals.shape[1])
            _ctu_seq = ctu_vals[:, t_start + r_rp + 1 : _ctu_end]
            if _ctu_seq.shape[1] == k:
                _cumul_ctu_norm_seq = ctx.normalizer.normalize_cumulative_ctu(_ctu_seq)

        cond_norm: torch.Tensor = self._data_prep["build_cond_norm"](
            g_vec, p_vec, _delta_ctu_cond, _cumul_ctu
        )
        g_vec_norm: torch.Tensor = self._data_prep["build_g_vec_norm"](g_vec)

        # HF-1d prior_ctx（operator-gated arch）
        _prior_ctx: Optional[dict] = None
        if getattr(ctx.model, "wants_prior_ctx", False) and _ctu_seq is not None:
            _prior_ctx = {
                "current_ctu_raw_seq": _ctu_seq,
                "l_char_px": p_vec[:, 2] if p_vec is not None else None,
                "prior_vel_norm": self._data_prep["get_prior_vel_norm"](),
            }

        # ── 前向 rollout + 損失計算 ──────────────────────────────────
        ctx.optimizer.zero_grad(set_to_none=True)
        _need_log: bool = ctx.step % _LOG_INTERVAL == 0

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            # RP pre-rollout（no_grad，梯度不回流）
            if use_rp:
                with torch.no_grad():
                    state = ctx.model(
                        state,
                        cond_norm,
                        fluid_mask,
                        r_rp,
                        return_trajectory=False,
                        delta_ctu_seq=delta_ctu_pre,
                        g_vec_norm=g_vec_norm,
                    )
                state = state.detach()

            # 監督 rollout
            _sup_kwargs: dict = (
                {"prior_ctx": _prior_ctx} if _prior_ctx is not None else {}
            )
            _, pred_traj = ctx.model(
                state,
                cond_norm,
                fluid_mask,
                k,
                return_trajectory=True,
                delta_ctu_seq=delta_ctu_seq,
                g_vec_norm=g_vec_norm,
                tf_ratio=tf_ratio,
                gt_traj=target_traj if tf_ratio > 0.0 else None,
                ss_detach=use_ss,
                cumul_ctu_norm_seq=_cumul_ctu_norm_seq,
                **_sup_kwargs,
            )

            ic_for_loss: torch.Tensor = (
                ic_for_delta if ic_for_delta is not None else ic_physics
            )
            loss, loss_metrics = ctx.loss_fn(
                pred_trajectory=pred_traj,
                target_traj=target_traj,
                ic_physics=ic_for_loss,
                static=static,
                fluid_mask=fluid_mask,
                loss_version=loss_version,
                k=k,
                p_vec=p_vec,
                collect_metrics=_need_log,
                current_ctu_seq=_ctu_seq,
            )

            # Dy-C/E：temporal KE loss（每 N 步才算）
            if cfg.temporal.enabled and (ctx.step % cfg.temporal.interval == 0):
                _l_char = p_vec[:, 2] if p_vec is not None else None
                l_temporal = self._regularizer["temporal_ke_loss"](
                    state, cond_norm, fluid_mask, g_vec_norm, _cumul_ctu, _l_char
                )
                loss = loss + float(cfg.temporal.weight) * l_temporal
                if isinstance(loss_metrics, dict):
                    loss_metrics["temporal_ke"] = float(l_temporal.detach().item())
                    loss_metrics["total"] = float(loss.detach().item())

            # Dy-V8/V9：Jacobian reg
            if cfg.jac_reg_weight > 0.0:
                _dt_first = (
                    delta_ctu_seq[:, 0:1] if delta_ctu_seq is not None else None
                )
                l_jac = self._regularizer["jacobian_reg_loss"](
                    state, cond_norm, fluid_mask, g_vec_norm, _dt_first, _prior_ctx
                )
                loss = loss + float(cfg.jac_reg_weight) * l_jac
                if isinstance(loss_metrics, dict):
                    loss_metrics["jac_reg"] = float(l_jac.detach().item())
                    loss_metrics["total"] = float(loss.detach().item())

        # ── NaN / spike 守衛 ─────────────────────────────────────────
        loss_val: Optional[float] = loss_metrics.get("total")
        if loss_val is None:
            loss_val = loss.detach().float().item()
            loss_metrics["total"] = loss_val
        is_bad: bool = (not math.isfinite(loss_val)) or (
            loss_val > cfg.loss_spike_threshold
        )
        if is_bad:
            self._nan_consecutive += 1
            logger.warning(
                "[step %d] unstable loss=%.4g  consecutive=%d/%d  (k=%d, v=%s)",
                ctx.step,
                loss_val,
                self._nan_consecutive,
                cfg.nan_patience,
                k,
                loss_version,
            )
            if self._nan_consecutive >= cfg.nan_patience:
                raise RuntimeError(
                    f"NaN early stop at step {ctx.step}: "
                    f"{self._nan_consecutive} consecutive unstable steps "
                    f"(threshold={cfg.loss_spike_threshold})"
                )
            ctx.step += 1
            return {"k": k, "loss_version": loss_version, "total": float("nan")}

        # ── 每 500 步定期檢查權重 NaN ────────────────────────────────
        if ctx.step % 500 == 0:
            for name, param in ctx.model.named_parameters():
                if not torch.isfinite(param).all():
                    logger.error(
                        "[step %d] NaN/Inf in param %s — emergency stop",
                        ctx.step,
                        name,
                    )
                    raise RuntimeError(
                        f"NaN weight detected in {name} at step {ctx.step}"
                    )

        # ── Backward + optimizer step ────────────────────────────────
        ctx.scaler.scale(loss).backward()
        ctx.scaler.unscale_(ctx.optimizer)
        grad_norm_tensor = nn.utils.clip_grad_norm_(
            ctx.model.parameters(), max_norm=cfg.grad_clip_norm
        )
        grad_norm: Optional[float] = grad_norm_tensor.item() if _need_log else None

        ctx.scaler.step(ctx.optimizer)
        ctx.scaler.update()
        ctx.scheduler.step()

        self._nan_consecutive = 0

        scaler_scale: Optional[float] = (
            ctx.scaler.get_scale() if (cfg.use_amp and _need_log) else None
        )
        gpu_mem_mb: Optional[float] = (
            torch.cuda.memory_allocated(ctx.device) / 1e6
            if (_need_log and ctx.device.type == "cuda")
            else None
        )

        ctx.step += 1
        out: dict = {
            "k": k,
            "loss_version": loss_version,
            "tf_ratio": tf_ratio,
            "p_teacher": p_teacher,
            "use_ss": int(use_ss),
            "use_rp": int(use_rp),
            "r_rp": r_rp,
            "grad_norm": grad_norm,
            "scaler_scale": scaler_scale,
            "gpu_mem_mb": gpu_mem_mb,
            "t_start": t_start,
            "global_t_start": global_t_start,
            **loss_metrics,
        }
        out["frame_window"] = k
        out["dcs_enabled"] = int(delta_ctu_seq is not None)
        if delta_ctu_seq is not None:
            with torch.no_grad():
                _dt = delta_ctu_seq.detach().float()
                out["delta_ctu_min"] = float(_dt.min().item())
                out["delta_ctu_mean"] = float(_dt.mean().item())
                out["delta_ctu_max"] = float(_dt.max().item())
                out["physical_horizon_ctu"] = float(_dt.sum(dim=1).mean().item())
                _flat = _dt.flatten()
                out["delta_ctu_bucket_0p2"] = int(
                    ((_flat >= 0.1) & (_flat < 0.5)).sum().item()
                )
                out["delta_ctu_bucket_1p0"] = int(
                    ((_flat >= 0.5) & (_flat < 1.5)).sum().item()
                )
                out["delta_ctu_bucket_2p0"] = int(
                    ((_flat >= 1.5) & (_flat < 3.0)).sum().item()
                )

        # cold/warm split MAE
        if _need_log:
            try:
                import numpy as np  # noqa: PLC0415

                ic_types = batch.get("ic_type")
                if isinstance(ic_types, list) and len(ic_types) > 0:
                    is_cold = np.array([str(x) == "cold" for x in ic_types])
                    out["batch_cold_frac"] = float(is_cold.mean())
                    with torch.no_grad():
                        pf = pred_traj[-1].float()
                        tf = target_traj[:, -1].float()
                        fm = fluid_mask.float()
                        denom = fm.sum(dim=(1, 2, 3)).clamp_min(1.0)
                        per_ch = (
                            (pf - tf).abs() * fm
                        ).sum(dim=(2, 3)) / denom.unsqueeze(1)
                    for split, sel in (("cold", is_cold), ("warm", ~is_cold)):
                        if sel.any():
                            idx = torch.tensor(
                                np.where(sel)[0], device=per_ch.device
                            )
                            cm = per_ch.index_select(0, idx).mean(dim=0)
                            out[f"rho_mae_{split}"] = float(cm[0].item())
                            out[f"jx_mae_{split}"] = float(cm[3].item())
                            out[f"jy_mae_{split}"] = float(cm[5].item())
            except Exception as _split_err:
                logger.debug(
                    "train split MAE skipped at step %d: %s", ctx.step, _split_err
                )
        return out

    # ------------------------------------------------------------------
    # 排程輔助
    # ------------------------------------------------------------------

    def _get_schedule_vals(self) -> dict[str, Any]:
        """取得當前步的所有 curriculum 排程值。

        Returns:
            包含 k、loss_version、tf_ratio、p_teacher、use_ss 的 dict。
        """
        from .. import schedule as _sched  # noqa: PLC0415

        ctx = self.ctx
        cfg = ctx.cfg
        k: int = _sched.sample_k(cfg.curriculum, ctx.step)
        loss_version: str = _sched.get_loss_version(cfg.curriculum, ctx.step)
        tf_ratio: float = _sched.get_tf_ratio(cfg.tfc, ctx.step)
        p_teacher: float = _sched.get_p_teacher(cfg.ss, ctx.step)
        use_ss: bool = cfg.ss.enabled
        return {
            "k": k,
            "loss_version": loss_version,
            "tf_ratio": tf_ratio,
            "p_teacher": p_teacher,
            "use_ss": use_ss,
        }
