"""mean_vis.py — mean_module rollout 監控組裝（compose=identity + 多 Re 序列圖）。

> ARCH_mean_turb_split_plan §4。組裝 = 共用核心 ``build_rollout_seq_cols(compose=None)``
> （ū fixed-point，收斂判定 ``mae_ssim_converged``）+ 時空繪圖 ``plot_eval_sequence_multi``。

職責分工：
    - ``build_mean_sample``：單一 case → ``(cols, curve, solid)``（薄包共用核心；mean 特化參數綁好）。
    - ``render_mean_vis``：多 case sample 列表 + curve 列表 → 寫 ``mean_eval_sequence_latest.png``
      （多 Re 序列）+ ``rollout_error_latest.png``（誤差曲線）。

caller（async ``lc_vis_external`` / 未來 ``MeanTrainer.save_monitor_vis``）各自跑 dataloader、
組 cond/renorm/state0，逐 case 呼 ``build_mean_sample`` 收集，最後 ``render_mean_vis`` 一次出圖。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from lbmnca.nca.plots.train.mean_eval_sequence import plot_eval_sequence_multi
from lbmnca.nca.plots.train.rollout_error import plot_rollout_error_lines
from lbmnca.nca.tes.eval.mean_rollout_vis import build_rollout_seq_cols


def build_mean_sample(
    roll_step: Callable[[torch.Tensor], torch.Tensor],
    state0: torch.Tensor,
    target_mean: torch.Tensor,
    ic_physics: torch.Tensor,
    fluid_mask: torch.Tensor,
    sd: int,
    pd: int,
    field_norm,
    converter,
    n_steps: int,
    n_seq_cols: int,
    ctu_per_step: float = 1.0,
) -> tuple[list[dict], dict, np.ndarray]:
    """單一 mean case → ``(cols, curve, solid)``（compose=identity、收斂=mae_ssim fixed-point）。

    Args:
        roll_step: ``callable(state)->state``（caller 綁好 cond/mask/g_vec）。
        state0: 初始 state ``[B, sd+pd(+hidden), H, W]``。
        target_mean: ``[B, 9, H, W]`` mean target（norm）。
        ic_physics: ``[B, 9, H, W]`` IC（parent_mean）。
        fluid_mask: ``[B, 1, H, W]`` 二值 mask（1=fluid）。
        sd / pd: static_dim / physics_dim。
        field_norm: FieldNormalizer。
        converter: FieldConverter。
        n_steps: rollout 步數。
        n_seq_cols: log-spaced snapshot 欄數。
        ctu_per_step: 每步 pseudo-CTU（async 0.2 / offline 1.0）。

    Returns:
        ``(cols, curve, solid)``，格式同 ``build_rollout_seq_cols``。
    """
    return build_rollout_seq_cols(
        roll_step=roll_step, state0=state0, target=target_mean, ic=ic_physics,
        fluid_mask=fluid_mask, sd=sd, pd=pd, field_norm=field_norm, converter=converter,
        n_steps=n_steps, n_seq_cols=n_seq_cols, ctu_per_step=ctu_per_step,
        compose_fn=None, converged_fn=None,
        ic_label="IC (parent_mean)", target_label="MEAN TARGET",
        snap_label_prefix="", phase="mean",
        snap_color="tab:blue", target_color="tab:green",
    )


def render_mean_vis(
    seq_samples: Sequence[dict],
    curves: Sequence[dict],
    plots_dir: Path,
    step: int,
    rollout_k: Optional[float] = None,
) -> None:
    """多 Re mean 序列圖 + rollout 誤差曲線輸出。

    Args:
        seq_samples: list of dict{re, cols, solid}（``plot_eval_sequence_multi`` 用）。
        curves: list of curve dict（``plot_rollout_error_lines`` 用）。
        plots_dir: 輸出目錄。
        step: 當前訓練 step（標題用）。
        rollout_k: 若給定，誤差曲線於 ``rollout_k`` 畫 transition 虛線。
    """
    plots_dir = Path(plots_dir)
    if seq_samples:
        plot_eval_sequence_multi(
            list(seq_samples),
            f"mean_eval_sequence (multi-Re)  |  step {step}",
            plots_dir / "mean_eval_sequence_latest.png",
        )
    if curves:
        plot_rollout_error_lines(
            list(curves), plots_dir / "rollout_error_latest.png",
            f"mean rollout error  |  step {step}",
            transition_ctu=(float(rollout_k) if rollout_k is not None else None),
        )
