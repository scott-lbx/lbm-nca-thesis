"""mean_rollout_vis.py — rollout 視覺序列「共用核心」（mean / turb 泛化）。

抽出 trainer 與其他呼叫端重複的 rollout-vis 內迴圈（roll → masked MAE/SSIM →
收斂判定 → 組 seq_cols），使 SDF threshold / masked SSIM / 收斂判定 改一次即全體一致
（治本 DRY 漂移）。

★設計：async 監控為主（caller 載 ckpt、不碰 live training process）。本模組只負責
「給定 model 單步 callable + IC/target/mask，roll N 步並組成多 Re 序列圖的一個 sample」，
caller 各自保留 context-specific setup（cond 組裝 / renorm / device / n_steps）。

★泛化：核心 ``build_rollout_seq_cols`` 以 ``compose_fn`` / ``converged_fn`` 兩個 hook
涵蓋 mean 與 turb 兩條 pipeline：
  - ``compose_fn(phys) -> renderable``：``None`` = identity（mean，ū 直接渲染）；
    turb 傳 ``lambda p: mean_g + p``（ū+u′，將 fluct 殘差合回 mean 全場再渲染）。
  - ``converged_fn(mae_history, ssim_history) -> (converged, onset_idx)``：``None`` =
    ``mae_ssim_converged``（mean fixed-point）；turb 可傳 ``None``（暫不判 limit-cycle）。
``build_mean_seq_cols`` 保留為 thin wrapper（``compose=None`` + 預設收斂判定）供既有
呼叫端使用不破。

對外：``build_rollout_seq_cols`` / ``build_mean_seq_cols``。
繪圖用 ``lbmnca.nca.plots.train.mean_eval_sequence.plot_eval_sequence_multi``。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from lbmnca.nca.plots.train.mean_eval_sequence import mae_ssim_converged
from lbmnca.nca.tes.eval.visual_metrics import struct_vel_rgb_ssim

__all__ = [
    "build_rollout_seq_cols",
    "build_mean_seq_cols",
    "vel_vort_np",
    "masked_mae_mse",
]


def vel_vort_np(converter, phys: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """phys [B,9,H,W](norm) → (vel_mag, vorticity) numpy（取 b=0）。"""
    vphy = converter.toVphy(phys.float(), is_norm=True)
    return (vphy["vel_mag"][0].cpu().numpy(), vphy["vorticity"][0].cpu().numpy())


def masked_mae_mse(pred: torch.Tensor, tgt: torch.Tensor,
                   mask: torch.Tensor) -> tuple[float, float]:
    """fluid-masked per-pixel MAE/MSE（÷ fluid_pixels×channels，跨 case 公平）。"""
    m = mask.float().expand_as(pred)
    diff = (pred - tgt) * m
    denom = m.sum().clamp_min(1.0)
    mae = float(diff.abs().sum().item() / denom.item())
    mse = float((diff * diff).sum().item() / denom.item())
    return mae, mse


def build_rollout_seq_cols(
    roll_step: Callable[[torch.Tensor], torch.Tensor],
    state0: torch.Tensor,
    target: torch.Tensor,
    ic: torch.Tensor,
    fluid_mask: torch.Tensor,
    sd: int,
    pd: int,
    field_norm,
    converter,
    n_steps: int,
    n_seq_cols: int,
    ctu_per_step: float = 1.0,
    compose_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    converged_fn: Optional[Callable[..., tuple[bool, int]]] = None,
    ic_label: str = "IC (parent_mean)",
    target_label: str = "MEAN TARGET",
    snap_label_prefix: str = "",
    phase: str = "mean",
    snap_color: str = "tab:blue",
    target_color: str = "tab:green",
) -> tuple[list[dict], dict, np.ndarray]:
    """roll module N 步 → 組單一 sample 的序列欄 + 誤差曲線 + solid mask（mean/turb 共用）。

    roll + log-spaced snapshot + masked MAE/SSIM + 收斂判定 + 組欄邏輯為兩條 pipeline 共用；
    mean 與 turb 的差異收斂進 ``compose_fn``（渲染前合成）與 ``converged_fn``（收斂判準）兩個 hook。

    Args:
        roll_step: ``callable(state)->state``，模型單步推進（caller 綁好 cond/mask/g_vec/mean_flow）。
        state0: 初始 state ``[B, sd+pd(+hidden), H, W]``（caller 由 model.init_state 取得）。
        target: ``[B, 9, H, W]`` 渲染/誤差目標（norm）。mean = mean target；turb = GT 瞬時場。
        ic: ``[B, 9, H, W]`` IC 渲染場（mean = parent_mean；turb = mean_g+fluct0，caller 合好傳入）。
        fluid_mask: ``[B, 1, H, W]``（1=fluid；用正確 SDF threshold 算好）。
        sd / pd: static_dim / physics_dim（physics 切片 ``state[:, sd:sd+pd]``）。
        field_norm: FieldNormalizer（masked struct_vel_rgb_ssim 用）。
        converter: FieldConverter（vel/vort 渲染用）。
        n_steps: rollout 步數（mean 線常用 500、turb 線 ≤120）。★單位是**步**不是 CTU。
        n_seq_cols: 中間 log-spaced snapshot 欄數。
        ctu_per_step: 每步 pseudo-CTU；欄標籤＝``{step·ctu:.0f}CTU``。呼叫端一律傳
            ``tes.inference.protocol.NOMINAL_DELTA_CTU_PER_STEP``（mean）或
            ``TURB_DELTA_CTU_PER_STEP``（turb），與 ``build_cond`` 實際餵給模型的
            ``delta_ctu=1.0`` 一致。
        compose_fn: ``callable(phys)->renderable``。``None`` = identity（mean，physics 即可渲染）；
            turb 傳 ``lambda p: mean_g + p``（ū+u′：fluct 殘差合回 mean 全場）。誤差/SSIM/渲染皆用
            composed 場（與 target 對齊）。
        converged_fn: ``callable(mae_history, ssim_history)->(converged, onset_idx)``。
            ``None`` = ``mae_ssim_converged``（mean fixed-point plateau-onset）；turb 暫傳 ``None``
            → 退回 ``mae_ssim_converged``。若要完全停用收斂標記，傳 ``lambda *_: (False, -1)``。
        ic_label / target_label: IC 欄 / TARGET 欄標籤（turb 自訂為 ``IC (mean+fluct0)`` / ``TURB TARGET``）。
        snap_label_prefix: snapshot 欄標籤前綴（turb 用 ``"turb "`` → ``turb 12CTU``）。
        phase: 誤差曲線 ``phase`` 欄（``"mean"`` / ``"turb"``，繪圖分色用）。
        snap_color / target_color: snapshot / TARGET 欄色。

    Returns:
        ``(sample_cols, curve, sample_solid)``：
          - sample_cols: ``plot_eval_sequence_multi`` / ``plot_eval_sequence`` 用的欄 list
            （IC → snapshots → TARGET）。
          - curve: ``plot_rollout_error_lines`` 用的單條誤差曲線 dict。
          - sample_solid: ``[H, W]`` bool（solid 區，由二值 fluid_mask 取）。
    """
    if compose_fn is None:
        def compose_fn(phys: torch.Tensor) -> torch.Tensor:  # identity（mean）
            return phys
    if converged_fn is None:
        converged_fn = mae_ssim_converged

    sample_solid = fluid_mask[0, 0].cpu().numpy() <= 0.5
    tgt_v, tgt_w = vel_vort_np(converter, target)
    ic_v, ic_w = vel_vort_np(converter, ic)

    _lt = np.logspace(0, np.log10(n_steps), n_seq_cols)
    cand = sorted(set(int(np.clip(round(t), 1, n_steps)) for t in _lt))

    state = state0
    snaps_d: dict[int, torch.Tensor] = {}
    ctu, mae_l, mse_l, ssim_l = [], [], [], []
    for s in range(1, n_steps + 1):
        state = roll_step(state)
        phys = compose_fn(state[:, sd:sd + pd].float())
        mae, mse = masked_mae_mse(phys, target, fluid_mask)
        ctu.append(s * ctu_per_step); mae_l.append(mae); mse_l.append(mse)
        ssim_l.append(struct_vel_rgb_ssim(phys, target, field_norm, fluid_mask))
        if s in cand:
            snaps_d[s] = phys.clone()
    snaps_d[n_steps] = compose_fn(state[:, sd:sd + pd].float()).clone()  # 確保末步

    _conv, _conv_idx = converged_fn(mae_l, ssim_l)
    _conv_step = (_conv_idx + 1) if _conv else -1
    snaps = sorted(snaps_d.items())
    _conv_col_step = None
    if _conv:
        _ge = [s for s, _ in snaps if s >= _conv_step]
        _conv_col_step = min(_ge) if _ge else None

    sample_cols: list[dict] = [
        {"label": ic_label, "vel": ic_v, "vort": ic_w, "err": None, "color": "black"}
    ]
    for s, phys in snaps:
        v, w = vel_vort_np(converter, phys)
        sample_cols.append({"label": f"{snap_label_prefix}{s * ctu_per_step:.0f}CTU",
                            "vel": v, "vort": w,
                            "err": np.abs(v - tgt_v), "color": snap_color,
                            "converged": (s == _conv_col_step)})
    sample_cols.append({"label": target_label, "vel": tgt_v, "vort": tgt_w,
                        "err": None, "color": target_color})

    curve = {"ctu": ctu, "mae": mae_l, "mse": mse_l, "ssim": ssim_l,
             "color": snap_color, "phase": phase, "converged_step": _conv_step}
    return sample_cols, curve, sample_solid


def build_mean_seq_cols(
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
    """roll mean_module N 步 → 組單一 sample 的序列欄 + 誤差曲線 + solid mask。

    ★thin wrapper：泛化核心 ``build_rollout_seq_cols`` 的 mean 特化（``compose=None`` =
    identity、收斂判定 = ``mae_ssim_converged`` fixed-point）。保留舊參數名
    （``target_mean`` / ``ic_physics``）供現有 3 caller 不破。

    Args:
        roll_step: ``callable(state)->state``，模型單步推進（caller 綁好 cond/mask/g_vec）。
        state0: 初始 state ``[B, sd+pd(+hidden), H, W]``（caller 由 model.init_state 取得）。
        target_mean: ``[B, 9, H, W]`` mean target（norm）。
        ic_physics: ``[B, 9, H, W]`` IC（parent_mean，供 IC 欄渲染）。
        fluid_mask: ``[B, 1, H, W]``（1=fluid；用正確 SDF threshold 算好）。
        sd / pd: static_dim / physics_dim（physics 切片 ``state[:, sd:sd+pd]``）。
        field_norm: FieldNormalizer（masked struct_vel_rgb_ssim 用）。
        converter: FieldConverter（vel/vort 渲染用）。
        n_steps: rollout 步數（mean 線常用 500）。
        n_seq_cols: 中間 log-spaced snapshot 欄數。
        ctu_per_step: 每步 pseudo-CTU；欄標籤＝``{step·ctu:.0f}CTU``。呼叫端一律傳
            ``tes.inference.protocol.NOMINAL_DELTA_CTU_PER_STEP``（＝1.0）。

    Returns:
        ``(sample_cols, curve, sample_solid)``，格式同 ``build_rollout_seq_cols``。
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
