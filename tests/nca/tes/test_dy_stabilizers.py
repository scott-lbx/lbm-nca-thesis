"""Dy 穩定化單元測試：update_clip（rate 軟夾）+ L_temporal（2CTU KE）行為。"""
from __future__ import annotations

import pytest
import torch

from lbmnca.nca.tes.config import TESModelConfig, TESTemporalConfig
from lbmnca.nca.tes.loss.loss_terms_v2 import (
    JX_CH,
    RHO_CH,
    jacobian_amp_loss,
    temporal_ke_2ctu,
)
from lbmnca.nca.tes.models.markov_nca_cell import MarkovNCACell
from lbmnca.nca.tes.models.model_factory import build_tes_model


class _IdNorm:
    """測試用 identity FieldNormalizer stub：denorm_mphy 回傳原值。"""

    def denorm_mphy(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _ke_pred(vel_per_step: torch.Tensor, B: int = 1, H: int = 4, W: int = 4) -> torch.Tensor:
    """建 [B,R,9,H,W]：rho=1，jx=vel_per_step（broadcast），其餘 0。"""
    R = vel_per_step.shape[0]
    pred = torch.zeros(B, R, 9, H, W)
    pred[:, :, RHO_CH] = 1.0
    pred[:, :, JX_CH] = vel_per_step.view(1, R, 1, 1)
    return pred

SD, PD, HD, H, W = 4, 9, 0, 8, 8


def _cell(clip: float = 0.0) -> MarkovNCACell:
    c = MarkovNCACell(state_dim=SD + PD + HD, nca_channels=16, physics_dim=PD, hidden_dim=HD)
    c.update_clip = clip
    return c


def test_clip_zero_init_identity():
    """zero-init project + clip → 物理變化為 0（identity 保留）。"""
    c = _cell(clip=0.3)
    state = torch.randn(2, SD + PD, H, W)
    fm = torch.ones(2, 1, H, W)
    gamma = torch.ones(2, 16); beta = torch.zeros(2, 16)
    out = c(state, gamma, beta, fm)
    change = (out[:, SD:] - state[:, SD:]).abs().max().item()
    assert change < 1e-6, f"zero-init 應 identity，得到 change={change}"


def test_update_clip_bounds_rate():
    """非零 project + clip=0.3 → 每步物理變化（delta_ctu=None）幅度 ≤ 0.3。"""
    c = _cell(clip=0.3)
    torch.nn.init.normal_(c.project.weight, std=5.0)  # 強迫大 rate
    torch.nn.init.normal_(c.project.bias, std=5.0)
    state = torch.randn(2, SD + PD, H, W)
    fm = torch.ones(2, 1, H, W)
    gamma = torch.ones(2, 16); beta = torch.zeros(2, 16)
    out = c(state, gamma, beta, fm)
    change = (out[:, SD:] - state[:, SD:]).abs().max().item()
    assert change <= 0.3 + 1e-4, f"clip=0.3 應界定變化 ≤0.3，得到 {change}"


def test_no_clip_unbounded():
    """clip=0（停用）→ 大 project 權重 → 變化可超過 0.3（確認 clip 真的有作用差異）。"""
    c = _cell(clip=0.0)
    torch.nn.init.normal_(c.project.weight, std=5.0)
    torch.nn.init.normal_(c.project.bias, std=5.0)
    state = torch.randn(2, SD + PD, H, W)
    fm = torch.ones(2, 1, H, W)
    gamma = torch.ones(2, 16); beta = torch.zeros(2, 16)
    out = c(state, gamma, beta, fm)
    change = (out[:, SD:] - state[:, SD:]).abs().max().item()
    assert change > 0.3, f"無 clip 應可超過 0.3，得到 {change}"


def test_factory_applies_clip_to_all_cells():
    """model_factory 應把 cfg.update_clip 套到所有 MarkovNCACell。"""
    cfg = TESModelConfig(arch="h_multirate", state_dim=13, static_dim=4, physics_dim=9,
                         hidden_dim=0, cond_dim=19, update_clip=0.25)
    model = build_tes_model(cfg)
    cells = [m for m in model.modules() if isinstance(m, MarkovNCACell)]
    assert len(cells) > 0
    assert all(abs(m.update_clip - 0.25) < 1e-9 for m in cells)


# ── L_temporal（solver 2CTU two-window KE）─────────────────────────────────

def test_temporal_ke_steady_zero():
    """穩態（KE 各步常數）→ ratio≈0 < eps → loss≈0。"""
    R = 20
    pred = _ke_pred(torch.full((R,), 0.7))  # 等速 → 等 KE
    ctu = torch.arange(1, R + 1).float().view(1, R) * 0.5  # 0.5..10，覆蓋 2×window=10
    fm = torch.ones(1, 1, 4, 4)
    loss = temporal_ke_2ctu(pred, ctu, fm, window_ctu=5.0, eps=0.05, normalizer=_IdNorm())
    assert loss.item() < 1e-4, f"穩態應≈0，得到 {loss.item()}"


def test_temporal_ke_drift_positive():
    """強漂移（KE 逐步上升）→ ratio>eps → loss>0。"""
    R = 20
    pred = _ke_pred(torch.linspace(0.2, 1.0, R))  # 速度線性上升 → KE 漂移
    ctu = torch.arange(1, R + 1).float().view(1, R) * 0.5
    fm = torch.ones(1, 1, 4, 4)
    loss = temporal_ke_2ctu(pred, ctu, fm, window_ctu=5.0, eps=0.05, normalizer=_IdNorm())
    assert loss.item() > 0.05, f"漂移應 >eps，得到 {loss.item()}"


def test_temporal_ke_differentiable():
    """L_temporal 可微：grad 回流到 pred_seq 且有限非零。"""
    R = 20
    pred = _ke_pred(torch.linspace(0.2, 1.0, R)).detach().requires_grad_(True)
    ctu = torch.arange(1, R + 1).float().view(1, R) * 0.5
    fm = torch.ones(1, 1, 4, 4)
    loss = temporal_ke_2ctu(pred, ctu, fm, normalizer=_IdNorm())
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_temporal_ke_short_window_safe():
    """rollout 太短（不足 2×window）→ 無 valid 元素 → loss=0（不崩）。"""
    R = 4
    pred = _ke_pred(torch.linspace(0.2, 1.0, R))
    ctu = torch.arange(1, R + 1).float().view(1, R) * 0.5  # 0.5..2，< 2×window=10
    fm = torch.ones(1, 1, 4, 4)
    loss = temporal_ke_2ctu(pred, ctu, fm, window_ctu=5.0, eps=0.05, normalizer=_IdNorm())
    assert loss.item() == 0.0


def test_temporal_config_default_off_and_validation():
    """temporal config 預設 off；grad_tail>r_total 應驗證失敗。"""
    c = TESTemporalConfig()
    assert c.enabled is False and c.r_total >= 4
    TESTemporalConfig(enabled=True, r_total=50, grad_tail=2)  # 合法
    with pytest.raises(Exception):
        TESTemporalConfig(r_total=10, grad_tail=100)


# ── Dy-V8/V9：Jacobian reg（全域收縮）+ JAWS（空間自適應）────────────────────

def test_jac_global_ratio_gamma():
    """ratio=1（f_xe=f_x+ε）：γ=1.0 → loss≈0；γ=0.9 → relu(1−0.9)=0.1。"""
    B, pd, H, W = 2, 9, 8, 8
    fx = torch.zeros(B, pd, H, W)
    eps = 0.01 * torch.randn(B, pd, H, W)
    fm = torch.ones(B, 1, H, W)
    fxe = fx + eps  # 局部放大率 = 1
    assert jacobian_amp_loss(fx, fxe, eps, fm, gamma=1.0).item() < 1e-5
    assert abs(jacobian_amp_loss(fx, fxe, eps, fm, gamma=0.9).item() - 0.1) < 0.02


def test_jac_global_expansion_positive():
    """ratio=2（放大）→ relu(2−1)=1 → loss>0.5。"""
    B, pd, H, W = 2, 9, 8, 8
    fx = torch.zeros(B, pd, H, W)
    eps = 0.01 * torch.randn(B, pd, H, W)
    fm = torch.ones(B, 1, H, W)
    fxe = fx + 2.0 * eps
    assert jacobian_amp_loss(fx, fxe, eps, fm, gamma=1.0).item() > 0.5


def test_jac_differentiable():
    """jac loss 可微：grad 回流到 f_xe。"""
    B, pd, H, W = 2, 9, 8, 8
    fx = torch.zeros(B, pd, H, W)
    eps = 0.01 * torch.randn(B, pd, H, W)
    fm = torch.ones(B, 1, H, W)
    fxe = (fx + 2.0 * eps).detach().requires_grad_(True)
    loss = jacobian_amp_loss(fx, fxe, eps, fm, gamma=1.0)
    loss.backward()
    assert fxe.grad is not None and torch.isfinite(fxe.grad).all()
    assert fxe.grad.abs().sum().item() > 0.0


def test_jaws_downweights_complex_region():
    """JAWS：λ_c 越大 → 高複雜度區權重越低 → loss 越小（收縮放鬆於渦核）。"""
    B, pd, H, W = 1, 9, 16, 16
    # f_x 有空間梯度（x 方向 ramp）→ 複雜度 >0
    ramp = torch.linspace(0.0, 1.0, W).view(1, 1, 1, W).expand(B, pd, H, W).contiguous()
    fx = ramp * 1.5
    eps = 0.01 * torch.randn(B, pd, H, W)
    fm = torch.ones(B, 1, H, W)
    fxe = fx + 2.0 * eps  # 均勻放大 ratio~2
    l0 = jacobian_amp_loss(fx, fxe, eps, fm, gamma=1.0, jaws_enabled=True, jaws_lambda_c=0.0).item()
    l5 = jacobian_amp_loss(fx, fxe, eps, fm, gamma=1.0, jaws_enabled=True, jaws_lambda_c=5.0).item()
    assert l5 < l0, f"JAWS λ_c↑ 應降 loss（複雜區放鬆），得到 l0={l0}, l5={l5}"
