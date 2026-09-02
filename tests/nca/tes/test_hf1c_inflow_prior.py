"""HF-1c：deterministic inflow prior loss-mask 測試。

1. identity：uniform region weights (1,1,1) → weighted MAE == 標準 fluid MAE（loss 不改模型；
   weight 全 1 時退化為 baseline，過不了不准進主跑）。
2. 方向（HF_code_plan §9 必加）：current_ctu 增加 → active 區往 downstream(+x) 擴張、
   鋒面帶質心往 +X_01 推進（front 隨時間前進，無 double-advance：純由 current_ctu 決定）。
3. no-leakage：mask 計算簽章不含 target（結構性保證）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.loss.inflow_prior import (  # noqa: E402
    compute_inflow_masks,
    inflow_weight_map_seq,
    obs_mphy_traj_mae_prior,
)
from lbmnca.nca.tes.loss.loss_terms_v2 import obs_mphy_traj_mae  # noqa: E402


def _x01_static(b: int, h: int, w: int) -> torch.Tensor:
    """static [B,4,H,W]，ch2 = X_01 = (col+0.5)/W。"""
    x = (torch.arange(w, dtype=torch.float32) + 0.5) / w
    x01 = x.view(1, 1, 1, w).expand(b, 1, h, w)
    static = torch.zeros(b, 4, h, w)
    static[:, 2:3] = x01
    return static


def test_uniform_weights_identity() -> None:
    b, k, h, w = 2, 3, 16, 32
    pred = torch.randn(b, k, 9, h, w)
    target = torch.randn(b, k, 9, h, w)
    fm = torch.ones(b, 1, h, w)
    static = _x01_static(b, h, w)
    cc = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]])
    lc = torch.tensor([106.0, 80.0])
    wmap = inflow_weight_map_seq(
        cc, lc, static, fm, w_active=1.0, w_transition=1.0, w_inactive=1.0
    )
    base = obs_mphy_traj_mae(pred, target, fm)
    weighted = obs_mphy_traj_mae_prior(pred, target, fm, wmap)
    assert torch.allclose(weighted, base, atol=1e-5), "uniform weights 應退化為標準 fluid MAE"


def test_none_weightmap_identity() -> None:
    b, k, h, w = 2, 2, 8, 16
    pred = torch.randn(b, k, 9, h, w)
    target = torch.randn(b, k, 9, h, w)
    fm = torch.ones(b, 1, h, w)
    assert torch.allclose(
        obs_mphy_traj_mae_prior(pred, target, fm, None),
        obs_mphy_traj_mae(pred, target, fm),
        atol=1e-6,
    )


def test_front_advances_downstream_with_ctu() -> None:
    b, h, w = 1, 16, 64
    fm = torch.ones(b, 1, h, w)
    static = _x01_static(b, h, w)
    x01 = static[:, 2:3]
    lc = torch.tensor([106.0])
    # 大 L_char 讓 front 在域內可見移動
    a_early, t_early, _ = compute_inflow_masks(torch.tensor([2.0]), lc, x01, fm)
    a_late, t_late, _ = compute_inflow_masks(torch.tensor([8.0]), lc, x01, fm)
    # active 面積隨 ctu 增加
    assert a_late.sum() > a_early.sum(), "current_ctu 增加 → active 區應擴張"
    # 鋒面帶質心（沿 X_01）往 downstream 推進
    def centroid(tb):
        wsum = (tb * x01).sum()
        return (wsum / tb.sum().clamp_min(1e-6)).item()
    assert centroid(t_late) > centroid(t_early), "front 鋒面帶質心應隨 ctu 往 +x 推進"


def test_ctu_zero_front_at_inlet() -> None:
    """ctu=0 → front 在 inlet(X_01≈0) → 域內幾乎全 inactive（active 面積極小）。"""
    b, h, w = 1, 16, 64
    fm = torch.ones(b, 1, h, w)
    static = _x01_static(b, h, w)
    a0, _, _ = compute_inflow_masks(torch.tensor([0.0]), torch.tensor([106.0]), static[:, 2:3], fm)
    # ctu=0：x_front_01=0，active=sigmoid(-X_01/w_front) < 0.5 everywhere → 總量 < 半域
    assert a0.mean() < 0.5
