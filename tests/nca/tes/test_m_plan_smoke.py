"""M-plan dummy-train smoke tests.

對 docs/research_plan/M_plan.md 列出的所有 ablation 變體與 TESSemanticLoss
註冊的全部 19 個 loss term 進行 forward+loss+backward 煙霧測試，
不依賴 zarr 資料、使用 synthetic tensors 並以 in-memory FieldNormalizer。

驗證目標：
1. TESSemanticLoss 每個註冊 term 個別都能 forward + backward 不 NaN。
2. M_base / M_no_mrp / M_no_hc / M_no_ss / M_no_rp / M_gvec 6 個 M-plan
   ablation 對應的 TESNCA 變體皆可 forward + 算 loss + backward 不爆。
3. SS (ss_detach=True + tf_ratio>0) 與 TFC (tf_ratio=1.0, ss_detach=False)
   兩種 schedule 路徑都跑得通。
4. Dual-pathway G_vec (gated mode) 與 cond_dim ∈ {3, 4, 5, 17, 19} 全部
   可建構 + forward。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.config import MRPConfig, TESModelConfig
from lbmnca.nca.tes.loss.semantic_loss import TESSemanticLoss, _REGISTRY
from lbmnca.nca.tes.models.tes_nca import TESNCA
from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B: int = 2
K: int = 3
H: int = 32
W: int = 32


def _make_field_normalizer() -> FieldNormalizer:
    """Minimal identity-stats FieldNormalizer（mean=0, std=1）。"""
    n = 9
    return FieldNormalizer(
        mf_mean=np.zeros(n, dtype=np.float32),
        mf_std=np.ones(n, dtype=np.float32),
        up_mean=np.zeros(n, dtype=np.float32),
        up_std=np.ones(n, dtype=np.float32),
        sdf_mean=0.0,
        sdf_std=1.0,
        px_mean=0.0,
        px_std=1.0,
        py_mean=0.0,
        py_std=1.0,
        cond_mean=np.zeros(1, dtype=np.float32),
        cond_std=np.ones(1, dtype=np.float32),
        is_physics_informed=False,
        # ★ DEEP_AUDIT v2 `L0.3-a`：vis_range 已無預設值（預設＝靜默 fallback）。
        #   本 fixture 先前隱性依賴預設 0.14，而 `eval/pod_metrics.py` 會讀
        #   `normalizer.vel_vis_max` → 現在必須顯式給。
        #   數值取自 `dataset_zarr/*_stats.json` 的 `vis_range`（global p99.5）。
        vis_vel_max=0.14,
        vis_vor_range=0.016,
        vis_p_center=0.3333,
        vis_p_range=0.01,
    )


def _make_static() -> torch.Tensor:
    """[B, 4, H, W] static with left quarter solid, rest fluid."""
    sdf = torch.ones(B, H, W)
    sdf[:, :, : W // 4] = -1.0
    static = torch.zeros(B, 4, H, W)
    static[:, 0] = sdf
    return static


def _make_traj(scale: float = 0.05) -> torch.Tensor:
    return torch.randn(B, K, 9, H, W) * scale


def _make_ic(scale: float = 0.05) -> torch.Tensor:
    return torch.randn(B, 9, H, W) * scale


def _fluid_mask_from_static(static: torch.Tensor) -> torch.Tensor:
    return (static[:, 0:1] > 0.0).float()


# ---------------------------------------------------------------------------
# Part 1：每個 loss term 個別 forward + backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("term_name", sorted(_REGISTRY))
def test_loss_term_forward_backward(term_name: str, tmp_path) -> None:
    """每個註冊 loss term 個別 forward + backward 不 NaN。"""
    fn = _make_field_normalizer()
    kwargs = {}
    if term_name == "StructVel_PODCoeffLoss":
        basis_path = tmp_path / "pod_basis.npz"
        modes = np.zeros((16, H, W), dtype=np.float32)
        for i in range(16):
            modes[i].reshape(-1)[i] = 1.0
        np.savez_compressed(
            basis_path,
            mean=np.zeros((H, W), dtype=np.float32),
            modes=modes,
            singular_values=np.ones(16, dtype=np.float32),
            explained_variance=np.ones(16, dtype=np.float32) / 16.0,
        )
        kwargs["pod_basis_path"] = str(basis_path)
    loss_fn = TESSemanticLoss({term_name: 1.0}, stats_path="", field_normalizer=fn, **kwargs)

    pred = _make_traj(0.05)
    target = _make_traj(0.05)
    ic = _make_ic(0.05)
    static = _make_static()
    fluid_mask = _fluid_mask_from_static(static)

    # 將 pred 製作為 leaf tensor with grad；list of tensors（每步）
    pred_list = [pred[:, i].clone().requires_grad_(True) for i in range(K)]

    total, metrics = loss_fn(
        pred_trajectory=pred_list,
        target_traj=target,
        ic_physics=ic,
        static=static,
        fluid_mask=fluid_mask,
        loss_version="v1",
        k=K,
    )

    assert torch.isfinite(total), f"{term_name}: total={total}"
    assert term_name.lower() in metrics, f"{term_name}: metric missing"
    assert "nan_keys" not in metrics, f"{term_name}: nan_keys={metrics.get('nan_keys')}"

    # backward 通過
    total.backward()
    grads = [p.grad for p in pred_list if p.grad is not None]
    assert grads, f"{term_name}: no grad propagated to pred_trajectory"
    for g in grads:
        assert torch.isfinite(g).all(), f"{term_name}: grad has NaN/Inf"


# ---------------------------------------------------------------------------
# Part 2：M-plan ablation 變體 forward + loss + backward
# ---------------------------------------------------------------------------


def _make_model_cfg(**overrides) -> TESModelConfig:
    """建立 M-plan-style TESModelConfig。

    預設為 M_base：state_dim=29, hidden_dim=16, cond_dim=19, mrp enabled。
    可由 kwargs 覆寫對應 ablation。
    """
    mrp_overrides = overrides.pop("mrp", {})
    default_mrp = dict(
        enabled=True,
        use_topo=True,
        use_stream=True,
        use_trans=True,
        num_learnable=8,    # 縮小供 smoke 用
        hormone_dim=4,
    )
    default_mrp.update(mrp_overrides)
    mrp = MRPConfig(**default_mrp)

    defaults = dict(
        state_dim=29,
        static_dim=4,
        physics_dim=9,
        hidden_dim=16,
        cond_dim=19,
        encoder_hidden=32,
        nca_channels=16,
        g_vec_dim=0,
        gvec_mode="none",
        mrp=mrp,
    )
    defaults.update(overrides)
    return TESModelConfig(**defaults)


@pytest.mark.parametrize("variant,cfg_kwargs", [
    ("M_base", dict()),
    ("M_no_mrp", dict(mrp=dict(enabled=False))),
    ("M_no_hc", dict(state_dim=13, hidden_dim=0)),
    ("M_no_ss", dict()),                                 # 由 schedule 差別決定
    ("M_no_rp", dict()),                                 # RP 由 trainer 控制，模型路徑與 M_base 同
    ("M_gvec", dict(cond_dim=4, g_vec_dim=14, gvec_mode="gated", encoder_hidden=64)),
])
def test_m_plan_ablation_forward(variant: str, cfg_kwargs: dict) -> None:
    """各 ablation 變體可建構 + forward + 取 trajectory。"""
    cfg = _make_model_cfg(**cfg_kwargs)
    model = TESNCA(cfg)

    static = _make_static()
    ic_physics = _make_ic(0.05)
    state = model.init_state(static, ic_physics)
    fluid_mask = _fluid_mask_from_static(static)
    cond_norm = torch.randn(B, cfg.cond_dim)
    g_vec_norm = torch.randn(B, cfg.g_vec_dim) if cfg.g_vec_dim > 0 else None

    final, traj = model(
        state, cond_norm, fluid_mask, K,
        return_trajectory=True, g_vec_norm=g_vec_norm,
    )

    assert final.shape == (B, cfg.state_dim, H, W), f"{variant}: bad final shape {final.shape}"
    assert len(traj) == K, f"{variant}: trajectory length {len(traj)} != {K}"
    for i, step in enumerate(traj):
        assert step.shape == (B, cfg.physics_dim, H, W), f"{variant} step{i}: {step.shape}"
        assert torch.isfinite(step).all(), f"{variant} step{i}: NaN/Inf in trajectory"


@pytest.mark.parametrize("variant,cfg_kwargs", [
    ("M_base", dict()),
    ("M_no_mrp", dict(mrp=dict(enabled=False))),
    ("M_no_hc", dict(state_dim=13, hidden_dim=0)),
])
def test_m_plan_ablation_loss_backward(variant: str, cfg_kwargs: dict) -> None:
    """M-plan ablation 各變體 forward + L_winner loss + backward 路徑通暢。"""
    cfg = _make_model_cfg(**cfg_kwargs)
    model = TESNCA(cfg)

    static = _make_static()
    ic_physics = _make_ic(0.05)
    state = model.init_state(static, ic_physics)
    fluid_mask = _fluid_mask_from_static(static)
    cond_norm = torch.randn(B, cfg.cond_dim)

    # L_winner loss
    fn = _make_field_normalizer()
    loss_fn = TESSemanticLoss(
        {
            "ObsMphy_Traj_Mae": 1.0,
            "ObsMphy_Delta_Mae": 0.5,
            "ObsVel_Vorticity_Mae": 0.5,
        },
        stats_path="",
        field_normalizer=fn,
    )

    target_traj = _make_traj(0.05)

    _, traj = model(state, cond_norm, fluid_mask, K, return_trajectory=True)
    total, metrics = loss_fn(
        pred_trajectory=traj,
        target_traj=target_traj,
        ic_physics=ic_physics,
        static=static,
        fluid_mask=fluid_mask,
        loss_version="v1",
        k=K,
    )

    assert torch.isfinite(total), f"{variant}: total loss non-finite"
    total.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    assert has_grad, f"{variant}: no finite gradient on model params"


# ---------------------------------------------------------------------------
# Part 3：Schedule 變體（TFC / SS / standard）
# ---------------------------------------------------------------------------


def _run_schedule(tf_ratio: float, ss_detach: bool) -> None:
    cfg = _make_model_cfg()
    model = TESNCA(cfg)

    static = _make_static()
    ic_physics = _make_ic(0.05)
    state = model.init_state(static, ic_physics)
    fluid_mask = _fluid_mask_from_static(static)
    cond_norm = torch.randn(B, cfg.cond_dim)
    gt_traj = _make_traj(0.05)

    _, traj = model(
        state, cond_norm, fluid_mask, K,
        return_trajectory=True,
        tf_ratio=tf_ratio,
        gt_traj=gt_traj,
        ss_detach=ss_detach,
    )
    assert len(traj) == K
    for step in traj:
        assert torch.isfinite(step).all()


def test_schedule_tfc_full() -> None:
    """純 TFC：tf_ratio=1.0, ss_detach=False。"""
    _run_schedule(tf_ratio=1.0, ss_detach=False)


def test_schedule_ss_detach() -> None:
    """Scheduled Sampling：tf_ratio=0.5, ss_detach=True。"""
    _run_schedule(tf_ratio=0.5, ss_detach=True)


def test_schedule_pure_freeroll() -> None:
    """純自迴歸：tf_ratio=0.0。"""
    _run_schedule(tf_ratio=0.0, ss_detach=False)


# ---------------------------------------------------------------------------
# Part 4：cond_dim 變體
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cond_dim,g_vec_dim,gvec_mode", [
    (3,  0,  "none"),
    (4,  0,  "none"),
    (5,  0,  "none"),
    (17, 0,  "none"),
    (19, 0,  "none"),
    (4, 14, "gated"),       # M_gvec 真實組合
])
def test_cond_dim_variants(cond_dim: int, g_vec_dim: int, gvec_mode: str) -> None:
    """各 cond_dim 與 dual-pathway 組合可正確 forward。"""
    cfg = _make_model_cfg(cond_dim=cond_dim, g_vec_dim=g_vec_dim, gvec_mode=gvec_mode)
    model = TESNCA(cfg)

    static = _make_static()
    state = model.init_state(static, _make_ic())
    fluid_mask = _fluid_mask_from_static(static)
    cond_norm = torch.randn(B, cond_dim)
    g_vec_norm = torch.randn(B, g_vec_dim) if g_vec_dim > 0 else None

    final = model(state, cond_norm, fluid_mask, K, return_trajectory=False,
                  g_vec_norm=g_vec_norm)
    assert final.shape == (B, cfg.state_dim, H, W)
    assert torch.isfinite(final).all()


def test_time_conditioned_cond18() -> None:
    """time_conditioned=True 時 cond_dim=18：cond_norm 仍為 17D，每步 concat dt → 18D。"""
    cfg = _make_model_cfg(cond_dim=18)
    model = TESNCA(cfg)

    static = _make_static()
    state = model.init_state(static, _make_ic())
    fluid_mask = _fluid_mask_from_static(static)
    cond_norm = torch.randn(B, 17)            # base 17D
    delta_ctu_seq = torch.ones(B, K) * 0.5    # per-step dt

    _, traj = model(state, cond_norm, fluid_mask, K, return_trajectory=True,
                    delta_ctu_seq=delta_ctu_seq)
    assert len(traj) == K
    for step in traj:
        assert torch.isfinite(step).all()
