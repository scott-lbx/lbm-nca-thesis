"""三態推論協定（`tes.inference.protocol`）單元測試 —— DEEP_AUDIT Stage 3.6/3.7/3.8。

守住四條協定不變量：
  1. ``resolve_ke_anchor`` 是 **rollout 起始 state 的純函式**（不吃監督目標、不跨呼叫累積）。
  2. ``ke_anchor_mode='ceiling'`` 時模型 forward 漏傳 anchor 會**直接報錯**，不再靜默 bypass。
  3. ``chain_handoff`` 保留 hidden；``init_state`` 交接則歸零（兩者不可混用）。
  4. step<->CTU 只有一個常數。
"""
from __future__ import annotations

import types

import pytest
import torch

from lbmnca.nca.tes.inference import protocol


def _cfg(mode: str = "ceiling", apply_ic_mask: bool = True, dilate: int = 0):
    """最小 config stub（僅含協定所需欄位；直接存取，缺欄位即報錯）。"""
    return types.SimpleNamespace(
        apply_ic_mask=apply_ic_mask,
        mean_chain_handoff_dilate_px=dilate,
        model=types.SimpleNamespace(
            static_dim=4, physics_dim=9, hidden_dim=8,
            transport=types.SimpleNamespace(ke_anchor_mode=mode),
        ),
    )


class _StubModel:
    """只實作協定會用到的 init_state / apply_ic_mask（語意同 HBaseNCA）。"""

    def __init__(self, sd=4, pd=9, hd=8):
        self._cfg = types.SimpleNamespace(static_dim=sd, physics_dim=pd, hidden_dim=hd)

    def init_state(self, static, ic_physics=None):
        b, _, h, w = static.shape
        c = self._cfg
        phys = (torch.zeros(b, c.physics_dim, h, w) if ic_physics is None else ic_physics)
        hid = torch.zeros(b, c.hidden_dim, h, w)
        return torch.cat([static, phys, hid], dim=1)

    def apply_ic_mask(self, state, solid_mask):
        c = self._cfg
        dyn = state[:, c.static_dim:].clone()
        dyn[solid_mask.expand(-1, c.physics_dim + c.hidden_dim, -1, -1)] = 0.0
        return torch.cat([state[:, : c.static_dim], dyn], dim=1)


def _fields(h=8, w=8):
    static = torch.zeros(1, 4, h, w)
    static[:, 0] = 1.0                 # SDF>0 -> fluid
    static[:, 0, :, :2] = -1.0         # 左兩欄為固體
    fluid = (static[:, 0:1] > 0.0).float()
    phys = torch.randn(1, 9, h, w)
    return static, phys, fluid


# ── 1. 錨點是起始 state 的純函式 ────────────────────────────────────────
def test_resolve_ke_anchor_off_returns_none():
    static, phys, fluid = _fields()
    model = _StubModel()
    state = model.init_state(static, phys)
    assert protocol.resolve_ke_anchor(_cfg("off"), state, fluid) is None


def test_resolve_ke_anchor_is_pure_function_of_state():
    """同一 state 兩次呼叫結果相同；不同 state 得不同值（無跨呼叫累積）。"""
    static, phys, fluid = _fields()
    model = _StubModel()
    cfg = _cfg("ceiling")
    s1 = model.init_state(static, phys)
    s2 = model.init_state(static, phys * 0.5)
    a1a = protocol.resolve_ke_anchor(cfg, s1, fluid)
    a1b = protocol.resolve_ke_anchor(cfg, s1, fluid)
    a2 = protocol.resolve_ke_anchor(cfg, s2, fluid)
    assert torch.equal(a1a, a1b)                     # 冪等
    assert not torch.allclose(a1a, a2)               # 隨輸入改變
    # 先呼叫 s2 再呼叫 s1 仍得原值（部署端舊 running-min 會在此失敗）
    assert torch.equal(protocol.resolve_ke_anchor(cfg, s1, fluid), a1a)


def test_resolve_ke_anchor_rejects_unknown_mode():
    static, phys, fluid = _fields()
    state = _StubModel().init_state(static, phys)
    with pytest.raises(ValueError):
        protocol.resolve_ke_anchor(_cfg("running_min"), state, fluid)


# ── 2. ceiling 模式漏傳 anchor 必須報錯（禁靜默 bypass）──────────────────
def test_arch_raises_when_ceiling_without_anchor():
    from lbmnca.nca.tes.config import TESModelConfig
    from lbmnca.nca.tes.models.model_factory import build_tes_model

    mcfg = TESModelConfig(
        arch="h3t_streamcollide", state_dim=21, static_dim=4, physics_dim=9,
        hidden_dim=8, cond_dim=19, g_vec_dim=0, gvec_mode="none",
        encoder_hidden=16, nca_channels=8,
    )
    mcfg.transport.ke_anchor_mode = "ceiling"
    model = build_tes_model(mcfg).eval()
    static, phys, fluid = _fields(16, 16)
    state = model.init_state(static, phys)
    cond = torch.zeros(1, 19)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="ke_anchor"):
            model(state, cond, fluid, 1, return_trajectory=False)


# ── 3. chain 交接保留 hidden；init_state 交接歸零 ───────────────────────
def test_chain_handoff_preserves_hidden():
    static, phys, fluid = _fields()
    model = _StubModel()
    cfg = _cfg("off")
    prev = model.init_state(static, phys)
    prev[:, 13:] = 0.7                      # 給 hidden 一個可辨識的非零值
    out = protocol.chain_handoff(model, cfg, prev, static, fluid, fluid)
    hidden = out[:, 13:]
    assert float(hidden[fluid.expand_as(hidden) > 0.5].abs().max()) > 0.0, \
        "chain_handoff 不得清掉流體區的 hidden"
    # 固體區必須歸零
    solid = (fluid <= 0.5).expand_as(hidden)
    assert float(hidden[solid].abs().max()) == 0.0


def test_init_state_handoff_zeroes_hidden_contrast():
    """對照組：舊評估腳本的 init_state 交接會把 hidden 歸零。"""
    static, phys, fluid = _fields()
    model = _StubModel()
    prev = model.init_state(static, phys)
    prev[:, 13:] = 0.7
    out = model.init_state(static, prev[:, 4:13])
    assert float(out[:, 13:].abs().max()) == 0.0


def test_initial_state_respects_apply_ic_mask():
    static, phys, fluid = _fields()
    model = _StubModel()
    solid = (fluid <= 0.5).expand(-1, 9, -1, -1)
    on = protocol.initial_state(model, _cfg("off", apply_ic_mask=True), static, phys, fluid)
    off = protocol.initial_state(model, _cfg("off", apply_ic_mask=False), static, phys, fluid)
    assert float(on[:, 4:13][solid].abs().max()) == 0.0
    assert float(off[:, 4:13][solid].abs().max()) > 0.0


# ── 4. step <-> CTU 單一常數 ────────────────────────────────────────────
def test_step_ctu_roundtrip():
    assert protocol.NOMINAL_DELTA_CTU_PER_STEP == 1.0
    assert protocol.steps_to_ctu(500) == 500.0
    assert protocol.ctu_to_steps(protocol.steps_to_ctu(37)) == 37
