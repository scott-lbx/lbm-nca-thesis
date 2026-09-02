"""ψ-curl 動量頭測試（FLUXSTREAM_DESIGN.md v2）。

1. identity-init：psi_curl_momentum=True 但 project_psi zero-init → forward 逐元素
   等於 psi_curl_momentum=False（兩者 jx/jy delta 起步皆為 0）。
2. div-free：隨機初始化 project_psi 權重後，量測輸出 jx/jy 場的離散散度，
   應遠小於「直接輸出 jx/jy」基線（機制驗證，非嚴格 0——邊界 fluid_mask 截斷）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.config import (  # noqa: E402
    MRPConfig,
    TESModelConfig,
    TESMultiScaleConfig,
    TESTransportConfig,
)
from lbmnca.nca.tes.models.h3t_streamcollide_nca import H3TStreamCollideNCA  # noqa: E402


def _cfg(psi_curl_momentum: bool) -> TESModelConfig:
    return TESModelConfig(
        arch="h3t_streamcollide",
        state_dim=21, static_dim=4, physics_dim=9, hidden_dim=8,
        cond_dim=19, encoder_hidden=32, nca_channels=16,
        g_vec_dim=0, gvec_mode="none",
        mrp=MRPConfig(enabled=False),
        multiscale=TESMultiScaleConfig(ratios=[8, 32], fine_substeps=2, scale_condition=True),
        transport=TESTransportConfig(skip=True),  # canonical SFA mean：STREAM 死碼
        psi_curl_momentum=psi_curl_momentum,
    )


def _inputs(cfg, b=2, h=32, w=32):
    static = torch.randn(b, cfg.static_dim, h, w)
    static[:, 0] = 1.0
    ic = torch.randn(b, cfg.physics_dim, h, w) * 0.1
    cond = torch.randn(b, cfg.cond_dim)
    return static, ic, cond


def test_psi_curl_identity_init_matches_baseline() -> None:
    """project_psi zero-init → psi_curl_momentum=True/False forward 逐元素相等。"""
    torch.manual_seed(0)
    model_off = H3TStreamCollideNCA(_cfg(False)).eval()
    torch.manual_seed(0)
    model_on = H3TStreamCollideNCA(_cfg(True)).eval()

    cfg = _cfg(False)
    static, ic, cond = _inputs(cfg)
    fm = model_off.get_fluid_mask(static)
    state = model_off.init_state(static, ic)
    dseq = torch.full((2, 3), 0.2)
    with torch.no_grad():
        o_off = model_off(state, cond, fm, 3, delta_ctu_seq=dseq)
        o_on = model_on(state, cond, fm, 3, delta_ctu_seq=dseq)
    assert torch.allclose(o_off, o_on, atol=1e-6), (
        "psi_curl_momentum=True 但 project_psi zero-init 應逐元素等於 baseline（identity-init 未破壞）"
    )


def test_psi_curl_produces_div_free_momentum() -> None:
    """隨機初始化 project_psi 後，輸出 jx/jy 場散度應遠小於直接輸出基線（fluid 內部）。"""
    torch.manual_seed(1)
    model_on = H3TStreamCollideNCA(_cfg(True)).eval()
    # 打破 zero-init，讓 psi 產生非零貢獻（否則測不到 curl 機制本身）
    for cell in list(model_on.coarse_cells) + [model_on.fine_cell]:
        torch.nn.init.normal_(cell.project_psi.weight, std=0.5)
        torch.nn.init.normal_(cell.project_psi.bias, std=0.1)

    cfg = _cfg(True)
    b, h, w = 1, 48, 48
    static = torch.randn(b, cfg.static_dim, h, w)
    static[:, 0] = 1.0
    ic = torch.randn(b, cfg.physics_dim, h, w) * 0.1
    cond = torch.randn(b, cfg.cond_dim)
    fm = model_on.get_fluid_mask(static)
    # 全 fluid（避免邊界截斷干擾本測試判準；邊界殘差另由 pilot 實測，見設計文件 §1-3）
    fm = torch.ones_like(fm)
    state = model_on.init_state(static, ic)
    dseq = torch.full((b, 1), 1.0)

    with torch.no_grad():
        out = model_on(state, cond, fm, 1, delta_ctu_seq=dseq)

    sd, pd = cfg.static_dim, cfg.physics_dim
    # ★量測 delta（curl 貢獻的增量），非最終場本身——IC 隨機初始化本身散度不為 0 是預期行為，
    # 只有「這一步新增的動量變化」由 curl 構造保證 div-free。
    djx = out[:, sd + 3] - state[:, sd + 3]
    djy = out[:, sd + 5] - state[:, sd + 5]
    # 中央差分散度（內部區域，避開最外緣 1px 的單邊差分誤差）
    ddx = (djx[:, :, 2:] - djx[:, :, :-2]) * 0.5
    ddy = (djy[:, 2:, :] - djy[:, :-2, :]) * 0.5
    div = ddx[:, 1:-1, :] + ddy[:, :, 1:-1]
    div_norm = div.abs().mean().item()

    assert div_norm < 1e-3, (
        f"ψ-curl 增量 Δ(jx,jy) 內部散度應趨近 0（curl 恆等式），實測 {div_norm:.3e}"
    )
