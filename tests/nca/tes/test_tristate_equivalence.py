"""三態（train / eval / deploy）真實路徑等價性 —— DEEP_AUDIT R3-G5 / GATE 3 判準 G3-6。

`test_inference_protocol.py` 只用 stub model 驗協定函式**自身**的不變量，從未比對三條
**真實路徑**的輸出；唯一比對過的是一次性腳本
``docs/reviews/DEEP_AUDIT_2026-07/tools/stage3_g36_verify.py``（不在測試套件內，回歸時
沒有任何東西會再跑它）。本檔把該驗證收進 pytest：

    TRAIN  : **不是** ``MeanStep`` 本體，而是就地重寫一次 ``mean_step`` 目前用的協定
             呼叫序列（``protocol.initial_state`` → ``protocol.resolve_ke_anchor`` →
             ``model(...)``）。見下方「本檔抓不到什麼」。
    EVAL   : ``NYCU_Thesis/figures/plots/_v4_infer.V4Infer``（CH4/CH5 數字來源）
    DEPLOY : ``src.inference.backends.tes_backend.TESBackend``，經 ``prepare()`` →
             ``step()`` 的真實入口（與 03 端 ``TESAdapter`` 共用同一份
             ``tes.inference.protocol``）

斷言：單階與三階鏈式的 ``train_vs_eval`` / ``train_vs_deploy`` / ``eval_vs_deploy``
max|Δ| 皆為 **0.0**（同一算子，逐位元相同）。

★★ 本檔抓不到什麼（2026-07-28 二輪誠實補述）：TRAIN 臂沒有實際建 ``MeanStep`` /
``TrainContext`` 跑一次訓練 step——那需要 optimizer、loss、pool 等一整套訓練狀態，
在推論用測試裡建不起來。因此 TRAIN 臂驗的是「協定函式組合起來的結果」，若
``mean_step`` 日後自行改寫 IC 或錨點而不再呼叫協定，**本測試不會變紅**。要真的釘住
``mean_step``，須把它建 state 的那段抽成可測函式再由此呼叫（尚未做）。
EVAL 與 DEPLOY 兩臂則是走真實入口（``V4Infer.rollout`` / ``TESBackend.prepare+step``）。

★ 需 GPU + canonical ckpt + C2-2 graph zarr；任一缺件即 ``skip``（CPU-only CI 不會紅）。
  以 ``pytest -m gpu tests/tes/test_tristate_equivalence.py`` 手動執行；訓練進行中勿跑
  （單一訓練規則 + 高 VRAM 訓練禁 GPU pytest）。
  ``gpu`` marker 於 ``tests/conftest.py`` 註冊（``--strict-markers`` 下不會 error）。

★ 本檔尚未實跑（round-1/2 皆只做 collect-only，GPU 被訓練佔用），max|Δ|==0.0 是否
  成立**未經驗證**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
THESIS_PLOTS = WORKSPACE / "NYCU_Thesis" / "figures" / "plots"

CONFIG = REPO_ROOT / "configs" / "SFA" / "main_v4.yaml"
CKPT = (REPO_ROOT / "runs" / "SFA2" / "main_v4"
        / "ckpt_final.pth")
GRAPH_ZARR = WORKSPACE / "dataset_zarr" / "C2-2_WarmGraph_M180C300R20.zarr"
K = 8                    # 每 stage rollout 步數
N_CHAIN_STAGES = 3

_MISSING = [
    str(p) for p in (CONFIG, CKPT, GRAPH_ZARR) if not p.exists()
]
_SKIP_REASON = (
    "需 CUDA" if not torch.cuda.is_available()
    else (f"缺件：{_MISSING}" if _MISSING else "")
)
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(bool(_SKIP_REASON),
                                                  reason=_SKIP_REASON or "ok")]


# ──────────────────────────────────────────────────────────────────────
# 共用建置（session 級：建 trainer / 載 ckpt 很慢，三個測試共用一份）
# ──────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def env():
    """建 model / backend / V4Infer 與 renorm 後的 node 張量。

    Returns:
        dict：cfg / model / backend / inf / nids / stages / conds / device 等。
    """
    import numpy as np
    import zarr

    if str(THESIS_PLOTS) not in sys.path:
        sys.path.insert(0, str(THESIS_PLOTS))

    from lbmnca.nca.tes.config import load_tes_config
    from lbmnca.nca.tes.data.graph_index import GraphIndex
    from lbmnca.nca.tes.data.normalizer import TESCondNormalizer
    from lbmnca.nca.tes.data.runtime_global_stats import (
        build_runtime_global_stats,
        resolve_stats_paths,
    )
    from lbmnca.nca.tes.models.model_factory import build_tes_model
    from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer

    device = torch.device("cuda")
    cfg = load_tes_config(str(CONFIG))
    zarr_paths = [str(zp) for zp in cfg.zarr_paths]
    stats = resolve_stats_paths(zarr_paths, cfg.stats_paths)
    gstats = build_runtime_global_stats(stats, zarr_paths=zarr_paths, dataset_names=None)
    gnorm = FieldNormalizer.from_stats_dict(gstats).to(device)
    src_idx = next(i for i, zp in enumerate(zarr_paths)
                   if Path(zp).stem == GRAPH_ZARR.stem)
    snorm = FieldNormalizer.from_file(stats[src_idx]).to(device)
    cnorm = TESCondNormalizer.from_stats_dict(gstats)

    graph = GraphIndex.for_zarr(str(GRAPH_ZARR))
    root = zarr.open(str(GRAPH_ZARR), mode="r")
    nids = sorted(root["nodes"].group_keys())[:N_CHAIN_STAGES]

    def _node(nid: str):
        """讀 node → (static_global, physics_global, g_vec, p_vec, fluid_mask, 原張量)。"""
        st_s = torch.from_numpy(
            np.asarray(root["nodes"][nid]["static"][:]))[None].to(device).float()
        ph_s = torch.from_numpy(
            np.asarray(root["nodes"][nid]["mean_flow"][:]))[None].to(device).float()
        st = st_s.clone()
        st[:, 0] = gnorm.norm_sdf(snorm.denorm_sdf(st_s[:, 0]))
        st[:, 1] = gnorm.norm_upstream(snorm.denorm_upstream(st_s[:, 1]))
        st[:, 2] = gnorm.norm_pos_x(snorm.denorm_pos_x(st_s[:, 2]))
        st[:, 3] = gnorm.norm_pos_y(snorm.denorm_pos_y(st_s[:, 3]))
        ph = gnorm.norm_mphy(snorm.denorm_mphy(ph_s))
        gv = torch.tensor(graph.g_vec(nid), dtype=torch.float32, device=device)[None]
        pv = torch.tensor(graph.p_vec(nid), dtype=torch.float32, device=device)[None]
        return st, ph, gv, pv, (st[:, 0:1] > 0.0).float(), st_s, ph_s

    stages = [_node(n) for n in nids]
    conds = [
        (cnorm.build_cond(cfg.model.cond_dim, gv, pv, None, None),
         (cnorm.normalize_g(gv)
          if cfg.model.gvec_mode in ("additive", "gated") and cfg.model.g_vec_dim > 0
          else None))
        for (_, _, gv, pv, _, _, _) in stages
    ]

    model = build_tes_model(cfg.model).to(device)
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    from _v4_infer import V4Infer  # noqa: E402（需 thesis plots 在 sys.path）
    from lbmnca.nca.inference.backends.tes_backend import TESBackend

    inf = V4Infer(str(CONFIG), str(CKPT), on_identity_mismatch="warn")
    dep_cfg = {
        "model": {"tes_run_config": str(CONFIG), "checkpoint": str(CKPT),
                  "delta_ctu": 1.0},
        "repos": {},
    }
    st0 = stages[0][0]
    backend = TESBackend(int(st0.shape[-2]), int(st0.shape[-1]), dep_cfg,
                         device=str(device))
    backend.model.load_state_dict(ck["model"], strict=False)
    backend.model.eval()

    return {
        "cfg": cfg, "model": model, "backend": backend, "inf": inf,
        "nids": nids, "stages": stages, "conds": conds, "device": device,
        "src_idx": src_idx,
    }


class _DeployGeo:
    """``GeometryInputs`` 佔位：``static_raw`` 不被使用（``_normalize_static`` 已被取代），
    ``cond_raw`` 則是 ``TESBackend._build_cond`` 真正會讀的欄位。"""

    def __init__(self, st: torch.Tensor, gv: torch.Tensor, pv: torch.Tensor) -> None:
        self.static_raw = st[0].cpu()
        self.cond_raw = {"g_vec": gv[0].cpu().tolist(),
                         "p_vec": pv[0].cpu().tolist()}


def _train_single(env_) -> torch.Tensor:
    """TRAIN 臂單階輸出 physics（就地重寫 mean_step 目前用的協定序列，非 MeanStep 本體）。"""
    from lbmnca.nca.tes.inference import protocol

    cfg, model = env_["cfg"], env_["model"]
    st, ph, _gv, _pv, fm, _, _ = env_["stages"][0]
    cond, gvn = env_["conds"][0]
    sd, pd = int(cfg.model.static_dim), int(cfg.model.physics_dim)
    with torch.inference_mode():
        s0 = protocol.initial_state(model, cfg, st, ph, fm)
        anchor = protocol.resolve_ke_anchor(cfg, s0, fm)
        s = model(s0, cond, fm, K, return_trajectory=False, g_vec_norm=gvn,
                  ke_anchor=anchor)
    return s[:, sd:sd + pd].float().clone()


def _eval_single(env_) -> torch.Tensor:
    """EVAL 路徑（論文 harness V4Infer.rollout）單階輸出 physics。"""
    inf = env_["inf"]
    _st, _ph, gv, pv, _fm, st_s, ph_s = env_["stages"][0]
    case = inf.prep_raw(st_s.cpu(), ph_s.cpu(), ph_s.cpu(), gv.cpu(), pv.cpu(),
                        env_["src_idx"], case_key=env_["nids"][0])
    return inf.rollout(case, K).float().clone()


def _deploy_single(env_) -> torch.Tensor:
    """DEPLOY 路徑（TESBackend.prepare(warm=True) → step）單階輸出 physics。

    以 monkeypatch ``_normalize_static`` 注入已 renorm 的 static（繞過 03 端 raw→norm
    前處理），確保比較的是「協定差異」而非「輸入差異」；IC 由 ``_steady`` 供給，走
    ``prepare()`` 的 warm 分支——**不繞過 prepare()**，否則 prepare 內的協定邏輯
    （initial_state / chain_handoff / resolve_ke_anchor）根本沒被測到，斷言會退化成
    「protocol 等於 protocol」的恆真式。

    ★``_steady_mask=None`` ＝「無有效舊幾何」，此時 ``chain_handoff`` 的重置區僅取
    新固體。**但兩者結果相同只在 ``cfg.apply_ic_mask=True`` 時成立**——``chain_handoff``
    無條件對重置區套 ``model.apply_ic_mask``，而 ``initial_state`` 只在
    ``cfg.apply_ic_mask`` 為真時才清固體區 IC。canonical ``main_v4.yaml``
    的 ``apply_ic_mask: true``（2026-07-28 實測），本檔也只在該 config 下執行；若換成
    ``apply_ic_mask: false`` 的 config，本臂與 TRAIN/EVAL 的 stage 0 **不再可比**，
    斷言會因這個已知原因而失敗，不是三態真的分歧。
    """
    from lbmnca.nca.tes.inference import protocol

    cfg, backend = env_["cfg"], env_["backend"]
    st, ph, gv, pv, fm, _, _ = env_["stages"][0]
    sd, pd = int(cfg.model.static_dim), int(cfg.model.physics_dim)

    backend._normalize_static = (lambda _g, _st=st: _st)  # noqa: E731
    backend._state = None
    backend._mask = fm
    backend._steady = protocol.initial_state(
        backend.model, backend.run_cfg, st, ph, fm)
    backend._steady_mask = None
    backend.prepare(_DeployGeo(st, gv, pv), warm=True)
    backend.step(K)
    return backend._state[:, sd:sd + pd].float().clone()


def _train_chain(env_) -> torch.Tensor:
    """TRAIN 臂三階鏈式（就地重寫協定序列，含 chain_handoff）末階 physics。"""
    from lbmnca.nca.tes.inference import protocol

    cfg, model = env_["cfg"], env_["model"]
    sd, pd = int(cfg.model.static_dim), int(cfg.model.physics_dim)
    prev_state, prev_fm = None, None
    with torch.inference_mode():
        for j, (st, ph, _gv, _pv, fm, _, _) in enumerate(env_["stages"]):
            cond, gvn = env_["conds"][j]
            state = (protocol.initial_state(model, cfg, st, ph, fm) if j == 0
                     else protocol.chain_handoff(model, cfg, prev_state, st, fm, prev_fm))
            anchor = protocol.resolve_ke_anchor(cfg, state, fm)
            state = model(state, cond, fm, K, return_trajectory=False,
                          g_vec_norm=gvn, ke_anchor=anchor)
            prev_state, prev_fm = state, fm
    return prev_state[:, sd:sd + pd].float().clone()


def _eval_chain(env_) -> torch.Tensor:
    """EVAL 路徑三階鏈式（V4Infer.chain_handoff + rollout）末階 physics。"""
    cfg, inf = env_["cfg"], env_["inf"]
    sd, pd = int(cfg.model.static_dim), int(cfg.model.physics_dim)
    prev_state, prev_fm = None, None
    for j, nid in enumerate(env_["nids"]):
        _st, _ph, gv, pv, _fm, st_s, ph_s = env_["stages"][j]
        case = inf.prep_raw(st_s.cpu(), ph_s.cpu(), ph_s.cpu(), gv.cpu(), pv.cpu(),
                            env_["src_idx"], case_key=nid)
        state0 = (case.state0 if j == 0 else
                  inf.chain_handoff(prev_state, case.static, case.fluid_mask, prev_fm))
        inf.rollout(case, K, state0=state0)
        prev_state, prev_fm = inf.last_state, case.fluid_mask
    return prev_state[:, sd:sd + pd].float().clone()


def _deploy_chain(env_) -> torch.Tensor:
    """DEPLOY 路徑三階鏈式（TESBackend.prepare(warm) + save_steady）末階 physics。"""
    from lbmnca.nca.tes.inference import protocol

    cfg, backend = env_["cfg"], env_["backend"]
    sd, pd = int(cfg.model.static_dim), int(cfg.model.physics_dim)

    for j, (st, ph, gv, pv, fm, _, _) in enumerate(env_["stages"]):
        backend._normalize_static = (lambda _g, _st=st: _st)  # noqa: E731
        if j == 0:
            backend._state = None
            backend._mask = fm
            backend._steady = protocol.initial_state(
                backend.model, backend.run_cfg, st, ph, fm)
            backend._steady_mask = None
        else:
            backend.save_steady()
        backend.prepare(_DeployGeo(st, gv, pv), warm=True)
        backend.step(K)
    return backend._state[:, sd:sd + pd].float().clone()


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


# ──────────────────────────────────────────────────────────────────────
def test_single_stage_tristate_identical(env):
    """單階：三態對同一輸入必須逐位元相同（G3-6）。"""
    tr, ev, dp = _train_single(env), _eval_single(env), _deploy_single(env)
    assert _maxdiff(tr, ev) == 0.0, "train vs eval 分歧（單階）"
    assert _maxdiff(tr, dp) == 0.0, "train vs deploy 分歧（單階）"
    assert _maxdiff(ev, dp) == 0.0, "eval vs deploy 分歧（單階）"


def test_chain_tristate_identical(env):
    """三階鏈式：交接（hidden / ic_mask / 錨點）是三態分歧的集中處。"""
    tr, ev, dp = _train_chain(env), _eval_chain(env), _deploy_chain(env)
    assert _maxdiff(tr, ev) == 0.0, "train vs eval 分歧（鏈式）"
    assert _maxdiff(tr, dp) == 0.0, "train vs deploy 分歧（鏈式）"
    assert _maxdiff(ev, dp) == 0.0, "eval vs deploy 分歧（鏈式）"
