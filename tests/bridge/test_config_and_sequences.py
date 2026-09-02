"""config.yaml / demo.config.json / send_sequence：設定完整性與訊息組裝。

這些測試會抓「config 指到已刪除的 ckpt / dataset」這類死指標——2026-08 一次資料清理
就把 03 的 mean ckpt、IC 生成器、analyze_zarr 資料集全部指空，而載入失敗只在啟動
main.py 時才炸。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lbmnca.paths import resolve_placeholders


@pytest.fixture(scope="module")
def cfg(repo: Path) -> dict:
    with open(repo / "configs" / "bridge" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def demo_cfg(repo: Path) -> dict:
    with open(repo / "demo" / "demo.config.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def send_sequence(repo: Path) -> ModuleType:
    """以檔案路徑載入 ``demo/send_sequence.py``（demo/ 不是套件，不能一般 import）。"""
    path = repo / "demo" / "send_sequence.py"
    spec = importlib.util.spec_from_file_location("_demo_send_sequence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve(repo: Path, rel: str) -> Path:
    """解析 config 路徑，展開 ``lbmnca.paths`` 的 ``${...}`` 佔位符（同 TESAdapter._resolve_path）。"""
    rel = resolve_placeholders(rel)
    p = Path(rel)
    return p if p.is_absolute() else (repo / rel).resolve()


# ── config.yaml：模型接線 ────────────────────────────────────────────────────

def test_backend_is_decompose(cfg):
    assert cfg["model"]["backend"] == "decompose"


@pytest.mark.parametrize("sub", ["mean", "ic_generator"])
def test_model_checkpoint_and_run_config_exist(cfg, repo, sub):
    """★死指標守門：ckpt 與其 02 run yaml 都必須存在。"""
    m = cfg["model"][sub]
    ck = resolve(repo, m["checkpoint"])
    rc = resolve(repo, m["tes_run_config"])
    assert ck.is_file(), f"{sub} checkpoint 不存在：{ck}"
    assert rc.is_file(), f"{sub} tes_run_config 不存在：{rc}"
    # 正規化統計也是死指標守門的一環：缺它就會退回讀 yaml 的絕對 zarr 路徑，
    # 在沒有訓練資料集的機器上直接失敗。
    gs = resolve(repo, m["global_stats"])
    assert gs.is_file(), f"{sub} global_stats 不存在：{gs}"


def test_ic_generator_declares_steps(cfg):
    """ic_steps 是直接存取（禁 fallback）；缺鍵會在 DecomposeAdapter 建構時 KeyError。"""
    assert int(cfg["model"]["ic_generator"]["ic_steps"]) >= 1


def test_ic_generator_is_not_the_steady_ablation_arm(cfg):
    """base_steady 是論文 P_old 消融臂（val mean_rel_l2 1.69），不可當 IC 生成器。

    它單次 forward 吐出約 11×GT 的速度並在 state_clip 飽和，mean 也拉不回來。
    """
    assert "base_steady" not in cfg["model"]["ic_generator"]["tes_run_config"]


def test_mean_run_config_matches_ckpt_series(cfg):
    """mean 的 run config 與 ckpt 必須同一個 series（檔名字首相同），否則 arch/正規化錯配。"""
    m = cfg["model"]["mean"]
    series = Path(m["tes_run_config"]).stem
    assert Path(m["checkpoint"]).stem.startswith(series), (
        f"tes_run_config={m['tes_run_config']!r} 與 checkpoint={m['checkpoint']!r} "
        "檔名字首不一致，可能不是同一個 series"
    )


def test_delta_ctu_matches_protocol_nominal(cfg):
    """step↔CTU 換算只能有一套；不一致會讓 KE 兩窗判準在錯的窗長上判定。"""
    assert float(cfg["model"]["delta_ctu"]) == 1.0


def test_gvec_fallback_disabled(cfg):
    """demo 必須 false：否則 G_vec 計算失敗會無聲退回常數向量。"""
    assert cfg["geometry"]["allow_gvec_fallback"] is False


def test_decompose_stop_keys_present(cfg):
    """DecomposeAdapter 直接存取這三個鍵，缺一即 KeyError。"""
    d = cfg["model"]["decompose"]
    for k in ("stop_metric", "consec_l2_eps", "stop_evidence_full_run"):
        assert k in d
    assert d["stop_metric"] in ("ke_two_window", "consec_l2_phy")


def test_turb_disabled_and_not_dangling(cfg):
    """mean-only：turb 關閉時 ckpt 應為 null（非指向不存在的 run）。"""
    assert cfg["model"]["decompose"]["enable_turb"] is False
    assert cfg["model"]["turb"]["checkpoint"] is None


def test_resolution_is_inference_domain(cfg):
    assert (cfg["resolution"]["H"], cfg["resolution"]["W"]) == (360, 640)


def test_repo_paths_exist(repo):
    """bridge 依賴 morphospace（G_vec）與 nca（TES 模型）子套件都必須存在。

    monorepo 合併後 config.yaml 不再有 ``repos.morphospace_src`` /
    ``repos.nca_cfd_src`` 這種跨 repo 路徑（舊 polyrepo 遺留，已隨 config.yaml 移除）；
    兩者現在都是 ``lbmnca`` 底下的子套件，改直接檢查子套件目錄存在。
    """
    for sub in ("morphospace", "nca"):
        assert (repo / "src" / "lbmnca" / sub).is_dir(), sub


def test_re_levels_cover_three_wind_levels(cfg):
    assert set(cfg["physics"]["re_levels"]) == {1, 2, 3}


# ── demo.config.json ────────────────────────────────────────────────────────

def test_demo_ws_url_matches_server_config(cfg, demo_cfg):
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    assert demo_cfg["ws_url"] == f"ws://{host}:{port}"


def test_demo_sim_domain_matches_ingest_constants(demo_cfg):
    from lbmnca.bridge.core.geometry_ingest import _SIM_H, _SIM_W

    assert demo_cfg["sim"]["width"] == _SIM_W
    assert demo_cfg["sim"]["height"] == _SIM_H


def test_demo_wind_levels_match_config(cfg, demo_cfg):
    vals = {int(w["value"]) for w in demo_cfg["wind_levels"]}
    assert vals == set(cfg["physics"]["re_levels"])
    assert sum(1 for w in demo_cfg["wind_levels"] if w.get("default")) == 1


def test_demo_viz_modes_are_known(demo_cfg):
    known = {"velocity", "vorticity", "pressure", "streamline", "comfort", "ibfv"}
    assert set(demo_cfg["viz_modes"]) <= known
    assert demo_cfg["default_viz_mode"] in demo_cfg["viz_modes"]


def test_sequence_events_are_well_formed(demo_cfg, send_sequence):
    for name, seq in demo_cfg["sequences"].items():
        assert seq["events"], f"{name} 無 event"
        ts = [float(e["t"]) for e in seq["events"]]
        assert ts == sorted(ts), f"{name} event 時間未遞增"
        for ev in seq["events"]:
            assert ev["action"] == "analyze" or ev["action"] in send_sequence._CONTROL_ACTIONS, ev


@pytest.mark.dataset
def test_analyze_zarr_events_point_at_existing_v4_data(demo_cfg, workspace):
    """★死指標守門：analyze_zarr 引用的 dataset 必須在 dataset_zarr_v4 下存在。

    舊 c2-2 序列引用的 C2-2_WarmGraph_* 隨 v3 資料一起被刪，序列因此整條失效。
    """
    seen = 0
    for seq in demo_cfg["sequences"].values():
        for ev in seq["events"]:
            if ev["action"] != "analyze_zarr":
                continue
            seen += 1
            z = workspace / "dataset_zarr_v4" / f"{ev['dataset']}_unified.zarr"
            assert z.is_dir(), f"dataset 不存在：{z}"
            assert (z / "nodes" / ev["case_id"]).is_dir(), \
                f"case 不存在：{ev['dataset']}/{ev['case_id']}"
    assert seen > 0, "沒有任何 analyze_zarr 事件可驗"


def test_warm_chain_injects_ic_only_on_root(demo_cfg):
    """warm chain：只有第一個 case 注入 GT mean_flow，其後純模型接鏈。"""
    for name, seq in demo_cfg["sequences"].items():
        zs = [e for e in seq["events"] if e["action"] == "analyze_zarr"]
        if not zs:
            continue
        assert zs[0]["inject_ic"] is True, name
        assert all(e["inject_ic"] is False for e in zs[1:]), name


# ── send_sequence 訊息組裝 ───────────────────────────────────────────────────

def test_analyze_message_is_flat(send_sequence):
    msg = send_sequence._build_message(
        {"action": "analyze", "mode": "warm",
         "buildings": [{"cx": 100, "cy": 200, "w": 10, "h": 20}]},
        default_wind=2, sim_h=1080, flip_y=False)
    assert msg["action"] == "analyze"
    assert "payload" not in msg, "analyze 必須頂層扁平"
    assert msg["buildings"][0]["cy"] == 200


def test_flip_y_matches_canvas_convention(send_sequence):
    """p5 canvas y 向下、GGUI y 向上 → 送出時 cy' = sim_h − cy。"""
    msg = send_sequence._build_message(
        {"action": "analyze", "buildings": [{"cx": 1, "cy": 200, "w": 3, "h": 4}]},
        default_wind=2, sim_h=1080, flip_y=True)
    assert msg["buildings"][0]["cy"] == 880
    assert msg["buildings"][0]["cx"] == 1        # x 不翻


def test_control_message_is_wrapped(send_sequence):
    msg = send_sequence._build_message({"action": "set_viz_mode", "viz_mode": "velocity"},
                         default_wind=2, sim_h=1080, flip_y=False)
    assert msg == {"action": "set_viz_mode", "payload": {"viz_mode": "velocity"}}


def test_analyze_zarr_control_keys_preserved(send_sequence):
    ev = {"action": "analyze_zarr", "dataset": "D", "case_id": "C",
          "inject_ic": True, "wind_level": 3}
    msg = send_sequence._build_message(ev, default_wind=2, sim_h=1080, flip_y=False)
    assert msg["payload"] == {"dataset": "D", "case_id": "C",
                              "inject_ic": True, "wind_level": 3}


def test_unknown_action_rejected(send_sequence):
    with pytest.raises(ValueError):
        send_sequence._build_message({"action": "launch_missiles"}, default_wind=2,
                       sim_h=1080, flip_y=False)


def test_every_config_sequence_builds(demo_cfg, send_sequence):
    """config 內每個序列的每個 event 都要能組成合法訊息。"""
    for name, seq in demo_cfg["sequences"].items():
        for ev in seq["events"]:
            msg = send_sequence._build_message(ev, default_wind=2, sim_h=1080, flip_y=True)
            assert msg["action"] == ev["action"], name
