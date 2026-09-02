"""v3/v4 格式判別的回歸測試（LBM_NCA-1 前置修正）。

背景：`train.py` 與 `runtime_global_stats._case_count()` 原本各自以
「同名 `{stem}_graph.json` 是否存在」判別 v4。該判準只認舊佈局
（`{ds}_v4.zarr` ＋ 獨立 graph.json），漏掉 unified 單檔
（`{ds}_unified.zarr`，topology 存在 `.zattrs`）——D 系列六本全是後者，
於是被誤判成 v3，落進 `cases/` 分支後以 `KeyError: 'cases'` 收場。

本檔用**臨時建的最小 zarr**，不依賴 `dataset_zarr/` 的實體資料
（那些是 gitignored 的 156 GB，CI／他機無法假設存在）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import zarr

from lbmnca.nca.tes.data.graph_index import GraphIndex, is_v4_zarr
from lbmnca.nca.tes.data.runtime_global_stats import _case_count


def _make_unified_zarr(path: Path, n_edges: int = 3) -> None:
    """建一個最小的 `tes_unified_v1` zarr（topology 在 `.zattrs`，無 graph.json）。"""
    root = zarr.open(str(path), mode="w")
    root.attrs["topology"] = {
        "version": "tes_v4",
        "format": "tes_unified_v1",
        "role_source": "dataset",
        "dataset_id": "UNIT_unified",
        "stats_space": "UNIT_unified",
        "grid": {"H": 4, "W": 4, "n_channels": 9},
        "nodes": [
            {
                "node_id": f"n{i}",
                "mask_id": f"m{i}",
                "re": 100.0,
                "ic_type": "cold",
                "tree_depth": 0,
                "canonical_edge_id": f"e{i}",
                "external": False,
                "n_postconv_frames": 1,
                "g_vec": [0.0] * 14,
                "p_vec": [100.0, 0.05, 20.0],
                "cube_geom": [0.0, 0.0, 0.0],
            }
            for i in range(n_edges)
        ],
        "edges": [
            {
                "edge_id": f"e{i}",
                "src_node": None,
                "dst_node": f"n{i}",
                "edge_type": "chain",
                "window_type": "full",
                "n_transit_frames": 2,
                "ic_source": "cold_start",
                "traj_ref": None,
                "attractor": None,
            }
            for i in range(n_edges)
        ],
    }


def _make_legacy_v4_zarr(zarr_path: Path, graph_json: Path, n_train: int = 2) -> None:
    """建舊佈局 v4：空 zarr ＋ 同名 `_graph.json`（帶 `split` 欄）。"""
    zarr.open(str(zarr_path), mode="w")
    graph = {
        "version": "tes_v4",
        "dataset_id": "UNIT_legacy",
        "stats_space": "UNIT_legacy",
        "postconv_zarr": "unused.zarr",
        "grid": {"H": 4, "W": 4, "n_channels": 9},
        "nodes": [
            {"node_id": "n0", "split": "train", "tree_depth": 0, "external": False,
             "canonical_edge_id": "e0", "g_vec": [0.0] * 14, "p_vec": [1.0, 1.0, 1.0]}
        ],
        "edges": [
            {"edge_id": f"e{i}", "src_node": None, "dst_node": "n0",
             "edge_type": "chain", "window_type": "full",
             "ic_source": "cold_start", "traj_ref": None, "attractor": None,
             "split": "train" if i < n_train else "val"}
            for i in range(n_train + 1)
        ],
    }
    graph_json.write_text(json.dumps(graph), encoding="utf-8")


def _make_v3_zarr(path: Path) -> None:
    """建 v3 扁平格式：只有 `cases/` group，無 topology、無 graph.json。"""
    root = zarr.open(str(path), mode="w")
    cases = root.create_group("cases")
    for i in range(2):
        g = cases.create_group(f"c{i}")
        g.attrs["dataset_name"] = "UNIT_v3"


def test_unified_zarr_detected_as_v4(tmp_path: Path) -> None:
    """★核心回歸：unified 單檔沒有 graph.json，仍必須判為 v4。"""
    p = tmp_path / "UNIT_unified.zarr"
    _make_unified_zarr(p)
    assert not (p.parent / f"{p.stem}_graph.json").exists(), "前提：unified 不該有 graph.json"
    assert is_v4_zarr(p) is True


def test_unified_case_count_uses_edge_total(tmp_path: Path) -> None:
    """unified 無 `split` 欄，權重＝全部 edge 數（整本同一角色）。"""
    p = tmp_path / "UNIT_unified.zarr"
    _make_unified_zarr(p, n_edges=5)
    assert _case_count(p, None) == 5


@pytest.mark.dataset
def test_legacy_v4_still_counts_only_train_edges(tmp_path: Path) -> None:
    """舊佈局的 DEEP_AUDIT Stage 3.3 行為（只數 train edge）不得被改掉。"""
    p = tmp_path / "UNIT_legacy_v4.zarr"
    gj = tmp_path / "UNIT_legacy_v4_graph.json"
    _make_legacy_v4_zarr(p, gj, n_train=2)
    assert is_v4_zarr(p) is True
    assert _case_count(p, None) == 2  # 3 條 edge，其中 1 條是 val


def test_v3_zarr_not_detected_as_v4(tmp_path: Path) -> None:
    """v3 扁平格式必須仍走 `cases/` 分支，否則舊 dataset 全壞。"""
    p = tmp_path / "UNIT_v3.zarr"
    _make_v3_zarr(p)
    assert is_v4_zarr(p) is False
    assert _case_count(p, None) == 2


def test_missing_path_raises_not_false(tmp_path: Path) -> None:
    """路徑打錯要當場炸，不可回 False——否則會被誤當成 v3，錯誤在下游才浮現。"""
    with pytest.raises(FileNotFoundError):
        is_v4_zarr(tmp_path / "does_not_exist.zarr")


def test_graph_index_agrees_with_detector(tmp_path: Path) -> None:
    """判別器與載入器必須同調：`is_v4_zarr` 為 True 時 `for_zarr` 必須載得起來。"""
    p = tmp_path / "UNIT_unified.zarr"
    _make_unified_zarr(p)
    assert is_v4_zarr(p)
    gi = GraphIndex.for_zarr(p)
    assert gi.is_unified is True
    assert len(gi.edges) == 3
