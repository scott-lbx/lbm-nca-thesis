"""多來源 TES Zarr 混訓的執行期全域統計量建構模組。

保持各來源 zarr 分開儲存的同時，提供 Trainer 一個共用的正規化空間。
來源正規化後的張量仍需透過下列流程轉換：

    source norm → raw → dataset-global norm
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import zarr

from .graph_index import GraphIndex, is_v4_zarr


def resolve_stats_paths(zarr_paths: list[str | Path], stats_paths: list[str] | None) -> list[str]:
    """將 stats JSON 路徑與 zarr_paths 對齊並解析。

    若 stats_paths 已明確提供，驗證長度後直接回傳。
    否則以 zarr 檔名自動推導（foo.zarr → foo_stats.json）。

    Args:
        zarr_paths: zarr 資料集路徑列表。
        stats_paths: 顯式指定的 stats JSON 路徑列表；None 表示自動推導。

    Returns:
        與 zarr_paths 一一對應的 stats 路徑字串列表。

    Raises:
        ValueError: stats_paths 長度與 zarr_paths 不一致。
        FileNotFoundError: 自動推導路徑不存在。
    """
    if stats_paths:
        if len(stats_paths) != len(zarr_paths):
            raise ValueError(
                "stats_paths 長度必須等於 zarr_paths 長度 "
                f"({len(stats_paths)} != {len(zarr_paths)})"
            )
        return [str(p) for p in stats_paths]

    resolved: list[str] = []
    for zp in zarr_paths:
        p = Path(zp)
        if p.suffix == ".zarr":
            cand: Path = p.with_name(f"{p.stem}_stats.json")
        else:
            cand = Path(str(p) + "_stats.json")
        if not cand.exists():
            raise FileNotFoundError(
                f"無法自動推導 stats_path: {cand}. 請在 config.stats_paths 明確指定。"
            )
        resolved.append(str(cand))
    return resolved


def _load_json(path: str | Path) -> dict:
    """讀取並解析 JSON 檔案。

    Args:
        path: JSON 檔案路徑。

    Returns:
        解析後的字典物件。
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _case_count(zarr_path: str | Path, dataset_names: Optional[Iterable[str]]) -> int:
    """計算 zarr 中符合 dataset_names 過濾條件的 case 數量。

    供加權合併統計量時計算各來源的樣本權重。

    Args:
        zarr_path: zarr 資料集路徑。
        dataset_names: 要納入計算的 dataset 名稱集合；None 表示不過濾。

    Returns:
        符合條件的 case 數量（至少為 1，避免除以零）。

    Note:
        v4（node/edge 分離）無 `cases/` group，改以 topology 的 edge 數作為權重
        —— edge 是 v4 的訓練樣本單位，等同 v3 的 case。格式判別走
        `graph_index.is_v4_zarr()`，同時涵蓋舊佈局（獨立 `{ds}_graph.json`）
        與 unified 單檔（topology 在 `.zattrs`）。

        ★ 舊佈局的權重只數 **train** edge。原本數全部
        edge，使 val/test 的樣本數影響跨 dataset 的統計加權 —— 屬正規化統計的
        val/test 成分之一。`split` 欄未填妥時直接 raise（禁預設值）。

        ★unified（A 方案）：資料層已無 `split` 欄，整本 dataset 屬單一角色，
        由 config 的 `zarr_paths` 決定。呼叫端（`build_tes_trainer`）只把 train
        角色的 zarr 傳進來，故此處全部 edge 即全部 train edge，直接數總數。

    Raises:
        ValueError: v4 舊佈局 graph.json 的 `split` 欄未填妥。
    """
    if is_v4_zarr(zarr_path):
        graph = GraphIndex.for_zarr(zarr_path)
        if dataset_names and graph.dataset_id not in set(dataset_names):
            return 1
        if graph.is_unified:
            return max(len(graph.edges), 1)
        valid = {"train", "val", "test", "dropped"}
        bad = [eid for eid, e in graph.edges.items() if e["split"] not in valid]
        if bad:
            raise ValueError(
                f"{zarr_path} 的 graph split 欄未填妥（{len(bad)} 條 edge，例 {sorted(bad)[:2]}）。"
                "統計加權必須只數 train edge；"
                "請先於資料生成端回填 split 欄（見 docs/DATA_GENERATION.md）。"
            )
        return max(sum(1 for e in graph.edges.values() if e["split"] == "train"), 1)

    ds_filter: Optional[set[str]] = set(dataset_names) if dataset_names else None
    root: zarr.Group = zarr.open(str(zarr_path), mode="r")
    n: int = 0
    for key in root["cases"].keys():
        attrs: dict = dict(root["cases"][key].attrs)
        if ds_filter is not None and attrs.get("dataset_name", "") not in ds_filter:
            continue
        n += 1
    return max(n, 1)


def _combine_mean_std(
    means: list[np.ndarray],
    stds: list[np.ndarray],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """以加權方式合併多組 mean / std 統計量。

    利用二階矩公式：E[X²] = Var[X] + E[X]²，將各來源的統計量合併
    為加權全局統計量，避免資訊損失。

    Args:
        means: 各來源的 mean 陣列列表（shape 須一致）。
        stds: 各來源的 std 陣列列表（shape 須一致）。
        weights: 各來源的樣本數權重陣列（shape [N]）。

    Returns:
        (global_mean, global_std) float32 陣列元組，std 含 1e-8 下限。
    """
    w: np.ndarray = weights.astype(np.float64)
    w = w / max(float(w.sum()), 1.0)
    mean_stack: np.ndarray = np.stack([m.astype(np.float64) for m in means], axis=0)
    std_stack: np.ndarray = np.stack([s.astype(np.float64) for s in stds], axis=0)
    mean: np.ndarray = (mean_stack * w[:, None]).sum(axis=0)
    second: np.ndarray = ((std_stack ** 2 + mean_stack ** 2) * w[:, None]).sum(axis=0)
    var: np.ndarray = np.maximum(second - mean ** 2, 0.0)
    return mean.astype(np.float32), (np.sqrt(var).astype(np.float32) + 1e-8)


def _combine_scalar_stats(
    stats: list[dict],
    section: str,
    key: str,
    weights: np.ndarray,
) -> dict:
    """從 stats dict 列表中提取並加權合併指定純量統計量。

    Args:
        stats: 各來源的 stats 字典列表。
        section: stats 的一級鍵名（如 "static_norm"）。
        key: 二級鍵名（如 "upstream"）。
        weights: 各來源的樣本數權重陣列（shape [N]）。

    Returns:
        含 "mean" 與 "std" 鍵的合併統計字典。
    """
    means: list[np.ndarray] = [np.array([s[section][key]["mean"]], dtype=np.float32) for s in stats]
    stds: list[np.ndarray] = [np.array([s[section][key]["std"]], dtype=np.float32) for s in stats]
    mean, std = _combine_mean_std(means, stds, weights)
    return {"mean": float(mean[0]), "std": float(std[0])}


def build_runtime_global_stats(
    stats_paths: list[str | Path],
    zarr_paths: list[str | Path] | None = None,
    dataset_names: Optional[list[str]] = None,
) -> dict:
    """從多個 stats JSON 建立單一記憶體內的全域統計字典。

    物理尺度以各通道最大值保守合併；static 與條件向量的 z-score 統計量
    在有 zarr_paths 時以 case 數加權合併，否則等權重合併。

    Args:
        stats_paths: 各來源的 stats JSON 路徑列表。
        zarr_paths: 對應的 zarr 路徑列表（用於計算加權合併的 case 數權重）；
            None 表示等權重合併。
        dataset_names: 計算 case 數時的 dataset 名稱過濾列表；None = 全部。

    Returns:
        全域統計字典，包含 snapshot_norm、physics_norm、static_norm、cond_stats 等欄位。

    Raises:
        ValueError: stats_paths 為空，或 stats 缺少 tes_v3 必要的 snapshot_norm 鍵，
            或 M_REF 在不同來源間不一致。
    """
    if not stats_paths:
        raise ValueError("build_runtime_global_stats requires at least one stats_path")

    stats: list[dict] = [_load_json(p) for p in stats_paths]
    weights: np.ndarray
    if zarr_paths:
        weights = np.array([_case_count(zp, dataset_names) for zp in zarr_paths], dtype=np.float64)
    else:
        weights = np.ones(len(stats), dtype=np.float64)

    first: dict = stats[0]
    out: dict = {
        "version": "tes_runtime_global",
        "runtime_global": True,
        "source_stats_paths": [str(p) for p in stats_paths],
    }

    if "snapshot_norm" not in first:
        raise ValueError(
            "build_runtime_global_stats requires tes_v3 stats (snapshot_norm key missing). "
            f"Found keys: {list(first.keys())}"
        )

    out["snapshot_norm"] = first["snapshot_norm"]  # metadata only, same across all v3 stats

    # vis_range：global SSOT colormap 範圍（跨全部 dataset 同一值）→ 直接取 first（strict，缺即報錯）
    out["vis_range"] = first["vis_range"]

    # ── physics 正規化 ────────────────────────────────────────────────────
    # ★ 凍結 schema（`frozen_norm`）：常數寫死在 `src/lbmnca/lbm/post_process/norm_frozen.py`，
    #   跨 dataset 逐位元相同是**必然**而非巧合 —— 故此處不做任何聚合，
    #   只斷言一致。若不一致，代表某本是用不同版本的程式碼轉的，
    #   那比數值差異更嚴重（同一個 tensor 值在兩本裡對應不同的物理量）。
    #
    # 舊 schema 沒有 `frozen_norm`，其常數由各自資料量測而來 → 沿用既有的
    # elementwise max 聚合。兩條路徑不可混用，故先確認全部同代。
    frozen = [("frozen_norm" in s) for s in stats]
    if any(frozen) and not all(frozen):
        raise ValueError(
            "stats_paths 混用了凍結 schema 與舊的量測 schema —— "
            "兩者的 S 相差 4-40 倍，混用不會報錯只會靜默產生不可比的結果。")

    if all(frozen):
        ref = first["frozen_norm"]
        for idx, s in enumerate(stats[1:], start=1):
            fz = s["frozen_norm"]
            if (fz["schema_version"] != ref["schema_version"]
                    or fz["S"] != ref["S"] or fz["M_REF"] != ref["M_REF"]):
                raise ValueError(
                    f"frozen_norm 在 stats[0] 與 stats[{idx}] 之間不一致："
                    f"{ref['schema_version']} vs {fz['schema_version']}")
        out["physics_norm"] = {
            "M_REF": list(ref["M_REF"]),
            "S_combined": list(ref["S"]),
        }
        out["frozen_norm"] = ref
    else:
        m_refs: list[np.ndarray] = [
            np.array(s["physics_norm"]["M_REF"], dtype=np.float32) for s in stats]
        for idx, m_ref in enumerate(m_refs[1:], start=1):
            if not np.allclose(m_refs[0], m_ref, atol=1e-6):
                raise ValueError(
                    f"physics_norm.M_REF differs between stats[0] and stats[{idx}]")
        s_arr: np.ndarray = np.maximum.reduce(
            [np.array(s["physics_norm"]["S"], dtype=np.float32) for s in stats])
        s_combined: np.ndarray = np.maximum.reduce([
            np.array(s["physics_norm"]["S_combined"], dtype=np.float32) for s in stats
        ])
        out["physics_norm"] = {
            "U_ref": float(max(float(s["physics_norm"].get("U_ref", 0.0)) for s in stats)),
            "M_REF": m_refs[0].tolist(),
            "S": s_arr.tolist(),
            "rescale_abs_max": (s_combined / np.maximum(s_arr, 1e-8)).astype(np.float32).tolist(),
            "S_combined": s_combined.tolist(),
        }

    # v3 static_norm: sdf=tanh_lchar (passthrough), pos_x/y=raw_01 (passthrough)
    out["static_norm"] = {
        "sdf": {"method": "tanh_lchar"},
        "upstream": _combine_scalar_stats(stats, "static_norm", "upstream", weights),
        "pos_x": {"method": "raw_01"},
        "pos_y": {"method": "raw_01"},
    }

    g_mean, g_std = _combine_mean_std(
        [np.array(s["cond_stats"]["g_vec"]["mean"], dtype=np.float32) for s in stats],
        [np.array(s["cond_stats"]["g_vec"]["std"], dtype=np.float32) for s in stats],
        weights,
    )
    p_mean, p_std = _combine_mean_std(
        [np.array(s["cond_stats"]["p_vec"]["mean"], dtype=np.float32) for s in stats],
        [np.array(s["cond_stats"]["p_vec"]["std"], dtype=np.float32) for s in stats],
        weights,
    )
    out["cond_stats"] = {
        "g_vec": {"mean": g_mean.tolist(), "std": g_std.tolist()},
        "p_vec": {"mean": p_mean.tolist(), "std": p_std.tolist()},
    }
    return out
