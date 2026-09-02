"""GVEC_ACTIVE_INDICES 定義完整性的回歸測試（DEEP_AUDIT_2026-07 Stage 2.7 後續）。

`gvec_api.GVEC_ACTIVE_INDICES` 是 14D 選取的唯一權威。DEEP_AUDIT_2026-07 發現時，
舊 polyrepo（lbm-nca-thesis）內另有 4 份手抄副本，其中 `src/lbmnca/lbm/post_process/
tes_h5_to_zarr.py` 位於 dataset 生成主線上——改主清單不會傳導、也不會報錯。

合併為單一套件 `lbmnca` 後，那 4 份手抄副本已全部消失：3 份未隨併入本 repo，
`tes_h5_to_zarr.py`（原最危險的那份）與 `gvec_bridge.py` 都已改成直接
``from lbmnca.morphospace.physics_proxy.gvec_api import GVEC_ACTIVE_INDICES``。
canonical 現在只有一份，「多份手抄副本互相漂移」的守衛對象已不存在；本測試改為
守兩件仍然會靜默出錯的事：canonical 本身的 14D 契約值，以及下游呼叫端沒有退化
回本地手抄副本（見 `scripts/qa/verify_gvec_active_indices.py`）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent.parent
_VERIFIER = _REPO_ROOT / "scripts" / "qa" / "verify_gvec_active_indices.py"


def _load_verifier() -> ModuleType:
    """以檔案路徑載入驗證器模組（``scripts/`` 不是套件）。

    Returns:
        已載入的 verify_gvec_active_indices 模組。
    """
    spec = importlib.util.spec_from_file_location("_verify_gvec_active_indices", _VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier() -> ModuleType:
    """驗證器模組 fixture。"""
    assert _VERIFIER.exists(), f"驗證器不存在：{_VERIFIER}"
    return _load_verifier()


def test_verifier_script_exists() -> None:
    """驗證器腳本必須存在（被誤刪時本測試先炸）。"""
    assert _VERIFIER.exists(), (
        f"缺少 {_VERIFIER}；GVEC_ACTIVE_INDICES 的完整性守衛已失效。"
    )


def test_gvec_active_indices_definition_is_sound(verifier: ModuleType) -> None:
    """canonical 索引合法、名實對照表對齊、下游呼叫端未退化為本地副本。"""
    failures = verifier.check_all(_REPO_ROOT, verbose=False)
    assert not failures, (
        "GVEC_ACTIVE_INDICES 定義出現問題：\n  "
        + "\n  ".join(failures)
        + "\n以 src/lbmnca/morphospace/physics_proxy/gvec_api.py 為準修正。"
    )


def test_canonical_is_the_expected_14d(verifier: ModuleType) -> None:
    """canonical 清單即 14D 契約值（維度數變更必須是有意識的決策）。"""
    canon, _ = verifier.read_literal_assign(
        _REPO_ROOT / verifier.CANONICAL[1], verifier.CANONICAL[2]
    )
    assert len(canon) == 14, (
        f"GVEC_DIM 由 14 變為 {len(canon)}。這會改變 zarr 的 g_vec 語義，"
        "且既有 zarr/graph.json 只有 pipeline schema 版本（tes_v4），"
        "沒有 g_vec 選取版本欄位。變更前請先讀 "
        "docs/reviews/DEEP_AUDIT_2026-07/stage2/STAGE2_567_Gvec維度命名守衛.md"
        "（若已隨併入流程歸檔則以本 repo 內對應文件為準）。"
    )
    assert list(canon) == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 21]


def test_verifier_actually_detects_a_downgraded_importer(
    verifier: ModuleType, tmp_path: Path
) -> None:
    """自我檢測：把下游呼叫端改成本地手抄副本，驗證器必須 FAIL（避免守衛恆真而形同虛設）。"""
    rel_paths = [verifier.CANONICAL[1], verifier.FEATURES_FILE]
    rel_paths += [rel for _, rel in verifier.IMPORTERS]

    for rel in rel_paths:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil_copy2(_REPO_ROOT / rel, dst)

    # 鏡像本身應該 PASS
    assert not verifier.check_all(tmp_path, verbose=False), "鏡像複本應與原始一致"

    # 把 dataset 生成主線那份改回本地手抄副本（且刻意錯一個索引）
    victim_rel = verifier.IMPORTERS[0][1]
    victim = tmp_path / victim_rel
    text = victim.read_text(encoding="utf-8-sig")
    injected = (
        "_GVEC_ACTIVE_INDICES = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 20]\n"
        + text
    )
    victim.write_text(injected, encoding="utf-8")

    failures = verifier.check_all(tmp_path, verbose=False)
    assert failures, "驗證器沒抓到人為注入的本地手抄副本 —— 守衛失效"
    assert any("lbm_tes_h5_to_zarr" in f for f in failures)


def shutil_copy2(src: Path, dst: Path) -> None:
    """薄包裝 ``shutil.copy2``，避免在模組頂層 import 未使用的符號散落各處。"""
    import shutil

    shutil.copy2(src, dst)


# ── Stage 2.5：有效條件維度的宣告 vs 資料層 ─────────────────────────────────
# lbm-nca-thesis 為 sibling workspace（見 tests/bridge/conftest.py 的 `workspace` fixture
# 同款推導），持有跨 repo 共用的 C-series canonical 語料；本 repo 內不隨附資料，機器
# 沒有時直接 skip（不是死指標——路徑本身正確，只是資料集非必要依賴）。
_WORKSPACE_ROOT = _REPO_ROOT.parent / "lbm-nca-thesis"
_ZARR_ROOT = _WORKSPACE_ROOT / "dataset_zarr"
_CANON_GRAPH = _ZARR_ROOT / "C1-3_Warm_M300C300R32S100_graph.json"


@pytest.mark.skipif(
    not _CANON_GRAPH.exists(),
    reason=f"本機無 {_ZARR_ROOT}（C-series canonical 語料），跳過資料層比對",
)
def test_effective_dim_declaration_matches_data() -> None:
    """`gvec_api` 宣告的常數維／有效維度必須與 C-series 資料層實測一致。

    這是 Stage 2.5 的守衛：宣告寫在 code 裡，事實在資料裡，兩者不得漂移。
    """
    import json

    import numpy as np

    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from lbmnca.morphospace.physics_proxy.gvec_api import (
        GVEC_DIM,
        GVEC_EFFECTIVE_DIM_C_SERIES,
        GVEC_FEATURE_NAMES,
        GVEC_INERT_DIMS_C_SERIES,
        find_degenerate_dims,
    )

    datasets = [
        "C0-3_Warm_M30C30R4S10",
        "C1-3_Warm_M300C300R32S100",
        "C2-1_loop_5_2",
        "C2-2_WarmGraph_M180C300R20",
    ]
    chunks = []
    for ds in datasets:
        p = _ZARR_ROOT / f"{ds}_graph.json"
        if not p.exists():
            pytest.skip(f"缺 {p.name}")
        graph = json.loads(p.read_text(encoding="utf-8"))
        chunks.append(np.array([n["g_vec"] for n in graph["nodes"]], dtype=np.float64))
    merged = np.concatenate(chunks, axis=0)

    deg = find_degenerate_dims(merged)
    declared = sorted(GVEC_FEATURE_NAMES.index(n) for n in GVEC_INERT_DIMS_C_SERIES)
    assert deg == declared, (
        f"GVEC_INERT_DIMS_C_SERIES 宣告 {[GVEC_FEATURE_NAMES[i] for i in declared]}，"
        f"資料層實測 {[GVEC_FEATURE_NAMES[i] for i in deg]}"
    )
    assert GVEC_EFFECTIVE_DIM_C_SERIES == GVEC_DIM - len(deg)
    for name, expected in GVEC_INERT_DIMS_C_SERIES.items():
        col = merged[:, GVEC_FEATURE_NAMES.index(name)]
        assert float(col[0]) == pytest.approx(expected), f"{name} 常數值與宣告不符"
