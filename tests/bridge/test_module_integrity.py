"""執行鏈完整性：每個檔案能編譯、能 import，且沒有指向已移除 API 的殘留呼叫。

只做 static / import 層級檢查，不建模型、不碰 GPU。
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# lbmnca.bridge 執行鏈可達的檔案（GPU/模型模組只編譯不 import）。
# main.py 是套件內 entrypoint；demo/ 下三支是獨立輔助腳本（非套件模組，只做
# compile 層級檢查）——analyze_component_standalone.py 依設計刻意不 import
# lbmnca（見檔頭註解：★自足，貼進 Grasshopper Python 元件單獨執行）。
CHAIN = [
    "src/lbmnca/bridge/main.py",
    "src/lbmnca/bridge/core/command_queue.py",
    "src/lbmnca/bridge/core/geometry_ingest.py",
    "src/lbmnca/bridge/core/server.py",
    "src/lbmnca/bridge/core/session.py",
    "src/lbmnca/bridge/core/static_builder.py",
    "src/lbmnca/bridge/geometry/angle_transform.py",
    "src/lbmnca/bridge/geometry/gvec_bridge.py",
    "src/lbmnca/bridge/geometry/jfa_sdf.py",
    "src/lbmnca/bridge/geometry/mask_rasterizer.py",
    "src/lbmnca/bridge/geometry/regime_features.py",
    "src/lbmnca/bridge/ggui/input_handler.py",
    "src/lbmnca/bridge/ggui/playback.py",
    "src/lbmnca/bridge/ggui/profiling.py",
    "src/lbmnca/bridge/ggui/ui_draw.py",
    "src/lbmnca/bridge/ggui/window.py",
    "src/lbmnca/bridge/integration/model_backend.py",
    "src/lbmnca/bridge/integration/adapters/__init__.py",
    "src/lbmnca/bridge/integration/adapters/decompose_adapter.py",
    "src/lbmnca/bridge/integration/adapters/dummy_adapter.py",
    "src/lbmnca/bridge/integration/adapters/tes_adapter.py",
    "src/lbmnca/bridge/render/colorize.py",
    "src/lbmnca/bridge/render/convergence_overlay.py",
    "src/lbmnca/bridge/render/field_ops.py",
    "src/lbmnca/bridge/render/gpu_fields.py",
    "src/lbmnca/bridge/render/ibfv_params.py",
    "src/lbmnca/bridge/render/ibfv_pass.py",
    "src/lbmnca/bridge/render/overlay.py",
    "src/lbmnca/bridge/render/playback_director.py",
    "src/lbmnca/bridge/render/streamlines.py",
    "src/lbmnca/bridge/simulation/convergence_guard.py",
    "src/lbmnca/bridge/simulation/trajectory.py",
    "demo/launcher.py",
    "demo/send_sequence.py",
    "demo/gh/analyze_component_standalone.py",
]

# 不需 GPU、`pip install -e .` 後即可 import 的套件模組
# （lbmnca.bridge.ggui / render / integration.adapters 需要 Taichi context
#  或 02 repo 的 checkpoint，不在此列；main.py 啟動時才 ti.init(arch=ti.cuda)）。
IMPORTABLE = [
    "lbmnca.bridge.core.command_queue",
    "lbmnca.bridge.core.geometry_ingest",
    "lbmnca.bridge.core.server",
    "lbmnca.bridge.core.session",
    "lbmnca.bridge.core.static_builder",
    "lbmnca.bridge.geometry.angle_transform",
    "lbmnca.bridge.geometry.regime_features",
    "lbmnca.bridge.integration.model_backend",
]


@pytest.mark.parametrize("rel", CHAIN)
def test_chain_file_exists_and_compiles(repo: Path, rel: str):
    p = repo / rel
    assert p.is_file(), f"執行鏈檔案不存在：{rel}"
    compile(p.read_text(encoding="utf-8"), str(p), "exec")


@pytest.mark.parametrize("mod", IMPORTABLE)
def test_module_imports(mod: str):
    importlib.import_module(mod)


def _names_used(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def test_no_double_rotation_left_in_window(repo: Path):
    """★座標契約：polygons 由 ingest 一次轉成 canonical，window 不得再轉。"""
    used = _names_used(repo / "src" / "lbmnca" / "bridge" / "ggui" / "window.py")
    assert "transform_polygons" not in used
    assert "AngleTransform" not in used


def test_ingest_does_the_rotation(repo: Path):
    """反向確認：旋轉確實發生在 geometry_ingest。"""
    used = _names_used(
        repo / "src" / "lbmnca" / "bridge" / "core" / "geometry_ingest.py"
    )
    assert "AngleTransform" in used
    assert "to_canonical" in used


def test_no_references_to_removed_velocity_rotation(repo: Path):
    """已移除的速度場逆旋轉 API 不得在任何執行鏈檔案殘留。"""
    for rel in CHAIN:
        src = (repo / rel).read_text(encoding="utf-8")
        assert "rotate_velocity_field" not in src, rel
        assert "get_inverse_params" not in src, rel


def test_analyze_zarr_uses_v4_layout(repo: Path):
    """v3 的 dataset_zarr/ 已刪；window.py 必須走 dataset_zarr_v4。"""
    src = (
        repo / "src" / "lbmnca" / "bridge" / "ggui" / "window.py"
    ).read_text(encoding="utf-8")
    assert "dataset_zarr_v4" in src
    assert '"dataset_zarr"' not in src and "'dataset_zarr'" not in src


def test_no_getattr_config_fallback_in_adapters(repo: Path):
    """專案鐵則：禁 ``getattr(cfg, "key", default)`` 式的 config 存取。"""
    for rel in (
        "src/lbmnca/bridge/integration/adapters/tes_adapter.py",
        "src/lbmnca/bridge/integration/adapters/decompose_adapter.py",
    ):
        tree = ast.parse((repo / rel).read_text(encoding="utf-8"), rel)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3
            ):
                pytest.fail(f"{rel}:{node.lineno} 出現帶 default 的 getattr")
