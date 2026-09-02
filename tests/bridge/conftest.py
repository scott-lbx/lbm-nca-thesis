"""pytest 共用設定：讓 `src.*` 可 import，並固定 Taichi 走 CPU。

本目錄下的測試**全部不碰 GPU**——訓練常常同時在跑，搶 VRAM 會把它 OOM 掉。
需要 Taichi 的測試一律 `ti.init(arch=ti.cpu)`（見 `ti_cpu` fixture）。
模型推論（PyTorch CUDA）不在此測，那屬於 GPU 空檔才跑的 smoke。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WORKSPACE = REPO.parent / "lbm-nca-thesis"


@pytest.fixture(scope="session")
def repo() -> Path:
    """lbm-nca-thesis repo 根目錄。"""
    return REPO


@pytest.fixture(scope="session")
def workspace() -> Path:
    """lbm-nca-thesis 根目錄。"""
    return WORKSPACE


@pytest.fixture(scope="session")
def ti_cpu():
    """在 CPU 上初始化 Taichi 一次，供需要 kernel/field 的測試共用。

    Taichi 每個 process 只該 init 一次，故 scope=session。
    """
    import taichi as ti

    ti.init(arch=ti.cpu, log_level=ti.WARN)
    return ti
