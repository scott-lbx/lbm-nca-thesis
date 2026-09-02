"""pytest 共用設定：只做 marker 註冊（DEEP_AUDIT R3-G5）。

背景：``tests/tes/test_tristate_equivalence.py`` 用 ``pytest.mark.gpu`` 標記需要 GPU +
canonical ckpt 的測試，但本 repo 沒有 ``pytest.ini`` / ``pyproject.toml`` / ``setup.cfg``，
marker 未註冊 → 執行時出 ``PytestUnknownMarkWarning``，加 ``--strict-markers`` 則直接
error。

刻意放在 ``tests/conftest.py`` 而非新增 repo 根的 ``pytest.ini``：後者會改變 pytest 的
rootdir 判定與設定檔優先序，影響範圍超出測試蒐集；conftest 只在蒐集 ``tests/`` 時載入，
除了註冊 marker 之外不改變任何行為。

★本檔**只**註冊 marker，不做 skip、不改 fixture、不設 addopts——是否要跑 gpu 測試由
呼叫端自行以 ``-m gpu`` / ``-m "not gpu"`` 決定。
"""
from __future__ import annotations


def pytest_configure(config) -> None:
    """註冊本 repo 自訂 marker。

    Args:
        config: pytest 的 ``Config`` 物件。
    """
    config.addinivalue_line(
        "markers",
        "gpu: 需要 CUDA（且通常需 canonical ckpt / dataset）的測試；"
        "訓練進行中勿跑（單一訓練規則 + 高 VRAM 訓練禁 GPU pytest）。",
    )
    config.addinivalue_line(
        "markers",
        "dataset: 需要本機重新生成的 Zarr 資料集；見 docs/DATA_GENERATION.md。",
    )
