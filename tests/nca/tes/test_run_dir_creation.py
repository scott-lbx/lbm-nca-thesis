"""run 目錄建立護欄測試（CPU-only，全部寫 tmp_path）。

覆蓋 DEEP_AUDIT GATE 0 新發現 N1：純推論的製圖 / 評估 harness 透過
``build_mean_training`` → ``build_tes_trainer`` → ``TESTrainer.__init__`` 建 trainer，
而 ``__init__`` 無條件 ``output_dir.mkdir()`` + 開 ``training.log`` / ``metrics.json``，
於是每跑一次製圖就在 ``runs/`` 留下一個沒有 ckpt 的空 run 目錄，讓 archive
「剩餘 run 100% 有 ckpt_final」的不變量自動失效。

鎖住兩件事：
  1. ``create_run_dir=True``（訓練預設）仍建 run 目錄並開 metrics.json。
  2. ``create_run_dir=False``（推論）完全不碰檔案系統，連父目錄都不生成。
另外鎖住三個入口（``TESTrainer`` / ``build_tes_trainer`` / ``build_mean_training``）
的 ``create_run_dir`` 參數存在且預設為 True，避免日後被移除或預設翻轉。

★不碰 GPU：device 固定 cpu、use_amp=False，且以 mock 取代需要 zarr/stats 的
建構步驟（scheduler / SDF 門檻 / step 子元件），本檔測的是 ``__init__`` 的
run 目錄副作用，與訓練數值無關。
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from lbmnca.nca.tes.config import TESTrainConfig
from lbmnca.nca.tes.train import build_tes_trainer
from lbmnca.nca.tes.train_mean import build_mean_training
from lbmnca.nca.tes.training import TESTrainer

RUN_NAME: str = "unit_run_dir_guard"


def _make_cfg(run_dir: Path) -> TESTrainConfig:
    """最小可用 config：output_dir 指向 tmp_path，關閉 AMP 與非同步產圖。

    Args:
        run_dir: 預期的 run 目錄路徑（尚未存在）。

    Returns:
        已設定 output_dir 的 ``TESTrainConfig``。
    """
    cfg = TESTrainConfig()
    cfg.output_dir = str(run_dir)
    cfg.use_amp = False
    cfg.async_eval_generation = False
    return cfg


def _build_trainer(run_dir: Path, create_run_dir: bool) -> TESTrainer:
    """建一個只跑得完 ``__init__`` 的 ``TESTrainer``（不碰 zarr / GPU）。

    Args:
        run_dir: ``cfg.output_dir``。
        create_run_dir: 傳給 ``TESTrainer`` 的同名參數。

    Returns:
        建構完成的 ``TESTrainer``。
    """
    model = nn.Linear(4, 4)
    with patch.object(TESTrainer, "_build_scheduler", return_value=None), patch.object(
        TESTrainer, "_load_sdf_threshold", return_value=0.0
    ), patch.object(TESTrainer, "_build_components", return_value=None):
        return TESTrainer(
            cfg=_make_cfg(run_dir),
            model=model,
            loss_fn=nn.Identity(),
            normalizer=None,
            train_loader=None,
            device=torch.device("cpu"),
            create_run_dir=create_run_dir,
        )


def _release_log_handles(trainer: TESTrainer, run_dir: Path) -> None:
    """關閉 trainer 開在 run_dir 內的檔案 handle（Windows 上不關會擋 tmp 清理）。

    Args:
        trainer: 待清理的 trainer。
        run_dir: run 目錄路徑，只移除指向此目錄的 root logger handler。
    """
    if trainer._metrics_fp is not None:
        trainer._metrics_fp.close()
        trainer._metrics_fp = None
    root = logging.getLogger()
    for handler in list(root.handlers):
        base = getattr(handler, "baseFilename", None)
        if base is not None and Path(base).parent == run_dir:
            handler.close()
            root.removeHandler(handler)


def test_training_mode_creates_run_dir(tmp_path) -> None:
    """訓練預設路徑（create_run_dir=True）行為不變：建目錄並開 metrics.json。"""
    run_dir = tmp_path / "runs" / RUN_NAME
    trainer = _build_trainer(run_dir, create_run_dir=True)
    try:
        assert run_dir.is_dir()
        assert (run_dir / "metrics.json").exists()
        assert trainer._metrics_fp is not None
    finally:
        _release_log_handles(trainer, run_dir)


def test_inference_mode_creates_no_run_dir(tmp_path) -> None:
    """★N1 護欄：推論路徑（create_run_dir=False）不得產生任何目錄或檔案。"""
    runs_root = tmp_path / "runs"
    run_dir = runs_root / RUN_NAME
    trainer = _build_trainer(run_dir, create_run_dir=False)
    try:
        assert not run_dir.exists()
        assert not runs_root.exists()  # 連父層都不該被 mkdir(parents=True) 建出來
        assert trainer._metrics_fp is None
        assert trainer._output_dir == run_dir  # 路徑仍保留供參考
    finally:
        _release_log_handles(trainer, run_dir)


@pytest.mark.parametrize(
    "func",
    [TESTrainer.__init__, build_tes_trainer, build_mean_training],
    ids=["TESTrainer.__init__", "build_tes_trainer", "build_mean_training"],
)
def test_create_run_dir_param_exists_and_defaults_true(func) -> None:
    """三個入口都必須有 create_run_dir 參數且預設 True（訓練行為不變）。"""
    param = inspect.signature(func).parameters.get("create_run_dir")
    assert param is not None, f"{func.__qualname__} 缺少 create_run_dir 參數"
    assert param.default is True, f"{func.__qualname__} 的 create_run_dir 預設必須為 True"
