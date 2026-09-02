"""存 ckpt 護欄測試：identity 必須落盤，缺則 raise（CPU-only，寫 tmp_path）。

覆蓋 GATE 0 複驗 [Stage R 護欄] 的三條路徑：
  1. 正常：存檔後 re-load 讀得到完整 identity。
  2. 失敗：``build_identity`` 失敗（config 缺欄位）→ ``CheckpointIdentityError``
     且**不留下**無身分 ckpt；``torch.save`` 之後 identity 不在檔案裡也要 raise。
  3. 降級：顯式參數或環境變數開啟時，才允許存無身分 ckpt。

同時鎖住 DRY：``Checkpointer.save`` 與 ``TESTrainer._save_checkpoint`` 都必須委派至
``save_checkpoint``，不得各自再組 payload / 呼叫 ``torch.save``。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from lbmnca.nca.tes.config import TESTrainConfig
from lbmnca.nca.tes.training import Checkpointer, TESTrainer
from lbmnca.nca.tes.training.checkpoint import checkpointer as ckpt_mod
from lbmnca.nca.tes.training.checkpoint import identity as ident_mod
from lbmnca.nca.tes.training.checkpoint.checkpointer import (
    ALLOW_UNIDENTIFIED_CKPT_ENV,
    CheckpointIdentityError,
    checkpoint_filename,
    save_checkpoint,
)
from lbmnca.nca.tes.training.checkpoint.identity import IDENTITY_KEY, extract_identity
from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer


def _field_normalizer() -> FieldNormalizer:
    """不依賴 stats 檔的 FieldNormalizer（tanh 模式）。"""
    nine = np.arange(9, dtype=np.float32)
    return FieldNormalizer(
        mf_mean=nine,
        mf_std=nine + 1.0,
        up_mean=np.zeros(9, dtype=np.float32),
        up_std=np.ones(9, dtype=np.float32),
        sdf_mean=0.0,
        sdf_std=1.0,
        px_mean=0.0,
        px_std=1.0,
        py_mean=0.0,
        py_std=1.0,
        cond_mean=np.zeros(1, dtype=np.float32),
        cond_std=np.ones(1, dtype=np.float32),
        is_tanh=True,
        upstream_mean=0.0,
        upstream_std=1.0,
    )


def _rig() -> dict:
    """最小 model/optimizer/scheduler/scaler 組合（不碰 GPU）。"""
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return {
        "model": model,
        "optimizer": opt,
        "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0),
        "scaler": torch.amp.GradScaler("cuda", enabled=False),
        "field_normalizer": _field_normalizer(),
        "cond_normalizer": None,
    }


@pytest.fixture()
def cfg() -> TESTrainConfig:
    """預設 config（psi_lowpass_sigma 設為可辨識值）。"""
    c = TESTrainConfig()
    c.model.psi_lowpass_sigma = 0.8
    c.seed = 42
    return c


def test_save_writes_identity_and_reload_sees_it(tmp_path, cfg) -> None:
    """正常路徑：ckpt 存出後 re-load 讀得到 identity 與五個原有 key。"""
    records: list[dict] = []
    path = save_checkpoint(
        tmp_path,
        step=500,
        tag=None,
        cfg=cfg,
        write_jsonl=lambda r, force_flush=False: records.append(r),
        **_rig(),
    )
    assert path.name == checkpoint_filename(500, None)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert all(k in loaded for k in ("step", "model", "optimizer", "scheduler", "scaler"))
    ident = extract_identity(loaded)
    assert ident is not None
    assert ident["inference_hparams"]["model.psi_lowpass_sigma"] == 0.8
    assert ident["denorm"]["field_normalizer"] is not None
    assert any(r["event"] == "checkpoint" and r["has_identity"] for r in records)


def test_missing_config_field_raises_and_leaves_no_ckpt(tmp_path, cfg, monkeypatch) -> None:
    """★核心護欄：config 缺欄位（欄位改名）→ raise，而非靜默存出無身分 ckpt。"""
    monkeypatch.setattr(
        ident_mod,
        "INFERENCE_SENSITIVE_TRAIN_FIELDS",
        ident_mod.INFERENCE_SENSITIVE_TRAIN_FIELDS + ("apply_ic_mask_RENAMED",),
    )
    records: list[dict] = []
    with pytest.raises(CheckpointIdentityError) as excinfo:
        save_checkpoint(
            tmp_path,
            step=600,
            tag=None,
            cfg=cfg,
            write_jsonl=lambda r, force_flush=False: records.append(r),
            **_rig(),
        )
    assert isinstance(excinfo.value.__cause__, AttributeError)
    assert not (tmp_path / checkpoint_filename(600, None)).exists()
    # 存檔端也必須留下 metrics.json 記錄（舊版只有一行 log warning）
    assert any(
        r["event"] == "ckpt_identity_error" and r["phase"] == "build" for r in records
    )


def test_identity_lost_during_serialization_raises(tmp_path, cfg, monkeypatch) -> None:
    """torch.save 之後 identity 不在檔案裡 → 落盤後 re-load 斷言必須攔下。"""
    orig_save = torch.save

    def _save_dropping_identity(obj, f, *args, **kwargs):
        if isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if k != IDENTITY_KEY}
        return orig_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(ckpt_mod.torch, "save", _save_dropping_identity)
    records: list[dict] = []
    with pytest.raises(CheckpointIdentityError, match="無身分資訊"):
        save_checkpoint(
            tmp_path,
            step=700,
            tag=None,
            cfg=cfg,
            write_jsonl=lambda r, force_flush=False: records.append(r),
            **_rig(),
        )
    assert any(r["phase"] == "verify_on_disk" for r in records)


@pytest.mark.parametrize("via_env", [False, True])
def test_explicit_downgrade_allows_unidentified_ckpt(
    tmp_path, cfg, monkeypatch, via_env: bool
) -> None:
    """降級開關（參數或環境變數）顯式開啟時，才允許存無身分 ckpt。"""
    monkeypatch.setattr(
        ident_mod,
        "INFERENCE_SENSITIVE_TRAIN_FIELDS",
        ident_mod.INFERENCE_SENSITIVE_TRAIN_FIELDS + ("apply_ic_mask_RENAMED",),
    )
    kwargs = {}
    if via_env:
        monkeypatch.setenv(ALLOW_UNIDENTIFIED_CKPT_ENV, "1")
    else:
        monkeypatch.delenv(ALLOW_UNIDENTIFIED_CKPT_ENV, raising=False)
        kwargs["allow_unidentified"] = True
    path = save_checkpoint(
        tmp_path, step=800, tag="degraded", cfg=cfg, **kwargs, **_rig()
    )
    assert path.exists()
    assert extract_identity(torch.load(path, map_location="cpu", weights_only=False)) is None


def test_default_is_raise_not_warning(tmp_path, cfg, monkeypatch) -> None:
    """未設環境變數時預設必為 raise（降級不得是預設行為）。"""
    monkeypatch.delenv(ALLOW_UNIDENTIFIED_CKPT_ENV, raising=False)
    monkeypatch.setattr(
        ident_mod,
        "INFERENCE_SENSITIVE_TRAIN_FIELDS",
        ident_mod.INFERENCE_SENSITIVE_TRAIN_FIELDS + ("apply_ic_mask_RENAMED",),
    )
    with pytest.raises(CheckpointIdentityError):
        save_checkpoint(tmp_path, step=801, tag=None, cfg=cfg, **_rig())


def test_checkpointer_save_delegates(tmp_path, cfg) -> None:
    """Checkpointer.save 走同一份實作，產出的 ckpt 同樣帶身分。"""
    rig = _rig()
    ctx = SimpleNamespace(
        output_dir=tmp_path,
        step=1234,
        model=rig["model"],
        optimizer=rig["optimizer"],
        scheduler=rig["scheduler"],
        scaler=rig["scaler"],
        cfg=cfg,
        field_normalizer=rig["field_normalizer"],
        normalizer=None,
        metrics_file=None,
    )
    cper = Checkpointer(ctx)
    records: list[dict] = []
    cper.write_jsonl = lambda r, force_flush=False: records.append(r)  # type: ignore[method-assign]
    path = cper.save()
    assert path.name == checkpoint_filename(1234, None)
    assert extract_identity(torch.load(path, map_location="cpu", weights_only=False)) is not None
    assert any(r["event"] == "checkpoint" for r in records)


@pytest.mark.parametrize(
    "func", [Checkpointer.save, TESTrainer._save_checkpoint], ids=["checkpointer", "trainer"]
)
def test_no_duplicated_save_implementation(func) -> None:
    """★DRY：兩個存檔入口都只委派，不得再各自組 payload / 建 identity / torch.save。"""
    src = inspect.getsource(func)
    assert "save_checkpoint(" in src
    assert "torch.save" not in src
    assert "build_identity" not in src
