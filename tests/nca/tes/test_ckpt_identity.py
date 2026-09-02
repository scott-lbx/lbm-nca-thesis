"""checkpoint 身分（identity）欄位測試（CPU-only，不碰 GPU / zarr / runs）。

覆蓋 DEEP_AUDIT_2026-07 Stage 0.2 的四項要求：
  1. 存 ckpt 時寫入身分欄位（git commit / config sha / seed / 推論敏感超參數 /
     反正規化常數）。
  2. 載入端能讀出並在不一致時報警（psi_lowpass 0.8/0.9 事故類型）。
  3. 向後相容：舊 ckpt 無 identity 不崩潰。
  4. 「沒有記錄」與「記錄為某值」可區分。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from lbmnca.nca.tes.config import TESTrainConfig
from lbmnca.nca.tes.training.checkpoint.identity import (
    IDENTITY_KEY,
    IDENTITY_SCHEMA_VERSION,
    NOT_RECORDED,
    build_identity,
    collect_inference_hparams,
    extract_identity,
    field_normalizer_snapshot,
    format_identity,
    git_info,
    sha256_of,
    sha256_of_file,
    verify_identity,
)
from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer


def _dummy_field_normalizer(scale: float = 1.0) -> FieldNormalizer:
    """建一個不依賴 stats 檔的 FieldNormalizer（tanh 模式），供反正規化比對測試。

    Args:
        scale: 乘在 mf_std 上的倍率，用來製造「不同正規化空間」。

    Returns:
        FieldNormalizer 實例。
    """
    nine = np.arange(9, dtype=np.float32)
    return FieldNormalizer(
        mf_mean=nine,
        mf_std=(nine + 1.0) * scale,
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


@pytest.fixture()
def cfg() -> TESTrainConfig:
    """預設 config，psi_lowpass_sigma 設為訓練值 0.8。"""
    c = TESTrainConfig()
    c.model.psi_lowpass_sigma = 0.8
    c.seed = 42
    return c


def test_identity_records_required_fields(cfg: TESTrainConfig) -> None:
    """身分區塊必含 git_commit / config sha / seed / 推論敏感超參數 / denorm。"""
    ident = build_identity(
        cfg,
        field_normalizer=_dummy_field_normalizer(),
        cond_normalizer=None,
        model=None,
        config_source="configs/dummy.yaml",
    )
    assert ident["schema_version"] == IDENTITY_SCHEMA_VERSION
    # git（不可用時為 None，但 key 必存在——「查不到」不等於「沒這欄」）
    assert "commit" in ident["git"]
    assert ident["seed"]["seed"] == 42
    assert len(ident["config"]["sha256"]) == 64
    assert len(ident["config"]["model_sha256"]) == 64
    hp = ident["inference_hparams"]
    for key in (
        "apply_ic_mask",
        "model.psi_lowpass_sigma",
        "model.transport.ke_anchor_mode",
        "stats_policy",
        "zarr_paths",
    ):
        assert key in hp, key
    assert hp["model.psi_lowpass_sigma"] == 0.8
    # 反正規化常數必須「內嵌」而非只存路徑
    assert ident["denorm"]["field_normalizer"] is not None
    assert "_mf_std" in ident["denorm"]["field_normalizer"]["buffers"]
    assert len(ident["denorm"]["sha256"]) == 64


def test_identity_is_json_primitive_only(cfg: TESTrainConfig) -> None:
    """身分區塊只含 JSON 原生型別（torch.load(weights_only=True) 相容）。"""
    ident = build_identity(
        cfg, field_normalizer=_dummy_field_normalizer(), cond_normalizer=None
    )

    def _check(o: object) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                assert isinstance(k, str)
                _check(v)
        elif isinstance(o, list):
            for v in o:
                _check(v)
        else:
            assert o is None or isinstance(o, (str, int, float, bool)), type(o)

    _check(ident)


def test_verify_detects_psi_lowpass_mismatch(cfg: TESTrainConfig) -> None:
    """★重演事故：ckpt 訓練於 0.8，載入端 config 為 0.9 → 必須偵測到。"""
    ckpt = {
        "step": 8000,
        "identity": build_identity(
            cfg, field_normalizer=_dummy_field_normalizer(), cond_normalizer=None
        ),
    }
    wrong = dataclasses.replace(cfg)
    wrong.model = dataclasses.replace(cfg.model, psi_lowpass_sigma=0.9)

    report = verify_identity(ckpt, wrong, on_mismatch="warn")
    keys = [m["key"] for m in report["mismatches"]]
    assert "model.psi_lowpass_sigma" in keys
    hit = next(m for m in report["mismatches"] if m["key"] == "model.psi_lowpass_sigma")
    assert hit["trained_with"] == 0.8
    assert hit["loading_with"] == 0.9

    with pytest.raises(ValueError, match="psi_lowpass_sigma"):
        verify_identity(ckpt, wrong, on_mismatch="raise")


def test_verify_passes_on_matching_config(cfg: TESTrainConfig) -> None:
    """同一份 config → 無 mismatch，denorm 相符。"""
    fn = _dummy_field_normalizer()
    ckpt = {"identity": build_identity(cfg, field_normalizer=fn, cond_normalizer=None)}
    report = verify_identity(ckpt, cfg, field_normalizer=fn, on_mismatch="raise")
    assert report["mismatches"] == []
    assert report["denorm"]["status"] == "match"
    assert report["n_checked"] > 40


def test_verify_detects_denorm_space_change(cfg: TESTrainConfig) -> None:
    """反正規化常數改變（等同抽掉一個 zarr 來源）→ denorm 判 mismatch。"""
    fn_train = _dummy_field_normalizer(scale=1.0)
    fn_other = _dummy_field_normalizer(scale=1.32)  # 全域尺度改變 32%
    ckpt = {
        "identity": build_identity(
            cfg, field_normalizer=fn_train, cond_normalizer=None
        )
    }
    report = verify_identity(ckpt, cfg, field_normalizer=fn_other, on_mismatch="warn")
    assert report["denorm"]["status"] == "mismatch"
    assert report["denorm"]["diff"]


def test_legacy_ckpt_without_identity_does_not_crash(cfg: TESTrainConfig) -> None:
    """向後相容：舊 ckpt（5 key）載入不崩潰，且明確標示『無身分資訊』。"""
    legacy = {
        "step": 8000,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
    }
    assert extract_identity(legacy) is None  # None = 沒有記錄，非預設值
    report = verify_identity(legacy, cfg, on_mismatch="raise")  # raise 模式也不得炸
    assert report["has_identity"] is False
    assert report["mismatches"] == []
    assert "無身分資訊" in report["summary"]
    assert "無身分資訊" in format_identity(None)


def test_not_recorded_is_distinct_from_recorded_value(cfg: TESTrainConfig) -> None:
    """區分『沒有記錄』與『記錄為某值』：缺欄位不得被當成值相同或值不同。"""
    ident = build_identity(
        cfg, field_normalizer=_dummy_field_normalizer(), cond_normalizer=None
    )
    ident["inference_hparams"].pop("model.psi_lowpass_sigma")
    ckpt = {"identity": ident}

    wrong = dataclasses.replace(cfg)
    wrong.model = dataclasses.replace(cfg.model, psi_lowpass_sigma=0.9)

    report = verify_identity(ckpt, wrong, on_mismatch="warn")
    assert "model.psi_lowpass_sigma" in report["not_recorded"]
    assert all(m["key"] != "model.psi_lowpass_sigma" for m in report["mismatches"])
    # 完整 model config 快照仍是第二道防線
    assert any(d["key"] == "psi_lowpass_sigma" for d in report["model_config_diff"])


def test_collect_inference_hparams_has_no_fallback() -> None:
    """禁 config fallback：缺欄位必須 AttributeError，不得靜默代入預設值。"""

    class _Broken:
        model = object()

    with pytest.raises(AttributeError):
        collect_inference_hparams(_Broken())


def test_save_load_roundtrip_via_torch(tmp_path, cfg: TESTrainConfig) -> None:
    """torch.save/load 往返後身分欄位完整保留，且檔案 sha256 可計算。"""
    ident = build_identity(
        cfg, field_normalizer=_dummy_field_normalizer(), cond_normalizer=None
    )
    path = tmp_path / "ckpt_test.pth"
    torch.save({"step": 1, "model": {}, IDENTITY_KEY: ident}, path)

    loaded = torch.load(str(path), map_location="cpu", weights_only=False)
    got = extract_identity(loaded)
    assert got is not None
    assert got["config"]["model_sha256"] == ident["config"]["model_sha256"]
    assert got["inference_hparams"]["model.psi_lowpass_sigma"] == 0.8
    assert len(sha256_of_file(path)) == 64


def test_field_normalizer_snapshot_is_deterministic() -> None:
    """相同 normalizer → 相同快照 hash（跨 process 可比對）。"""
    a = field_normalizer_snapshot(_dummy_field_normalizer())
    b = field_normalizer_snapshot(_dummy_field_normalizer())
    assert sha256_of(a) == sha256_of(b)


def test_git_info_shape() -> None:
    """git_info 必回傳固定 schema；查不到時 commit 為 None（不填假值）。"""
    info = git_info()
    for key in ("available", "commit", "branch", "dirty", "workspace_commit"):
        assert key in info
    if not info["available"]:
        assert info["commit"] is None


def test_not_recorded_sentinel_is_falsy_singleton() -> None:
    """NOT_RECORDED 為單例且 falsy，避免與 0/False/'' 等真實值混淆。"""
    assert NOT_RECORDED is type(NOT_RECORDED)()
    assert not NOT_RECORDED
    assert repr(NOT_RECORDED) == "<NOT_RECORDED>"
