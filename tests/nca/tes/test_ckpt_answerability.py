"""ckpt 可追溯性驗收測試（DEEP_AUDIT_2026-07 v2 ``L0.0-b``）。

驗收斷言：**任一 ckpt 可單獨回答「哪個 config／哪份資料／哪個 commit」**。

本檔與 ``test_ckpt_save_guard.py`` 的分工
------------------------------------------------------------------
``test_ckpt_save_guard.py`` 測的是「identity 有沒有走到硬碟上」（存在性）；
本檔測的是「走上去的那份**答不答得出三個問題**」（充分性）。舊版落盤斷言只查
``inference_hparams`` 非空 / ``config.sha256`` 存在 / ``denorm`` 是 dict，
三個驗收問題**沒有一個**被真正檢查——且三個缺口全部朝「看起來通過」失效。

另含兩項防回歸：
  * ``torch.save`` 繞過 checkpointer 的 callsite 清單（以 ``ast`` 掃，非 regex）。
  * 覆寫既有 ckpt 檔必須留下記錄（``torch.save`` 本身是無聲覆蓋）。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from lbmnca.nca.tes.config import TESTrainConfig
from lbmnca.nca.tes.training.checkpoint import checkpointer as ckpt_mod
from lbmnca.nca.tes.training.checkpoint import identity as ident_mod
from lbmnca.nca.tes.training.checkpoint.checkpointer import (
    CheckpointIdentityError,
    _identity_answerability_problems,
    save_checkpoint,
)
from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer

#: lbmnca.nca repo root。
REPO_ROOT = Path(__file__).resolve().parents[3]

#: ★允許存在裸 ``torch.save`` 的**活躍**檔案（repo 相對 posix 路徑）。
#:
#: 這三支是 v1 ``R2`` 點名、實測仍在的訓練腳本：它們自行組 payload 存 ckpt，
#: 完全不經 :func:`save_checkpoint`，產出的 ckpt 永遠無身分。列在此處＝**已知債**，
#: ⛔ 不是核可 —— 清單只准變短。新增任何一支都會讓本測試失敗。
KNOWN_BARE_SAVE_FILES: frozenset[str] = frozenset(
    {
        "scripts/train_seed_fluct.py",
        "scripts/turb_redesign_train.py",
        "scripts/turb_redesign_pool_train.py",
        # POD 基底產物（非訓練 ckpt，但同樣是無身分 .pt，且被 identity 以路徑引用）
        "scripts/fit_turb_pod_basis.py",
        # canonical 存檔點本身
        "src/lbmnca/nca/tes/training/checkpoint/checkpointer.py",
    }
)

#: 掃描範圍（repo 相對）。
LIVE_SCAN_DIRS: tuple[str, ...] = ("src/lbmnca/nca/tes", "scripts")


# ---------------------------------------------------------------------------
# 共用治具
# ---------------------------------------------------------------------------

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


def _rig(with_normalizer: bool = True) -> dict:
    """最小 model/optimizer/scheduler/scaler 組合（不碰 GPU）。"""
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return {
        "model": model,
        "optimizer": opt,
        "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0),
        "scaler": torch.amp.GradScaler("cuda", enabled=False),
        "field_normalizer": _field_normalizer() if with_normalizer else None,
        "cond_normalizer": None,
    }


@pytest.fixture()
def cfg() -> TESTrainConfig:
    """預設 config（帶可辨識的 dataset 來源，模擬真實訓練）。"""
    c = TESTrainConfig()
    c.model.psi_lowpass_sigma = 0.8
    c.seed = 42
    c.dataset_names = ["C2-1_loop_5_2"]
    return c


# ---------------------------------------------------------------------------
# 1. 端到端：存出來的 ckpt 真的答得出三個問題
# ---------------------------------------------------------------------------

def test_saved_ckpt_answers_config_data_commit(tmp_path, cfg) -> None:
    """★驗收正控：ckpt **單獨**打開即可回答三個問題（不靠 run 目錄旁的檔案）。"""
    path = save_checkpoint(tmp_path, step=100, tag="final", cfg=cfg, **_rig())
    ident = torch.load(path, map_location="cpu", weights_only=False)["identity"]

    # 哪個 commit
    assert ident["git"]["commit"], "ckpt 內無 commit"
    # 哪個 config
    assert ident["config"]["sha256"]
    assert ident["config"]["model"]["psi_lowpass_sigma"] == 0.8
    # 哪份資料
    assert ident["denorm"]["field_normalizer"] is not None
    assert ident["denorm"]["stats_ref"]["dataset_names"] == ["C2-1_loop_5_2"]
    # 三問皆可答 ⇒ 規則函式回空
    assert _identity_answerability_problems(ident) == []


# ---------------------------------------------------------------------------
# 2. 「哪個 commit」：git 不可用 → 舊版靜默通過，新版必須擋
# ---------------------------------------------------------------------------

def test_missing_git_commit_is_rejected(tmp_path, cfg, monkeypatch) -> None:
    """★git_info 查不到 commit（回 None）時，ckpt 不得被當成合格身分存出。

    ``git_info`` 在 git 不可用時**刻意**回 ``commit=None``（不填假值）。
    舊落盤斷言完全沒看 git 區塊 → ckpt 照樣存出且「驗證通過」，這顆權重此後
    永遠對不回程式碼版本。失效方向＝看起來通過。
    """
    monkeypatch.setattr(
        ident_mod,
        "git_info",
        lambda refresh=False: {
            "available": False,
            "repo": "lbmnca.nca",
            "commit": None,
            "branch": None,
            "dirty": None,
            "dirty_files": [],
            "workspace_commit": None,
        },
    )
    records: list[dict] = []
    with pytest.raises(CheckpointIdentityError, match="git.commit"):
        save_checkpoint(
            tmp_path,
            step=101,
            tag=None,
            cfg=cfg,
            write_jsonl=lambda r, force_flush=False: records.append(r),
            **_rig(),
        )
    assert any(r.get("phase") == "verify_on_disk" for r in records)


# ---------------------------------------------------------------------------
# 3. 「哪份資料」：來源全空且未內嵌常數 → 必須擋
# ---------------------------------------------------------------------------

def test_unidentifiable_data_source_is_rejected(tmp_path) -> None:
    """★denorm 既無來源清單、也未內嵌反正規化常數 → 無法回答「哪份資料」。

    舊斷言只有 ``isinstance(denorm, dict)``；這種 denorm 仍是 dict，因此通過。
    """
    c = TESTrainConfig()  # zarr_path/zarr_paths/stats_path/dataset_names 皆為空
    c.seed = 42
    with pytest.raises(CheckpointIdentityError, match="哪份資料"):
        save_checkpoint(
            tmp_path, step=102, tag=None, cfg=c, **_rig(with_normalizer=False)
        )


def test_data_source_ok_when_only_stats_ref_present() -> None:
    """只要 stats_ref 指得出來源，未內嵌常數仍算答得出「哪份資料」（不過度嚴格）。"""
    ident = _minimal_identity()
    ident["denorm"]["field_normalizer"] = None
    ident["denorm"]["field_normalizer_absent_reason"] = "stats_policy=single；未建立"
    ident["denorm"]["stats_ref"]["zarr_paths"] = ["D:/x.zarr"]
    assert _identity_answerability_problems(ident) == []


def test_absent_normalizer_without_reason_is_rejected() -> None:
    """未內嵌反正規化常數卻**沒說原因** → 靜默省略，必須擋。"""
    ident = _minimal_identity()
    ident["denorm"]["field_normalizer"] = None
    ident["denorm"]["field_normalizer_absent_reason"] = None
    ident["denorm"]["stats_ref"]["zarr_paths"] = ["D:/x.zarr"]
    probs = _identity_answerability_problems(ident)
    assert any("absent_reason" in p for p in probs), probs


# ---------------------------------------------------------------------------
# 4. 「哪個 config」/ seed / JSON 型別
# ---------------------------------------------------------------------------

def _minimal_identity() -> dict:
    """組一份「三問皆可答」的最小 identity，供逐項拔欄位測試。"""
    return {
        "schema_version": 1,
        "git": {"commit": "a" * 40},
        "seed": {"seed": 42, "split_seed": 7},
        "config": {"sha256": "b" * 64, "model": {"arch": "x"}},
        "inference_hparams": {"apply_ic_mask": True},
        "denorm": {
            "sha256": "c" * 64,
            "stats_ref": {"zarr_paths": [], "dataset_names": []},
            "field_normalizer": {"buffers": {}},
            "field_normalizer_absent_reason": None,
        },
    }


def test_minimal_identity_is_answerable() -> None:
    """負控：完整 identity 必須零問題（否則後面的拔欄位測試沒有意義）。"""
    assert _identity_answerability_problems(_minimal_identity()) == []


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda i: i["config"].pop("model"), "config.model"),
        (lambda i: i["config"].__setitem__("model", {}), "config.model"),
        (lambda i: i.pop("seed"), "seed"),
        (lambda i: i["git"].__setitem__("commit", None), "git.commit"),
        (lambda i: i["denorm"].__setitem__("sha256", None), "denorm.sha256"),
        (lambda i: i.__setitem__("inference_hparams", {}), "inference_hparams"),
    ],
)
def test_each_missing_field_is_detected(mutate, expect: str) -> None:
    """逐一拔掉必要欄位，規則函式都必須指名該欄位（而非籠統通過）。"""
    ident = _minimal_identity()
    mutate(ident)
    probs = _identity_answerability_problems(ident)
    assert any(expect in p for p in probs), (expect, probs)


def test_non_json_primitive_identity_is_rejected(tmp_path, cfg, monkeypatch) -> None:
    """identity 混入 Path 物件 → 下游 weights_only=True 載入會壞，必須在落盤時擋。"""
    real_build = ident_mod.build_identity

    def _build_with_path(*args, **kwargs):
        ident = real_build(*args, **kwargs)
        ident["config"]["config_source"] = Path("C:/x/y.yaml")  # 非 JSON 原生型別
        return ident

    monkeypatch.setattr(ckpt_mod, "build_identity", _build_with_path)
    with pytest.raises(CheckpointIdentityError, match="JSON"):
        save_checkpoint(tmp_path, step=103, tag=None, cfg=cfg, **_rig())


# ---------------------------------------------------------------------------
# 5. 覆寫可見化
# ---------------------------------------------------------------------------

def test_overwriting_existing_ckpt_is_recorded(tmp_path, cfg) -> None:
    """``torch.save`` 無聲覆蓋同路徑舊檔 → 必須留下 ``ckpt_overwrite`` 記錄。"""
    records: list[dict] = []
    sink = lambda r, force_flush=False: records.append(r)  # noqa: E731
    save_checkpoint(tmp_path, step=200, tag="final", cfg=cfg, write_jsonl=sink, **_rig())
    assert not [r for r in records if r["event"] == "ckpt_overwrite"]
    save_checkpoint(tmp_path, step=200, tag="final", cfg=cfg, write_jsonl=sink, **_rig())
    ow = [r for r in records if r["event"] == "ckpt_overwrite"]
    assert len(ow) == 1 and ow[0]["prev_size_bytes"] > 0


# ---------------------------------------------------------------------------
# 6. 繞過 checkpointer 的 torch.save（ast，非 regex）
# ---------------------------------------------------------------------------

def _torch_save_lines(source: str, filename: str = "<t>") -> list[int]:
    """以 ``ast`` 找出 ``torch.save`` callsite 的行號（含 import 別名形式）。

    ⛔ 不用 regex：regex 會掃到註解／docstring 內的字樣，也看不出別名。
    """
    tree = ast.parse(source, filename=filename)
    mods = {"torch"}
    funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "torch" and a.asname:
                    mods.add(a.asname)
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            for a in node.names:
                if a.name == "save":
                    funcs.add(a.asname or "save")

    def _dotted(n: ast.AST) -> str:
        parts: list[str] = []
        cur = n
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return ""

    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        head = name.split(".")[0] if name else ""
        if (name.endswith(".save") and head in mods) or name in funcs:
            hits.append(node.lineno)
    return sorted(hits)


def test_scanner_finds_all_call_forms_and_ignores_text() -> None:
    """★反測掃描器本身：三種寫法都要抓到，註解/字串裡的字樣不得誤報。

    沒有這個反測，「掃不到 ⇒ 沒有繞過」是不可採信的結論（本專案已踩過
    「工具靜默回答沒問題」四次）。
    """
    src = (
        '"""docstring 提到 torch.save 不算"""\n'
        "import torch\n"
        "import torch as th\n"
        "from torch import save as tsave\n"
        "# torch.save(x, p) 這是註解\n"
        "MSG = 'torch.save(x, p)'\n"
        "torch.save(1, 'a')\n"
        "th.save(2, 'b')\n"
        "tsave(3, 'c')\n"
    )
    assert _torch_save_lines(src) == [7, 8, 9]


def test_no_new_bare_torch_save_in_live_code() -> None:
    """★防回歸：``src/lbmnca/nca/tes`` 與 ``scripts`` 下裸 ``torch.save`` 的檔案清單只准變短。

    已知債見 :data:`KNOWN_BARE_SAVE_FILES`。新增任何未登錄的 callsite ⇒ 失敗，
    因為那代表又多一條「產出無身分 ckpt」的路徑。
    """
    found: dict[str, list[int]] = {}
    scanned = 0
    for d in LIVE_SCAN_DIRS:
        for p in (REPO_ROOT / d).rglob("*.py"):
            if "__pycache__" in p.parts or "archive" in [x.lower() for x in p.parts]:
                continue
            scanned += 1
            hits = _torch_save_lines(p.read_bytes().decode("utf-8-sig"), str(p))
            if hits:
                found[p.relative_to(REPO_ROOT).as_posix()] = hits
    # 掃描器真的看到了東西（避免路徑寫錯造成「零發現＝通過」）
    assert scanned > 50, f"只掃到 {scanned} 個檔，掃描路徑可能有誤"
    assert found, "一個 torch.save 都沒掃到 —— 掃描範圍必定有誤"
    unexpected = {k: v for k, v in found.items() if k not in KNOWN_BARE_SAVE_FILES}
    assert not unexpected, f"新增未登錄的裸 torch.save：{json.dumps(unexpected, indent=2)}"
