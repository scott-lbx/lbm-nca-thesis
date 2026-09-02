"""訓練檢查點管理器：儲存/載入模型狀態 + 非同步視覺化生成 + JSONL 指標記錄。

★ TES 訓練主線的存檔實作：:func:`save_checkpoint`
------------------------------------------------------------------
``Checkpointer.save`` 與 ``TESTrainer._save_checkpoint`` 過去是兩份幾乎逐行相同的
實作（payload 組裝、identity 建立、``torch.save``、jsonl 記錄），必須同步維護，
同一邏輯多份實作正是難以維護的根因。現在**這兩者**都委派至本模組的
:func:`save_checkpoint`（``checkpointer.Checkpointer.save`` 與
``training/trainer.py::TESTrainer._save_checkpoint`` 各有一處呼叫），行為改一次即兩處生效。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader

from .....paths import REPO_ROOT
from ..jsonl import dumps_record

from .identity import (
    IDENTITY_KEY,
    build_identity,
    extract_identity,
    log_identity_on_load,
    sha256_of,
    verify_identity,
)


logger = logging.getLogger(__name__)


#: ★顯式降級開關：設為 ``1``/``true``/``yes``/``on`` 時，允許存出「無身分」ckpt。
#:
#: 預設（未設或設為其他值）＝**不允許**：identity 建立失敗或落盤後驗證不到 identity
#: 一律 :class:`CheckpointIdentityError`，而非只留一行 warning。理由見
#: :func:`save_checkpoint` docstring。
ALLOW_UNIDENTIFIED_CKPT_ENV = "TES_ALLOW_UNIDENTIFIED_CKPT"

#: ★細粒度逃生口：僅豁免 **git 區塊**。
#:
#: 背景：``git_info()`` 在 git 不可用（未安裝／不在 PATH／非 repo）時刻意回
#: ``commit=None``，而落盤斷言把 ``git.commit`` 非空列為必要條件 → 在無 git 的環境
#: 訓練會在**第一顆 ckpt** 直接 :class:`CheckpointIdentityError` 中止。
#: 此時唯一的逃生口是 :data:`ALLOW_UNIDENTIFIED_CKPT_ENV`，但那會把整個身分機制關掉
#: （config／資料／seed 的檢查一併失效）—— 粗到把嬰兒和洗澡水一起倒掉。
#:
#: 設為 ``1``/``true``/``yes``/``on`` 時：**只**跳過「哪個 commit」這一條，其餘
#: answerability 條件維持強制。預設未設 ＝ 維持現行嚴格行為。
ALLOW_MISSING_GIT_ENV = "TES_ALLOW_MISSING_GIT_IDENTITY"

#: 環境變數視為「開啟」的字面值。
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class CheckpointIdentityError(RuntimeError):
    """checkpoint 身分區塊建立失敗，或落盤後 re-load 驗證不到身分。

    刻意設計為**中止訓練**：ckpt 的身分資訊是「這顆權重能不能被正確重現/部署」目前**僅有的
    程式層**依據（``src/lbmnca/nca/tes`` 內無第二套 ckpt 溯源機制；run 目錄
    ``metrics.json`` 的 startup event 屬外部記錄，只能人工核對、且可與權重分離），
    靜默退回無身分等同於整輪訓練的產出不可追溯。
    """


def _env_allows_unidentified() -> bool:
    """讀環境變數 :data:`ALLOW_UNIDENTIFIED_CKPT_ENV`，判斷是否顯式允許無身分 ckpt。

    Returns:
        True 表示使用者已顯式開啟降級；預設 False。
    """
    return os.environ.get(ALLOW_UNIDENTIFIED_CKPT_ENV, "").strip().lower() in _TRUTHY


def _env_allows_missing_git() -> bool:
    """讀環境變數 :data:`ALLOW_MISSING_GIT_ENV`，判斷是否只豁免 git 區塊。

    Returns:
        True 表示使用者已顯式允許「git 不可用時仍存 ckpt」；預設 False。
    """
    return os.environ.get(ALLOW_MISSING_GIT_ENV, "").strip().lower() in _TRUTHY


def checkpoint_filename(step: int, tag: Optional[str]) -> str:
    """依步數/標籤產生 checkpoint 檔名（**TES 訓練主線**存檔命名的來源）。

    本函式在 :func:`save_checkpoint` 內呼叫，是走此主線時檔名產生的唯一來源。

    Args:
        step: 目前訓練步數。
        tag: 可選標籤（如 ``"final"``）；None 時用步數命名。

    Returns:
        檔名字串（``ckpt_step{step:07d}.pth`` 或 ``ckpt_{tag}.pth``）。
    """
    return f"ckpt_step{step:07d}.pth" if tag is None else f"ckpt_{tag}.pth"


def _emit(write_jsonl: Optional[Callable[..., None]], record: dict) -> None:
    """若呼叫端提供了 jsonl writer 就寫一筆（force flush）；否則略過。

    Args:
        write_jsonl: ``(record, force_flush=...)`` 介面的 writer，可為 None。
        record: 待寫入的 dict。
    """
    if write_jsonl is None:
        return
    try:
        write_jsonl(record, force_flush=True)
    except Exception as exc:  # noqa: BLE001 — 記錄失敗不得反過來遮蔽主錯誤
        logger.warning("checkpoint jsonl 記錄失敗：%s", exc)


def _assert_identity_on_disk(
    path: Path,
    *,
    expected: Optional[dict],
    allow_unidentified: bool,
) -> Optional[dict]:
    """把剛存好的 ckpt **重新讀回**，斷言 identity 確實存在且**足以回答三個問題**。

    只檢查「有沒有走到硬碟上」——``build_identity`` 成功不代表 ``torch.save`` 之後
    讀回來還在（序列化路徑、weights_only 相容性、寫入截斷都可能吃掉它），故驗證
    必須發生在落盤之後。

    ★ 驗收斷言
    ------------------------------------------------------------------
    「任一 ckpt 可**單獨**回答：哪個 config／哪份資料／哪個 commit」。
    舊版斷言只檢查三件事（``inference_hparams`` 非空、``config.sha256`` 存在、
    ``denorm`` 是 dict），三個驗收問題**沒有一個被真正檢查到**：

    ====================  ========================================================
    驗收問題              舊版為何擋不住
    ====================  ========================================================
    哪個 commit           ``git`` 區塊完全未檢查。``git_info()`` 在 git 不可用時
                          回傳 ``commit=None``（刻意不填假值），ckpt 照樣存出且
                          斷言通過 → 這顆權重永遠無法對應到程式碼版本。
    哪份資料              只檢查 ``isinstance(denorm, dict)``。一個
                          ``stats_ref`` 全空、``field_normalizer=None`` 的
                          denorm 區塊仍是 dict → 通過，但反正規化空間不可還原。
    哪個 config           只檢查 ``sha256`` 這個**不可反解**的字串。缺 ``model``
                          快照時無法逐欄 diff，只能得到「不同」而說不出差在哪。
    ====================  ========================================================

    三個缺口全部朝「看起來通過」失效，與 L0 首輪 9/9 缺口同向。

    另檢查 identity 是否 JSON-serialisable：``identity.py`` 的設計原則是身分區塊
    只由 JSON 原生型別構成（供 ``weights_only=True`` 安全載入）；混入 ``Path`` /
    ``numpy`` 物件時舊版斷言不會察覺，卻會讓下游安全載入端整個載不動。

    Args:
        path: 已存檔的 ckpt 路徑。
        expected: 存檔前建立的 identity（None 表示建立階段已降級）。
        allow_unidentified: True 時允許「讀不到 identity」，僅記 warning。

    Returns:
        讀回的 identity dict；降級路徑下可能為 None。

    Raises:
        CheckpointIdentityError: 讀不回檔案、identity 缺失、結構不完整、無法回答
            三個驗收問題，或 round-trip 後內容與寫入前不一致（且未顯式降級）。
    """
    try:
        reloaded = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 — 轉成明確的 ckpt 身分錯誤
        raise CheckpointIdentityError(
            f"checkpoint 落盤後無法重新載入驗證：{path}（{type(exc).__name__}: {exc}）"
        ) from exc

    got = extract_identity(reloaded)
    if got is None:
        msg = (
            f"checkpoint 已寫出但**無身分資訊**：{path}\n"
            "  → 此 ckpt 無 commit / config sha / 推論敏感超參數 / 反正規化常數，"
            "無法在載入端做任何比對（psi_lowpass 0.8/0.9 事故類型無法攔截）。\n"
            f"  → 若確實需要存無身分 ckpt，請顯式設定環境變數 "
            f"{ALLOW_UNIDENTIFIED_CKPT_ENV}=1。"
        )
        if allow_unidentified:
            logger.warning("%s（已由 %s 顯式降級）", msg, ALLOW_UNIDENTIFIED_CKPT_ENV)
            return None
        raise CheckpointIdentityError(msg)

    problems = _identity_answerability_problems(
        got, allow_missing_git=_env_allows_missing_git()
    )
    if expected is not None and sha256_of(got) != sha256_of(expected):
        problems.append("round-trip 後 identity 內容與寫入前不一致")
    if problems:
        raise CheckpointIdentityError(
            f"checkpoint 身分區塊不完整：{path}\n  → " + "\n  → ".join(problems)
        )
    return got


def _identity_answerability_problems(
    got: dict, *, allow_missing_git: bool = False
) -> list[str]:
    """列出 identity **無法回答哪個驗收問題**；全部可回答時回傳空 list。

    抽成獨立函式的理由：這份規則必須由落盤斷言與離線盤點工具**共用同一份**，否則會出現
    「訓練端說合格、盤點端說不合格」的兩套標準。新增消費者請一律 import 本函式，
    勿另寫規則副本。

    Args:
        got: 從 ckpt 讀回的 identity 區塊。
        allow_missing_git: True 時**只**豁免 ``git.commit`` 這一條（其餘不變），
            由 :data:`ALLOW_MISSING_GIT_ENV` 顯式開啟；預設 False（維持嚴格）。

    Returns:
        問題描述字串清單（空 list ＝ 三個驗收問題皆可回答）。
    """
    problems: list[str] = []

    # ── 推論敏感超參數（psi_lowpass 事故的直接攔截面）──────────────
    hp = got.get("inference_hparams")
    if not isinstance(hp, dict) or not hp:
        problems.append("inference_hparams 缺失或為空")

    # ── Q1「哪個 config」：sha256（可比對）＋ model 快照（可逐欄 diff）──
    cfg_blk = got.get("config")
    if not isinstance(cfg_blk, dict) or not cfg_blk.get("sha256"):
        problems.append("config.sha256 缺失 → 無法回答「哪個 config」")
    elif not isinstance(cfg_blk.get("model"), dict) or not cfg_blk["model"]:
        problems.append(
            "config.model 快照缺失或為空 → 不一致時只能得到「有差異」而說不出差在哪"
        )

    # ── Q2「哪份資料」：來源清單 或 內嵌反正規化常數，至少要有其一 ──
    denorm = got.get("denorm")
    if not isinstance(denorm, dict):
        problems.append("denorm 區塊缺失 → 無法回答「哪份資料」")
    else:
        ref = denorm.get("stats_ref")
        ref = ref if isinstance(ref, dict) else {}
        has_source = any(
            ref.get(k) for k in ("zarr_path", "zarr_paths", "stats_path", "stats_paths", "dataset_names")
        )
        has_embedded = denorm.get("field_normalizer") is not None
        if not has_source and not has_embedded:
            problems.append(
                "denorm 既無 stats_ref 來源（zarr/stats/dataset 全空）也未內嵌反正規化常數"
                " → 無法回答「哪份資料」，正規化空間不可還原"
            )
        if not has_embedded and not str(denorm.get("field_normalizer_absent_reason") or "").strip():
            # 未內嵌時必須**明說原因**；靜默省略與「本來就沒有」不可混為一談。
            problems.append(
                "denorm.field_normalizer 未內嵌，且未記錄 field_normalizer_absent_reason"
                "（不得靜默省略）"
            )
        if not denorm.get("sha256"):
            problems.append("denorm.sha256 缺失")

    # ── Q3「哪個 commit」：git 不可用時 git_info 回傳 None，舊版完全未檢查 ──
    git_blk = got.get("git")
    if not isinstance(git_blk, dict) or not git_blk.get("commit"):
        if allow_missing_git:
            logger.warning(
                "[ckpt_identity] git.commit 缺失但已由 %s 顯式豁免："
                "此 ckpt 無法回答「哪個 commit」，仍會被存出。",
                ALLOW_MISSING_GIT_ENV,
            )
        else:
            problems.append(
                "git.commit 缺失（git 不可用時 git_info 刻意回 None）"
                " → 無法回答「哪個 commit」，此權重永遠對不回程式碼版本。"
                f"git 確實不可用時，請顯式設 {ALLOW_MISSING_GIT_ENV}=1"
                f"（只豁免 git 一條；⛔ 不要用 {ALLOW_UNIDENTIFIED_CKPT_ENV}，那會關掉整個身分機制）"
            )

    # ── seed：重現性的最低限度記錄 ──────────────────────────────────
    seed_blk = got.get("seed")
    if not isinstance(seed_blk, dict) or "seed" not in seed_blk:
        problems.append("seed 區塊缺失 → 無法回答「哪個 seed」")

    # ── JSON 原生型別紀律（weights_only=True 安全載入的前提）──────────
    #   ⛔ 刻意**不加** ``allow_nan=False``：``_vis_*`` 等 buffer 在 stats 未提供
    #   vis_range 時合法為 NaN（見 stage0 報告「新缺口」段），那是另一層的問題，
    #   不該由存檔端擋。這裡只驗型別：不得混入 Path / numpy / dataclass 物件。
    try:
        json.dumps(got)
    except TypeError as exc:
        problems.append(
            f"identity 含非 JSON 原生型別（{type(exc).__name__}: {exc}）"
            " → 下游 weights_only=True 安全載入會失敗"
        )

    return problems


def save_checkpoint(
    output_dir: str | Path,
    *,
    step: int,
    tag: Optional[str],
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    cfg: Any,
    field_normalizer: Any,
    cond_normalizer: Any,
    write_jsonl: Optional[Callable[..., None]] = None,
    allow_unidentified: Optional[bool] = None,
) -> Path:
    """★**TES 訓練主線**存 checkpoint 的實作：組 payload → 建 identity → 存檔 → 落盤後驗證。

    「單一」僅在「``Checkpointer.save`` 與 ``TESTrainer._save_checkpoint`` 兩個入口共用同一份
    實作」這個意義下成立。

    設計理由
    ------------------------------------------------------------------
    ``collect_inference_hparams`` 刻意用**無 default** 的 ``getattr`` 讀 59 個欄位，
    只要 config dataclass 少任一欄就 ``AttributeError``。舊版把 ``build_identity``
    包在 ``try/except`` 裡只記 warning，等於「最需要身分機制生效的情境（新增/改名
    config 欄位）正好是它靜默失效的情境」，而且 warning 不進 ``metrics.json``。
    現在改為：

    1. identity 建立失敗 → **不存檔**、寫 jsonl ``event=ckpt_identity_error``、
       拋 :class:`CheckpointIdentityError`（訓練在第一顆 ckpt 就大聲失敗，
       而非跑完整輪才發現全部 ckpt 無身分）。
    2. ``torch.save`` 之後 **re-load 驗證** identity 真的在檔案裡且可解析
       （見 :func:`_assert_identity_on_disk`）。
    3. 需要存無身分 ckpt 時，必須**顯式**開啟環境變數
       ``TES_ALLOW_UNIDENTIFIED_CKPT=1``（或傳 ``allow_unidentified=True``）。

    Args:
        output_dir: run 輸出目錄。
        step: 目前訓練步數。
        tag: 可選標籤（如 ``"final"`` / ``"early_stop_best"``）；None 時用步數命名。
        model: 模型（取 ``state_dict`` 與參數量）。
        optimizer: optimizer（取 ``state_dict``）。
        scheduler: scheduler（取 ``state_dict``）。
        scaler: AMP GradScaler（取 ``state_dict``）。
        cfg: ``TESTrainConfig``，供 :func:`~.identity.build_identity`。
        field_normalizer: ``FieldNormalizer``（內嵌反正規化常數）；可為 None。
        cond_normalizer: ``TESCondNormalizer``；可為 None。
        write_jsonl: ``(record, force_flush=...)`` 介面的 metrics.json writer；
            存檔事件與身分錯誤都會寫入（舊版存檔端完全沒有 jsonl 記錄）。
        allow_unidentified: 顯式降級開關；None 時讀環境變數
            :data:`ALLOW_UNIDENTIFIED_CKPT_ENV`（預設不允許）。

    Returns:
        存檔後的 checkpoint 路徑。

    Raises:
        CheckpointIdentityError: identity 建立失敗或落盤後驗證不到身分，
            且未顯式降級。
    """
    if allow_unidentified is None:
        allow_unidentified = _env_allows_unidentified()
    path = Path(output_dir) / checkpoint_filename(step, tag)

    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }

    identity: Optional[dict] = None
    try:
        identity = build_identity(
            cfg,
            field_normalizer=field_normalizer,
            cond_normalizer=cond_normalizer,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 — 轉為顯式失敗（預設）或顯式降級
        _emit(
            write_jsonl,
            {
                "event": "ckpt_identity_error",
                "phase": "build",
                "step": step,
                "tag": tag,
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
                "allow_unidentified": allow_unidentified,
            },
        )
        msg = (
            f"checkpoint identity 建立失敗（step={step}, tag={tag}）："
            f"{type(exc).__name__}: {exc}\n"
            "  → 多半是 config dataclass 缺少 INFERENCE_SENSITIVE_* 清單上的欄位"
            "（collect_inference_hparams 刻意不 fallback）。\n"
            "  → 修 config 欄位；若確實需要存無身分 ckpt，請顯式設定環境變數 "
            f"{ALLOW_UNIDENTIFIED_CKPT_ENV}=1。"
        )
        if not allow_unidentified:
            logger.error("%s", msg, exc_info=True)
            raise CheckpointIdentityError(msg) from exc
        logger.warning(
            "%s（已由 %s 顯式降級，ckpt 仍會儲存但無身分資訊）",
            msg,
            ALLOW_UNIDENTIFIED_CKPT_ENV,
        )
    else:
        payload[IDENTITY_KEY] = identity

    # ★覆寫可見化：`torch.save` 會無聲蓋掉同路徑舊檔。同一 output_dir 被重跑時，
    #   論文數字所引用的那顆權重就這樣消失，而 log 裡看不出發生過。不阻擋（阻擋會
    #   壞掉正常續訓），但一定要留下記錄。
    if path.exists():
        stat = path.stat()
        logger.warning(
            "checkpoint 覆寫既有檔案：%s（舊檔 %d bytes, mtime=%.0f）——"
            "舊權重無法復原，若該檔曾被引用請先備份 run 目錄。",
            path,
            stat.st_size,
            stat.st_mtime,
        )
        _emit(
            write_jsonl,
            {
                "event": "ckpt_overwrite",
                "step": step,
                "tag": tag,
                "path": str(path),
                "prev_size_bytes": stat.st_size,
                "prev_mtime": stat.st_mtime,
            },
        )

    torch.save(payload, path)

    # ★落盤後驗證：build 成功不等於檔案裡有；缺則 raise（不是 warning）。
    try:
        _assert_identity_on_disk(
            path, expected=identity, allow_unidentified=allow_unidentified
        )
    except CheckpointIdentityError as exc:
        _emit(
            write_jsonl,
            {
                "event": "ckpt_identity_error",
                "phase": "verify_on_disk",
                "step": step,
                "tag": tag,
                "path": str(path),
                "error": str(exc),
                "allow_unidentified": allow_unidentified,
            },
        )
        logger.error("%s", exc)
        raise

    logger.info("Checkpoint 已儲存：%s", path)
    _emit(
        write_jsonl,
        {
            "event": "checkpoint",
            "step": step,
            "path": str(path),
            "tag": tag,
            "has_identity": identity is not None,
        },
    )
    return path


def load_model_state(
    model: Any,
    ckpt: dict,
    cfg: Any,
    *,
    ckpt_path: str | Path | None = None,
    field_normalizer: Any | None = None,
    on_mismatch: Optional[str] = None,
    context: str = "",
    strict: bool = False,
    allow_sidecar: bool = False,
) -> dict:
    """評測／製圖／部署端載入 ckpt 權重的**建議共用實作**：先核身分，再 load_state_dict。

    ⚠ 這**不是**唯一入口：codebase 內仍有檔案自行呼叫 ``load_state_dict`` 而完全
    不經本函式，各自寫著同一段 ``model.load_state_dict(ckpt["model"], strict=False)``
    + ``print``，不呼叫 :func:`~.identity.verify_identity`。

    設計理由
    ------------------------------------------------------------------
    多份各自實作的後果是：修好其中一份不代表其他各份會跟上（同一邏輯多份實作
    在載入側的同構版本）。

    ``strict=False`` 本身不是錯（warm-restart 新模組確實會有 missing keys），錯的是
    **沒有任何人看那些 keys**。本函式一律把 missing/unexpected 記進 log，
    並在 unexpected 非空時升為 warning（那代表 ckpt 有模型吃不下的權重）。

    Args:
        model: 已建構好的 ``nn.Module``。
        ckpt: ``torch.load`` 得到的 checkpoint dict（需含 ``"model"``）。
        cfg: 這次要用的 config（``TESTrainConfig``）。
        ckpt_path: ckpt 路徑（訊息 / sidecar 用）。
        field_normalizer: 這次建立的 ``FieldNormalizer``；提供時額外比對反正規化常數。
        on_mismatch: ``"warn"`` / ``"raise"``；``None`` 時讀 ``cfg.ckpt_identity_mismatch``
            （⛔ 禁 fallback：cfg 缺該欄位即 AttributeError）。該欄位預設 ``"raise"``；
            續訓路徑另走 ``cfg.ckpt_identity_mismatch_resume``（預設 ``"warn"``）。
        context: 呼叫端標識（寫進 log，如 ``"rq2_chain_eval"``）。
        strict: 傳給 ``load_state_dict``；預設 False 維持既有各腳本行為。
        allow_sidecar: 是否允許以旁掛 ``{ckpt}.identity.json`` 補身分（顯式出聲）。

    Returns:
        :func:`~.identity.verify_identity` 的報告 dict。

    Raises:
        ValueError: ``on_mismatch="raise"`` 且身分不一致。
    """
    if on_mismatch is None:
        on_mismatch = cfg.ckpt_identity_mismatch
    log_identity_on_load(
        ckpt, ckpt_path=ckpt_path, context=context, allow_sidecar=allow_sidecar
    )
    report = verify_identity(
        ckpt,
        cfg,
        field_normalizer=field_normalizer,
        ckpt_path=ckpt_path,
        on_mismatch=on_mismatch,
        context=context,
        allow_sidecar=allow_sidecar,
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if missing:
        logger.info("[ckpt_load/%s] missing keys（%d）：%s", context, len(missing), missing)
    if unexpected:
        logger.warning(
            "[ckpt_load/%s] unexpected keys（%d）：%s —— ckpt 內有此模型吃不下的權重，"
            "多半代表 config 與訓練時不同。",
            context,
            len(unexpected),
            unexpected,
        )
    return report


class Checkpointer:
    """訓練檢查點管理器：儲存/載入模型狀態 + 非同步視覺化生成 + JSONL 指標記錄。

    所有方法透過 ``TrainerContext`` 存取訓練器共享狀態，不直接依賴 TESTrainer。
    """

    def __init__(self, ctx: Any) -> None:
        """初始化 Checkpointer。

        Args:
            ctx: TrainerContext 共享狀態容器。
        """
        self.ctx = ctx
        self._async_eval_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # 儲存 / 載入
    # ------------------------------------------------------------------

    def save(self, tag: Optional[str] = None) -> Path:
        """儲存訓練 checkpoint 至 output_dir（含 ``identity`` 身分區塊）。

        本方法只做 ctx → 參數的搬運，實際存檔邏輯全部委派給模組級的
        :func:`save_checkpoint`（與 ``TESTrainer._save_checkpoint`` 共用同一份
        實作）。identity 缺失的處置、落盤後驗證、降級開關皆見該函式。

        Args:
            tag: 可選標籤（如 ``"final"``、``"emergency"``）；
                 ``None`` 時用目前步數命名（``ckpt_step{step:07d}.pth``）。

        Returns:
            儲存後的 checkpoint 檔案 Path。

        Raises:
            CheckpointIdentityError: 身分區塊建立/落盤驗證失敗且未顯式降級。
        """
        ctx = self.ctx
        return save_checkpoint(
            ctx.output_dir,
            step=ctx.step,
            tag=tag,
            model=ctx.model,
            optimizer=ctx.optimizer,
            scheduler=ctx.scheduler,
            scaler=ctx.scaler,
            cfg=ctx.cfg,
            field_normalizer=ctx.field_normalizer,
            cond_normalizer=ctx.normalizer,
            write_jsonl=self.write_jsonl,
        )

    def load(self, ckpt_path: Path) -> None:
        """從 checkpoint 恢復完整訓練狀態（step + model + optimizer + scheduler + scaler）。

        載入前先比對 ckpt 內 ``identity`` 與當前 config 的推論敏感超參數；
        不一致時記 warning（舊 ckpt 無 identity 時明確標示「無身分資訊」）。

        Args:
            ckpt_path: checkpoint .pth 檔路徑。
        """
        ctx = self.ctx
        ckpt = torch.load(ckpt_path, map_location=ctx.device)
        self._check_identity(ckpt, ckpt_path)
        ctx.step = ckpt["step"]
        ctx.model.load_state_dict(ckpt["model"])
        ctx.optimizer.load_state_dict(ckpt["optimizer"])
        ctx.scheduler.load_state_dict(ckpt["scheduler"])
        ctx.scaler.load_state_dict(ckpt["scaler"])
        logger.info("已從 %s 恢復，step=%d", ckpt_path, ctx.step)

    def load_weights_only(self, ckpt_path: Path) -> None:
        """只載入 model weights；optimizer / scheduler / step 全部重置（warm restart 用）。

        Args:
            ckpt_path: checkpoint .pth 檔路徑。
        """
        ctx = self.ctx
        ckpt = torch.load(ckpt_path, map_location=ctx.device)
        self._check_identity(ckpt, ckpt_path)
        missing, unexpected = ctx.model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            logger.info(
                "Warm restart missing keys (expected for new modules): %s", missing
            )
        if unexpected:
            logger.warning("Warm restart unexpected keys: %s", unexpected)
        logger.info(
            "Warm restart：只載入 model weights from %s，step 從 0 重新計算",
            ckpt_path,
        )

    def _check_identity(self, ckpt: dict, ckpt_path: Path) -> None:
        """比對 ckpt 身分與當前 config；處置由 ``cfg.ckpt_identity_mismatch_resume`` 決定。

        本方法只服務**續訓／warm-restart 載入**，故讀 resume 專用欄位（預設 ``"warn"``），
        與評測／製圖／部署端的 ``cfg.ckpt_identity_mismatch``（預設 ``"raise"``）分開。
        取 ``"warn"`` 的理由是 identity 比對本身異常不得讓跑到一半的訓練無法續跑；
        不一致與「無身分」兩種情形一律留下 log 與 metrics.json 的 ``ckpt_identity`` 事件。

        Args:
            ckpt: 已 ``torch.load`` 的 checkpoint dict。
            ckpt_path: checkpoint 路徑（訊息用）。

        Raises:
            ValueError: ``cfg.ckpt_identity_mismatch_resume == "raise"`` 且身分不一致。
        """
        ctx = self.ctx
        # ⛔ 禁 fallback 存取：缺欄位即 AttributeError（config.py 是單一真相）。
        on_mismatch = ctx.cfg.ckpt_identity_mismatch_resume
        try:
            log_identity_on_load(ckpt, ckpt_path=ckpt_path, context="trainer")
            report = verify_identity(
                ckpt,
                ctx.cfg,
                field_normalizer=ctx.field_normalizer,
                ckpt_path=ckpt_path,
                on_mismatch=on_mismatch,
                context="trainer",
            )
        except ValueError:
            # ★ on_mismatch="raise" 的 ValueError 必須穿過這層 try —— 否則下面那個
            #   寬 except 會把使用者顯式要求的「中止」再度吞成一行 warning。
            if on_mismatch == "raise":
                raise
            logger.warning("checkpoint identity 比對失敗（ValueError）", exc_info=True)
            return
        except Exception as exc:  # noqa: BLE001 — 比對失敗不得阻擋載入
            logger.warning("checkpoint identity 比對失敗：%s", exc, exc_info=True)
            return
        self.write_jsonl(
            {
                "event": "ckpt_identity",
                "ckpt": str(ckpt_path),
                "has_identity": report["has_identity"],
                "identity_source": report["identity_source"],
                "n_mismatch": len(report["mismatches"]),
                "mismatches": report["mismatches"],
                "denorm_status": report["denorm"].get("status"),
                "trained_at_commit": report["git"].get("trained_at_commit"),
            },
            force_flush=True,
        )

    # ------------------------------------------------------------------
    # 視覺化生成
    # ------------------------------------------------------------------

    def generate_images(self) -> None:
        """於 checkpoint 產生評估影像：序列條、HBand 頻譜、分層報告。

        輸出至 ``eval_output/eval_ckpt_{step}/``，並自動清理舊目錄（保留最新 2 份）。
        全程 try/except 包覆：相依不可用或失敗時僅記 warning，不中斷訓練。
        """
        import json as _json

        import numpy as _np

        ctx = self.ctx

        try:
            from ...eval.eval_cleanup import cleanup_eval_ckpts
            from ...eval.eval_plots import (
                plot_hband_spectrum,
                plot_icswap_demo,
                plot_sequence_strip,
                plot_stratified_report,
            )
            from ....plots.train.debug_hband_gt import plot_hband_debug
            from ....plots.train.report_hband_artifact import plot_hband_artifact

            eval_dir = ctx.output_dir / "eval_output"
            ckpt_dir = eval_dir / f"eval_ckpt_{ctx.step:07d}"
            seq_dir = ckpt_dir / "sequence"
            seq_dir.mkdir(parents=True, exist_ok=True)

            fn = ctx.field_normalizer
            k_vis = min(20, max(_get_val_k(ctx) * 2, 8))
            # ★ 顯示速度上限需與 SSOT（`stats.json` 的 `vis_range.vel_max`）一致，
            #   不可硬編：偏寬會把像素值壓向 0、使弱結構被系統性淡化，而訓練期
            #   比對圖是「一票否決」的審查核心——偏誤方向對「通過」有利。
            #   改由 normalizer 取（與 eval／deploy 同源），⛔ 不得再硬編。
            _VEL_MAX = float(fn.vel_vis_max)

            def _vm(t: torch.Tensor) -> "_np.ndarray":
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                raw = fn.denorm_mphy(t.to(ctx.device))
                rho = raw[:, 0].clamp_min(1e-6)
                vm = (
                    (raw[:, 3] / rho).square() + (raw[:, 5] / rho).square()
                ).sqrt()
                return _np.clip(
                    (vm / _VEL_MAX).squeeze(0).cpu().numpy(), 0.0, 1.0
                )

            cases: list[dict] = []

            def _gather_vis(loader: Optional[DataLoader], n_take: int) -> None:
                if loader is None:
                    return
                taken = 0
                with torch.inference_mode():
                    for batch in loader:
                        if taken >= n_take:
                            break
                        traj_cpu = batch["trajectory"]
                        B, T, C, H, W = traj_cpu.shape
                        n_tgt = min(k_vis, T - 1)
                        if n_tgt < 1:
                            continue
                        n_steps = n_tgt
                        if ctx.cfg.get("mean_module_enabled", False):
                            n_steps = max(n_tgt, min(20, k_vis))
                        static = batch["static"].to(ctx.device)
                        if ctx.cfg.get("mean_module_enabled", False) and "parent_mean_renorm" in batch:
                            ic_phys = batch["parent_mean_renorm"][:1].to(ctx.device).float()
                        else:
                            ic_phys = traj_cpu[:1, 0].to(ctx.device).float()
                        if ctx.cfg.get("mean_module_enabled", False) and "mean_flow_post_conv" in batch:
                            mf = batch["mean_flow_post_conv"][:1].to(ctx.device).float()
                            tgt_slice = mf.unsqueeze(1).expand(-1, n_steps, -1, -1, -1).contiguous()
                        else:
                            tgt_slice = traj_cpu[:1, 1 : 1 + n_tgt].to(ctx.device).float()
                        g_vec = batch["g_vec"][:1].to(ctx.device)
                        p_vec_b = batch["p_vec"][:1].to(ctx.device)
                        static_1 = static[:1]
                        batch_1 = {
                            k: v[:1] if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        static_1, ic_phys, tgt_slice = _renorm_source_to_global(
                            ctx, static_1, ic_phys, tgt_slice, batch_1
                        )
                        fluid_mask = _get_fluid_mask(ctx, static_1)
                        _vis_delta_ctu: torch.Tensor | None = None
                        _vis_cumul_ctu: torch.Tensor | None = None
                        _vis_cumul_seq_norm: torch.Tensor | None = None
                        if ctx.cfg["model"]["cond_dim"] in (5, 19):
                            _vis_cumul_ctu = torch.zeros(1, device=ctx.device)
                            if "delta_ctu" in batch:
                                _dc = batch["delta_ctu"][:1].to(ctx.device)
                                _vis_delta_ctu = (
                                    _dc[:, 0] if _dc.shape[1] > 0 else None
                                )
                            if "ctu_values" in batch:
                                _cv = batch["ctu_values"][:1].to(ctx.device)
                                _end = min(1 + n_steps, _cv.shape[1])
                                _cseq = _cv[:, 1:_end]
                                if _cseq.shape[1] == n_steps:
                                    _vis_cumul_seq_norm = (
                                        ctx.normalizer.normalize_cumulative_ctu(_cseq)
                                    )
                        cond_norm = _build_cond_norm(
                            ctx, g_vec, p_vec_b, _vis_delta_ctu, _vis_cumul_ctu
                        )
                        g_vec_norm = _build_g_vec_norm(ctx, g_vec)
                        delta_seq: torch.Tensor | None = None
                        if _uses_per_step_delta_ctu(ctx) and "delta_ctu" in batch:
                            delta_seq = batch["delta_ctu"][:1, :n_steps].to(ctx.device)
                        state = ctx.model.init_state(static_1, ic_phys)
                        with torch.amp.autocast("cuda", enabled=ctx.cfg["use_amp"]):
                            _, pred_traj = ctx.model(
                                state,
                                cond_norm,
                                fluid_mask,
                                n_steps,
                                True,
                                delta_ctu_seq=delta_seq,
                                g_vec_norm=g_vec_norm,
                                cumul_ctu_norm_seq=_vis_cumul_seq_norm,
                            )
                        ck_b = batch.get("case_key", [""])
                        ic_b = batch.get("ic_type", ["unknown"])
                        cases.append(
                            {
                                "pred_traj": [p[0].cpu().float() for p in pred_traj],
                                "tgt_traj": tgt_slice[0].cpu().float(),
                                "ic": ic_phys[0].cpu().float(),
                                "fm": fluid_mask[0].cpu().float(),
                                "case_key": ck_b[0] if isinstance(ck_b, list) else "",
                                "ic_type": ic_b[0] if isinstance(ic_b, list) else "unknown",
                                "re": float(p_vec_b[0, 0].item()),
                                "n_steps": n_steps,
                                "delta_ctu_val": float(batch["delta_ctu"][0, 0].item()) if "delta_ctu" in batch and batch["delta_ctu"].shape[1] > 0 else 1.0,
                            }
                        )
                        taken += 1

            ctx.model.eval()
            _gather_vis(ctx.val_cold_loader, 2)
            _gather_vis(ctx.val_warm_loader, 2)
            ctx.model.train()

            for case in cases:
                n_st = case["n_steps"]
                preds = case["pred_traj"]
                tgts = case["tgt_traj"]
                ic = case["ic"]
                fm = case["fm"]
                ck = (
                    (case["case_key"] or "case")[:12]
                    .replace("/", "_")
                    .replace("\\", "_")
                )
                mode = case["ic_type"]
                title = f"{case['case_key']} | {mode} | Re={case['re']:.0f}"

                d_ctu = case.get("delta_ctu_val", 1.0)
                if n_st > 1:
                    fidxs = _np.unique(
                        _np.concatenate([[0], _np.round(
                            _np.geomspace(1, n_st - 1, 9)
                        ).astype(int)])
                    ).tolist()[:10]
                else:
                    fidxs = [0]
                n_tgt_avail = tgts.shape[0]
                gt_f = [_vm(tgts[min(fi, n_tgt_avail - 1)]) for fi in fidxs]
                pred_f = [_vm(preds[min(fi, len(preds) - 1)]) for fi in fidxs]
                labels = [f"CTU={fi * d_ctu:.1f}" for fi in fidxs]
                fm_np = fm.squeeze(0).numpy()

                plot_sequence_strip(
                    gt_f,
                    pred_f,
                    labels,
                    title,
                    seq_dir / f"{mode}_{ck}.png",
                )
                # plot_hband_spectrum(
                #     _vm(preds[-1]),
                #     _vm(tgts[-1]),
                #     fm_np,
                #     title,
                #     ckpt_dir / f"hband_{mode}_{ck}.png",
                # )
                # plot_hband_debug(
                #     _vm(preds[-1]),
                #     _vm(tgts[-1]),
                #     fm_np,
                #     title,
                #     ckpt_dir / f"hband_{mode}_{ck}_debug.png",
                # )
                # plot_hband_artifact(
                #     _vm(preds[-1]),
                #     _vm(tgts[-1]),
                #     fm_np,
                #     title,
                #     ckpt_dir / f"hband_{mode}_{ck}_artifact.png",
                # )
                # if mode.startswith("warm"):
                #     di = _np.unique(
                #         _np.round(_np.linspace(0, n_st - 1, 5)).astype(int)
                #     ).tolist()
                #     d_gt = [_vm(tgts[min(i, n_st - 1)]) for i in di]
                #     d_pred = [_vm(preds[min(i, len(preds) - 1)]) for i in di]
                #     plot_icswap_demo(
                #         _vm(ic),
                #         d_pred,
                #         d_gt,
                #         [f"t={i}" for i in di],
                #         title,
                #         ckpt_dir / f"icswap_{ck}.png",
                #     )

            # ── Loss breakdown plot ──────────────────────────────────────
            _metrics_path = ctx.output_dir / "metrics.json"
            if _metrics_path.exists():
                try:
                    import matplotlib as _mpl_mod
                    _mpl_mod.use("Agg")
                    import matplotlib.pyplot as _plt2

                    _train_recs: list[dict] = []
                    with open(_metrics_path, encoding="utf-8") as _mf:
                        for _line in _mf:
                            try:
                                _r = _json.loads(_line)
                                if _r.get("event") == "train":
                                    _train_recs.append(_r)
                            except Exception:
                                pass

                    if _train_recs:
                        _steps_h = [_r["step"] for _r in _train_recs]
                        _PANEL_GROUPS = [
                            ("Loss Total",    ["total"]),
                            ("Fluct MSE",     ["mse", "mse_wf"]),
                            ("ACC / GRAFT",   ["acc_term", "rij_term", "tke_term", "spec_term"]),
                            ("Mean MAE",      ["obsmphy_traj_mae", "obsvphy_vel_mae", "mean_mae"]),
                        ]
                        _valid_groups = [
                            (title, [k for k in keys
                                     if any(k in _r for _r in _train_recs)])
                            for title, keys in _PANEL_GROUPS
                        ]
                        _valid_groups = [(t, ks) for t, ks in _valid_groups if ks]
                        if _valid_groups:
                            _nrows = len(_valid_groups)
                            _fig2, _axes2 = _plt2.subplots(
                                _nrows, 1, figsize=(10, 2.5 * _nrows), sharex=True
                            )
                            if _nrows == 1:
                                _axes2 = [_axes2]
                            for _ai, (_title, _keys) in enumerate(_valid_groups):
                                _ax = _axes2[_ai]
                                for _k in _keys:
                                    _vals = [_r.get(_k, float("nan")) for _r in _train_recs]
                                    _ax.plot(_steps_h, _vals, lw=0.9, label=_k)
                                _ax.set_ylabel(_title, fontsize=8)
                                _ax.legend(fontsize=7, loc="upper right")
                                _ax.grid(alpha=0.3)
                            _axes2[-1].set_xlabel("Step")
                            _fig2.suptitle(
                                f"Loss Breakdown — step {ctx.step}", fontsize=10
                            )
                            _plt2.tight_layout()
                            _plots_dir = ctx.output_dir / "plots"
                            _plots_dir.mkdir(parents=True, exist_ok=True)
                            _lb_path = _plots_dir / "loss_breakdown.png"
                            _fig2.savefig(_lb_path, dpi=90, bbox_inches="tight")
                            _plt2.close(_fig2)
                            logger.info("Loss breakdown → %s", _lb_path)
                except Exception as _lb_err:
                    logger.warning("Loss breakdown plot failed: %s", _lb_err)

            vg_path = eval_dir / "vis_gate_detailed.json"
            if vg_path.exists():
                with open(vg_path, encoding="utf-8") as _f:
                    plot_stratified_report(
                        _json.load(_f), ckpt_dir / "stratified_report.png"
                    )

            cleanup_eval_ckpts(eval_dir, keep_n=2)
            logger.info("Checkpoint eval images → %s", ckpt_dir.name)

        except Exception as _img_err:
            logger.warning(
                "Checkpoint eval images failed at step %d: %s",
                ctx.step,
                _img_err,
                exc_info=True,
            )

    def launch_async_eval(self, ckpt_path: Path) -> None:
        """在獨立子行程啟動 checkpoint 影像/報告生成（不阻塞訓練）。

        前一 async eval 仍在執行時略過本次啟動；腳本缺失時記 warning 後返回。

        Args:
            ckpt_path: 觸發 eval 的 checkpoint 檔路徑。
        """
        ctx = self.ctx
        if (
            self._async_eval_proc is not None
            and self._async_eval_proc.poll() is None
        ):
            logger.info(
                "Async eval still running; skip launch for step %d", ctx.step
            )
            return

        script = REPO_ROOT / "scripts" / "run_async_eval_generation.py"
        if not script.exists():
            logger.warning("Async eval script missing: %s", script)
            return

        log_path = ctx.output_dir / "async_eval.log"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with log_path.open("ab") as log_fp:
            self._async_eval_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_async_eval_generation",
                    "--run-dir",
                    str(ctx.output_dir),
                    "--checkpoint",
                    str(ckpt_path),
                    "--device",
                    "cpu",
                ],
                cwd=str(repo_root),
                stdout=log_fp,
                stderr=log_fp,
                creationflags=creationflags,
            )
        logger.info("Async eval launched for %s", ckpt_path.name)

    # ------------------------------------------------------------------
    # Logging 設定 + JSONL 記錄
    # ------------------------------------------------------------------

    def setup_logging(self) -> None:
        """設定 logging 輸出至 stdout + 檔案，並開啟 metrics.json（line-delimited）。"""
        ctx = self.ctx
        log_path = ctx.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8"),
            ],
        )
        ctx.metrics_file = (ctx.output_dir / "metrics.json").open(
            "a", encoding="utf-8"
        )

    def write_jsonl(self, record: dict, force_flush: bool = False) -> None:
        """寫一筆 JSON record 至 metrics.json（line-delimited 格式）。

        Args:
            record: 待寫入的 dict。
            force_flush: True 時立即 flush（event/error/warning 用）；
                         False 時僅 write buffer，由 100 步 train_step 統一 flush。
        """
        ctx = self.ctx
        if ctx.metrics_file is not None:
            ctx.metrics_file.write(dumps_record(record) + "\n")
            if force_flush:
                ctx.metrics_file.flush()

    # ------------------------------------------------------------------
    # 背景工作
    # ------------------------------------------------------------------

    def submit_background_job(
        self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """在背景執行緒跑 CPU 端生成工作，避免阻塞訓練迴圈。

        Args:
            name: 工作名稱（log 用）。
            fn: 待執行的 callable。
            *args: 傳給 fn 的位置參數。
            **kwargs: 傳給 fn 的關鍵字參數。
        """
        ctx = self.ctx
        if ctx.executor is None:
            fn(*args, **kwargs)
            return

        def _wrapped() -> None:
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                logger.debug(
                    "Background %s skipped at step %d: %s", name, ctx.step, exc
                )

        # 清除已完成的 job（依賴 trainer 端提供 _bg_jobs list；若無則跳過清理）
        bg_jobs = getattr(ctx, "_bg_jobs", None)
        if bg_jobs is not None:
            bg_jobs[:] = [job for job in bg_jobs if not job.done()]
            if bg_jobs:
                logger.debug(
                    "Background %s skipped at step %d: previous job still running",
                    name,
                    ctx.step,
                )
                return
            bg_jobs.append(ctx.executor.submit(_wrapped))
        else:
            ctx.executor.submit(_wrapped)


# ---------------------------------------------------------------------------
# 內部輔助函式（最小化對 trainer private method 的依賴）
# ---------------------------------------------------------------------------

def _get_val_k(ctx: Any) -> int:
    """回傳目前步數對應的 val k（委派至 schedule.get_val_k）。"""
    from .. import schedule as _sched

    return _sched.get_val_k(ctx.cfg["curriculum"], ctx.step)


def _uses_per_step_delta_ctu(ctx: Any) -> bool:
    """判斷是否使用 per-step delta_ctu_seq。"""
    return bool(
        ctx.cfg.get("time_conditioned", False)
        or (
            ctx.cfg["model"].get("arch") == "h_multirate"
            and ctx.cfg["model"].get("cond_dim") == 19
        )
    )


def _get_fluid_mask(ctx: Any, static: torch.Tensor) -> torch.Tensor:
    """從 static 張量取 fluid mask（channel 0，threshold=sdf_threshold）。"""
    return (static[:, 0:1] > ctx.sdf_threshold).float()


def _renorm_source_to_global(
    ctx: Any,
    static: torch.Tensor,
    ic_physics: torch.Tensor,
    tgt: torch.Tensor,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """將 source dataset 正規化轉換至全域正規化（無 source normalizers 時為 identity）。

    Args:
        ctx: TrainerContext。
        static: 靜態場張量。
        ic_physics: 初始條件物理場。
        tgt: 目標序列。
        batch: 原始 batch dict（含 source_key 資訊）。

    Returns:
        轉換後的 (static, ic_physics, tgt) 三元組。
    """
    # 若無 source normalizers，直接回傳（identity）
    if not ctx.source_field_normalizers:
        return static, ic_physics, tgt
    # 若有 source normalizers，需委派給 trainer 的 _renorm_source_to_global
    # 此處提供 fallback（無法從 ctx 取得 trainer instance 時）
    logger.debug(
        "Checkpointer._renorm_source_to_global: source normalizers present but "
        "trainer method unavailable; returning as-is"
    )
    return static, ic_physics, tgt


def _build_cond_norm(
    ctx: Any,
    g_vec: torch.Tensor,
    p_vec: torch.Tensor,
    delta_ctu: Optional[torch.Tensor],
    cumul_ctu: Optional[torch.Tensor],
) -> torch.Tensor:
    """建立正規化後的條件向量（委派至 normalizer）。

    Args:
        ctx: TrainerContext。
        g_vec: 幾何向量 [B, g_vec_dim]。
        p_vec: 物理向量 [B, p_vec_dim]。
        delta_ctu: 可選 delta CTU [B]。
        cumul_ctu: 可選 cumulative CTU [B]。

    Returns:
        正規化條件向量 [B, cond_dim]。
    """
    return ctx.normalizer.normalize_cond(
        g_vec, p_vec, delta_ctu=delta_ctu, cumul_ctu=cumul_ctu
    )


def _build_g_vec_norm(ctx: Any, g_vec: torch.Tensor) -> torch.Tensor:
    """建立正規化後的 g_vec（委派至 normalizer）。

    Args:
        ctx: TrainerContext。
        g_vec: 幾何向量 [B, g_vec_dim]。

    Returns:
        正規化 g_vec [B, g_vec_dim]。
    """
    return ctx.normalizer.normalize_g_vec(g_vec)
