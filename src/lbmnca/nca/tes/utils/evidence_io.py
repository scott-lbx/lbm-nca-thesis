"""evidence_io.py — 證據檔身分後綴 + 禁止同名覆寫。

## 問題

評估產物（診斷 JSON、量測圖）若採**同名覆寫、無 ckpt/日期後綴**，換一顆 checkpoint
重跑就直接蓋掉前一代的產物，論文裡引用的數字與其自稱的來源檔會互斥，且被蓋掉的那份
已無存活來源可查。

## 解法

證據檔名一律帶身分後綴：

    {stem}__{ckpt_sha8}_{YYYYMMDD}{ext}

`ckpt_sha8` = 產生此證據所用 checkpoint 檔案 sha256 的前 8 碼（**內容雜湊**，不是路徑或
run 名稱——同一路徑被覆寫重訓時 sha 會變，路徑不會）。無 checkpoint 的證據（純 GT／
純資料統計）呼叫端必須**明示** `ckpt=None`，token 記為 ``nockpt``。

寫入時 **檔案已存在即 raise `FileExistsError`**——不靜默覆寫、也不靜默跳過。
要覆寫必須顯式傳 `overwrite=True`（CLI 對應 `--allow-overwrite`）。

同時把 provenance（ckpt 絕對路徑與完整 sha256、日期、腳本、argv、git commit、主機）
內嵌進 JSON 的 ``_provenance`` 欄位，使證據檔即使被搬移、改名也還能自證身分。

## 向後相容

**既有檔案一律不動**（它們是歷史證據）。只有**新產生**的檔案帶身分後綴。
讀取端用 `resolve_latest()`：優先取新格式中日期最新者；若完全沒有新格式檔案，
才退回舊的無後綴同名檔並在 stderr 明示「這是無身分的 legacy 檔」。
此退回是**顯式且會出聲**的相容路徑，與專案禁止的「靜默 config fallback」不同。

## 用法

    from lbmnca.nca.tes.utils.evidence_io import resolve_latest, write_json_evidence

    # 寫（自動加後綴 + 防覆寫 + 內嵌 provenance）
    p = write_json_evidence(out_dir, "em2_core_metrics", payload, ckpt=ckpt)

    # 對應的圖檔用同一身分 token（用 reserve_evidence_path，它會做防覆寫檢查）
    fig.savefig(reserve_evidence_path(out_dir, "em2_core_metrics", ckpt=ckpt, ext=".png"))

    # 讀（新格式最新者；無則退回 legacy 同名檔）
    src = resolve_latest(out_dir, "em2_core_metrics")

## 逃生口

`_assert_absent` 的「已存在即 `FileExistsError`」是刻意設計（見上）；但呼叫端不一定
都有 argparse（只寫 npz/json 的繪圖腳本多半無 CLI），撞到時無法從命令列解除。
逃生口有兩層，**優先序由上到下**（任一成立即放行覆寫）：

    1. 呼叫端顯式傳入 overwrite=True
       （有 CLI 的腳本對應 --allow-overwrite，見 `add_overwrite_arg`）。
    2. 環境變數 ``NCA_EVIDENCE_ALLOW_OVERWRITE``（見 `ALLOW_OVERWRITE_ENV`）設為
       truthy 字面值之一（``"1"`` / ``"true"`` / ``"yes"`` / ``"on"``，大小寫不拘）。
       此為「無 CLI 腳本」的統一逃生口，比照 ``TES_ALLOW_UNIDENTIFIED_CKPT`` 慣例
       （見 ``lbmnca.nca.tes.training.checkpoint.checkpointer``）。

環境變數生效時**一律 print 一行含被覆寫檔名的警告到 stderr**，不靜默；沒設或設成
非 truthy 字面值（例如空字串、``"0"``、任意其他字串）一律視為**不允許**——刻意不用
`os.environ.get(k, default)` 之後直接當 bool 吃任意字串（本專案禁止 config
fallback 式的模糊真值判定）。

    # 命令列（有 CLI 的腳本）
    python fig_xxx.py --allow-overwrite

    # 無 CLI 腳本（Windows PowerShell）
    $env:NCA_EVIDENCE_ALLOW_OVERWRITE = "1"
    python some_no_cli_script.py

逃生口一旦生效（無論哪一層），**被覆寫這件事會寫進新檔的 ``_provenance.overwrite``**
（含被取代檔的大小與 mtime），使「這份證據蓋掉過另一份」本身也是可追溯的事實，
而不是只有當下 stderr 那一行會被 log rotation 沖掉。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "SEP",
    "NO_CKPT_TOKEN",
    "ALLOW_OVERWRITE_ENV",
    "ckpt_sha256",
    "ckpt_sha8",
    "identity_suffix",
    "parse_identity",
    "evidence_path",
    "reserve_evidence_path",
    "write_json_evidence",
    "write_png_evidence",
    "mp4_metadata_args",
    "build_provenance",
    "list_evidence",
    "resolve_latest",
    "add_overwrite_arg",
]

# stem 本身常含底線（`mean_chain_rollout_c21loop`），故身分後綴用雙底線分隔，
# 讓 glob/parse 不會誤切 stem。
SEP = "__"
NO_CKPT_TOKEN = "nockpt"

#: 無 CLI 腳本的覆寫逃生口：設為 truthy 字面值時，允許 `_assert_absent` 放行覆寫。
#: 優先序見 module docstring「逃生口」一節：`overwrite=True`（CLI `--allow-overwrite`）
#: > 本環境變數 > 預設 `FileExistsError`。
ALLOW_OVERWRITE_ENV = "NCA_EVIDENCE_ALLOW_OVERWRITE"

#: 環境變數視為「開啟」的字面值（大小寫不拘）；其餘一律視為未開啟——刻意不用
#: `os.environ.get(k, default)` 後直接當 bool 吃任意字串。
_ALLOW_OVERWRITE_TRUTHY = frozenset({"1", "true", "yes", "on"})

# {stem}__{sha8}_{YYYYMMDD}{ext}
_SUFFIX_RE = re.compile(r"^(?P<sha8>[0-9a-f]{8}|" + NO_CKPT_TOKEN + r")_(?P<date>\d{8})$")

_SHA_CACHE: dict[tuple[str, int, int], str] = {}


# ──────────────────────────────────────────────────────────────────────────
# checkpoint 身分
# ──────────────────────────────────────────────────────────────────────────
def ckpt_sha256(ckpt: str | Path) -> str:
    """回傳 checkpoint 檔案內容的完整 sha256（hex）。

    以 (絕對路徑, 大小, mtime_ns) 為 key 做 process 內快取——同一支腳本多次寫證據
    不會重複讀 ~90MB 檔案。檔案被改寫時 mtime/size 會變，快取自然失效。

    Args:
        ckpt: checkpoint 檔案路徑。必須存在，不存在直接 `FileNotFoundError`。

    Returns:
        64 字元的小寫 hex sha256。
    """
    p = Path(ckpt).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"checkpoint 不存在或不是檔案：{p}\n"
            "證據檔的身分後綴以 ckpt 內容雜湊為準，缺 ckpt 即無法標身分——"
            "純 GT／純資料統計的證據請明示 ckpt=None。"
        )
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_CACHE[key] = digest
    return digest


def ckpt_sha8(ckpt: str | Path | None) -> str:
    """checkpoint 身分 token：sha256 前 8 碼；`ckpt` 為 None 時回 ``nockpt``。

    Args:
        ckpt: checkpoint 路徑，或明示 None（純 GT／純資料證據）。
    """
    if ckpt is None:
        return NO_CKPT_TOKEN
    return ckpt_sha256(ckpt)[:8]


def identity_suffix(*, ckpt: str | Path | None, date: str | None = None) -> str:
    """組出 ``{ckpt_sha8}_{YYYYMMDD}`` 身分字串。

    Args:
        ckpt: checkpoint 路徑，或明示 None。
        date: `YYYYMMDD`；None 表示用本機今日日期。
    """
    if date is None:
        date = _dt.date.today().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"date 必須是 YYYYMMDD 八碼數字，收到：{date!r}")
    return f"{ckpt_sha8(ckpt)}_{date}"


# ──────────────────────────────────────────────────────────────────────────
# 路徑組裝
# ──────────────────────────────────────────────────────────────────────────
def evidence_path(
    out_dir: str | Path,
    stem: str,
    *,
    ckpt: str | Path | None,
    ext: str = ".json",
    date: str | None = None,
    overwrite: bool = False,
    check_absent: bool = True,
) -> Path:
    """組出帶身分後綴的證據檔路徑，**預設同時做防覆寫檢查**。

    檔名帶身分後綴只保證「看得出是哪一代」，同 ckpt 同日重跑仍會撞名，故本函式預設
    呼叫 :func:`_assert_absent`。真的只想組字串（例如要先看檔名長相、或要自行處理已
    存在的情形）必須顯式傳 ``check_absent=False``，讓「不檢查」是呼叫端寫在程式裡的
    決定。

    Args:
        out_dir: 輸出目錄（會自動建立）。
        stem: 檔名主體，**不含**副檔名與身分後綴，例如 ``em2_core_metrics``。
            不可含 :data:`SEP`、路徑分隔符或 ``..``。
        ckpt: 產生此證據的 checkpoint 路徑；純 GT／純資料證據明示 None。
        ext: 副檔名（含點），預設 ``.json``；圖檔傳 ``.png``。
        date: `YYYYMMDD`；None 表示今日。
        overwrite: 目標已存在時是否允許覆寫（僅在 `check_absent=True` 時有意義）。
        check_absent: 是否做「已存在即報錯」檢查。預設 True。

    Returns:
        `{out_dir}/{stem}__{ckpt_sha8}_{YYYYMMDD}{ext}`

    Raises:
        ValueError: `stem` 含保留序列／路徑分隔符／`..`，或 `ext` 未以點開頭。
        FileExistsError: `check_absent=True`、目標已存在且未開逃生口。
    """
    if SEP in stem:
        raise ValueError(f"stem 不可含 {SEP!r}（該序列保留給身分後綴）：{stem!r}")
    if "/" in stem or "\\" in stem or ".." in stem or Path(stem).name != stem:
        raise ValueError(
            f"stem 不可含路徑分隔符或 '..'（會逃出 out_dir，使證據落在非預期目錄）：{stem!r}"
        )
    if not ext.startswith("."):
        raise ValueError(f"ext 必須以點開頭，收到：{ext!r}")
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stem}{SEP}{identity_suffix(ckpt=ckpt, date=date)}{ext}"
    if check_absent:
        _assert_absent(path, overwrite=overwrite)
    return path


def _env_allows_overwrite() -> bool:
    """讀環境變數 :data:`ALLOW_OVERWRITE_ENV`，判斷是否顯式允許覆寫既有證據檔。

    Returns:
        True 表示使用者已顯式開啟逃生口；未設定或設成非 truthy 字面值一律 False。
    """
    return os.environ.get(ALLOW_OVERWRITE_ENV, "").strip().lower() in _ALLOW_OVERWRITE_TRUTHY


def _assert_absent(path: Path, *, overwrite: bool) -> dict[str, Any]:
    """檔案已存在則報錯（除非顯式 overwrite，或環境變數逃生口生效）。

    優先序：`overwrite=True` > 環境變數 :data:`ALLOW_OVERWRITE_ENV` > 預設拒絕。
    兩種逃生口都會 print 一行含檔名的警告到 stderr，不靜默。

    Args:
        path: 目標路徑。
        overwrite: 呼叫端是否顯式允許覆寫。

    Returns:
        覆寫紀錄 dict，供 :func:`build_provenance` 內嵌進證據檔本身，
        使「這份證據蓋掉過另一份」不只存在於當下的 stderr。

    Raises:
        FileExistsError: 目標已存在且兩層逃生口都未開。
    """
    record: dict[str, Any] = {
        "used": False,
        "reason": None,
        "replaced_existing": False,
        "replaced_size_bytes": None,
        "replaced_mtime_utc": None,
    }
    if not path.exists():
        return record
    st = path.stat()
    record["replaced_existing"] = True
    record["replaced_size_bytes"] = st.st_size
    record["replaced_mtime_utc"] = _dt.datetime.fromtimestamp(
        st.st_mtime, tz=_dt.timezone.utc).isoformat(timespec="seconds")
    if overwrite:
        record["used"] = True
        record["reason"] = "explicit_overwrite_arg"
        print(
            f"[evidence_io] [WARN] 呼叫端顯式 overwrite=True，覆寫既有證據檔："
            f"{path}（原檔 {st.st_size} bytes, mtime {record['replaced_mtime_utc']}）",
            file=sys.stderr,
        )
        return record
    if _env_allows_overwrite():
        record["used"] = True
        record["reason"] = f"env:{ALLOW_OVERWRITE_ENV}"
        print(
            f"[evidence_io] [WARN] 環境變數 {ALLOW_OVERWRITE_ENV} 已顯式設定，"
            f"允許覆寫既有證據檔：{path}",
            file=sys.stderr,
        )
        return record
    raise FileExistsError(
        f"證據檔已存在，拒絕覆寫：{path}\n"
        "同一顆 ckpt + 同一天已寫過這份證據，預設不覆寫。\n"
        "  · 若上一次跑到一半失敗、要重來 → 顯式傳 overwrite=True"
        "（CLI 為 --allow-overwrite），或先手動把舊檔移走。\n"
        "  · 若這是換了 ckpt 的新量測 → 檔名裡的 sha8 應該不同，"
        "請確認 ckpt 參數指到正確檔案。\n"
        f"  · 若呼叫端無 CLI（無法傳 --allow-overwrite）→ 顯式設環境變數 "
        f"{ALLOW_OVERWRITE_ENV}=1 後重跑（會在 stderr 印出警告，非靜默）。"
    )


# ──────────────────────────────────────────────────────────────────────────
# provenance
# ──────────────────────────────────────────────────────────────────────────
def _git_info(start: Path) -> dict[str, Any]:
    """回傳 `start` 所在 git repo 的 commit / dirty 狀態。

    取不到時把原因寫進 ``error`` 欄位——三個欄位一律填 None 的話，事後無法分辨
    「不是 git repo」「git 不在 PATH」「timeout」，等同無聲失敗。
    """
    info: dict[str, Any] = {
        "repo_root": None, "commit": None, "dirty": None, "error": None,
    }
    try:
        root = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        if root.returncode != 0:
            info["error"] = f"rev-parse --show-toplevel rc={root.returncode}: {root.stderr.strip()[:200]}"
            return info
        info["repo_root"] = root.stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        if head.returncode == 0:
            info["commit"] = head.stdout.strip()
        else:
            info["error"] = f"rev-parse HEAD rc={head.returncode}: {head.stderr.strip()[:200]}"
        status = subprocess.run(
            ["git", "-C", str(start), "status", "--porcelain"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
        else:
            info["error"] = f"status --porcelain rc={status.returncode}: {status.stderr.strip()[:200]}"
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    return info


def build_provenance(
    *,
    ckpt: str | Path | None,
    date: str | None = None,
    extra: dict[str, Any] | None = None,
    overwrite_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """組出可內嵌進證據 JSON 的 ``_provenance`` 區塊。

    Args:
        ckpt: checkpoint 路徑，或明示 None。
        date: `YYYYMMDD`；None 表示今日。
        extra: 呼叫端要額外記的欄位（例如 config 路徑、dataset 名、seed）。
        overwrite_record: :func:`_assert_absent` 的回傳值，使「本檔是覆寫產生的」
            寫進檔案本身而非只出現在 stderr。None（未經 `_assert_absent`）記為
            `{"used": None, ...}`，明確表示「不知道」而非「沒覆寫」。

    Returns:
        provenance dict。ckpt 相關欄位在 `ckpt is None` 時為 None。
    """
    if date is None:
        date = _dt.date.today().strftime("%Y%m%d")
    script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd()
    prov: dict[str, Any] = {
        "schema": "evidence_io/v1",
        "date_tag": date,
        "written_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "ckpt": None,
        "ckpt_sha256": None,
        "ckpt_sha8": ckpt_sha8(ckpt),
        "ckpt_size_bytes": None,
        "ckpt_mtime_utc": None,
        "script": str(script),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git": _git_info(script.parent if script.is_file() else Path.cwd()),
        "overwrite": overwrite_record if overwrite_record is not None else {
            "used": None,
            "reason": "not_checked（呼叫端未經 _assert_absent，覆寫與否不可知）",
            "replaced_existing": None,
            "replaced_size_bytes": None,
            "replaced_mtime_utc": None,
        },
    }
    if ckpt is not None:
        p = Path(ckpt).resolve()
        st = p.stat()
        prov["ckpt"] = str(p)
        prov["ckpt_sha256"] = ckpt_sha256(p)
        prov["ckpt_size_bytes"] = st.st_size
        prov["ckpt_mtime_utc"] = _dt.datetime.fromtimestamp(
            st.st_mtime, tz=_dt.timezone.utc).isoformat(timespec="seconds")
    if extra is not None:
        prov["extra"] = extra
    return prov


# ──────────────────────────────────────────────────────────────────────────
# 寫入
# ──────────────────────────────────────────────────────────────────────────
def write_json_evidence(
    out_dir: str | Path,
    stem: str,
    payload: dict[str, Any],
    *,
    ckpt: str | Path | None,
    date: str | None = None,
    overwrite: bool = False,
    extra_provenance: dict[str, Any] | None = None,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Any = None,
) -> Path:
    """把 `payload` 寫成帶身分後綴的證據 JSON，並內嵌 ``_provenance``。

    Args:
        out_dir: 輸出目錄。
        stem: 檔名主體（不含副檔名／身分後綴）。
        payload: 要寫的 dict。會被淺層複製後加入 ``_provenance``，原 dict 不被改動。
        ckpt: 產生此證據的 checkpoint 路徑；純 GT／純資料證據明示 None。
        date: `YYYYMMDD`；None 表示今日。
        overwrite: 目標已存在時是否覆寫。預設 False → 已存在即 `FileExistsError`。
        extra_provenance: 併入 ``_provenance.extra`` 的額外欄位。
        indent / ensure_ascii / default: 直接轉給 `json.dump`。

    Returns:
        實際寫出的路徑。

    Raises:
        FileExistsError: 目標已存在且 `overwrite=False`。
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"payload 必須是 dict（才塞得下 _provenance），收到 {type(payload).__name__}。"
            "若原本是 list，請包成 {'items': [...]}。"
        )
    path = evidence_path(out_dir, stem, ckpt=ckpt, ext=".json", date=date,
                         check_absent=False)
    overwrite_record = _assert_absent(path, overwrite=overwrite)
    body = dict(payload)
    body["_provenance"] = build_provenance(
        ckpt=ckpt, date=date, extra=extra_provenance,
        overwrite_record=overwrite_record,
    )
    # `.tmp` 中繼檔用 "x"（獨佔建立）而非 "w"：殘留或併發的同名 .tmp 若被靜默截斷，
    # 等同兩個 process 各寫一半再互相 os.replace。
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as f:
            json.dump(body, f, indent=indent, ensure_ascii=ensure_ascii, default=default)
    except FileExistsError as exc:
        raise FileExistsError(
            f"中繼檔已存在，拒絕覆寫：{tmp}\n"
            "可能是上一次寫到一半失敗留下的殘骸，或有另一個 process 正在寫同一份證據。"
            "確認無其他 process 後手動刪除該 .tmp 再重跑。"
        ) from exc
    # 收斂 TOCTOU：_assert_absent 之後、os.replace 之前若目標才出現，
    # os.replace 會靜默蓋掉。此處在不允許覆寫時再驗一次。
    if not overwrite_record["used"] and path.exists():
        tmp.unlink(missing_ok=True)
        raise FileExistsError(
            f"證據檔在檢查之後、落盤之前才出現，拒絕覆寫：{path}\n"
            "表示有另一個 process 同時在寫同一份證據（同 ckpt 同日期）。"
            "請確認是否重複執行，必要時顯式傳 overwrite=True。"
        )
    os.replace(tmp, path)
    return path


def reserve_evidence_path(
    out_dir: str | Path,
    stem: str,
    *,
    ckpt: str | Path | None,
    ext: str,
    date: str | None = None,
    overwrite: bool = False,
) -> Path:
    """給非 JSON 產物（PNG / MP4 …）：組路徑並先做「已存在則報錯」檢查。

    Args:
        out_dir: 輸出目錄。
        stem: 檔名主體。
        ckpt: checkpoint 路徑，或明示 None。
        ext: 副檔名（含點）。
        date: `YYYYMMDD`；None 表示今日。
        overwrite: 已存在時是否允許覆寫。

    Returns:
        可直接交給 `fig.savefig()` 等的路徑。

    Note:
        本函式只保證**檔名帶身分**與**當下不存在**，不內嵌 provenance。
        PNG 請改用 :func:`write_png_evidence`（provenance 寫進 PNG tEXt chunk），
        MP4 請把 :func:`mp4_metadata_args` 的回傳值接到 ffmpeg 命令列。
        仍需要純路徑（其他格式）時才用本函式。
    """
    path = evidence_path(out_dir, stem, ckpt=ckpt, ext=ext, date=date,
                         check_absent=False)
    _assert_absent(path, overwrite=overwrite)
    return path


def write_png_evidence(
    fig: Any,
    out_dir: str | Path,
    stem: str,
    *,
    ckpt: str | Path | None,
    date: str | None = None,
    overwrite: bool = False,
    extra_provenance: dict[str, Any] | None = None,
    **savefig_kwargs: Any,
) -> Path:
    """把 matplotlib figure 存成帶身分後綴的 PNG，**並把 provenance 內嵌進檔案本身**。

    `reserve_evidence_path` 只讓檔名帶身分，檔案一旦被複製到簡報／論文目錄並改名，
    身分即完全消失。PNG 格式本身支援 tEXt 文字區塊，matplotlib 的
    ``savefig(..., metadata={...})`` 會原樣寫入，因此 provenance 可以跟著像素走。

    讀回方式（唯讀，不需本模組）::

        from PIL import Image
        Image.open(p).text["provenance"]      # → JSON 字串

    Args:
        fig: matplotlib ``Figure``（或任何有 ``savefig`` 的物件）。
        out_dir: 輸出目錄。
        stem: 檔名主體。
        ckpt: 產生此圖的 checkpoint 路徑；純 GT／純資料圖明示 None。
        date: `YYYYMMDD`；None 表示今日。
        overwrite: 已存在時是否允許覆寫。
        extra_provenance: 併入 ``_provenance.extra`` 的額外欄位。
        **savefig_kwargs: 轉給 ``fig.savefig``（dpi / bbox_inches …）。
            不接受 ``metadata``：那正是本函式要寫的欄位。

    Returns:
        實際寫出的 PNG 路徑。

    Raises:
        TypeError: 呼叫端自己傳了 ``metadata``（會蓋掉 provenance）。
        FileExistsError: 目標已存在且未允許覆寫。
    """
    if "metadata" in savefig_kwargs:
        raise TypeError(
            "write_png_evidence 不接受 metadata 參數：provenance 就是寫在該欄位。"
            "額外欄位請用 extra_provenance。"
        )
    path = evidence_path(out_dir, stem, ckpt=ckpt, ext=".png", date=date,
                         check_absent=False)
    overwrite_record = _assert_absent(path, overwrite=overwrite)
    prov = build_provenance(
        ckpt=ckpt, date=date, extra=extra_provenance,
        overwrite_record=overwrite_record,
    )
    # PNG tEXt 只吃字串；key 用 "provenance"，另存一份人眼可讀的短摘要。
    meta = {
        "provenance": json.dumps(prov, ensure_ascii=False, default=str),
        "Software": "lbmnca/evidence_io",
        "Description": (
            f"stem={stem} ckpt_sha8={prov['ckpt_sha8']} date={prov['date_tag']}"
        ),
    }
    fig.savefig(path, metadata=meta, **savefig_kwargs)
    return path


def mp4_metadata_args(
    *,
    ckpt: str | Path | None,
    date: str | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> list[str]:
    """回傳可直接插進 ffmpeg 命令列的 ``-metadata`` 參數（MP4 provenance 內嵌）。

    用法::

        args = mp4_metadata_args(ckpt=ckpt)
        subprocess.run(["ffmpeg", "-i", src, *args, str(out_path)], check=True)

    MP4 的 ``comment`` 欄位長度沒有硬性上限但不宜過長，故只放精簡欄位
    （完整 provenance 仍應同時寫一份同 stem 的 JSON 證據）。

    Args:
        ckpt: 產生此影片的 checkpoint 路徑，或明示 None。
        date: `YYYYMMDD`；None 表示今日。
        extra_provenance: 額外欄位（會併入 comment 的 JSON）。

    Returns:
        ``["-metadata", "comment=...", "-metadata", "encoding_tool=..."]``。
    """
    prov = build_provenance(ckpt=ckpt, date=date, extra=extra_provenance)
    compact = {
        "schema": prov["schema"],
        "ckpt": prov["ckpt"],
        "ckpt_sha256": prov["ckpt_sha256"],
        "date_tag": prov["date_tag"],
        "written_at": prov["written_at"],
        "script": prov["script"],
        "git": prov["git"],
        "extra": prov.get("extra"),
    }
    return [
        "-metadata",
        "comment=" + json.dumps(compact, ensure_ascii=False, default=str),
        "-metadata",
        "encoding_tool=lbmnca/evidence_io",
    ]


# ──────────────────────────────────────────────────────────────────────────
# 讀取端解析（向後相容）
# ──────────────────────────────────────────────────────────────────────────
def list_evidence(out_dir: str | Path, stem: str, ext: str = ".json") -> list[Path]:
    """列出 `stem` 的所有帶身分後綴證據檔，依 (日期, mtime) 由新到舊排序。"""
    d = Path(out_dir)
    if not d.is_dir():
        return []
    hits: list[tuple[str, float, Path]] = []
    for p in d.glob(f"{stem}{SEP}*{ext}"):
        ident = parse_identity(p.name, stem, ext)
        if ident is None:
            continue
        hits.append((ident[1], p.stat().st_mtime, p))
    hits.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _, _, p in hits]


def parse_identity(name: str, stem: str, ext: str) -> tuple[str, str] | None:
    """從檔名解出 ``(ckpt_sha8, date)``；不符身分格式回 None。

    Args:
        name: 檔名（含副檔名），例如 ``em2__0a0dc0e7_20260728.json``。
        stem: 檔名主體。
        ext: 副檔名（含點）。
    """
    prefix = f"{stem}{SEP}"
    if not name.startswith(prefix) or not name.endswith(ext):
        return None
    m = _SUFFIX_RE.match(name[len(prefix):-len(ext)])
    if m is None:
        return None
    return m.group("sha8"), m.group("date")


def _warn_if_ambiguous(hits: list[Path], *, ext: str) -> None:
    """同一天有多顆不同 ckpt 寫過同一 stem 時，出聲列出所有候選。

    若直接依 (date, mtime) 取第一個回傳，「今天兩顆 ckpt 都跑過同一份證據」會被靜默
    化約成一個檔，讀取端因此把數字歸給 mtime 較新的那顆 ckpt。
    """
    top = hits[0]
    stem = top.name.split(SEP)[0]
    ident = parse_identity(top.name, stem, ext)
    if ident is None:
        return
    date = ident[1]
    same_day = [p for p in hits
                if (i := parse_identity(p.name, stem, ext)) is not None and i[1] == date]
    shas = {parse_identity(p.name, stem, ext)[0] for p in same_day}
    if len(shas) <= 1:
        return
    print(
        f"[evidence_io] [WARN] 同一天（{date}）有 {len(shas)} 顆不同 ckpt 寫過 "
        f"'{stem}{ext}'，resolve_latest 只能依 mtime 取其一：\n"
        + "\n".join(f"[evidence_io]     {'-> ' if p == top else '   '}{p}" for p in same_day)
        + "\n[evidence_io]   讀取端請改用 list_evidence() 明示挑選你要的 ckpt_sha8。",
        file=sys.stderr,
    )


def resolve_latest(
    out_dir: str | Path,
    stem: str,
    ext: str = ".json",
    *,
    allow_legacy: bool = True,
) -> Path:
    """讀取端解析：優先取最新的帶身分後綴檔；沒有才退回舊的無後綴同名檔。

    退回 legacy 時會在 stderr 明示警告（不是靜默 fallback）。

    Args:
        out_dir: 目錄。
        stem: 檔名主體。
        ext: 副檔名（含點）。
        allow_legacy: 是否允許退回舊的 `{stem}{ext}`。設 False 時找不到新格式即報錯。

    Returns:
        存在的檔案路徑。

    Raises:
        FileNotFoundError: 新舊格式都找不到。
    """
    hits = list_evidence(out_dir, stem, ext)
    if hits:
        _warn_if_ambiguous(hits, ext=ext)
        return hits[0]
    legacy = Path(out_dir) / f"{stem}{ext}"
    # 同一 stem 可能在**別的副檔名**下已有新格式檔（例如 .json 已重測、要的 .png 還沒
    # 重畫）。訊息若只說「無身分 legacy」／「找不到」，讀取端會誤以為整個 stem 從未
    # 重測過，於是安心地拿舊世代檔繼續用。
    other_ext_hits = sorted(
        p for p in Path(out_dir).glob(f"{stem}{SEP}*")
        if p.is_file() and p.suffix != ext and parse_identity(p.name, stem, p.suffix)
    ) if Path(out_dir).is_dir() else []
    if allow_legacy and legacy.is_file():
        msg = (
            f"[evidence_io] [WARN] 使用無身分後綴的 legacy 證據檔：{legacy}\n"
            f"[evidence_io]   此檔無 ckpt/日期身分，來源世代不可驗證。"
            f"重跑產生端即可取得帶身分的新檔。"
        )
        if other_ext_hits:
            msg += (
                f"\n[evidence_io]   注意：本 stem **已有新格式產物**，只是副檔名不是 "
                f"{ext}（{', '.join(p.name for p in other_ext_hits[:5])}）。\n"
                f"[evidence_io]   即「{stem} 已重測過，但你要的 {ext} 停留在舊世代」"
                "—— 此時混用等於把新舊兩代數字放進同一張圖／表。"
            )
        print(msg, file=sys.stderr)
        return legacy
    if other_ext_hits:
        raise FileNotFoundError(
            f"找不到證據檔：{legacy}（含帶身分後綴的 {stem}{SEP}*{ext}）。\n"
            f"  → 但本 stem **已有新格式產物**，只是副檔名不是 {ext}：\n"
            + "\n".join(f"      {p.name}" for p in other_ext_hits[:10])
            + f"\n  → 即「{stem} 已重測過，但你要的 {ext} 還沒重新產生」。"
            "請重跑產生該副檔名的那一支腳本，不要改讀舊世代檔混用。"
        )
    raise FileNotFoundError(
        f"找不到證據檔：{Path(out_dir) / (stem + ext)}（含帶身分後綴的 "
        f"{stem}{SEP}*{ext}）。請先執行對應的產生端腳本。"
    )


def add_overwrite_arg(parser) -> None:
    """替 argparse parser 加上標準的 `--allow-overwrite` 旗標。

    與環境變數 :data:`ALLOW_OVERWRITE_ENV` 為同一個逃生口的兩種觸發方式，
    優先序見 module docstring「逃生口」一節：本旗標（顯式 `overwrite=True`）
    優先於環境變數。

    Args:
        parser: `argparse.ArgumentParser` 實例。
    """
    parser.add_argument(
        "--allow-overwrite", action="store_true",
        help=(
            "允許覆寫同 ckpt 同日期的既有證據檔（預設不允許，已存在即報錯）。"
            f"無 CLI 的呼叫端可改用環境變數 {ALLOW_OVERWRITE_ENV}=1。"
        ),
    )
