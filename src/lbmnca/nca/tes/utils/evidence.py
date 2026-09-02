"""證據寫入的單一入口（DRY）。

底層實作在同目錄的 :mod:`evidence_io`；本模組再加一層訓練期專用的
:func:`write_run_evidence`，並把整組 API 集中 re-export，其餘模組一律
`from ..utils.evidence import write_json_evidence` 委派，不直接 import 底層模組。

## 兩種 evidence 語義

| 類型 | 函式 | 檔名 | 覆寫 |
| --- | --- | --- | --- |
| 離線 eval（CLI，ckpt 已落地） | `write_json_evidence` | `{stem}__{ckpt_sha8}_{date}.json` | 預設禁止 |
| 訓練期滾動指標（run 目錄內） | `write_run_evidence` | `{stem}__nockpt_{date}.json` | 允許 |

訓練期的 `eval_summary_latest` / `vis_gate_detailed` 走 `write_run_evidence`：

* **ckpt=None 是實質判斷不是偷懶**——寫入時該 step 的 ckpt 尚未（或不會）落地，
  沒有可雜湊的檔案內容；此類 evidence 的世代身分由 **run 目錄 + step** 承載，
  兩者都寫進 `_provenance.extra`。
* **overwrite=True 是設計意圖**——這是 run 內每次 eval 都重寫的滾動指標檔，
  非跨世代 canonical 證據；若改成禁覆寫，第二次 eval 會直接中斷訓練。
  跨世代覆寫風險由 run 目錄隔離承擔。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .evidence_io import (
    add_overwrite_arg,
    build_provenance,
    ckpt_sha8,
    evidence_path,
    list_evidence,
    reserve_evidence_path,
    resolve_latest,
    write_json_evidence,
)

__all__ = [
    "add_overwrite_arg",
    "build_provenance",
    "ckpt_sha8",
    "evidence_path",
    "list_evidence",
    "reserve_evidence_path",
    "resolve_latest",
    "write_json_evidence",
    "write_run_evidence",
]


def write_run_evidence(
    out_dir: str | Path,
    stem: str,
    payload: dict[str, Any],
    *,
    run_dir: str | Path,
    step: int | None,
    extra_provenance: dict[str, Any] | None = None,
) -> Path:
    """寫訓練期「run 內滾動」evidence（帶日期後綴 + `_provenance`，允許覆寫）。

    與 `write_json_evidence` 的差異見模組 docstring：此類檔案在單一 run 內每次
    eval 都重寫，ckpt 尚未落地故無法取內容雜湊，身分由 run 目錄 + step 承擔。

    Args:
        out_dir: 輸出目錄（通常為 `{run_dir}/eval_output`）。
        stem: 檔名主體（不含副檔名與身分後綴）。
        payload: 要寫的 dict。
        run_dir: 產生此 evidence 的 run 目錄（寫入 `_provenance.extra.run_dir`）。
        step: 訓練 step（寫入 `_provenance.extra.step`）；未知傳 None。
        extra_provenance: 併入 `_provenance.extra` 的額外欄位。

    Returns:
        實際寫出的路徑。
    """
    extra: dict[str, Any] = {
        "evidence_kind": "training_rolling",
        "run_dir": str(run_dir),
        "step": step,
    }
    if extra_provenance is not None:
        extra.update(extra_provenance)
    return write_json_evidence(
        out_dir, stem, payload,
        ckpt=None, overwrite=True, extra_provenance=extra, default=str,
    )
