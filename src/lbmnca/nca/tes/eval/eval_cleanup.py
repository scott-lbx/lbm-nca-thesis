"""eval 輸出目錄自動清理：保留最新的 N 個 eval_ckpt_* 影像目錄，刪除較舊的。

JSON 指標檔案絕不刪除，僅清理影像 checkpoint 目錄。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_eval_ckpts(eval_output_dir: Path, keep_n: int = 2) -> None:
    """刪除 eval_ckpt_* 影像目錄中最舊的，僅保留最新的 keep_n 個。

    目錄名稱須符合 eval_ckpt_XXXXXXX 格式（零補齊步驟編號）。
    由於步驟編號一致零補齊，字典序排序即等同於時間序排序。

    Args:
        eval_output_dir: runs/{exp}/eval_output/ 的路徑。
        keep_n: 保留最新 checkpoint 目錄的數量。
    """
    if not eval_output_dir.exists():
        return

    ckpt_dirs = sorted(
        [
            d
            for d in eval_output_dir.iterdir()
            if d.is_dir() and d.name.startswith("eval_ckpt_")
        ],
        key=lambda d: d.name,
    )

    to_delete: list[Path] = ckpt_dirs[:-keep_n] if len(ckpt_dirs) > keep_n else []
    for d in to_delete:
        try:
            shutil.rmtree(d)
            logger.info("Eval cleanup: removed %s", d.name)
        except OSError as e:
            logger.warning("Failed to remove eval dir %s: %s", d, e)

    if to_delete:
        kept = min(len(ckpt_dirs), keep_n)
        logger.info(
            "Eval cleanup: deleted %d old dirs, kept %d (keep_n=%d)",
            len(to_delete),
            kept,
            keep_n,
        )
