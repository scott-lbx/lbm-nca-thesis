"""early-stop 的分派邏輯（純函式，與 model／loader／GPU 無關）。

★為何獨立成模組：``TESTrainer`` 與 ``TrainerBase``
（``MeanTrainer``／``TurbTrainer``——``train_mean.py``／``train_turb.py`` 實際
實例化的類——所走的路徑）若各自維護一份分派邏輯與各自的 monitor 解析器，容易在
其中一份修正後、另一份仍留著同構缺陷（回傳型別 int vs float、非數值輸入是否
``ValueError``、``vg_*`` 鍵是否支援、訊息內容等）而不自覺。

分派邏輯本身若耦合在 trainer 內就難以單獨測試（要測它就得實例化 trainer，需要
GPU + dataset），容易讓測試檔自行重寫一份分支複製品，與生產碼的連結只剩字串比對，
使得生產碼路徑上的修正不會被測試覆蓋到。

本模組把分派抽成不需 GPU／dataset 的純函式，兩個 trainer 共用同一支，
monitor 解析一律委派
:func:`~lbmnca.nca.tes.training.schedule.select_early_stop_monitor`（單一實作）。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .schedule import select_early_stop_monitor

__all__ = ["dispatch_early_stop"]


def dispatch_early_stop(
    *,
    es_cfg: Any,
    step: int,
    cold_metrics: dict[str, float],
    warm_metrics: dict[str, float],
    vg_metrics: Optional[dict[str, float]] = None,
    early_stop: Any,
    state_dict_fn: Callable[[], dict],
    write_jsonl: Callable[..., None],
) -> bool:
    """執行一次 early-stop 判定，回傳是否應該停止訓練。

    語義（兩個 trainer 共用，不得再分岔）：

    1. ``es_cfg.enabled`` 為 False → **仍寫一筆** ``event=early_stop`` 的稽核紀錄
       （明示 ``enabled=False``）後回傳 ``False``。稽核軌跡不可隨 enabled 消失
       ——它是唯一能事後證明「早停有沒有被啟用／有沒有作用」的紀錄。
    2. 啟用 → monitor **必須解得出**，解不出即 ``KeyError`` 並列出可用 key。
       原始缺陷是解析器回 ``None``、呼叫端 ``if x is not None`` 靜默跳過整段，
       使 ``monitor='total'``（dataclass 預設，mean 線根本沒有這個 key）
       這種組態錯誤永遠不可見。

    Args:
        es_cfg: ``TESEarlyStopConfig``（需有 ``enabled``／``monitor``／``restore_best``）。
        step: 當前訓練步。
        cold_metrics: cold-start split 的 val 指標。
        warm_metrics: warm-start split 的 val 指標。
        vg_metrics: VIS-GATE 指標；``TrainerBase`` 路徑無視覺門控，傳 ``None``。
        early_stop: ``TESEarlyStopping`` 實例。
        state_dict_fn: 取得 model state_dict 的 callable（延後求值，避免無謂複製）。
        write_jsonl: 寫 metrics.json 的 callable，簽章為 ``(record, force_flush=...)``。

    Returns:
        ``True`` 表示應停止訓練（呼叫端負責 restore_best 與存 ckpt）。

    Raises:
        KeyError: 啟用早停但 ``monitor`` 在三組 metrics 中皆不存在。
    """
    monitor = es_cfg.monitor
    if not es_cfg.enabled:
        write_jsonl(
            {"event": "early_stop", "step": step, "enabled": False, "monitor": monitor},
            force_flush=True,
        )
        return False

    primary = select_early_stop_monitor(
        monitor, cold_metrics, warm_metrics, vg_metrics or {}
    )
    stop = early_stop.update(primary, step, state_dict_fn())
    write_jsonl(early_stop.as_jsonl_record(step), force_flush=True)
    return bool(stop)
