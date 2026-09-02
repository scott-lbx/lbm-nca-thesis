"""訓練期具名 RNG 串流與 determinism 記錄。

背景
----
訓練熱路徑上**用標準庫 ``random`` 的**離散抽樣共用同一條全域序列——curriculum
k 抽樣、RollingPool 使用機率、rollout ``t_start``、mean 額外步數、chain 分支
機率、視覺取樣都串在同一條流上。任一路徑的抽樣次數改變（開關某個功能、
config 微調、走到不同分支）就會整條位移其餘路徑的結果，使「同 config 同
seed」的兩次訓練無法逐步對齊，也讓 ablation 的差異混入純粹的 RNG 位移。

本模組提供以**固定 offset** 從全域 seed 派生的具名 ``random.Random`` 實例，
供**已遷移的呼叫點**取得彼此獨立的序列。

★涵蓋範圍（**這一節是限定語，不是窮舉保證**）
----------------------------------------------
已遷移的呼叫點集合、以及呼叫端用到的方法集合，由
``tests/nca/tes/test_r7_countable_claims.py`` 的
``test_rng_streams_consumer_inventory`` 與
``test_rng_stream_chained_methods_are_equivalence_safe`` 維護（斷言即清單）。

**本模組不涵蓋的抽樣（三類；它們各自掛在某條共用全域序列上，但不是同一條，
本模組的具名串流一律管不到）**：

1. **走 torch 全域 RNG 的離散抽樣**（本模組完全管不到，它們共用的是
   ``torch.manual_seed`` 那條序列）。
2. **dataset 端的抽樣**：``data/tes_dataset.py`` 內的裸全域 ``random``
   **刻意不遷移**——``num_workers>0`` 時 worker 以 spawn 重新 import 本模組，
   ``_ENABLED`` 會回到初始的 ``False``，遷移後只會在單行程與多行程之間產生不一致
   的行為。dataset 端的可重現性由 ``build_tes_dataloader(seed=...)`` 的專屬
   ``torch.Generator`` + ``_seed_worker`` 負責，是另一條機制。

因此 ``rng_stream_isolation=True`` 時得到的是**部分隔離**，不可讀成「全訓練期
完全隔離」，也不可據此宣稱「訓練期抽樣全部具名化」。

回歸測試 ``tests/nca/tes/test_steps_rng_streams.py`` 的守備範圍同樣有限，必須照字面
理解：它只以 ast 掃 ``src/lbmnca/nca/tes/training/steps/*.py``，鎖住「除允許清單外，不得出現
``import random`` 後的 ``random.<attr>``，也不得出現 ``from random import <函式>``
後直接呼叫該函式」。它**不掃** ``steps/`` 以外的目錄，也**不管** torch 全域 RNG。

★預設停用（零行為變更）
------------------------
模組初始狀態 ``_ENABLED = False``，此時 :func:`stream` / :func:`fresh_stream`
直接回傳標準庫 ``random`` **模組本身**。呼叫端用到的方法在模組與
``random.Random`` 實例上同名同語義（模組層函式即一個隱藏 ``Random`` 實例的
繫結方法），故停用時抽樣序列與現行程式**逐一相同**。

只有 :func:`configure` 被以 ``enabled=True`` 呼叫（由
``cfg.rng_stream_isolation`` 驅動）之後才切換成隔離串流。該切換**會改變
抽樣序列**，因此屬於「需重新訓練才能定案」的變更，不可在既有
run 的續訓中途開啟後仍宣稱與原 run 可比。

用法
----
    from ..rng_streams import stream

    k = stream("schedule").randint(k_min, k_max)

Attributes:
    STREAM_OFFSETS: 具名串流 → 固定 seed offset。**offset 一旦寫定即不可更動**
        （更動等同換 seed，會使已標記 ``rng_stream_isolation=true`` 的 run 失去
        可比性）。
"""
from __future__ import annotations

import logging
import os
import random
from types import ModuleType
from typing import Union

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "STREAM_OFFSETS",
    "DETERMINISM_MODES",
    "configure",
    "is_enabled",
    "stream",
    "fresh_stream",
    "streams_summary",
    "snapshot_runtime_flags",
    "apply_determinism_mode",
]

# ── 具名串流固定 offset（★寫定後不可更動）────────────────────────────
STREAM_OFFSETS: dict[str, int] = {
    "schedule": 10_000,      # curriculum k / phase 抽樣（schedule.sample_k）
    "rollout": 20_000,       # rollout t_start / 額外步數
    "pool": 30_000,          # RollingPool 使用機率、slot 抽樣
    "damage": 40_000,        # pool damage 觸發機率
    "chain": 50_000,         # multi-edge pushforward 分支機率
    "vis": 60_000,           # 訓練期視覺取樣（t0 / n）
    # 70_000 原為 "turb" 串流，turb 已移出範疇。偏移量刻意留空
    # 不回收——重用它等於讓新用途拿到與舊 run 相同的抽樣序列。
}

DETERMINISM_MODES: tuple[str, ...] = ("off", "warn_only", "strict")

_ENABLED: bool = False
_SEED: int = 0
_STREAMS: dict[str, random.Random] = {}

# ``random`` 模組與 ``random.Random`` 實例的公開介面相同，可互換使用。
RngLike = Union[random.Random, ModuleType]


def configure(seed: int, enabled: bool) -> None:
    """設定具名 RNG 串流（由訓練入口在 ``seed_everything()`` 之後呼叫一次）。

    Args:
        seed: 全域 seed（``cfg.seed``）；各串流種子為 ``seed + STREAM_OFFSETS[name]``。
        enabled: True 才啟用隔離串流；False（預設）維持共用全域 ``random`` 序列。
    """
    global _ENABLED, _SEED
    _ENABLED = bool(enabled)
    _SEED = int(seed)
    _STREAMS.clear()
    if _ENABLED:
        logger.warning(
            "rng_stream_isolation=True：訓練抽樣改用具名獨立 RNG 串流（seed=%d，"
            "串流 %s）。★抽樣序列與 isolation=False 的既有 run 不可比。",
            _SEED,
            sorted(STREAM_OFFSETS),
        )
    else:
        logger.info("rng_stream_isolation=False：沿用共用全域 random 序列（現行行為）")


def is_enabled() -> bool:
    """回傳目前是否啟用具名串流隔離。

    Returns:
        True 表示 :func:`stream` 回傳獨立 ``random.Random``；False 表示回傳全域模組。
    """
    return _ENABLED


def stream(name: str) -> RngLike:
    """取得具名 RNG（停用時回傳標準庫 ``random`` 模組本身）。

    Args:
        name: 串流名稱，必須是 :data:`STREAM_OFFSETS` 的鍵。

    Returns:
        ``random.Random`` 實例（啟用時，跨呼叫持續推進）或 ``random`` 模組（停用時）。

    Raises:
        KeyError: 串流名稱未在 :data:`STREAM_OFFSETS` 登記（禁止臨時造名）。
    """
    if name not in STREAM_OFFSETS:
        raise KeyError(
            f"未登記的 RNG 串流名稱：{name!r}；合法值={sorted(STREAM_OFFSETS)}。"
            "新增用途請先在 rng_streams.STREAM_OFFSETS 登記固定 offset。"
        )
    if not _ENABLED:
        return random
    if name not in _STREAMS:
        _STREAMS[name] = random.Random(_SEED + STREAM_OFFSETS[name])
    return _STREAMS[name]


def fresh_stream(name: str) -> RngLike:
    """取得**每次重建**的具名 RNG（停用時回傳標準庫 ``random`` 模組本身）。

    用於希望「跨 checkpoint 看到同一組抽樣」的場景（如訓練期視覺取樣：每次
    ``save_val_fields`` 重建，使不同 step 的視覺圖採同一組 ``(t0, n)`` 便於比對）。

    Args:
        name: 串流名稱，必須是 :data:`STREAM_OFFSETS` 的鍵。

    Returns:
        全新的 ``random.Random``（啟用時，狀態不跨呼叫延續）或 ``random`` 模組（停用時）。

    Raises:
        KeyError: 串流名稱未在 :data:`STREAM_OFFSETS` 登記。
    """
    if name not in STREAM_OFFSETS:
        raise KeyError(
            f"未登記的 RNG 串流名稱：{name!r}；合法值={sorted(STREAM_OFFSETS)}。"
        )
    if not _ENABLED:
        return random
    return random.Random(_SEED + STREAM_OFFSETS[name])


def streams_summary() -> dict[str, object]:
    """回傳串流設定摘要（供 log / 證據 JSON 記錄）。

    Returns:
        含 ``enabled`` / ``seed`` / ``offsets`` 三鍵的 dict。
    """
    return {
        "enabled": _ENABLED,
        "seed": _SEED,
        "offsets": dict(STREAM_OFFSETS),
    }


# ── determinism ────────────────────────────────────────────────────────
def snapshot_runtime_flags() -> dict[str, object]:
    """擷取影響 kernel 決定性的執行期旗標現值（唯讀，不改任何設定）。

    Returns:
        含 torch 版本、cuda 版本、cudnn deterministic/benchmark、TF32 開關、
        ``CUBLAS_WORKSPACE_CONFIG`` 與 ``PYTHONHASHSEED`` 環境變數的 dict。
        ``PYTHONHASHSEED`` 若為 None 表示直譯器啟動時未設定 → hash randomization
        不受控（見 :func:`lbmnca.nca.tes.train.seed_everything` docstring）。
    """
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "env_cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "env_pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def apply_determinism_mode(mode: str, cudnn_benchmark: bool = False) -> dict[str, object]:
    """依 ``cfg.determinism_mode`` 套用（或僅記錄）決定性設定。

    三個檔位：

    * ``"off"``（**預設**）：完全不動任何旗標、不記錄 → 與現行行為逐字相同。
    * ``"warn_only"``：**只記錄並提醒**。擷取 :func:`snapshot_runtime_flags`
      寫入 log，並警告 determinism 未被強制。★不呼叫
      ``torch.use_deterministic_algorithms``、不改 cudnn/TF32，
      因此**不改變任何 kernel 選擇**（與 torch 自身的 ``warn_only=True``
      語義不同，此處刻意只做記錄）。
    * ``"strict"``：啟用 ``torch.use_deterministic_algorithms(True)`` 與
      ``cudnn.deterministic=True`` / ``benchmark=False``。★本專案模型含
      ``grid_sample`` 等無 deterministic GPU kernel 的算子，開啟後極可能
      直接 ``RuntimeError``（見 ``train.seed_everything`` docstring）；
      此檔位僅供作者驗證用，尚未在本專案通過實測。

    ★``cudnn_benchmark``（``cfg.cudnn_benchmark``）與本檔位**正交但互斥**：
    autotune 會依實測選最快 kernel，而「最快」在 deterministic 模式下不成立，
    故 ``strict`` 一律強制關閉並在 log 明示覆寫，不讓 config 值靜默失效。

    ★效能成本：``strict`` 的 ``use_deterministic_algorithms(True)`` 會讓每步
    明顯變慢（實測約 1.5×），再關掉 cudnn autotune 又更慢一些。

    Args:
        mode: 三檔位之一。
        cudnn_benchmark: 是否啟用 cuDNN autotune。``strict`` 下強制為 False。

    Returns:
        執行期旗標快照（``"off"`` 檔位仍會套用 ``cudnn_benchmark``）。

    Raises:
        ValueError: mode 不在 :data:`DETERMINISM_MODES`。
    """
    if mode not in DETERMINISM_MODES:
        raise ValueError(
            f"determinism_mode 必須是 {DETERMINISM_MODES} 之一，得到 {mode!r}"
        )
    if mode == "strict" and cudnn_benchmark:
        logger.warning(
            "determinism_mode=strict 與 cudnn_benchmark=True 互斥："
            "autotune 選出的 kernel 不保證 deterministic，故本次強制關閉 autotune。"
        )
        cudnn_benchmark = False
    if mode != "strict":
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    if mode == "off":
        return {}

    if mode == "strict":
        # CUBLAS_WORKSPACE_CONFIG 需在 cuBLAS handle 建立前設定；此處為盡力而為，
        # 最可靠的做法仍是由啟動腳本以環境變數設定。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        logger.warning(
            "determinism_mode=strict：已啟用 use_deterministic_algorithms(True)。"
            "★本專案含 grid_sample/scatter 等無 deterministic kernel 的算子，"
            "訓練可能直接 RuntimeError。"
        )

    flags = snapshot_runtime_flags()
    logger.info("determinism_mode=%s；執行期旗標=%s", mode, flags)
    if mode == "warn_only":
        logger.warning(
            "determinism_mode=warn_only：僅記錄執行期旗標，未強制 deterministic "
            "kernel；run-to-run 數值差異仍可能存在，請以 repeat_tolerance 判讀。"
        )
    if flags["env_pythonhashseed"] is None:
        logger.warning(
            "PYTHONHASHSEED 未設定：Python hash randomization 不受控。"
            "需完全可重現時請由啟動腳本設定該環境變數（程式內設定無效）。"
        )
    return flags
