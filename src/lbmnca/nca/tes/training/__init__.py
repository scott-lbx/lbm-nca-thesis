"""TES-NCA 訓練框架（組合模式：TESTrainer 持有子元件實例）。

公開 API：
    TESTrainer        — 主訓練器（持有所有子元件；現行 train.py 入口）
    TrainerBase       — module-agnostic 主迴圈骨架（additive 拆分 §3；abstract hook）
    MeanTrainer       — TrainerBase 子類：mean module（委派 TESTrainer 子元件）
    TrainerContext    — 訓練狀態容器（step、epoch、config 等）
    TESEarlyStopping  — plateau-based early stopping
    BaseStep          — 訓練步驟基底類別
    MeanStep          — 平均流場訓練步驟
    Checkpointer      — Checkpoint 儲存與讀取（save 委派 save_checkpoint）
    save_checkpoint   — ★存 ckpt 的單一實作（含落盤後 identity 斷言）
    CheckpointIdentityError — 身分建立/落盤驗證失敗（預設中止）
    ConditionBuilder  — 條件向量組裝
    DataRenormalizer  — 資料反正規化工具
    ICOps             — 初始條件操作
    Regularizer       — 正則化計算
    schedule          — 純函式 curriculum 工具集
        sample_k          — 取樣展開步數 k
        get_val_k         — 取得驗證 k
        get_loss_version  — 取得 loss 版本
        get_tf_ratio      — 取得 teacher forcing 比例
        get_p_teacher     — 取得 teacher forcing 機率
"""
from __future__ import annotations

from . import schedule
from .checkpoint import (
    ALLOW_UNIDENTIFIED_CKPT_ENV,
    CheckpointIdentityError,
    Checkpointer,
    save_checkpoint,
)
from .context import TrainerContext
from .data_prep import ConditionBuilder, DataRenormalizer, ICOps
from .early_stopping import TESEarlyStopping
from .regularization import Regularizer
from .schedule import (
    get_loss_version,
    get_p_teacher,
    get_tf_ratio,
    get_val_k,
    sample_k,
)
from .steps import BaseStep, MeanStep
from .trainer import TESTrainer
from .trainer_base import TrainerBase
from .mean_trainer import MeanTrainer
from .validation import aggregate_val_metrics

__all__ = [
    "TESTrainer",
    "TrainerBase",
    "MeanTrainer",
    "TrainerContext",
    "TESEarlyStopping",
    "BaseStep",
    "MeanStep",
    "aggregate_val_metrics",
    "Checkpointer",
    "save_checkpoint",
    "CheckpointIdentityError",
    "ALLOW_UNIDENTIFIED_CKPT_ENV",
    "ConditionBuilder",
    "DataRenormalizer",
    "ICOps",
    "Regularizer",
    "schedule",
    "sample_k",
    "get_val_k",
    "get_loss_version",
    "get_tf_ratio",
    "get_p_teacher",
]
