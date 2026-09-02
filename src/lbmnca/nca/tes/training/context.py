"""TESTrainer 共享狀態容器模組。

本模組定義 TrainerContext dataclass，作為所有 TESTrainer 子元件之間的統一
共享介面，取代子元件直接存取 trainer self 的方式，提升模組解耦性與可測試性。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainerContext:
    """TESTrainer 共享狀態容器，傳遞給所有子元件取代直接存取 trainer self。

    所有 Trainer 子元件（ConditionBuilder、DataRenormalizer、ICOps 等）
    均透過此 dataclass 存取模型、設定、正規化器與可變訓練狀態。
    子元件更新 ctx.step 等可變狀態是允許的。

    Args:
        model: TES-NCA 模型實例（nn.Module）。
        loss_fn: 語意損失模組（TESSemanticLoss）。
        optimizer: AdamW 最佳化器。
        scheduler: LambdaLR 學習率排程器。
        scaler: AMP 用 GradScaler；非 AMP 模式時為 None。
        early_stopping: TESEarlyStopping 早停監控器。
        train_loader: 訓練 DataLoader（cold + warm 合併）。
        val_cold_loader: cold 驗證 DataLoader；無 cold cases 時為 None。
        val_warm_loader: warm 驗證 DataLoader；無 warm cases 時為 None。
        normalizer: TESCondNormalizer 條件正規化器。
        field_normalizer: FieldNormalizer；尚未載入時為 None。
        source_field_normalizers: 多 zarr source 各自 FieldNormalizer 的列表。
        pool: TESRollingPool；pool.mode != 'rolling' 時為 None。
        cfg: TESTrainConfig 完整設定。
        device: 訓練裝置（cuda / cpu）。
        output_dir: 訓練輸出目錄路徑。
        sdf_threshold: SDF_norm 流體判定門檻值（tanh_lchar 下為 0.0）。
        step: 當前訓練步數（可被子元件更新）。
        metrics_file: metrics.json 已開啟的 IO 物件；訓練前為 None。
        logger: 模組 logger；預設 None 由子元件各自取得。
        executor: 背景執行緒池（ThreadPoolExecutor）；停用時為 None。
    """

    # 核心元件
    model: nn.Module
    loss_fn: Any                    # TESSemanticLoss
    optimizer: torch.optim.Optimizer
    scheduler: Any                  # LambdaLR
    scaler: Any | None              # GradScaler（AMP 用）
    early_stopping: Any             # TESEarlyStopping
    # 資料
    train_loader: DataLoader
    val_cold_loader: DataLoader
    val_warm_loader: DataLoader
    normalizer: Any                 # TESCondNormalizer
    field_normalizer: Any | None    # FieldNormalizer
    source_field_normalizers: dict[str, Any]
    pool: Any | None                # TESRollingPool
    # 設定
    cfg: Any                        # TESTrainConfig
    device: torch.device
    output_dir: Path
    sdf_threshold: float
    # 可變狀態
    step: int = field(default=0)
    metrics_file: Any = field(default=None)
    logger: logging.Logger = field(default=None)
    executor: Any = field(default=None)  # ThreadPoolExecutor
