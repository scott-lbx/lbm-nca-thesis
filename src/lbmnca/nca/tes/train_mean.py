"""LC decompose mean module 訓練入口（薄入口）。

使用方式：
    python -m lbmnca.nca.tes.train_mean --config configs/nca/main.yaml
    python -m lbmnca.nca.tes.train_mean --config configs/nca/ablations/base_modern.yaml --finetune checkpoints/base_modern.pth

★定位：
    本檔為 **純新增薄入口**，與既有 ``train.py`` 並存、互不影響。流程：
        load_mean_config → build model（factory）→ build dataloaders
        → MeanTrainer(...) → .train()

★安全鐵則：本檔不接入 ``train.py``、不改動 ``trainer.py`` / ``config.py`` /
    ``models/`` / ``loss/terms/`` / ``data/``。

★橋接：
    ``build_mean_training(config_path, finetune)`` =
      (1) ``load_mean_config`` 驗證 split-config（乾淨介面 + 早期錯誤）；
      (2) ``load_tes_config`` 載 legacy ``TESTrainConfig``（同一 yaml，proven build 路徑）；
      (3) 斷言 ``mean_module_enabled``（mean yaml 必有）；
      (4) ``train.build_tes_trainer`` 共用 build flow 建 backing ``TESTrainer``；
      (5) 包成 ``MeanTrainer``（委派 backing 的已驗證 mean 子元件）。
    過渡期：build 仍走 legacy path（proven）；split-config 用於驗證 + 乾淨介面。
    ★不改 trainer.py / config legacy / models / loss / data。
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.rcParams["axes.unicode_minus"] = False

from .config import load_mean_config, load_tes_config
from .train import build_tes_trainer
from .training import MeanTrainer

logger = logging.getLogger(__name__)


def build_mean_training(
    config_path: str,
    finetune: str | None,
    *,
    create_run_dir: bool = True,
) -> MeanTrainer:
    """從 mean YAML 組裝 mean module 訓練（split-config 驗證 + legacy proven build + MeanTrainer）。

    Args:
        config_path: mean YAML 設定檔路徑。
        finetune: warm restart checkpoint 路徑；None 表示從頭訓練。
        create_run_dir: 是否建立帶 timestamp 的 run 目錄並接上檔案 logging。訓練一律
            為 True（預設，行為與過去完全相同）。純推論呼叫端（製圖 / 評估 harness）
            須傳 False，否則每跑一次就在 ``runs/`` 生一個沒有 ckpt 的空 run 目錄。

    Returns:
        已組裝完成、可呼叫 ``.train()`` 的 ``MeanTrainer``。

    Raises:
        ValueError: yaml 非 mean module（``mean_module_enabled`` 非 True）。
    """
    cfg_mean = load_mean_config(config_path)   # 驗證 split-config（乾淨介面 + 早期錯誤）
    logger.info("MeanTrainConfig 驗證通過：arch.state_dim=%d cond_dim=%d",
                cfg_mean.arch.state_dim, cfg_mean.arch.cond_dim)
    legacy_cfg = load_tes_config(config_path)  # proven legacy build 路徑（同一 yaml）
    if not legacy_cfg.mean_module_enabled:
        raise ValueError(
            f"train_mean 要求 mean module yaml（mean_module_enabled=True）；"
            f"{config_path} 得到 mean_module_enabled={legacy_cfg.mean_module_enabled}"
        )
    backing = build_tes_trainer(
        legacy_cfg, finetune=finetune, create_run_dir=create_run_dir
    )
    return MeanTrainer(backing=backing)


def main() -> None:
    """mean module 訓練主函式（薄入口：parse args → build → train）。"""
    parser = argparse.ArgumentParser(description="LC decompose Phase 0 mean module 訓練")
    parser.add_argument("--config", required=True, help="mean YAML 設定檔路徑")
    parser.add_argument(
        "--finetune",
        default=None,
        help="warm restart：只載入 model weights，optimizer/scheduler 重置",
    )
    args: argparse.Namespace = parser.parse_args()

    logger.info("載入 mean config：%s", args.config)
    trainer = build_mean_training(args.config, args.finetune)
    trainer.train()


if __name__ == "__main__":
    main()
