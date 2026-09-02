"""TES-NCA 訓練入口 — 載入 config → 建立資料集 → 啟動 TESTrainer。

使用方式：
    python -m lbmnca.nca.tes.train --config configs/nca/main.yaml
    python -m lbmnca.nca.tes.train --config configs/nca/main.yaml --resume checkpoints/main_v4.pth
"""
from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.rcParams["axes.unicode_minus"] = False

import torch

from .config import load_tes_config
from .data.normalizer import TESCondNormalizer
from .data.runtime_global_stats import build_runtime_global_stats, resolve_stats_paths
from .data.graph_dataset import collate_chain_batch, collate_graph_batch
from .data.graph_index import GraphIndex, is_v4_zarr
from .data.graph_split import build_chain_train_dataset, split_graph_datasets
from .data.loader import build_tes_dataloader
from .loss.semantic_loss import TESSemanticLoss
from .models.model_factory import build_tes_model
from .training import rng_streams
from .training.trainer import TESTrainer
from ..utils.FieldNormalizer import FieldNormalizer

logger = logging.getLogger(__name__)

# ── per-loader seed 偏移（僅在 cfg.loader_seed_offset=True 時生效）──
# 四個 loader 過去共用同一個 cfg.seed，導致 train_loader 與 chain_loader（皆
# shuffle=True）拿到相同的 permutation 串流與相同的 worker base_seed，兩邊
# worker 的 vflip 決策序列完全相關。偏移量寫定後不可更動（等同換 seed）。
LOADER_SEED_OFFSET_TRAIN: int = 0
LOADER_SEED_OFFSET_VAL_COLD: int = 1
LOADER_SEED_OFFSET_VAL_WARM: int = 2
LOADER_SEED_OFFSET_CHAIN: int = 3


def loader_seed(cfg: Any, offset: int) -> int:
    """回傳指定 loader 應使用的 seed。

    Args:
        cfg: legacy ``TESTrainConfig``（需含 ``seed`` 與 ``loader_seed_offset``）。
        offset: ``LOADER_SEED_OFFSET_*`` 之一。

    Returns:
        ``cfg.loader_seed_offset`` 為 True 時回傳 ``cfg.seed + offset``；
        否則回傳 ``cfg.seed``（預設，維持現行行為）。
    """
    return int(cfg.seed) + int(offset) if cfg.loader_seed_offset else int(cfg.seed)


def seed_everything(seed: int) -> None:
    """設定 python / numpy / torch / cuda 全域 RNG（可重現）。

    不啟用 ``torch.use_deterministic_algorithms``：模型含 grid_sample / scatter 等
    無 deterministic GPU kernel 的算子，且 RP 重現門檻為「近似」（GPU/cuDNN 非
    bitwise）。此函式只負責 seed，使 run-to-run 落在 tolerance 內可比。
    需要強制 deterministic kernel 時走 ``cfg.determinism_mode="strict"``
    （見 ``training/rng_streams.py::apply_determinism_mode``）。

    ★``PYTHONHASHSEED`` **不在此設定**：該環境變數只在直譯器啟動時
    被讀取，程式內 ``os.environ[...] = ...`` 是 no-op（hash randomization 早已
    固定），屬「看起來有防護、實際沒有」。需要控制 hash 隨機化時必須由**啟動
    腳本／環境變數**設定，例如：

        $env:PYTHONHASHSEED = "42"; python -m lbmnca.nca.tes.train --config ...

    未設定時 ``rng_streams.apply_determinism_mode()`` 會在 log 中明示不受控。

    Args:
        seed: 全域亂數種子值。
    """
    import random as _random

    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("seed_everything：seed=%d（python/numpy/torch/cuda）", seed)



#: 保留集（DT）的 zarr 一律放在含此片段的路徑下。訓練期讀到即中止。
_HOLDOUT_MARKERS: tuple[str, ...] = ("dataset_zarr_holdout", "DT_Stress")


def _assert_no_holdout(zarr_paths: list[str]) -> None:
    """訓練路徑上出現保留集即中止。

    DT 是論文最終實驗用的保留集，一旦在訓練或模型選擇階段被讀過，
    它報出的數字就不再是 held-out 的了——而這種污染事後無法從結果反推，
    只能在入口擋掉。物理隔離（獨立目錄）之外再加這道程式攔截，
    因為目錄可以被複製、config 可以被貼上。
    """
    bad = [p for p in zarr_paths
           if any(m in str(p) for m in _HOLDOUT_MARKERS)]
    if bad:
        raise RuntimeError(
            "訓練 config 的 zarr_paths 含保留集（DT）："
            + "、".join(bad)
            + "。DT 只在論文最終實驗使用，不得進入訓練或模型選擇階段。"
              f"（判定依據：路徑含 {_HOLDOUT_MARKERS} 任一片段）"
        )


def build_tes_trainer(
    cfg: Any,
    *,
    finetune: Optional[str] = None,
    resume: Optional[str] = None,
    tag: Optional[str] = None,
    create_run_dir: bool = True,
) -> "TESTrainer":
    """從 legacy ``TESTrainConfig`` 組裝可訓練的 ``TESTrainer``（★共用 build flow）。

    抽出原 ``main()`` 的 seed → output_dir timestamp → zarr/stats → split → dataloaders →
    normalizer → model → loss → TESTrainer → load ckpt 全流程，供 ``train.py`` /
    ``train_mean.py`` / ``train_turb.py`` 共用（DRY，避免三入口各自重寫 build flow）。

    Args:
        cfg: 已載入（且 seed/max_steps 等 override 已套用）的 legacy ``TESTrainConfig``。
        finetune: warm restart checkpoint（只載 weights，reset optim/step）。
        resume: 恢復訓練 checkpoint（載 optim/scheduler/step）。
        tag: output_dir 額外後綴（seed 標記等）。
        create_run_dir: 是否建立帶 timestamp 的 run 目錄並接上檔案 logging。訓練一律
            為 True（預設，行為與過去完全相同）。純推論呼叫端（製圖 / 評估 harness）
            須傳 False，否則每跑一次就在 ``runs/`` 生一個沒有 ckpt 的空 run 目錄。

    Returns:
        已建構並（若指定）載入 ckpt 的 ``TESTrainer``，可直接 ``.train()`` 或包成 Mean/TurbTrainer。
    """
    # ── 全域 RNG seed（可重現）─────────────────────────────────
    seed_everything(cfg.seed)
    # 具名 RNG 串流 + determinism 記錄（兩者預設皆為現行行為）
    rng_streams.configure(seed=cfg.seed, enabled=cfg.rng_stream_isolation)
    rng_streams.apply_determinism_mode(cfg.determinism_mode, cfg.cudnn_benchmark)
    if cfg.loader_seed_offset:
        logger.warning(
            "loader_seed_offset=True：四個 loader 改用 cfg.seed+{0,1,2,3} 偏移，"
            "資料順序與既有 run 不同"
        )

    # ── output_dir 永遠加 timestamp 後綴（resume 時沿用 config 原值）──
    # 推論模式不改寫 output_dir 也不建目錄；此路徑不會寫出任何檔案。
    if create_run_dir and not resume:
        _base: Path = Path(cfg.output_dir)
        _ts: str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        _suffix: str = f"_{tag}" if tag else ""
        cfg.output_dir = str(_base.parent / f"{_base.name}{_suffix}_{_ts}")
    logger.info(
        "output_dir：%s（create_run_dir=%s）", cfg.output_dir, create_run_dir
    )

    # ── 確定 zarr 路徑列表 ────────────────────────────────────────────
    zarr_paths: list[str]
    if cfg.zarr_paths:
        for zp in cfg.zarr_paths:
            if not Path(zp).exists():
                raise FileNotFoundError(f"TES Zarr 資料集不存在：{zp}")
        zarr_paths = [str(zp) for zp in cfg.zarr_paths]
    else:
        if not Path(cfg.zarr_path).exists():
            raise FileNotFoundError(
                f"TES Zarr 資料集不存在：{cfg.zarr_path}\n"
                "請先執行 LBM pipeline 生成 TES Zarr（見 docs/DATA_GENERATION.md）"
            )
        zarr_paths = [str(cfg.zarr_path)]

    _assert_no_holdout(zarr_paths)

    # ★正規化空間唯一形式：多來源 stats 合併成一組 runtime global 常數。
    #   單 zarr 的 `stats_path` 路徑已移除（見上）。
    if cfg.stats_policy != "dataset_global_runtime":
        raise ValueError(
            f"stats_policy 必須為 'dataset_global_runtime'，得到 {cfg.stats_policy!r}。"
            " 單 zarr stats 路徑不再支援。"
        )
    use_runtime_global_stats: bool = True
    source_stats_paths: list[str] = []
    # dict 值型別為 Any：global_stats 結構由 build_runtime_global_stats 決定
    global_stats: Optional[dict[str, Any]] = None
    field_normalizer: Optional[FieldNormalizer] = None
    source_field_normalizers: Optional[list[FieldNormalizer]] = None

    if use_runtime_global_stats:
        source_stats_paths = resolve_stats_paths(zarr_paths, cfg.stats_paths)
        global_stats = build_runtime_global_stats(
            source_stats_paths,
            zarr_paths=zarr_paths,
            dataset_names=cfg.dataset_names if cfg.dataset_names else None,
        )
        logger.info(
            "Runtime dataset-global stats enabled: %d source stats -> one global normalizer",
            len(source_stats_paths),
        )

    source_stats_indices: Optional[dict[str, int]] = (
        {str(zp): i for i, zp in enumerate(zarr_paths)}
        if use_runtime_global_stats
        else None
    )

    # ── dataset：只認 v4 unified（tes_unified_v1）───────────────────────
    # ★v3（`cases/` 扁平格式）已全數歸檔，本專案不再有格式分支。
    #   非 unified 一律在入口 raise —— 沒有退路、沒有 fallback，錯的資料不會
    #   悄悄走進一條「還能跑但語義不同」的舊路徑。
    _not_unified = [zp for zp in zarr_paths if not is_v4_zarr(zp)]
    if _not_unified:
        raise ValueError(
            "zarr_paths 含非 v4 unified 資料：\n"
            + "\n".join(f"  {zp}" for zp in _not_unified)
            + "\n訓練端只接受 v4 unified（tes_unified_v1）格式。"
        )
    for zp in zarr_paths:
        if not GraphIndex.for_zarr(zp).is_unified:
            raise ValueError(
                f"{zp} 是 v4 舊佈局（獨立 graph.json），非 tes_unified_v1。"
                " 請先轉為 unified 單檔（見 docs/DATA_GENERATION.md）。"
            )

    if not cfg.mean_module_enabled:
        raise ValueError(
            "mean_module_enabled 必須為 True —— 本專案只包含 mean pipeline。"
        )
    if not cfg.val_zarr_paths:
        raise ValueError(
            "必須在 config 指定 `val_zarr_paths`（DE 系列）。角色由 config 決定，"
            "不從資料推導——unified 格式的資料層沒有 split 欄。"
        )
    _assert_no_holdout([str(p) for p in cfg.val_zarr_paths])

    logger.info(
        "v4 unified：train %d 本（D 系列）／val %d 本（DE 系列）",
        len(zarr_paths), len(cfg.val_zarr_paths),
    )
    if cfg.mean_ablation_arm != "use_edge":
        logger.warning(
            "★三臂消融：本次訓練為 %s 臂，非現狀 use_edge", cfg.mean_ablation_arm
        )

    _tr = split_graph_datasets(
        zarr_paths=zarr_paths, role="train",
        vflip_aug_prob=cfg.mean_vflip_aug_prob,
        ablation_arm=cfg.mean_ablation_arm,
    )
    _va = split_graph_datasets(
        zarr_paths=[str(p) for p in cfg.val_zarr_paths], role="val",
        vflip_aug_prob=0.0, ablation_arm=cfg.mean_ablation_arm,
    )
    train_ds, val_cold_ds, val_warm_ds = _tr.train, _va.val_cold, _va.val_warm
    _collate = collate_graph_batch

    logger.info(
        "訓練集：%d cases | val cold：%d cases | val warm：%d cases",
        len(train_ds),
        len(val_cold_ds) if val_cold_ds is not None else 0,
        len(val_warm_ds) if val_warm_ds is not None else 0,
    )

    train_loader = build_tes_dataloader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=_collate, seed=loader_seed(cfg, LOADER_SEED_OFFSET_TRAIN),
    )
    val_cold_loader = (
        build_tes_dataloader(
            val_cold_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.val_num_workers,   # ★見 config 註解：不與 train 共用
            collate_fn=_collate,
            seed=loader_seed(cfg, LOADER_SEED_OFFSET_VAL_COLD),
        )
        if val_cold_ds is not None
        else None
    )
    val_warm_loader = (
        build_tes_dataloader(
            val_warm_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.val_num_workers,   # ★見 config 註解：不與 train 共用
            collate_fn=_collate,
            seed=loader_seed(cfg, LOADER_SEED_OFFSET_VAL_WARM),
        )
        if val_warm_ds is not None
        else None
    )

    # ── multi-edge pushforward chain loader（缺陷 b lever）──────────────
    chain_loader: Optional[DataLoader] = None
    if cfg.mean_chain_enabled:
        chain_ds = build_chain_train_dataset(
            zarr_paths=zarr_paths,
            chain_len=cfg.mean_chain_len, ablation_arm=cfg.mean_ablation_arm,
        )
        logger.info("multi-edge pushforward：chain 訓練集 %d 條（len=%d）",
                    len(chain_ds), cfg.mean_chain_len)
        chain_loader = build_tes_dataloader(
            chain_ds, batch_size=cfg.mean_chain_batch_size, shuffle=True,
            num_workers=cfg.val_num_workers, collate_fn=collate_chain_batch,
            seed=loader_seed(cfg, LOADER_SEED_OFFSET_CHAIN),
        )

    # ── 正規化器（runtime dataset-global stats）──────────────────────
    normalizer: TESCondNormalizer = TESCondNormalizer.from_stats_dict(global_stats)
    field_normalizer = FieldNormalizer.from_stats_dict(global_stats)
    source_field_normalizers = [
        FieldNormalizer.from_file(p) for p in source_stats_paths
    ]

    # ── 模型 ──────────────────────────────────────────────────────────
    model = build_tes_model(cfg.model)
    n_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "TES model (%s) 參數量：%s（%.2f M）",
        cfg.model.arch,
        f"{n_params:,}",
        n_params / 1e6,
    )

    # ── 損失函數 ──────────────────────────────────────────────────────
    if not cfg.loss_terms:
        raise ValueError(
            "TES training now requires semantic loss_terms. "
            "Legacy numeric loss configs are archived and no longer supported."
        )
    logger.info("Semantic loss terms: %s", cfg.loss_terms)
    loss_fn: TESSemanticLoss = TESSemanticLoss(
        cfg.loss_terms,
        source_stats_paths[0],
        field_normalizer=field_normalizer,
        pod_basis_path=cfg.pod_basis_path,
        pod_modes=cfg.pod_modes,
        ssim_range_from_stats=cfg.ssim_range_from_stats,
    )

    # ── Trainer ───────────────────────────────────────────────────────
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用裝置：%s", device)

    trainer: TESTrainer = TESTrainer(
        cfg=cfg,
        model=model,
        loss_fn=loss_fn,
        normalizer=normalizer,
        train_loader=train_loader,
        val_cold_loader=val_cold_loader,
        val_warm_loader=val_warm_loader,
        device=device,
        field_normalizer=field_normalizer,
        source_field_normalizers=source_field_normalizers,
        chain_loader=chain_loader,
        create_run_dir=create_run_dir,
    )

    # ── RNG / determinism provenance（純記錄，不改任何旗標或抽樣）──────
    # `apply_determinism_mode` 在預設檔位 "off" 直接 return {} 且不寫任何 log，
    # 使 canonical run 事後完全無法回答「當時的 cudnn/TF32/PYTHONHASHSEED 是什麼」。
    # 此處無條件補記一次：logger 已由 TESTrainer._setup_logging 接上 training.log，
    # 另寫一筆 rng_provenance 事件進 metrics.json（create_run_dir=False 時
    # `_write_jsonl` 內部的 `self._metrics_fp is not None` 檢查會讓它成為 no-op）。
    # ★覆蓋範圍限縮（勿外推）：只有走本函式 `build_tes_trainer` 的 run 會有
    #   metrics.json 的 rng_provenance 事件；評估腳本沒有 provenance 事件。
    #   ckpt identity 側亦尚未寫入。
    _rng_provenance: dict[str, Any] = {
        "event": "rng_provenance",
        "seed": int(cfg.seed),
        "determinism_mode": str(cfg.determinism_mode),
        "loader_seed_offset": bool(cfg.loader_seed_offset),
        "loader_seeds": {
            "train": loader_seed(cfg, LOADER_SEED_OFFSET_TRAIN),
            "val_cold": loader_seed(cfg, LOADER_SEED_OFFSET_VAL_COLD),
            "val_warm": loader_seed(cfg, LOADER_SEED_OFFSET_VAL_WARM),
            "chain": loader_seed(cfg, LOADER_SEED_OFFSET_CHAIN),
        },
        "streams": rng_streams.streams_summary(),
        "runtime_flags": rng_streams.snapshot_runtime_flags(),
    }
    logger.info("[R7][rng_provenance] %s", _rng_provenance)
    trainer._write_jsonl(_rng_provenance, force_flush=True)

    if resume:
        trainer.load_checkpoint(resume)
    elif finetune:
        trainer.load_weights_only(finetune)

    return trainer


def main() -> None:
    """TES-NCA 訓練主函式（薄入口：parse args → load config → build_tes_trainer → train）。"""
    parser = argparse.ArgumentParser(description="TES-NCA 訓練")
    parser.add_argument("--config", required=True, help="YAML 設定檔路徑")
    parser.add_argument("--resume", default=None, help="從 checkpoint 恢復訓練")
    parser.add_argument(
        "--finetune",
        default=None,
        help="warm restart：只載入 model weights，optimizer/scheduler 重置",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆寫 cfg.seed（RP 多 seed 用，不改 config）",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="覆寫 cfg.max_steps")
    parser.add_argument("--tag", default=None, help="output_dir 額外後綴（如 s1/s2 seed 標記）")
    args: argparse.Namespace = parser.parse_args()

    cfg = load_tes_config(args.config)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.max_steps is not None:
        cfg.max_steps = int(args.max_steps)

    trainer = build_tes_trainer(
        cfg, finetune=args.finetune, resume=args.resume, tag=args.tag
    )
    trainer.train()


if __name__ == "__main__":
    main()
