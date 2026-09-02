"""DataLoader worker 預算的回歸測試（DEEP_AUDIT W4，2026-08-02 長跑實測）。

問題形狀：``train.py`` 建 4 個 DataLoader（train／val_cold／val_warm／chain），
先前**四個都吃同一個 ``cfg.num_workers``** → ``num_workers=4`` 實際 spawn
**16 個** worker 進程，合計佔 ~28 GB RAM（單一 0.11–3.59 GB）；系統 61.7 GB 中
僅餘 14.0 GB，終止該進程樹後回到 42.4 GB。**以上是直接量測的事實**，也是本測試
要守的性質：設定裡寫「4」而執行期是「16」，使用者無從得知。

★2026-08-02 第五輪訂正：本 docstring 原本接著把一次長跑的減速歸因於此，
並引「每 2 小時由 ~1700 掉到 ~700（2.4×）」與「GPU 端指標全部正常」。
**該因果歸因已撤回**——(a) 症狀數字錯一個量級（由 metrics.json 的 ``t`` 欄重算
實為 4199→3800→2500→700→600→300 步/2h，首末 14×，且沒有任何窗是 ~1700）；
(b) 用來排除 GPU 側的 ``dt_loss_primary_ms``／``gpu_mem_mb`` 只有 step 1–100 的
n=44 樣本，而減速發生在 t≈5.5 h，零證據力；(c) 16 個 worker 從 t≈0 就存在，
無法解釋數小時後的下滑，且同機另一 LBM 批次於 t≈5.2 h 啟動與急降重合。

本測試守的是「worker 預算不得被四個 loader 共用放大」這個**組態事實**，
與那次減速的成因無關——即使成因另有其人，這個放大本身仍是缺陷。

全部純 CPU（只讀原始碼與 config 預設值）。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAIN_PY = REPO / "src" / "lbmnca" / "nca" / "tes" / "train.py"


def _num_workers_exprs() -> list[str]:
    """列出 ``train.py`` 中所有 ``build_tes_dataloader(...)`` 的 num_workers 引數原始碼。"""
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    out: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_tes_dataloader"
        ):
            continue
        for kw in node.keywords:
            if kw.arg == "num_workers":
                out.append(ast.unparse(kw.value))
    return out


def test_only_train_loader_uses_num_workers() -> None:
    """只有 train loader 可以吃 ``cfg.num_workers``；其餘走 ``cfg.val_num_workers``。

    ★這就是 2.4× 減速的回歸鎖：四個 loader 共用同一個值時，設定裡的「4」
    在執行期是「16」，而使用者無從得知。
    """
    exprs = _num_workers_exprs()
    assert len(exprs) >= 4, f"只找到 {len(exprs)} 個 loader，掃描條件可能壞了"
    n_train = sum(1 for e in exprs if e == "cfg.num_workers")
    assert n_train == 1, (
        f"有 {n_train} 個 loader 吃 cfg.num_workers（應只有 train）：{exprs}"
    )
    for e in exprs:
        assert e in ("cfg.num_workers", "cfg.val_num_workers"), (
            f"未預期的 num_workers 來源 {e!r}——新 loader 必須明示走哪一組預算"
        )


def test_val_num_workers_defaults_to_zero() -> None:
    """``val_num_workers`` 預設 0：val/chain loader 每 val_interval 才用一次，
    常駐 worker 的 RAM 代價遠大於省下的那幾秒。"""
    from lbmnca.nca.tes.config._legacy import TESTrainConfig

    assert TESTrainConfig().val_num_workers == 0


def test_worker_budget_is_bounded() -> None:
    """以預設值算出的**實際** worker 總數必須遠小於舊行為。

    舊：4 個 loader × num_workers → 4×4 = 16。
    新：train 4 個 ＋ 其餘 0 個 = 4。
    """
    from lbmnca.nca.tes.config._legacy import TESTrainConfig

    cfg = TESTrainConfig()
    n_loaders_using_val = sum(1 for e in _num_workers_exprs() if e == "cfg.val_num_workers")
    total = cfg.num_workers + n_loaders_using_val * cfg.val_num_workers
    old_total = (1 + n_loaders_using_val) * cfg.num_workers
    assert total <= cfg.num_workers, f"worker 總數 {total} 超出單一 loader 預算"
    assert total < old_total, f"未低於舊行為（新 {total} vs 舊 {old_total}）"
