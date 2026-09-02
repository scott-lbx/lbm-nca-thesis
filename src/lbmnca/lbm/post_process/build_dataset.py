r"""D／DE／DT 全量轉檔驅動：H5 → v3 → mean-only unified 單檔 → 隔離驗證。

## 為什麼要有這支

六個 dataset、四個步驟、三種角色，每一步的參數都不同（誰算 static/cond 統計、
誰沿用、誰寫到 holdout 根目錄）。手打一次錯一次——尤其 `--stats-from` 漏掉會讓
DE／DT 各自算自己的 static/cond 統計，分數與訓練集不可比，而且**不會報錯**。

## 關鍵設計

**physics 正規化已凍結。** `M_REF` 與 `S` 取自 `norm_frozen`，不由任何資料集
量測產生 → 「跨 dataset 尺度不一致」在構造上不可能發生。掃描的角色因此從
「產生常數」變成「驗收資料裝不裝得下」，逐本 PASS 才算數。

**static/cond 仍是量測值。** 幾何與條件向量的語義與場通道不同，未納入本次凍結
範圍 → `--stats-from` 機制對這兩項仍然必要，順序也仍然不可換
（host 先產出，其餘五本沿用）。

**mean-only。** zarr 不保存湍流時序：每個 case 只留時均場，加上冷啟動 edge 的
第 0 幀（靜止態 IC）。凍結尺度是對**時均場**校準的，存過渡段等於存一份
反正規化不回來的資料。體積因此由 164.5 GB 降到約 14 GB。

**角色是 dataset 級屬性。** 一個 dataset 一個角色（幾何互斥由生成端保證），
故由 `--dataset-role` 直接宣告即可，資料層不寫 split 欄。訓練端的角色來自
config 的 `zarr_paths`／`val_zarr_paths` 分組。

**DT 寫在 `zarr_holdout/`。** 路徑本身就是 `lbmnca/nca/tes/train.py`
的 `_assert_no_holdout` 判別依據。與訓練資料同目錄的保留集遲早會被誤讀。

## 用法

```bash
python -m lbmnca.lbm.post_process.build_dataset            # dry-run，只印計畫
python -m lbmnca.lbm.post_process.build_dataset --execute
python -m lbmnca.lbm.post_process.build_dataset --execute --from-step 2
```
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from lbmnca.paths import REPO_ROOT, data_root, dataset_dir, zarr_dir

#: 執行子步驟用的 python 直譯器——沿用啟動本腳本的同一個直譯器，
#: 避免對 venv 目錄結構做任何假設。
_PY = sys.executable
#: 輸出根目錄。可由 CLI 覆寫 —— 重轉時先寫到新目錄、驗證通過再換掉舊的，
#: 失敗時才有回頭路（舊 zarr 是既有訓練結果的唯一來源）。
_ZARR = zarr_dir()
_HOLD = data_root() / "zarr_holdout"

#: 訓練集三本。host 擺第一個 —— 統計由它掃全部三本後產出。
TRAIN_HOST = "D1-1_Cold_M400C400"
TRAIN_OTHERS = ["D0-1_Cold_M40C40", "D2-1_Warm_M200C200R24S60"]
#: 驗證集（val），沿用訓練集常數
EVAL_SETS = ["DE0-1_Cold_M12", "DE1-1_Warm_M48R12"]
#: 保留集（test），沿用訓練集常數，且寫在另一個根目錄
TEST_SETS = ["DT_Stress"]

STATS_SRC = _ZARR / f"{TRAIN_HOST}_stats.json"


def run(cmd: list[str], execute: bool, cwd: Path | None = None) -> None:
    """執行子步驟。任何非零退出即中止——後續步驟都依賴前一步的產物。"""
    pretty = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"\n$ {pretty}", flush=True)
    if not execute:
        return
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd or REPO_ROOT))
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"步驟失敗（exit {r.returncode}，{dt/60:.1f} 分）：{pretty}")
    print(f"  ✓ {dt/60:.1f} 分", flush=True)


def convert(ds: str, role: str, root: Path, execute: bool,
            scan_dirs: list[str] | None = None,
            stats_from: Path | None = None) -> None:
    cmd = [_PY, "-u", "-m", "lbmnca.lbm.post_process.tes_h5_to_zarr",
           "--dataset-dir", dataset_dir(ds),
           "--dataset-role", role,
           "--output-root", root]
    if scan_dirs:
        cmd += ["--stats-scan-dirs"] + [dataset_dir(d) for d in scan_dirs]
    if stats_from:
        cmd += ["--stats-from", stats_from]
    run(cmd, execute)


def step1_guard_existing(execute: bool) -> None:
    """既有產物即中止。重轉必須是明確的決定，不能被這支腳本悄悄覆寫掉。"""
    hits = []
    for ds in [TRAIN_HOST] + TRAIN_OTHERS + EVAL_SETS:
        hits += [p for p in (_ZARR / f"{ds}_unified.zarr",
                             _ZARR / f"{ds}.zarr", _ZARR / f"{ds}_v4.zarr") if p.exists()]
    for ds in TEST_SETS:
        hits += [p for p in (_HOLD / f"{ds}_unified.zarr",
                             _HOLD / f"{ds}.zarr", _HOLD / f"{ds}_v4.zarr") if p.exists()]
    if hits:
        raise RuntimeError(
            "既有 zarr 產物存在，拒絕重轉（會覆寫或與舊檔混雜）：\n  "
            + "\n  ".join(str(p) for p in hits)
            + "\n確定要重轉請先自行移除上列路徑。")
    print("  無既有產物 ✓")


def step2_convert(execute: bool) -> None:
    """★順序不可換：host 先產出常數，其餘五本才有東西可沿用。"""
    convert(TRAIN_HOST, "train", _ZARR, execute, scan_dirs=TRAIN_OTHERS)
    for ds in TRAIN_OTHERS:
        convert(ds, "train", _ZARR, execute, stats_from=STATS_SRC)
    for ds in EVAL_SETS:
        convert(ds, "val", _ZARR, execute, stats_from=STATS_SRC)
    for ds in TEST_SETS:
        convert(ds, "test", _HOLD, execute, stats_from=STATS_SRC)


def step3_sidecar(execute: bool) -> None:
    """v4 migration 需要 `{ds}_cases.json` 的 parent 樹來合成 node/edge。"""
    run([_PY, "-m", "lbmnca.lbm.post_process.make_cases_sidecar", "--datasets",
         TRAIN_HOST, *TRAIN_OTHERS, *EVAL_SETS, "--out-root", _ZARR], execute)
    run([_PY, "-m", "lbmnca.lbm.post_process.make_cases_sidecar", "--datasets", *TEST_SETS,
         "--out-root", _HOLD], execute)


def step4_unified(execute: bool) -> None:
    """v3 cases → mean-only unified 單檔（`tes_unified_mean_v1`）。

    ★ 取代舊的 v3→v4→unified 兩步。mean-only 範疇下 `postconv` 與過渡段
    `snapshots` 沒有任何讀取端，而舊 migrate 的不變式（``n_transit +
    n_postconv == T``）在只存一幀的資料上必然失敗 —— 那些不變式保護的東西
    已經不存在。見 `build_unified_zarr.py` 的模組說明。
    """
    for ds in [TRAIN_HOST] + TRAIN_OTHERS + EVAL_SETS:
        run([_PY, "-u", "-m", "lbmnca.lbm.post_process.build_unified_zarr", "--dataset", ds,
             "--src-root", _ZARR, "--execute", "--overwrite"], execute)
    for ds in TEST_SETS:
        run([_PY, "-u", "-m", "lbmnca.lbm.post_process.build_unified_zarr", "--dataset", ds,
             "--src-root", _HOLD, "--execute", "--overwrite"], execute)


def step6_verify(execute: bool) -> None:
    """四項驗證：常數逐位元同源、符合度全 PASS、DT 隔離、訓練端 guard 有效。"""
    if not execute:
        print("\n[verify] （dry-run 略過）")
        return
    print("\n=== 驗證 ===")
    from lbmnca.lbm.post_process import norm_frozen as nf   # noqa: PLC0415

    bad: list[str] = []
    for ds, root in ([(d, _ZARR) for d in [TRAIN_HOST] + TRAIN_OTHERS + EVAL_SETS]
                     + [(d, _HOLD) for d in TEST_SETS]):
        s = json.loads((root / f"{ds}_stats.json").read_text(encoding="utf-8"))
        fz = s["frozen_norm"]
        # ★ 凍結常數是寫死的，逐位元一致是**必然**而非巧合 —— 不一致代表
        #   某一本是用不同版本的程式碼轉的，那比數值差異更嚴重。
        same = (fz["schema_version"] == nf.SCHEMA_VERSION
                and fz["S"] == nf.S.tolist()
                and fz["M_REF"] == nf.M_REF.tolist()
                and fz["ATANH_A"] == nf.ATANH_A)
        comp = s["compliance"]
        ok = same and comp["verdict"] == "PASS"
        tag = "  [holdout]" if root is _HOLD else ""
        print(f"  {ds:28s} 常數{'✓' if same else '✗'} "
              f"符合度 {comp['verdict']} "
              f"軌道比 {comp['util_ratio_orbit']:.2f}{tag}")
        if not ok:
            bad.append(ds)
    if bad:
        raise RuntimeError(f"常數不同源或符合度 FAIL：{bad}")

    leaked = [p.name for p in _ZARR.glob("DT*")]
    if leaked:
        raise RuntimeError(f"DT 產物出現在訓練根目錄：{leaked}")
    print("  dataset_zarr/ 無 DT 產物 ✓")

    guard = subprocess.run(
        [_PY, "-c",
         "from lbmnca.nca.tes.train import _assert_no_holdout as g; "
         f"g([r'{_HOLD / 'DT_Stress_unified.zarr'}'])"],
        capture_output=True, text=True)
    if guard.returncode == 0:
        raise RuntimeError("★訓練端 guard 沒擋住 DT —— 保留集會進訓練")
    print("  訓練 guard 擋下 DT ✓")


_STEPS = [
    ("1 既有產物檢查", step1_guard_existing),
    ("2 H5 → zarr v3（凍結常數）", step2_convert),
    ("3 cases sidecar", step3_sidecar),
    ("4 → mean-only unified 單檔", step4_unified),
    ("5 驗證", step6_verify),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description="D／DE／DT 全量轉檔驅動")
    ap.add_argument("--execute", action="store_true", help="實際執行（預設 dry-run）")
    ap.add_argument("--from-step", type=int, default=1, help="從第 N 步開始（1-6）")
    ap.add_argument("--to-step", type=int, default=len(_STEPS))
    ap.add_argument("--zarr-root", default=None, help="訓練/驗證集輸出根目錄")
    ap.add_argument("--hold-root", default=None, help="保留集（DT）輸出根目錄")
    a = ap.parse_args()

    global _ZARR, _HOLD, STATS_SRC
    if a.zarr_root:
        _ZARR = Path(a.zarr_root)
    if a.hold_root:
        _HOLD = Path(a.hold_root)
    STATS_SRC = _ZARR / f"{TRAIN_HOST}_stats.json"
    _ZARR.mkdir(parents=True, exist_ok=True)
    _HOLD.mkdir(parents=True, exist_ok=True)
    print(f"輸出根目錄：train/val={_ZARR}  holdout={_HOLD}")

    os.environ["PYTHONIOENCODING"] = "utf-8"
    mode = "EXECUTE" if a.execute else "DRY-RUN"
    print(f"=== D 系列全量轉檔 [{mode}] 步驟 {a.from_step}–{a.to_step} ===")
    t0 = time.time()
    for i, (name, fn) in enumerate(_STEPS, 1):
        if not (a.from_step <= i <= a.to_step):
            continue
        print(f"\n{'='*70}\n── 步驟 {name} ──\n{'='*70}", flush=True)
        fn(a.execute)
    print(f"\n全部完成，耗時 {(time.time()-t0)/60:.1f} 分")
    return 0


if __name__ == "__main__":
    sys.exit(main())
