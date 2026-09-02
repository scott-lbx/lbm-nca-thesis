"""verify_gvec_active_indices.py — GVEC_ACTIVE_INDICES 的定義完整性驗證。

背景
----
DEEP_AUDIT_2026-07 Stage 2.7（`B2-02-宣稱-10`）發現時，`gvec_api.GVEC_ACTIVE_INDICES`
在舊 polyrepo（lbm-nca-thesis）內另有 4 份手抄副本，其中 `src/lbmnca/lbm/post_process/
tes_h5_to_zarr.py` **位於 dataset 生成主線上**，改主清單不會傳導、也不會報錯。

合併為單一套件 `lbmnca` 後，那 4 份手抄副本已全部移除：
  - `tes_fps_selector.py` / `fix_missing_masks_and_stats.py` / `patch_zero_gvec.py`
    三份未隨併入本 repo（不在 monorepo 的呼叫路徑上）。
  - `tes_h5_to_zarr.py`（原最危險的那份）與 `gvec_bridge.py` 都已改成直接
    ``from lbmnca.morphospace.physics_proxy.gvec_api import GVEC_ACTIVE_INDICES``。

canonical 清單現在**只有一份**（`src/lbmnca/morphospace/physics_proxy/gvec_api.py`），
「多份手抄副本互相漂移」的風險已不存在。本檢查因此改為守兩件仍然會靜默出錯的事：

檢查項目
--------
1. canonical 索引本身合法（範圍／嚴格遞增／無重複，落在 `FEATURE_NAMES` 之內）
2. `GVEC_FEATURE_SPEC`（Stage 2.6 名實對照表）的 `code` 欄與選取出的 14 個名稱
   逐項對齊 —— 改索引卻忘了改誠實描述表，仍是靜默漂移
3. 兩個下游呼叫端（dataset 生成主線 `tes_h5_to_zarr.py`、前端橋接 `gvec_bridge.py`）
   確實還在 import canonical，沒有退化回本地手抄副本

實作方式
--------
**純 AST 靜態解析，不 import 任何 `lbmnca` 子套件。**
`scripts/dataset/*.py` 類腳本在 module 層可能有副作用（插 sys.path、印訊息、
甚至讀取外接磁碟路徑），import 它們只為讀一個常數並不安全；直接對原始碼做
AST 解析可以繞開這個問題。

用法
----
    python scripts/qa/verify_gvec_active_indices.py
    python scripts/qa/verify_gvec_active_indices.py --quiet

退出碼：0 = PASS，1 = FAIL。唯讀，不修改任何檔案。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# ── canonical（SSOT）──────────────────────────────────────────────────────
#    (標籤, 相對 repo root 的路徑, 變數名)
CANONICAL: tuple[str, str, str] = (
    "canonical",
    "src/lbmnca/morphospace/physics_proxy/gvec_api.py",
    "GVEC_ACTIVE_INDICES",
)

FEATURES_FILE: str = "src/lbmnca/morphospace/physics_proxy/features.py"

# 下游呼叫端 —— 必須維持 import canonical，不得退化為本地手抄副本
IMPORTERS: list[tuple[str, str]] = [
    ("lbm_tes_h5_to_zarr", "src/lbmnca/lbm/post_process/tes_h5_to_zarr.py"),
    ("bridge_gvec_bridge", "src/lbmnca/bridge/geometry/gvec_bridge.py"),
]


class GvecConsistencyError(AssertionError):
    """GVEC_ACTIVE_INDICES 的定義或下游引用出現問題。"""


def _parse_module(path: Path) -> ast.Module:
    """以 AST 解析單一 .py 檔（不執行）。

    Args:
        path: 待解析檔案的絕對路徑。

    Returns:
        解析後的 AST module。

    Raises:
        GvecConsistencyError: 檔案不存在或無法解析。
    """
    if not path.exists():
        raise GvecConsistencyError(f"檔案不存在：{path}")
    try:
        # utf-8-sig：repo 內部分檔案帶 BOM，用純 utf-8 讀會讓 ast.parse 在第 1 行報 U+FEFF。
        return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError as exc:                                  # pragma: no cover
        raise GvecConsistencyError(f"無法解析 {path}：{exc}") from exc


def read_literal_assign(path: Path, var_name: str) -> tuple[list, int]:
    """從 .py 檔中取出某個 module-level 常數的字面值（不 import）。

    支援 ``x = [...]``（Assign）與 ``x: list[int] = [...]``（AnnAssign）兩種寫法。

    Args:
        path: 目標 .py 檔絕對路徑。
        var_name: module-level 變數名稱。

    Returns:
        (字面值, 該賦值所在行號)。

    Raises:
        GvecConsistencyError: 找不到該變數，或其右值不是字面常數。
    """
    tree = _parse_module(path)
    for node in tree.body:
        target_name: str | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Name
        ):
            target_name = node.targets[0].id
        if target_name != var_name or node.value is None:
            continue
        try:
            return ast.literal_eval(node.value), node.lineno
        except ValueError as exc:
            raise GvecConsistencyError(
                f"{path}:{node.lineno} 的 {var_name} 不是字面常數（無法靜態比對）：{exc}"
            ) from exc
    raise GvecConsistencyError(f"{path} 找不到 module-level 的 {var_name}")


def _has_top_level_literal_named_like(path: Path, base_name: str) -> bool:
    """檢查檔案是否有 top-level、名稱為 ``base_name`` 或 ``_{base_name}`` 的字面清單賦值。

    下游檔案慣例上會把 import 進來的符號用底線前綴另存一個私有名稱（例如
    ``tes_h5_to_zarr.py`` 的 ``GVEC_ACTIVE_INDICES as _GVEC_ACTIVE_INDICES``）；
    若有人把 import 退化回本地手抄副本，最可能沿用的正是這個既有名稱，而不是
    改用完全相同的 ``base_name``。兩種名稱都要查，才抓得到真實會發生的退化。

    Args:
        path: 目標 .py 檔絕對路徑。
        base_name: 基準變數名（不含底線前綴）。

    Returns:
        True 表示該檔在 module-level 有對應的字面清單賦值。
    """
    for candidate in (base_name, f"_{base_name}"):
        try:
            read_literal_assign(path, candidate)
            return True
        except GvecConsistencyError:
            continue
    return False


def module_imports_symbol(path: Path, symbol: str) -> bool:
    """檢查某 .py 檔是否以 ``from ... import symbol`` 匯入指定名稱（不限 module-level）。

    Args:
        path: 目標 .py 檔絕對路徑。
        symbol: 待確認的匯入名稱。

    Returns:
        True 表示該檔確實 import 了該符號。
    """
    tree = _parse_module(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == symbol:
                    return True
    return False


def check_all(repo_root: Path, verbose: bool = True) -> list[str]:
    """執行全部檢查。

    Args:
        repo_root: lbm-nca-thesis repo 根目錄。
        verbose: 是否印出逐項明細。

    Returns:
        失敗訊息清單；空清單代表全部 PASS。
    """
    failures: list[str] = []

    def say(msg: str) -> None:
        if verbose:
            print(msg)

    # ── 1. canonical 索引本身的合法性 ────────────────────────────────────
    label_c, rel_c, var_c = CANONICAL
    canon, line_c = read_literal_assign(repo_root / rel_c, var_c)
    say("=" * 78)
    say("[1] canonical 索引合法性（範圍／遞增／無重複）")
    say(f"    {label_c:<24} {rel_c}:{line_c}")
    say(f"    canonical = {canon}")

    feature_names, fn_line = read_literal_assign(repo_root / FEATURES_FILE, "FEATURE_NAMES")
    n_feat = len(feature_names)
    if any((not isinstance(i, int)) or i < 0 or i >= n_feat for i in canon):
        failures.append(f"[1] canonical 索引超出 FEATURE_NAMES 範圍 [0, {n_feat})：{canon}")
        say(f"    [FAIL] 索引超出 [0, {n_feat})")
    elif list(canon) != sorted(set(canon)):
        failures.append(f"[1] canonical 索引未嚴格遞增或有重複：{canon}")
        say("    [FAIL] 未嚴格遞增或有重複")
    else:
        say(f"    [OK  ] {len(canon)} 個索引全部落在 [0, {n_feat}) 且嚴格遞增無重複")

    derived = [feature_names[i] for i in canon]

    # ── 2. GVEC_FEATURE_SPEC 名實對照表與 14 維選取對齊 ──────────────────
    say("")
    say("[2] GVEC_FEATURE_SPEC（名實對照表）與 14 維選取對齊")
    try:
        spec, spec_line = read_literal_assign(repo_root / rel_c, "GVEC_FEATURE_SPEC")
    except GvecConsistencyError as exc:
        failures.append(f"[2] {exc}")
        say(f"    [FAIL] {exc}")
    else:
        spec_codes = [s["code"] for s in spec]
        if spec_codes == derived:
            say(f"    [OK  ] 14 列 code 欄與 FEATURE_NAMES[indices] 逐項相同"
                f"（{rel_c}:{spec_line}）")
        else:
            failures.append(
                f"[2] {rel_c}:{spec_line} 的 GVEC_FEATURE_SPEC code 欄與選取結果不符：\n"
                f"      selected = {derived}\n"
                f"      spec     = {spec_codes}"
            )
            say("    [FAIL] code 欄與選取結果不符（見結尾摘要）")
        missing_keys = [
            s.get("code", "?")
            for s in spec
            if not {"code", "honest", "scope", "computed", "verdict", "claims_physics"}
            <= set(s)
        ]
        if missing_keys:
            failures.append(f"[2] GVEC_FEATURE_SPEC 下列項目缺欄位：{missing_keys}")
            say(f"    [FAIL] 缺欄位：{missing_keys}")

    # ── 3. 下游呼叫端仍在 import canonical，未退化為本地手抄副本 ─────────
    say("")
    say("[3] 下游呼叫端未退化為本地手抄副本")
    for label, rel in IMPORTERS:
        path = repo_root / rel
        try:
            imports_ok = module_imports_symbol(path, "GVEC_ACTIVE_INDICES")
        except GvecConsistencyError as exc:
            failures.append(f"[3] {label}: {exc}")
            say(f"    [FAIL] {label:<20} {exc}")
            continue
        # 裸名與底線前綴兩種名稱都要查：下游慣例是把 import 進來的符號另存成
        # `_GVEC_ACTIVE_INDICES`，退化成手抄副本時最可能沿用的正是那個名字。
        # 只查裸名的話，真實會發生的那種退化剛好漏掉。
        has_own_copy = _has_top_level_literal_named_like(path, "GVEC_ACTIVE_INDICES")
        if imports_ok and not has_own_copy:
            say(f"    [OK  ] {label:<20} {rel}（import canonical，無本地副本）")
        else:
            reason = (
                "未 import GVEC_ACTIVE_INDICES"
                if not imports_ok
                else "出現本地硬編副本（退化為手抄清單）"
            )
            failures.append(f"[3] {label} ({rel}) {reason}")
            say(f"    [FAIL] {label:<20} {reason}")

    return failures


def main() -> int:
    """CLI 入口。

    Returns:
        0 = 全部 PASS；1 = 有問題。
    """
    parser = argparse.ArgumentParser(description="GVEC_ACTIVE_INDICES 定義完整性驗證")
    parser.add_argument("--quiet", action="store_true", help="只印最終結論")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="lbm-nca-thesis repo 根目錄（預設由本檔位置推導）",
    )
    args = parser.parse_args()

    failures = check_all(args.repo_root, verbose=not args.quiet)

    print("")
    print("=" * 78)
    if failures:
        print(f"[FAIL] GVEC_ACTIVE_INDICES 完整性檢查失敗（{len(failures)} 項）")
        for f in failures:
            print(f"  - {f}")
        print("")
        print("  修法：以 src/lbmnca/morphospace/physics_proxy/gvec_api.py 為準，")
        print("  下游呼叫端一律 import，不得再手抄清單。")
        return 1
    print("[PASS] canonical 索引合法；名實對照表一致；2 個下游呼叫端仍 import canonical，")
    print("       未退化為本地手抄副本。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
