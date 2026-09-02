# ★2026-08-05：以下斷言的定位錨（long_rollout / pool_step / dy2_step /
#   rollout_runner / run_async_eval_generation / eval_test_split）已隨 v3、turb
#   與 dead arms 一併歸檔。台帳存在的意義是「文件寫的數字＝程式碼實況」，
#   當被稽核的程式碼整段不存在時，正確處置是撤下該條斷言而不是改個數字讓它綠。
#   歸檔源碼見 Archive/{v3_dataset,gen8_turb,dead_arms}_2026-08-05/。
"""訓練 step 的 RNG 來源靜態檢查（R7；純 ast，不 import 任何訓練模組）。

DEEP_AUDIT_2026-07 根因 R7 的解法是把訓練期離散抽樣改走
``training/rng_streams.py`` 的具名串流。本測試以 ast 掃描
``src/tes/training/steps/*.py``，鎖住「除已知未遷移檔外，不得再從標準庫
``random`` 的**共用全域序列**抽樣」，避免新 step 又把抽樣掛回同一條流。

★守備範圍（請照字面理解，不要外推）
------------------------------------
偵測得到的兩種寫法：

* ``import random`` / ``import random as _random`` 後的 ``<別名>.<attr>``；
* ``from random import randint`` / ``from random import random as _r`` 後
  **直接呼叫**該名稱。

**不偵測**的情形（已知且刻意）：

* ``from random import Random`` / ``SystemRandom`` 後自建實例——那是獨立序列，
  不是本測試要防的共用全域序列（見 :data:`_INDEPENDENT_RNG_NAMES`）；
* ``from random import *``（星號匯入）——:func:`_random_global_function_names`
  一律跳過 ``alias.name == "*"``，故 star-import 後直接呼叫 ``randint(...)``
  **不會**被抓到，儘管它確實共用全域序列。這是為了不必維護 ``random.__all__``
  的名單而刻意留下的盲區。掃描器這個行為由
  :func:`test_scanner_blindspot_star_import` 鎖住，「steps/ 內 0 處 star-import」
  這個前提則由 :func:`test_steps_dir_premises_for_known_blindspots` 持續查驗；
* ``importlib`` / ``__import__`` 等動態取得 ``random`` 模組的寫法；
* ``steps/`` **以外**的目錄（如 ``training/mean_pool.py``、``data/tes_dataset.py``）；
* ``torch`` 全域 RNG（``torch.randint`` / ``randperm`` / ``rand``）——它是另一條
  全域序列，本模組與本測試都管不到。

★兩種寫法的判定**不對稱**（刻意，且與上列第一項有張力）：``Random`` /
``SystemRandom`` 的豁免**只套用在 ``from random import`` 形式**；
``import random`` 後的 ``random.Random(0)`` 走 :func:`_scan_source` 的
Attribute 分支，該分支不看屬性名，因此會被判為違規（假陽性）。
現況 ``steps/`` 內 ``random.Random`` 命中 0 處，故尚未實際誤報（由
:func:`test_steps_dir_premises_for_known_blindspots` 持續查驗）；掃描器這個
不對稱行為由 :func:`test_scanner_attribute_form_has_no_class_exemption` 鎖住，
日後若要消除假陽性，須同時改該分支與本段敘述。

所以本測試綠燈只代表「掃描範圍內沒有這兩種寫法」，**不等於**「訓練期抽樣全部
具名化」。完整的涵蓋範圍清單見 ``training/rng_streams.py`` 模組 docstring。

★允許清單（``_UNMIGRATED``）記錄**目前仍未遷移**的檔案，屬 R3 所有權。
清單只放寬「這些檔案不檢查」，不要求它們一定要有裸 ``random.*``，故 R3
把它們遷移完成後本測試仍會通過（屆時可自行把該檔從清單移除以收緊）。

本檔只做原始碼解析（``ast.parse``），不執行也不 import 被掃描的模組，
故不碰 GPU、不讀資料集。
"""

from __future__ import annotations

import ast
from pathlib import Path

_STEPS_DIR: Path = Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca" / "tes" / "training" / "steps"

# 已知仍掛在共用全域 random 序列上的檔案（R3 所有權；見 rng_streams.py 模組 docstring）。
# ★2026-07-29 清空：mean_step.py 的四處裸 `_random` 已遷移至具名串流
#   （GT-phase 抽樣與 curriculum k → "schedule"；額外 rollout 步數與 inv depth
#   → "rollout"），R7-X3 核銷。清單為空即「steps/ 下無檔案掛在共用全域序列」，
#   任何回歸都會由下方掃描直接紅燈。
_UNMIGRATED: frozenset[str] = frozenset()

# ``from random import X`` 匯入這些名稱時建立的是**獨立** RNG 實例，不共用全域序列，
# 故不列為違規（本測試防的是共用序列，不是「禁止使用 random 這個套件」）。
_INDEPENDENT_RNG_NAMES: frozenset[str] = frozenset({"Random", "SystemRandom"})


def _random_module_aliases(tree: ast.Module) -> set[str]:
    """找出檔案內以 ``import`` 敘述綁定到標準庫 ``random`` 模組的名稱。

    涵蓋 ``import random``、``import random as _random``（含函式內的區域 import）。
    **不涵蓋** ``importlib.import_module("random")`` / ``__import__`` 等動態取得。

    ★本函式只回傳「模組別名」，不區分後續存取的屬性；:data:`_INDEPENDENT_RNG_NAMES`
    的豁免**不套用**於這條路徑（詳見模組 docstring「兩種寫法的判定不對稱」）。

    Args:
        tree: 已解析的模組 AST。

    Returns:
        指向 ``random`` 模組的名稱集合（可能為空）。
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "random":
                    aliases.add(alias.asname or "random")
    return aliases


def _random_global_function_names(tree: ast.Module) -> dict[str, str]:
    """找出由 ``from random import X`` 綁入、且共用全域序列的函式名。

    ``random`` 的模組層函式是同一個隱藏 ``Random`` 實例的繫結方法，因此
    ``from random import randint`` 之後直接呼叫 ``randint(...)`` 與
    ``random.randint(...)`` 消費的是同一條序列——這正是 R7 要防的。
    :data:`_INDEPENDENT_RNG_NAMES` 內的名稱（``Random`` / ``SystemRandom``）
    會建立獨立實例，故排除。

    Args:
        tree: 已解析的模組 AST。

    Returns:
        ``本地名稱 -> 原始 random 屬性名`` 的對照（可能為空）。
    """
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "random" and node.level == 0:
            for alias in node.names:
                if alias.name in _INDEPENDENT_RNG_NAMES or alias.name == "*":
                    continue
                names[alias.asname or alias.name] = alias.name
    return names


def _scan_source(source: str, label: str) -> list[str]:
    """掃描一段原始碼，回傳共用全域 ``random`` 序列的抽樣位置。

    偵測兩種寫法（見模組 docstring 的守備範圍）：
    ``<random 別名>.<attr>``，以及 ``from random import X`` 後直接呼叫 ``X(...)``。

    ★Attribute 分支**不看屬性名**，故 ``random.Random(0)`` / ``random.SystemRandom()``
    也會被列為命中；:data:`_INDEPENDENT_RNG_NAMES` 的豁免僅作用於
    ``from random import`` 分支（見模組 docstring「兩種寫法的判定不對稱」）。

    Args:
        source: Python 原始碼字串。
        label: 顯示在結果字串前綴的名稱（通常是檔名）。

    Returns:
        ``"label:行號 寫法"`` 字串列表，依字母序（無命中則為空）。
    """
    tree = ast.parse(source, filename=label)
    aliases = _random_module_aliases(tree)
    from_imports = _random_global_function_names(tree)
    if not aliases and not from_imports:
        return []
    hits: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            hits.append(f"{label}:{node.lineno} {node.value.id}.{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in from_imports
        ):
            hits.append(
                f"{label}:{node.lineno} {node.func.id}()"
                f"（from random import {from_imports[node.func.id]}）"
            )
    return sorted(hits)


def _bare_random_calls(path: Path) -> list[str]:
    """回傳檔案內**偵測得到的**共用全域 ``random`` 抽樣位置。

    偵測範圍（含已知盲區）見模組 docstring 的「守備範圍」一節。

    Args:
        path: 待掃描的 .py 檔。

    Returns:
        ``"檔名:行號 寫法"`` 字串列表（無命中則為空）。
    """
    return _scan_source(path.read_text(encoding="utf-8"), path.name)


def test_steps_dir_exists() -> None:
    """掃描目標目錄必須存在（防路徑寫死後被搬走而變成空掃描）。"""
    assert _STEPS_DIR.is_dir(), f"找不到 steps 目錄：{_STEPS_DIR}"
    assert list(_STEPS_DIR.glob("*.py")), "steps 目錄內沒有任何 .py，掃描範圍可疑"


def test_no_bare_random_in_migrated_steps() -> None:
    """已遷移的 step 檔不得直接使用 ``random.*``（須走 ``rng_streams.stream()``）。"""
    offenders: list[str] = []
    for path in sorted(_STEPS_DIR.glob("*.py")):
        if path.name in _UNMIGRATED:
            continue
        offenders.extend(_bare_random_calls(path))
    assert not offenders, (
        "以下 step 直接使用標準庫 random（會掛回共用全域序列，違反 R7）：\n  "
        + "\n  ".join(offenders)
        + "\n請改用 `from ..rng_streams import stream` 後以 stream(\"<name>\") 取樣。"
    )




# ── 掃描器自身的行為鎖（讓 docstring 宣稱的守備範圍可被機器查驗）────────────
def test_scanner_detects_module_attribute_form() -> None:
    """``import random`` + ``random.<attr>``（含 as 別名）必須被抓到。"""
    assert _scan_source("import random\nrandom.randint(0, 3)\n", "t.py")
    assert _scan_source("import random as _r\n_r.random()\n", "t.py")


def test_scanner_detects_from_import_form() -> None:
    """``from random import X`` 後直接呼叫 ``X(...)`` 必須被抓到（round-3 修補點）。

    round-2 的掃描器只認 ``ast.Import``，這種寫法可整條逃逸，
    使模組 docstring 的「鎖住裸 random」讀起來比實際保護範圍寬。
    """
    hits = _scan_source("from random import randint\nk = randint(1, 5)\n", "t.py")
    assert hits, "from random import 形式未被偵測"
    assert "randint" in hits[0]
    assert _scan_source("from random import random as _rr\n_rr()\n", "t.py")


def test_scanner_ignores_independent_rng_classes() -> None:
    """``from random import Random`` 建立的是獨立序列，不應被判違規。"""
    assert not _scan_source("from random import Random\nrng = Random(0)\nrng.random()\n", "t.py")


def test_scanner_ignores_unused_import() -> None:
    """只 ``import random`` 而未使用（如 dy2_step.py 的殘留 import）不算違規。"""
    assert not _scan_source("import random\n\n\ndef f():\n    return 1\n", "t.py")


def test_scanner_blindspot_star_import() -> None:
    """``from random import *`` 是**已知盲區**——本測試鎖住「掃不到」這個現況。

    星號匯入後直接呼叫 ``randint(...)`` 確實共用全域序列，但
    :func:`_random_global_function_names` 一律跳過 ``*``（不想維護
    ``random.__all__`` 名單）。此處刻意斷言「無命中」，使模組 docstring 的
    盲區清單成為機器可驗事實；日後若補上這條偵測，本測試會紅並提醒同步改敘述。
    """
    assert not _scan_source("from random import *\nk = randint(1, 5)\n", "t.py")


def test_scanner_attribute_form_has_no_class_exemption() -> None:
    """``import random`` 後的 ``random.Random(0)`` **會**被判命中（已知不對稱）。

    :data:`_INDEPENDENT_RNG_NAMES` 的豁免只作用於 ``from random import`` 分支；
    Attribute 分支不看屬性名。本測試把這個假陽性風險寫成機器可驗的現況，
    與模組 docstring「兩種寫法的判定不對稱」一段互相對應。
    """
    assert _scan_source("import random\nrng = random.Random(0)\n", "t.py")


def test_steps_dir_premises_for_known_blindspots() -> None:
    """鎖住模組 docstring 兩個盲區敘述所依賴的**現況前提**（含允許清單內的檔案）。

    前提一：``steps/`` 內 0 處 ``from random import *``——星號匯入是刻意留下的
    盲區，一旦有人寫進來，本測試會紅並提醒盲區已從「理論上」變成「實際上」。
    前提二：``steps/`` 內 0 處 ``random.Random`` / ``random.SystemRandom``——
    Attribute 分支不做類別豁免，出現即是假陽性；兩者任一出現都必須先裁決
    要改掃描器還是改敘述。
    """
    star_imports: list[str] = []
    class_uses: list[str] = []
    for path in sorted(_STEPS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        aliases = _random_module_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "random":
                if any(alias.name == "*" for alias in node.names):
                    star_imports.append(f"{path.name}:{node.lineno}")
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr in _INDEPENDENT_RNG_NAMES
            ):
                class_uses.append(f"{path.name}:{node.lineno} {node.value.id}.{node.attr}")
    assert not star_imports, (
        "steps/ 出現 `from random import *`（本掃描器的已知盲區，抓不到後續呼叫）：\n  "
        + "\n  ".join(star_imports)
        + "\n請改為具名 import 或改走 rng_streams.stream()；並更新模組 docstring 的盲區敘述。"
    )
    assert not class_uses, (
        "steps/ 出現 `random.Random` / `random.SystemRandom`（Attribute 分支不做類別豁免，"
        "會被 _scan_source 判為違規＝假陽性）：\n  "
        + "\n  ".join(class_uses)
        + "\n請裁決要在 Attribute 分支比照 _INDEPENDENT_RNG_NAMES 排除，還是改寫模組 docstring。"
    )


def test_unmigrated_allowlist_matches_reality() -> None:
    """允許清單內的檔案必須存在。

    防「允許清單寫了已不存在的檔名」而讓守衛看起來比實際嚴格。
    ★本測試**不**檢查清單外的檔案是否乾淨——那由
    :func:`test_no_bare_random_in_migrated_steps` 負責。
    """
    for name in _UNMIGRATED:
        assert (_STEPS_DIR / name).is_file(), f"允許清單列了不存在的檔案：{name}"
