# ★2026-08-05：以下斷言的定位錨（long_rollout / pool_step / dy2_step /
#   rollout_runner / run_async_eval_generation / eval_test_split）已隨 v3、turb
#   與 dead arms 一併歸檔。台帳存在的意義是「文件寫的數字＝程式碼實況」，
#   當被稽核的程式碼整段不存在時，正確處置是撤下該條斷言而不是改個數字讓它綠。
#   歸檔源碼見 Archive/{v3_dataset,gen8_turb,dead_arms}_2026-08-05/。
"""metrics.json 寫入淨化的回歸測試（DEEP_AUDIT W4 should_fix S1）。

守住的事實：``json.dumps`` 預設 ``allow_nan=True``，會把 ``float('nan')``／``inf``
寫成裸 token ``NaN``／``Infinity``——那不是合法 JSON（RFC 8259 沒有這些字面值）。
Python 的 ``json.loads`` 容忍它，所以整條證據鏈從來不會報錯；但論文證據一旦經過
非 Python 的嚴格解析，不是失敗就是被轉成 ``null``／字串，而缺漏恰好與被量的性質
相關（發散的步才會有 NaN）——MNAR，會讓聚合值系統性偏樂觀。

全部純 CPU。
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import pytest

from lbmnca.nca.tes.training.jsonl import (
    NONFINITE_SUFFIX,
    dumps_record,
    restore_record,
    sanitize_record,
)

SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca" / "tes" / "training"

#: 承載 metrics.json 的 file handle 名稱。凡是對這些名字呼叫 ``.write()`` 的地方，
#: 就是一個 metrics.json 寫入點。
#:
#: ★三輪訂正（must_fix）：本檔第一版用的是人工列的三元素 ``METRICS_WRITERS``，
#: 註解卻寫「四個實際寫 metrics.json 的位置」（清單只有 3 條、另一處註解又寫
#: 「三支檔案」）。★2026-08-05：``pool_step.py`` 與 ``dy2_step.py`` 已隨 dead arms 歸檔，實際寫入點 **4 個**——原文
#: 兩處走裸 ``json.dumps`` 完全不在清單內，於是「四個寫入點全部接上」這句宣稱
#: 通過了自己的測試。人工清單就是漏掉的原因，改為全 repo AST 掃描。
METRICS_HANDLES = ("metrics_file", "_metrics_fp")

#: 掃描範圍（整個 training 套件）。
SCAN_ROOT = SRC_ROOT


# ---------------------------------------------------------------------------
# 1. 輸出必須是合法 JSON（嚴格模式解得開）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_becomes_null_and_is_strict_json(bad: float) -> None:
    """非有限值 → ``null``，且以 ``parse_constant`` 攔截確認沒有裸 token 殘留。"""
    line = dumps_record({"step": 100, "loss": bad, "ok": 1.5})

    def _boom(tok: str) -> None:
        raise AssertionError(f"輸出含非法 JSON 字面值 {tok!r}")

    got = json.loads(line, parse_constant=_boom)
    assert got["loss"] is None
    assert got["ok"] == 1.5
    assert got["step"] == 100


@pytest.mark.parametrize(
    ("bad", "tag"),
    [(math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf")],
)
def test_nonfinite_is_disclosed_not_silently_dropped(bad: float, tag: str) -> None:
    """換成 null 的同時必須留下 ``__nonfinite`` 標記，否則「發散」變成無聲缺漏。"""
    got = json.loads(dumps_record({"loss": bad}))
    assert got["loss"] is None
    assert got[f"loss{NONFINITE_SUFFIX}"] == tag


def test_finite_values_pass_through_bit_exact() -> None:
    """有限值不得被改動——淨化層不能順手改數字。"""
    rec = {"a": 1.0, "b": -3.25, "c": 0.0, "d": 1e-30, "e": 6.02e23}
    got = json.loads(dumps_record(rec))
    assert got == rec
    assert NONFINITE_SUFFIX not in dumps_record(rec)


def test_nested_structures_are_sanitized() -> None:
    """巢狀 dict／list 內的非有限值也要處理（metrics record 有巢狀欄位）。"""
    rec = {
        "outer": {"inner": math.nan, "fine": 2.0},
        "curve": [1.0, math.inf, 3.0],
        "deep": [{"x": -math.inf}],
    }
    got = json.loads(dumps_record(rec), parse_constant=lambda t: pytest.fail(t))
    assert got["outer"]["inner"] is None
    assert got["outer"][f"inner{NONFINITE_SUFFIX}"] == "nan"
    assert got["outer"]["fine"] == 2.0
    assert got["curve"] == [1.0, None, 3.0]
    assert got["deep"][0]["x"] is None


def test_bool_is_not_treated_as_float() -> None:
    """``bool`` 是 ``int`` 子類但不是浮點；不得被誤動。"""
    got = json.loads(dumps_record({"enabled": False, "flag": True}))
    assert got["enabled"] is False
    assert got["flag"] is True


def test_non_serializable_still_falls_back_to_str() -> None:
    """原本的 ``default=str`` 行為保留（record 內混有 Path／tensor 等物件）。"""
    got = json.loads(dumps_record({"p": Path("a/b")}))
    assert isinstance(got["p"], str)


def test_existing_marker_key_is_not_overwritten() -> None:
    """使用者若已有同名的 ``__nonfinite`` 鍵，既有值優先（不得被標記覆蓋）。"""
    rec = {"loss": math.nan, f"loss{NONFINITE_SUFFIX}": "使用者自己寫的"}
    got = json.loads(dumps_record(rec))
    assert got[f"loss{NONFINITE_SUFFIX}"] == "使用者自己寫的"


def test_sanitize_does_not_mutate_input() -> None:
    """淨化不得就地修改 record——呼叫端可能還要用同一個 dict。"""
    rec = {"loss": math.nan, "nested": {"x": math.inf}}
    sanitize_record(rec)
    assert math.isnan(rec["loss"])
    assert math.isinf(rec["nested"]["x"])


def test_dumps_is_single_line() -> None:
    """必須是單行（line-delimited JSON 的前提）。"""
    assert "\n" not in dumps_record({"a": 1, "b": {"c": [1, 2]}})


# ---------------------------------------------------------------------------
# 2. 四個寫入點都必須走淨化層
# ---------------------------------------------------------------------------

def _writes_to_metrics_handle(node: ast.Call) -> bool:
    """判斷一個 ``X.write(...)`` 呼叫的 ``X`` 是否為 metrics.json 的 handle。

    涵蓋 ``ctx.metrics_file.write``、``self._metrics_fp.write``，以及先把 handle
    取進區域變數再寫的形式（``fp = ctx.metrics_file`` → ``fp.write``）——後者
    無法只靠語法判斷，故由 :func:`_local_handle_aliases` 補上別名。
    """
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "write"):
        return False
    target = ast.unparse(node.func.value)
    return any(h in target for h in METRICS_HANDLES)


def _local_handle_aliases(tree: ast.AST) -> set[str]:
    """找出 ``x = <...metrics handle...>`` 這種區域別名的變數名。

    ★這是繞過本測試最明顯的方式（把 handle 存進區域變數再 write），
    ``pool_step.py`` 原本就是這個形狀（``fp = getattr(ctx, "metrics_file", None)``）。
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            src = ast.unparse(node.value)
            if any(h in src for h in METRICS_HANDLES):
                aliases.add(tgt.id)
    return aliases


def _value_aliases(tree: ast.AST) -> dict[str, str]:
    """收集 ``name = <expr>`` 的一層別名（``{變數名: expr 原始碼}``）。

    ★需要這層是因為寫入點常寫成兩行：``line = dumps_record(rec)`` 然後
    ``fp.write(line + "\\n")``。只看 write 的引數會誤判成「繞過淨化層」。
    同名多次賦值時全部串接，只要有任何一次經過淨化就算數（保守方向：
    寧可漏報也不要因寫法差異假紅——真正的把關是下面的計數測試）。
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                out[tgt.id] = out.get(tgt.id, "") + " " + ast.unparse(node.value)
    return out


def _expand(arg_src: str, aliases: dict[str, str]) -> str:
    """把引數原始碼裡出現的區域別名展開一層。"""
    parts = [arg_src]
    for name in re.findall(r"[A-Za-z_]\w*", arg_src):
        if name in aliases:
            parts.append(aliases[name])
    return " ".join(parts)


def _scan_metrics_writers() -> dict[Path, list[tuple[int, str]]]:
    """全 repo 掃出所有 metrics.json 寫入點。

    Returns:
        ``{檔案: [(行號, 展開別名後的引數原始碼), ...]}``。
    """
    found: dict[Path, list[tuple[int, str]]] = {}
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        handles = _local_handle_aliases(tree)
        values = _value_aliases(tree)
        hits: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute) and node.func.attr == "write"
            ):
                continue
            target = ast.unparse(node.func.value)
            if _writes_to_metrics_handle(node) or target in handles:
                arg_src = " ".join(ast.unparse(a) for a in node.args)
                hits.append((node.lineno, _expand(arg_src, values)))
        if hits:
            found[path] = hits
    return found


def test_every_metrics_writer_goes_through_dumps_record() -> None:
    """**全 repo** 的 metrics.json 寫入點都必須走淨化層。

    不再靠人工清單——人工清單正是上一版漏掉 ``pool_step.py`` 與 ``dy2_step.py``
    的原因，而「四個寫入點全部接上」那句宣稱因此通過了自己的測試。
    """
    found = _scan_metrics_writers()
    offenders: list[str] = []
    for path, hits in found.items():
        for lineno, arg_src in hits:
            if "dumps_record" not in arg_src:
                offenders.append(
                    f"{path.relative_to(SCAN_ROOT)}:{lineno}: {arg_src[:100]}"
                )
    assert not offenders, (
        "以下 metrics.json 寫入點繞過淨化層（裸 json.dumps 會寫出非法 JSON 的 "
        "NaN／Infinity 裸 token）：\n  " + "\n  ".join(offenders)
    )


def test_scan_found_the_expected_writer_count() -> None:
    """守住掃描本身：掃到的寫入點數必須符合實測，太少代表掃描條件壞了。

    實測 4 處（trainer×2、trainer_base、checkpointer）。★原 6 處，pool_step/dy2_step 已歸檔。
    新增寫入點時本測試會紅，提醒同步確認它有走淨化層。
    """
    found = _scan_metrics_writers()
    total = sum(len(v) for v in found.values())
    names = sorted(p.name for p in found)
    # ★2026-08-05：pool_step / dy2_step 隨 dead arms 歸檔，寫入點由 6 降為 4。
    assert total >= 4, f"只掃到 {total} 處寫入點（檔案 {names}），掃描條件可能壞了"
    for expected in (
        "trainer.py", "trainer_base.py", "checkpointer.py",
    ):
        assert expected in names, f"{expected} 未被掃到"


# ---------------------------------------------------------------------------
# 3. numpy／torch 標量（三輪 must_fix：allow_nan=False 對它們結構性失效）
# ---------------------------------------------------------------------------

def test_numpy_scalar_nonfinite_is_caught() -> None:
    """numpy 標量的非有限值必須與 Python float 同等處理。

    ★舊行為：``np.float32('nan')`` → ``{"x": "nan"}``（字串），不 raise、不標記。
    根因是 ``default=str`` 在 ``allow_nan=False`` 之前就把它吃掉了——也就是說
    那道「安全網」對非 Python-float 型別結構性失效。
    """
    np = pytest.importorskip("numpy")
    for dtype in (np.float16, np.float32, np.float64):
        got = json.loads(dumps_record({"x": dtype("nan")}))
        assert got["x"] is None, f"{dtype.__name__} 的 nan 未被攔截"
        assert got[f"x{NONFINITE_SUFFIX}"] == "nan"
    got = json.loads(dumps_record({"x": np.float32("inf")}))
    assert got["x"] is None and got[f"x{NONFINITE_SUFFIX}"] == "inf"


def test_numpy_scalar_finite_stays_a_number_not_a_string() -> None:
    """有限的 numpy 標量必須是 JSON 數值，不得被 ``default=str`` 變成字串。

    ★這比 NaN 那條影響更廣：任何 numpy 標量指標先前都以**字串**落進
    metrics.json（``np.float32(1.5)`` → ``"1.5"``），下游做數值比較會全錯。
    """
    np = pytest.importorskip("numpy")
    got = json.loads(dumps_record({"a": np.float32(1.5), "b": np.int64(7)}))
    assert got["a"] == 1.5 and isinstance(got["a"], float)
    assert got["b"] == 7 and isinstance(got["b"], int)


def test_torch_zero_dim_tensor_is_caught() -> None:
    """0 維 torch tensor（訓練 record 最常見的來源）同樣要處理。

    ★舊行為：``torch.tensor(nan)`` → ``{"x": "tensor(nan)"}``，連 repr 都混進去。
    """
    torch = pytest.importorskip("torch")
    got = json.loads(dumps_record({"x": torch.tensor(float("nan"))}))
    assert got["x"] is None and got[f"x{NONFINITE_SUFFIX}"] == "nan"
    got = json.loads(dumps_record({"x": torch.tensor(2.5)}))
    assert got["x"] == 2.5


def test_multi_dim_tensor_still_falls_back_to_str() -> None:
    """非 0 維 tensor 不是標量，維持 ``default=str`` 既有行為（不假裝處理得了）。"""
    torch = pytest.importorskip("torch")
    got = json.loads(dumps_record({"x": torch.tensor([1.0, 2.0])}))
    assert isinstance(got["x"], str)


def test_nonstring_key_nonfinite_is_still_disclosed() -> None:
    """key 不是字串時掛不上 ``<key>__nonfinite``，必須改用集中欄位揭露。"""
    got = json.loads(dumps_record({1: float("nan"), "ok": 2}))
    assert got["1"] is None
    assert got[NONFINITE_SUFFIX] == [{"key": "1", "was": "nan"}]


# ---------------------------------------------------------------------------
# 4. round-trip 還原（★第四輪：GPU 實跑抓到的迴歸）
# ---------------------------------------------------------------------------

def test_sanitize_restore_round_trip_is_lossless() -> None:
    """``restore_record(sanitize_record(x))`` 必須還原成 ``x``。

    ★這不是理論需求：``metrics.json`` 的 startup event 內含 config dump，
    ``run_async_eval_generation.py`` 會把它讀回來寫成 ``async_eval_config.yaml``
    再 ``load_tes_config``。淨化把 ``TESTrainConfig`` 四個以 ``float("nan")``
    當哨兵的欄位（tempw/jacw）變成 ``None``，config round-trip 因此把 NaN 換成
    ``None``，且多出的 ``__nonfinite`` 鍵讓載入器噴「未知欄位」——GPU 實跑的
    ``async_eval.log`` 直接印出了這四個鍵。標記存在的目的就是讓這件事可逆。
    """
    orig = {
        "tempw_init": math.nan, "tempw_final": math.nan,
        "jacw_init": math.inf, "jacw_final": -math.inf,
        "lr": 3e-4, "name": "v28",
        "nested": {"x": math.nan, "y": 1.0},
        "flag": False,
    }
    got = restore_record(json.loads(dumps_record(orig)))

    assert math.isnan(got["tempw_init"]) and math.isnan(got["tempw_final"])
    assert got["jacw_init"] == math.inf
    assert got["jacw_final"] == -math.inf
    assert got["lr"] == 3e-4 and got["name"] == "v28" and got["flag"] is False
    assert math.isnan(got["nested"]["x"]) and got["nested"]["y"] == 1.0
    assert not any(NONFINITE_SUFFIX in k for k in got), "標記鍵未被移除"
    assert not any(NONFINITE_SUFFIX in k for k in got["nested"]), "巢狀標記鍵未被移除"


def test_restore_is_noop_on_clean_record() -> None:
    """沒有標記的紀錄過 restore 不得改變（避免誤動一般資料）。"""
    clean = {"a": 1.0, "b": None, "c": {"d": "x"}, "e": [1, 2]}
    assert restore_record(clean) == clean




def test_marker_collision_does_not_lose_disclosure() -> None:
    """標記鍵撞名時，既有值保留、揭露改掛 ``__dup``，兩筆都不得消失。"""
    got = json.loads(
        dumps_record({"loss": float("nan"), f"loss{NONFINITE_SUFFIX}": "使用者的"})
    )
    assert got[f"loss{NONFINITE_SUFFIX}"] == "使用者的"
    assert got[f"loss{NONFINITE_SUFFIX}__dup"] == "nan"


def test_round_trip_is_lossy_on_marker_collision() -> None:
    """釘住可逆性的**已知邊界**：標記鍵撞名時 round-trip 不可逆。

    ★第四輪訂正：``restore_record`` 的 docstring 原本讀來像無條件可逆，
    ``test_sanitize_restore_round_trip_is_lossless`` 也沒覆蓋這個情形。
    實測 ``{"loss": nan, "loss__nonfinite": nan}`` 還原後 ``loss`` 變 ``None``。

    該情形實務上不可達（只有本模組自己產生 ``__nonfinite`` 後綴的鍵），
    故選擇**揭露而非增加複雜度**——但不得宣稱無條件可逆。
    若哪天真的要支援，本測試會是規格的起點。
    """
    orig = {"loss": math.nan, f"loss{NONFINITE_SUFFIX}": math.nan}
    got = restore_record(sanitize_record(orig))
    assert got["loss"] is None, (
        "撞名情形已可逆——若這是刻意修好的，請同步改 restore_record 的 docstring"
    )
    assert f"loss{NONFINITE_SUFFIX}__dup" in got, "被擠開的標記應仍可見，不得靜默消失"
