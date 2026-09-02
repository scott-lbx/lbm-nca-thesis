"""loss 項白名單：計入 ``total`` 的項必須有 per-term 欄位（DEEP_AUDIT W4 should_fix S8）。

問題形狀：``mean_step`` 的 ``ret`` dict 是唯一的 per-term 觀測面，也是
``loss_breakdown.png`` 與所有事後診斷的資料來源。但沒有任何機制保證「一個項被加進
``total``」與「該項有 breakdown 欄位」同步——``rel_l2``（canonical 權重 1.5）就這樣
在 total 裡待了整段訓練期而完全不可觀測，且兩輪訂正的註解都只是**記下**這件事，
沒有任何東西會在下次有人再加一個項時發出聲音。

本檔的機制：以 AST 數出每個函式裡 ``loss = loss + …`` 形式的累加次數，與下方
:data:`LEDGER` 對帳。任何新增／刪除的項都會讓數字對不上而變紅，強迫作者回來這裡
宣告該項是**有記錄**還是**具名例外**（含理由）。這不保證欄位語義正確，但保證
「悄悄多一個不可觀測的項」不可能發生。
"""

from __future__ import annotations

import ast
from pathlib import Path

MEAN_STEP = (
    Path(__file__).resolve().parents[3]
    / "src" / "lbmnca" / "nca" / "tes" / "training" / "steps" / "mean_step.py"
)

#: 累加目標變數名（``loss`` 是各函式的主累加器，``l_p`` 是 primary 閉包的）。
ACCUM_NAMES = ("loss", "l_p")

#: 「這是初始化不是累加」的判定標記（出現在賦值右側即放行）。
INIT_MARKERS = ("new_zeros", "zeros(", "torch.tensor(", "_moment_w * l_m")

#: 每個函式的**直接**累加次數（不含巢狀函式），以及該函式的觀測狀態。
#:
#: 改動任一函式的項數時，必須同步更新此表並在 ``note`` 說明新項記在哪個欄位、
#: 或為何仍是例外。
LEDGER: dict[str, dict] = {
    "run": {
        "n_accum": 16,
        "observed": True,
        "note": (
            "canonical mean 支，16 個直接累加點（逐一以 AST 列舉核對）："
            "stay／vel_ssim／vort_ssim／hband／enstrophy／velgrad／band／"
            "directional_band／delta_focus／overflow／mass_divj／fp_reg／inv／"
            "spec／specmatch／jac。ret dict 逐項記錄。"
        ),
    },
    "_primary_field_loss": {
        "n_accum": 3,
        "observed": True,
        "note": (
            "run() 內的閉包（moment＋chuber＋vgg＋rel_l2）。chuber／vgg 由回傳值帶出、"
            "rel_l2 經 nonlocal 累加器帶出；moment 併入 `mse`。"
        ),
    },
    "_chain_stage_loss": {
        "n_accum": 12,
        "observed": False,
        "note": (
            "★具名例外：chain 支的 12 個項**全數**沒有 per-term 欄位，只有一個合計"
            "標量進 total。這不是 rel_l2 一項的問題，而是整條路徑缺觀測面；而 chain"
            "分支是以 0.5 機率抽樣的——mean 與 chain 兩支的 total 中位數實測差 3.03×，"
            "所以 total 的批間跳動有多少來自分支抽樣、多少來自模型，目前無法拆解。"
            "修法是讓本函式回傳 breakdown dict 並在 ret 內以 `chain_` 前綴併入，"
            "屬 ret schema 變動，須與 S3 的重訓一起做才不會產生半新半舊的 metrics.json。"
        ),
    },
    "_chain_contract_loss": {
        "n_accum": 2,
        "observed": False,
        "note": "★具名例外：同上，contract 兩項併入 chain 合計標量，無獨立欄位。",
    },
}


def _is_accum(child: ast.AST) -> bool:
    """判斷一個語句是否為對 :data:`ACCUM_NAMES` 的累加。

    涵蓋三種等價寫法——★三輪訂正（must_fix）：第一版只認 ``x = x + term``，
    實測 ``x += term`` 與 ``x = term + x`` **都不被計數**，而該檔 docstring 卻寫
    「保證『悄悄多一個不可觀測的項』不可能發生」。一行 ``loss += x`` 就能繞過，
    那道保證是假的。三種形式現在都算。
    """
    # 1) x += term
    if (
        isinstance(child, ast.AugAssign)
        and isinstance(child.op, ast.Add)
        and isinstance(child.target, ast.Name)
        and child.target.id in ACCUM_NAMES
    ):
        return True
    # 2) x = x + term  ／  3) x = term + x
    if (
        isinstance(child, ast.Assign)
        and len(child.targets) == 1
        and isinstance(child.targets[0], ast.Name)
        and child.targets[0].id in ACCUM_NAMES
        and isinstance(child.value, ast.BinOp)
        and isinstance(child.value.op, ast.Add)
    ):
        name = child.targets[0].id
        for side in (child.value.left, child.value.right):
            if isinstance(side, ast.Name) and side.id == name:
                return True
    return False


def _direct_accum_count(func: ast.FunctionDef) -> int:
    """數出該函式**本體**的累加次數（跳過巢狀函式）。"""
    count = 0

    def walk(node: ast.AST) -> None:
        nonlocal count
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # 巢狀函式各自記帳
            if _is_accum(child):
                count += 1
            walk(child)

    walk(func)
    return count


def _unexplained_accum_writes(func: ast.FunctionDef) -> list[int]:
    """找出「寫入累加變數但不是已知累加形式」的行號（不變量守衛）。

    ★沒有這道的話，改用 ``loss = _combine(loss, term)`` 或
    ``loss = sum([loss, term])`` 之類的寫法就能完全脫離帳本而不被察覺。
    初始化（``loss = x.new_zeros(())``／``loss = pred.new_zeros(())``）與
    註記為初值的賦值屬正常，由 :data:`INIT_MARKERS` 白名單放行。
    """
    bad: list[int] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            is_write = (
                isinstance(child, ast.Assign)
                and len(child.targets) == 1
                and isinstance(child.targets[0], ast.Name)
                and child.targets[0].id in ACCUM_NAMES
            ) or (
                isinstance(child, ast.AugAssign)
                and isinstance(child.target, ast.Name)
                and child.target.id in ACCUM_NAMES
            )
            if is_write and not _is_accum(child):
                src = ast.unparse(child.value)
                if not any(m in src for m in INIT_MARKERS):
                    bad.append(child.lineno)
            walk(child)

    walk(func)
    return bad


def _scan() -> dict[str, int]:
    """回傳 ``{函式名: 直接累加次數}``（只收有累加的函式）。"""
    tree = ast.parse(MEAN_STEP.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            n = _direct_accum_count(node)
            if n:
                out[node.name] = n
    return out


def test_no_undeclared_loss_accumulator_function() -> None:
    """不得出現 LEDGER 沒登記的累加函式（新開一條 loss 路徑必須先來這裡宣告）。"""
    found = _scan()
    assert set(found) == set(LEDGER), (
        f"LEDGER 與實際不符：多出 {set(found) - set(LEDGER)}，"
        f"少了 {set(LEDGER) - set(found)}"
    )


def test_accumulation_counts_match_ledger() -> None:
    """每個函式的項數必須與 LEDGER 一致。

    ★這就是 ``rel_l2`` 事件的回歸鎖：多加一個項而不更新此表 → 紅。
    """
    found = _scan()
    mismatched = [
        f"{name}: 實際 {found[name]} 項，LEDGER 記 {LEDGER[name]['n_accum']} 項"
        f"（{LEDGER[name]['note'][:40]}…）"
        for name in sorted(found)
        if found[name] != LEDGER[name]["n_accum"]
    ]
    assert not mismatched, (
        "loss 項數與白名單不符。若確實新增了項，請更新 LEDGER 並說明該項記在哪個 "
        "ret 欄位；若刻意不記錄，請寫成具名例外並說明理由：\n  "
        + "\n  ".join(mismatched)
    )


def test_observed_functions_expose_rel_l2_field() -> None:
    """``run`` 的 ret dict 必須含 ``rel_l2`` 欄位（權重 1.5 的項不得再不可觀測）。"""
    src = MEAN_STEP.read_text(encoding="utf-8")
    assert '"rel_l2": loss_rel_l2' in src, "rel_l2 又從 ret dict 消失"
    assert '"overflow": loss_overflow' in src
    assert '"fp_reg": loss_fp_reg' in src


def test_every_unobserved_function_has_a_written_reason() -> None:
    """具名例外必須附理由，且理由要夠長到說明白（避免用空字串規避）。"""
    for name, entry in LEDGER.items():
        if entry["observed"]:
            continue
        assert len(entry["note"]) >= 40, f"{name} 的例外理由過短"
        assert "例外" in entry["note"], f"{name} 未標明是具名例外"


def test_all_three_accumulation_forms_are_counted() -> None:
    """三種等價累加寫法都必須被計數（★三輪 must_fix 的回歸鎖）。

    第一版只認 ``x = x + term``，``x += term`` 與 ``x = term + x`` 都能悄悄
    加項而完全不被察覺——而該檔 docstring 宣稱「保證不可能發生」。
    """
    src = (
        "def f():\n"
        "    loss = t.new_zeros(())\n"
        "    loss = loss + a\n"      # 形式 1
        "    loss += b\n"            # 形式 2
        "    loss = c + loss\n"      # 形式 3
        "    return loss\n"
    )
    fn = ast.parse(src).body[0]
    assert _direct_accum_count(fn) == 3, "三種寫法未被同等計數"


def test_unexplained_write_to_accumulator_is_flagged() -> None:
    """繞過三種形式（例如 ``loss = _combine(loss, x)``）必須被不變量守衛抓到。"""
    ok = ast.parse(
        "def f():\n    loss = t.new_zeros(())\n    loss = loss + a\n    return loss\n"
    ).body[0]
    assert _unexplained_accum_writes(ok) == []

    bad = ast.parse(
        "def f():\n"
        "    loss = t.new_zeros(())\n"
        "    loss = _combine(loss, a)\n"
        "    return loss\n"
    ).body[0]
    assert _unexplained_accum_writes(bad), "脫離帳本的寫法未被抓到"


def test_production_code_has_no_unexplained_accumulator_writes() -> None:
    """生產碼本身不得出現脫離三種形式的累加器寫入。"""
    tree = ast.parse(MEAN_STEP.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in LEDGER:
            for lineno in _unexplained_accum_writes(node):
                offenders.append(f"{node.name} @ mean_step.py:{lineno}")
    assert not offenders, (
        "以下位置寫入了 loss 累加器但不是已知的累加形式，帳本無法涵蓋：\n  "
        + "\n  ".join(offenders)
    )


def test_recorded_terms_reach_the_plot_whitelist() -> None:
    """記進 ret dict 的 per-term 欄位必須畫得出來（★第四輪 must_fix）。

    先前 W4 宣稱「補上 per-term 欄位」，但 ``src/plots/train/loss_breakdown.py``
    的 ``_MEAN_MODULE_KEYS`` 沒收 ``overflow``／``fp_reg``／``rel_l2``——欄位進了
    metrics.json 卻到不了使用者唯一會看的那張圖，等於補了半套。
    另一條 ``key.startswith("w_")`` 的納入路徑走不通：canonical metrics.json 的
    ``w_`` 鍵集合實測為空集合。
    """
    plot_src = (
        Path(__file__).resolve().parents[3]
        / "src" / "lbmnca" / "nca" / "plots" / "train" / "loss_breakdown.py"
    ).read_text(encoding="utf-8")
    for key in ("overflow", "fp_reg", "rel_l2"):
        assert f'"{key}"' in plot_src, (
            f"{key} 有 ret 欄位但不在 loss_breakdown 白名單中，畫不出來"
        )


def test_ledger_name_referenced_in_source_actually_exists() -> None:
    """程式碼註解引用的白名單名稱必須真的存在（★三輪 must_fix）。

    ``mean_step.py`` 曾寫「已列為 ``LOSS_TERM_LEDGER`` 的具名例外（見模組頂部）」
    ——全 repo 只有那行註解本身命中該字串，真正的白名單叫 ``LEDGER``、
    在本測試檔而不是模組頂部。指路指到不存在的地方比不指路更糟。
    """
    src = MEAN_STEP.read_text(encoding="utf-8")
    assert "LOSS_TERM_LEDGER" not in src, (
        "mean_step.py 又引用了不存在的 LOSS_TERM_LEDGER；正確名稱是本檔的 LEDGER"
    )
    if "LEDGER" in src:
        assert "test_loss_breakdown_ledger" in src, (
            "引用 LEDGER 時必須同時指出它在 tests/tes/test_loss_breakdown_ledger.py"
        )
