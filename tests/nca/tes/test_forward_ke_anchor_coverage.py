"""ke_anchor 漏接掃描器 —— DEEP_AUDIT R3-G1（防再長出新的漏接點）。

背景：``h3t_streamcollide`` 在 ``ke_anchor_mode='ceiling'`` 且 ``ke_anchor is None``
時直接 ``RuntimeError``（Stage 3.6）。canonical config 正是 ``ceiling``，因此**任何**
自寫 rollout 迴圈卻不傳 ``ke_anchor`` 的推論 harness，在 canonical 下要嘛直接崩、要嘛
（若改 ``off``）跑的是「無修正器」算子——與部署不是同一個算子。

用 :mod:`ast` 而非 regex：regex 會把註解、docstring、字串裡的 ``ke_anchor`` 算成命中，
正好朝「看起來有傳」的方向失效。

**掃描範圍（``_in_scope``）**：

    1. :data:`SCOPE_DIRS` 所列的**整個目錄**（推論／評估層），不論檔案有沒有被人記得
       加進白名單；
    2. 任何 import ``_v4_infer`` / ``V4Infer`` 的檔案（新製圖腳本自動納入）；
    3. :data:`MEAN_HARNESS` 明列的個別 02 scripts / 論文製圖檔。

★ 2026-07-28 二輪修正：原版的 ``_in_scope`` 只認「手寫白名單 + 用 V4Infer 的檔案」，
剛好只涵蓋當時已修好的檔案，於是 ``src/tes/eval/`` 與 ``src/tes/inference/`` 七個真實
漏接點全落在盲區、報 ``missing=0``。範圍改為「先畫目錄、再以具名 ALLOWLIST 減項」，
使「忘了把新檔加進清單」不再等於「掃描器看不到」。

排除範圍：``src/tes/archive/``、``__pycache__``（見 :data:`EXCLUDE_PARTS`），
以及 :data:`ALLOWLIST` 逐筆具名並附理由者。

**本掃描器不宣稱的事**：

    * 它只看 :data:`ROOTS` × :data:`SCOPE_DIRS` 之內的檔案，``missing=0`` 只代表「這些
      目錄內沒有漏接」，不代表全 workspace 無漏接；訓練層（``src/tes/training/``）與
      turb 專用 harness 不在範圍內。
    * 它**只查 ``ke_anchor``**，對 ``apply_ic_mask`` 一字未證——IC 政策的護欄是另一支
      ``test_ic_mask_policy_coverage.py``，兩者的 ``missing=0`` 不可互相借用。
    * 它是靜態掃描：只看「有沒有寫出來」，不看「傳進去的值是不是 None／空 dict」
      （見 :func:`_has_anchor` 的已知漏洞段）。

CLI：``python tests/tes/test_forward_ke_anchor_coverage.py``（列出全部範圍內呼叫點）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
# 單一 repo，相對路徑一律以 repo 根表示。
WORKSPACE = REPO_ROOT

# 掃描根目錄（存在才掃）。
ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    WORKSPACE / "lbmnca.bridge" / "src",
)

# 整目錄納入掃描的推論／評估層（相對 workspace 的 POSIX 路徑前綴）。
# 目錄制優先於白名單制：新增檔案自動被看見，要排除必須在 ALLOWLIST 具名。
SCOPE_DIRS: tuple[str, ...] = (
    "src/lbmnca/nca/tes/eval/",
    "src/lbmnca/nca/tes/inference/",
    "src/lbmnca/nca/inference/",
    "src/lbmnca/bridge/integration/adapters/",
)

# 不在 SCOPE_DIRS 之下、但同樣受本規則管的 mean 線 harness，逐檔明列。
# 只收本 repo 實際存在的檔案：一條指向不存在路徑的規則看起來像有在保護什麼，
# 實際上什麼都沒掃到，比沒有這條規則更糟。
MEAN_HARNESS: frozenset[str] = frozenset({
    "src/lbmnca/nca/inference/backends/tes_backend.py",
})

# 已知且**有理由**暫不修的呼叫點 ``(相對路徑, 呼叫文字, 所在函式 qualname)``。
# 新增豁免必須在此寫明理由，不得靜默擴大掃描盲區。
#
# ★2026-07-28 四輪收緊（原為 ``(相對路徑, 關鍵字)`` 且以**子字串**比對）：舊寫法的
# ``"ad.model"`` 會連帶豁免同檔任何文字含 ``ad.model`` 的 forward——例如日後若有人寫
# ``self._mean_ad.model(...)``（mean 線、本該被管），也會被同一筆靜默放行。實測目前
# 兩檔都沒有這種呼叫（mean 步走 ``self._mean_ad.step(1)``），所以還沒被觸發，但那是
# 運氣不是護欄。現在改為三元組＋**完全相等**比對，並多帶一個「所在函式」限定：
# 換函式、換寫法都會讓豁免失效而重新變紅（比行號穩定，不隨上方增刪行漂移）。
ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset({
    # ★2026-07-28 三輪：randomwalk_v4.py 的豁免已移除 —— 該檔已改走 inf.rollout
    #   （內部依協定取 ke_anchor）＋ inf.chain_handoff，不再自寫 forward 迴圈。
    # ── 以下兩筆為 turb 線（非 mean 三態協定治理範圍）──
    # turb_long_rollout：turb_module 長 rollout（傳 mean_flow_norm）。turb config 的
    # ke_anchor_mode 為 'off'，且 turb arch（turb_perception_a）的 forward 不收
    # ke_anchor；硬要求傳入只會製造假訊號。
    ("src/lbmnca/nca/tes/eval/turb_long_rollout.py", "turb_model", "rollout_case"),
    # turb_redesign_metrics：TurbReNCA 專用簽章 model(state, m_bar, g, p, mask, ...)，
    # 與 TES mean forward 不同族，無 ke_anchor 參數。
    ("src/lbmnca/nca/tes/eval/turb_redesign_metrics.py", "model", "free_rollout_fluct"),
    # ── 以下兩筆為 decompose 的 TURB 步（2026-07-28 三輪由「假展開」改為具名豁免）──
    # 二輪這兩處寫的是 `**anchor_kwargs`，而 anchor_kwargs 恆為 {}（兩個 prepare() 都不
    # 呼叫 _turb_ad.prepare()，_ke_anchor 恆 None）——等於用一個永遠為空的展開把呼叫點
    # 洗成綠燈。三輪已移除該死碼，改在此具名豁免：turb arch（turb_perception_a）的
    # forward 不收 ke_anchor，且 turb config 為 ke_anchor_mode='off'。
    # 兩處的 ``ad`` 都是同一函式內 ``ad = self._turb_ad`` 的區域別名（故限定
    # ``_step_one``；mean 步在別的函式、且走 ``self._mean_ad.step(1)``，不受本豁免涵蓋）。
    ("src/lbmnca/nca/inference/backends/decompose_backend.py", "ad.model",
     "DecomposeBackend._step_one"),
    ("src/lbmnca/bridge/integration/adapters/decompose_adapter.py", "ad.model",
     "DecomposeAdapter._step_one"),
})

# TES forward 專屬關鍵字引數（用來認出「這確實是一次 forward」）。
FORWARD_KWARGS: frozenset[str] = frozenset({
    "return_trajectory", "g_vec_norm", "mean_flow_norm", "ke_anchor",
    "tf_ratio", "gt_traj", "ss_detach", "delta_ctu_seq", "cumul_ctu_norm_seq",
})

# 第 4 個位置引數（n_steps）常見寫法：整數常數或這些變數名。
N_STEPS_NAMES: frozenset[str] = frozenset({
    "n", "n_steps", "k", "steps", "rollout_k", "rk", "preroll", "grad_steps",
})

# 排除目錄片段（歸檔碼與快取）。
EXCLUDE_PARTS: tuple[str, ...] = ("__pycache__", "archive", "_archive", "Archive")


def _rel(p: Path) -> str:
    return p.relative_to(WORKSPACE).as_posix()


def _func_text(node: ast.Call) -> str:
    try:
        return ast.unparse(node.func)
    except Exception:  # pragma: no cover - 僅防禦
        return "<unparse-failed>"


def _is_model_call(node: ast.Call) -> bool:
    """是否為「模型 forward」呼叫：名稱像 model/net/forward 且引數 >= 4。

    TES forward 簽章前四個位置引數為 ``(state, cond, fluid_mask, n_steps)``，故以 4 為
    門檻濾掉 ``model(x)`` 這類單引數呼叫。
    """
    t = _func_text(node)
    tail = t.rsplit(".", 1)[-1]
    looks_like_model = (
        tail in {"model", "net", "forward"}
        or tail.endswith("_model")
        or tail.startswith("model")
        or ".model" in t
    )
    if not looks_like_model:
        return False
    if len(node.args) + len(node.keywords) < 4:
        return False
    return _looks_like_forward_signature(node)


def _looks_like_forward_signature(node: ast.Call) -> bool:
    """引數形狀是否像一次 TES forward。

    僅靠名字會把 ``eval_model(name, paradigm, ...)`` 這類助手函式誤判成 forward。
    真正的 forward 要嘛帶專屬關鍵字引數，要嘛第 4 個位置引數是 ``n_steps``。
    """
    if any(kw.arg in FORWARD_KWARGS for kw in node.keywords):
        return True
    if len(node.args) >= 4:
        fourth = node.args[3]
        if isinstance(fourth, ast.Constant) and isinstance(fourth.value, int):
            return True
        if isinstance(fourth, ast.Name) and fourth.id in N_STEPS_NAMES:
            return True
    return False


def _uses_v4_infer(src: str) -> bool:
    return "_v4_infer" in src or "V4Infer" in src


def _in_scope(rel_path: str, src: str) -> bool:
    """該檔是否在掃描範圍內（目錄制 ∪ 明列 harness ∪ 用 V4Infer 者）。"""
    if any(rel_path.startswith(d) for d in SCOPE_DIRS):
        return True
    if rel_path in MEAN_HARNESS:
        return True
    return _uses_v4_infer(src)


def _has_anchor(node: ast.Call) -> bool:
    """該 forward 呼叫是否已帶錨點。

    兩種合格寫法：

        * 顯式 ``ke_anchor=...``；
        * ``**<name>`` 展開，且展開對象的名稱含 ``anchor``——各 arch 的 forward 簽章不
          一致（``flat`` / ``turb_perception_a`` 不收 ``ke_anchor``），泛用 rollout 函式
          只能用 ``**protocol.forward_anchor_kwargs(...)`` 的結果有條件傳入。

    ★這個啟發式的已知漏洞（2026-07-28 三輪記錄）：第二條**只看名字**，不驗證那個 dict
    有沒有可能非空。二輪的兩支 decompose TURB 步就是寫成 ``**anchor_kwargs`` 而該變數
    在該控制流下恆為 ``{}``（``prepare()`` 從不呼叫 ``_turb_ad.prepare()``），於是被判成
    OK。三輪已移除那段死碼並改列 :data:`ALLOWLIST`，但**啟發式本身沒有變**——任何人寫
    一個名字含 ``anchor`` 的空 dict 展開，這裡仍會給綠燈。要判定「錨點真的傳進去了」
    需要執行期證據，靜態掃描做不到。

    Args:
        node: 一個已判定為 forward 的 ``ast.Call``。

    Returns:
        是否視為已帶錨點。
    """
    for kw in node.keywords:
        if kw.arg == "ke_anchor":
            return True
        if kw.arg is None:                      # **kwargs 展開
            try:
                text = ast.unparse(kw.value)
            except Exception:                   # pragma: no cover - 僅防禦
                continue
            if "anchor" in text:
                return True
    return False


def _enclosing_funcs(tree: ast.AST) -> list[tuple[int, int, str]]:
    """列出所有函式定義的行區間與 qualname（含 class 前綴）。

    用於把 :data:`ALLOWLIST` 的豁免限定在「某函式內」，比行號穩定——上方增刪行不會
    讓豁免漂掉，但把呼叫搬到別的函式會讓它重新變紅。

    Args:
        tree: 已 parse 的模組 AST。

    Returns:
        ``[(起始行, 結束行, qualname)]``，未排序。
    """
    spans: list[tuple[int, int, str]] = []

    def _walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                qual = f"{prefix}{child.name}"
                end = child.end_lineno or child.lineno
                if not isinstance(child, ast.ClassDef):
                    spans.append((child.lineno, end, qual))
                _walk(child, f"{qual}.")
            else:
                _walk(child, prefix)

    _walk(tree, "")
    return spans


def _enclosing_of(spans: list[tuple[int, int, str]], lineno: int) -> str:
    """回傳 ``lineno`` 所在的最內層函式 qualname；不在任何函式內回 ``"<module>"``。"""
    best = "<module>"
    best_start = -1
    for start, end, qual in spans:
        if start <= lineno <= end and start > best_start:
            best, best_start = qual, start
    return best


def scan() -> list[tuple[str, int, str, bool, str]]:
    """掃描範圍內所有模型 forward 呼叫點。

    Returns:
        ``[(相對路徑, 行號, 呼叫文字, 是否帶錨點, 所在函式 qualname)]``。
    """
    rows: list[tuple[str, int, str, bool, str]] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            try:
                src = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = _rel(path)
            if not _in_scope(rel, src):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            spans = _enclosing_funcs(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_model_call(node):
                    rows.append((rel, node.lineno, _func_text(node),
                                 _has_anchor(node),
                                 _enclosing_of(spans, node.lineno)))
    return rows


def missing() -> list[tuple[str, int, str]]:
    """範圍內未帶錨點的呼叫點（已扣除 ALLOWLIST）。

    ★豁免比對為 ``(路徑, 呼叫文字, 所在函式)`` 三者**完全相等**，不是子字串——見
    :data:`ALLOWLIST` 上方的四輪收緊說明。
    """
    out: list[tuple[str, int, str]] = []
    for rel, lineno, text, has, encl in scan():
        if has:
            continue
        if (rel, text, encl) in ALLOWLIST:
            continue
        out.append((rel, lineno, text))
    return out


# ──────────────────────────────────────────────────────────────────────
def test_scanner_sees_known_good_call_site():
    """反測：掃描器必須抓得到已知**有**傳錨點的呼叫點。

    否則「沒找到問題」只是掃描器沒在看（本檔最容易失效的方式）。
    """
    rows = scan()
    assert rows, "掃描器在此環境找不到任何模型 forward 呼叫點（範圍或啟發式失效）"
    ok_sites = {rel for rel, _ln, _t, has, _encl in rows if has}
    assert "src/lbmnca/nca/inference/backends/tes_backend.py" in ok_sites, (
        "tes_backend 的 step() 已知會傳 ke_anchor，掃描器卻沒標為 OK → 掃描器失效"
    )


def test_scope_actually_covers_eval_and_inference_dirs():
    """反測二：掃描器必須真的看進 :data:`SCOPE_DIRS`，不能只看白名單。

    round-1 的失效模式不是啟發式抓不到呼叫，而是**範圍畫得太小**——白名單剛好只列
    已修好的檔案，於是 ``src/tes/eval/`` 的漏接點全部落在盲區而報 ``missing=0``。
    此測試以「這兩個目錄至少各有一個被掃到的 forward 呼叫」把範圍本身釘住。
    """
    rows = scan()
    seen = {rel for rel, _ln, _t, _has, _encl in rows}
    for prefix in ("src/lbmnca/nca/tes/eval/", "src/lbmnca/nca/tes/inference/"):
        assert any(rel.startswith(prefix) for rel in seen), (
            f"SCOPE_DIRS 內的 {prefix} 沒有任何 forward 呼叫被掃到 → 範圍或啟發式失效"
        )


def test_no_forward_call_missing_ke_anchor():
    """mean 線 harness 不得有未帶 ke_anchor 的 forward 呼叫（R3-G1）。"""
    bad = missing()
    detail = "\n".join(f"  {rel}:{ln}  {txt}" for rel, ln, txt in bad)
    assert not bad, (
        "以下 forward 呼叫未傳 ke_anchor（canonical ke_anchor_mode='ceiling' 下會 "
        f"RuntimeError，或在 'off' 下跑到與部署不同的算子）：\n{detail}\n"
        "修法：先取 protocol.resolve_ke_anchor(cfg, state, fluid_mask)"
        "（V4Infer 用 inf.ke_anchor(state, fluid_mask)）再以 ke_anchor= 傳入。"
    )


if __name__ == "__main__":  # pragma: no cover - CLI 用法
    _rows = scan()
    print(f"in-scope forward calls = {len(_rows)}")
    for _path, _line, _call, _ok, _encl in _rows:
        print(f"  [{'OK  ' if _ok else 'MISS'}] {_path}:{_line}  {_call}  @{_encl}")
    _bad = missing()
    print(f"missing ke_anchor (ALLOWLIST 扣除後) = {len(_bad)}")
    for _path, _line, _call in _bad:
        print(f"  MISS {_path}:{_line}  {_call}")
    sys.exit(1 if _bad else 0)
