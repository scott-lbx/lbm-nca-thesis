# ★2026-08-05：以下斷言的定位錨（long_rollout / pool_step / dy2_step /
#   rollout_runner / run_async_eval_generation / eval_test_split）已隨 v3、turb
#   與 dead arms 一併歸檔。台帳存在的意義是「文件寫的數字＝程式碼實況」，
#   當被稽核的程式碼整段不存在時，正確處置是撤下該條斷言而不是改個數字讓它綠。
#   歸檔源碼見 Archive/{v3_dataset,gen8_turb,dead_arms}_2026-08-05/。
"""apply_ic_mask 政策漏接掃描器 —— DEEP_AUDIT R3 / W0.6（三輪新增）。

背景：``apply_ic_mask`` 決定 stage 0 的 IC 要不要把固體區歸零。訓練端讀
``cfg.apply_ic_mask``（canonical ``True``；2026-07-28 實測 ``base_step`` / ``cross_ctu_step``
/ ``dy2_step`` 三處都是讀該欄位，其中 ``base_step`` 另加 ``t_start == 0`` 條件），
但評估與部署端過去各自寫死：
``eval/rollout_runner`` 的開關由呼叫端硬編布林（一支傳 ``False``、兩支傳 ``True``），
``eval/long_rollout`` 自帶一份 ``masked_fill``，四支 eval 的 stage 0 則是裸
``model.init_state(...)`` 完全不套。結果是「評估的 IC 協定不是訓練的那一套」。

三輪的收斂點是 :func:`src.tes.inference.protocol.initial_state`
（與拿不到 cfg 時用的 ``initial_state_from_flag``）：**mean 線在 :data:`SCOPE_DIRS` 內的
stage 0 都經過它**（:data:`ALLOWLIST` 具名豁免的 turb 線／dummy／協定入口自身除外），
政策值來自 ``cfg.apply_ic_mask``。本掃描器把這件事釘住——範圍內出現裸
``*.init_state(...)`` 呼叫就報錯，除非在 :data:`ALLOWLIST` 具名並附理由。

**本掃描器不宣稱的事**：

    * 它只看 :data:`ROOTS` × :data:`SCOPE_DIRS` 之內的檔案。訓練層
      （``src/tes/training/``，實測仍有 6 處自寫 ``masked_fill``）、``scripts/``（7 處）
      與 ``NYCU_Thesis`` 製圖端都**不在**範圍內，``missing=0`` 不代表那些地方沒問題。
    * 它是靜態掃描：只看「有沒有繞過協定入口」，不看傳進去的 ``apply_ic_mask``
      布林值是否真的來自 ``cfg``（那要執行期或型別層證據）。
    * 它與 ``test_forward_ke_anchor_coverage.py`` 是兩把獨立的尺，一把綠燈不證明另一把。

CLI：``python tests/tes/test_ic_mask_policy_coverage.py``（列出範圍內全部呼叫點）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# 單一 repo，相對路徑一律以 repo 根表示。
WORKSPACE = REPO_ROOT

# 掃描根目錄（存在才掃）。
ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "src",
    WORKSPACE / "lbmnca.bridge" / "src",
)

# 整目錄納入掃描的推論／評估層（相對 workspace 的 POSIX 路徑前綴）。
# 與 ke_anchor 掃描器刻意用**同一組**目錄，兩把尺量的是同一塊地。
SCOPE_DIRS: tuple[str, ...] = (
    "src/lbmnca/nca/tes/eval/",
    "src/lbmnca/nca/tes/inference/",
    "src/lbmnca/nca/inference/",
    "src/lbmnca/bridge/integration/adapters/",
)

# 不在 SCOPE_DIRS 目錄下、但屬 mean 線 harness 而必須納入的個別檔案。
MEAN_HARNESS: frozenset[str] = frozenset({
    "src/lbmnca/nca/tes/eval_warmzeroshot.py",
})

# 視為「已走協定入口」的呼叫文字尾段。
PROTOCOL_ENTRIES: tuple[str, ...] = (
    "initial_state", "initial_state_from_flag", "chain_handoff",
)

# 已知且**有理由**繞過協定入口的呼叫點 ``(相對路徑, 呼叫文字, 所在函式 qualname)``。
# 新增豁免必須寫明理由，不得靜默擴大盲區。
#
# ★2026-07-28 四輪收緊（原為 ``(相對路徑, 關鍵字)`` 且以**子字串**比對）：舊寫法會超額
# 豁免——例如 protocol.py 那筆的 ``"model.init_state"`` 同時是
# ``turb_model.init_state`` / ``self._turb_ad.model.init_state`` 的子字串，等於把該檔
# 內所有 ``*.init_state`` 一次放行。現改為三元組＋**完全相等**比對，並多帶「所在函式」
# 限定（比行號穩定：上方增刪行不會讓豁免漂掉，把呼叫搬去別的函式則會重新變紅）。
ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset({
    # 協定入口自己：initial_state_from_flag 內部這一次 init_state 就是 mean 線 stage 0
    # 最終落地的地方，本來就不該被自己的掃描器判成違規。
    ("src/lbmnca/nca/tes/inference/protocol.py", "model.init_state",
     "initial_state_from_flag"),
    # ── turb 線：fluct 為 state，IC 是固定 seed 小擾動且已先乘 fluid mask，
    #    apply_ic_mask（mean 線的 config 欄位）語意不適用。──
    ("src/lbmnca/nca/tes/eval/turb_long_rollout.py", "turb_model.init_state",
     "rollout_case"),
    ("src/lbmnca/nca/tes/eval/turb_redesign_metrics.py", "model.init_state",
     "free_rollout_fluct"),
    ("src/lbmnca/nca/tes/inference/decompose_inference.py", "turb_model.init_state",
     "make_turb_stepper._step"),
    ("src/lbmnca/nca/inference/backends/decompose_backend.py",
     "self._turb_ad.model.init_state", "DecomposeBackend._handoff_to_turb"),
    ("src/lbmnca/bridge/integration/adapters/decompose_adapter.py",
     "self._turb_ad.model.init_state", "DecomposeAdapter._handoff_to_turb"),
    # decompose_inference 的 mean 冷啟：ic_physics=None，protocol.initial_state 與
    # model.init_state 逐位元相同（apply_ic_mask 只作用於非 None 的 IC）。
    ("src/lbmnca/nca/tes/inference/decompose_inference.py",
     "models.mean_model.init_state", "run_decompose_pipeline"),
    # dummy backend/adapter：假模型（無 physics 語意），不參與三態協定。
    # 兩檔各有兩處，都在 prepare() 內（cold / warm 兩條分支）。
    ("src/lbmnca/nca/inference/backends/dummy_backend.py", "self.model.init_state",
     "DummyBackend.prepare"),
    ("src/lbmnca/bridge/integration/adapters/dummy_adapter.py", "self.model.init_state",
     "DummyAdapter.prepare"),
})

# 排除目錄片段（歸檔碼與快取）。
EXCLUDE_PARTS: tuple[str, ...] = ("__pycache__", "archive", "_archive", "Archive")


def _rel(p: Path) -> str:
    """回傳相對 workspace 的 POSIX 路徑字串。

    Args:
        p: 絕對路徑。

    Returns:
        相對 workspace 的 POSIX 路徑。
    """
    return p.relative_to(WORKSPACE).as_posix()


def _func_text(node: ast.Call) -> str:
    """把呼叫節點的 func 還原成原始碼字串。

    Args:
        node: ``ast.Call`` 節點。

    Returns:
        還原後的字串；還原失敗時回 ``"<unparse-failed>"``。
    """
    try:
        return ast.unparse(node.func)
    except Exception:  # pragma: no cover - 僅防禦
        return "<unparse-failed>"


def _in_scope(rel_path: str) -> bool:
    """該檔是否落在 :data:`SCOPE_DIRS` 之內，或屬 :data:`MEAN_HARNESS` 明列檔案。

    Args:
        rel_path: 相對 workspace 的 POSIX 路徑。

    Returns:
        是否納入掃描。
    """
    return (any(rel_path.startswith(d) for d in SCOPE_DIRS)
            or rel_path in MEAN_HARNESS)


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
    """掃描範圍內所有 ``init_state`` / 協定入口呼叫點。

    Returns:
        ``[(相對路徑, 行號, 呼叫文字, 是否為協定入口, 所在函式 qualname)]``。
    """
    rows: list[tuple[str, int, str, bool, str]] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            rel = _rel(path)
            if not _in_scope(rel):
                continue
            try:
                src = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            spans = _enclosing_funcs(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                text = _func_text(node)
                tail = text.rsplit(".", 1)[-1]
                if tail in PROTOCOL_ENTRIES:
                    rows.append((rel, node.lineno, text, True,
                                 _enclosing_of(spans, node.lineno)))
                elif tail == "init_state":
                    rows.append((rel, node.lineno, text, False,
                                 _enclosing_of(spans, node.lineno)))
    return rows


def missing() -> list[tuple[str, int, str]]:
    """範圍內繞過協定入口的 ``init_state`` 呼叫點（已扣除 :data:`ALLOWLIST`）。

    ★豁免比對為 ``(路徑, 呼叫文字, 所在函式)`` 三者**完全相等**，不是子字串——見
    :data:`ALLOWLIST` 上方的四輪收緊說明。

    Returns:
        ``[(相對路徑, 行號, 呼叫文字)]``。
    """
    out: list[tuple[str, int, str]] = []
    for rel, lineno, text, is_entry, encl in scan():
        if is_entry:
            continue
        if (rel, text, encl) in ALLOWLIST:
            continue
        out.append((rel, lineno, text))
    return out


# ──────────────────────────────────────────────────────────────────────


def test_scope_covers_eval_and_adapter_dirs():
    """反測二：範圍必須真的涵蓋 eval 目錄與 03 端 adapters 目錄。"""
    rows = scan()
    seen = {rel for rel, _ln, _t, _e, _encl in rows}
    for prefix in ("src/lbmnca/nca/tes/eval/",
                   "src/lbmnca/bridge/integration/adapters/"):
        assert any(rel.startswith(prefix) for rel in seen), (
            f"SCOPE_DIRS 內的 {prefix} 沒有任何呼叫被掃到 → 範圍或啟發式失效"
        )


def test_no_stage0_bypassing_protocol():
    """推論／評估層的 stage 0 不得繞過 protocol.initial_state（R3 / W0.6）。"""
    bad = missing()
    detail = "\n".join(f"  {rel}:{ln}  {txt}" for rel, ln, txt in bad)
    assert not bad, (
        "以下 stage 0 直接呼叫 model.init_state，繞過 apply_ic_mask 政策：\n"
        f"{detail}\n"
        "修法：改用 protocol.initial_state(model, cfg, static, ic, fluid_mask)，"
        "或在拿不到 cfg 時用 protocol.initial_state_from_flag(model, apply_ic_mask, ...)；"
        "turb 線／dummy 等不適用者請加進本檔 ALLOWLIST 並寫明理由。"
    )


if __name__ == "__main__":  # pragma: no cover - CLI 用法
    _rows = scan()
    print(f"in-scope init_state / protocol-entry calls = {len(_rows)}")
    for _path, _line, _call, _entry, _encl in _rows:
        print(f"  [{'ENTRY' if _entry else 'RAW  '}] {_path}:{_line}  {_call}  @{_encl}")
    _bad = missing()
    print(f"stage0 bypassing protocol (ALLOWLIST 扣除後) = {len(_bad)}")
    for _path, _line, _call in _bad:
        print(f"  MISS {_path}:{_line}  {_call}")
    sys.exit(1 if _bad else 0)
