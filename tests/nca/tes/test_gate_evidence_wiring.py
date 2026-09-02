"""`L0-c` 回歸守衛：gate 證據管線的接線不得再度斷掉。

★ 為什麼需要這個測試（DEEP_AUDIT_2026-07 v2 `L0-c`）
------------------------------------------------------
`trainer.py` 一直呼叫 ``self._cfg.gate_thresholds``，（`validation/validator.py` 原為平行副本，W4 三輪已刪除死碼）
但該欄位**從未被定義** → 每次 AttributeError，而呼叫被包在
``except Exception: logger.debug(...)`` 裡被靜默吞掉，
緊接其後的 ``save_vis_gate_detailed(output_dir / "eval_output", ...)`` 從未執行。

實測後果：``runs/LBM_NCA-8/`` 全部 run（含論文引用的 canonical ``v28_psilp08``）
**零個**有 ``eval_output/``；全 ``runs/`` 樹最新的 ``eval_output/`` 停在 2026-07-09。
per-case gate 證據管線自該日起就是死的。

這正是稽核 [E3]「六組論文數字與其自稱來源互斥、三值全 workspace 已無存活來源」
的機制 —— **不是被同名覆寫，是根本沒有被寫出來過**。

本檔把「靜默斷線」變成「測試失敗」。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from lbmnca.nca.tes.config._legacy import TESTrainConfig
from lbmnca.nca.tes.eval.case_metrics import DEFAULT_GATE_THRESHOLDS, evaluate_gate

_SRC = Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca" / "tes"


def test_gate_thresholds_field_exists_on_the_config_the_trainer_actually_holds() -> None:
    """★ 核心斷言：`TESTrainConfig` 必須有 `gate_thresholds`。

    這是 `trainer.py` / `validator.py` 實際存取的那個 config。
    """
    cfg = TESTrainConfig()
    assert hasattr(cfg, "gate_thresholds"), (
        "TESTrainConfig 缺 gate_thresholds —— 呼叫端會 AttributeError，"
        "而該錯誤被 defensive except 吞掉，導致 eval_output/ 永遠不產生"
    )
    assert cfg.gate_thresholds is None, "預設須為 None（＝用 DEFAULT_GATE_THRESHOLDS）"


def test_none_is_a_valid_thresholds_argument() -> None:
    """預設值 `None` 必須是 `evaluate_gate` 接受的輸入，否則補欄位沒有意義。"""
    sig = inspect.signature(evaluate_gate)
    assert "thresholds" in sig.parameters
    assert sig.parameters["thresholds"].default is None


def test_default_thresholds_are_non_empty() -> None:
    """`None` 走的預設閾值表不得為空 —— 空表等於 gate 沒有判準。"""
    assert DEFAULT_GATE_THRESHOLDS, "DEFAULT_GATE_THRESHOLDS 不得為空"
    assert all(isinstance(v, (int, float)) for v in DEFAULT_GATE_THRESHOLDS.values())


@pytest.mark.parametrize(
    "rel_path",
    ["training/trainer.py"],  # ★W4 三輪：validation/validator.py 為零實例化死碼已刪除
)
def test_every_cfg_attribute_in_the_gate_evidence_block_exists(rel_path: str) -> None:
    """呼叫端在 gate 證據區塊存取的每個 `self._cfg.<x>`，都必須真的存在。

    這條比只檢查 `gate_thresholds` 更強：它擋的是**同一類**錯誤再次發生
    （新增一個 config 欄位的存取，卻忘了在 dataclass 定義）。
    """
    # ⚠ 必須用 AST 而非 regex：regex 會掃到 docstring／註解裡提及的欄位名，
    #   產生看起來很真實的假陽性（本測試初版即被自己的說明文字騙過一次）。
    tree = ast.parse((_SRC / rel_path).read_text(encoding="utf-8"))
    attrs = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_cfg"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    }
    cfg = TESTrainConfig()
    missing = sorted(a for a in attrs if not hasattr(cfg, a))
    assert not missing, (
        f"{rel_path} 存取了 TESTrainConfig 沒有的欄位：{missing}。"
        " 這類存取一旦落在 defensive except 內就會被靜默吞掉。"
    )


@pytest.mark.parametrize(
    "rel_path",
    ["training/trainer.py"],  # ★W4 三輪：validation/validator.py 為零實例化死碼已刪除
)
def test_gate_evidence_failure_is_not_logged_at_debug(rel_path: str) -> None:
    """證據檔寫入失敗**不得**只記在 DEBUG —— 那正是它死了 19 天沒人發現的原因。"""
    text = (_SRC / rel_path).read_text(encoding="utf-8")
    assert 'logger.debug("Per-case metrics failed' not in text, (
        f"{rel_path} 仍以 logger.debug 記錄證據檔寫入失敗；"
        " 須為 warning 以上，否則失敗不可見"
    )
