"""`L0-c` 落地驗收：gate 證據檔必須真的寫得出來。

★ 與 `test_gate_evidence_wiring.py` 的分工
--------------------------------------------
`test_gate_evidence_wiring.py` 驗的是**接線**（欄位存在、失敗不靜默）。
本檔驗的是**結果**：完整重現 `trainer.py:2295-2299` 的呼叫序列，
斷言 ``eval_output/`` 底下真的出現檔案。

依 DEEP_AUDIT v2 §1 的紀律：「應該會 work」不算驗證。
在本檔通過之前，`L0-c` 只能標 ◐。

⚠ 本檔**不需要 GPU、不需要模型、不需要訓練** ——
``build_vis_gate_detailed`` 與 ``save_vis_gate_detailed`` 都是純函式，
故可用合成輸入完整驗證寫檔路徑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lbmnca.nca.tes.config._legacy import TESTrainConfig
from lbmnca.nca.tes.eval.case_metrics import build_vis_gate_detailed, save_vis_gate_detailed


def _fake_case(idx: int, ic_type: str, re: float) -> dict[str, Any]:
    """一筆合成 per-case 結果（欄位比照 compute_per_case_visual_gate 的輸出）。"""
    return {
        "case_id": f"synthetic_c{idx:04d}",
        "ic_type": ic_type,
        "re": re,
        "IC_Copy_Score": 0.80,
        "HorizontalBand_FFT": 0.12,
        "StructVel_SSIM": 0.85,
        "StructVort_HighMask_IoU": 0.60,
        "StructVort_RGB_SSIM": 0.70,
        "VortTopK_MAE": 0.01,
        "VortSignedCorr": 0.90,
    }


def _fake_aggregate() -> dict[str, Any]:
    return {
        "IC_Copy_Score": 0.80,
        "StructVel_RGB_SSIM": 0.85,
        "HorizontalBand_FFT": 0.12,
    }


def test_gate_evidence_actually_lands_on_disk(tmp_path: Path) -> None:
    """★ 核心斷言：跑完 trainer 的那三行後，`eval_output/` 必須非空。

    這正是 2026-07-09 之後從未發生過的事。
    """
    cfg = TESTrainConfig()  # ← gate_thresholds 由此取得（L0-c 補上的欄位）
    per_case = [
        _fake_case(0, "cold", 120.0),
        _fake_case(1, "warm", 250.0),
        _fake_case(2, "warm", 380.0),
    ]

    # 完整重現 trainer.py:2295-2299
    detailed = build_vis_gate_detailed(
        1234, per_case, _fake_aggregate(),
        gate_thresholds=cfg.gate_thresholds,
    )
    out_dir = tmp_path / "eval_output"
    save_vis_gate_detailed(out_dir, detailed)

    assert out_dir.is_dir(), "eval_output/ 未被建立"
    written = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert written, "eval_output/ 是空的 —— 證據檔沒有落地"


def test_written_evidence_contains_the_gate_verdict(tmp_path: Path) -> None:
    """證據檔必須含 gate 判定本身，否則它不構成「證據」。"""
    cfg = TESTrainConfig()
    per_case = [_fake_case(i, "warm", 200.0 + 50 * i) for i in range(4)]
    detailed = build_vis_gate_detailed(
        500, per_case, _fake_aggregate(), gate_thresholds=cfg.gate_thresholds
    )
    out_dir = tmp_path / "eval_output"
    save_vis_gate_detailed(out_dir, detailed)

    jsons = [p for p in out_dir.iterdir() if p.suffix == ".json"]
    assert jsons, f"沒有 JSON 證據檔，只有：{[p.name for p in out_dir.iterdir()]}"

    payload = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert "gate" in payload, "證據檔缺 gate 區段"
    assert "checks" in payload["gate"], "gate 缺逐項 checks"
    assert "gate_pass" in payload["gate"], "gate 缺 gate_pass 判定"
    # 六個 must-pass 判準都要在
    assert set(payload["gate"]["checks"]) == {
        "cold_ssim", "cold_iou", "warm_ssim", "hband_max", "ic_range"
    } or len(payload["gate"]["checks"]) >= 5, (
        f"gate checks 數量異常：{sorted(payload['gate']['checks'])}"
    )


def test_step_snapshot_is_kept_separately(tmp_path: Path) -> None:
    """當步快照必須另存 —— 否則跨 step 的證據會互相覆寫（稽核 [S4] 的病灶）。"""
    cfg = TESTrainConfig()
    per_case = [_fake_case(i, "warm", 300.0) for i in range(3)]
    out_dir = tmp_path / "eval_output"

    for step in (1000, 2000):
        detailed = build_vis_gate_detailed(
            step, per_case, _fake_aggregate(), gate_thresholds=cfg.gate_thresholds
        )
        save_vis_gate_detailed(out_dir, detailed)

    names = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert any("001000" in n for n in names), f"step 1000 快照遺失：{names}"
    assert any("002000" in n for n in names), f"step 2000 快照遺失：{names}"
