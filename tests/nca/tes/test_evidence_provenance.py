"""稽核證據檔的 provenance 契約（DEEP_AUDIT W4 第五輪）。

★為何需要（第五輪 synth 的跨條目裁決）：本 campaign 產出的 12 份證據 json
**全部沒有 provenance**——無 git commit、無 argv、無 eval_seed、無輸入清單。
對照 repo 自己已寫好的規範 ``scripts/run_mean_gate_eval.py``（有 ``--eval-seed``
且明文要求 seed 必須落盤、寫進 ``agg["eval_seed"]``），W4 的證據檔全部繞過了它。

後果是可量的：第五輪逐條複核時抓到**三處數字微漂與一處量級錯誤**，全部源於
「人工把數字轉寫進報告／註解」而非「腳本輸出」——

* 「後 6 點均值高於最佳 3.6%」→ 實算 3.5025%
* 「架構修正 mse +1.05%」→ 實算 +1.0683%
* 「每 2 小時 ~1700 → ~700（2.4×）」→ 實算 4199→3800→2500→700→600→300（14×）
* 「jac ratio ≈0.9998、算子不收縮」→ 量到的是**未訓練模型**；訓練後為 0.726、算子收縮

本檔把「新產出的證據檔必須自證來源」變成測試，而不是靠自律。純 CPU。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "docs" / "reviews" / "DEEP_AUDIT_2026-07" / "tools" / "_out"
)

#: provenance 至少要能回答「誰、用什麼、在哪個版本、對什麼輸入」。
REQUIRED_KEYS = ("git_commit", "argv", "generated_at", "inputs")

#: 既有檔案的豁免名單。★這些是本規範**建立之前**產出的，逐一列名而非用萬用字元——
#: 萬用字元豁免等於沒有規範。新增檔案不得加進來，要嘛帶 provenance、要嘛不要落盤。
GRANDFATHERED = frozenset({
    "convergence_analysis_2026-08-01.json",
    "gate_eval_v4_first_run_2026-08-01.json",
    "jac_gamma_probe_2026-08-01.json",
    "longrun_plateau_and_throughput_2026-08-02.json",
    "red_on_revert_2026-07-31.json",
    "red_on_revert_2026-08-01.json",
    "w4_review_round2_synth.json",
    "w4_round2_findings.json",
    "w4_round3_result.json",
    "w4_round3_open_items.json",
    "w4_round4_result.json",
    "w4_round5_input.json",
    "w4_round5_result.json",
})


def _evidence_files() -> list[Path]:
    if not OUT_DIR.is_dir():
        return []
    return sorted(OUT_DIR.glob("*.json"))


@pytest.mark.skipif(not _evidence_files(), reason="證據目錄不在此環境")
def test_new_evidence_files_carry_provenance() -> None:
    """豁免名單之外的證據檔必須帶 ``_provenance``，且含四個必要欄位。"""
    offenders: list[str] = []
    for p in _evidence_files():
        if p.name in GRANDFATHERED:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            offenders.append(f"{p.name}: 非合法 JSON（{exc}）")
            continue
        prov = d.get("_provenance") if isinstance(d, dict) else None
        if not isinstance(prov, dict):
            offenders.append(f"{p.name}: 缺 _provenance")
            continue
        missing = [k for k in REQUIRED_KEYS if k not in prov]
        if missing:
            offenders.append(f"{p.name}: _provenance 缺 {missing}")
    assert not offenders, (
        "以下證據檔無法自證來源（新檔不得加進 GRANDFATHERED）：\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.skipif(not _evidence_files(), reason="證據目錄不在此環境")
def test_grandfather_list_does_not_grow_silently() -> None:
    """豁免名單只能縮不能長——它是負債清單，不是後門。

    每一個名字都對應一份「數字無法由腳本重算」的證據；第五輪正是在這些檔裡
    抓到三處數字微漂與一處量級錯誤。修好一份就從名單移除一份。
    """
    present = {p.name for p in _evidence_files()}
    stale = GRANDFATHERED - present
    assert len(GRANDFATHERED) <= 13, (
        f"豁免名單長到 {len(GRANDFATHERED)} 筆——新證據檔應帶 provenance 而非加豁免"
    )
    # 名單裡指向已不存在的檔案 → 應該清掉，避免掩護未來的同名新檔
    assert not stale, f"豁免名單有失效項（檔案已不存在）：{sorted(stale)}，請移除"


@pytest.mark.skipif(not _evidence_files(), reason="證據目錄不在此環境")
def test_retracted_claims_are_marked_in_place() -> None:
    """被撤回的宣稱必須在**原證據檔內**標記，不得只在別處寫訂正。

    ★理由：讀者拿到的是那份 json，不是 commit message。第五輪撤回了三條宣稱
    （jac 方向錯、吞吐因果、plateau 劣化），若只在 commit 裡寫，下一個人打開
    json 看到的仍是原始錯誤結論。
    """
    expect = {
        "jac_gamma_probe_2026-08-01.json": "RETRACTION_2026-08-02",
        "longrun_plateau_and_throughput_2026-08-02.json": "RETRACTION_2026-08-02_throughput",
        "convergence_analysis_2026-08-01.json": "CORRECTION_2026-08-02",
    }
    for name, key in expect.items():
        p = OUT_DIR / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        assert key in d, f"{name} 缺撤回／訂正標記 {key}——讀者會看到原始錯誤結論"
