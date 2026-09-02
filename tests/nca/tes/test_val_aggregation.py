"""val 指標聚合的回歸測試（DEEP_AUDIT W4 P0-1）。

守住三件事，每一件都對應本 campaign 實際發生過的缺陷：

1. **加權 ≠ 池化**：``mean_mse`` 與 ``mean_rel_l2`` 必須走精確池化。
   把它們改回 batch 加權平均會系統性偏離，本檔以固定 fixture 釘住差距方向。
2. **NaN 韌性**：單一發散 batch 不得毒化整個 split。W4-B 二輪把池化累加放在
   ``math.isfinite`` 判斷之外，使一個 NaN batch 讓兩個主指標變 NaN，同時還輸出
   ``__n_dropped`` 宣稱已剔除——比未訂正前更糟，且 771 個測試全綠。
3. **契約**：``mean_gate_metrics`` 的回傳鍵集合固定。二輪一度把三個內部累加量
   塞進去，同樣 771 全綠——那些量會沿「metrics dict → 聚合 → 論文證據 JSON」
   洩漏成指標欄位。★四輪訂正：本段原寫「它有 12 個呼叫端，其中數支會迭代所有鍵」，
   實測是 **9 處／6 檔**，且 ``em2_core_metrics``／``rq2_chain_eval`` 都是逐鍵
   具名取值不迭代；真正整包寫進證據 JSON 的是 ``run_mean_gate_eval.py``。

全部純 CPU，不讀 zarr、不載 ckpt。
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch

from lbmnca.nca.tes.eval.mean_gate_eval import mean_gate_metrics, mean_gate_pool_terms
from lbmnca.nca.tes.training.validation.aggregate import (
    COVERAGE_KEY,
    ESTIMATOR_KEY,
    N_DROPPED_SUFFIX,
    N_USED_SUFFIX,
    POOL_KEY_2K,
    POOL_KEYS,
    POOLED_METRICS,
    aggregate_val_metrics,
)


def _pair(batch: int, seed: int, scale: float = 1.0, fluid_cols: int = 8):
    """造一組 (pred, target, mask)。

    Args:
        batch: batch size。
        seed: 亂數種子。
        scale: 噪聲倍率（造出批間誤差量級差異）。
        fluid_cols: 流體欄數（16 欄中的前 N 欄）。**批間不同的流體佔比是
            pooled 與 batch 加權產生差異的必要條件**——若各批每樣本的 n_fluid
            相同，``Σsq_err/Σn_fluid`` 與 batch 加權平均在數學上恆等。
    """
    g = torch.Generator().manual_seed(seed)
    tgt = (torch.rand(batch, 9, 16, 16, generator=g) * 0.4 - 0.2)
    pred = tgt + torch.randn(batch, 9, 16, 16, generator=g) * 0.05 * scale
    mask = torch.zeros(batch, 1, 16, 16)
    mask[:, :, :, :fluid_cols] = 1.0
    return pred, tgt, mask


def _record(batch: int, seed: int, scale: float = 1.0, fluid_cols: int = 8):
    """造一筆 aggregate_val_metrics 的 record。"""
    pred, tgt, mask = _pair(batch, seed, scale, fluid_cols)
    return batch, mean_gate_metrics(pred, tgt, mask), mean_gate_pool_terms(pred, tgt, mask)


# ---------------------------------------------------------------------------
# 1. 池化量與單 batch 指標一致
# ---------------------------------------------------------------------------

def test_pool_terms_matches_single_batch_metrics() -> None:
    """單一 batch 下，池化量重算的 mse／rel_l2 必須等於 mean_gate_metrics 的值。

    ★用 ``rel`` 而非 ``abs``：兩者是不同的 float32 累加路徑，絕對誤差可達 1e-9 量級，
    ``abs=1e-9`` 會假紅。
    """
    pred, tgt, mask = _pair(3, seed=0)
    m = mean_gate_metrics(pred, tgt, mask)
    pool = mean_gate_pool_terms(pred, tgt, mask)

    assert pool["sq_err"] / pool["n_fluid"] == pytest.approx(m["mean_mse"], rel=1e-6)
    assert math.sqrt(pool["sq_err"]) / math.sqrt(pool["sq_tgt"]) == pytest.approx(
        m["mean_rel_l2"], rel=1e-6
    )


# ---------------------------------------------------------------------------
# 2. 加權 ≠ 池化（釘住「不可改回加權平均」）
# ---------------------------------------------------------------------------

def test_pooled_mse_differs_from_weighted_when_fluid_fraction_varies() -> None:
    """流體佔比批間不同時，池化 mse 必須與 batch 加權平均不同。

    ★實測發現（W4 三輪）：若各批每樣本的 ``n_fluid`` 相同，
    ``Σsq_err/Σn_fluid`` 與 batch 加權平均**在數學上恆等**——mse 的池化差異
    只在流體佔比不同時才出現（真實資料中不同幾何的流體佔比本來就不同）。
    本測試因此刻意讓兩批的 fluid_cols 相差 4 倍。
    """
    records = [
        _record(4, seed=1, scale=1.0, fluid_cols=2),
        _record(2, seed=2, scale=6.0, fluid_cols=14),
    ]
    out = aggregate_val_metrics(records)

    weighted_mse = sum(m["mean_mse"] * b for b, m, _ in records) / sum(
        b for b, _, _ in records
    )
    pooled_mse = out["mean_mse"]

    assert pooled_mse != pytest.approx(weighted_mse, rel=1e-3), (
        "池化與加權相同 → mse 可能被改回加權平均"
    )
    lo = min(m["mean_mse"] for _, m, _ in records)
    hi = max(m["mean_mse"] for _, m, _ in records)
    assert lo <= pooled_mse <= hi, "池化值必須落在各批之間"


def test_pooled_rel_l2_differs_from_weighted_even_with_equal_masks() -> None:
    """``mean_rel_l2`` 是比值，即使各批流體佔比相同也不等於加權平均。

    這是它與 ``mean_mse`` 的關鍵差異：mse 只在佔比不同時才有池化差異，
    rel_l2 因為是 ``sqrt(Σd²)/sqrt(Σt²)``，任何批間誤差量級差異都會造成不同。
    """
    records = [_record(4, seed=11, scale=1.0), _record(4, seed=12, scale=8.0)]
    out = aggregate_val_metrics(records)

    weighted = sum(m["mean_rel_l2"] * b for b, m, _ in records) / sum(
        b for b, _, _ in records
    )
    assert out["mean_rel_l2"] != pytest.approx(weighted, rel=1e-3), (
        "rel_l2 池化與加權相同 → 可能被改回加權平均"
    )


def test_pooled_matches_manual_formula() -> None:
    """池化值必須精確等於 Σsq_err/Σn_fluid 與 sqrt(Σsq_err)/sqrt(Σsq_tgt)。"""
    records = [_record(4, seed=3), _record(2, seed=4, scale=3.0), _record(3, seed=5)]
    out = aggregate_val_metrics(records)

    sq_err = sum(p["sq_err"] for _, _, p in records)
    sq_tgt = sum(p["sq_tgt"] for _, _, p in records)
    n_fluid = sum(p["n_fluid"] for _, _, p in records)

    assert out["mean_mse"] == pytest.approx(sq_err / n_fluid, rel=1e-9)
    assert out["mean_rel_l2"] == pytest.approx(
        math.sqrt(sq_err) / math.sqrt(sq_tgt), rel=1e-9
    )


def test_general_metrics_are_batch_size_weighted() -> None:
    """非池化指標走 batch size 加權（尾批不滿時不得被高估）。"""
    records = [
        (4, {"foo": 0.1}, {k: 1.0 for k in POOL_KEYS}),
        (4, {"foo": 0.2}, {k: 1.0 for k in POOL_KEYS}),
        (3, {"foo": 0.9}, {k: 1.0 for k in POOL_KEYS}),
    ]
    out = aggregate_val_metrics(records)
    expected = (0.1 * 4 + 0.2 * 4 + 0.9 * 3) / 11
    assert out["foo"] == pytest.approx(expected, rel=1e-12)
    # 未加權平均會是 0.4；確認我們不是那個
    assert out["foo"] != pytest.approx((0.1 + 0.2 + 0.9) / 3, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. NaN 韌性（本輪 M1 阻斷級缺陷的回歸鎖）
# ---------------------------------------------------------------------------

def test_nan_batch_does_not_poison_pooled_metrics() -> None:
    """單一發散 batch 必須被整組排除，不得使整個 split 的池化指標變 NaN。

    這是 W4-B 二輪的實際迴歸：池化累加在 ``isfinite`` 判斷之外，一個 NaN batch
    讓 ``mean_mse``／``mean_rel_l2`` 全變 NaN，而 ``__n_dropped`` 仍宣稱已剔除。
    """
    good = _record(4, seed=6)
    bad_metrics = {"mean_mse": math.nan, "mean_rel_l2": math.nan, "mean_jx_ssim": 0.5}
    bad_pool = {"sq_err": math.nan, "sq_tgt": math.nan, "n_fluid": 100.0}
    out = aggregate_val_metrics([good, (2, bad_metrics, bad_pool)])

    assert math.isfinite(out["mean_mse"]), "NaN batch 毒化了池化的 mean_mse"
    assert math.isfinite(out["mean_rel_l2"]), "NaN batch 毒化了池化的 mean_rel_l2"
    # 池化值應等於只用 good batch 算出來的
    assert out["mean_mse"] == pytest.approx(
        good[2]["sq_err"] / good[2]["n_fluid"], rel=1e-9
    )
    # 剔除必須被揭露，且單位是樣本數
    assert out[f"mean_mse{N_DROPPED_SUFFIX}"] == pytest.approx(2.0)
    assert out[f"mean_mse{N_USED_SUFFIX}"] == pytest.approx(4.0)


def test_all_batches_nan_keeps_key_absent_rather_than_nan() -> None:
    """全部 batch 皆非有限時，指標鍵不出現，而不是輸出 NaN。"""
    bad = (2, {"mean_mse": math.nan, "mean_rel_l2": math.inf},
           {"sq_err": math.nan, "sq_tgt": 1.0, "n_fluid": 1.0})
    out = aggregate_val_metrics([bad, bad])
    for name in POOLED_METRICS:
        assert name not in out
        assert out[f"{name}{N_DROPPED_SUFFIX}"] == pytest.approx(4.0)


def test_empty_records_returns_empty_dict() -> None:
    """空 records 回空字典，不得丟例外或回 NaN。"""
    assert aggregate_val_metrics([]) == {}


# ---------------------------------------------------------------------------
# 4. 回傳鍵集合契約（守住「不得再塞內部欄位進論文指標」）
# ---------------------------------------------------------------------------

def test_mean_gate_metrics_key_contract() -> None:
    """``mean_gate_metrics`` 的回傳鍵集合是對外契約。

    ``run_mean_gate_eval.py`` 會把聚合結果整包寫進論文證據 JSON，因此任何混進
    這個 dict 的內部量都會變成論文指標欄位。新增鍵前必須先改本測試，
    並說明該鍵為何屬論文指標。（★四輪訂正：原寫「12 個呼叫端…迭代所有鍵」，
    實測 9 處／6 檔，且被點名的兩支其實是逐鍵具名取值。）
    """
    pred, tgt, mask = _pair(2, seed=7)
    assert set(mean_gate_metrics(pred, tgt, mask)) == {
        "mean_mse", "mean_rel_l2", "mean_jx_ssim",
    }


def test_pool_terms_key_contract_and_no_underscore_disguise() -> None:
    """池化量走獨立函式、鍵名不以底線偽裝成「私有」。"""
    pred, tgt, mask = _pair(2, seed=8)
    keys = set(mean_gate_pool_terms(pred, tgt, mask))
    assert keys == set(POOL_KEYS)
    assert not any(k.startswith("_") for k in keys)


# ---------------------------------------------------------------------------
# 5. k→2k 估計量不得混用（should_fix S2 的回歸鎖）
# ---------------------------------------------------------------------------

def _record_2k(batch: int, seed: int, scale: float = 1.0, fluid_cols: int = 8):
    """造一筆含 sq_err_2k 的 record（模擬 self-rollout 續推到 2k 步）。"""
    pred_k, tgt, mask = _pair(batch, seed, scale, fluid_cols)
    pred_2k, _, _ = _pair(batch, seed + 500, scale * 1.7, fluid_cols)
    m = mean_gate_metrics(pred_k, tgt, mask)
    mse_2k = mean_gate_metrics(pred_2k, tgt, mask)["mean_mse"]
    m["mean_mse_2k"] = float(mse_2k)
    m["mean_drift_2k"] = float(mse_2k - m["mean_mse"])
    pool = mean_gate_pool_terms(pred_k, tgt, mask)
    pool[POOL_KEY_2K] = mean_gate_pool_terms(pred_2k, tgt, mask)["sq_err"]
    return batch, m, pool


def test_drift_identity_holds_after_aggregation() -> None:
    """聚合後必須滿足 ``mean_drift_2k == mean_mse_2k - mean_mse``。

    ★這正是 S2 的缺陷：``mean_mse`` 走精確池化、``mean_mse_2k``／``mean_drift_2k``
    走 batch 加權，三個數並存於同一筆 val 紀錄，讀者自行相減會得到與紀錄不同的值。
    """
    records = [
        _record_2k(4, seed=21, scale=1.0, fluid_cols=3),
        _record_2k(2, seed=22, scale=5.0, fluid_cols=13),
        _record_2k(3, seed=23, scale=2.0, fluid_cols=8),
    ]
    out = aggregate_val_metrics(records)
    assert out["mean_drift_2k"] == pytest.approx(
        out["mean_mse_2k"] - out["mean_mse"], rel=1e-9
    ), "drift 恆等式在 split 層級不成立 → 估計量又被混用"


def test_pooled_2k_matches_manual_formula() -> None:
    """``mean_mse_2k`` 必須精確等於 ``Σ sq_err_2k / Σ n_fluid``（與 mean_mse 同分母）。"""
    records = [_record_2k(4, seed=31, fluid_cols=3), _record_2k(2, seed=32, fluid_cols=13)]
    out = aggregate_val_metrics(records)
    sq_2k = sum(p[POOL_KEY_2K] for _, _, p in records)
    n_fluid = sum(p["n_fluid"] for _, _, p in records)
    assert out["mean_mse_2k"] == pytest.approx(sq_2k / n_fluid, rel=1e-9)

    weighted = sum(m["mean_mse_2k"] * b for b, m, _ in records) / sum(
        b for b, _, _ in records
    )
    assert out["mean_mse_2k"] != pytest.approx(weighted, rel=1e-3), (
        "池化與加權相同 → mean_mse_2k 可能被改回加權平均"
    )


def test_missing_sq_err_2k_falls_back_to_weighted_without_crashing() -> None:
    """沒有提供 ``sq_err_2k`` 時（如 turb 線）不得丟例外，沿用加權平均。"""
    records = [_record(4, seed=41), _record(2, seed=42)]
    for _, m, _ in records:
        m["mean_mse_2k"] = 0.5
    out = aggregate_val_metrics(records)
    assert out["mean_mse_2k"] == pytest.approx(0.5, rel=1e-9)


def test_estimator_switch_is_disclosed_not_silent() -> None:
    """★第四輪 must_fix：2k 估計量切換必須自證，不得靜默。

    覆蓋率不足時退回加權平均，恆等式 ``drift == mse_2k - mse`` 在該分支**不成立**，
    而先前輸出的鍵集合與池化路徑完全相同——讀者無從得知自己能不能相減。
    該分支 live 可達：任一批的 2k rollout 發散使 ``sq_err_2k`` 非有限就會走到，
    而那正是 ``mean_drift_2k`` 存在的目的所要偵測的情形。
    """
    full = [_record_2k(4, seed=71, fluid_cols=3), _record_2k(2, seed=72, fluid_cols=13)]
    out_full = aggregate_val_metrics(full)
    assert out_full[ESTIMATOR_KEY] == 1.0
    assert COVERAGE_KEY not in out_full
    assert out_full["mean_drift_2k"] == pytest.approx(
        out_full["mean_mse_2k"] - out_full["mean_mse"], rel=1e-9
    )

    partial = [_record_2k(4, seed=71, fluid_cols=3), _record_2k(2, seed=72, fluid_cols=13)]
    del partial[1][2][POOL_KEY_2K]
    out_part = aggregate_val_metrics(partial)
    assert out_part[ESTIMATOR_KEY] == 0.0, "退回加權卻沒有揭露"
    assert 0.0 < out_part[COVERAGE_KEY] < 1.0
    # 恆等式在此分支確實不成立——揭露欄位存在的理由
    assert out_part["mean_drift_2k"] != pytest.approx(
        out_part["mean_mse_2k"] - out_part["mean_mse"], rel=1e-6
    )


def test_disclosure_values_are_floats_not_strings() -> None:
    """揭露欄位必須是可被 ``f"{v:.4f}"`` 格式化的數值。

    ★消費端（``trainer_base.py``／``trainer.py`` 的 VAL log）會對**每一個** value
    做 ``:.4f``，字串會直接 ValueError。塞不能格式化的值進共用指標 dict 正是本
    campaign 被點名過的缺陷類別（``_pool_*`` 鍵事件）。
    """
    out = aggregate_val_metrics([_record_2k(2, seed=81)])
    for key, val in out.items():
        assert isinstance(val, float), f"{key} 不是 float：{val!r}"
        f"{val:.4f}"  # 不得丟例外


def test_no_2k_disclosure_when_metric_absent() -> None:
    """完全沒有 2k 指標時不得憑空生出揭露欄位（避免污染 turb 等其他線）。"""
    out = aggregate_val_metrics([_record(4, seed=91), _record(2, seed=92)])
    assert ESTIMATOR_KEY not in out
    assert COVERAGE_KEY not in out


def test_partial_sq_err_2k_does_not_override() -> None:
    """只有部分 batch 提供 ``sq_err_2k`` → 不得覆寫。

    分母 ``Σ n_fluid`` 來自**全部**納入池化的 batch；分子只累加了一部分，
    覆寫會系統性低估。此時應保留加權平均（有偏但至少估計量一致）。
    """
    full = _record_2k(4, seed=51, fluid_cols=3)
    partial = _record_2k(2, seed=52, fluid_cols=13)
    del partial[2][POOL_KEY_2K]
    out = aggregate_val_metrics([full, partial])

    n_fluid = full[2]["n_fluid"] + partial[2]["n_fluid"]
    bad = full[2][POOL_KEY_2K] / n_fluid
    assert out["mean_mse_2k"] != pytest.approx(bad, rel=1e-6), (
        "分子少了一批、分母沒少 → 低估的池化值被寫了出去"
    )
    weighted = (full[1]["mean_mse_2k"] * 4 + partial[1]["mean_mse_2k"] * 2) / 6
    assert out["mean_mse_2k"] == pytest.approx(weighted, rel=1e-9)


def test_nan_batch_excluded_from_2k_pool_too() -> None:
    """發散 batch 也必須同時退出 2k 池化（分子分母同進同退）。"""
    good = _record_2k(4, seed=61)
    bad = (2,
           {"mean_mse": math.nan, "mean_rel_l2": math.nan, "mean_mse_2k": math.nan},
           {"sq_err": math.nan, "sq_tgt": math.nan, "n_fluid": 999.0,
            POOL_KEY_2K: math.nan})
    out = aggregate_val_metrics([good, bad])
    assert out["mean_mse_2k"] == pytest.approx(
        good[2][POOL_KEY_2K] / good[2]["n_fluid"], rel=1e-9
    )
    assert out["mean_drift_2k"] == pytest.approx(
        out["mean_mse_2k"] - out["mean_mse"], rel=1e-9
    )


def test_disclosure_suffixes_carry_their_unit() -> None:
    """剔除揭露的後綴必須自帶單位（should_fix S5）。

    ★2026-08-05：對照組 ``eval/rp_aggregate.py``（n 為 **seed 數** 3–7，本檔為
    **樣本數** 數百）已隨 RP 實驗臂歸檔至 ``Archive/dead_arms_2026-08-05/``，
    故該側的對照斷言撤下。本檔這一側的不變量仍守著：後綴必須自帶單位，
    否則兩種 n 並列在論文證據鏈裡讀者無從分辨。
    """
    assert N_USED_SUFFIX.endswith("_samples")
    assert N_DROPPED_SUFFIX.endswith("_samples")


def test_all_solid_batch_does_not_pollute_pool() -> None:
    """全固體 batch（``fluid_mask`` 全 0）不得往池化分母灌幽靈格數（★第四輪 should_fix）。

    ``mean_gate_metrics`` 為避免除零用 ``clamp_min(1.0)``，但那個 1.0 若跟著進
    ``mean_gate_pool_terms`` 的累加量，Σn_fluid 就會憑空 +1 而分子是 0——整個
    split 的 mean_mse 被無聲稀釋。真實 val batch 不會全固體，但「不會發生所以
    不用處理」正是本 campaign 反覆出事的形狀。
    """
    pred, tgt, _ = _pair(2, seed=101)
    solid = torch.zeros(2, 1, 16, 16)
    pool = mean_gate_pool_terms(pred, tgt, solid)
    assert pool["n_fluid"] == 0.0, "全固體 batch 貢獻了幽靈 n_fluid"
    assert pool["sq_err"] == 0.0 and pool["sq_tgt"] == 0.0

    good = _record(4, seed=102)
    m_solid = mean_gate_metrics(pred, tgt, solid)
    out = aggregate_val_metrics([good, (2, m_solid, pool)])
    # 池化值必須與「只有 good batch」完全相同（全固體批不改變分子也不改變分母）
    assert out["mean_mse"] == pytest.approx(
        good[2]["sq_err"] / good[2]["n_fluid"], rel=1e-9
    )


def test_metrics_still_guards_division_by_zero_on_all_solid() -> None:
    """``mean_gate_metrics`` 本身仍不得除零（clamp 只留在那條，不外洩到累加量）。"""
    pred, tgt, _ = _pair(2, seed=103)
    solid = torch.zeros(2, 1, 16, 16)
    m = mean_gate_metrics(pred, tgt, solid)
    assert math.isfinite(m["mean_mse"]) and m["mean_mse"] == 0.0
    assert math.isfinite(m["mean_rel_l2"])
