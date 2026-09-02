"""釘死「消融的損失空間權重必須取自覆寫前的真實 parent」這條不變式。

2026-08-19 踩過兩次：

1. `mean_step` 改成讀 `batch["true_parent_mean_renorm"]`，但 `graph_dataset._FIELD_KEYS`
   沒加該鍵，collate 直接把它丟掉，`batch.get()` 回 None 而靜默退回用覆寫後的
   `ic_physics`。`far`／`rand` 兩臂的 `ic_physics` 在資料層就已被 `src_override`
   換掉，於是修正變成空操作，兩臂各約四小時的重訓白跑。
2. 只把鍵加進 `_FIELD_KEYS` 而漏掉 `_maybe_vflip`，會讓半數樣本（`vflip_aug_prob 0.5`）
   的權重高值落在流域的錯誤半邊，且五臂同受影響、不會在臂間比較中露出來。

兩者都不會讓訓練報錯，只會讓數字悄悄失去可歸因性。故以測試釘死。
"""
from __future__ import annotations

import inspect

import torch

from lbmnca.nca.tes.data.graph_dataset import _FIELD_KEYS, EdgeMeanDataset, collate_graph_batch

KEY = "true_parent_mean_renorm"


def test_field_keys_contains_true_parent() -> None:
    """collate 只 stack `_FIELD_KEYS` 內的鍵，漏掉即等於該修正無效。"""
    assert KEY in _FIELD_KEYS, (
        f"{KEY} 不在 _FIELD_KEYS，collate 會丟掉它，"
        "mean_step 會退回用覆寫後的 ic_physics 建損失權重"
    )


def test_collate_preserves_true_parent() -> None:
    """端到端：sample 有這個鍵，collate 後 batch 也必須有。"""
    def mk() -> dict[str, torch.Tensor]:
        return {
            "static": torch.zeros(4, 8, 8),
            "mean_flow_post_conv": torch.zeros(9, 8, 8),
            "parent_mean_renorm": torch.ones(9, 8, 8),
            KEY: torch.full((9, 8, 8), 2.0),
            "g_vec": torch.zeros(14),
            "p_vec": torch.zeros(3),
            "source_stats_idx": torch.tensor(0),
        }

    batch = collate_graph_batch([mk(), mk()])
    assert KEY in batch, f"collate 後 {KEY} 消失"
    assert batch[KEY].shape == (2, 9, 8, 8)
    # 兩者必須是不同的張量，否則「覆寫前／後」的區分沒有意義
    assert not torch.equal(batch[KEY], batch["parent_mean_renorm"])


def test_vflip_flips_true_parent_too() -> None:
    """vflip 必須同步翻轉 true_parent，否則權重落在錯誤半邊。"""
    src = inspect.getsource(EdgeMeanDataset._maybe_vflip)
    assert KEY in src, (
        f"_maybe_vflip 沒有翻轉 {KEY}。vflip_aug_prob 為 0.5 時，"
        "半數樣本的監督目標已翻轉而權重來源未翻，權重高值會落在流域錯誤半邊"
    )
    # 與 parent 在同一個翻轉清單裡（同一行 for 迴圈）
    assert "parent_mean_renorm" in src


def test_mean_step_raises_when_key_missing() -> None:
    """缺鍵必須報錯，不得靜默 fallback —— 靜默正是上次失效沒被發現的原因。"""
    from lbmnca.nca.tes.training.steps import mean_step

    src = inspect.getsource(mean_step)
    assert "缺 true_parent_mean_renorm" in src, (
        "mean_step 應在有 parent_mean_renorm 卻缺 true_parent_mean_renorm 時 raise，"
        "而非 fallback 到覆寫後的 ic_physics"
    )


def test_meanpool_seed_ic_separate_from_parent() -> None:
    """MeanPool 的 seed 重置 IC 必須與 wake 權重的 parent 分開。

    2026-08-19 把 `parent_baseline` 統一成真實 parent 以修損失權重共線，
    但 `refill` 也用它灌 pool，於是 seed 重置（`n_seed = round(B * 0.15)`，
    `batch_size 5` 時即每批 20%）把真實 parent 當 IC 注入**全部消融臂**：
    `abl_no` 的「完全沒有暖起點」不成立、`far`/`rand` 有 20% 樣本等同 `abl_use`。
    效應量因此朝零衰減，且衰減量無法從量測資料反推。
    """
    import inspect

    from lbmnca.nca.tes.training import mean_pool
    from lbmnca.nca.tes.training.steps import mean_step

    src = inspect.getsource(mean_pool)
    assert "_seed_ic" in src, "MeanPool 缺 _seed_ic 欄位"
    assert "model.init_state(_static, seed_ic[:n_seed])" in src, (
        "seed 重置仍用 parent 當 IC；那是真實 parent，會還原消融臂被移除的暖起點"
    )
    assert "seed_ic: torch.Tensor" in src, "refill 缺 seed_ic 參數"

    # 呼叫是多行寫的，先把連續空白壓成單一空格再比對。原本的單行子字串比對
    # 因換行與縮排而永遠對不上——那種斷言看起來是綠的，其實從來沒驗過任何東西。
    step_src = " ".join(inspect.getsource(mean_step).split())
    assert "refill( state, target_mean, parent_baseline, ic_physics" in step_src, (
        "mean_step 未把該臂覆寫後的 ic_physics 當 seed_ic 傳入"
    )
