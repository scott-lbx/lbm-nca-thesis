"""val 指標的跨 batch 聚合（純函式，與 model／loader／GPU 無關）。

此聚合邏輯獨立成純函式的理由：若內嵌在 ``TESTrainer._validate_split_mean``
的迴圈裡，會綁死 model forward／renorm／protocol，無法在不載 ckpt 的情況下
測試；抽成純函式後，聚合語義可用 dict fixture 完整測試。

聚合規則（三者不同，混用會出錯）：

* **一般指標**：以 batch size 加權平均。原本的「批次均值再平均」在尾批不滿
  ``batch_size`` 時會高估該批每個樣本的權重。
* ``mean_mse``：分母是 ``n_fluid`` 而非 batch size，故以
  ``Σ sq_err / Σ n_fluid`` 精確池化。
* ``mean_rel_l2``：是 ``sqrt(Σd²)/sqrt(Σt²)`` 的**比值**，跨 batch 取任何形式的
  算術平均都不等於整個 split 的 rel_L2，故以
  ``sqrt(Σ sq_err) / sqrt(Σ sq_tgt)`` 精確池化。

非有限值處理：某個 batch 的 ``mean_mse`` 或 ``mean_rel_l2`` 若非有限（訓練發散），
該 batch 的池化量**整組**排除（三個量必須同進同退，否則分子與分母來自不同批），
並計入 ``__n_dropped``。

★估計量混用：``mean_mse_2k`` 是 k→2k self-rollout 的 MSE，與 ``mean_mse``
同單位同定義，若只有 ``mean_mse`` 走精確池化、``mean_mse_2k``／``mean_drift_2k``
走 batch 加權平均，兩種估計量在同一份 val 紀錄裡並存，會使**恆等式
``mean_drift_2k == mean_mse_2k - mean_mse`` 在聚合後不成立**——讀者拿三個數
自行相減會得到與紀錄不同的值，且差距方向不定。現在 ``sq_err_2k`` 一併池化、
``mean_drift_2k`` 於聚合後才相減，恆等式在 split 層級恢復成立——**但只在
每一個納入池化的 batch 都提供了 ``sq_err_2k`` 時**。覆蓋率不足（例如某批的
2k rollout 發散）時退回加權平均，此時恆等式不成立，故一律輸出
``mean_mse_2k__pooled``（1.0=池化／0.0=加權）與退回時的
``mean_mse_2k__pool_coverage`` 明示用的是哪一種。
"""

from __future__ import annotations

import math

__all__ = [
    "aggregate_val_metrics",
    "POOL_KEYS",
    "POOLED_METRICS",
    "POOL_KEY_2K",
    "N_USED_SUFFIX",
    "N_DROPPED_SUFFIX",
    "ESTIMATOR_KEY",
    "COVERAGE_KEY",
]

#: ``mean_gate_pool_terms`` 回傳的三個累加量鍵名。
POOL_KEYS: tuple[str, ...] = ("sq_err", "sq_tgt", "n_fluid")

#: 由 ``POOL_KEYS`` 精確池化、而非加權平均的指標名。
POOLED_METRICS: tuple[str, ...] = ("mean_mse", "mean_rel_l2")

#: 剔除揭露的鍵名後綴（★單位寫進名字：本模組聚合的 n 是**樣本數**，
#: 其他聚合模組可能用不同單位，例如 seed 數，注意不要混淆）。
N_USED_SUFFIX: str = "__n_used_samples"
N_DROPPED_SUFFIX: str = "__n_dropped_samples"

#: k→2k self-rollout 的平方誤差累加量（選配）。呼叫端若在 ``pool_terms`` 內提供
#: 此鍵，``mean_mse_2k``／``mean_drift_2k`` 改走精確池化；未提供則沿用加權平均。
POOL_KEY_2K: str = "sq_err_2k"

#: 估計量揭露欄位：覆蓋率不足時退回加權平均，鍵集合與池化路徑完全相同，
#: 若不揭露就會讓 drift 恆等式在該分支悄悄不成立。
#:
#: ★值刻意用 ``float`` 而非 ``"pooled"``／``"weighted"`` 字串：本 dict 的消費端
#: （``trainer_base.py:173``、``trainer.py:774``）會對**每一個** value 做
#: ``f"{v:.4f}"``，字串會直接 ValueError。塞不能格式化的值進共用指標 dict
#: 是需要避免的缺陷類別（``_pool_*`` 鍵事件），不重蹈。
ESTIMATOR_KEY: str = "mean_mse_2k__pooled"
COVERAGE_KEY: str = "mean_mse_2k__pool_coverage"


def aggregate_val_metrics(
    records: list[tuple[int, dict[str, float], dict[str, float]]],
) -> dict[str, float]:
    """把逐 batch 的 val 指標聚合成整個 split 的指標。

    Args:
        records: 每個元素為 ``(batch_size, metrics, pool_terms)``。
            ``metrics`` 是該 batch 的指標字典（``mean_gate_metrics`` 及其後續
            ``update`` 的結果）；``pool_terms`` 是 ``mean_gate_pool_terms`` 的輸出，
            需含 :data:`POOL_KEYS` 三鍵。

    Returns:
        聚合後的指標字典。``mean_mse``／``mean_rel_l2`` 走精確池化，其餘走 batch
        加權平均；另含每個曾被剔除的指標的 ``<key>__n_used_samples``／
        ``<key>__n_dropped_samples``（單位為**樣本數**，非批次數、非 seed 數）。若所有納入池化的 batch 都提供了
        :data:`POOL_KEY_2K`，``mean_mse_2k`` 亦改走精確池化，且
        ``mean_drift_2k`` 於聚合後才以 ``mean_mse_2k - mean_mse`` 重算，使該恆等
        式在 split 層級成立；覆蓋率不足時退回加權平均且恆等式**不**成立，
        由 :data:`ESTIMATOR_KEY`（1.0／0.0）與 :data:`COVERAGE_KEY` 揭露。
        ``records`` 為空時回傳空字典。
    """
    if not records:
        return {}

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    dropped: dict[str, int] = {}
    pool: dict[str, float] = {k: 0.0 for k in POOL_KEYS}
    pool_n: int = 0
    pool_2k: float = 0.0
    pool_2k_n: int = 0

    for batch_size, metrics, pool_terms in records:
        bs = int(batch_size)

        for key, val in metrics.items():
            if math.isfinite(val):
                sums[key] = sums.get(key, 0.0) + float(val) * bs
                counts[key] = counts.get(key, 0) + bs
            else:
                dropped[key] = dropped.get(key, 0) + bs

        # ★池化量與逐 batch 指標用**同一把尺**：只有該 batch 的兩個被池化指標
        #   皆為有限值時才納入，且三個累加量同進同退。
        pooled_ok = all(
            math.isfinite(metrics.get(name, math.nan)) for name in POOLED_METRICS
        ) and all(math.isfinite(pool_terms.get(k, math.nan)) for k in POOL_KEYS)
        if pooled_ok:
            for k in POOL_KEYS:
                pool[k] += float(pool_terms[k])
            pool_n += bs
            # 2k 池化共用同一個 n_fluid 分母，故只能在同一批被納入時累加
            if math.isfinite(pool_terms.get(POOL_KEY_2K, math.nan)):
                pool_2k += float(pool_terms[POOL_KEY_2K])
                pool_2k_n += bs

    out: dict[str, float] = {k: sums[k] / counts[k] for k in sums}

    # 精確池化覆寫（只在有可用批次時；否則保留加權平均，不製造 NaN）
    if pool_n > 0:
        if pool["n_fluid"] > 0.0:
            out["mean_mse"] = pool["sq_err"] / pool["n_fluid"]
        if pool["sq_tgt"] > 0.0:
            out["mean_rel_l2"] = math.sqrt(pool["sq_err"]) / math.sqrt(pool["sq_tgt"])
        # ★2k 分母就是 pool["n_fluid"]，因此只有在**每一個**納入池化的 batch
        #   都提供了 sq_err_2k 時才可覆寫；否則分子少了幾批、分母沒少，會低估。
        #
        # 覆蓋率不足時退回加權平均——鍵集合完全相同、若不揭露則恆等式
        #   drift == mse_2k - mse 在該分支不成立。該分支 live 可達：任何一批的
        #   2k rollout 發散使 sq_err_2k 非有限就會走到，而那正是 mean_drift_2k
        #   這個指標存在的目的所要偵測的情形。現在一律輸出 ESTIMATOR_KEY
        #   （＝ "mean_mse_2k__pooled"，1.0=池化／0.0=加權），明示用的是哪一種，
        #   讀者不必翻程式碼就知道能不能自行相減。
        if "mean_mse_2k" in out or pool_2k_n > 0:
            pooled_ok_2k = (
                pool_2k_n == pool_n and pool["n_fluid"] > 0.0 and "mean_mse" in out
            )
            if pooled_ok_2k:
                out["mean_mse_2k"] = pool_2k / pool["n_fluid"]
                out["mean_drift_2k"] = out["mean_mse_2k"] - out["mean_mse"]
                out[ESTIMATOR_KEY] = 1.0          # 1 = 精確池化，恆等式成立
            else:
                out[ESTIMATOR_KEY] = 0.0          # 0 = 退回加權，恆等式不成立
                out[COVERAGE_KEY] = (
                    float(pool_2k_n) / float(pool_n) if pool_n else 0.0
                )

    # 剔除揭露：後綴自帶單位。本模組的 n 是**樣本數**，其他聚合模組可能用不同
    # 單位（例如 seed 數），讀者若只看鍵名 `__n_used` 會無從分辨。把單位寫進
    # 鍵名，比另開一個 `__n_unit` 欄位更難誤讀。
    for key, n_drop in dropped.items():
        out[f"{key}{N_DROPPED_SUFFIX}"] = float(n_drop)
        out[f"{key}{N_USED_SUFFIX}"] = float(counts.get(key, 0))

    return out
