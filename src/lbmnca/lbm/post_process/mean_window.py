"""GT 時均場的取樣不確定度量化（DEEP_AUDIT_2026-07 修正 1.5，1.16 後僅存部分）。

本模組提供與 I/O 解耦的純函式（不碰 h5py/zarr，呼叫端傳 array 進來），
回答「GT 時均場本身的取樣不確定度是多少」——用以區分「模型能力不足」與
「GT 標的本身取樣雜訊過大」。見 :func:`compute_sem_eff`。

★ **2026-07-28（Stage 1.14 裁決 → 1.16 執行）：脫落週期偵測與 T_shed-based
時均窗選擇已整批移除。**

移除的是 ``detect_shedding_period`` / ``ShedPeriodResult`` / ``_dominant_frequency``
/ ``select_mean_window`` / ``MeanWindowResult`` / ``FALLBACK_*`` 及其常數。
理由（Stage 1.13 + 1.14 實測）：

  - 全域積分 KE 幾乎不含渦脫落訊號 —— Re=100 圓柱上真實脫落諧波比 FFT 主峰
    弱 **4744 倍**，回報的 ``T_shed`` 高估真值 **3.59 倍**。
  - 唯一原理上可行的補救（雙訊號交叉驗證）實測出局：KE 與 enstrophy 逐 bin
    選中**同一個錯誤主峰** → 會回報「一致、可信」再給出錯 3.6-4.0 倍的週期，
    比單訊號更糟，因為它製造假的信心。
  - 生產端無真值可驗收任何偵測器 → 該路線在生產上原本就不可驗收。

移除成本實測接近零：``MEAN_WINDOW_POLICY`` 預設一直是 ``"post_conv"``，
``T_shed`` **從未進入生產路徑**；4 個 canonical dataset 的 2063 個 node/edge
group 中，含週期相關欄位者為 **0 個** → 無任何既有產物需要標記作廢或重轉。

被移除的程式碼逐字保存在
``docs/reviews/DEEP_AUDIT_2026-07/tools/_retired_period_detection.py``，
供 Stage 1.9/1.13/1.14/phase1-5 的稽核工具復現當年結論之用，**不得**被生產
管線 import。

⚠ 倖存的 :func:`compute_sem_eff` **不依賴** ``T_shed``（改用積分自相關時間
τ_int），故 GT 不確定度量化（GATE 1 的 G1-4）不受本次移除影響。
"""

from __future__ import annotations

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 常數
# ══════════════════════════════════════════════════════════════════════════════

MIN_SAMPLES_FOR_SEM: int = 2
"""計算 :func:`compute_sem_eff` 所需的最少樣本數。"""

AUTOCORR_MAX_LAG_FRAC: float = 0.5
"""估計積分自相關時間 τ_int 時，搜尋 lag 的上限（占樣本數比例）。避免大 lag
時樣本自相關估計本身雜訊過大，導致 τ_int 高估甚至發散。"""


def _integrated_autocorr_time(x: np.ndarray, max_lag_frac: float) -> float:
    """以 initial positive sequence（IPS，Geyer 1992 簡化版）估計積分自相關時間。

    樣本自相關函數 ρ(k)（k=1,2,...）從 k=1 開始累加，在自相關首次 ≤ 0 處截斷
    （不含該項）：

        τ_int = 0.5 + Σ_{k=1}^{K} ρ(k)，K = 首個使 ρ(k) ≤ 0 的 k 減一

    搜尋範圍上限為 ``max_lag_frac × n``，避免大 lag 時樣本自相關估計本身雜訊
    過大導致 τ_int 高估甚至發散。

    Args:
        x: [n] 時間序列（未去均值亦可，內部會去均值）。
        max_lag_frac: lag 搜尋上限占樣本數的比例。

    Returns:
        τ_int（float，恆 >= 0.5）。
    """
    n = x.shape[0]
    max_lag = max(1, int(n * max_lag_frac))
    x_c = x - x.mean()
    var = float(np.dot(x_c, x_c) / n)
    if var <= 0.0:
        return 0.5
    tau = 0.5
    for k in range(1, max_lag + 1):
        rho_k = float(np.dot(x_c[: n - k], x_c[k:]) / n / var)
        if rho_k <= 0.0:
            break
        tau += rho_k
    return tau


def compute_sem_eff(values: np.ndarray, max_lag_frac: float = AUTOCORR_MAX_LAG_FRAC) -> float:
    """有效樣本量修正的標準誤（SEM_eff）。

    時間序列相鄰 frame 高度相關（例如同一 case 內連續 snapshot），樸素
    ``std / sqrt(n)`` 會低估不確定度（把相關樣本當獨立樣本計數）。改用積分
    自相關時間 τ_int 修正有效樣本數：

        n_eff = n / (2 τ_int)
        SEM_eff = std / sqrt(n_eff)

    τ_int 估計見 :func:`_integrated_autocorr_time`（initial positive sequence
    法）。此數字回答「GT 時均場本身的取樣不確定度是多少」——時間相關性愈強
    （τ_int 愈大），n_eff 愈小、SEM_eff 相對樸素估計愈大。

    Args:
        values: [n] 純量時間序列（例如逐 frame 的速度大小或動量通道空間平均
            值；呼叫端決定用哪個通道，見呼叫處說明）。
        max_lag_frac: 傳遞給 :func:`_integrated_autocorr_time`。

    Returns:
        SEM_eff（float，恆 >= 0）。

    Raises:
        ValueError: ``values`` 非 1D，或樣本數 < :data:`MIN_SAMPLES_FOR_SEM`。
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values 須為 1D 陣列")
    n = values.shape[0]
    if n < MIN_SAMPLES_FOR_SEM:
        raise ValueError(f"樣本數不足（{n} < {MIN_SAMPLES_FOR_SEM}），無法計算 SEM_eff")

    std = float(np.std(values, ddof=1))
    if std == 0.0:
        return 0.0

    tau_int = _integrated_autocorr_time(values, max_lag_frac)
    n_eff = max(n / (2.0 * tau_int), 1.0)
    return std / np.sqrt(n_eff)
