"""徑向能譜的共用純函式（radial-averaged spectrum 與其對數斜率）。

背景：`long_rollout._radial_spectrum` 已是通用純函式，但只掛在 turb 的 long-rollout
路徑上；mean 側 canonical 評測完全不呼叫，且全 repo 沒有任何「能譜對數斜率」實作
（既有的 `_spectrum_logmse` 量的是 pred 與 GT 的 log 譜距離，不是斜率）。本模組把
`radial_spectrum` 上移為單一真相，並新增 `radial_spectrum_logslope`。

★本模組是**純量測**：import 條數與「無 `lbmnca.nca.tes.loss.*`」由
`tests/nca/tes/test_r4_countable_claims.py` 以 AST 機器複驗（數字不寫進本 docstring）。
不接受任何可調權重、不設通過門檻。這是防「判準自我實現」的結構性保證——
量測與最佳化目標之間沒有可接線的介面。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "radial_spectrum",
    "radial_spectrum_logslope",
]

_EPS: float = 1e-12


def radial_spectrum(field: np.ndarray) -> np.ndarray:
    """2D 場的 radial-averaged 能譜：|FFT(field)|² 對 |k| 環平均。

    ★計算與 `long_rollout._radial_spectrum`（本函式的原始出處）逐行相同，唯一差異是
    `ValueError` 訊息裡的函式名（`_radial_spectrum` → `radial_spectrum`）。上移後由
    `long_rollout` 以 ``as _radial_spectrum`` 反向 import，既有 turb 評測數值零變更。

    Args:
        field: 2D 實數陣列 ``[H, W]``。

    Returns:
        1D radial 能譜（length = max radius + 1）；含 DC bin（index 0）。

    Raises:
        ValueError: 輸入陣列非 2D 時拋出。
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"radial_spectrum 需 2D 場，收到 ndim={f.ndim}")
    h, w = f.shape
    power = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
    rmax = int(r.max())
    tbin = np.bincount(r.ravel(), weights=power.ravel(), minlength=rmax + 1)
    nbin = np.bincount(r.ravel(), minlength=rmax + 1).clip(min=1)
    return tbin / nbin


def radial_spectrum_logslope(
    field: np.ndarray,
    *,
    k_lo_frac: float,
    k_hi_frac: float,
    r2_min: float,
) -> dict[str, float | bool | int]:
    """log E(k) 對 log k 在指定波數區間的最小平方斜率。

    慣性次區間（inertial subrange）內 2D 能譜近似冪律 ``E(k) ∝ k^s``；斜率 s 是
    「模型把能量放在哪個尺度」的直接量測。偏離 GT 斜率為正 → 高頻能量過多
    （grain / checkerboard）；為負 → 過度耗散（糊）。

    ★`k_lo_frac` / `k_hi_frac` / `r2_min` 皆為**必填**，刻意不給預設值：
    區間選擇會直接改變斜率數值，任何隱含預設等同於偷偷替作者做判準決定。

    Args:
        field: 2D 實數陣列 ``[H, W]``。
        k_lo_frac: 擬合下界，佔最大可用波數（bin index 上界）的比例，(0, 1)。
        k_hi_frac: 擬合上界，同上，須大於 ``k_lo_frac``。
        r2_min: 擬合可信度門檻；決定係數低於此值時 ``ill_fit`` 標記為 True。
            ★這是「擬合本身是否成立」的統計門檻，不是物理判準的通過門檻。

    Returns:
        包含以下鍵的字典：
        - ``logslope``: 最小平方斜率 s（擬合失敗回 nan）。
        - ``r2``: 決定係數（擬合失敗回 nan）。
        - ``ill_fit``: ``r2 < r2_min`` 或擬合失敗時為 True。
        - ``n_points``: 實際參與擬合的 bin 數。
        - ``k_lo``: 實際使用的下界 bin index。
        - ``k_hi``: 實際使用的上界 bin index。

    Raises:
        ValueError: 區間參數不合法（非 (0,1) 或 lo >= hi）時拋出。
    """
    if not (0.0 < k_lo_frac < 1.0) or not (0.0 < k_hi_frac <= 1.0):
        raise ValueError(f"k_lo_frac/k_hi_frac 須落在 (0,1]：{k_lo_frac}, {k_hi_frac}")
    if k_lo_frac >= k_hi_frac:
        raise ValueError(f"k_lo_frac 須小於 k_hi_frac：{k_lo_frac} >= {k_hi_frac}")

    spec = radial_spectrum(field)
    k_max = spec.size - 1                        # bin index 上界（index 0 = DC）
    k_lo = max(1, int(round(k_lo_frac * k_max)))  # 排除 DC bin
    k_hi = min(k_max, int(round(k_hi_frac * k_max)))
    fail = {
        "logslope": float("nan"),
        "r2": float("nan"),
        "ill_fit": True,
        "n_points": 0,
        "k_lo": int(k_lo),
        "k_hi": int(k_hi),
    }
    if k_hi - k_lo < 2:                           # 至少 3 點才談得上擬合
        return fail

    k = np.arange(k_lo, k_hi + 1, dtype=np.float64)
    e = spec[k_lo : k_hi + 1]
    ok = np.isfinite(e) & (e > 0.0)
    if int(ok.sum()) < 3:
        return fail

    log_k = np.log10(k[ok])
    log_e = np.log10(e[ok] + _EPS)
    slope, intercept = np.polyfit(log_k, log_e, 1)
    resid = log_e - (slope * log_k + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    r2 = float("nan") if ss_tot <= _EPS else 1.0 - ss_res / ss_tot
    if not np.isfinite(slope) or not np.isfinite(r2):
        return fail
    return {
        "logslope": float(slope),
        "r2": r2,
        "ill_fit": bool(r2 < float(r2_min)),
        "n_points": int(ok.sum()),
        "k_lo": int(k_lo),
        "k_hi": int(k_hi),
    }
