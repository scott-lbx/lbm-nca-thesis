"""``src/post_process/mean_window.py`` 單元測試（DEEP_AUDIT_2026-07 修正 1.5）。

純 CPU、不需 GPU、不需真 H5、不需 zarr，全部使用合成訊號。

★ 2026-07-28（Stage 1.14 裁決 → 1.16 執行）：週期偵測與 T_shed-based 窗選擇
已從生產模組移除，對應的 4 個測試
（``test_detect_shedding_period_sine_within_5pct`` /
``..._white_noise_not_detected`` / ``..._nonuniform_sampling`` /
``test_select_mean_window_truncated_when_insufficient_periods``）一併刪除。

⚠ 那 4 個測試**全部曾經通過**——因為它們餵的是合成純正弦訊號，而失效模式是
「真實全域 KE 裡根本沒有脫落訊號」（1.13：真實諧波弱 4744 倍）。
合成訊號測試對這個失效模式**原理上無效**，這正是它們沒有攔下 1.13 的原因，
也是為什麼新增判準時不能只用合成訊號驗收。

執行：
    python tests/test_mean_window.py
    # 或
    pytest tests/test_mean_window.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 確保 repo root（lbmnca/lbm/）在 path，讓 `from src...` import 生效
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lbmnca.lbm.post_process.mean_window import compute_sem_eff  # noqa: E402

# Windows 預設 cp950 console 對部分 Unicode 字元（希臘字母等）無法輸出，
# 若測試印出這類字元會使全過的測試因 print 崩潰而誤判失敗。強制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def test_compute_sem_eff_ar1_exceeds_naive() -> None:
    """已知自相關的 AR(1) 序列 → SEM_eff 明顯大於樸素 std/sqrt(n)。"""
    rng = np.random.default_rng(2)
    n = 3000
    phi = 0.8
    x = np.zeros(n)
    noise = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]

    naive_sem = float(np.std(x, ddof=1) / np.sqrt(n))
    sem_eff = compute_sem_eff(x)

    assert sem_eff > 1.5 * naive_sem, (
        f"SEM_eff（{sem_eff:.5f}）應明顯大於樸素 SEM（{naive_sem:.5f}），"
        f"AR(1) phi={phi} 理論 tau_int≈(1+phi)/(1-phi)=9"
    )
    print(f"[PASS] AR(1) naive_sem={naive_sem:.5f} sem_eff={sem_eff:.5f} "
          f"ratio={sem_eff/naive_sem:.2f}x")


def test_compute_sem_eff_iid_matches_naive() -> None:
    """獨立同分布序列（tau_int -> 0.5）→ SEM_eff 應接近樸素 SEM。

    這是 AR(1) 測試的對照組：確認 tau_int 修正不會對無自相關的序列
    無故放大不確定度。
    """
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000)

    naive_sem = float(np.std(x, ddof=1) / np.sqrt(x.shape[0]))
    sem_eff = compute_sem_eff(x)
    ratio = sem_eff / naive_sem

    assert 0.8 < ratio < 1.3, (
        f"iid 序列的 SEM_eff/naive = {ratio:.3f} 應接近 1（tau_int ~= 0.5）"
    )
    print(f"[PASS] iid naive_sem={naive_sem:.5f} sem_eff={sem_eff:.5f} "
          f"ratio={ratio:.2f}x")


def test_period_detection_symbols_removed() -> None:
    """迴歸測試：生產模組不得再提供任何週期偵測 API（1.14 路徑 2）。

    這道測試是為了防止「某次重構把退役複本又搬回生產模組」——
    1.13/1.14 的結論是該 API **在生產上不可驗收**，不是「暫時關掉」。
    """
    import lbmnca.lbm.post_process.mean_window as mw

    banned = (
        "detect_shedding_period", "ShedPeriodResult", "_dominant_frequency",
        "_to_uniform_grid", "MIN_PEAK_POWER_RATIO", "MIN_SAMPLES_FOR_FFT",
        "NYQUIST_SAFETY_FACTOR", "select_mean_window", "MeanWindowResult",
        "FALLBACK_LAST_N_PERIODS", "FALLBACK_LAST_N_PERIODS_TRUNCATED",
        "FALLBACK_PERIOD_UNDETECTED",
    )
    present = [b for b in banned if hasattr(mw, b)]
    assert not present, f"生產模組仍含已撤除的週期偵測符號：{present}"

    import lbmnca.lbm.post_process.tes_h5_to_zarr as tz

    banned_zarr = ("MEAN_WINDOW_POLICY", "N_SHEDDING_PERIODS_FOR_MEAN")
    present_zarr = [b for b in banned_zarr if hasattr(tz, b)]
    assert not present_zarr, f"轉檔器仍含已撤除的週期窗策略：{present_zarr}"

    print(f"[PASS] 週期偵測符號已全數移除（檢查 {len(banned) + len(banned_zarr)} 個）")


if __name__ == "__main__":
    test_compute_sem_eff_ar1_exceeds_naive()
    test_compute_sem_eff_iid_matches_naive()
    test_period_detection_symbols_removed()
    print("\n所有 mean_window 測試通過")
