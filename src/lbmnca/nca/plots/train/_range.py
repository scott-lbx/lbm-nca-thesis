"""Per-sample 配色範圍：從 GT(target) 推導 vmin/vmax，IC/GT/pred 共用。

固定全域範圍（vel [0,0.20]、vor ±0.04）對過阻尼暗場太寬 → 全黑、看不出 pred
比 GT 暗多少。改用 GT 的 percentile 自適應：pred 仍按「同欄 GT 的 scale」畫，
暗則誠實顯暗（不會 auto-range fake similarity，因為基準是 GT 不是 pred 自身）。

每個 sample（compare 圖的一欄 / audit 圖的一列）各自 GT 推導範圍 → 結構最清楚。
代價：欄與欄量值不可直接比（量值寫在欄標題）。
"""
from __future__ import annotations

import numpy as np

# floor 防止近零死場把噪聲放大成滿屏
_VEL_FLOOR = 0.02
_VOR_FLOOR = 0.005
_PCT = 99.0


def _masked(field_hw: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    return np.asarray(field_hw)[np.asarray(mask_hw) > 0.5]


def gt_vel_range(target_hw: np.ndarray, mask_hw: np.ndarray,
                 floor: float = _VEL_FLOOR) -> tuple[float, float]:
    """vel magnitude：[0, max(p99(GT), floor)]。"""
    v = _masked(target_hw, mask_hw)
    vmax = float(np.percentile(np.abs(v), _PCT)) if v.size else floor
    return 0.0, max(vmax, floor)


def gt_vor_range(target_hw: np.ndarray, mask_hw: np.ndarray,
                 floor: float = _VOR_FLOOR) -> tuple[float, float]:
    """vorticity：對稱 [-a, a]，a = max(p99(|GT|), floor)。"""
    v = _masked(target_hw, mask_hw)
    a = float(np.percentile(np.abs(v), _PCT)) if v.size else floor
    a = max(a, floor)
    return -a, a


def err_range(pred_hw: np.ndarray, target_hw: np.ndarray, mask_hw: np.ndarray,
              pct: float = 95.0, floor: float = 1e-6) -> tuple[float, float]:
    """|pred-GT| 的 [0, p_pct]，per-sample。"""
    e = np.abs(np.asarray(pred_hw) - np.asarray(target_hw))[np.asarray(mask_hw) > 0.5]
    emax = float(np.percentile(e, pct)) if e.size else floor
    return 0.0, max(emax, floor)
