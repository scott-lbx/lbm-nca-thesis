"""IBFV（Image Based Flow Visualization）參數單一真相（SSOT）。

集中所有 IBFV pass 的可調參數，避免散落於 kernel 與 window 主迴圈。
依 `docs/research_plan/smooth_transition/SSOT_SPEC_dual_layer_ibfv.md` §5 起手式建議值。

設計原則（對齊專案 config 規則）：
    - dataclass 欄位即單一真相，存取一律 `cfg.field`（無 getattr fallback）。
    - 新增層（mean / turb）各持一份 `IBFVParams`，差異僅 noise 頻率與 α。

座標與單位約定（與 `streamlines.py` / `field_ops.py` 一致）：
    - 驅動速度場為 LBM 格子速度（`vel = j / rho`，未換算 m/s）。
    - advect 位移 `dt_advect` 直接以「像素 / frame」為單位作用於格子速度，
      故落地時需令 `‖v‖ · dt_advect ≈ 0.5–1.5 px`（見下方 `dt_advect` 註解）。
"""
from dataclasses import dataclass


@dataclass
class IBFVParams:
    """單層 IBFV pass 的參數。

    Attributes:
        alpha: 時間混合係數 α，控制新噪聲注入比例與拖尾長度。
            建議 0.05–0.10：越小拖尾越長、畫面越平滑、反應越慢。
        dt_advect: 每 render frame 的平流步長（作用於格子速度，單位 px/frame）。
            目標 `‖v‖ · dt_advect ≈ 0.5–1.5 px`；典型格子速度量級 ~0.1，
            故起手式約 5.0–15.0。落地時依實測速度量級校正（見 D10a y 軸測試）。
        noise_freq: animated 噪聲的空間頻率（每單位歸一化座標的格數）。
            mean 層用粗噪聲（小值），turb 層用細噪聲（mean 的 2–4×）。
        noise_time_rate: 噪聲時間相位推進速率（rad / frame）。
            >0 才有持續流動感（animated noise）；0 = 靜態噪聲（不建議）。
        injection_floor: 新噪聲注入下限，避免 α 過小時紋理長期不更新導致死灰。
            預設 0（純 IBFV）；需要時可微調。
    """

    alpha: float = 0.08
    dt_advect: float = 8.0
    noise_freq: float = 24.0
    noise_time_rate: float = 0.15
    injection_floor: float = 0.0


# ── 兩層建議預設（dual_layer SPEC §2/§5）──────────────────────────────
# mean 層：粗噪聲、長拖尾、低反應（當底，表達大尺度結構）。
MEAN_LAYER_PARAMS = IBFVParams(
    alpha=0.06,
    dt_advect=8.0,
    noise_freq=18.0,
    noise_time_rate=0.10,
)

# turb 層：細噪聲（mean 的 ~3×）、稍短拖尾（脈動感更明顯）。
TURB_LAYER_PARAMS = IBFVParams(
    alpha=0.09,
    dt_advect=8.0,
    noise_freq=54.0,
    noise_time_rate=0.20,
)
