"""訓練 curriculum 純函數工具。

從 ``TESTrainer`` 拆出來的無狀態取樣／排程函式：

* ``sample_k`` — k curriculum 依步數隨機取樣 rollout 步數
* ``get_val_k`` — 對應相位的 k_max
* ``get_loss_version`` — Loss curriculum 版本
* ``get_tf_ratio`` — Teacher Forcing 線性退火
* ``get_p_teacher`` — Scheduled Sampling 線性退火
* ``select_early_stop_monitor`` — 從 val/VIS-GATE metrics 中解析 early-stop 監控值
* ``add_vg_composite`` — 將 VIS-GATE composite 寫入 metric dict

行為與舊版 ``TESTrainer._*`` 對應方法完全一致。
"""
from __future__ import annotations

import math
from typing import Any, Optional

from ..config import TESCurriculumConfig, TESSScheduleConfig, TESTFCConfig
from .rng_streams import stream


def sample_k(cur: TESCurriculumConfig, step: int) -> int:
    """依目前步數從 k curriculum 取樣 rollout 步數。

    抽樣走 ``rng_streams.stream("schedule")``：``cfg.rng_stream_isolation=False``
    （預設）時即標準庫 ``random`` 模組本身，行為與舊版逐字相同。

    Args:
        cur: k curriculum 設定。
        step: 目前訓練步數。

    Returns:
        隨機取樣的 rollout 步數。
    """
    rng = stream("schedule")
    if step < cur.phase1_end:
        return rng.randint(cur.phase1_k_min, cur.phase1_k_max)
    if step < cur.phase2_end:
        return rng.randint(cur.phase2_k_min, cur.phase2_k_max)
    return rng.randint(cur.phase3_k_min, cur.phase3_k_max)


def get_val_k(cur: TESCurriculumConfig, step: int) -> int:
    """依目前步數回傳 val 評估使用的 k（與訓練同相位的 k_max）。

    Args:
        cur: k curriculum 設定。
        step: 目前訓練步數。

    Returns:
        對應訓練相位的 k_max。
    """
    if step < cur.phase1_end:
        return cur.phase1_k_max
    if step < cur.phase2_end:
        return cur.phase2_k_max
    return cur.phase3_k_max


def get_loss_version(cur: TESCurriculumConfig, step: int) -> str:
    """依目前步數回傳 Loss curriculum 版本（``v1``、``v2`` 或 ``v3``）。

    Args:
        cur: k curriculum 設定（含 loss_v2_start / loss_v3_start 欄位）。
        step: 目前訓練步數。

    Returns:
        Loss curriculum 版本字串：``"v1"``、``"v2"`` 或 ``"v3"``。
    """
    if step < cur.loss_v2_start:
        return "v1"
    if step < cur.loss_v3_start:
        return "v2"
    return "v3"


def get_tf_ratio(tfc: TESTFCConfig, step: int) -> float:
    """依目前步數計算 Teacher Forcing ratio（0.0=純自迴歸，1.0=全 TF）。

    Args:
        tfc: Teacher Forcing 設定。
        step: 目前訓練步數。

    Returns:
        Teacher Forcing ratio（[0.0, 1.0]）。
    """
    if not tfc.enabled:
        return 0.0
    if step < tfc.phase1_end:
        return tfc.tf_ratio_init
    if step < tfc.phase2_end:
        progress = (step - tfc.phase1_end) / max(1, tfc.phase2_end - tfc.phase1_end)
        return tfc.tf_ratio_init + (tfc.tf_ratio_final - tfc.tf_ratio_init) * progress
    return tfc.tf_ratio_final


def get_p_teacher(ss: TESSScheduleConfig, step: int) -> float:
    """Scheduled Sampling 的 ``p_teacher`` 退火值。

    Args:
        ss: Scheduled Sampling 設定。
        step: 目前訓練步數。

    Returns:
        p_teacher 退火值（[0.0, 1.0]）。
    """
    if not ss.enabled:
        return 0.0
    if step < ss.phase1_end:
        return ss.p_teacher_init
    if step < ss.phase2_end:
        progress = (step - ss.phase1_end) / max(1, ss.phase2_end - ss.phase1_end)
        return ss.p_teacher_init + (ss.p_teacher_final - ss.p_teacher_init) * progress
    return ss.p_teacher_final


def _ramp_in_window(
    step: int,
    start: int,
    end: int,
    val_init: float,
    val_final: float,
) -> float:
    """在 phase 窗 ``[start, end)`` 內線性 ramp，窗外維持端值。

    錯開 schedule 的共用線性插值核心：

    * ``step < start``：回 ``val_init``（窗前，尚未開始 ramp）。
    * ``start <= step < end``：依 ``(step-start)/(end-start)`` 線性內插。
    * ``step >= end``：回 ``val_final``（窗後，維持終值）。

    當 ``end <= start``（退化窗 / 未設定）時直接回 ``val_init``，
    等價於「不排程、維持常數」的安全 default。

    Args:
        step: 目前訓練步數。
        start: phase 窗起點（含）。
        end: phase 窗終點（不含）。
        val_init: 窗前 / ramp 起始值。
        val_final: 窗後 / ramp 終止值。

    Returns:
        該步對應的排程值（float）。
    """
    if end <= start:
        # 退化窗或未設定 → 不排程，回起始值（常數行為）
        return val_init
    if step < start:
        return val_init
    if step >= end:
        return val_final
    progress = (step - start) / (end - start)
    return val_init + (val_final - val_init) * progress


def get_dk(step: int, cfg: Any) -> float:
    """detached self-exposure 暴露深度 dk 排程（軸 2，0→大）。

    在 ``[dk_phase_start, dk_phase_end)`` 線性 ramp，窗外維持端值。

    cfg 必須包含以下欄位（直接存取，缺少則 AttributeError）：
        ``dk_phase_start``、``dk_phase_end``、``dk_init``、``dk_final``。

    Args:
        step: 目前訓練步數。
        cfg: 含 dk 排程欄位的設定物件（dataclass / SimpleNamespace 皆可）。

    Returns:
        該步的 dk 值（float；呼叫端視需要 round 成 int）。
    """
    start = int(cfg.dk_phase_start)
    end = int(cfg.dk_phase_end)
    val_init = float(cfg.dk_init)
    val_final = float(cfg.dk_final)
    return _ramp_in_window(step, start, end, val_init, val_final)


def get_lk(step: int, cfg: Any) -> float:
    """grad rollout 步數 lk 排程（軸 3，小→略大，封頂 ~8）。

    在 ``[lk_phase_start, lk_phase_end)`` 線性 ramp，窗外維持端值。

    cfg 必須包含以下欄位（直接存取，缺少則 AttributeError）：
        ``lk_phase_start``、``lk_phase_end``、``lk_init``、``lk_final``。

    Args:
        step: 目前訓練步數。
        cfg: 含 lk 排程欄位的設定物件。

    Returns:
        該步的 lk 值（float；呼叫端視需要 round 成 int）。
    """
    start = int(cfg.lk_phase_start)
    end = int(cfg.lk_phase_end)
    val_init = float(cfg.lk_init)
    val_final = float(cfg.lk_final)
    return _ramp_in_window(step, start, end, val_init, val_final)


def get_noise_sigma(step: int, cfg: Any) -> float:
    """training noise σ 排程（軸 4，0→大）。

    在 ``[noise_phase_start, noise_phase_end)`` 線性 ramp，窗外維持端值。

    cfg 必須包含以下欄位（直接存取，缺少則 AttributeError）：
        ``noise_phase_start``、``noise_phase_end``、``noise_sigma_init``、
        ``noise_sigma_final``。

    Args:
        step: 目前訓練步數。
        cfg: 含 noise 排程欄位的設定物件。

    Returns:
        該步的 noise σ（float）。
    """
    start = int(cfg.noise_phase_start)
    end = int(cfg.noise_phase_end)
    val_init = float(cfg.noise_sigma_init)
    val_final = float(cfg.noise_sigma_final)
    return _ramp_in_window(step, start, end, val_init, val_final)


def get_jac_gamma(step: int, cfg: Any) -> float:
    """Jacobian reg 收縮目標 γ 排程（軸 5，1.0→0.95 收縮漸強）。

    在 ``[jac_phase_start, jac_phase_end)`` 線性 ramp，窗外維持端值。
    γ 越小代表收縮越強（‖J‖<γ）。

    cfg 必須包含以下欄位（直接存取，缺少則 AttributeError）：
        ``jac_phase_start``、``jac_phase_end``、``jac_gamma_init``、
        ``jac_gamma_final``。

    Args:
        step: 目前訓練步數。
        cfg: 含 jac 排程欄位的設定物件。

    Returns:
        該步的 Jacobian γ（float）。
    """
    start = int(cfg.jac_phase_start)
    end = int(cfg.jac_phase_end)
    val_init = float(cfg.jac_gamma_init)
    val_final = float(cfg.jac_gamma_final)
    return _ramp_in_window(step, start, end, val_init, val_final)


def _resolve_weight_ramp(step: int, cfg: Any, prefix: str, fixed_default: float) -> float:
    """通用 weight ramp：讀 ``<prefix>_phase_start/end`` + ``<prefix>_init/final``。

    ``<prefix>_init/final`` 為 NaN（未設）時 fallback 至 ``fixed_default``（即沿用
    config 固定權重，向後相容無 schedule 的舊 config）。設定後走 `_ramp_in_window`。

    Args:
        step: 目前訓練步數。
        cfg: 設定物件。
        prefix: 欄位前綴（``tempw`` / ``jacw``）。
        fixed_default: init/final 未設時的固定 fallback 值。

    Returns:
        該步權重值（float）。
    """
    # 動態欄位名（prefix∈{dk,lk,noise,jac,tempw,jacw}）→ getattr 必要；移除 default fallback
    # （欄位皆存在於 TESTrainConfig；缺即 AttributeError，不靜默 fallback，符禁則）。
    start = int(getattr(cfg, f"{prefix}_phase_start"))
    end = int(getattr(cfg, f"{prefix}_phase_end"))

    def _val(name: str) -> float:
        v = getattr(cfg, f"{prefix}_{name}")  # nan 由下方 isnan 檢查 → fixed_default
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return fixed_default
        return float(v)

    return _ramp_in_window(step, start, end, _val("init"), _val("final"))


def get_temporal_weight(step: int, cfg: Any) -> float:
    """temporal 2CTU loss 權重排程（Phase E，0→0.2）。

    讀 ``tempw_phase_start/end`` + ``tempw_init/final``；未設時 fallback 至
    ``cfg.temporal.weight``（向後相容 noSched 固定權重）。

    Args:
        step: 目前訓練步數。
        cfg: 含 temporal sub-config 與 tempw 排程欄位的設定物件。

    Returns:
        該步的 temporal loss 權重（float）。
    """
    fixed = 0.0
    t = cfg.temporal
    if t is not None:
        fixed = float(t.weight or 0.0)
    return _resolve_weight_ramp(step, cfg, "tempw", fixed)


def get_jac_weight(step: int, cfg: Any) -> float:
    """Jacobian reg loss 權重排程（Phase D，0→0.05）。

    讀 ``jacw_phase_start/end`` + ``jacw_init/final``；未設時 fallback 至
    ``cfg.jac_reg_weight``（向後相容 noSched 固定權重）。

    Args:
        step: 目前訓練步數。
        cfg: 含 jac_reg_weight 與 jacw 排程欄位的設定物件。

    Returns:
        該步的 Jacobian reg 權重（float）。
    """
    fixed = float(cfg.jac_reg_weight or 0.0)
    return _resolve_weight_ramp(step, cfg, "jacw", fixed)


def get_batch_ratios(step: int, cfg: Any) -> list[float]:
    """batch 4 臂比例 phase gating（Phase A 只 t0/tn）。

    detached(arm2) 於 ``dk_phase_start`` 前、noise(arm3) 於 ``noise_phase_start``
    前暫不啟用，其比例併入 tn(arm1)。**僅當該軸有設未來開窗（phase_start>0）才
    gate**；phase_start<=0（noSched/smoke）→ 全臂恆開（向後相容、行為不變）。

    Args:
        step: 目前訓練步數。
        cfg: 含 dy2_batch_ratios 與 dk/noise phase 欄位的設定物件。

    Returns:
        長度 4 的有效比例 list ``[t0, tn, detached, noise]``。
    """
    base = list(cfg.dy2_batch_ratios)
    while len(base) < 4:
        base.append(0.0)
    r = [float(x) for x in base[:4]]

    dk_start = int(cfg.dk_phase_start)
    noise_start = int(cfg.noise_phase_start)
    det_on = dk_start <= 0 or step >= dk_start
    noise_on = noise_start <= 0 or step >= noise_start

    moved = 0.0
    if not det_on:
        moved += r[2]
        r[2] = 0.0
    if not noise_on:
        moved += r[3]
        r[3] = 0.0
    r[1] += moved  # 未啟用臂的比例併入 tn 臂
    return r


def assert_schedules_staggered(cfg: Any) -> None:
    """檢查各 schedule phase 窗互不重疊（錯開鐵律）。

    錯開原則：任兩個難度軸不可同時 ramp。本函式收集所有 **已啟用**
    （``end > start``，即非退化窗）的排程軸的 ``[start, end)`` 區間，
    兩兩檢查是否重疊；重疊則 raise ValueError。

    端點相接（``a.end == b.start``）視為不重疊（半開區間 ``[start, end)``）。
    未啟用（退化窗）的軸不參與檢查 —— cfg 沒設這些欄位時所有軸皆退化，
    視為「全不排程」，自然通過（安全 default）。

    涵蓋軸：dk、lk、noise、jac（與 ``get_dk`` / ``get_lk`` /
    ``get_noise_sigma`` / ``get_jac_gamma`` 讀取的欄位一致）。

    Args:
        cfg: 含各排程 phase 欄位的設定物件。

    Raises:
        ValueError: 任兩個已啟用排程軸的 phase 窗重疊。
    """
    def _axis_window(prefix: str) -> tuple[int, int]:
        """讀取 ``<prefix>_phase_start/end``；cfg 完全未定義該軸欄位時視為退化窗。

        與 ``get_dk`` 等排程函式不同（那些函式必須算出實際排程數值，缺欄位
        直接 AttributeError）：本函式只做「是否啟用」的結構性檢查，欄位整組
        不存在即代表 cfg 沒有配置這條軸的排程，等價於退化窗 ``(0, 0)``——
        對應本函式文件開頭的「全不排程視為通過」合約。
        """
        start_attr, end_attr = f"{prefix}_phase_start", f"{prefix}_phase_end"
        if not hasattr(cfg, start_attr) or not hasattr(cfg, end_attr):
            return (0, 0)
        return (int(getattr(cfg, start_attr)), int(getattr(cfg, end_attr)))

    axes: list[tuple[str, int, int]] = [
        ("dk", *_axis_window("dk")),
        ("lk", *_axis_window("lk")),
        ("noise", *_axis_window("noise")),
        ("jac", *_axis_window("jac")),
    ]
    # 只取已啟用（非退化）窗
    active = [(name, s, e) for name, s, e in axes if e > s]
    for i in range(len(active)):
        name_a, start_a, end_a = active[i]
        for j in range(i + 1, len(active)):
            name_b, start_b, end_b = active[j]
            # 半開區間 [start, end) 重疊判定：start_a < end_b 且 start_b < end_a
            if start_a < end_b and start_b < end_a:
                raise ValueError(
                    "Dy2 §5.2 錯開鐵律違反："
                    f"schedule 軸 '{name_a}' [{start_a}, {end_a}) 與 "
                    f"'{name_b}' [{start_b}, {end_b}) phase 窗重疊；"
                    "任兩難度軸不可同時 ramp，請錯開 phase 邊界。"
                )


def add_vg_composite(vg_metrics: dict[str, float]) -> dict[str, float]:
    """將 VIS-GATE composite 寫入 ``vg_metrics`` 並回傳同一個 dict（in-place）。

    Composite = clamp(ic, 0, 1) * max(ssim, 0) * (1 - clamp(hband, 0, 1))。

    Args:
        vg_metrics: VIS-GATE 指標 dict（key 為指標名稱，value 為數值）。

    Returns:
        已寫入 ``"vg_composite"`` 鍵的同一個 dict。
    """
    if not vg_metrics:
        return vg_metrics
    ic = min(max(float(vg_metrics.get("vg_IC_Copy_Score", 0.0)), 0.0), 1.0)
    ssim = max(float(vg_metrics.get("vg_StructVel_RGB_SSIM", 0.0)), 0.0)
    hband = min(max(float(vg_metrics.get("vg_HorizontalBand_FFT", 1.0)), 0.0), 1.0)
    vg_metrics["vg_composite"] = ic * ssim * (1.0 - hband)
    return vg_metrics


def select_early_stop_monitor(
    monitor: str,
    cold_metrics: dict[str, float],
    warm_metrics: dict[str, float],
    vg_metrics: dict[str, float],
) -> float:
    """從 ``cold/warm/vg`` 三組 metrics 中按優先序解析 early-stop 監控值。

    依序查找 cold_metrics → warm_metrics → vg_metrics，回傳第一個命中的數值。

    依專案鐵則「缺欄位即報錯，不得 fallback」：三組皆查無 key 時直接 raise 並
    列出當下可用的 key，使組態錯誤在第一次驗證就浮現，而不是靜默跳過整個
    early stopping（``TESEarlyStopConfig.monitor`` 預設為 ``"total"``，而 mean
    模組的 val 指標沒有 ``total`` 這個 key）。

    ``tests/nca/tes/test_early_stop_contract.py`` 的掃描只認**恆存在**的 key
    （``ALWAYS_PRESENT``）加 ``vg_`` 前綴，並另有兩支測試釘住以下兩個邊界情況：

    * **漏報**：``monitor: ke_ratio`` 會通過靜態掃描（它確實是 val 指標名），但
      ``ke_curve_convergence`` 在 KE 曲線退化時回 ``nan``，而
      ``validation/aggregate.py`` 會把非有限的 key **整個丟掉**——掃描是綠的、
      執行期照樣 ``KeyError``。「靜態掃描綠 ⇒ 執行期不炸」這個推論本身有洞。
    * **誤報**：``vg_*``（VIS-GATE 指標，``TESTrainer`` 傳的第三組）是合法 monitor，
      ``known`` 集合須納入，否則合法 config 會被判違規。

    掃描擋得住「monitor 名稱根本不存在」這一類，擋不住「名稱存在但執行期非
    有限」。保留 raise 是為了讓未來真的開啟早停時，組態錯誤在第一次驗證就
    浮現，而不是靜默跳過。

    Args:
        monitor: early-stop 監控 key 名稱。
        cold_metrics: cold-start 評估指標 dict。
        warm_metrics: warm-start 評估指標 dict。
        vg_metrics: VIS-GATE 指標 dict。

    Returns:
        對應 monitor key 的數值。

    Raises:
        KeyError: 三組 metrics 皆無 ``monitor`` 這個 key。
    """
    if monitor in cold_metrics:
        return cold_metrics[monitor]
    if monitor in warm_metrics:
        return warm_metrics[monitor]
    if monitor in vg_metrics:
        return vg_metrics[monitor]
    raise KeyError(
        f"early_stop.monitor='{monitor}' 在三組 val metrics 中皆不存在，early stopping 無法作用。"
        f" 可用 key —— cold: {sorted(cold_metrics)}；warm: {sorted(warm_metrics)}；"
        f" vis_gate: {sorted(vg_metrics)}。"
        " 請在 config 指定實際存在的 key（mean 線常用 'mean_rel_l2' 或 'mean_mse'），"
        " 或設 early_stop.enabled=false 明示不使用早停。"
    )


__all__ = [
    "sample_k",
    "get_val_k",
    "get_loss_version",
    "get_tf_ratio",
    "get_p_teacher",
    "get_dk",
    "get_lk",
    "get_noise_sigma",
    "get_jac_gamma",
    "assert_schedules_staggered",
    "add_vg_composite",
    "select_early_stop_monitor",
]
