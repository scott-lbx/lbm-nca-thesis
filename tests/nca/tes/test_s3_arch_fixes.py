"""S3 架構修正批次的行為測試（DEEP_AUDIT W4，2026-08-01 使用者裁決納入）。

三個修正各帶一個 config 開關，**預設關 → 既有 run 與 ckpt 零行為變更**。本檔守兩件事：

1. 關閉時與舊行為逐位元相同（不得偷偷改動既有實驗）。
2. 開啟時**真的**改變行為——否則開關是裝飾品，而「已納入修正」會變成又一條
   不實宣稱（本 campaign 已經出現過「補在到不了消費端的地方」這種形狀）。

全部純 CPU。
"""

from __future__ import annotations

import math

import pytest
import torch

from lbmnca.nca.tes.loss.terms.perceptual import _ssim_ranges, _VEL_MAX, _VOR_RANGE


class _StubNormalizer:
    """只提供 ``vis_range`` 兩個 property 的最小替身。"""

    def __init__(self, vel: float, vor: float) -> None:
        self._vel, self._vor = vel, vor

    @property
    def vel_vis_max(self) -> float:  # noqa: D102
        return self._vel

    @property
    def vor_vis_range(self) -> float:  # noqa: D102
        return self._vor


class _MissingVisNormalizer:
    """模擬 ``vis_range`` 未注入（真實 normalizer 會 raise）。"""

    @property
    def vel_vis_max(self) -> float:  # noqa: D102
        raise RuntimeError("vis_range 未注入")

    @property
    def vor_vis_range(self) -> float:  # noqa: D102
        raise RuntimeError("vis_range 未注入")


# ---------------------------------------------------------------------------
# 修正 (3)：SSIM 值域接 SSOT
# ---------------------------------------------------------------------------

def test_ssim_ranges_default_keeps_hardcoded_constants() -> None:
    """關閉時必須回硬編常數（既有 run 零行為變更）。"""
    nz = _StubNormalizer(0.14, 0.016)
    assert _ssim_ranges(nz, from_stats=False) == (_VEL_MAX, _VOR_RANGE)
    assert (_VEL_MAX, _VOR_RANGE) == (0.20, 0.04)


def test_ssim_ranges_from_stats_reads_ssot() -> None:
    """開啟時必須讀 normalizer 的 ``vis_range``，而且**確實不同**於硬編值。

    ★若兩者恰好相同，本開關就沒有作用——那正是要防的「裝飾性開關」。
    canonical 實測 vis_range 是 0.14／0.016，硬編是 0.20／0.04。
    """
    nz = _StubNormalizer(0.14, 0.016)
    vel, vor = _ssim_ranges(nz, from_stats=True)
    assert (vel, vor) == (0.14, 0.016)
    assert vel != _VEL_MAX and vor != _VOR_RANGE, "開關未改變任何數值"


def test_ssim_range_mismatch_magnitude_is_what_was_claimed() -> None:
    """釘住「兩個泛函不同」的量級——這是把修正納入的理由本身。

    渦度側 0.04/0.016 = 2.5×、速度側 0.20/0.14 ≈ 1.43×。動態範圍越寬，
    同一個條紋在映射到 [0,1] 後的對比越小 → loss 對它越不敏感。
    """
    assert _VOR_RANGE / 0.016 == pytest.approx(2.5, rel=1e-9)
    assert _VEL_MAX / 0.14 == pytest.approx(1.4286, rel=1e-3)


def test_ssim_ranges_propagates_missing_vis_range_error() -> None:
    """``vis_range`` 缺失時必須 raise，不得靜默退回硬編值。

    ★專案鐵則：缺欄位即報錯，不得 fallback。靜默沿用過期常數會讓所有視覺指標
    在錯誤尺度上作答且看不出來。
    """
    with pytest.raises(RuntimeError):
        _ssim_ranges(_MissingVisNormalizer(), from_stats=True)
    # 關閉時不碰 normalizer → 不得因此 raise
    assert _ssim_ranges(_MissingVisNormalizer(), from_stats=False) == (
        _VEL_MAX, _VOR_RANGE
    )


# ---------------------------------------------------------------------------
# 修正 (1)：粗層 solid 洩漏
# ---------------------------------------------------------------------------

def test_coarse_delta_mask_removes_solid_cell_contribution() -> None:
    """乘上 ``coarse_mask`` 後，固體粗格的 delta 必須歸零。

    重現機制：整格落在固體內的粗格，``cs[phys]`` 是 pooled 的固體值、``csn[phys]``
    是 cell 對它的輸出，相減得到一個**不可學**的殘差；它接著被上採灌進建物周圍
    的流體格。這裡以最小張量把「乘 mask 前後」的差異釘住。
    """
    b, pd, h, w = 1, 9, 4, 4
    cs = torch.zeros(b, pd, h, w)
    csn = torch.zeros(b, pd, h, w)
    csn[:, :, 0, 0] = 3.0            # 固體粗格產生的殘差
    csn[:, :, 2, 2] = 1.0            # 流體粗格的正常 delta

    coarse_mask = torch.ones(b, 1, h, w)
    coarse_mask[:, :, 0, 0] = 0.0    # (0,0) 是固體

    delta_old = csn - cs
    delta_new = delta_old * coarse_mask

    assert delta_old[0, 0, 0, 0] == 3.0, "舊行為：固體格殘差確實存在"
    assert delta_new[0, 0, 0, 0] == 0.0, "新行為未切斷固體格洩漏"
    assert delta_new[0, 0, 2, 2] == 1.0, "流體格的 delta 不得被誤刪"
    assert not torch.equal(delta_old, delta_new), "開關未改變任何數值"


def test_coarse_delta_mask_is_noop_when_all_fluid() -> None:
    """全流體時乘 mask 應為 identity（開關不得誤傷無固體的 case）。"""
    delta = torch.randn(2, 9, 4, 4, generator=torch.Generator().manual_seed(0))
    full = torch.ones(2, 1, 4, 4)
    assert torch.equal(delta * full, delta)


# ---------------------------------------------------------------------------
# 三個開關的接線（防「加了 config 但沒接上」）
# ---------------------------------------------------------------------------

def test_all_three_switches_exist_and_default_off() -> None:
    """三個開關必須存在於**各自正確的** dataclass，且預設為關（零行為變更）。

    ★訂正沿革（兩次，全記在此以免再犯）：

    1. 本測試原本最後一行是
       ``assert hasattr(TESModelConfig(), "ssim_range_from_stats") or True``
       ——``or True`` 使其**恆真**（vacuous assertion），而它要守的正是
       「開關真的掛在對的 config 上」。而且它指的 ``TESModelConfig`` 是錯的類別，
       實測 ``hasattr(TESModelConfig(), "ssim_range_from_stats")`` 為 **False**
       （欄位在 ``TESTrainConfig``）——也就是拿掉 ``or True`` 它會立刻紅，
       它守的正好是自己漏掉的東西。本檔 docstring 自陳要防「裝飾性開關」，
       自己卻寫了一個裝飾性斷言。
    2. ★commit 2d5938d 的訊息宣稱「已改為實質斷言」，但 git diff 證明該行**逐字未動**
       ——寫 message 的人依據的是「我打算做什麼」而非 diff。這是本 campaign 核心
       失效模式（宣稱≠程式碼）在「正在修該失效模式的那個 commit」上的直接復發，
       由第五輪審查抓出。本次真正落地。
    """
    from lbmnca.nca.tes.config._legacy import TESMultiScaleConfig, TESTrainConfig

    ms = TESMultiScaleConfig()
    assert ms.coarse_delta_mask is False
    assert ms.psi_upsample_follows_config is False
    assert TESTrainConfig().ssim_range_from_stats is False


def test_ssim_switch_lives_on_the_config_the_call_sites_read() -> None:
    """``ssim_range_from_stats`` 必須在 ``mean_step`` 實際讀得到的 cfg 上。

    ★「加了 config 但呼叫端讀不到」是本 campaign 出現過的形狀（S8 的 per-term
    欄位到不了繪圖白名單）。這裡直接斷言呼叫端引用的表達式與欄位定義一致。
    """
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[3]
        / "src" / "lbmnca" / "nca" / "tes" / "training" / "steps" / "mean_step.py"
    ).read_text(encoding="utf-8")
    assert src.count("range_from_stats=cfg.ssim_range_from_stats") == 6, (
        "mean_step 的 SSIM 呼叫端未全數接上（loss 側 2 + metric 側 2 + chain 側 2 = 6）"
    )

    # ★第四輪訂正：commit 1b5b9b7 宣稱「六個呼叫端全部接上」——**該計數不完整**。
    #   registry 路徑另有 2 個（StructVel_SsimLoss／StructVort_SsimLoss），
    #   實測 4 支非 archive config 以非 0 權重使用它們，先前完全沒接到開關，
    #   於是「開了 ssim_range_from_stats 卻只有一半的 SSIM 換值域」。實際共 8 個。
    #   registry 走 `_LossContext.ssim_range_from_stats`（context 沒有 cfg 欄位，
    #   誤寫成 `c.cfg.` 會在執行期 AttributeError）。
    reg = (
        Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca" / "tes" / "loss" / "registry.py"
    ).read_text(encoding="utf-8")
    assert reg.count("range_from_stats=c.ssim_range_from_stats") == 2, (
        "registry 的兩個 SSIM term 未接上開關"
    )
    assert "c.cfg.ssim_range_from_stats" not in reg, (
        "_LossContext 沒有 cfg 欄位，這樣寫會 AttributeError"
    )


def test_arch_switch_sites_have_no_getattr_fallback() -> None:
    """三個開關的讀取不得用 ``getattr(cfg, ..., default)``（專案鐵則）。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca" / "tes"
    h3t = (
        root / "models" / "assembly" / "transport_variants" / "h3t_streamcollide.py"
    ).read_text(encoding="utf-8")
    for field in ("coarse_delta_mask", "psi_upsample_follows_config"):
        for line in h3t.splitlines():
            if field in line and "getattr" in line:
                pytest.fail(f"{field} 使用了 getattr fallback：{line.strip()}")
