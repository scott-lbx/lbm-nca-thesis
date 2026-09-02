"""`R.0u` 守衛：ψ-curl 動量頭與 STREAM 不得靜默共存。

★ 為什麼需要這個測試（DEEP_AUDIT_2026-07 v2 `R.0u`，D2 裁決附帶項）
--------------------------------------------------------------------
動量更新鏈為 ``j_{n+1} = warp(j_n) + curl(psi)``
（``h3t_streamcollide.py:604,609-612``）。``curl(psi)`` 恆 div-free，
但 ``warp(j_n)`` 不與散度算子交換 → **增量不再 div-free**。

已量化（``B3-03-數值-08``）：解析 div-free 場的 ``mean|div|`` 由 1.641e-07
→ 20px 位移後 2.084e-03（**×12697**，約場 RMS 的 9%）。

這會使 **D3 裁決**所依賴的「ψ-curl 增量無散度性」宣稱失效。
而在本守衛加入之前**沒有任何檢查** —— 掃全部 408 個 yaml，兩者從未共存，
所以一旦有人同時設定，得到的是**靜默錯誤的物理**，不是錯誤訊息。

本檔把「靜默」變成「立刻失敗」。
"""

from __future__ import annotations

import pytest

from lbmnca.nca.tes.config._legacy import TESModelConfig, TESTransportConfig


def _model(psi_curl: bool, skip: bool) -> TESModelConfig:
    return TESModelConfig(
        arch="h3t_streamcollide",
        psi_curl_momentum=psi_curl,
        transport=TESTransportConfig(skip=skip),
    )


def test_rejects_psi_curl_with_stream_enabled() -> None:
    """★ 核心斷言：兩者共存必須 raise，⛔ 不得靜默通過。"""
    with pytest.raises(Exception) as exc:
        _model(psi_curl=True, skip=False)
    msg = str(exc.value)
    assert "psi_curl_momentum" in msg
    assert "transport.skip" in msg


def test_error_message_points_to_the_reason_and_the_way_out() -> None:
    """錯誤訊息必須說明**為什麼**，並指出可行路徑 —— 否則下一個人只會把守衛刪掉。

    ★措辭基準隨 R5 更新（commit 298c089）：守衛訊息原寫「不再 div-free」，R5 把保證
    範圍由「恆 div-free」限縮為「**增量**在流體內部無散度」後改成中文措辭
    （`_legacy.py` `psi_curl_momentum` 欄位下方 (a)(b)(c)）。本斷言同步改抓中文詞，
    要求的仍是同一件事（訊息必須點名失效的是無散度性），並**加碼**要求點出「增量」
    ——訊息若退回「恆 div-free」這種過度宣稱，本測試仍會擋下。
    """
    with pytest.raises(Exception) as exc:
        _model(psi_curl=True, skip=False)
    msg = str(exc.value)
    assert "無散度" in msg, "須說明失效的是無散度性"
    assert "增量" in msg, "須點明失效的是『增量』無散度，不得退回『恆 div-free』的過度宣稱"
    assert "flux-form" in msg or "D3" in msg, "須指出可行路徑（flux-form 或重裁 D3）"


@pytest.mark.parametrize(
    "psi_curl, skip",
    [
        (True, True),    # ★ canonical v28：ψ-curl 開、STREAM 停用
        (False, False),  # turb 分支：STREAM 開、ψ-curl 關
        (False, True),   # 兩者皆關
    ],
)
def test_allows_every_combination_that_is_not_the_forbidden_one(
    psi_curl: bool, skip: bool
) -> None:
    """其餘三種組合必須照常通過 —— 守衛只擋一種，不得誤傷。"""
    cfg = _model(psi_curl=psi_curl, skip=skip)
    assert cfg.psi_curl_momentum is psi_curl
    assert cfg.transport.skip is skip


def test_canonical_v28_combination_is_the_allowed_one() -> None:
    """canonical `main_v4` 的組合（ψ-curl 開＋STREAM 停用）必須可建。

    這條是回歸保護：若日後有人把守衛寫反，canonical 會立刻建不起來。
    """
    cfg = _model(psi_curl=True, skip=True)
    assert cfg.psi_curl_momentum is True
    assert cfg.transport.skip is True
