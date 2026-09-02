"""`L0.3-a` 守衛：`vis_range` 必須來自 SSOT，不得硬編。

★ 為什麼需要這個測試（DEEP_AUDIT_2026-07 v2 `L0.3-a`）
--------------------------------------------------------
顯示範圍的單一真相是 ``stats.json`` 的 ``vis_range``
（`tes_h5_to_zarr.py` 寫入 → `FieldNormalizer` → `FieldColorizer`；
`FieldColorizer` 未注入即 raise）。實測值為 ``vel_max = 0.14`` / ``vor_range = 0.016``。

但監督者實測發現**五處硬編逃逸**，其中兩處在**評估影像輸出路徑**內
（`trainer.py` 與 `checkpoint/checkpointer.py` 的 ``_VEL_MAX = 0.20``）：

* 0.20 比 SSOT 寬 **1.43×** → 所有像素值被壓向 0 → **弱結構被系統性淡化**
* 而訓練期比對圖正是專案規則裡「**一票否決**」的審查核心
* → 偏誤方向**對「通過」有利**：缺陷看起來比實際輕

同源的第三處在 loss 側（``_VOR_RANGE = 0.04`` vs SSOT 0.016，差 **2.5×**），
其後果是 ``vort_ssim`` 在約三分之二的記錄步**恰好讀作 0**
—— 一個權重 0.3 的損失項被寫錯的常數實質關閉（見 `L0-d`）。

本檔擋的是**同一類錯誤再次出現**。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src" / "lbmnca" / "nca"

# 活路徑（`src/tes/archive/` 為歷史碼，README §7.11 明令禁 import，故不納入）
_LIVE_PATHS = [
    "tes/training/trainer.py",
    "tes/training/checkpoint/checkpointer.py",
    "tes/loss/terms/perceptual.py",
]

_GUARDED_NAMES = {"_VEL_MAX", "_VOR_RANGE", "VEL_MAX", "VOR_RANGE"}


def _hardcoded_vis_constants(path: Path) -> list[tuple[str, float, int]]:
    """回傳 `(名稱, 數值, 行號)`：被指派**數值字面值**的顯示範圍常數。

    ⚠ 必須用 AST 而非 regex —— regex 會掃到註解與 docstring 裡對舊值的說明，
    產生看起來很真實的假陽性（本專案已被同類假陽性騙過數次）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, float, int]] = []
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, (int, float)):
            continue
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id in _GUARDED_NAMES:
                out.append((tgt.id, float(value.value), node.lineno))
    return out


@pytest.mark.parametrize(
    "rel_path",
    [
        "tes/training/trainer.py",
        "tes/training/checkpoint/checkpointer.py",
        pytest.param(
            "tes/loss/terms/perceptual.py",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "DEEP_AUDIT v2 `L0-d` 尚未裁決：loss 側仍硬編 _VEL_MAX=0.20 / "
                    "_VOR_RANGE=0.04。改成 SSOT（0.14 / 0.016）會讓目前約 2/3 記錄步"
                    "讀作 0 的 vort_ssim（權重 0.3）真正生效 —— 屬**訓練行為變更**，"
                    "依專案規則須 from-scratch 驗證，且可能動到 CHAIN_PASS。"
                    "★ strict=True：L0-d 一旦修好本項會 XPASS 而失敗，"
                    "屆時**必須移除這個 xfail**，不得放著。"
                ),
            ),
        ),
    ],
)
def test_no_hardcoded_vis_range_constant_in_live_paths(rel_path: str) -> None:
    """★ 核心斷言：活路徑不得把顯示範圍寫成數值字面值。

    正確做法是由 `FieldNormalizer` 取（``fn.vel_vis_max`` / ``fn.vor_vis_range``），
    與 eval／deploy 同源。
    """
    found = _hardcoded_vis_constants(_SRC / rel_path)
    assert not found, (
        f"{rel_path} 仍硬編顯示範圍常數：{found}。"
        " SSOT 是 stats.json 的 vis_range，須改由 FieldNormalizer 取"
        "（fn.vel_vis_max / fn.vor_vis_range）。"
        " 過寬的範圍會壓縮訊號：實測 loss 側 _VOR_RANGE 寬 2.5 倍使 vort_ssim"
        " 在約 2/3 記錄步讀作 0；審查圖側寬 1.43 倍使視覺缺陷被淡化。"
    )


def test_field_normalizer_exposes_the_ssot_accessors() -> None:
    """`FieldNormalizer` 必須提供這兩個存取子 —— 否則呼叫端無處可取。"""
    from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer

    for attr in ("vel_vis_max", "vor_vis_range"):
        assert hasattr(FieldNormalizer, attr), f"FieldNormalizer 缺 {attr}"


def test_review_image_paths_read_from_the_normalizer() -> None:
    """兩處審查影像輸出必須實際讀 `fn.vel_vis_max`（不只是「沒有硬編」）。

    只斷言「沒有字面值」不夠 —— 有人可能改成別的錯誤來源。
    """
    for rel in ("tes/training/trainer.py", "tes/training/checkpoint/checkpointer.py"):
        tree = ast.parse((_SRC / rel).read_text(encoding="utf-8"))
        reads = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and n.attr == "vel_vis_max"
        ]
        assert reads, f"{rel} 未讀取 fn.vel_vis_max —— 審查圖可能仍用非 SSOT 的範圍"


# ── vis_range 注入守衛（`L0.3-a` 第 6 處：預設值＝靜默 fallback）─────────────


def _normalizer_without_vis():
    """建構一個**未注入 vis_range** 的 FieldNormalizer。"""
    import numpy as np

    from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer

    n = np.zeros(9, dtype=np.float32)
    return FieldNormalizer(
        mf_mean=n, mf_std=n + 1.0, up_mean=n, up_std=n + 1.0,
        sdf_mean=0.0, sdf_std=1.0, px_mean=0.0, px_std=1.0,
        py_mean=0.0, py_std=1.0,
        cond_mean=np.zeros(1, dtype=np.float32),
        cond_std=np.ones(1, dtype=np.float32),
    )


def test_construction_without_vis_range_is_allowed() -> None:
    """未注入 vis_range **仍可建構** —— 不需要顯示範圍的用途不該被擋。"""
    assert _normalizer_without_vis() is not None


@pytest.mark.parametrize(
    "attr", ["vel_vis_max", "vor_vis_range", "p_vis_center", "p_vis_range"]
)
def test_reading_vis_range_without_injection_raises(attr: str) -> None:
    """★ 核心斷言：**讀取**未注入的顯示範圍必須 raise，⛔ 不得回傳預設值。

    真正的風險是「讀到過期的預設」而非「建構時沒傳」，故守在讀取端：
    D1 重生成後 `vis_range` 會改變，而寫死的預設不會 ——
    任何未經 `from_stats_dict` 的路徑會靜默沿用舊值，且**看不出來**。
    """
    fn = _normalizer_without_vis()
    with pytest.raises(RuntimeError) as exc:
        getattr(fn, attr)
    assert "vis_range" in str(exc.value)
    assert "from_stats_dict" in str(exc.value), "錯誤訊息須指出正確建構方式"


def test_from_stats_dict_injects_all_four() -> None:
    """`from_stats_dict` 走完後，四個顯示範圍都必須可讀（不 raise）。"""
    import inspect

    from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer

    src = inspect.getsource(FieldNormalizer.from_stats_dict)
    for key in ("vel_max", "vor_range", "p_center", "p_range"):
        assert f'vr["{key}"]' in src, (
            f"from_stats_dict 未以 bracket 存取 vr[{key!r}] —— "
            "缺 key 必須 KeyError，不得靜默略過"
        )
