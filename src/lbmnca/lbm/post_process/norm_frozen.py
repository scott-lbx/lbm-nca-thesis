r"""凍結正規化常數與正／反變換（`norm_frozen_v3`）。

## 設計原則

    out = tanh( (raw - M_REF) / S )
    raw = atanh( clip(out, ±CLIP) ) * S + M_REF

三條不可違反的性質，依序由三個設計決定達成：

**1. 零點是物理常數，正負語義不變。**
``M_REF`` 是 D2Q9 MRT 的靜止平衡態 :math:`m^{ref}(\rho_0{=}1, u{=}0)`，
**由理論推導、不由資料量測**。``tanh`` 是奇函數且單調 → 輸出的正負完全由
:math:`(raw - M_{REF})` 決定，零點恰在 :math:`raw = M_{REF}`。
ch0 因此有乾淨語義：正 = 壓縮、負 = 稀疏、0 = 靜止密度。
``S`` 必須是**單一正純量**（正負兩側同尺度），否則對稱性破壞。

**2. 反變換的像集合恆在物理可容許區內（構造性保證）。**
``clip`` 使反變換的輸出**先驗有界**：

.. math::
    raw \in [\,M_{REF} - A\,S,\ \ M_{REF} + A\,S\,],\qquad A = \operatorname{atanh}(CLIP)

於是「模型輸出什麼」與「反變換落在哪」解耦 —— 不論模型權重是隨機的、
是發散的、還是被推進一百步，反變換的結果都不可能離開這個盒。
常數的選取使該盒滿足兩條物理約束（見 :func:`_assert_admissible`）：

* **密度恆正**：:math:`A\,S_\rho < \rho_0` → :math:`\rho \in [0.8167,\ 1.1833]`
* **次音速**：:math:`\sqrt{2}\,A\,S_j / (\rho_0 - A\,S_\rho) < c_s`
  → :math:`|u| \le 0.5343 < c_s = 0.5774`，即 :math:`\mathrm{Ma} \le 0.9254`
  （:math:`|j| \le \sqrt2 \max(|j_x|,|j_y|)`，配上最小可能的 :math:`\rho`
  ——這是最壞情況組合，故為充分條件）

★ 馬赫條件是**跨通道耦合**的，逐通道的盒本身表達不了它；
此處是靠**同時**選 :math:`S_\rho` 與 :math:`S_j` 讓兩個盒的最壞組合仍次音速。
:math:`e,\ \varepsilon,\ p_{xx},\ p_{xy}` 沒有單通道的符號約束，
其盒的意義是「落在 GT 觀測範圍的 1.2 倍以上」，不是可實現性條件。

**3. 常數凍結，與資料集無關。**
``S`` **不由任何資料集量測產生**。它是依 §常數來源 一次校準後寫死的憲法。
新資料集不符合時，修的是資料集或報告限制，**不是常數** ——
一旦允許「資料不合就調常數」，跨資料集可比性就名存實亡。

## 分組：equivariance-minimal

共用尺度**只在旋轉會混合的通道之間**才是必要的：

===================  ==========================  ===================================
通道                 張量型別                    是否須共用尺度
===================  ==========================  ===================================
``rho``              scalar                      否（獨立）
``e``                scalar                      否（獨立）
``eps``              scalar                      否（獨立）
``(jx, jy)``         vector（旋轉 θ）            **是** —— 90° 旋轉會互換
``(qx, qy)``         vector（旋轉 θ）            **是**
``(pxx, pxy)``       spin-2（旋轉 2θ）           **是** —— 45° 旋轉會互換
===================  ==========================  ===================================

故自由尺度有 **6 個**，不是 4 個。把 ``e`` 與 ``eps`` 綁在一起、或把
``(jx,jy)`` 與 ``(qx,qy)`` 綁在一起，都會犧牲範圍利用率而換不到任何等變性。

軌道**內**的振幅差（``jy/jx``、``pxy/pxx``）是流場組態的物理事實，
逐通道拉平會人為破壞等變性 —— 同一物理狀態旋轉後餵進去會得到不同的正規化結果。
**這部分的不平衡不可消除**，量化值見 :data:`WITHIN_ORBIT_NOTE`。

## 常數來源（一次校準，之後凍結）

在**乾淨流體遮罩**下（見 :func:`fluid_mask_from_phi`），掃 train 三本
（D0-1 + D1-1 + D2-1）的**全母體 571 node** 的 ``mean_flow`` 與
``mean_flow_running``，取每通道 :math:`|raw - M_{REF}|` 的分位與 max，
依上表取軌道內的統計量，再依 :data:`S` docstring 的**兩條判準**定尺度。
實測值見 :data:`CALIBRATION`。

★ **校準對象是時均場，不是瞬時快照。** 範疇為 mean-only：zarr 不保存湍流時序，
模型的狀態空間裡只會有時均場（以及冷啟動的靜止態，其偏差恆為 0）。
用快照校準會讓尺度被過渡段的極值撐大 4–5 倍，而那些值根本不會進入模型 ——
白白犧牲時均場的範圍利用率，並讓馬赫條件變成綁定約束。

``S_e = 2 S_rho``（因 :math:`e^{eq} = -2\rho + 3|j|^2/\rho` 的首階項）、
``S_q = S_j``（因 :math:`q^{eq} = -j`，實測
:math:`\mathrm{corr}(j,q) = -0.999999`）—— 這兩個等式是物理關係，不是巧合。
:math:`S_\varepsilon` 則是**獨立**的（見 :data:`S`）。
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION: str = "norm_frozen_v3"
"""凍結常數的版本指紋。**任何常數變更都必須升版並重跑全部實驗。**"""

# ── 物理常數（理論推導，不隨資料變）──────────────────────────────────────
RHO_0: float = 1.0
CS: float = float(np.sqrt(1.0 / 3.0))
"""格子聲速 :math:`c_s = 1/\\sqrt3`。"""

M_REF: np.ndarray = np.array(
    [RHO_0, -2.0 * RHO_0, RHO_0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)
"""D2Q9 MRT 靜止平衡態 :math:`[\\rho, e, \\varepsilon, j_x, q_x, j_y, q_y, p_{xx}, p_{xy}]`。"""

CH_NAMES: tuple[str, ...] = (
    "rho", "e", "eps", "jx", "qx", "jy", "qy", "pxx", "pxy",
)

#: equivariance-minimal 軌道：只有這些集合內部**必須**共用尺度。
ORBITS: dict[str, tuple[int, ...]] = {
    "rho":      (0,),
    "e":        (1,),
    "eps":      (2,),   # 獨立尺度：見 S_EPS 的說明
    "momentum": (3, 5),   # (jx, jy) — vector
    "eflux":    (4, 6),   # (qx, qy) — vector
    "stress":   (7, 8),   # (pxx, pxy) — spin-2
}

# ── ★ 凍結尺度 ───────────────────────────────────────────────────────────
S_RHO: float = 0.079
S_E: float = 2.0 * S_RHO          # e ≈ -2ρ 首階
S_EPS: float = 0.112              # ★ 獨立於 S_RHO：ε 是自身的 scalar 軌道
S_J: float = 0.133
S_Q: float = S_J                  # q^eq = -j
S_P: float = 0.014

S: np.ndarray = np.array(
    [S_RHO, S_E, S_EPS, S_J, S_Q, S_J, S_Q, S_P, S_P], dtype=np.float64
)
"""凍結尺度 (9,)。**定案後永不修改。**

★★ **兩條判準，分別管兩類軌道 —— 「範圍利用率」不是判準。**

**(A) 向量與 spin-2 軌道（momentum／eflux／stress）：工作點的導數。**
:math:`p_{99.5}` 落在 :math:`|x| \\approx 0.76`，該處
:math:`\\tanh'(x) = 1 - \\tanh^2(x) \\approx 0.59`。
實測工作點：動量 :math:`|x| = 0.762`（導數 **0.588**）、
應力 :math:`|x| = 0.767`（導數 0.584）。

**(B) scalar 軌道（rho／e／eps）：共用 conv 的輸入能量占比。**
:math:`p_{99.5}` 落在 :math:`|x| \\approx 0.31`（導數 ≈ 0.91，近線性段）。

判準之所以分兩類，是因為 **scalar 三通道彼此高度冗餘**
（:math:`\\mathrm{corr}` 0.76–0.97），而 :math:`q \\equiv -j` 更是精確重複。
把 scalar 也放到 :math:`|x| = 0.76`，等於在共用 conv 的輸入端塞進近半的冗餘能量，
把動量的資訊占比擠掉 —— 實測 train 三本 90 node 的通道變異數占比：

===========  ===========  ===========
家族         :math:`|x|{=}0.76`  **本版**
===========  ===========  ===========
scalar       47.6 %       **15.1 %**
momentum     18.6 %       **30.2 %**
eflux        18.6 %       30.2 %
stress       15.1 %       24.5 %
===========  ===========  ===========

★ 三件事互相獨立，不可混為一談：

* **密度必須有振幅** —— 舊常數的 :math:`S_\\rho = 1.0` 使振幅僅 0.025，
  模型實質上不必預測密度。本版 0.303，是舊版的 12 倍。
* **動量不可壓縮** —— :math:`S_j = 0.133` 與舊版的 0.1343 幾乎相同。
  曾經把它推到 :math:`|x| = 1.535`（導數 **0.170**，少 3.5 倍）去「最大化利用率」，
  結果是峰值區的物理誤差在 norm 空間幾乎不生梯度，長推進累積過衝：
  k=100 的 `Peak_Ratio` 1.68、`HF_Band_Ratio` 1.59、`RAPSD_Shape` 差 3.8 倍，
  而 `rel_l2` 只退 8 %（L2 對頻譜退化結構性失明）。
* **scalar 不可與動量等高** —— 見上表。

其餘三條約束：

1. **不截斷**　:math:`\\tanh(\\max/S) < CLIP`。GT 不得有任何一格落在盒外。
2. **物理界**　見 :func:`_assert_admissible`。
3. **覆蓋餘裕**　:math:`A\\,S / \\max \\ge 1.25`。實測 **1.59–4.83** 倍。

軌道層級利用率 0.455–0.915（比 2.01）—— 該比值不再是最佳化目標，
故容許它大於 1，換來的是上表的能量占比。

★ :data:`ATANH_A` **不影響工作點**（:math:`x = p_{99.5}/S` 與 A 無關），
只決定盒大小，故可獨立放寬到馬赫上限而不付出任何條件數代價。

★ :data:`S_EPS` **不等於** :data:`S_RHO`：ε 是獨立的 scalar 軌道，
綁在一起會讓 ε 的尾巴反過來撐大密度的盒。
"""

ATANH_A: float = 2.32
"""狀態盒半徑（在 :math:`\\operatorname{atanh}` 空間）。``CLIP = tanh(ATANH_A)``。

上限由馬赫條件決定（:math:`A \\le 2.4705`），取 2.32。
GT 的 max 落在 :math:`|x| \\le 1.459`（tanh 0.8975），故 GT 永不被 clip；
長推進的實測峰值 :math:`|u| = 0.343`（`-7`，k=100，900 個 case×k）對盒界 0.534
還有 1.56 倍餘裕，保險絲不會在正常操作下觸發。
"""

CLIP: float = float(np.tanh(ATANH_A))
"""狀態空間的閉包半徑 ≈ 0.980869。正變換的像、反變換的定義域都是 :math:`[-CLIP, CLIP]^9`。"""

R: np.ndarray = ATANH_A * S
"""物理空間的盒半徑 (9,)：``raw ∈ M_REF ± R``。"""

CALIBRATION: dict[str, dict[str, float]] = {
    # 乾淨流體遮罩下，train **全母體 571 node** 的 `mean_flow` ＋
    # `mean_flow_running` 實測（raw 單位）。max 為精確值，分位為抽樣。
    # ★ 校準對象是時均場，不是瞬時快照 —— 範疇為 mean-only，zarr 不保存湍流時序，
    #   模型的狀態空間裡只會有時均場（以及冷啟動的靜止態）。
    "rho": {"p99.5": 0.02475, "p99.99": 0.03535, "max": 0.03794},
    "e":   {"p99.5": 0.04614, "p99.99": 0.07162, "max": 0.09558},
    "eps": {"p99.5": 0.03562, "p99.99": 0.05928, "max": 0.07394},
    "jx":  {"p99.5": 0.10132, "p99.99": 0.12736, "max": 0.13765},
    "qx":  {"p99.5": 0.10134, "p99.99": 0.12723, "max": 0.13767},
    "jy":  {"p99.5": 0.04445, "p99.99": 0.09029, "max": 0.14008},
    "qy":  {"p99.5": 0.04452, "p99.99": 0.08988, "max": 0.14010},
    "pxx": {"p99.5": 0.01074, "p99.99": 0.01674, "max": 0.02043},
    "pxy": {"p99.5": 0.00272, "p99.99": 0.00632, "max": 0.01100},
}
"""校準當時的實測分布。**僅供追溯，執行期不讀取** —— 常數已凍結在 :data:`S`。"""

WITHIN_ORBIT_NOTE: str = (
    "軌道內殘餘不平衡（不可消除，消除即破壞等變性）："
    "jy/jx = 0.60、qy/qx = 0.60、pxy/pxx = 0.35（P99.5 比值）。"
)

TANH_FP16_SAFE: float = 0.999
"""float16 儲存下仍能可靠反轉的 ``|out|`` 上限。

fp16 在 1.0 附近的間距是 :math:`2^{-11}`，而 :math:`\\mathrm{d}(\\operatorname{atanh})
/\\mathrm{d}t = 1/(1-t^2)` 在此發散 —— 超過此值的往返誤差會被量化階距放大。
:data:`CLIP` (0.980869) 已在此門檻之下，故**盒的邊界本身就是 fp16 安全的**。
"""


# ══════════════════════════════════════════════════════════════════════════
# 構造性保證的斷言
# ══════════════════════════════════════════════════════════════════════════


def _assert_admissible() -> None:
    """import 時檢查凍結常數確實滿足兩條物理約束。

    這兩條是本模組的**全部價值**：只要它們成立，反變換就不可能產生
    負密度或超音速狀態，與模型權重、推進步數、是否訓練過完全無關。

    Raises:
        AssertionError: 常數被改成不滿足物理約束的值。
    """
    rho_min = RHO_0 - R[0]
    assert rho_min > 0.0, (
        f"密度可能為負：rho_min = {rho_min:.6f}。"
        f"需 ATANH_A * S_RHO < {RHO_0}，現為 {R[0]:.6f}"
    )
    # |j| = sqrt(jx² + jy²) ≤ √2 · max(|jx|, |jy|) ≤ √2 · R_j
    u_max = np.sqrt(2.0) * R[3] / rho_min
    assert u_max < CS, (
        f"可能超音速：|u|_max = {u_max:.6f} >= c_s = {CS:.6f}。"
        f"需 sqrt(2)*ATANH_A*S_J / (RHO_0 - ATANH_A*S_RHO) < c_s"
    )
    assert CLIP <= TANH_FP16_SAFE, (
        f"CLIP = {CLIP:.6f} 超過 fp16 可靠反轉上限 {TANH_FP16_SAFE}"
    )
    assert np.all(S > 0.0), "S 必須逐通道為正純量"
    for name, idx in ORBITS.items():
        vals = {float(S[i]) for i in idx}
        assert len(vals) == 1, f"軌道 {name} 內尺度不一致（破壞等變性）：{vals}"


_assert_admissible()


def admissible_box() -> dict[str, tuple[float, float]]:
    """反變換的像集合（逐通道），以及由此導出的巨觀量界限。

    Returns:
        ``{通道名或量名: (下界, 上界)}``。任何通過 :func:`inverse` 的輸出
        都保證落在其中，**不需要任何 runtime 檢查或投影**。
    """
    out: dict[str, tuple[float, float]] = {
        n: (float(M_REF[i] - R[i]), float(M_REF[i] + R[i]))
        for i, n in enumerate(CH_NAMES)
    }
    rho_lo, rho_hi = out["rho"]
    out["|j|"] = (0.0, float(np.sqrt(2.0) * R[3]))
    out["|u|"] = (0.0, float(np.sqrt(2.0) * R[3] / rho_lo))
    out["p"] = (rho_lo / 3.0, rho_hi / 3.0)          # p = ρ c_s²
    out["Ma"] = (0.0, float(np.sqrt(2.0) * R[3] / rho_lo / CS))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 正／反變換
# ══════════════════════════════════════════════════════════════════════════


def _bcast(v: np.ndarray, ndim: int) -> np.ndarray:
    """把 (9,) 攤成可對 (9,H,W) 或 (...,9,H,W) 廣播的形狀。"""
    if ndim == 3:
        return v.reshape(9, 1, 1)
    if ndim == 4:
        return v.reshape(1, 9, 1, 1)
    if ndim == 5:
        return v.reshape(1, 1, 9, 1, 1)
    raise ValueError(f"不支援的 ndim={ndim}（預期 3/4/5）")


def forward(raw: np.ndarray) -> np.ndarray:
    """raw MRT moments → 正規化狀態 ∈ (-1, 1)。

    Args:
        raw: ``(9,H,W)`` / ``(T,9,H,W)`` / ``(B,T,9,H,W)``。

    Returns:
        與輸入同形狀的 float32。**未 clamp** —— 是否落在盒內由
        :func:`clamp_state` 決定，讓「GT 是否超出盒」可被偵測而非被靜默吸收。
    """
    n = raw.ndim
    return np.tanh((raw - _bcast(M_REF, n)) / _bcast(S, n)).astype(np.float32)


def count_nonfinite(state: np.ndarray) -> int:
    """狀態中的 ``inf`` / ``nan`` 個數。

    :func:`clamp_state` 會把它們消掉以維持全稱保證，但消掉不等於沒發生 ——
    這個計數就是為了讓「上游壞掉」不被靜默吸收。呼叫端應記錄並在非零時告警。
    """
    return int((~np.isfinite(state)).sum())


def clamp_state(state: np.ndarray) -> np.ndarray:
    """把狀態夾回閉盒 :math:`[-CLIP, CLIP]^9`，並消除非有限值。

    ★ 這是狀態空間閉包的唯一機制。模型每一步之後施加，
    則「更新算子把盒映進盒」成為構造性事實，與權重無關。

    ``nan`` 映到 0（＝靜止態 :math:`M_{REF}`）、``±inf`` 映到 :math:`\\pm CLIP`。
    若不處理，``np.clip(nan)`` 會原樣回傳 ``nan`` ——
    盒的保證就不是全稱的，而「所有通道恆在物理範圍內」這句話會有一個洞。
    非有限值的出現本身是 bug，用 :func:`count_nonfinite` 記錄，不要靜默吞掉。
    """
    s = np.nan_to_num(state, nan=0.0, posinf=CLIP, neginf=-CLIP)
    return np.clip(s, -CLIP, CLIP)


def inverse(state: np.ndarray) -> np.ndarray:
    """正規化狀態 → raw MRT moments。

    先 :func:`clamp_state` 再 ``atanh``，故輸出**無條件**落在
    :func:`admissible_box` —— 與 state 的來源、量級、是否有限皆無關。

    Args:
        state: 與 :func:`forward` 同形狀。

    Returns:
        raw moments，float64。
    """
    n = state.ndim
    c = clamp_state(state.astype(np.float64))
    return np.arctanh(c) * _bcast(S, n) + _bcast(M_REF, n)


# ══════════════════════════════════════════════════════════════════════════
# 巨觀量（可追蹤的 LBM 物理量）
# ══════════════════════════════════════════════════════════════════════════


def macroscopic(raw: np.ndarray) -> dict[str, np.ndarray]:
    """由 raw moments 取出可追蹤的 LBM 物理量。

    Args:
        raw: ``(9,H,W)``。

    Returns:
        ``rho``（密度）、``ux``/``uy``（格子速度）、``umag``、``p``（壓力
        :math:`\\rho c_s^2`）、``Ma``（馬赫數）、``pxx``/``pxy``（偏差應力矩）。
    """
    rho = raw[0]
    ux, uy = raw[3] / rho, raw[5] / rho
    umag = np.sqrt(ux ** 2 + uy ** 2)
    return {
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "umag": umag,
        "p": rho * CS ** 2,
        "Ma": umag / CS,
        "pxx": raw[7],
        "pxy": raw[8],
    }


# ══════════════════════════════════════════════════════════════════════════
# 乾淨流體遮罩
# ══════════════════════════════════════════════════════════════════════════

H5_H: int = 360
H5_W: int = 640
H5_DOWNSAMPLE: int = 3


def solid_fraction_from_png(png_path: str) -> np.ndarray:
    """由 ROI 解析度的 mask PNG 算每個 H5s 格的固體占比 :math:`\\varphi`。

    ★ 存在理由：``tes_writer`` 對 moments 用 ``cv2.INTER_AREA``（面積平均）、
    對 mask 用 ``cv2.INTER_NEAREST``（最近鄰）。兩者不一致 → 部分固體的格
    其 moments 是流固加權平均（:math:`\\rho` 量化在 :math:`k/3`），
    但 mask 可能標成流體。實測這類格佔流體的 0.098 %、100 % 貼著固體，
    且貢獻了 :math:`|\\delta\\rho|` 尾巴的全部（0.116 → 0.698）。

    Args:
        png_path: ``dataset/{ds}/masks/{geometry_id}.png``（1080×1920）。

    Returns:
        ``(360, 640)`` float64，每格的固體面積占比 ∈ [0, 1]。
    """
    from PIL import Image

    a = np.array(Image.open(png_path).convert("L"))
    exp = (H5_H * H5_DOWNSAMPLE, H5_W * H5_DOWNSAMPLE)
    if a.shape != exp:
        raise ValueError(f"mask PNG 尺寸 {a.shape} != 預期 {exp}：{png_path}")
    solid = (a < 128).astype(np.float64)   # PNG 慣例：255=free, 0=building
    return solid.reshape(H5_H, H5_DOWNSAMPLE, H5_W, H5_DOWNSAMPLE).mean(axis=(1, 3))


def fluid_mask_from_phi(h5_mask: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """乾淨流體遮罩：只有**完全不含固體**的格才算流體。

    Args:
        h5_mask: ``geometry/mask``（True = 固體，含 LBM 邊界牆）。
        phi: :func:`solid_fraction_from_png` 的輸出。

    Returns:
        ``(360, 640)`` bool，True = 純流體格。
    """
    return ~(np.asarray(h5_mask, dtype=bool) | (phi > 0.0))


__all__ = [
    "SCHEMA_VERSION", "M_REF", "S", "CLIP", "ATANH_A", "R", "CS", "RHO_0",
    "CH_NAMES", "ORBITS", "CALIBRATION", "WITHIN_ORBIT_NOTE",
    "forward", "inverse", "clamp_state", "macroscopic", "admissible_box",
    "solid_fraction_from_png", "fluid_mask_from_phi",
]
