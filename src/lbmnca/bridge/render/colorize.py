"""
Taichi GPU colormap kernel 集合。

所有 kernel 全在 GPU 執行。
solid 像素（mask < 0.5）一律輸出 RGB(0.15, 0.15, 0.15)。
流體像素依各 kernel 上色。

★DEEP_AUDIT R4-G9：`colorize_comfort` 把 LBM 格子速度乘上 `config.physics.lbm_to_ms`
當成 m/s 餵進 Lawson/Beaufort 分級。該係數在 `config.yaml` 現值為 **1.0＝未校準的哨兵
值**——LBM 入口速度為 0.05 lattice unit，乘 1.0 後全場都遠低於 5.5 m/s，分級因此恆綠、
畫面看起來永遠「舒適」。正確係數需要作者確定 LBM→物理的長度與速度尺度對應（D-R4-7），
屬研究設定問題，本層不自行填數。本模組提供 `comfort_scale_uncalibrated` /
`comfort_scale_warning` 供顯示層在未校準時明示，**不改變任何上色行為**。
"""
# ★勿加 `from __future__ import annotations`：它會把 kernel 的 `ti.template()`
# 註解變成字串，Taichi 解析時直接 TaichiSyntaxError（實測）。型別註解一律用 typing。
from typing import Optional

import taichi as ti

COMFORT_SCALE_SENTINEL: float = 1.0
"""`lbm_to_ms` 的未校準哨兵值。恆等換算＝「還沒有人決定尺度對應」，不是一個物理選擇。"""


def comfort_scale_uncalibrated(lbm_to_ms: float) -> bool:
    """判斷 comfort 分級的速度換算係數是否仍是未校準哨兵值。

    Args:
        lbm_to_ms: `config["physics"]["lbm_to_ms"]`。

    Returns:
        係數等於 `COMFORT_SCALE_SENTINEL` 時為 True。
    """
    return float(lbm_to_ms) == COMFORT_SCALE_SENTINEL


def comfort_scale_warning(lbm_to_ms: float) -> Optional[str]:
    """未校準時回傳警示字串，已校準回 None。

    字串刻意為純 ASCII —— Taichi GGUI 的 `gui.text` 無法渲染中文（見
    `src/ggui/window.py::_draw_status_overlay` 的既有去非 ASCII 處理）。

    Args:
        lbm_to_ms: `config["physics"]["lbm_to_ms"]`。

    Returns:
        警示字串，或 None。
    """
    if not comfort_scale_uncalibrated(lbm_to_ms):
        return None
    return ("comfort scale UNCALIBRATED (lbm_to_ms=1.0 sentinel) - "
            "Lawson/Beaufort bands are meaningless at this value")


@ti.kernel
def to_display(src: ti.template(), dst: ti.template()):
    """轉置 rgb (H, W) → display (W, H) 供 canvas.set_image。

    運算慣例 src[i, j]（i=row=y∈H, j=col=x∈W）→ dst[j, i]（x=W, y=H）。
    配合 Taichi canvas y 軸由下而上 → row 0 顯示在底部（對齊 dataset_zarr origin="lower"）。
    修正 set_image 直接吃 (H, W) 造成的壓縮/變形。
    """
    for i, j in src:
        dst[j, i] = src[i, j]


@ti.func
def sample_lut(lut: ti.template(), t: ti.f32) -> ti.Vector:
    """在 256-entry colormap LUT 中查表（GPU 端，不走 CPU）。

    t 為 [0, 1] 的歸一化強度值，超出範圍自動 clamp。
    回傳 RGB ti.Vector[3]。
    """
    idx = ti.cast(ti.math.clamp(t, 0.0, 1.0) * 255.0, ti.i32)
    return lut[idx]


_SOLID_COLOR = ti.Vector([0.15, 0.15, 0.15])


@ti.kernel
def colorize_velocity(
    vel_mag: ti.template(),
    lut_plasma: ti.template(),
    mask: ti.template(),
    vmax: ti.f32,
    rgb: ti.template(),
):
    """速度量值 → plasma colormap。

    vmax: 速度量值上界（用於歸一化），若 0 則所有流體顯示最低色。
    """
    for i, j in vel_mag:
        if mask[i, j] < 0.5:
            rgb[i, j] = _SOLID_COLOR
        else:
            t = vel_mag[i, j] / (vmax + 1e-8)
            rgb[i, j] = sample_lut(lut_plasma, t)


@ti.kernel
def colorize_vorticity(
    vorticity: ti.template(),
    lut_vort: ti.template(),
    mask: ti.template(),
    vrange: ti.f32,
    rgb: ti.template(),
):
    """渦度 → 雙色 colormap（負=青，零=黑，正=黃）。

    vrange: 渦度對稱範圍（−vrange ~ +vrange），決定顯示飽和點。
    """
    for i, j in vorticity:
        if mask[i, j] < 0.5:
            rgb[i, j] = _SOLID_COLOR
        else:
            # 映射 [-vrange, +vrange] → [0, 1]
            t = (vorticity[i, j] / (vrange + 1e-8)) * 0.5 + 0.5
            rgb[i, j] = sample_lut(lut_vort, ti.math.clamp(t, 0.0, 1.0))


@ti.kernel
def colorize_pressure(
    pressure: ti.template(),
    lut_rdbu: ti.template(),
    mask: ti.template(),
    p_center: ti.f32,
    p_range: ti.f32,
    rgb: ti.template(),
):
    """壓力場 → RdBu_r colormap（高壓=藍，低壓=紅）。

    p_center: 參考壓力（通常 rho/3 ≈ 1/3）。
    p_range:  顯示範圍半寬。
    """
    for i, j in pressure:
        if mask[i, j] < 0.5:
            rgb[i, j] = _SOLID_COLOR
        else:
            t = (pressure[i, j] - p_center) / (p_range + 1e-8) * 0.5 + 0.5
            rgb[i, j] = sample_lut(lut_rdbu, ti.math.clamp(t, 0.0, 1.0))


@ti.kernel
def colorize_ibfv(
    ibfv_tex: ti.template(),
    vel_mag: ti.template(),
    lut_plasma: ti.template(),
    mask: ti.template(),
    vmax: ti.f32,
    tex_gain: ti.f32,
    rgb: ti.template(),
):
    """IBFV 灰階紋理 × 速度量值 plasma 著色（smooth_transition Phase A）。

    底色取速度量值的 plasma colormap（與 ``colorize_velocity`` 同語意），
    再以 IBFV 平流紋理做明暗調變，呈現「沿流場流動的條紋」。solid 區輸出
    固定灰。

    Args:
        ibfv_tex: IBFV pass 輸出紋理 [H, W]，灰階 [0, 1]。
        vel_mag: 速度量值 [H, W]（底色用）。
        lut_plasma: plasma colormap LUT（256 entry）。
        mask: 幾何遮罩 [H, W]，< 0.5 = solid。
        vmax: 速度量值上界（底色歸一化）。
        tex_gain: 紋理調變強度（0 = 純底色、1 = 全紋理；建議 0.6–0.8）。
        rgb: 輸出 RGB field [H, W]（就地寫入）。
    """
    for i, j in ibfv_tex:
        if mask[i, j] < 0.5:
            rgb[i, j] = _SOLID_COLOR
        else:
            t = vel_mag[i, j] / (vmax + 1e-8)
            base = sample_lut(lut_plasma, t)
            # 紋理在 [0,1]，以 (1-gain) + gain*tex 調變明暗（避免全黑）
            shade = (1.0 - tex_gain) + tex_gain * ti.math.clamp(ibfv_tex[i, j], 0.0, 1.0)
            rgb[i, j] = base * shade


@ti.kernel
def colorize_comfort(
    vel_mag: ti.template(),
    mask: ti.template(),
    lbm_to_ms: ti.f32,
    lut_comfort: ti.template(),
    rgb: ti.template(),
):
    """速度量值 → Beaufort 步行舒適度分區。

    上色邏輯（m/s 斷點，Lawson comfort criteria 近似）：
        0   – 5.5  m/s → 綠  (BF 0-3, comfortable)
        5.5 – 10.8 m/s → 黃  (BF 4-5, acceptable)
        10.8– 17.2 m/s → 橘  (BF 6-7, uncomfortable)
        17.2+       m/s → 紅  (BF 8+,  dangerous)

    lbm_to_ms: LBM 格子速度 → m/s 換算係數（config 中設定）。
        ★為 1.0 時代表**未校準**（見模組 docstring 與 `comfort_scale_uncalibrated`），
        此時上色結果不具物理意義，顯示層須明示。本 kernel 行為不因此改變。
    lut_comfort: 預烘好的舒適度 LUT（256 entry）。
    """
    for i, j in vel_mag:
        if mask[i, j] < 0.5:
            rgb[i, j] = _SOLID_COLOR
        else:
            speed_ms = vel_mag[i, j] * lbm_to_ms
            # 映射 0~20 m/s → [0, 1]，20 m/s 以上 clamp 至最高分區
            t = ti.math.clamp(speed_ms / 20.0, 0.0, 1.0)
            rgb[i, j] = sample_lut(lut_comfort, t)
