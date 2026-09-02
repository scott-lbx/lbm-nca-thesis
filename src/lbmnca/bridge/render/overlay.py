"""
建物 overlay 和 UI 元素繪製。
全在 GPU（Taichi kernel）或歸一化 canvas 坐標（自動縮放）。
"""
import taichi as ti


@ti.kernel
def overlay_buildings(
    mask: ti.template(),
    rgb: ti.template(),
):
    """將建物固體疊在 rgb 場上。

    mask: [H, W]，0=solid, 1=fluid
    rgb: [H, W] ti.Vector.field(3)

    固體（含邊緣）→ RGB(0.5, 0.5, 0.5)（中灰，對齊 02 FieldColorizer.OBSTACLE_GRAY，
    reference mp4/thesis 圖皆無邊框，統一純色避免多出 GT 沒有的視覺 pattern）
    流體 → 不修改（保留 colorize 結果）
    """
    for i, j in mask:
        if mask[i, j] < 0.5:  # solid
            rgb[i, j] = ti.Vector([0.5, 0.5, 0.5])


def draw_wind_arrow(canvas: "ti.ui.Canvas"):
    """在視窗左側畫靜態入口風向箭頭（歸一化坐標，自動縮放）。

    三條水平線段表示風從左進入。
    """
    import numpy as np

    # 箭頭頭部 x 座標（左側 3%）
    x_tip = 0.03
    # 三條流線的 y 位置
    ys = [0.35, 0.50, 0.65]
    shaft_len = 0.05

    arrow_verts = []
    for y in ys:
        # 水平線段
        arrow_verts.extend([x_tip, y, x_tip + shaft_len, y])

    verts_np = np.array(arrow_verts, dtype=np.float32).reshape(-1, 2)

    # 暫時用 canvas.circles 替代 lines（GGUI lines API 需要配對 begin/end Vector field）
    # 後續優化：改用 canvas.lines 傳入 begin/end Vector.field(2) 對
    for i, y in enumerate(ys):
        canvas.circles(
            centers=np.array([[x_tip + shaft_len * 0.5, y]], dtype=np.float32),
            radius=0.003,
            color=(0.9, 0.9, 0.9),
        )
