"""
GPU 流線追蹤 Taichi kernel。

全程在 GPU，不走 CPU。
種子點 (N_SEEDS=100) 在速度場中 Euler 積分，最多 MAX_STEPS=300 步。
死路（固體/出界/速度過小）後剩餘頂點填 ti.Vector([-1.0, -1.0])。
canvas.lines() 不渲染超出 [0,1] 範圍的頂點，因此 -1 值有效跳過。

★2026-07-23：原 N_SEEDS=200/MAX_STEPS=100（dt=0.003 → 每條線最長僅覆蓋
0.3 個歸一化寬度）。200 條互相緊貼、每條又短，疊在一起視覺上呈密集交叉的
「陰影網格」，容易誤判成方向錯誤的斜紋（實測個別種子路徑方向其實正確，
見 docs/audit/0723/IBFV_STREAMLINE_INLET_FIX_LOG_2026-07-23.md Round 2）。
改成種子減半＋步數 3 倍（可覆蓋 0.9 寬度），線更長、更稀疏，觀感貼近一般
流線圖（少量長線而非大量短線）。
"""
import taichi as ti

N_SEEDS: int = 100
MAX_STEPS: int = 300


@ti.kernel
def init_seeds(seeds: ti.template(), H: int, W: int):
    """初始化均勻網格分布種子點（歸一化坐標 [0,1]×[0,1]）。

    100 個種子 = 10 列 × 10 行，x 從 0.05 到 0.95，y 從 0.05 到 0.95。
    """
    for s in seeds:
        row = s // 10
        col = s % 10
        seeds[s] = ti.Vector([
            0.05 + col * 0.10,    # x: 0.05 ~ 0.95 (10 columns)
            0.05 + row * 0.09,    # y: 0.05 ~ 0.95 (10 rows)
        ])


@ti.func
def bilinear_vel(
    vel_x: ti.template(),
    vel_y: ti.template(),
    px: ti.f32,
    py: ti.f32,
    W: int,
    H: int,
) -> ti.types.vector(2, ti.f32):
    """雙線性插值取速度值（GPU 端，供 trace_streamlines 內部呼叫）。

    px, py: 歸一化坐標 [0,1]
    回傳: ti.Vector([u, v])
    """
    xi = px * float(W) - 0.5
    yi = py * float(H) - 0.5
    x0 = ti.cast(ti.floor(xi), ti.i32)
    y0 = ti.cast(ti.floor(yi), ti.i32)
    x1 = ti.min(x0 + 1, W - 1)
    y1 = ti.min(y0 + 1, H - 1)
    x0 = ti.max(x0, 0)
    y0 = ti.max(y0, 0)

    fx = xi - ti.cast(x0, ti.f32)
    fy = yi - ti.cast(y0, ti.f32)
    fx = ti.math.clamp(fx, 0.0, 1.0)
    fy = ti.math.clamp(fy, 0.0, 1.0)

    u = (vel_x[y0, x0] * (1.0 - fx) * (1.0 - fy)
       + vel_x[y0, x1] * fx * (1.0 - fy)
       + vel_x[y1, x0] * (1.0 - fx) * fy
       + vel_x[y1, x1] * fx * fy)
    v = (vel_y[y0, x0] * (1.0 - fx) * (1.0 - fy)
       + vel_y[y0, x1] * fx * (1.0 - fy)
       + vel_y[y1, x0] * (1.0 - fx) * fy
       + vel_y[y1, x1] * fx * fy)
    return ti.Vector([u, v])


@ti.kernel
def trace_streamlines(
    vel_x: ti.template(),
    vel_y: ti.template(),
    mask: ti.template(),
    seeds: ti.template(),
    verts: ti.template(),
    dt: ti.f32,
    H: int,
    W: int,
):
    """追蹤所有種子的流線，輸出至 verts field。

    verts[s * MAX_STEPS + step] = ti.Vector([px, py])（歸一化坐標）
    終止條件：固體像素 / 出界 / 速度過小
    終止後剩餘頂點填 ti.Vector([-1.0, -1.0])（超出 [0,1] 範圍，繪製端會跳過）

    ★2026-07-23：本函式算出的軌跡經 matplotlib 驗證完全正確（自由流近水平、
    尾流正確繞行/成渦）。曾出現「畫面上一堆左下到右上的粗斜線」是繪製端
    （原 `canvas.lines`）的問題，不是這裡——已改用 `canvas.circles` 逐點畫
    (`src/ggui/window.py`)，繞開整個 line-connecting 語意。詳見
    docs/audit/0723/IBFV_STREAMLINE_INLET_FIX_LOG_2026-07-23.md。

    vel_x, vel_y: [H, W] 速度場（LBM 格子速度，已正規化）
    mask: [H, W] 0=solid, 1=fluid
    seeds: [N_SEEDS] ti.Vector.field(2)，歸一化初始位置
    verts: [N_SEEDS * MAX_STEPS] ti.Vector.field(2)，輸出頂點
    dt: 積分步長（config streamline.dt）
    H, W: 解析度
    """
    for s in range(N_SEEDS):
        px = seeds[s][0]
        py = seeds[s][1]
        alive = True

        for step in range(MAX_STEPS):
            if alive:
                # 邊界檢查
                if px < 0.0 or px > 1.0 or py < 0.0 or py > 1.0:
                    verts[s * MAX_STEPS + step] = ti.Vector([-1.0, -1.0])
                    alive = False
                else:
                    # 固體檢查（轉換為像素坐標）
                    ci = ti.cast(py * float(H), ti.i32)
                    cj = ti.cast(px * float(W), ti.i32)
                    ci = ti.max(0, ti.min(ci, H - 1))
                    cj = ti.max(0, ti.min(cj, W - 1))

                    if mask[ci, cj] < 0.5:  # solid
                        verts[s * MAX_STEPS + step] = ti.Vector([-1.0, -1.0])
                        alive = False
                    else:
                        verts[s * MAX_STEPS + step] = ti.Vector([px, py])
                        uv = bilinear_vel(vel_x, vel_y, px, py, W, H)
                        u = uv[0]
                        v = uv[1]
                        mag = ti.math.sqrt(u * u + v * v)
                        if mag < 1e-5:  # 速度過小，停止
                            alive = False
                            # 這個頂點已填入，下一個才填 -1
                        else:
                            px += u / (mag + 1e-8) * dt
                            py += v / (mag + 1e-8) * dt
            else:
                verts[s * MAX_STEPS + step] = ti.Vector([-1.0, -1.0])
