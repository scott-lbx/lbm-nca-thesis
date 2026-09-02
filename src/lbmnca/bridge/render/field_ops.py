"""
物理場衍生計算 Taichi kernels。
從 9ch MRT moments 計算速度、壓力、渦度。
全程在 GPU，不走 CPU。
"""
import taichi as ti


@ti.kernel
def derive_fields(
    physics: ti.template(),
    vel_x: ti.template(),
    vel_y: ti.template(),
    vel_mag: ti.template(),
    pressure: ti.template(),
):
    """從 MRT 矩量推導速度場與壓力場。

    TES MRT 9ch 通道定義（依 FieldConverter.py MPHY_*_IDX）：
        ch0 = ρ（密度）
        ch1 = e（能量模）
        ch2 = ε（能量平方模）
        ch3 = j_x = ρ·vel_x（x 動量通量）
        ch4 = Q_x
        ch5 = j_y = ρ·vel_y（y 動量通量）
        ch6 = Q_y
        ch7 = P_xx
        ch8 = P_xy

    速度：vel_x = j_x/ρ（ch3/ch0），vel_y = j_y/ρ（ch5/ch0）
    壓力：p = ρ·cs² = ρ/3（LBM D2Q9，cs²=1/3）

    注意：physics field 需為已反正規化的 raw MRT 矩量（非 Mphy_lattice_norm）。
    反正規化：m_raw = m_norm × S_combined + M_REF
    """
    for i, j in vel_x:
        rho = physics[0, i, j]
        jx  = physics[3, i, j]   # j_x = ρ·vel_x
        jy  = physics[5, i, j]   # j_y = ρ·vel_y
        eps = 1e-8
        ux  = jx / (rho + eps)
        uy  = jy / (rho + eps)
        vel_x[i, j]    = ux
        vel_y[i, j]    = uy
        vel_mag[i, j]  = ti.math.sqrt(ux * ux + uy * uy)
        pressure[i, j] = rho / 3.0


@ti.kernel
def derive_vorticity(
    vel_x: ti.template(),
    vel_y: ti.template(),
    vorticity: ti.template(),
    H: int,
    W: int,
):
    """計算 z 向渦度：ω = ∂v/∂x − ∂u/∂y（中央差分）。"""
    for i, j in vorticity:
        ip = ti.min(i + 1, H - 1)
        im = ti.max(i - 1, 0)
        jp = ti.min(j + 1, W - 1)
        jm = ti.max(j - 1, 0)
        du_dy = (vel_x[ip, j] - vel_x[im, j]) * 0.5
        dv_dx = (vel_y[i, jp] - vel_y[i, jm]) * 0.5
        vorticity[i, j] = dv_dx - du_dy


@ti.kernel
def compute_vmax(
    vel_mag: ti.template(),
    mask: ti.template(),
    result: ti.template(),  # shape=() scalar field
):
    """計算流體區域速度最大值（Taichi atomic reduction）。"""
    result[None] = 0.0
    for i, j in vel_mag:
        if mask[i, j] > 0.5:  # 只計算流體
            ti.atomic_max(result[None], vel_mag[i, j])


@ti.kernel
def compute_stats(
    vel_mag: ti.template(),
    mask: ti.template(),
    vmax: ti.template(),   # shape=() scalar
    vmean: ti.template(),  # shape=() scalar
    count: ti.template(),  # shape=() int field
):
    """計算流體區域速度統計（最大值、均值）。"""
    vmax[None] = 0.0
    vmean[None] = 0.0
    count[None] = 0
    for i, j in vel_mag:
        if mask[i, j] > 0.5:
            ti.atomic_max(vmax[None], vel_mag[i, j])
            ti.atomic_add(vmean[None], vel_mag[i, j])
            ti.atomic_add(count[None], 1)
