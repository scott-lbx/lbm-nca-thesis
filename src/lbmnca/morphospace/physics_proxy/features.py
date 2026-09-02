"""Physics-Proxy Feature 計算模組（Phase G4: 22-d + FULL/BBOX split）。

從 binary geometry mask 解析式計算 22 個物理代理特徵，無需執行任何 CFD 模擬。
不再依賴 R=64 canonical resize：直接對輸入 mask 計算（任何尺寸均可）。

特徵分群（共 22 維）：
    A (A1–A5): 障礙物空間分布
    B (B1–B4): 流路幾何
    C (C1–C4): 阻塞與曝露
    D (D1–D3): 壁面距離統計
    E (E1–E3): 跨特徵（幾何 × 模擬設定）
    F (F1–F3): 入口/出口邊界特徵

FULL vs BBOX crop 分組：
    FULL (6)：B2, C2, E2, F1, F2, F3 — LBM 域邊界 / 全域路徑特徵
    BBOX (16)：A1-A5, B1, B3, B4, C1, C3, C4, D1, D2, D3, E1, E3 — 設計區幾何

BBOX 自動偵測 obstacles bounding box + margin（預設 25%）。

輸入慣例：
    mask : np.ndarray [H, W], dtype uint8 / bool / int
           1（True）= 固體（建物），0（False）= 流體（自由流）
    流向為 x 方向：左 (x=0) → 右 (x=W-1)
    mask[y, x]：y = 垂直軸，x = 水平流向軸
"""

from __future__ import annotations

from collections import deque
from typing import TypedDict

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric


# ---------------------------------------------------------------------------
# 特徵名稱、維度、crop 分組
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "A1_min_inter_obstacle_gap_px",
    "A2_mean_inter_obstacle_gap_px",
    "A3_obstacle_count",
    "A4_mean_obstacle_area_ratio",
    "A5_obstacle_size_cv",
    "B1_throat_width_px",
    "B2_tortuosity",
    "B3_void_connectivity",
    "B4_max_enclosed_fluid_area_ratio",
    "C1_frontal_area_density",
    "C2_blockage_ratio",
    "C3_shelter_index",
    "C4_ud_bcr_ratio",
    "D1_mean_wall_distance_px",
    "D2_std_wall_distance_px",
    "D3_hydraulic_diameter_est_px",
    "E1_re_eff",
    "E2_blockage_corrected_velocity",
    "E3_canyon_ar_mean",
    "F1_inlet_fluid_fraction",
    "F2_inlet_entropy",
    "F3_outlet_reachability",
]

FEATURE_DIM: int = len(FEATURE_NAMES)  # 22

FULL_CROP_FEATURES: set[str] = {
    "B2_tortuosity",
    "C2_blockage_ratio",
    "E2_blockage_corrected_velocity",
    "F1_inlet_fluid_fraction",
    "F2_inlet_entropy",
    "F3_outlet_reachability",
}
BBOX_CROP_FEATURES: set[str] = set(FEATURE_NAMES) - FULL_CROP_FEATURES

DEFAULT_BBOX_MARGIN_FACTOR: float = 0.25

# Phase G4 加入 KDTree-based gap + skimage MCP-based BFS 後，native 解析度可接受。
# 預設 0 = 不 downsample（直接吃原圖）。需 downsample 時可設正整數。
DEFAULT_BBOX_MAX_DIM: int = 0
DEFAULT_FULL_MAX_DIM: int = 0


# ---------------------------------------------------------------------------
# 模擬設定 TypedDict
# ---------------------------------------------------------------------------


class SimConfig(TypedDict):
    """E1（re_eff）、E2（blockage_corrected_velocity）會讀取。其他 features 不需要。"""

    Re: float
    U_inlet_lu: float
    L_char_px: float


# ---------------------------------------------------------------------------
# BBOX 自動偵測
# ---------------------------------------------------------------------------


def auto_bbox(
    solid: np.ndarray,
    margin_factor: float = DEFAULT_BBOX_MARGIN_FACTOR,
) -> tuple[int, int, int, int]:
    """偵測 obstacles 的 tight bounding box，加 margin 後 clamp 到 mask 範圍。

    Args:
        solid: bool [H, W]，True = 固體。
        margin_factor: bbox 邊長乘以此比例作為 margin（雙邊各加）。

    Returns:
        (y0, y1, x0, x1)：half-open，可直接 solid[y0:y1, x0:x1]。
        若無 obstacles，回傳整個 mask 範圍。
    """
    H, W = solid.shape
    rows = np.where(solid.any(axis=1))[0]
    cols = np.where(solid.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, H, 0, W

    y0_tight = int(rows[0])
    y1_tight = int(rows[-1]) + 1
    x0_tight = int(cols[0])
    x1_tight = int(cols[-1]) + 1

    h_tight = y1_tight - y0_tight
    w_tight = x1_tight - x0_tight
    margin_y = int(round(h_tight * margin_factor))
    margin_x = int(round(w_tight * margin_factor))

    y0 = max(0, y0_tight - margin_y)
    y1 = min(H, y1_tight + margin_y)
    x0 = max(0, x0_tight - margin_x)
    x1 = min(W, x1_tight + margin_x)
    return y0, y1, x0, x1


# ---------------------------------------------------------------------------
# 共用 helpers
# ---------------------------------------------------------------------------


def _bfs_from_inlet(fluid: np.ndarray) -> np.ndarray:
    """從入口邊（x=0）對流體域做 BFS，回傳每個流體格到入口的步數。

    使用 skimage.graph.MCP_Geometric（C-optimized Dijkstra）；非流體格設成 inf cost
    使路徑只能走流體。比 pure Python deque 快 ~100×（在 2M pixel mask 上）。
    """
    H, W = fluid.shape
    # cost: 流體格 cost=1（每步 +1），非流體格 cost=inf（不可走）
    cost = np.where(fluid, 1.0, np.inf).astype(np.float64)
    mcp = MCP_Geometric(cost, fully_connected=False)  # 4-connectivity
    starts = [(y, 0) for y in range(H) if fluid[y, 0]]
    if not starts:
        return np.full((H, W), np.inf, dtype=np.float32)
    cum_costs, _ = mcp.find_costs(starts)
    return cum_costs.astype(np.float32)


def _label_solid(solid: np.ndarray) -> tuple[np.ndarray, int]:
    labeled: np.ndarray
    n: int
    labeled, n = ndimage.label(solid)
    return labeled, n


# ---------------------------------------------------------------------------
# Group A：障礙物空間分布（A1–A5）— BBOX
# ---------------------------------------------------------------------------


def _compute_obstacle_gaps(
    solid: np.ndarray,
    labeled: np.ndarray,
    n_obstacles: int,
    H: int,
    W: int,
) -> tuple[float, float]:
    """A1（最小間距）與 A2（平均最近鄰間距），單位 px。

    KDTree-based：對 binary mask 取每個 obstacle 的 boundary pixels（boundary =
    solid - erosion(solid)），用 cKDTree 在 boundary 點集上做 nearest neighbor
    query。比舊版 distance_transform_edt × n_obstacles 快 ~50-100×。

    ⚠ 兩點名實限制（Stage 2.6）：
      * ``n_obstacles <= 1`` 時兩者都回傳 ``max(H, W)`` 的 **sentinel**，
        那是「沒有間距可算」的哨兵值，不是間距。C-series 670 節點中有 2 個
        （0.3%）落在此分支。
      * A2 是「每個障礙**對其餘障礙的最近鄰距離**」再取平均，
        不是所有障礙配對距離的平均。

    Args:
        solid: [H, W] bool（BBOX crop；本函式僅用其形狀）。
        labeled: `ndimage.label` 的固體標記圖。
        n_obstacles: 固體連通元件數。
        H: BBOX 高。
        W: BBOX 寬。

    Returns:
        (min_gap_px, mean_nearest_neighbour_gap_px)。
    """
    fallback = float(max(H, W))
    if n_obstacles <= 1:
        return fallback, fallback

    # 取每個 obstacle 的 boundary（4-connectivity erosion）
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    boundary_pts_per_obs: list[np.ndarray] = []  # 每個 obstacle 的 boundary [N_i, 2]
    for i in range(1, n_obstacles + 1):
        comp = labeled == i
        eroded = ndimage.binary_erosion(comp, structure=structure)
        boundary = comp & ~eroded
        ys, xs = np.where(boundary)
        if ys.size > 0:
            boundary_pts_per_obs.append(np.stack([ys, xs], axis=1).astype(np.float64))
        else:
            # 單像素 obstacle：boundary = obstacle 本身
            ys, xs = np.where(comp)
            boundary_pts_per_obs.append(np.stack([ys, xs], axis=1).astype(np.float64))

    nearest_gaps: list[float] = []
    global_min = float("inf")
    for i in range(n_obstacles):
        # 把「除自己外」所有 boundary 點集合
        other_pts_list = [boundary_pts_per_obs[j] for j in range(n_obstacles) if j != i]
        if not other_pts_list:
            continue
        other_pts = np.concatenate(other_pts_list, axis=0)
        if other_pts.size == 0 or boundary_pts_per_obs[i].size == 0:
            continue
        tree = cKDTree(other_pts)
        dists, _ = tree.query(boundary_pts_per_obs[i], k=1)
        gap_i = float(dists.min())
        nearest_gaps.append(gap_i)
        global_min = min(global_min, gap_i)

    min_gap = global_min if np.isfinite(global_min) else fallback
    mean_gap = float(np.mean(nearest_gaps)) if nearest_gaps else fallback
    return min_gap, mean_gap


def _compute_obstacle_size_stats(
    labeled: np.ndarray,
    n_obstacles: int,
    H: int,
    W: int,
) -> tuple[float, float]:
    """A4（mean area ratio）、A5（size CV）。

    ⚠ A4 的分母是傳入的 ``H × W``，在 `compute_all` 的呼叫點是
    **BBOX crop 面積**，不是 LBM 全域面積、也不是基地面積 → A4 不是建蔽率
    （Stage 2.6）。A5 為變異係數、尺度不變，不受此影響。

    Args:
        labeled: `ndimage.label` 的固體標記圖。
        n_obstacles: 固體連通元件數。
        H: 分母用的高（呼叫點傳 BBOX 高）。
        W: 分母用的寬（呼叫點傳 BBOX 寬）。

    Returns:
        (mean_area_ratio, size_cv)。
    """
    if n_obstacles == 0:
        return 0.0, 0.0
    total_px = float(H * W)
    sizes = np.array(
        [float((labeled == i).sum()) / total_px for i in range(1, n_obstacles + 1)],
        dtype=np.float64,
    )
    mean_ratio = float(sizes.mean())
    cv = float(sizes.std() / (sizes.mean() + 1e-8))
    return mean_ratio, cv


# ---------------------------------------------------------------------------
# Group B：流路幾何（B1, B3, B4 — BBOX；B2 — FULL）
# ---------------------------------------------------------------------------


def _compute_throat_width(fluid: np.ndarray) -> float:
    """B1：最小自由截面（px）—— ★ 名稱誤導，見下。

    實際計算：逐 x 截面的流體像素數取最小值，
    等價於 ``H - max_column_solid_count``，即**最大阻塞截面的補集**。

    ⚠ 這**不是**「最窄流道寬度」：同一截面上多條並排的縫隙會被**加總**，
    故本量與任何一條連通縫隙的寬度無關。14 維 G_vec 中沒有任何一維量到
    最窄連通縫隙（DEEP_AUDIT_2026-07 Stage 2.6，`B2-02-物理-01`）。
    誠實名稱見 ``gvec_api.GVEC_FEATURE_SPEC``。

    Args:
        fluid: [H, W] bool，True = 流體（BBOX crop 後）。

    Returns:
        最小自由截面的流體像素數。
    """
    col_fluid_counts: np.ndarray = fluid.sum(axis=0)
    return float(col_fluid_counts.min())


def _compute_tortuosity(bfs_dist: np.ndarray, fluid: np.ndarray, W: int) -> float:
    """B2：≈(1 + 流路彎曲度) — FULL 計算（BFS inlet→outlet 走全 LBM 域）。

    ⚠ 回傳值為 ``(mean_geodesic + (W-1)) / W``，比標準 tortuosity 定義
    （測地長 / 直線長）**多了一個 ≈1 的常數偏移**；C-series 實測值域
    [1.99911, 2.03926] 即由此而來（DEEP_AUDIT_2026-07 Stage 2.6）。
    常數偏移會被下游 z-score 吸收，故**對模型數值無影響**，
    但論文附錄的定義式若寫成「平均路徑長 / 域寬」即為錯誤。

    Args:
        bfs_dist: [H, W] 自入口的測地距離場（不可達為 inf）。
        fluid: [H, W] bool，True = 流體。
        W: FULL mask 寬度。

    Returns:
        ``(mean_geodesic + W - 1) / W``；出口無可達流體格時回 inf。
    """
    outlet_fluid = fluid[:, W - 1]
    outlet_dists = bfs_dist[:, W - 1][outlet_fluid]
    finite_dists = outlet_dists[np.isfinite(outlet_dists)]
    if len(finite_dists) == 0:
        return float("inf")
    mean_path = float(finite_dists.mean()) + (W - 1)
    return mean_path / W


def _compute_max_enclosed_fluid_area(
    solid: np.ndarray,
    fluid: np.ndarray,
    H: int,
    W: int,
) -> float:
    """B4：最大封閉流體口袋面積 / BBOX 面積 — BBOX。

    ⚠ 分母同 A4 為 **BBOX crop 面積**，不是全域面積（Stage 2.6）。
    本維在 C-series canonical 語料上恆為 0（遮罩生成不含圍合中庭），
    但**並非本質退化** —— 合成中庭幾何實測可得 0.198。
    見 ``gvec_api.GVEC_INERT_DIMS_C_SERIES``。

    Args:
        solid: [H, W] bool（未使用，保留簽章相容）。
        fluid: [H, W] bool，True = 流體（BBOX crop）。
        H: BBOX 高。
        W: BBOX 寬。

    Returns:
        最大封閉流體連通元件面積 / (H × W)。
    """
    padded_fluid = np.pad(fluid.astype(np.uint8), 1, constant_values=1)
    labeled_pad, _ = ndimage.label(padded_fluid)
    outer_label = int(labeled_pad[0, 0])
    labels_orig = labeled_pad[1 : H + 1, 1 : W + 1]
    internal_fluid = fluid & (labels_orig != outer_label)
    if not internal_fluid.any():
        return 0.0
    labeled_int, n_int = ndimage.label(internal_fluid)
    if n_int == 0:
        return 0.0
    max_area = int(max((labeled_int == i).sum() for i in range(1, n_int + 1)))
    return float(max_area) / (H * W)


# ---------------------------------------------------------------------------
# Group C：阻塞與曝露（C1, C3, C4 — BBOX；C2 — FULL）
# ---------------------------------------------------------------------------


def _compute_frontal_area_density(
    fluid: np.ndarray,
    solid: np.ndarray,
    H: int,
    W: int,
) -> float:
    """C1：迎風前緣像素數 / BBOX 面積 — BBOX。

    ⚠ 名稱借用都市氣候學的 frontal area density（λ_f），但 λ_f 是**無因次量**
    （迎風投影面積 / 基地面積），本量的分子是前緣像素**個數**（長度量）、
    分母是面積 → 量綱為 [1/px]，且隨解析度變化
    （FIX_C 實測：解析度 ×3 → 本量 ×2.989）。
    僅為可計算的形狀描述子，不宣稱對應 λ_f（Stage 2.6，`B2-02-物理-05`）。

    Args:
        fluid: [H, W] bool，True = 流體（BBOX crop）。
        solid: [H, W] bool，True = 固體（BBOX crop）。
        H: BBOX 高。
        W: BBOX 寬。

    Returns:
        迎風（+x 方向）固體前緣像素數 / (H × W)。
    """
    leading_edges: np.ndarray = fluid[:, :-1] & solid[:, 1:]
    return float(leading_edges.sum()) / (H * W)


def _compute_shelter_index(fluid: np.ndarray, H: int, W: int) -> float:
    """C3：位於「該列首個固體」下游的流體佔比 — BBOX。

    逐 row 找出該列第一個固體像素的 x，把該 x 之後（含）的所有流體格計為
    sheltered。**不考慮距離衰減，也不考慮流體是否在更下游重新暴露**，
    故非空氣動力意義上的遮蔽率（Stage 2.6）。

    Args:
        fluid: [H, W] bool，True = 流體（BBOX crop）。
        H: BBOX 高。
        W: BBOX 寬（未使用，保留簽章相容）。

    Returns:
        sheltered 流體格數 / 全部流體格數。
    """
    total_fluid = int(fluid.sum())
    if total_fluid == 0:
        return 0.0
    sheltered = 0
    for y in range(H):
        row_fluid = fluid[y, :]
        solid_positions = np.where(~row_fluid)[0]
        if len(solid_positions) == 0:
            continue
        first_solid_x = int(solid_positions[0])
        sheltered += int(row_fluid[first_solid_x:].sum())
    return float(sheltered) / float(total_fluid)


def _compute_ud_bcr_ratio(solid: np.ndarray, W: int) -> float:
    """C4：上下游 BCR 對稱 log-ratio — BBOX。

    `log((up+ε) / (down+ε))`，ε=1e-3。
    """
    mid = W // 2
    upstream_bcr = float(solid[:, :mid].mean())
    downstream_bcr = float(solid[:, mid:].mean())
    eps = 1e-3
    return float(np.log((upstream_bcr + eps) / (downstream_bcr + eps)))


# ---------------------------------------------------------------------------
# Group D：壁面距離統計（D1, D2, D3）— BBOX
# ---------------------------------------------------------------------------


def _compute_wall_distance_stats(
    fluid: np.ndarray,
    solid: np.ndarray,
) -> tuple[float, float, float]:
    """D1（mean）、D2（std）、D3（4A/P 形狀描述子）— 全部 BBOX。

    ⚠ 兩點名實限制（Stage 2.6）：
      * D1/D2 的「wall」**只含建物**：`distance_transform_edt` 只把固體當 0，
        BBOX 裁切邊界與 LBM 域壁都不算牆 → 值偏大且隨 bbox margin 變動。
      * D3 的濕周長 P 同樣只計 solid–fluid 界面像素（不含域壁），
        稀疏幾何時 P 很小 → C-series 實測 493/670 節點（73.6%）的 D3
        **大於整個域高 1080 px**，不可解讀為物理上的直徑。

    Args:
        fluid: [H, W] bool，True = 流體（BBOX crop）。
        solid: [H, W] bool，True = 固體（BBOX crop）。

    Returns:
        (mean_wall_distance_px, std_wall_distance_px, 4A/P)。
    """
    dist_to_wall: np.ndarray = ndimage.distance_transform_edt(fluid)
    fluid_dists = dist_to_wall[fluid]
    if fluid_dists.size == 0:
        return 0.0, 0.0, 0.0
    mean_wall = float(fluid_dists.mean())
    std_wall = float(fluid_dists.std())
    fluid_area = int(fluid.sum())
    h_iface = int(
        (solid[:, :-1] & fluid[:, 1:]).sum() + (fluid[:, :-1] & solid[:, 1:]).sum()
    )
    v_iface = int(
        (solid[:-1, :] & fluid[1:, :]).sum() + (fluid[:-1, :] & solid[1:, :]).sum()
    )
    perimeter = h_iface + v_iface
    hydraulic_diam = 4.0 * fluid_area / (perimeter + 1e-8)
    return mean_wall, std_wall, hydraulic_diam


# ---------------------------------------------------------------------------
# Group E：跨特徵（E1, E3 — BBOX；E2 — FULL）
# ---------------------------------------------------------------------------


def _compute_canyon_ar_mean(solid: np.ndarray, H: int, W: int) -> float:
    """E3：平均峽谷深寬比 — BBOX。"""
    labeled, n = ndimage.label(solid)
    if n == 0:
        return 0.0
    x_depths: list[float] = []
    for i in range(1, n + 1):
        comp = labeled == i
        x_cols = np.where(comp.any(axis=0))[0]
        if len(x_cols) >= 1:
            x_depths.append(float(x_cols[-1] - x_cols[0] + 1))
    if not x_depths:
        return 0.0
    mean_depth = float(np.mean(x_depths))
    gap_widths: list[float] = []
    for x in range(W):
        col = solid[:, x]
        in_gap = False
        gap_start = 0
        prev_solid = False
        for y in range(H):
            is_solid = bool(col[y])
            if not is_solid and not in_gap and prev_solid:
                in_gap = True
                gap_start = y
            elif is_solid and in_gap:
                gap_widths.append(float(y - gap_start))
                in_gap = False
            prev_solid = is_solid
    if not gap_widths:
        return 0.0
    mean_gap = float(np.mean(gap_widths))
    return mean_depth / (mean_gap + 1e-8)


# ---------------------------------------------------------------------------
# Group F：入口/出口邊界（F1, F2, F3）— FULL
# ---------------------------------------------------------------------------


def _compute_inlet_entropy(solid: np.ndarray) -> float:
    """F2：入口截面 (x=0) 二元熵 — FULL。"""
    p_solid = float(solid[:, 0].mean())
    p_fluid = 1.0 - p_solid
    if p_solid <= 0.0 or p_fluid <= 0.0:
        return 0.0
    entropy = -(p_solid * np.log2(p_solid) + p_fluid * np.log2(p_fluid))
    return float(entropy)


def _compute_outlet_reachability(
    bfs_dist: np.ndarray,
    fluid: np.ndarray,
    W: int,
) -> float:
    """F3：出口可達性 — FULL。"""
    outlet_fluid_mask = fluid[:, W - 1]
    total_outlet = int(outlet_fluid_mask.sum())
    if total_outlet == 0:
        return 0.0
    outlet_dists = bfs_dist[:, W - 1]
    reached = int(np.isfinite(outlet_dists[outlet_fluid_mask]).sum())
    return float(reached) / float(total_outlet)


# ---------------------------------------------------------------------------
# 主公共函式
# ---------------------------------------------------------------------------


def _resize_bbox_if_large(
    solid_bb: np.ndarray,
    max_dim: int,
) -> np.ndarray:
    """若 bbox 邊長 > max_dim，等比 downsample 至 max_dim（保長寬比，最近鄰）。

    Args:
        solid_bb: [H, W] bool/uint8。
        max_dim: 邊長上限。

    Returns:
        possibly-resized solid bool array。
    """
    H, W = solid_bb.shape
    long_dim = max(H, W)
    if long_dim <= max_dim:
        return solid_bb.astype(bool)
    scale = max_dim / long_dim
    new_H = max(8, int(round(H * scale)))
    new_W = max(8, int(round(W * scale)))
    # 最近鄰 downsample（保 binary）
    src_y = np.linspace(0, H - 1, new_H).astype(np.int32)
    src_x = np.linspace(0, W - 1, new_W).astype(np.int32)
    return solid_bb[np.ix_(src_y, src_x)].astype(bool)


def compute_all(
    mask: np.ndarray,
    sim_cfg: SimConfig | None = None,
    bbox_margin_factor: float = DEFAULT_BBOX_MARGIN_FACTOR,
    bbox_max_dim: int = DEFAULT_BBOX_MAX_DIM,
    full_max_dim: int = DEFAULT_FULL_MAX_DIM,
) -> np.ndarray:
    """計算全部 22 個 physics-proxy 特徵（raw，未歸一化）。

    內部分兩階段：
        1. FULL mask 上計算 6 個 LBM 域邊界 features（B2, C2, E2, F1, F2, F3）
           可選 downsample 至 full_max_dim 加速 BFS
        2. 自動偵測 obstacles bbox + margin，downsample 後計算 16 個設計區 features

    Args:
        mask: Binary geometry mask [H, W]，1=固體，0=流體。
              x 方向為流向（左→右）。
        sim_cfg: 模擬設定，含 Re、U_inlet_lu、L_char_px。
                 用於 E1 (Re·L/B1) 與 E2 (U/(1-C2))。None 時這兩項 = 0。
        bbox_margin_factor: BBOX 邊距比例（預設 0.25）。
        bbox_max_dim: BBOX 最大邊長（預設 256）。0 = 不 downsample。
        full_max_dim: FULL mask 最大邊長（預設 384，加速 BFS）。0 = 不 downsample。

    Returns:
        features: [22] float32（raw 值）。順序對應 FEATURE_NAMES。

    Raises:
        ValueError: mask 全固體或全流體。
    """
    solid_native: np.ndarray = mask.astype(bool)
    H_native, W_native = mask.shape

    if not (~solid_native).any():
        raise ValueError("mask 全為固體（無流體域）")
    if not solid_native.any():
        raise ValueError("mask 全為流體（無障礙物）")

    # ===== Phase 1: FULL mask（可選 downsample）=====
    if full_max_dim > 0:
        solid_full = _resize_bbox_if_large(solid_native, full_max_dim)
    else:
        solid_full = solid_native
    fluid_full = ~solid_full
    H_full, W_full = solid_full.shape

    bfs_dist_full = _bfs_from_inlet(fluid_full)
    b2 = _compute_tortuosity(bfs_dist_full, fluid_full, W_full)
    # ★ C2「blockage_ratio」名實不符（Stage 2.6，`B2-02-物理-01`）：
    #   實際只是**入口列 x=0 的固體佔比**，與障礙物的迎風投影完全無關，
    #   且數值上恆等於 1 − f1（與同批算出的 F1_inlet_fluid_fraction 完全冗餘）。
    #   C-series canonical 語料上恆為 0（建物限於 inner domain、入口段恆空），
    #   但非本質退化 —— 貼入口量體的合成幾何實測可得 0.278。
    c2 = float(solid_full[:, 0].mean())
    f1 = float(fluid_full[:, 0].mean())
    f2 = _compute_inlet_entropy(solid_full)
    f3 = _compute_outlet_reachability(bfs_dist_full, fluid_full, W_full)

    if sim_cfg is not None:
        e2 = float(sim_cfg["U_inlet_lu"]) / (1.0 - c2 + 1e-8)
    else:
        e2 = 0.0

    # ===== Phase 2: BBOX crop（從 native 取，再可選 downsample）=====
    y0, y1, x0, x1 = auto_bbox(solid_native, margin_factor=bbox_margin_factor)
    solid_bb_native = solid_native[y0:y1, x0:x1]
    if bbox_max_dim > 0:
        solid_bb = _resize_bbox_if_large(solid_bb_native, bbox_max_dim)
    else:
        solid_bb = solid_bb_native.astype(bool)
    fluid_bb = ~solid_bb
    H_bb, W_bb = solid_bb.shape

    labeled_bb, n_obs = _label_solid(solid_bb)
    a1, a2 = _compute_obstacle_gaps(solid_bb, labeled_bb, n_obs, H_bb, W_bb)
    a3 = float(n_obs)
    a4, a5 = _compute_obstacle_size_stats(labeled_bb, n_obs, H_bb, W_bb)
    b1 = _compute_throat_width(fluid_bb)
    _, n_fluid_comps = ndimage.label(fluid_bb)
    b3 = float(n_fluid_comps)
    b4 = _compute_max_enclosed_fluid_area(solid_bb, fluid_bb, H_bb, W_bb)
    c1 = _compute_frontal_area_density(fluid_bb, solid_bb, H_bb, W_bb)
    c3 = _compute_shelter_index(fluid_bb, H_bb, W_bb)
    c4 = _compute_ud_bcr_ratio(solid_bb, W_bb)
    d1, d2, d3 = _compute_wall_distance_stats(fluid_bb, solid_bb)

    if sim_cfg is not None:
        e1 = float(sim_cfg["Re"]) * float(sim_cfg["L_char_px"]) / (b1 + 1e-8)
    else:
        e1 = 0.0
    e3 = _compute_canyon_ar_mean(solid_bb, H_bb, W_bb)

    return np.array(
        [
            a1, a2, a3, a4, a5,           # A1-A5 (BBOX)
            b1, b2, b3, b4,               # B1, B3, B4 (BBOX); B2 (FULL)
            c1, c2, c3, c4,               # C1, C3, C4 (BBOX); C2 (FULL)
            d1, d2, d3,                   # D1-D3 (BBOX)
            e1, e2, e3,                   # E1, E3 (BBOX); E2 (FULL)
            f1, f2, f3,                   # F1-F3 (FULL)
        ],
        dtype=np.float32,
    )
