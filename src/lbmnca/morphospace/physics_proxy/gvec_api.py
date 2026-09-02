"""G_vec 公開 API（CEDS-NCA 介面層）。

從 binary geometry mask 計算 14D G_vec（physics-proxy features 的 NCA 激活子集）。
此 API 是 lbmnca.morphospace → lbmnca.lbm / lbmnca.nca 的穩定介面。

選取的 14D（幾何純量，不含物理設定相依項 E1_re_eff / E2_blockage_corrected_velocity）：
    A1, A2, A3, A4, A5  — 障礙物空間分布（5D）
    B1, B2, B4           — 流路幾何 (3D，跳過 B3 void_connectivity)
    C1, C2, C3           — 阻塞與曝露 (3D)
    D1, D3               — 壁面距離 (2D，跳過 D2 std)
    F3                   — 出口可達性 (1D)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .features import FEATURE_NAMES, compute_all

# ── 14D 激活子集定義 ──────────────────────────────────────────────────────────

GVEC_ACTIVE_INDICES: list[int] = [
    0,   # A1_min_inter_obstacle_gap_px
    1,   # A2_mean_inter_obstacle_gap_px
    2,   # A3_obstacle_count
    3,   # A4_mean_obstacle_area_ratio
    4,   # A5_obstacle_size_cv
    5,   # B1_throat_width_px
    6,   # B2_tortuosity
    8,   # B4_max_enclosed_fluid_area_ratio
    9,   # C1_frontal_area_density
    10,  # C2_blockage_ratio
    11,  # C3_shelter_index
    13,  # D1_mean_wall_distance_px
    15,  # D3_hydraulic_diameter_est_px
    21,  # F3_outlet_reachability
]

GVEC_DIM: int = len(GVEC_ACTIVE_INDICES)   # 14
GVEC_FEATURE_NAMES: list[str] = [FEATURE_NAMES[i] for i in GVEC_ACTIVE_INDICES]

# ⚠ 本清單另有 4 份手抄副本（tes_fps_selector / tes_h5_to_zarr / scripts/dataset ×2）。
#   一致性守衛：``scripts/qa/verify_gvec_active_indices.py``
#              ＋ ``lbmnca/morphospace/tests/test_gvec_active_indices.py``
#   修改本清單後**必須**跑過該驗證，否則 dataset 生成端會靜默沿用舊索引
#   （zarr/graph.json 只有 pipeline schema 版本 tes_v4，沒有 g_vec 選取版本欄位）。


# ── ★ 名實對照表（DEEP_AUDIT_2026-07 Stage 2.6）────────────────────────────────
#
# 這 14 維的識別字沿用 2025 年 physics-proxy 設計時的命名，其中多數**名稱借用了
# 都市氣候學／流體力學的既有術語，但實際計算並不等於該術語的定義**。
#
# 處置決定（2026-07-28）：**保留識別字，附上誠實描述 metadata**，不改名。
# 理由：識別字已寫入 `the design-space statistics file`、`p_stats_combined400.json`、
# `lbm_sample_L1.json` 等既有資料檔的 `feature_names` 欄位，以及 tier1/tier3 歷史報告；
# 改名會讓這些既有產物對不上，而**改名並不會改善任何數值**。
# （zarr / graph.json 內只存裸 [14] 陣列、不存名稱，故改名與否對資料層無影響。）
#
# 論文與圖表若要顯示維度名稱，**一律引用本表的 ``honest`` 欄，不要用 ``code`` 欄**，
# 並沿用 ``claims_physics=False`` 的宣告：這些量僅為**可計算的形狀描述子，
# 不宣稱對應該物理量**。
#
# 每列欄位：
#   code           : 程式識別字（= GVEC_FEATURE_NAMES 內的字串，勿更動）
#   honest         : 誠實描述實際計算的名稱
#   scope          : "FULL"（整張 LBM 域）或 "BBOX"（障礙物 bounding box + 25% margin 裁切）
#   computed       : 實際計算內容（一句話，可對照 features.py）
#   verdict        : "符合" / "部分不符" / "名實不符"
#   claims_physics : 是否可宣稱對應同名的物理量（全部為 False，見上）
GVEC_FEATURE_SPEC: list[dict[str, object]] = [
    {
        "code": "A1_min_inter_obstacle_gap_px",
        "honest": "min_obstacle_boundary_distance_px",
        "scope": "BBOX",
        "computed": "各固體連通元件邊界點集互取 KDTree 最近鄰歐氏距離的全域最小值；"
                    "n_obstacles <= 1 時改回傳 max(H_bbox, W_bbox) 的 sentinel（非間距）",
        "verdict": "部分不符",
        "claims_physics": False,
    },
    {
        "code": "A2_mean_inter_obstacle_gap_px",
        "honest": "mean_nearest_neighbour_obstacle_distance_px",
        "scope": "BBOX",
        "computed": "每個障礙對其餘所有障礙的最近鄰距離，再取平均（非所有配對的平均）；"
                    "同樣有 n_obstacles <= 1 的 sentinel 分支",
        "verdict": "部分不符",
        "claims_physics": False,
    },
    {
        "code": "A3_obstacle_count",
        "honest": "solid_connected_component_count",
        "scope": "BBOX",
        "computed": "4-connectivity 固體連通元件數（相接的量體併為 1）",
        "verdict": "符合",
        "claims_physics": False,
    },
    {
        "code": "A4_mean_obstacle_area_ratio",
        "honest": "mean_obstacle_area_over_bbox_area",
        "scope": "BBOX",
        "computed": "各障礙像素數 / (H_bbox × W_bbox) 的平均。"
                    "分母是 bbox 裁切面積，**不是** LBM 全域面積，也不是基地面積 → 不是建蔽率",
        "verdict": "部分不符",
        "claims_physics": False,
    },
    {
        "code": "A5_obstacle_size_cv",
        "honest": "obstacle_size_coefficient_of_variation",
        "scope": "BBOX",
        "computed": "障礙面積比的 std / mean（尺度不變，分母語義不影響）",
        "verdict": "符合",
        "claims_physics": False,
    },
    {
        "code": "B1_throat_width_px",
        "honest": "min_free_cross_section_px",
        "scope": "BBOX",
        "computed": "逐 x 截面的流體像素數取最小值 == H_bbox − max_column_solid_count，"
                    "即『最大阻塞截面的補集』。多條並排縫隙會被加總，"
                    "**故不是任何一條縫隙的寬度**",
        "verdict": "名實不符",
        "claims_physics": False,
    },
    {
        "code": "B2_tortuosity",
        "honest": "one_plus_geodesic_over_domain_width",
        "scope": "FULL",
        "computed": "自 x=0 做 MCP_Geometric BFS，出口列可達格的平均測地距離**加上 (W−1)**"
                    "後除以 W → 值為 ≈(1 + 真彎曲度)，非彎曲度本身",
        "verdict": "名實不符",
        "claims_physics": False,
    },
    {
        "code": "B4_max_enclosed_fluid_area_ratio",
        "honest": "max_enclosed_fluid_pocket_over_bbox_area",
        "scope": "BBOX",
        "computed": "與外部不連通的流體連通元件之最大面積 / (H_bbox × W_bbox)；"
                    "分母同 A4 為 bbox 面積而非全域面積",
        "verdict": "部分不符",
        "claims_physics": False,
        # ★DEEP_AUDIT R4-G10 實測（`scripts/audit_gvec_degenerate_dims.py`，673 節點 /
        #   7 個 dataset）：本維**恆為 0.0**（C-series 幾何全部不產生封閉流體口袋）。
        "degenerate_on_c_series": True,
    },
    {
        "code": "C1_frontal_area_density",
        "honest": "windward_edge_pixel_count_per_bbox_area",
        "scope": "BBOX",
        "computed": "迎風前緣像素**個數**（長度量） / bbox **面積** → 量綱 [1/px]。"
                    "都市氣候學的 frontal area density λ_f 是無因次量，本量不是；"
                    "且隨解析度變（FIX_C 實測 ×3 解析度 → ×2.989）",
        "verdict": "名實不符",
        "claims_physics": False,
    },
    {
        "code": "C2_blockage_ratio",
        "honest": "inlet_column_solid_fraction",
        "scope": "FULL",
        "computed": "**只看 x=0 那一列**的固體佔比，與障礙物的迎風投影完全無關；"
                    "數值上恆等於 1 − F1_inlet_fluid_fraction（與未選入的 F1 完全冗餘）",
        "verdict": "名實不符",
        "claims_physics": False,
        # ★DEEP_AUDIT R4-G10：本維在 C-series canonical 語料上恆為 0（建物限於 inner
        #   domain、入口段恆空）→ 14D G_vec 有一維是常數，對模型不提供任何訊息。
        #   非本質退化：貼入口量體的合成幾何實測可得 0.278（見 features.py 該處註解）。
        #   量化證據由 `scripts/audit_gvec_degenerate_dims.py` 逐維產出（唯讀掃描）。
        #   ★處置為待裁決項 D-R4-6：改為真正的迎風投影阻塞比會改變 G_vec 數值，
        #   需重跑整批 zarr 的 g_vec 且需重訓 → 本欄位只是 metadata，未改任何算式。
        "degenerate_on_c_series": True,
    },
    {
        "code": "C3_shelter_index",
        "honest": "fluid_fraction_downstream_of_first_solid",
        "scope": "BBOX",
        "computed": "逐 row 找該列第一個固體像素的 x，把該 x 之後（含）的所有流體格計為 sheltered；"
                    "不考慮距離衰減，也不考慮流體是否在下游重新暴露",
        "verdict": "部分不符",
        "claims_physics": False,
    },
    {
        "code": "D1_mean_wall_distance_px",
        "honest": "mean_edt_distance_to_nearest_solid_px",
        "scope": "BBOX",
        "computed": "EDT 到最近**建物**的距離對流體格取平均。"
                    "bbox 邊界與 LBM 域壁都不算牆 → 值偏大且隨 bbox margin 變動",
        "verdict": "部分不符",
        "claims_physics": False,
    },
    {
        "code": "D3_hydraulic_diameter_est_px",
        "honest": "shape_descriptor_4A_over_solid_perimeter_px",
        "scope": "BBOX",
        "computed": "4 × 流體面積 / 濕周長，濕周長只計 solid–fluid 界面像素（不含域壁）→ "
                    "稀疏幾何時分母趨近 0；C-series 實測 493/670 節點（73.6%）的值"
                    "大於整個域高 1080 px，**不可解讀為物理上的直徑**",
        "verdict": "部分不符",
        "claims_physics": False,
    },
    {
        "code": "F3_outlet_reachability",
        "honest": "reachable_outlet_fluid_fraction",
        "scope": "FULL",
        "computed": "出口列 (x=W−1) 的流體格中，自入口 BFS 可達者的比例",
        "verdict": "符合",
        "claims_physics": False,
        # ★DEEP_AUDIT R4-G10 實測（同上，673 節點）：本維**恆為 1.0**（C-series 幾何
        #   全部完全連通，出口永遠全可達）。名實相符但在本語料無鑑別力。
        "degenerate_on_c_series": True,
    },
]

#: 誠實名稱清單，順序同 GVEC_FEATURE_NAMES。論文／圖表標籤請用這份。
GVEC_HONEST_NAMES: list[str] = [str(s["honest"]) for s in GVEC_FEATURE_SPEC]


# ── ★ 有效維度（DEEP_AUDIT_2026-07 Stage 2.5）─────────────────────────────────
#
# 下列三維在 C-series canonical 語料（670 node，4 個 dataset 合併）上樣本標準差
# **精確為 0**（n_unique == 1）。這是遮罩生成慣例的副產物 ——
# 建物限於 inner domain（入口段恆空）、不生成圍合中庭、不生成橫幅擋牆 ——
# **不是特徵本身的退化**。合成幾何實測（1920×1080）：
#
#     幾何            B4        C2        F3
#     訓練分布代表    0.000     0.000     1.000
#     圍合中庭        0.198     0.000     1.000
#     貼入口量體      0.000     0.278     1.000
#     全高橫幅牆      0.000     0.000     0.000
#
# → 三維皆對「行人風環境最關切」的都市形態有反應，只是訓練語料一個都沒有。
#
# 部署端行為：`src/lbmnca/nca/tes/data/normalizer.py::normalize_g` 對
# std <= 1e-6 的維度直接輸出 0（FIX_C 止血），故**部署端這三維同樣不注入任何資訊**，
# 有效條件維度亦為 11。理由是模型從未在這些維上收過梯度
# （訓練期 x ≡ μ ⇒ 正規化後恆 0 ⇒ 第一層對應權重欄從未更新），
# 餵真值只會乘上未訓練的隨機權重。
#
# ⚠ 退化集合是**由所載入的 stats 決定的**，不是固定 3 維：
#   單獨用 `C2-1_loop_5_2_stats.json` 時退化維為 6 個
#   （另加 A1 / A3 / B2，因該 dataset 只有 5 個 root 幾何）。
#   canonical 訓練與部署走 `stats_policy: dataset_global_runtime`（四個 stats 合併）
#   → 退化集合回到 3 維、有效維度 11。
#
# 資料層查詢工具：``scripts/qa/verify_gvec_effective_dim.py``
#: C-series canonical 語料上的常數維（名稱 → 該語料上的固定值）。
GVEC_INERT_DIMS_C_SERIES: dict[str, float] = {
    "B4_max_enclosed_fluid_area_ratio": 0.0,
    "C2_blockage_ratio": 0.0,
    "F3_outlet_reachability": 1.0,
}

#: C-series canonical 語料上的有效條件維度（14 − 3）。論文請引用此數而非 GVEC_DIM。
GVEC_EFFECTIVE_DIM_C_SERIES: int = GVEC_DIM - len(GVEC_INERT_DIMS_C_SERIES)   # 11


def find_degenerate_dims(
    g_vecs: np.ndarray,
    *,
    std_threshold: float = 1e-6,
) -> list[int]:
    """在一批 raw G_vec 上找出樣本標準差低於門檻的維度索引。

    與 `lbmnca.nca` 的 `TESCondNormalizer.DEGENERATE_STD` 同一判準（1e-6）：
    落在此清單內的維度，正規化後一律輸出 0、對模型不注入任何資訊。

    Args:
        g_vecs: [N, GVEC_DIM] raw（未正規化）G_vec 批次。
        std_threshold: 判定為常數維的樣本標準差門檻。

    Returns:
        退化維在 G_vec 內的索引清單（升序）。

    Raises:
        ValueError: 輸入形狀不是 [N, GVEC_DIM]。
    """
    arr = np.asarray(g_vecs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != GVEC_DIM:
        raise ValueError(f"g_vecs 形狀應為 [N, {GVEC_DIM}]，實際收到 {arr.shape}")
    stds = arr.std(axis=0)
    return [int(i) for i in np.where(stds <= std_threshold)[0]]


def _load_mask(mask_path: str | Path, *, white_is_solid: bool = True) -> np.ndarray:
    """從 PNG 載入 binary mask，輸出 1=固體, 0=流體。

    Args:
        mask_path: PNG 路徑。
        white_is_solid: True（預設）→ 白色(>127)=固體；
                        False → 黑色(≤127)=固體（CEDS 慣例：0=building, 255=fluid）。
    """
    img = Image.open(str(mask_path)).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if white_is_solid:
        return (arr > 127).astype(np.uint8)
    else:
        return (arr <= 127).astype(np.uint8)


def compute_gvec(mask_path: str | Path, *, white_is_solid: bool = True) -> np.ndarray:
    """從單一 mask PNG 計算 14D G_vec（raw，未歸一化）。

    Args:
        mask_path: binary geometry mask PNG 路徑。
        white_is_solid: True（預設）→ 白色=固體；
                        False → 黑色=固體（CEDS 慣例：0=building, 255=fluid）。

    Returns:
        g_vec: [14] float32，對應 GVEC_FEATURE_NAMES 順序。

    Raises:
        ValueError: mask 全固體或全流體（無效幾何）。
    """
    mask = _load_mask(mask_path, white_is_solid=white_is_solid)
    all_features = compute_all(mask, sim_cfg=None)   # [22]
    return all_features[GVEC_ACTIVE_INDICES].astype(np.float32)


def compute_gvec_from_mask(mask: np.ndarray, *, solid_is_zero: bool = True) -> np.ndarray:
    """從 in-memory binary mask 計算 14D G_vec（raw，未歸一化）。

    供 lbmnca.bridge gvec_bridge 呼叫 live 幾何（不落地 PNG）。與 :func:`compute_gvec`
    等價，唯輸入為記憶體陣列而非路徑。

    Args:
        mask: [H, W] array。solid_is_zero=True（03 mask_rasterizer 慣例）→ 0=solid、
            非 0=fluid；內部轉為 compute_all 期望的 1=solid / 0=fluid。
        solid_is_zero: mask 極性。False → 直接視為 1=solid / 0=fluid（同 _load_mask 輸出）。

    Returns:
        g_vec: [14] float32，對應 GVEC_FEATURE_NAMES 順序。

    Raises:
        ValueError: mask 全固體或全流體（無效幾何）。
    """
    arr = np.asarray(mask)
    if solid_is_zero:
        solid = (arr < 0.5).astype(np.uint8)   # 0=solid → 1=solid
    else:
        solid = (arr > 0.5).astype(np.uint8)
    all_features = compute_all(solid, sim_cfg=None)   # [22]
    return all_features[GVEC_ACTIVE_INDICES].astype(np.float32)


def compute_gvec_batch(
    mask_paths: Sequence[str | Path],
    *,
    white_is_solid: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """批量計算 G_vec。失敗的 mask 對應列為 NaN。

    Args:
        mask_paths: mask PNG 路徑序列。
        white_is_solid: mask 慣例（見 compute_gvec）。
        verbose: 是否印出每個 mask 的進度。

    Returns:
        g_vecs: [N, 14] float32，對應輸入順序。
    """
    n = len(mask_paths)
    g_vecs = np.full((n, GVEC_DIM), np.nan, dtype=np.float32)
    for i, path in enumerate(mask_paths):
        if verbose:
            print(f"  [{i + 1}/{n}] {Path(path).name}")
        try:
            g_vecs[i] = compute_gvec(path, white_is_solid=white_is_solid)
        except Exception as exc:
            print(f"  [Warning] 跳過 {Path(path).name}: {exc}")
    return g_vecs
