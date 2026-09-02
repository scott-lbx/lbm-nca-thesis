"""
case_config.py — Generic C-series YAML config + dataset_meta generator.

Reads dataset/{name}/cases.json → generates:
  - dataset/{name}/configs/{case_id}.yaml  (per case)
  - dataset/{name}/dataset_meta.json

Handles node types:
  - ic_type="cold"  + status=None/pending → standard cold YAML
  - ic_type="cold"  + status="precomputed" → skip (use parent raw dir)
  - ic_type="warm"                         → warm YAML with parent_h5_path
  - sim_subtype="shortcut"                 → warm YAML with shortcut parent

Usage:
    python -m lbmnca.lbm.config_gen.case_config --dataset-name C0-1_Cold_M30C30
    python -m lbmnca.lbm.config_gen.case_config --dataset-name C0-2_Warm_M30C30R4S10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lbmnca.lbm.config_gen.config_utils.feasibility import check_feasibility
from lbmnca.morphospace.physics_proxy.lchar_api import l_char_from_buildings
from lbmnca.paths import data_root, dataset_dir

# ── Physics constants (same as gen_s1_5_configs.py) ──────────────────────────
U_INLET       = 0.05
RHO_OUT       = 1.0
RHO_IN        = RHO_OUT + 1.5 * U_INLET**2   # ~1.00375
MASK_W        = 1920
MASK_H        = 1080

SPONGE_IN  = 80;  SPONGE_OUT = 240; SPONGE_TOP = 90;  SPONGE_BOT = 90
BUFFER_IN  = 80;  BUFFER_OUT = 240; BUFFER_TOP = 90;  BUFFER_BOT = 90

NX_SOLVER = MASK_W + SPONGE_IN + BUFFER_IN + SPONGE_OUT + BUFFER_OUT   # 2560
NY_SOLVER = MASK_H + SPONGE_TOP + BUFFER_TOP + SPONGE_BOT + BUFFER_BOT  # 1440

EMBED_COL  = SPONGE_IN  + BUFFER_IN   # 160
EMBED_ROW  = SPONGE_TOP + BUFFER_TOP  # 180

# ★S1（2026-07-29）：入口速度 cosine ramp（緩啟動）。
#   實測依據 STAGE1_12「真實 C-series warmup 外推驗證」：三個真實 case 的
#   warmup 效應 Δ_mean = 0.233 / 0.280 / 0.351，佔模型誤差 0.229 的
#   **102% / 122% / 153%**——`warmup_steps=0` 的 impulsive start 造成的 GT 污染
#   比模型本身的誤差還大。機制：impulsive start 產生 water-hammer 密度波
#   δρ = ρ·u/c_s = 8.66%，而流體動力壓力訊號僅約 0.4%（訊噪比 ~1:20），
#   且 600 CTU 後仍未衰乾淨（實測 cylinder benchmark 殘留 2.1% 流向梯度）。
#   STAGE1_9 另證：10 CTU ramp 是唯一讓 4 個尾流探針主峰收斂到同一頻率的設定。
#   步數逐 case 由 L_char 決定（1 CTU = L_char / u_inlet 步）。
WARMUP_CTUS = 10.0

# ★S1（2026-07-29）：求解迴圈內時均累加器（Stage 1.15/1.16）。
#   **不設此區塊 = 累加器關閉**（solver 會印「沿用 snapshot 事後平均」），
#   那正是 D1/C1 鏈取樣混疊的根因，故 D-series 一律啟用。
#   ★段長政策（2026-07-29 改為固定 CTU）：原設計為「k 個聲學週期」，理由是
#   「窗長取週期整數倍 → 相位殘差歸零」。P2-C 複核以同一 case 的兩份平行 bench
#   直接對測後**證偽該理由**：同窗長下整數週期與非整數週期的時均誤差差異 <0.001
#   （40 CTU 窗：0.0553 vs 0.0550），分通道段間離散比 1.25–2.00 全落在純隨機噪聲
#   預測值 1.734 附近，無任何通道呈現相干模態主導的特徵。機制上聲學駐波是**空間
#   相干**模態、主宰全域積分量（KE），但**逐點時均場**誤差由局部去相關的湍流主導，
#   聲學僅貢獻約 6%（平方和）。
#   改固定 CTU 的實益：聲學週期 T_ac = 893.5/L_char CTU，生產 L_char∈[25,220] 使
#   T_ac∈[4.06, 35.74] CTU；沿用聲學段長時，小 L_char 的 case 在 100 CTU 內只有
#   2–3 段（半窗自檢退化、窗起點被靜默推遲最多 35.7 CTU）。固定 5 CTU（≈1 個渦脫
#   週期）對所有 L_char 都給出一致粒度，儲存量與原設定相當。
MEAN_ACCUM_ENABLED           = True
MEAN_ACCUM_SEGMENT_CTU       = 5.0   # 段長（CTU）；≈1 個渦脫週期，與 L_char 無關
# ★2026-07-30 由 1 提到 20：實測 stride=1 讓吞吐從 806 掉到 661 步/s（-18%），
#   而累加的統計價值由**物理自相關時間**決定、不是取樣數——實測積分自相關約
#   2.7 CTU（L=220 時 = 11880 步），stride=20 的取樣間隔 100 步仍遠小於它，
#   時均誤差完全不變。換句話說 stride=1 有 95% 的累加是白做的。
MEAN_ACCUM_STRIDE            = 20

# ⚠ 以下這段推導**已於 2026-07-29 作廢**，保留供追溯：
#   「P2-C 實測 SE ≈ 0.916·T^(−0.611)，故 200 CTU 使 GT 雜訊佔模型誤差 14%」。
#   該指數經同一 case 的兩窗長對測證偽（見下方 BURNIN_RE_FACTOR 註解 §2）。
# ★★2026-07-29 L3 驗收：TOTAL_CTUS 由固定值改為**逐 case 依 Re 計算**。
#
#   總長 = WARMUP_CTUS（ramp）+ burn_in（丟棄）+ MEAN_WINDOW_CTUS（時均窗）
#
#   實測依據（`bench_l3_consistency.py`，六個代表性 case）：
#   1. 動量擴散橫跨 L_char 的時間有閉式：
#          L²/ν = L² / (u·L/Re) = Re · (L/u) = **Re 個 CTU**
#      四個 case 完美吻合（Re=100→100.0、500→500.0、300→300.0、300→300.3 CTU）。
#   2. R1（Re=100）在 ramp 固定 10 CTU 下掃 burn-in，出現明確階梯：
#          0.5×Re → 半窗差 14.25%（FAIL）
#          1.0×Re → 半窗差  4.47%（PASS）   ← 轉折點
#          1.5×Re → 半窗差  4.36%（報酬遞減）
#   3. 舊設定（burn-in 僅 0.2×total = 40 CTU）在 R1/R3/R5 上半窗差達
#      15.6% / 34.0% / 26.7%——**GT 時均大部分落在瞬態內**。
#
#   ★同時作廢原 TOTAL_CTUS=200 的推導依據：該值源於 P2-C 的
#     `SE ≈ 0.916·T^(−0.611)`，而該指數已於 2026-07-29 實測證偽——同一 case
#     量 40 與 160 CTU 兩點得到的指數逐 case 為 +0.046 / −0.228 / −0.338，
#     不服從任何 T^(−p) 律，因為半窗差由**瞬態殘留**（系統性）主導而非統計雜訊。
#
#   ★warm case 的係數較小：它由 parent 的收斂態出發，瞬態本就短。
#     ⚠ WARM_BURNIN_RE_FACTOR 尚未以 warm case 實測驗證，暫取 0.2（= 舊協定的
#       等效比例），列為待驗項；cold 的 1.0 已有上述實測。
# ★2026-07-30 由 1.0 降到 0.3。原值建立在**半窗差**這個度量上（R1 掃描的
#   35.4% → 11.25% 階梯），但半窗差只有 1 個自由度、且含每半內部的變異，
#   實測同窗長僅滑動起點 40 CTU 就跳動 2 倍。改用 batch-means（MCMC 標準估計）
#   後重測，burn-in 對 SE 幾乎無影響：
#       c0010 窗200：burn 0.3xRe → 4.8%   burn 1.0xRe → 6.2%
#       c0034 窗200：burn 0.3xRe → 17.4%  burn 1.0xRe → 16.7%
#   進一步直接比較兩者的**時均場本身**（而非 SE），差異/雜訊 = 0.97-1.63
#   （純雜訊的期望值是 1.0）——縮短 burn-in 引入的偏差被統計雜訊淹沒。
#   在雜訊本身就有 5-25% 的情況下，多花 70% 計算時間消除一個 1.4x 雜訊的
#   偏差不划算。0.3 保留了對「明顯瞬態」的防護，同時省掉 70% 的丟棄成本。
BURNIN_RE_FACTOR      = 0.3     # cold：burn_in(CTU) = factor × Re
WARM_BURNIN_RE_FACTOR = 0.2     # warm：待實測驗證
# ★2026-07-29 由 80 提到 200：修正 G5.1 的度量缺陷後（分母原被 rho≈1 的常數項
#   稀釋 20-50 倍）實測，生產設定下的真實單窗 SE 為
#       c0027(Re=63) 1.53%   c0010(Re=269) 17.69%   c0034(Re=480) 26.90%
#   ——80 CTU 對中高 Re 明顯不足。窗長掃描（用同一份 H5 重組，未重跑）：
#       c0010 burn150：窗 80->SE 15.35%  120->7.30%  200->6.91%
#       c0034 burn300：窗 80->SE 56.60%  120->18.23% 260->22.16%
#   200 CTU 讓中低 Re 真正達標；c0034 卡在 ~22% 是週期 >260 CTU 的低頻尾流
#   擺動所致，屬物理限制（外推需 867 CTU 才到 10%），不再加長。
# ★2026-07-30 由 200 降到 150：batch-means 實測兩者幾乎相同
#   （c0010 5.1% → 5.9%、c0034 18.5% → 17.7%），但省 50 CTU/case。
MEAN_WINDOW_CTUS      = 150.0   # 時均窗長（burn-in 之後全取）
SPINUP_CTUS       = 0.0
SMAGORINSKY       = 0.1
COMPUTE_STEP_SIZE = 200
# ★早停關閉（2026-07-29）：P2-C 以零漂移純正弦做 null-control 證明，現行兩個
#   判準都是「(窗長/主導週期) 與相位」的函數、不是穩態性的函數——生產參數
#   （window=5.0）下對一個**真值漂移恆為 0** 的訊號，two-window 判 FAIL
#   (ratio=0.283)、trend gate 也判 FAIL (drift=1.193)，型一誤差 ≈100%。
#   真實軌跡（Re=100 圓柱 600 幀）：526 次檢查中 ratio<0.05 僅 17 次、**最長連續
#   命中 1 次**，k_required=3 永不可能達成；取樣相位/間距掃描 16 組全部 never。
#   實務後果是所有振盪 case 都跑滿預算後 hard_cutoff，早停等於死條文，卻讓
#   收斂與未收斂的 case 拿到**不同窗長**（52 vs 80 CTU → SE 差 30%）的異質 GT。
#   → 明確關閉早停，讓時均窗成為**固定協定常數**（burn-in 0.2×total 之後全取），
#     消除跨 case 的 GT 品質異質。三個 CONV_* 參數保留供診斷欄位使用。
ENABLE_CONV_CHECK = False
CONV_WINDOW_CTU   = 5.0
CONV_EPS          = 0.05
# ★ 持續性收斂判準（2026-07-26，DEEP_AUDIT_2026-07 修正 1+3）。
# 舊做法為「two-window ratio 首次低於 eps 即永久閂鎖」，TIER0 實測顯示那是在量
# 「首次隨機低於門檻的時刻」而非持續狀態：母體 74.6% 卡在 floor，宣告收斂後
# 89% 的 case 重套同一判準會 FAIL，KE 仍中位再漲 +82%。
#   CONV_K_REQUIRED：需連續成立次數。3 次涵蓋 15 CTU，以實測 T_shed≈8.3 CTU
#                    計約 1.8 個脫落週期，足以排除隨機夾點。
#   CONV_CUM_EPS   ：trend-line 累積漂移閘門（arXiv:2511.22618）。
CONV_K_REQUIRED   = 3
CONV_CUM_EPS      = 0.05
CONV_TREND_GATE   = True
EXTRA_CTU_CONV    = 50.0   # cold 收斂後時均窗 ≥~10 渦脫週期（audit B4：舊 10≈2 週期，時均殘留 ~10-20%）
GHOST_MOMENTS_S   = 1.2
SPONGE_STRENGTH   = 3.0

# ── Warm-only overrides (uniform 0.2 CTU, mature 50 CTU, 2026-07-06) ──
# 全程 0.2 CTU 均勻取樣（取消原 post-conv 1.0 coarse）；收斂後成熟窗 50 CTU，
# 對應 turb 50-CTU horizon 目標（@0.2 spacing = 250 frames 成熟段）。
# ⚠ WARM_TOTAL_CTUS 已作廢（2026-07-29）：warm 的總長改由
#   WARM_BURNIN_RE_FACTOR 逐 case 算出。保留常數僅為避免舊引用崩潰。
WARM_TOTAL_CTUS            = 90.0    # DEPRECATED，不再被 build_case_config 使用
WARM_EXTRA_CTU_CONV        = 50.0    # 收斂後成熟窗時長（= 50 CTU）
WARM_POST_CONV_SPACING_CTU = 0.2     # 均勻 0.2（原 1.0 coarse 取消）


def compute_l_char(buildings: list[list[float]]) -> int:
    # ★ 2026-07-26：委派至 L_char 單一真相（lbmnca.morphospace.physics_proxy.lchar_api）。
    # 公式不變（迎風橫向 y-extent，即 tuple (cx,cy,w,h) 的 max(h)），只是不再各自內嵌——
    # 原本 5 個檔案各有一份，其中 2 份落後未同步（見 FIX_B_Lchar統一.md）。
    return l_char_from_buildings(buildings)


def compute_bcr(buildings: list[list[float]]) -> float:
    return sum(w * h for _cx, _cy, w, h in buildings) / (MASK_W * MASK_H)


def resolve_parent_h5(node: dict, dataset_name: str) -> str:
    """Return absolute path to parent H5 file."""
    parent_ds   = node.get("parent_dataset", dataset_name)
    parent_id   = node.get("parent_case_id", "")
    if not parent_id:
        raise ValueError(f"warm node {node['case_id']} missing parent_case_id")
    raw_dir = dataset_dir(parent_ds) / "raw"
    return str(raw_dir / f"{parent_id}.h5")


def build_yaml_config(node: dict, dataset_name: str,
                      masks_dir: Path, ws_root: Path) -> dict[str, Any]:
    """組出單一 case 的求解器 YAML config。

    Args:
        node: cases.json 的一筆節點。
        dataset_name: 本 dataset 名稱。
        masks_dir: mask PNG 所在目錄。
        ws_root: 保留給呼叫端相容（如 `c2_2_graph_sampler.py`），本函式內部
            一律用 `lbmnca.paths.dataset_dir` 解析路徑，不使用此參數。
    """
    case_id   = node["case_id"]
    mask_id   = node["mask_id"]
    re        = float(node["re"])
    buildings = node["buildings"]
    ic_type   = node["ic_type"]
    depth     = int(node.get("tree_depth", node.get("depth", 0)))

    l_char = compute_l_char(buildings)
    bcr    = compute_bcr(buildings)
    nu     = U_INLET * l_char / re
    # 緩啟動步數：WARMUP_CTUS 個 CTU，1 CTU = L_char / u_inlet 步（逐 case 不同）。
    warmup_steps = int(round(WARMUP_CTUS * l_char / U_INLET))

    # ★2026-07-29（P2-B/P2-C 建議）：由 warn-only 改為**硬阻擋**。原註解說
    #   「守門政策待 regen 後決策」——現在正是 regen，所以決策就是擋。
    #   理由：warn-only 會讓不可行 config 靜默出貨，跑到一半才因 NaN 崩掉，
    #   浪費整個 case 的 wall time；而 solver 端的 tau 守門只擋 tau<=0.5 這個
    #   極端，Ma 過高與 tau 貼邊（實測 D1-1 有 case tau=0.5088、餘裕 0.0008）
    #   都要靠這裡擋。禁 fallback：不可行就 raise，不代選替代參數。
    _ok, _reason = check_feasibility(U_INLET, nu, l_char)
    if not _ok:
        raise ValueError(
            f"case {case_id} 物理不可行：{_reason}"
            f"（Re={re:.0f}, L_char={l_char}, nu={nu:.8g}, u_inlet={U_INLET}）。"
            "請調整該 case 的 Re 或幾何；不得生成不可行 config。"
        )

    mask_path = str(masks_dir / f"{mask_id}.png")

    # ★逐 case 的時間預算（2026-07-29 L3）：burn-in 依 Re，總長隨之。
    _burnin_factor = (WARM_BURNIN_RE_FACTOR if ic_type == "warm"
                      else BURNIN_RE_FACTOR)
    burnin_ctus = round(_burnin_factor * float(re), 1)
    total_ctus_case = round(WARMUP_CTUS + burnin_ctus + MEAN_WINDOW_CTUS, 1)

    tes: dict[str, Any] = {
        "case_id":             case_id,
        "geometry_id":         mask_id,
        "dataset_name":        dataset_name,
        "parent_dataset":      node.get("parent_dataset", dataset_name),
        "ic_type":             ic_type,
        "total_ctus":          total_ctus_case,
        # ★時均窗的三段結構明寫進 config，供轉檔層直接讀（不再靠固定比例猜）
        "warmup_ctus":         WARMUP_CTUS,
        "burnin_ctus":         burnin_ctus,
        "burnin_re_factor":    _burnin_factor,
        "mean_window_ctus":    MEAN_WINDOW_CTUS,
        "spinup_ctus":         SPINUP_CTUS,
        "re_target":           int(re),
        "n_obstacles":         int(node.get("n_buildings", len(buildings))),
        "bcr":                 round(bcr, 6),
        "g_vec":               [],
        "enable_conv_check":   ENABLE_CONV_CHECK,
        "conv_window_ctu":     CONV_WINDOW_CTU,
        "conv_eps":            CONV_EPS,
        "conv_k_required":     CONV_K_REQUIRED,
        "conv_cum_eps":        CONV_CUM_EPS,
        "conv_trend_gate":     CONV_TREND_GATE,
        "extra_ctu_after_conv": EXTRA_CTU_CONV,
        "warm_k_level":        depth,
        "perturb_type":        "",
        "perturb_magnitude":   0.0,
        # ★S1：求解迴圈內分段時均累加（Stage 1.15/1.16）。缺此區塊即關閉。
        "mean_accum": {
            "enabled":       MEAN_ACCUM_ENABLED,
            # 段長固定 MEAN_ACCUM_SEGMENT_CTU 個 CTU（1 CTU = L_char/u_inlet 步）。
            "segment_steps": int(round(MEAN_ACCUM_SEGMENT_CTU * l_char / U_INLET)),
            "stride":        MEAN_ACCUM_STRIDE,
        },
    }

    if ic_type == "warm":
        parent_h5 = resolve_parent_h5(node, dataset_name)
        parent_id = node.get("parent_case_id", "")
        tes["parent_h5_path"]           = parent_h5
        tes["parent_case_id"]           = parent_id
        # ★2026-08-02：warm 改用與 cold 相同的 staircase 排程。
        #   `dense_0.2`（每 0.2 CTU 一幀，約 1000 幀）是 turb 世代的產物——
        #   transit／脈動研究需要密集軌跡。turb 已於 2026-07-30 全面移出範圍，
        #   而 mean module 只用 `mean_flow`，它由 `/mean_accum` 逐步累加算出，
        #   **與 snapshot 排程完全無關**。
        #
        #   代價是實測出來的：dense 讓 warm 的 H5 達 1.89 GB/case，是 cold 的
        #   4.7 倍；260 個 warm case 需要 492 GB，而當時可用空間只有 145 GB。
        #   換 staircase 後回到約 0.4 GB/case。
        #
        #   若日後恢復 turb 研究，把這裡改回 dense_0.2 並重新評估磁碟即可；
        #   warm IC 與 mean_accum 的行為不受本項影響。
        tes["snapshot_schedule"]        = "staircase_ctu"
        # ★2026-07-29：不再覆寫 total_ctus——warm 的預算已由
        #   WARM_BURNIN_RE_FACTOR 在上方逐 case 算出（WARM_TOTAL_CTUS 作廢）。
        tes["extra_ctu_after_conv"]     = WARM_EXTRA_CTU_CONV
        tes["post_conv_spacing_ctu"]    = WARM_POST_CONV_SPACING_CTU
    else:
        tes["snapshot_schedule"] = "staircase_ctu"

    return {
        "simulation": {
            "nx":                    NX_SOLVER,
            "ny":                    NY_SOLVER,
            "name":                  case_id,
            "u_inlet":               U_INLET,   # 明寫入口速度（audit B2：solver 改 bracket 存取、禁 fallback）
            "nu":                    round(nu, 8),
            "ghost_moments_s":       GHOST_MOMENTS_S,
            "characteristic_length": float(l_char),
            "warmup_steps":          warmup_steps,
            "rho_in":                round(RHO_IN, 6),
            "rho_out":               RHO_OUT,
            "smagorinsky_constant":  SMAGORINSKY,
            "compute_step_size":     COMPUTE_STEP_SIZE,
            "max_steps":             0,
        },
        "boundary_condition": {
            "type":  [0, 2, 1, 2],
            "value": [[U_INLET, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        },
        "mask": {
            "enable":           True,
            "type":             "png",
            "invert":           False,
            "path":             mask_path,
            "embed_offset_col": EMBED_COL,
            "embed_offset_row": EMBED_ROW,
        },
        "domain_zones": {
            "sponge_in":                SPONGE_IN,
            "sponge_out":               SPONGE_OUT,
            "sponge_top":               SPONGE_TOP,
            "sponge_bot":               SPONGE_BOT,
            "buffer_in":                BUFFER_IN,
            "buffer_out":               BUFFER_OUT,
            "buffer_top":               BUFFER_TOP,
            "buffer_bot":               BUFFER_BOT,
            "buffer":                   0,
            "sponge_strength":          SPONGE_STRENGTH,
            "sponge_relax_strength":    0.0,
            "sponge_reference_u_factor": 1.0,
            "sponge_reference_v_target": 0.0,
            "sponge_reference_mode":    "uniform_flow",
        },
        "tes": tes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True)
    cfg = ap.parse_args()

    ds_dir     = dataset_dir(cfg.dataset_name)
    cases_json = ds_dir / "cases.json"
    masks_dir  = ds_dir / "masks"
    configs_dir = ds_dir / "configs"
    raw_dir    = ds_dir / "raw"
    meta_out   = ds_dir / "dataset_meta.json"

    configs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cases_data = json.loads(cases_json.read_text(encoding="utf-8"))
    nodes      = cases_data["nodes"]
    sim_order  = cases_data.get("sim_order", [n["case_id"] for n in nodes])

    cold_count = warm_count = skip_count = 0
    case_list  = []

    for node in nodes:
        if node.get("status") == "precomputed":
            skip_count += 1
            # Add to meta but skip config gen
            cold_dataset = node.get("cold_dataset", "")
            parent_h5 = str(dataset_dir(cold_dataset) / "raw" / f"{node['case_id']}.h5")
            case_list.append({
                "case_id":       node["case_id"],
                "mask_id":       node["mask_id"],
                "re":            int(node["re"]),
                "ic_type":       "cold",
                "status":        "precomputed",
                "h5_output_path": parent_h5,
                "cold_dataset":  cold_dataset,
            })
            continue

        cfg_dict  = build_yaml_config(node, cfg.dataset_name, masks_dir, data_root())
        out_path  = configs_dir / f"{node['case_id']}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        ic_type = node["ic_type"]
        if ic_type == "cold":
            cold_count += 1
        else:
            warm_count += 1

        entry: dict[str, Any] = {
            "case_id":       node["case_id"],
            "mask_id":       node["mask_id"],
            "re":            int(node["re"]),
            "ic_type":       ic_type,
            "depth":         int(node.get("tree_depth", node.get("depth", 0))),
            "config_path":   f"configs/{node['case_id']}.yaml",
            "h5_output_path": f"raw/{node['case_id']}.h5",
            "status":        "pending",
        }
        if node.get("sim_subtype") == "shortcut":
            entry["sim_subtype"]    = "shortcut"
            entry["shortcut_of"]    = node.get("shortcut_of", "")
        if ic_type == "warm":
            entry["parent_case_id"] = node.get("parent_case_id", "")
            entry["parent_dataset"] = node.get("parent_dataset", cfg.dataset_name)
            entry["warm_k_level"]   = int(node.get("tree_depth", node.get("depth", 1)))
        case_list.append(entry)

    meta = {
        "dataset_id":    cfg.dataset_name,
        "cold_dataset":  cases_data.get("cold_dataset", ""),
        "method":        cases_data.get("method", ""),
        "space":         cases_data.get("space", ""),
        "re_range":      cases_data.get("re_range", []),
        "n_roots":       cases_data.get("n_roots", 0),
        "n_cases":       cases_data.get("n_cases", 0),
        "n_shortcuts":   cases_data.get("n_shortcuts", 0),
        # 逐 case 不同，此處記錄公式而非單一值
        "total_ctus_formula": ("warmup + burnin + mean_window；"
                               f"burnin = {BURNIN_RE_FACTOR}×Re（cold）／"
                               f"{WARM_BURNIN_RE_FACTOR}×Re（warm）"),
        "warmup_ctus":   WARMUP_CTUS,
        "mean_window_ctus": MEAN_WINDOW_CTUS,
        "spinup_ctus":   SPINUP_CTUS,
        "schedule_cold": "staircase_ctu",
        "schedule_warm": "staircase_ctu",
        "snapshot_spacing_ctu_warm": 0.2,
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "sim_order":     sim_order,
        "cases":         case_list,
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[case_config] {cfg.dataset_name}")
    print(f"  cold={cold_count}  warm={warm_count}  precomputed_skip={skip_count}")
    print(f"  configs → {configs_dir}")
    print(f"  meta    → {meta_out}")


if __name__ == "__main__":
    main()
