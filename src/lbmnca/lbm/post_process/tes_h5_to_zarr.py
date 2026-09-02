"""TES H5 → TES Zarr 轉換 pipeline (v2).

兩遍掃描：
  Pass 1a: 掃描所有 H5，計算 static(upstream) z-score 與 g_vec / p_vec 統計
  Pass 1b: **驗收**資料是否裝得下凍結尺度（不再產生任何常數）
  Pass 2:  正規化所有欄位，寫入 Zarr；輸出 {dataset}_stats.json

★ physics 9 通道的 M_REF / S 一律取自 `norm_frozen`（凍結常數），
  不由任何資料集量測產生。static/cond 仍是量測值，故 `--stats-from` 仍必要。

輸出 Zarr 結構（供 src/lbmnca/nca/tes/data/tes_dataset.TESZarrDataset 讀取）：

    {dataset_name}.zarr/
    ├── .zattrs  {version, H, W, n_cases, created_at}
    └── cases/
        └── {case_key}/           e.g. "cfg_0001-M0001-RE150"
            ├── .zattrs     {ic_type, mask_id, re, n_snapshots, ctu_values: [...]}
            ├── snapshots          [T, 9, H, W]  float16  (Mphy_lattice_norm, rescaled baseline-deviation)
            ├── static             [4, H, W]     float32  (SDF_norm, upstream_norm, X_norm, Y_norm, z-score)
            ├── g_vec              [14]          float32  (raw)
            ├── p_vec              [3]           float32  (raw: Re, U_inlet_lu, L_char_px)
            ├── delta_ctus         [T-1]         float32  (np.diff(ctu_values)，DataLoader 直接切片取用)
            ├── mean_flow_post_conv [9, H, W]   float32  (時均流場，收斂後所有 frame 的 tanh-norm 平均)
            └── mean_flow_post_conv_running [9, H, W] float32
                                                 (★ Stage 1.15：求解迴圈內逐步累加的時均，
                                                  僅當 H5 含 /mean_accum 時寫出)

輸出路徑：{paths.zarr_dir()}/{dataset_name}.zarr
          {paths.zarr_dir()}/{dataset_name}_stats.json

用法（.venv 已啟動）：
    python -m lbmnca.lbm.post_process.tes_h5_to_zarr \\
        --dataset-dir dataset/TES_dataset/S1_1-Cold_M30C90
    # 可選：--output-root /path/to/output（預設：paths.zarr_dir()，見 LBMNCA_DATA_ROOT）
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

# ── lbmnca 內部模組 ───────────────────────────────────────────────────────────
from lbmnca.lbm.lbm_mrt_les.public.static_features import build_static_feature_stack
from lbmnca.lbm.post_process.mean_accum_io import (
    delta_norm_fluid,
    has_mean_accum,
    read_mean_accum,
    window_mean,
)
from lbmnca.lbm.post_process.mean_window import (
    MIN_SAMPLES_FOR_SEM,
    compute_sem_eff,
)
from lbmnca.lbm.post_process import norm_frozen as nf
from lbmnca.lbm.post_process.norm_utils import (
    CH_JX,
    CH_JY,
    NUM_CHANNELS,
    RESCALE_PERCENTILE,
    RHO_0,
)
from lbmnca.paths import zarr_dir

# ── ★ physics 正規化改用凍結常數（`norm_frozen`）────────────────────────────
# `M_REF` 與 `S` 不再由資料量測產生，掃描的角色從「產生常數」變成「驗收資料
# 是否裝得下」。跨 dataset 的尺度不一致因此在**構造上**不可能發生。
#
# ⚠ `static_norm.upstream` 與 `cond_stats` **仍是量測值**（幾何／條件的語義與
#   場通道不同，未納入本次凍結範圍）→ `--stats-from` 機制對這兩項仍然必要。
M_REF: np.ndarray = nf.M_REF.astype(np.float32)
S_FROZEN: np.ndarray = nf.S.astype(np.float32)

# ── G_vec：直接引用 canonical 定義，本模組不留副本 ────────────────────────────
# 本模組位於 dataset 生成主線上。若這裡持有一份手抄的索引集而 canonical 變動了，
# 產出的會是「形狀同為 [14]、語義錯位」的 g_vec；zarr/graph.json 只記 pipeline
# schema 版本（tes_v4）、沒有 g_vec 選取版本欄位，事後無從偵測。直接 import
# canonical 即從結構上排除這種分岔。
from lbmnca.morphospace.physics_proxy.features import compute_all as _compute_all_features
from lbmnca.morphospace.physics_proxy.gvec_api import GVEC_ACTIVE_INDICES as _GVEC_ACTIVE_INDICES

# ── 常數 ──────────────────────────────────────────────────────────────────────
COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
RESCALE_SAMPLES_PER_SNAPSHOT: int = 20_000  # 每快照隨機取樣像素數
H5_H: int = 360   # TES H5s 高度（3× 降採樣後）
H5_W: int = 640   # TES H5s 寬度
ROI_H: int = 1080  # ROI = Mask_Size 高度（降採樣前）
ROI_W: int = 1920  # ROI = Mask_Size 寬度
H5_DOWNSAMPLE: int = ROI_H // H5_H   # = 3，用於 SDF per-case tanh 縮放
# C2-2 short-edge attractor QA（spec §6.3，2026-07-07 校準）：
# 15 CTU 窗時均在 tanh-norm 空間帶 ~3 渦脫週期的相位殘留，連 canonical 自己的
# 15-vs-50 CTU 窗時均 rel-L2 都達 0.04-0.31 → 固定門檻不可行。改自校準比值：
#   ratio = rel_l2(short_tail_mean, canon_50) / rel_l2(canon_同長窗_mean, canon_50)
# 同 attractor 時 ratio ~ O(1)（兩個獨立相位殘留）；不同 attractor → ratio >> 1。
ATTRACTOR_QA_RATIO_MAX: float = 3.0

# ── 時均窗函數（DEEP_AUDIT_2026-07 修正 2）─────────────────────────────────
# 方窗（等權平均）對週期流的時均誤差收斂階僅 O(k^-1)（k = 平均了幾個脫落週期），
# 且殘留相位偏差取決於窗長是否為週期的整數倍。因每個 case 的 St 不同，
# 該偏差**逐 case 不同** → 形成資料集層級的系統性雜訊。
#   Hann   w(τ)=2sin²(πτ)      → O(k^-3)
#   Hann²  w(τ)=(8/3)sin⁴(πτ)  → O(k^-5)，且不需事先知道週期長度
# 依據：Krakos, Wang, Hall & Darmofal (2012), JCP 231(8):3228-3245；arXiv:2412.00604
#
# TIER0 實測（38 case）：‖方窗 − Hann²‖/‖Hann²‖ 中位 = 9.85%、最大 18.7%。
#
# ⚠ 預設維持 "square"（= 既有行為），因為切換會改變 mean_flow 的定義：
#   若只重轉部分 dataset，會與既有 zarr 產生**靜默不一致**。
#   切換時機由 FIX_A Tier 1 對照結果決定，屆時需全量重轉。
#   實際使用的窗型一律寫入 zarr attrs（mean_window），使不一致可被偵測。
MEAN_WINDOW: str = os.environ.get("TES_MEAN_WINDOW", "square")
_VALID_MEAN_WINDOWS = ("square", "hann", "hann_sq")
if MEAN_WINDOW not in _VALID_MEAN_WINDOWS:
    raise ValueError(
        f"未知 TES_MEAN_WINDOW={MEAN_WINDOW!r}（可用 {_VALID_MEAN_WINDOWS}）"
    )

# ── 時均窗起點（DEEP_AUDIT_2026-07 Stage 1.16：把收斂偵測降級為粗略提示）────
#
# 【已撤除】原有的 ``MEAN_WINDOW_POLICY``（"post_conv" / "last_n_periods"）與
# ``N_SHEDDING_PERIODS_FOR_MEAN``。1.14 裁決路徑 2：全域 KE 無法定義脫落週期
# （1.13：真實脫落諧波弱 4744 倍、T_shed 高估 3.59 倍），"last_n_periods" 這條
# 分支從未進入生產且原理上不可驗收 → 整批移除。
#
# 【現行】時均窗起點由**兩種基準**擇一，明寫入 zarr attr ``mean_window_basis``：
#
#   "running_accum_fixed_start"（H5 含 /mean_accum 時）
#       起點 = ``MEAN_ACCUM_START_FRAC × 軌跡總 CTU``，**協定常數，逐 case 相同，
#       完全不看 convergence_ctu**。理由（1.16）：收斂閂鎖的 two-window 判準
#       用兩個 5 CTU 半窗（合計 10 CTU）監測一個主導模態週期約 11.1 CTU 的訊號
#       （1.11 步驟 4），窗比週期短 → 原理上測不到穩態，只測到相位差；
#       生產母體 31.1% 的 post-conv 池 |漂移| > 2×SEM_eff 即其後果。
#       1.15 §4.6 實測起點 0→40 CTU 只讓時均變 0.020–0.089，遠小於窗長效應
#       → **起點該由一個固定協定決定，不該由一個不可靠的偵測器決定。**
#       殘餘的起點敏感度逐 case 量測並寫入 ``mean_running_start_delta_1seg``，
#       不靠假設它可忽略。
#
#   "post_conv_latch"（舊 H5，無 /mean_accum）
#       起點 = ``convergence_ctu``（既有行為，一位元不改）。既有 canonical
#       dataset 全部走這條，重轉結果與現存 zarr 逐位元一致。
MEAN_ACCUM_START_FRAC: float = 0.2
"""含 ``/mean_accum`` 的 H5，時均窗起點占軌跡總 CTU 的比例（burn-in 比例）。

0.2 在生產設定（``total_ctus=100``）下 = 20.0 CTU，即 Stage 1.15 的驗收窗起點；
用比例而非絕對值，是為了讓不同 ``total_ctus`` 的 case 共用同一條協定。

⚠ **這是協定常數，不是量測量**，不宣稱「20 CTU 之後已達穩態」。它可被辯護的
理由只有一條：1.15 §4.6 實測起點在 0–40 CTU 之間只讓時均變 0.020–0.089，
故任何固定選擇的代價**有界**，且逐 case 記錄於 ``mean_running_start_delta_1seg``。
"""

# ── 時均來源（DEEP_AUDIT_2026-07 Stage 1.15）─────────────────────────────────
# "snapshot"：mean_flow_post_conv = 對 post-conv **snapshot** 取平均（既有行為，
#             預設）。每 case 僅 26-29 幀、間距 ~1-2 CTU，與脫落週期同量級
#             → 抽樣雜訊 + 混疊（實測 SE 達模型誤差的 29-73%）。
# "running" ：mean_flow_post_conv = 由 H5 ``/mean_accum`` 分段和重組的
#             **求解迴圈內逐步累加**時均（每個 LBM 步都入和）。
#
# ★2026-07-29 S1：預設由 "snapshot" 改為 "running"。實測依據（P2-C 複核）：
#   1. snapshot 路徑被**聲學混疊**污染：cold staircase 後段 Δ=2.0 CTU，
#      L_char=220 的聲學週期 T_ac=4.0614 CTU → Δ/T_ac = 0.4924（幾乎正好半週期、
#      逐幀反相）。實測 41 幀窗的偶/奇幀均值 rel_l2 = 1.6467（tanh 值域上限 1.0，
#      **完全飽和**），只因幀數 41 為奇數才靠不完全抵消殘留約 1/41 振幅。
#      母體殘留因子 R：cold median 0.0137、max 0.1113。**這是系統性偏差**，
#      逐步累加器可完全消除。
#   2. 兩版 GT 差 0.105 rel-L2 = 模型誤差 0.229 的 **46%**；
#      mean_snapshot_delta_self = 0.169 vs mean_running_delta_self = 0.096
#      （累加器的自我一致性好 1.8 倍）。
#   3. 舊預設走 os.environ + 預設值，違反本專案「禁 fallback config 存取」鐵則，
#      且讓 Stage 1.15/1.16 的解耦成果沒有進入 canonical 交付。
#   無論設哪個，只要 H5 帶 /mean_accum 就**額外**寫一份
#   `mean_flow_post_conv_running`；實際來源寫入 case attr `mean_flow_source`。
#   ★既有 zarr 不受影響（本常數只在**新轉檔**時生效）。
MEAN_SOURCE: str = os.environ.get("TES_MEAN_SOURCE", "running")
_VALID_MEAN_SOURCES = ("snapshot", "running")
if MEAN_SOURCE not in _VALID_MEAN_SOURCES:
    raise ValueError(
        f"未知 TES_MEAN_SOURCE={MEAN_SOURCE!r}（可用 {_VALID_MEAN_SOURCES}）"
    )

GT_QUALITY_SENTINEL: float = -1.0
"""GT 品質欄位（mean_sem_eff / mean_*_delta_self 等）在不可算時（樣本不足、
上游未提供）的 sentinel 值，比照既有 convergence_ctu=-1.0 慣例。

⚠ 一律寫 sentinel，**不靜默填 0 或猜測值** —— 避免下游把「未量測」誤讀成
「量測值為 0」。"""


def _resolve_mean_start_ctu(
    convergence_ctu: float,
    total_ctu: float,
    has_accum: bool,
    burnin_ctu: float | None = None,
    warmup_ctu: float | None = None,
) -> tuple[float, str]:
    """決定時均窗起點，並回報它是依什麼基準訂的。

    ★ DEEP_AUDIT_2026-07 Stage 1.16：把收斂偵測從「正確性的關鍵路徑」降級為
    「何時開始累加的粗略提示」。有分段累加器時，窗起點改由協定常數決定，
    **完全不看 ``convergence_ctu``**；此時閂鎖是否觸發不再影響任何 GT 數值，
    只留在 ``convergence_ctu`` / ``converged`` attr 裡當狀態紀錄。

    為什麼不繼續用閂鎖（1.16 的三條實測依據）：

      1. ``conv_window_ctu=5.0`` 的 two-window 判準總跨度只有 10 CTU，短於它所
         監測訊號的主導模態週期（~11.1 CTU，1.11 步驟 4）→ **原理上測不出穩態**，
         只測到相位差。
      2. 生產母體 74.6% 的 ``convergence_ctu`` 卡在 floor（``2×window+5=15``），
         即判準一被允許檢查就立刻通過 —— 那不是物理時刻，是 floor 本身。
      3. 1.15 §4.6 實測：窗起點 0→40 CTU 只讓時均變 0.020–0.089，
         而窗長 < 30 CTU 時 ``Δ_self`` 爆到 0.26–1.56 →
         **起點遠不如窗長重要**，不值得綁一個不可靠的偵測器。

    Args:
        convergence_ctu: H5 的收斂閂鎖時刻（<0 表示從未觸發）。
        total_ctu: 軌跡最後一個 snapshot 的 CTU。
        has_accum: H5 是否含 ``/mean_accum`` 分段累加器。
        burnin_ctu: config 明寫的丟棄窗長（CTU）；``None``/負值 = 舊 H5 無此欄位。
        warmup_ctu: config 明寫的 ramp 長度（CTU），與 burn-in 相加得窗起點。

    Returns:
        ``(start_ctu, basis)``：窗起點（CTU）與基準字串
        （``"running_accum_fixed_start"`` 或 ``"post_conv_latch"``）。

    Raises:
        ValueError: 舊 H5（無累加器）且閂鎖從未觸發 —— 此時時均窗無定義。
            這種 case 應已在 :func:`_filter_readable_h5` 被擋掉。
    """
    # ★優先序（2026-07-29 L3）：config 明寫的 burn-in > 固定比例 > 收斂閂鎖。
    #   burn-in 由 `gen_c_series_configs.BURNIN_RE_FACTOR × Re` 算出，物理依據是
    #   動量擴散橫跨 L_char 的時間 = Re 個 CTU（四 case 實測吻合）。
    #   固定比例 0.2 是它之前的替代品，對高 Re 系統性不足（Re=500 只丟 40 CTU）。
    if has_accum and burnin_ctu is not None and burnin_ctu >= 0.0:
        _ramp = warmup_ctu if (warmup_ctu is not None and warmup_ctu > 0) else 0.0
        return _ramp + burnin_ctu, "config_burnin_by_Re"
    if has_accum:
        return MEAN_ACCUM_START_FRAC * total_ctu, "running_accum_fixed_start"
    if convergence_ctu >= 0.0:
        return convergence_ctu, "post_conv_latch"
    raise ValueError(
        "舊 H5（無 /mean_accum）且 convergence_ctu < 0 → 時均窗無定義。"
        "此 case 應已被 _filter_readable_h5 過濾，請檢查過濾器。"
    )


def _time_mean_weights(n: int, window: str) -> np.ndarray:
    """時均權重，歸一化使 sum(w) = 1。

    Args:
        n: 幀數。
        window: "square" / "hann" / "hann_sq"。

    Returns:
        [n] float64 權重。
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n < 3 or window == "square":
        return np.full(n, 1.0 / n, dtype=np.float64)
    tau = (np.arange(n, dtype=np.float64) + 0.5) / n
    if window == "hann":
        w = 2.0 * np.sin(np.pi * tau) ** 2
    else:  # hann_sq
        w = (8.0 / 3.0) * np.sin(np.pi * tau) ** 4
    s = w.sum()
    if s <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return w / s


def _weighted_time_mean(arr: np.ndarray, window: str) -> np.ndarray:
    """對 axis=0（時間軸）做加權時均。

    Args:
        arr: [N, ...] 物理空間 moments。
        window: 窗型，見 :func:`_time_mean_weights`。

    Returns:
        [...] 加權時均。
    """
    n = arr.shape[0]
    if n == 0:
        raise ValueError("empty time axis")
    if window == "square":
        return arr.mean(axis=0)
    w = _time_mean_weights(n, window).astype(arr.dtype, copy=False)
    return np.tensordot(w, arr, axes=(0, 0))


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════


def _list_h5_files(input_dir: Path) -> list[Path]:
    """列出目錄內所有 .h5 檔案，按檔名排序。"""
    files = sorted(input_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 中找不到任何 .h5 檔案")
    return files


#: 代表「已通過收錄」的 status。`precomputed` 是 warm dataset 引用的 cold root，
#: 它的驗收發生在其原生 dataset，不在這裡重判。
_ADMITTED_STATUS = ("success", "converged", "precomputed")


def _filter_admitted(h5_files: list[Path], dataset_dir: Path,
                     allow_no_meta: bool = False) -> list[Path]:
    """只保留 `dataset_meta.json` 標記為已收錄的 case。

    ★這道過濾不可省略。H5 的 ``converged`` attr 是 **KE 收斂閂鎖**的結果，
    與 `validation.case_physics` 的物理驗收閘門是兩回事——實測 D 系列 24 個
    未通過閘門的 case（含 3 個 KE 發散）其 ``converged`` **全部是 True**。
    下游若沿用 ``converged`` 當收錄判據，等於用一個比閘門寬鬆得多的標準，
    被判退的資料會全數流進訓練集，而且不會有任何地方再發現。

    收錄與否的單一真相是 `dataset_meta.json` 的 ``status``。

    Args:
        h5_files: 待過濾的 H5 路徑清單。
        dataset_dir: dataset 根目錄（含 `dataset_meta.json`）。
        allow_no_meta: dataset 無 meta 時是否放行。僅供 bench／臨時 dataset，
            生產資料不得使用——沒有 meta 就代表收錄狀態未定義。

    Returns:
        通過過濾的 H5 路徑清單。
    """
    meta_p = dataset_dir / "dataset_meta.json"
    if not meta_p.exists():
        if not allow_no_meta:
            raise FileNotFoundError(
                f"找不到 {meta_p}；收錄狀態未定義，不得靜默把全部 H5 當成已收錄。"
                "臨時／bench dataset 請明確加 --allow-no-meta。")
        print(f"  [warn] 無 dataset_meta.json，未套用收錄過濾"
              f"（--allow-no-meta）：{dataset_dir.name}")
        return h5_files

    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    admitted = {str(c.get("case_id")) for c in meta.get("cases", [])
                if str(c.get("status")) in _ADMITTED_STATUS}
    ok = [f for f in h5_files if f.stem in admitted]
    dropped = [f for f in h5_files if f.stem not in admitted]
    print(f"  收錄過濾：{len(ok)} 個通過物理閘門，排除 {len(dropped)} 個")
    for f in dropped[:10]:
        entry = next((c for c in meta.get("cases", [])
                      if str(c.get("case_id")) == f.stem), {})
        print(f"    排除 {f.stem}  status={entry.get('status', '不在 meta 中')!r}"
              f"  {str(entry.get('error', ''))[:50]}")
    if len(dropped) > 10:
        print(f"    ...另有 {len(dropped) - 10} 個")
    if not ok:
        raise RuntimeError(
            f"{dataset_dir.name}：收錄過濾後無任何 case。"
            "請確認物理驗收是否已跑過（dataset_meta.json 的 status）。")
    return ok


def _filter_readable_h5(h5_files: list[Path]) -> list[Path]:
    """過濾只保留可讀取、且時均窗有明確定義的 H5 檔案。

    兩條准入路徑（★ Stage 1.16 改）：

      A. H5 含 ``/mean_accum`` → **無條件准入**，與 ``convergence_ctu`` 無關。
         此時時均窗起點來自協定常數（見 :data:`MEAN_ACCUM_START_FRAC`），
         不經過收斂閂鎖，故「閂鎖沒觸發」不再是排除理由。
         1.16 的根因：閂鎖的 two-window 判準（兩個 5 CTU 半窗，合計 10 CTU）
         短於它所監測訊號的主導模態週期（~11.1 CTU）→ 觸不觸發帶有隨機性，
         用它當准入條件等於用擲骰子挑資料。1.15 的驗收 case 正是被這條擋掉，
         當時只能手動改 ``convergence_ctu`` 才跑得通管線。

      B. H5 不含 ``/mean_accum``（舊資料）→ 維持既有規則：``converged=True``
         且 ``convergence_ctu >= 0``。這條路徑的時均窗**只能**由閂鎖定義，
         閂鎖沒觸發就真的沒有窗，其 mean 會用全幀含 transient → 必須排除。
         既有 canonical dataset 全部走這條，行為一位元不變。

    stability_failure 案例的 H5 可能有 dirty consistency flags → 跳過。

    Args:
        h5_files: 待過濾的 H5 路徑清單。

    Returns:
        通過過濾的 H5 路徑清單。
    """
    ok: list[Path] = []
    skipped: list[str] = []
    n_accum = 0
    for p in h5_files:
        try:
            with h5py.File(p, "r") as f:
                converged = bool(f.attrs.get("converged", True))
                stop      = str(f.attrs.get("stop_reason", ""))
                cctu      = float(f.attrs.get("convergence_ctu", -1.0))
                has_traj  = "trajectory/moments" in f and "trajectory/ctu" in f
                has_accum = has_mean_accum(f)
        except OSError:
            skipped.append(f"{p.name} (open error — consistency flags dirty, skipping)")
            continue
        if not has_traj:
            skipped.append(f"{p.name} (missing trajectory/moments — partial write)")
            continue
        if has_accum:
            n_accum += 1
            ok.append(p)
            continue
        # 舊路徑（無 /mean_accum）：時均窗只能由閂鎖定義 → 沿用 DD-03 規則。
        if not converged or cctu < 0.0:
            skipped.append(
                f"{p.name} (no /mean_accum and not KE-converged: "
                f"converged={converged} cctu={cctu:.1f} stop={stop})"
            )
            continue
        ok.append(p)
    if skipped:
        print(f"  [filter] 跳過 {len(skipped)} 個 H5 (無有效時均窗 / 不可讀):")
        for s in skipped:
            print(f"    {s}")
    print(f"  [filter] 保留 {len(ok)} / {len(h5_files)} 個 H5"
          f"（其中 {n_accum} 個含 /mean_accum，時均窗與收斂閂鎖無關）")
    return ok


def _split_by_window(h5_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """依 window_type attr 分流 (full_files, short_files)。

    C2-2 WarmGraph relation edge（window_type="short"）只錄 transient +
    短驗證窗，不得進入全域 normalization 統計、不自算 mean_flow。
    舊 dataset 無 window_type attr → 視為 full。
    """
    full_files: list[Path] = []
    short_files: list[Path] = []
    for p in h5_files:
        with h5py.File(p, "r") as f:
            wt = str(f.attrs.get("window_type", ""))  # optional — 舊 H5 無此 attr
        (short_files if wt == "short" else full_files).append(p)
    if short_files:
        print(f"  [window] full={len(full_files)}  short={len(short_files)} "
              f"(short 排除於全域統計、mean_flow 引用 canonical)")
    return full_files, short_files


def _stats_population(full_files: list[Path], role: str,
                      has_stats_from: bool) -> list[Path]:
    """依 dataset 角色決定正規化統計的取樣母體。

    ★ DEEP_AUDIT Stage 3.3：upstream z-score / cond z-score 若在 train+val+test
    全集上計算並烘焙進 zarr 陣列值，val/test 的尺度資訊會回流訓練。
    故只有 train dataset 得進入母體。

    ★ physics 尺度已凍結（`norm_frozen`），不由任何資料產生 ——
    那條洩漏路徑因此在構造上消失，本函式只剩 static/cond 要守。

    角色是 **dataset 級**屬性（一個 dataset 一個角色，幾何互斥由生成端保證），
    不是 case 級屬性——故由 `--dataset-role` 直接宣告，不需要逐 case manifest。

    Args:
        full_files: 候選 H5（已過 `_split_by_window` 的 full-window 集）。
        role: `train` / `val` / `test`。
        has_stats_from: 是否指定了 `--stats-from`（沿用既有常數）。

    Returns:
        母體 H5 路徑列表；非 train 角色回傳空列表（常數一律沿用）。

    Raises:
        RuntimeError: train 角色卻無 full-window case；或非 train 角色未指定
            `--stats-from`（那會使 val/test 自行算尺度，分數不可比）。
    """
    if role != "train":
        if not has_stats_from:
            raise RuntimeError(
                f"--dataset-role={role} 必須同時指定 --stats-from 指向訓練集的 "
                "_stats.json。驗證／測試集若各自算自己的尺度，模型評估時看到的"
                "數值範圍與訓練時不同，分數不可比——而且不會報錯。")
        print(f"  [role] {role}：不進入統計母體，正規化常數一律沿用 --stats-from")
        return []
    if not full_files:
        raise RuntimeError("train dataset 無 full-window case，無法計算全域統計。")
    print(f"  [role] train：全域統計母體 {len(full_files)} 個 full-window case")
    return list(full_files)


def _make_case_key(attrs: dict[str, Any]) -> str:
    """從 H5 attrs 產生 case key（供 Zarr cases/ 使用）。

    新格式（case_id starts with 'cfg_'）：直接使用 case_id
        e.g. cfg_0001-M0001-RE150

    舊格式（legacy fallback）：
        Cold: {geometry_id}_re{re}_{ic_type}  → geo001_re150_cold
        Warm: {geometry_id}_from{parent_geo}_re{re}_{ic_type}
    """
    # H5 attrs 為系統邊界（solver 寫出）；以下 .get() 均為 optional 欄位
    case_id = str(attrs.get("case_id", ""))  # optional — 舊 H5 可能無此 attr
    # DD-01 fix：只要有 case_id 就直接用（每 case 唯一，含 _sc0/_sc1 shortcut、
    # _r1 replacement）。舊 legacy 用 geometry_id 組 key，shortcut（geometry_id
    # 不含 _sc 後綴、同 Re）與 base 組出相同 key → require_group overwrite 覆寫
    # → warm node 大面積流失 + pair parent 歸屬錯置。
    if case_id:
        return case_id

    # Legacy fallback（僅 case_id 缺失的舊 H5）
    geo_id = str(attrs.get("geometry_id", "unknown"))  # optional — H5 attrs system boundary
    re_val = int(round(float(attrs.get("re", 0))))     # optional — legacy H5 fallback
    ic_type = str(attrs.get("ic_type", "cold"))        # optional — 預設 cold

    if ic_type == "warm":
        parent_case_id = str(attrs.get("parent_case_id", ""))  # optional — warm 才有
        if parent_case_id:
            parts = parent_case_id.split("_")
            parent_geo = next((p for p in parts if p.startswith("geo")), None)
            if parent_geo:
                return f"{geo_id}_from{parent_geo}_re{re_val}_{ic_type}"

    return f"{geo_id}_re{re_val}_{ic_type}"


def _read_h5_attrs(f: h5py.File) -> dict[str, Any]:
    """讀取 H5 根屬性為 Python dict。"""
    return {k: v for k, v in f.attrs.items()}


def _extract_pvec(attrs: dict[str, Any]) -> np.ndarray:
    """從 attrs 提取 P_vec = [Re, U_inlet_lu, L_char_px]。"""
    return np.array(
        [
            float(attrs["re"]),
            float(attrs["u_inlet_lu"]),
            float(attrs["l_char_px"]),
        ],
        dtype=np.float32,
    )


def _find_masks_dir(h5_path: Path) -> Path:
    """從 H5 路徑推算 masks/ 目錄（dataset_dir/masks/）。

    H5 可能在 dataset_dir/raw/*.h5 或 dataset_dir/*.h5。
    """
    parent = h5_path.parent
    dataset_dir = parent.parent if parent.name == "raw" else parent
    return dataset_dir / "masks"


def _clean_solid_mask(h5_mask: np.ndarray, masks_dir: Path,
                      geometry_id: str) -> tuple[np.ndarray, int]:
    """乾淨固體遮罩：H5 遮罩 ∪「PNG 面積降採樣後固體占比 > 0」的格。

    ★ 為什麼需要（`tes_writer.py:263` vs `:286`）：moments 用 ``cv2.INTER_AREA``
    降採樣（3×3 區塊面積平均），mask 卻用 ``cv2.INTER_NEAREST``（取中心格）。
    一個 2/3 是固體的區塊，其 moments 平均後 :math:`\\rho = 1/3`，但只要中心格
    是流體就被標成流體。實測這類格佔流體的 0.098 %、**100 % 貼著固體**、
    :math:`\\rho` 精確量化在 :math:`k/3`（``INTER_AREA`` 的指紋），
    且**貢獻了 :math:`|\\delta\\rho|` 尾巴的全部**（0.116 → 0.698）。

    不修的話，正規化尺度會被這 0.098 % 的標籤錯誤綁架 ——
    密度尺度得放大 6 倍去容納一群根本不該算流體的格。

    Args:
        h5_mask: ``geometry/mask``（True = 固體，含 LBM 邊界牆）。
        masks_dir: ``dataset/{ds}/masks/``。
        geometry_id: mask PNG 檔名（不含副檔名）。

    Returns:
        ``(clean_solid, n_reclassified)``：乾淨固體遮罩（bool）與被重新分類的格數。

    Raises:
        FileNotFoundError: 找不到 mask PNG —— 無法判定部分格，拒絕沿用髒遮罩。
    """
    png = masks_dir / f"{geometry_id}.png"
    if not png.exists():
        raise FileNotFoundError(
            f"找不到 {png}；無法重建固體占比 φ。舊的 INTER_NEAREST 遮罩會把部分"
            "固體格標成流體並污染正規化統計，故拒絕沿用。")
    phi = nf.solid_fraction_from_png(str(png))
    base = np.asarray(h5_mask, dtype=bool)
    clean = base | (phi > 0.0)
    return clean, int((clean & ~base).sum())


def _load_mask_png_solid(masks_dir: Path, geometry_id: str) -> np.ndarray | None:
    """載入 mask PNG → uint8 solid mask（1=solid, 0=fluid）。

    PNG 慣例（lbmnca.morphospace）：255=free, 0=building(solid)。
    H5 mask 包含 LBM 邊界牆（全框 solid 連通）→ 不能直接用於 g_vec 計算。
    """
    png_path = masks_dir / f"{geometry_id}.png"
    if not png_path.exists():
        return None
    from PIL import Image
    arr = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
    return (arr < 128).astype(np.uint8)  # 1=solid(building), 0=fluid


def _compute_gvec_from_mask(solid_hw: np.ndarray | None) -> np.ndarray:
    """solid uint8 mask → 14D G_vec。乾淨資料原則：失敗即 raise，禁靜默零向量污染。"""
    if _compute_all_features is None:
        raise RuntimeError(
            f"morphospace features 未載入（{_MORPHOSPACE_FEATURES_PATH}）；"
            "拒絕寫入零向量 g_vec。請確認 lbmnca.morphospace 路徑存在。"
        )
    if solid_hw is None:
        raise FileNotFoundError("mask PNG 不存在，無法計算 g_vec（拒絕零向量回退）。")
    all_features = _compute_all_features(solid_hw)  # [22] float32；計算例外直接向上拋
    return all_features[_GVEC_ACTIVE_INDICES].astype(np.float32)


def _compute_static_raw(mask_hw: np.ndarray, l_char_px: float) -> np.ndarray:
    """從 H5 mask 計算靜態特徵（raw，未 z-score）。

    Args:
        mask_hw: (H, W) bool 遮罩（H5s 尺寸，360×640），True = 固體。
        l_char_px: 特徵長度（原始解析度像素值，降採樣前）。

    Returns:
        float32 陣列 (4, H, W) = [SDF_raw, upstream_raw, X_raw, Y_raw]。
    """
    return build_static_feature_stack(
        binary_mask_hw=mask_hw.astype(np.float32),
        l_char_px=float(l_char_px),
        crop_w_px=float(ROI_W),
        crop_h_px=float(ROI_H),
        target_h=H5_H,
        target_w=H5_W,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pass 1a — static / cond stats
# ══════════════════════════════════════════════════════════════════════════════



def pass1a_scan_stats(h5_files: list[Path]) -> dict[str, Any]:
    """Pass 1a：掃描所有 H5，計算 upstream z-score 與 cond 統計。

    ★ 也順帶回報 max velocity，但它**不再進入任何正規化** ——
    physics 尺度已凍結，此值只作為資料特性的紀錄。

    Static 各通道 norm 策略：
      C0 SDF      → per-case tanh(SDF / L_char_h5s)，不需全域 stats
      C1 upstream → 全域 z-score，此處收集 mean/std
      C2 X_01     → [0,1] fixed，不需全域 stats
      C3 Y_01     → [0,1] fixed，不需全域 stats

    Args:
        h5_files: H5 檔案路徑列表。

    Returns:
        dict with keys: max_vel, upstream_mean, upstream_std, g_vecs, p_vecs
    """
    max_vel: float = 0.0
    all_upstream: list[np.ndarray] = []

    all_g_vecs: list[np.ndarray] = []
    all_p_vecs: list[np.ndarray] = []

    for h5_path in tqdm(h5_files, desc="[Pass 1a] 掃描 H5 統計"):
        with h5py.File(h5_path, "r") as f:
            attrs = _read_h5_attrs(f)
            moments = f["trajectory/moments"][:].astype(np.float32)  # [T, 9, H, W]
            mask_hw = f["geometry/mask"][:]                            # [H, W] bool

        # 優先從原始 mask PNG 計算 g_vec（H5 mask 含 LBM 邊界牆，不可用）
        masks_dir = _find_masks_dir(h5_path)
        solid_hw = _load_mask_png_solid(masks_dir, str(attrs.get("geometry_id", "")))  # optional — H5 attrs
        g_vec = _compute_gvec_from_mask(solid_hw)

        # ── max velocity：僅紀錄用，不進入正規化 ──
        rho = np.maximum(moments[:, 0], 1e-6)  # [T, H, W]
        vel_x = moments[:, CH_JX] / rho
        vel_y = moments[:, CH_JY] / rho
        vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2)
        max_vel = max(max_vel, float(vel_mag.max()))

        # ── C1 upstream：收集全域統計（其他 static 通道不需全域 stats）──
        l_char_px = float(attrs["l_char_px"])
        static_raw = _compute_static_raw(mask_hw, l_char_px)  # (4, H, W)
        all_upstream.append(static_raw[1].ravel())  # C1 upstream [0,1]

        # ── cond 向量 ──
        all_g_vecs.append(g_vec)
        try:
            p_vec = _extract_pvec(attrs)
        except KeyError as e:
            raise KeyError(
                f"[tes_h5_to_zarr] 缺少必要 attr {e} in {h5_path.name}. "
                "確認 tes_writer 有寫入 re/u_inlet_lu/l_char_px。"
            ) from e
        all_p_vecs.append(p_vec)

    upstream_cat = np.concatenate(all_upstream)
    upstream_mean = float(upstream_cat.mean())
    upstream_std = float(upstream_cat.std()) + 1e-8
    print(f"  upstream mean={upstream_mean:.4f}  std={upstream_std:.4f}")

    return {
        "max_vel": max_vel,
        "upstream_mean": upstream_mean,
        "upstream_std":  upstream_std,
        "g_vecs": np.stack(all_g_vecs, axis=0),   # [N, 14]
        "p_vecs": np.stack(all_p_vecs, axis=0),    # [N, 3]
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pass 1b — rescale_abs_max
# ══════════════════════════════════════════════════════════════════════════════


class ComplianceAccumulator:
    """凍結尺度的**驗收**器（不產生任何常數）。

    ★ 掃描的角色已反轉。以前是「掃資料 → 算出尺度」，現在是
    「尺度是憲法 → 掃資料只驗證裝不裝得下」。

    ★ **驗收對象是 `mean_flow`**，不是瞬時快照。範疇為 mean-only：
    zarr 不保存湍流時序，模型的狀態空間裡只會有時均場（以及冷啟動的靜止態，
    其偏差恆為 0）。拿過渡段的極值來驗收，會用一批**永遠不會進入模型**的值
    去否決一組對時均場完全夠用的常數。

    ⚠ **紀律**：回報 FAIL 時，要修的是資料或報告限制，
    **不是** :data:`norm_frozen.S`。一旦允許「資料不合就調常數」，
    跨資料集可比性就名存實亡。
    """

    def __init__(self) -> None:
        self._mx = np.zeros(NUM_CHANNELS, dtype=np.float64)
        self._samp: list[list[np.ndarray]] = [[] for _ in range(NUM_CHANNELS)]
        self._n = 0
        self._rng = np.random.default_rng(42)

    def add(self, raw_mean: np.ndarray, fluid: np.ndarray,
            n_sample: int = 6000) -> None:
        """納入一個 case 的時均場（**物理空間**，未正規化）。

        Args:
            raw_mean: ``(9, H, W)`` raw moments 的時均。
            fluid: ``(H, W)`` bool，乾淨流體遮罩。
            n_sample: 分位統計的每場取樣像素數（max 一律取精確值）。
        """
        idx = np.flatnonzero(fluid.reshape(-1))
        if idx.size == 0:
            return
        pick = idx[self._rng.choice(idx.size, min(n_sample, idx.size), replace=False)]
        for ch in range(NUM_CHANNELS):
            v = np.abs(raw_mean[ch] - M_REF[ch]).reshape(-1)
            self._mx[ch] = max(self._mx[ch], float(v[idx].max()))
            self._samp[ch].append(v[pick])
        self._n += 1

    def verdict(self) -> dict[str, Any]:
        """產生符合度報告；有任何值超出可表達範圍即 raise。

        Raises:
            RuntimeError: 有值 ``tanh(|dev|/S) > CLIP`` —— 反正規化後不可逆。
        """
        per_ch: dict[str, dict[str, float]] = {}
        n_clip = 0
        for ch, name in enumerate(nf.CH_NAMES):
            v = np.concatenate(self._samp[ch])
            t_max = float(np.tanh(self._mx[ch] / S_FROZEN[ch]))
            n_clip += int(t_max > nf.CLIP)
            per_ch[name] = {
                "raw_p99.5": float(np.percentile(v, 99.5)),
                "raw_p99.99": float(np.percentile(v, 99.99)),
                "raw_max": float(self._mx[ch]),
                "util_p99.5": float(np.tanh(np.percentile(v, 99.5) / S_FROZEN[ch])),
                "tanh_max": t_max,
                "coverage_margin": float(nf.R[ch] / max(self._mx[ch], 1e-12)),
            }
        util = [per_ch[n]["util_p99.5"] for n in nf.CH_NAMES]
        orb = [max(per_ch[nf.CH_NAMES[i]]["util_p99.5"] for i in idx)
               for idx in nf.ORBITS.values()]
        rep = {
            "norm_schema": nf.SCHEMA_VERSION,
            "scope": "mean_flow (mean-only 範疇；不含過渡段)",
            "n_fields": self._n,
            "per_channel": per_ch,
            "util_ratio_per_channel": max(util) / max(min(util), 1e-12),
            "util_ratio_orbit": max(orb) / max(min(orb), 1e-12),
            "n_channels_clipped": n_clip,
            "verdict": "PASS" if n_clip == 0 else "FAIL",
        }
        print(f"\n  ── 凍結尺度符合度（{self._n} 個時均場）：{rep['verdict']} ──")
        print("    ch     raw_max   餘裕   util  tanh_max")
        for name in nf.CH_NAMES:
            d = per_ch[name]
            flag = "  ★越界" if d["tanh_max"] > nf.CLIP else ""
            print(f"    {name:5s} {d['raw_max']:9.5f} {d['coverage_margin']:6.2f} "
                  f"{d['util_p99.5']:6.3f} {d['tanh_max']:9.4f}{flag}")
        print(f"    軌道利用率比 {rep['util_ratio_orbit']:.2f}"
              f"（逐通道 {rep['util_ratio_per_channel']:.2f}）")
        if n_clip:
            raise RuntimeError(
                f"★ 有 {n_clip} 個通道的時均場超出凍結尺度的可表達範圍。"
                "這些值反正規化後不可逆。依凍結紀律，要修的是資料或報告限制，"
                "**不是** norm_frozen.S —— 改常數會讓所有既有結果失去可比性。")
        return rep


# ══════════════════════════════════════════════════════════════════════════════
# Pass 2 — Zarr 寫入
# ══════════════════════════════════════════════════════════════════════════════


def pass2_write_zarr(
    h5_files: list[Path],
    output_zarr: Path,
    upstream_mean: float,
    upstream_std: float,
    snapshot_mode: str = "ic_only",
) -> dict[str, Any]:
    """Pass 2：正規化所有欄位，寫入 TES Zarr（v4）。

    physics 通道一律用 :mod:`norm_frozen` 的凍結常數，不接受任何量測輸入 ——
    這是「跨資料集同一組尺度」在**構造上**成立的地方。

    Static 各通道 norm：
      C0 SDF      → tanh(SDF_h5s / L_char_h5s)，per-case，零點保留
      C1 upstream → (upstream - upstream_mean) / upstream_std，全域 z-score
      C2 X_01     → raw [0,1]，入口=0，出口=1
      C3 Y_01     → raw [0,1]，底=0，頂=1

    Args:
        h5_files: H5 檔案路徑列表。
        output_zarr: 輸出 Zarr 路徑。
        upstream_mean: C1 upstream 全域均值（pass1a 計算；static 仍為量測值）。
        upstream_std: C1 upstream 全域標準差。
        snapshot_mode: 湍流時序的保存範圍。

            * ``"none"``    —— 完全不存 ``snapshots``。
            * ``"ic_only"`` —— 只存第 0 幀（**預設**）。mean-only 範疇下
              下游唯一會讀的就是它（冷啟動 IC，見
              ``src/lbmnca/nca/tes/data/graph_dataset.py``：``src`` 為 None 時
              取 ``snapshots[0:1]``），而它是靜止態、偏差恆為 0。
            * ``"full"``    —— 存完整軌跡。⚠ 凍結尺度是對**時均場**校準的，
              過渡段的極值會超出可表達範圍而被截斷、**反正規化不可逆**。
              只在明確需要湍流時序時使用，並自負不可逆的後果。

    Returns:
        凍結尺度的符合度報告（``ComplianceAccumulator.verdict()``）。
    """
    if snapshot_mode not in ("none", "ic_only", "full"):
        raise ValueError(f"未知 snapshot_mode={snapshot_mode!r}")
    m_ref = M_REF.reshape(NUM_CHANNELS, 1, 1)
    s_combined = S_FROZEN.reshape(NUM_CHANNELS, 1, 1)
    n_reclassified = 0
    compliance = ComplianceAccumulator()

    root = zarr.open_group(str(output_zarr), mode="w")
    root.attrs.update({
        # ★ v4：physics 正規化改用凍結常數、固體格覆寫為 M_REF、遮罩改用面積占比。
        #   下游必須據此拒絕舊版（數值意義不同），故版本字串一併升。
        "version": "tes_v4",
        "norm_schema": nf.SCHEMA_VERSION,
        "solid_overwritten_to_m_ref": True,
        "solid_mask_source": "h5_mask | area_downsampled_png_phi>0",
        "snapshot_mode": snapshot_mode,
        "scope": "mean_only",
        "H": H5_H,
        "W": H5_W,
        "n_cases": len(h5_files),
        "created_at": datetime.now(timezone.utc).isoformat(),
        # ★ Stage 1.16（1.15 三項強制後續之 A1）：所有 mean_*_delta_* / *_se 品質
        # 欄位都是 **tanh 正規化空間**的 rel_l2，其數值直接取決於 S_combined。
        # 1.15 報告與 v4 zarr 存檔對不上（0.1853 vs 0.17103 等）的唯一原因，就是
        # 兩邊用了不同 dataset 的 stats.json。此處把常數本身寫進資料層，
        # 使任何引用都能自證是在哪組正規化下量的，同類混淆不會再發生。
        "gt_quality_metric": "tanh_fluid_masked_9ch_rel_l2",
        "gt_quality_norm_dataset": output_zarr.stem,
        "gt_quality_norm_S_combined": S_FROZEN.tolist(),
        "gt_quality_norm_M_REF": M_REF.tolist(),
    })
    cases_group = root.require_group("cases")

    # C2-2 WarmGraph：full-window case 的 mean_flow 快取（case_id → [9,H,W]）
    # 與 h5 路徑，供 short-window relation edge 做 attractor 一致性 QA。
    # h5_files 已排序 full 在前（main 端 _split_by_window），canonical 必先處理。
    canonical_means: dict[str, np.ndarray] = {}
    canonical_paths: dict[str, Path] = {}
    canonical_starts: dict[str, float] = {}   # ★ 1.16：canonical 的實際時均窗起點
    attractor_qa: list[tuple[str, str, float, float, float]] = []

    for h5_path in tqdm(h5_files, desc="[Pass 2] 寫入 Zarr"):
        with h5py.File(h5_path, "r") as f:
            attrs = _read_h5_attrs(f)
            moments = f["trajectory/moments"][:].astype(np.float32)  # [T, 9, H, W]
            mask_hw = f["geometry/mask"][:]                            # [H, W] bool
            ctu_values = f["trajectory/ctu"][:].astype(np.float32)    # [T]
            # ★ Stage 1.15：求解迴圈內逐步累加的分段時均（新 H5 才有）。
            # 明確的欄位存在性檢查（`in f`），非帶預設值的 config fallback。
            mean_accum = read_mean_accum(f)

        # 優先從原始 mask PNG 計算 g_vec（H5 mask 含 LBM 邊界牆，不可用）
        masks_dir = _find_masks_dir(h5_path)
        _gid = str(attrs.get("geometry_id", ""))  # optional — H5 attrs
        solid_hw = _load_mask_png_solid(masks_dir, _gid)
        g_vec = _compute_gvec_from_mask(solid_hw)

        # ── ★ 乾淨固體遮罩 ＋ 固體格覆寫為 M_REF ──────────────────────────
        # 覆寫的四個理由：
        #   1. 固體格不帶流場資訊；幾何由 static ch0 的 SDF 表達，零點即界線
        #   2. 消除流固界面上 ch0-2 的硬階躍 —— NCA 的 conv 會吃到那個階躍
        #   3. 使 clip_rate 在**全域**（含固體）也恆為 0，保證才是全稱的
        #   4. 損失端本就套 fluid_mask，故不改變任何梯度語義
        clean_solid, n_reclass = _clean_solid_mask(mask_hw, masks_dir, _gid)
        n_reclassified += n_reclass
        moments[:, :, clean_solid] = M_REF[None, :, None]
        mask_hw = clean_solid          # 之後的 SDF／fluid_mask 一律用乾淨遮罩

        case_key = _make_case_key(attrs)
        case_grp = cases_group.require_group(case_key)

        # ── case attrs ──
        # H5 attrs 為系統邊界，以下欄位均為 optional（舊 H5 可能缺失）
        _cctu = float(attrs.get("convergence_ctu", -1.0))
        case_grp.attrs.update({
            "ic_type":     str(attrs.get("ic_type", "cold")),             # optional
            "mask_id":     str(attrs.get("geometry_id", "unknown")),       # optional
            "re":          float(attrs.get("re", 0.0)),                    # optional
            "n_snapshots": int(attrs.get("n_snapshots", moments.shape[0])), # optional — fallback to T
            "ctu_values":  ctu_values.tolist(),
            # DD-02：補收斂 attrs（訓練/eval/pair loader 需）。DD-03：converged 以
            # convergence_ctu>=0 判定（非 H5 status-based 的 converged=True）。
            "case_id":         str(attrs.get("case_id", "")),
            "convergence_ctu": _cctu,
            "converged":       bool(_cctu >= 0.0),
            "stop_reason":     str(attrs.get("stop_reason", "")),
        })

        # ── C2-2 WarmGraph edge attrs（graph dataset 才有；空值表舊 dataset）──
        _edge_attrs = {
            k: str(attrs.get(k, ""))
            for k in ("edge_id", "edge_type", "edit_axis",
                      "window_type", "canonical_case_id")
        }
        if any(_edge_attrs.values()):
            case_grp.attrs.update(_edge_attrs)

        # ── snapshots：physics-informed normalization + tanh soft-clip → float16 ──
        # out = tanh((raw - M_REF) / S_combined)  ∈ (-1, 1)
        # P99.5 值映射到 tanh(1) ≈ 0.76；群組共享 A 保留通道間量值比例
        # ★ mean-only：預設只保留第 0 幀（冷啟動 IC）。凍結尺度是對時均場校準的，
        #   過渡段極值會超出可表達範圍 —— 存下去就是存一份反正規化不回來的資料。
        if snapshot_mode != "none":
            _sel = slice(0, 1) if snapshot_mode == "ic_only" else slice(None)
            snaps_norm = np.tanh((moments[_sel] - m_ref) / s_combined).astype(np.float16)
            case_grp.create_dataset(
                "snapshots",
                data=snaps_norm,
                chunks=(1, 9, H5_H, H5_W),
                dtype="float16",
                compressor=COMPRESSOR,
                overwrite=True,
            )

        # ── static：per-channel norm → float32 ──
        l_char_px = float(attrs["l_char_px"])
        l_char_h5s = l_char_px / H5_DOWNSAMPLE           # L_char 在 H5s px 單位
        static_raw = _compute_static_raw(mask_hw, l_char_px)  # (4, H, W)
        c0_sdf  = np.tanh(static_raw[0] / l_char_h5s)         # tanh，零點保留
        c1_ups  = (static_raw[1] - upstream_mean) / upstream_std  # z-score
        c2_x    = static_raw[2]                                # raw [0,1]
        c3_y    = static_raw[3]                                # raw [0,1]
        static_norm = np.stack([c0_sdf, c1_ups, c2_x, c3_y], axis=0).astype(np.float32)
        case_grp.create_dataset(
            "static",
            data=static_norm,
            chunks=(4, H5_H, H5_W),
            dtype="float32",
            compressor=COMPRESSOR,
            overwrite=True,
        )

        # ── g_vec：raw ──
        case_grp.create_dataset("g_vec", data=g_vec, dtype="float32", overwrite=True)

        # ── p_vec：raw [Re, U_inlet_lu, L_char_px] ──
        p_vec = _extract_pvec(attrs)
        case_grp.create_dataset("p_vec", data=p_vec, dtype="float32", overwrite=True)

        # ── delta_ctus：ctu_values 差分 [T-1]，DataLoader 直接切片，無需 runtime 計算 ──
        delta_ctus = np.diff(ctu_values).astype(np.float32)
        case_grp.create_dataset("delta_ctus", data=delta_ctus, dtype="float32", overwrite=True)

        # ── mean_flow_post_conv：時均流場 [9, H, W] float32 ──
        # ★ Stage 1.16：時均窗起點改由 _resolve_mean_start_ctu 決定。
        #   有 /mean_accum → 協定常數（與收斂閂鎖無關）；無 → 沿用 convergence_ctu。
        convergence_ctu = float(attrs.get("convergence_ctu", -1.0))
        # ★2026-07-29 L3：新 config 明寫 warmup/burnin，優先於固定比例。
        #   舊 H5 沒有這兩個 attr（或為 -1 哨兵）→ 自動落回原路徑。
        mean_start_ctu, mean_window_basis = _resolve_mean_start_ctu(
            convergence_ctu=convergence_ctu,
            total_ctu=float(ctu_values[-1]),
            has_accum=mean_accum is not None,
            burnin_ctu=float(attrs.get("burnin_ctus", -1.0)),
            warmup_ctu=float(attrs.get("warmup_ctus", -1.0)),
        )
        # snapshot 路徑與累加路徑用**同一個起點**，兩法才是同窗可比。
        time_mean_mask = ctu_values >= mean_start_ctu
        if not time_mean_mask.any():
            raise ValueError(
                f"{case_key}: 時均窗起點 {mean_start_ctu:.2f} CTU 之後無任何 snapshot"
                f"（軌跡終點 {float(ctu_values[-1]):.2f} CTU）"
            )

        # DD-07：在物理（raw moment）空間做 Reynolds 時均，再 tanh-normalize。
        # 舊版 snaps_norm.mean = E[tanh(x)] ≠ tanh(E[x])，在飽和區（±0.97）系統性偏差。
        # ★ 2026-07-26：時均窗函數改為可選（DEEP_AUDIT_2026-07 修正 2），見 MEAN_WINDOW。
        mean_moments = _weighted_time_mean(moments[time_mean_mask], MEAN_WINDOW)
        mean_flow = np.tanh((mean_moments - m_ref) / s_combined).astype(np.float32)

        # ── Stage 1.15：求解迴圈內逐步累加時均（新欄位；舊 H5 無 → 退回舊法）──
        fluid_hw = ~np.asarray(mask_hw, dtype=bool)   # True = 流體（已是乾淨遮罩）
        running = window_mean(mean_accum, mean_start_ctu) if mean_accum is not None else None
        # ★ 累加時均直接讀自 H5 的 /mean_accum，**沒有**經過上面對 moments 的覆寫，
        #   故固體格仍是 raw 0 → 正規化後會是 tanh(-1/S_rho) = -1（飽和）。
        #   這正是覆寫要消除的界面硬階躍，兩條路徑都得補上。
        if running is not None:
            for _k in ("mean", "half1", "half2"):
                if running.get(_k) is not None:
                    running[_k][:, clean_solid] = M_REF[:, None]
        # ★ 凍結尺度的驗收對象：兩條時均路徑都納入（模型可能吃到任一條）
        compliance.add(mean_moments, fluid_hw)
        if running is not None:
            compliance.add(running["mean"], fluid_hw)
        mean_flow_running = (
            np.tanh((running["mean"] - m_ref) / s_combined).astype(np.float32)
            if running is not None else None
        )

        # ★ Stage 1.16：起點敏感度（把「收斂偵測只是粗略提示」這句話量化）。
        # 把窗起點往後推一段，量時均差多少 → 若這個數字小，起點選錯的代價就小；
        # 大則代表該 case 的時均仍受起點支配，必須在資料層看得到，不能假設可忽略。
        run_start_delta_1seg = GT_QUALITY_SENTINEL
        if running is not None and mean_accum is not None:
            # ★ 位移基準必須是**窗內第一段的起點**（running["ctu0"]），不是
            # mean_start_ctu。用後者會退化：當 mean_start_ctu 落在兩個段界之間時，
            # 「下一個 ctu0 > mean_start_ctu 的段」就是窗的第一段本身 → 選段集合
            # 完全相同 → 差量恆為 0，這個 attr 會變成永遠 0 的假訊號。
            # （聲學分段實測踩到：start=20.00 落在段界 20.318 之前 → 量到 0.0）
            seg_ctu0 = mean_accum["segment_ctu0"]
            later = seg_ctu0[seg_ctu0 > float(running["ctu0"]) + 1e-9]
            if later.size > 0:
                shifted = window_mean(mean_accum, float(later[0]))
                if shifted is not None:
                    run_start_delta_1seg = delta_norm_fluid(
                        shifted["mean"], running["mean"], m_ref, s_combined, fluid_hw
                    )

        # snapshot 法的自我不確定度（同度量：tanh、fluid-masked、9ch rel_l2）：
        # 把時均窗的 snapshot 前後對切，量兩半均值的差 → GT 時均自身的抽樣雜訊。
        # 這是 1.12 null-control 的量法，此處常態化成每個 case 的 GT 品質欄位。
        snap_delta_self = GT_QUALITY_SENTINEL
        idx_win = np.flatnonzero(time_mean_mask)
        if idx_win.size >= 4:
            _cut = idx_win.size // 2
            _m1 = moments[idx_win[:_cut]].mean(axis=0)
            _m2 = moments[idx_win[_cut:]].mean(axis=0)
            snap_delta_self = delta_norm_fluid(_m2, _m1, m_ref, s_combined, fluid_hw)

        run_delta_self = GT_QUALITY_SENTINEL
        if running is not None and running["half1"] is not None:
            run_delta_self = delta_norm_fluid(
                running["half2"], running["half1"], m_ref, s_combined, fluid_hw
            )

        # ★ 來源切換：預設 snapshot（既有行為，既有 zarr 不受影響）。
        mean_flow_source = "snapshot"
        if MEAN_SOURCE == "running" and mean_flow_running is not None:
            mean_flow = mean_flow_running
            mean_flow_source = "running_accum"

        window_type = str(attrs.get("window_type", ""))  # optional — 舊 H5 無此 attr
        if window_type == "short":
            # C2-2 relation edge：15 CTU 驗證窗統計不足以當 mean/turb 監督訊號，
            # 不寫 mean_flow_post_conv；下游經 mean_ref 引用 canonical node。
            # 本地短窗 mean 只用於 attractor 一致性 QA（自校準比值，見常數區）。
            canon_id = str(attrs.get("canonical_case_id", ""))
            case_grp.attrs["mean_ref"] = canon_id
            canon_mean = canonical_means.get(canon_id)
            if canon_mean is None:
                print(f"  [QA][WARN] {case_key}: canonical '{canon_id}' 不在本批 "
                      f"full-window cases，attractor QA 跳過")
                case_grp.attrs["attractor_rel_l2"] = -1.0
                case_grp.attrs["attractor_qa_ratio"] = -1.0
                case_grp.attrs["attractor_mismatch"] = True
            else:
                rel_l2 = float(
                    np.linalg.norm(mean_flow - canon_mean)
                    / (np.linalg.norm(canon_mean) + 1e-12)
                )
                # baseline：canonical 自己「同長度窗」時均 vs 全窗時均的 rel-L2
                # = 該 case 在此窗長下的固有相位殘留（噪聲底）
                # ★ 1.16：兩邊都改用 canonical_starts 記錄的實際窗起點
                #   （與 mean_window_basis 一致），不再各自重讀 convergence_ctu。
                span = float(ctu_values[-1]) - mean_start_ctu
                c_conv = canonical_starts[canon_id]
                with h5py.File(canonical_paths[canon_id], "r") as cf:
                    c_mom = cf["trajectory/moments"][:].astype(np.float32)
                    c_ctu = cf["trajectory/ctu"][:].astype(np.float32)
                c_sel = (c_ctu >= c_conv) & (c_ctu <= c_conv + span)
                c_win_mean = np.tanh(
                    (c_mom[c_sel].mean(axis=0) - m_ref) / s_combined
                ).astype(np.float32)
                baseline = float(
                    np.linalg.norm(c_win_mean - canon_mean)
                    / (np.linalg.norm(canon_mean) + 1e-12)
                )
                ratio = rel_l2 / max(baseline, 1e-6)
                case_grp.attrs["attractor_rel_l2"] = rel_l2
                case_grp.attrs["attractor_baseline"] = baseline
                case_grp.attrs["attractor_qa_ratio"] = ratio
                case_grp.attrs["attractor_mismatch"] = bool(
                    ratio > ATTRACTOR_QA_RATIO_MAX)
                attractor_qa.append((case_key, canon_id, rel_l2, baseline, ratio))
        else:
            case_grp.create_dataset(
                "mean_flow_post_conv",
                data=mean_flow,
                chunks=(9, H5_H, H5_W),
                dtype="float32",
                compressor=COMPRESSOR,
                overwrite=True,
            )
            # ★ 記錄實際使用的時均窗型，使跨 dataset 的不一致可被偵測
            #（DEEP_AUDIT_2026-07 修正 2）。
            case_grp.attrs["mean_window"] = MEAN_WINDOW
            case_grp.attrs["mean_n_frames"] = int(time_mean_mask.sum())
            case_grp.attrs["mean_flow_source"] = mean_flow_source
            # ★ Stage 1.16：時均窗起點的**基準**與**數值**明寫入資料層，
            # 使「這個時均是不是依賴收斂閂鎖」可被下游直接查詢，不必反推。
            case_grp.attrs["mean_window_basis"] = mean_window_basis
            case_grp.attrs["mean_window_start_ctu"] = float(mean_start_ctu)
            case_grp.attrs["mean_window_end_ctu"] = float(ctu_values[-1])
            # snapshot 法的自我不確定度（Δ_self 與 SE = Δ/2，見上方計算）
            case_grp.attrs["mean_snapshot_delta_self"] = float(snap_delta_self)
            case_grp.attrs["mean_snapshot_se"] = float(
                snap_delta_self / 2.0 if snap_delta_self > 0 else GT_QUALITY_SENTINEL
            )

            # ── ★ 兩段獨立半窗時均（VQI 上錨＝GT 自噪的唯一來源）──────────
            #   `e2_vqi.run_selfnoise` 原本切 `postconv` 快照序列求兩段均值，
            #   但 mean-only 不存快照。改在此直接落盤 —— 反正累加器早就算出
            #   `half1`/`half2`，而它們是**時均**、振幅在凍結尺度內。
            #   ⚠ 這兩份不可用 `mean_flow` 與 `mean_flow_running` 代替：
            #     兩者逐位元相同（`MEAN_SOURCE='running'` 時前者就是後者），
            #     拿來當「兩段獨立窗」會量到 0，噪聲底線直接消失。
            if running is not None and running.get("half1") is not None:
                for _tag in ("half1", "half2"):
                    case_grp.create_dataset(
                        f"mean_flow_{_tag}",
                        data=np.tanh((running[_tag] - m_ref) / s_combined).astype(np.float32),
                        chunks=(9, H5_H, H5_W), dtype="float32",
                        compressor=COMPRESSOR, overwrite=True,
                    )
                case_grp.attrs["mean_halves_available"] = True
            else:
                case_grp.attrs["mean_halves_available"] = False

            # ── Stage 1.15：累加時均 ──
            # ★ `MEAN_SOURCE='running'` 時 `mean_flow_post_conv` **就是**這一份
            #   （實測逐位元相同）。存第二份只是把同一個陣列寫兩次。
            if mean_flow_running is not None and mean_flow_source != "running_accum":
                case_grp.create_dataset(
                    "mean_flow_post_conv_running",
                    data=mean_flow_running,
                    chunks=(9, H5_H, H5_W),
                    dtype="float32",
                    compressor=COMPRESSOR,
                    overwrite=True,
                )
                case_grp.attrs["mean_running_available"] = True
                case_grp.attrs["mean_running_n_steps"] = int(running["n_steps"])
                case_grp.attrs["mean_running_n_segments"] = int(running["n_segments"])
                case_grp.attrs["mean_running_ctu0"] = float(running["ctu0"])
                case_grp.attrs["mean_running_ctu1"] = float(running["ctu1"])
                case_grp.attrs["mean_running_delta_self"] = float(run_delta_self)
                case_grp.attrs["mean_running_se"] = float(
                    run_delta_self / 2.0 if run_delta_self > 0 else GT_QUALITY_SENTINEL
                )
                case_grp.attrs["mean_running_vs_snapshot"] = float(
                    delta_norm_fluid(running["mean"], mean_moments,
                                     m_ref, s_combined, fluid_hw)
                )
                case_grp.attrs["mean_running_start_delta_1seg"] = float(
                    run_start_delta_1seg
                )
                case_grp.attrs["mean_accum_stride"] = int(mean_accum["attrs"]["stride"])
                case_grp.attrs["mean_accum_segment_ctu"] = float(
                    mean_accum["attrs"]["segment_ctu_target"]
                )
                # ── ★ Stage 1.16 修 3：聲學可公度性（**記錄用，非門檻**）──────
                # 1.17：主導 GT 時均不確定度的是求解域縱向聲學四分之一波共振
                # （T = 4*nx/c_s /(1-M^2)，在**步**單位下是母體常數），不是渦脫。
                # SE(L) = A*|sin(pi L/T)|/(pi L/T) → 窗長取 T 的整數倍時相位殘差歸零
                # （1.17 實測窗長差 2.5% → SE 差 15.6 倍）。
                # 以下欄位讓下游**一眼看出窗是否可公度**；
                # ⛔ 但不據此過濾或加權任何 case —— 這是最佳化不是正確性判準，
                #    且 1.17 只在 Re=100 圓柱層流極限環驗證過，高 Re 密集多建物
                #    case 上聲學佔比可能被寬頻尾流稀釋、增益等比下降（未實測）。
                _acc_attrs = mean_accum["attrs"]
                _t_ac = float(_acc_attrs["acoustic_period_steps"]) if (
                    "acoustic_period_steps" in _acc_attrs) else GT_QUALITY_SENTINEL
                case_grp.attrs["acoustic_period_steps"] = _t_ac
                case_grp.attrs["mean_accum_segment_steps"] = int(
                    _acc_attrs["segment_steps_target"]
                ) if "segment_steps_target" in _acc_attrs else -1
                case_grp.attrs["mean_accum_segment_basis"] = str(
                    _acc_attrs["segment_basis"]
                ) if "segment_basis" in _acc_attrs else "ctu"
                if _t_ac > 0.0:
                    _n_ac = float(running["n_steps"]) / _t_ac
                    case_grp.attrs["n_acoustic_periods_in_window"] = _n_ac
                    case_grp.attrs["acoustic_commensurability_eps"] = float(
                        abs(_n_ac - round(_n_ac)) / max(round(_n_ac), 1.0)
                    )
                else:
                    case_grp.attrs["n_acoustic_periods_in_window"] = GT_QUALITY_SENTINEL
                    case_grp.attrs["acoustic_commensurability_eps"] = GT_QUALITY_SENTINEL
            else:
                case_grp.attrs["mean_running_available"] = False

            # ── ★ Stage 1.16 修 3：Δ_self 的語義註記（防誤用）────────────────
            # 1.17 實測 (Δ_self/2) / 真實系綜 SE 在 **0.40–5.29** 之間雙向跳動
            # （@L=80 高估 3.65 倍、@L=100 高估 5.06 倍）。原因：Δ_self 比的是兩個
            # 半窗，其相位歸零條件 L=2kT 與全窗的 L=kT **錯開半週期**。
            # → mean_*_delta_self / mean_*_se **不是標準誤**，不可當 SE 使用。
            #   真正的 SE 需要多窗系綜，單一 case 單一窗算不出來。
            case_grp.attrs["delta_self_semantics"] = (
                "half-window difference; NOT a standard error. "
                "(delta_self/2) vs true ensemble SE measured at 0.40-5.29x (Stage 1.17): "
                "half-window nulls at L=2kT, full-window at L=kT -> offset by half a period. "
                "True SE requires a multi-window ensemble."
            )

            # ── 收斂閂鎖的狀態（★ Stage 1.16：記錄，不當守衛）─────────────────
            # convergence_ctu / converged 仍原樣保留（誠實記錄求解器當時判了什麼），
            # 但它**已不決定** mean_flow 的窗（見 mean_window_basis）。
            # quasi_steady_check 直接透傳求解器的狀態字串，讓
            # 「converged=True 但收尾 KE 變化 > 5%」這個組合在資料層可被查詢。
            # ⚠ 這是**記錄欄位不是門檻**：1.13/1.11 已示範「不可能失敗的判準」
            #   比沒有判準更糟，故此處不據以過濾任何 case。
            for _k, _default in (
                ("quasi_steady_check", "unknown"),
                ("quasi_steady_source", ""),
            ):
                case_grp.attrs[_k] = str(attrs.get(_k, _default))  # optional — 舊 H5 無此 attr
            case_grp.attrs["quasi_steady_ke_change_pct"] = float(
                attrs.get("quasi_steady_ke_change_pct", GT_QUALITY_SENTINEL)  # optional
            )

            # SEM_eff：用時均窗內逐 frame「速度大小」空間平均值的時間序列估計
            # GT 時均場本身的取樣不確定度（見 mean_window.compute_sem_eff docstring）。
            n_mean_frames = int(time_mean_mask.sum())
            if n_mean_frames >= MIN_SAMPLES_FOR_SEM:
                rho_w  = np.maximum(moments[time_mean_mask, 0], 1e-6)         # [nf, H, W]
                velx_w = moments[time_mean_mask, CH_JX] / rho_w
                vely_w = moments[time_mean_mask, CH_JY] / rho_w
                vel_mag_series = np.sqrt(velx_w ** 2 + vely_w ** 2).mean(axis=(1, 2))  # [nf]
                case_grp.attrs["mean_sem_eff"] = float(
                    compute_sem_eff(vel_mag_series.astype(np.float64))
                )
            else:
                case_grp.attrs["mean_sem_eff"] = GT_QUALITY_SENTINEL

            canonical_means[case_key] = mean_flow
            canonical_paths[case_key] = h5_path
            canonical_starts[case_key] = float(mean_start_ctu)

    if attractor_qa:
        n_bad = sum(1 for *_x, ratio in attractor_qa
                    if ratio > ATTRACTOR_QA_RATIO_MAX)
        print(f"  [QA] attractor 一致性（自校準 ratio <= {ATTRACTOR_QA_RATIO_MAX}）："
              f"{len(attractor_qa) - n_bad} PASS / {n_bad} MISMATCH")
        for ck, cid, rl2, base, ratio in attractor_qa:
            flag = "  [!] MISMATCH" if ratio > ATTRACTOR_QA_RATIO_MAX else ""
            print(f"    {ck} -> {cid}  rel_l2={rl2:.4f} baseline={base:.4f} "
                  f"ratio={ratio:.2f}{flag}")

    print(f"  Zarr 寫入完成：{output_zarr}  ({len(h5_files)} cases)")
    print(f"  遮罩修正：{n_reclassified} 個部分固體格由「流體」改判為固體"
          f"（moments 覆寫為 M_REF）")
    print(f"  snapshot_mode={snapshot_mode}")
    return compliance.verdict()


# ══════════════════════════════════════════════════════════════════════════════
# Stats JSON 輸出
# ══════════════════════════════════════════════════════════════════════════════


def _build_stats_json(
    physics_norm: dict[str, Any],
    upstream_mean: float,
    upstream_std: float,
    g_vecs: np.ndarray | None,
    p_vecs: np.ndarray | None,
    eps: float = 1e-8,
    cond_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """組合完整 {dataset}_stats.json dict（tes_v3 格式）。

    Args:
        cond_stats: 已算好的 `cond_stats`（`--stats-from` 沿用訓練集常數時傳入）。
            給定時 `g_vecs`/`p_vecs` 不使用，可為 None。cond z-score 與
            static upstream 同屬「訓練集量測」，兩者必須同源。
    """
    if cond_stats is None:
        if g_vecs is None or p_vecs is None:
            raise ValueError("未提供 cond_stats 時，g_vecs/p_vecs 不可為 None")
        cond_stats = {
            "g_vec": {"mean": g_vecs.mean(axis=0).tolist(),
                      "std": (g_vecs.std(axis=0) + eps).tolist()},
            "p_vec": {"mean": p_vecs.mean(axis=0).tolist(),
                      "std": (p_vecs.std(axis=0) + eps).tolist()},
        }

    return {
        "version": "tes_v4",
        "snapshot_norm": {
            "method":      "tanh",
            "note":        f"out=tanh((raw-M_REF)/S); inv=atanh(clip(out,±{nf.CLIP:.7f}))*S+M_REF",
            "norm_groups": {k: list(v) for k, v in nf.ORBITS.items()},
        },
        # ── ★ A 類：凍結的物理常數（不隨資料變，重轉時不重算）──────────────
        "frozen_norm": {
            "_note": "norm_frozen.py 的常數副本，供追溯；執行期一律以程式碼為準",
            "schema_version": nf.SCHEMA_VERSION,
            "M_REF": nf.M_REF.tolist(),
            "S":     nf.S.tolist(),
            "ATANH_A": nf.ATANH_A,
            "CLIP":  nf.CLIP,
            "R":     nf.R.tolist(),
            "admissible_box": {k: list(v) for k, v in nf.admissible_box().items()},
        },
        # 向後相容欄位：下游若仍讀 physics_norm.S_combined 可直接取用。
        # `U_ref` / `rescale_abs_max` 已無意義（S 不再由量測合成），故不再輸出。
        "physics_norm": {
            "M_REF":      nf.M_REF.tolist(),
            "S_combined": nf.S.tolist(),
        },
        "compliance": physics_norm.get("compliance"),
        "static_norm": {
            "sdf":      {"method": "tanh_lchar",
                         "note":   "tanh(SDF_h5s / (L_char_px / H5_DOWNSAMPLE)), per-case, zero=boundary"},
            "upstream": {"method": "zscore",
                         "mean":   upstream_mean,
                         "std":    upstream_std},
            "pos_x":    {"method": "raw_01",
                         "note":   "(x+0.5)/W_h5s, 0=inlet, 1=outlet"},
            "pos_y":    {"method": "raw_01",
                         "note":   "(y+0.5)/H_h5s, 0=bottom, 1=top"},
        },
        "cond_stats": cond_stats,
        # colormap SSOT（denorm/lattice units，與 tanh norm 正交）：single global
        # robust p99.5（跨全部 C-series dataset × cases × frames，fluid-masked）。
        # 02 訓練 SSIM metric / sim-viz mp4 / 03 部署 colormap 三階段共用此固定尺度
        # → 不閃、跨 run/ckpt/dataset 可比。LBM Ma-limited 使各 dataset 範圍緊密一致
        # (vel p99.5 0.11-0.16, vor 0.014-0.018) → 單一 global 合理。
        "vis_range": {
            "vel_max":   0.14,     # global p99.5 = 0.1373
            "vor_range": 0.016,    # global p99.5 = 0.0162
            "p_center":  0.3333,   # rho/3, rho~1
            "p_range":   0.01,     # 壓力 colormap 半寬
            "pct":       99.5,
            "scope":     "global",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """TES H5→Zarr pipeline 主入口。"""
    parser = argparse.ArgumentParser(description="TES H5 → TES Zarr (v2) 轉換 pipeline")
    parser.add_argument(
        "--dataset-dir", required=True,
        help="TES dataset 目錄（含 raw/*.h5，e.g. dataset/TES_dataset/S1_1-Cold_M30C90）",
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Zarr 輸出根目錄（預設：<workspace>/dataset_zarr）",
    )
    parser.add_argument(
        "--dataset-role", required=True, choices=("train", "val", "test"),
        help=(
            "本 dataset 的角色。一個 dataset 一個角色，幾何互斥由生成端保證。"
            "★ Pass 1a/1b 的全域統計只納 train dataset（DEEP_AUDIT Stage 3.3）；"
            "val／test 必須搭配 --stats-from 沿用訓練集常數。"
            "刻意設為必填：省略即代表統計母體未定義，不得靜默退回全集。"
        ),
    )
    parser.add_argument(
        "--stats-from", default=None,
        help=("沿用既有 {ds}_stats.json 的正規化常數，不自行計算。"
              "★驗證／測試集**必須**用此選項指向訓練集的 stats——若各自算自己"
              "的尺度，模型評估時看到的數值範圍與訓練時不同，分數不可比。"
              "同時這也是它們的 manifest 無 train case 時唯一可行的路徑"
              "（全域統計母體為空，Pass 1a 會 raise）。"),
    )
    parser.add_argument(
        "--stats-scan-dirs", nargs="*", default=[],
        help=("額外納入全域統計母體的 dataset 目錄（與 --dataset-dir 同格式）。"
              "★用途：訓練集由多個 dataset 組成（D0-1 + D1-1 + D2-1）時，"
              "各自算自己的 static/cond 統計會讓不同 dataset 落在不同基準上。"
              "指定此選項可一次掃過全部訓練 dataset 產出單一組量測值，"
              "其餘 dataset 再以 --stats-from 沿用。"
              "（physics 尺度已凍結，不受此參數影響。）"
              "這些目錄同樣套用收錄過濾與 split=='train' 過濾。"),
    )
    parser.add_argument(
        "--snapshot-mode", default="ic_only", choices=("none", "ic_only", "full"),
        help=("湍流時序的保存範圍。預設 `ic_only`（只存第 0 幀＝冷啟動 IC）——"
              "mean-only 範疇下下游只讀這一幀。`full` 會存完整軌跡，但凍結尺度是對"
              "**時均場**校準的，過渡段極值會被截斷且反正規化不可逆，"
              "只在明確需要湍流時序時使用。"),
    )
    parser.add_argument(
        "--allow-no-meta", action="store_true",
        help=("dataset 無 dataset_meta.json 時放行（不套用收錄過濾）。"
              "僅供 bench／臨時 dataset；生產資料不得使用。"),
    )
    args = parser.parse_args()

    dataset_dir  = Path(args.dataset_dir).resolve()
    dataset_name = dataset_dir.name

    if args.output_root is not None:
        output_root = Path(args.output_root)
    else:
        output_root = zarr_dir()

    output_zarr  = output_root / f"{dataset_name}.zarr"
    output_stats = output_root / f"{dataset_name}_stats.json"
    input_dir    = dataset_dir / "raw"
    if not input_dir.is_dir():
        input_dir = dataset_dir

    print(f"Dataset:      {dataset_name}")
    print(f"Input H5 dir: {input_dir}")
    print(f"Output Zarr:  {output_zarr}")
    print(f"Output stats: {output_stats}")

    h5_files = _list_h5_files(input_dir)
    print(f"找到 {len(h5_files)} 個 H5 案例")

    # 過濾：只保留可讀取且 converged=True 的案例（stability_failure 排除）
    print("\n── 過濾非收斂 / 不可讀 H5 ──")
    # 先用物理驗收閘門的結果過濾，再談可讀性。順序不能反——
    # _filter_readable_h5 只看 H5 自身的旗標，擋不住通過 KE 閂鎖卻沒通過
    # 物理閘門的 case。
    h5_files = _filter_admitted(h5_files, dataset_dir, args.allow_no_meta)
    h5_files = _filter_readable_h5(h5_files)
    if not h5_files:
        raise RuntimeError("過濾後無有效 H5 案例，請確認 dataset 是否完整")

    # ── window 分流：short（C2-2 relation edge）排除於全域統計 ─────────────
    full_files, short_files = _split_by_window(h5_files)
    if not full_files:
        raise RuntimeError("無 full-window H5 案例 — 全域統計需至少一個 full case")
    # Pass 2 順序：full 在前，確保 canonical mean 先於其 short edge 建立
    h5_files = full_files + short_files

    # ── 統計母體：限 train dataset（DEEP_AUDIT Stage 3.3）────────────────────
    stats_files = _stats_population(full_files, args.dataset_role,
                                    has_stats_from=bool(args.stats_from))

    # ── 額外統計母體：跨 dataset 的共用正規化常數 ───────────────────────────
    # 只影響 Pass 1a/1b 的統計，不影響本次寫入 zarr 的 case 集合（h5_files）。
    extra_dirs = [Path(d).resolve() for d in args.stats_scan_dirs]
    if extra_dirs and args.dataset_role != "train":
        raise ValueError(
            f"--stats-scan-dirs 只能在 --dataset-role=train 時使用"
            f"（現為 {args.dataset_role}）。非 train 角色一律沿用 --stats-from。")
    n_own = len(stats_files)
    for d in extra_dirs:
        if d == dataset_dir:
            raise ValueError(f"--stats-scan-dirs 不可包含 --dataset-dir 本身：{d}")
        e_in = d / "raw" if (d / "raw").is_dir() else d
        e_files = _filter_readable_h5(
            _filter_admitted(_list_h5_files(e_in), d, args.allow_no_meta))
        e_full, _ = _split_by_window(e_files)
        # ★掃進來的 dataset 由呼叫端負責確保同為 train 角色——這個參數的用途就是
        #   「多本 train dataset 共用一組常數」，混入非 train 會犯 Stage 3.3 的錯。
        print(f"  [統計母體 +] {d.name}: {len(e_full)} 個 full-window case")
        stats_files = stats_files + e_full
    if extra_dirs:
        print(f"  統計母體合計 {len(stats_files)} case"
              f"（本 dataset {n_own} ＋ 跨集 {len(stats_files) - n_own}）")

    # ── Pass 1a：static / cond 統計（physics 已凍結，不再量測）────────────────
    print("\n── Pass 1a：static / cond 統計（只納 full-window 的 train case）──")
    print(f"  [凍結] physics 正規化用 {nf.SCHEMA_VERSION}，不由資料產生")
    print(f"         M_REF = {nf.M_REF.tolist()}")
    print(f"         S     = {nf.S.tolist()}")
    borrowed = None
    if args.stats_from:
        _sp = Path(args.stats_from)
        if not _sp.exists():
            raise FileNotFoundError(f"--stats-from 指向的檔案不存在：{_sp}")
        borrowed = json.loads(_sp.read_text(encoding="utf-8"))
        _sc = borrowed.get("stats_scope", {})
        print(f"  [沿用] static/cond 統計取自 {_sp.name}"
              f"（role={_sc.get('dataset_role')} n_cases={_sc.get('n_cases')}）")
        print("         static/cond 仍是量測值，驗證／測試集必須沿用訓練集的")
        p1a = {"max_vel": float("nan"),
               "upstream_mean": float(borrowed["static_norm"]["upstream"]["mean"]),
               "upstream_std": float(borrowed["static_norm"]["upstream"]["std"]),
               "g_vecs": None, "p_vecs": None}
        borrowed_cond = borrowed["cond_stats"]
    else:
        p1a = pass1a_scan_stats(stats_files)
        borrowed_cond = None

    # ★ 凍結尺度的符合度驗收改在 Pass 2 內累積 —— 驗收對象是 mean_flow，
    #   而 mean_flow 正是在 Pass 2 算出來的（見 ComplianceAccumulator）。
    physics_norm: dict[str, Any] = {"M_REF": nf.M_REF, "S_combined": nf.S}

    # ── upstream 統計 ────────────────────────────────────────────────────────
    upstream_mean: float = p1a["upstream_mean"]
    upstream_std:  float = p1a["upstream_std"]
    print(f"\n  upstream: mean={upstream_mean:.4f}  std={upstream_std:.4f}")

    # ── Pass 2 ───────────────────────────────────────────────────────────────
    print("\n── Pass 2：寫入 Zarr ──")
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    physics_norm["compliance"] = pass2_write_zarr(
        h5_files, output_zarr, upstream_mean, upstream_std,
        snapshot_mode=args.snapshot_mode)

    # ── 輸出 stats JSON ───────────────────────────────────────────────────────
    stats_dict = _build_stats_json(
        physics_norm, upstream_mean, upstream_std,
        p1a["g_vecs"], p1a["p_vecs"], cond_stats=borrowed_cond,
    )
    # 統計母體 provenance（DEEP_AUDIT Stage 3.3：正規化常數不得含 val/test 成分）
    # ★ 沿用時，n_cases 必須報**來源**的母體大小。報本 dataset 的 0 會讓
    #   provenance 看起來像「統計由空集算出」，是誤導。
    _src_scope = (borrowed or {}).get("stats_scope", {})
    stats_dict["stats_scope"] = {
        "dataset_role": args.dataset_role,
        "n_cases": int(_src_scope.get("n_cases", 0)) if borrowed else len(stats_files),
        "n_full_window_cases": len(full_files),
        "stats_from": str(Path(args.stats_from).resolve()) if args.stats_from else None,
        "scan_dirs": [str(d) for d in extra_dirs],
        # ★ physics 已凍結，不在此列 —— 這裡只列**量測**而來、需要 train-only
        #   保護的項目。physics 的來源是 norm_frozen，與 dataset 角色無關。
        "applies_to": ["static_norm.upstream", "cond_stats"],
        "frozen_applies_to": ["physics_norm.M_REF", "physics_norm.S_combined"],
    }
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"\n  global_stats.json 寫入：{output_stats}")
    print("\nTES H5->Zarr pipeline DONE")


if __name__ == "__main__":
    main()
