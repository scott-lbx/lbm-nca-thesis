"""
config_utils/geometry.py -- 薄包裝層，委派至 public.geometry。

正式實作位於 src/lbm_mrt_les/public/geometry.py。
本模組保留 fill_geometry / calc_l_char / calc_max_blockage 簽名，
供 config_batch_gen.py 原有呼叫不需變動。

v2: 優先使用 metadata 中的 OSM-based building_dims（來自 generate_masks），
    僅在 metadata 缺少此欄位時才 fallback 到 pixel-based 計算。
"""

from typing import Optional

from lbmnca.lbm.lbm_mrt_les.public.geometry import (
    compute_blockage_metrics_from_png,
    compute_characteristic_length_from_png,
)


def fill_geometry(
    mask_ctx: dict,
    sim_ctx: dict,
    meta_entry: Optional[dict] = None,
) -> None:
    """計算 L_char 與 max_blockage，結果寫入 mask_ctx。

    優先使用 metadata 中 OSM-based 的 building_dims.l_char_median_px，
    僅在 metadata 缺少此欄位時 fallback 到 pixel-based connected component 計算。

    Args:
        mask_ctx: per-mask context 字典（會被就地修改）。
        sim_ctx: 共用 sim context。
        meta_entry: metadata.json 中對應條目（含 building_dims）。
    """
    # --- L_char: OSM-based 優先 ---
    # ⚠ 語義分歧警告（DEEP_AUDIT_2026-07 / FIX_B_Lchar統一.md §第四套語義）：
    # 此分支使用 OSM metadata 的 `l_char_median_px`＝**全體建物短邊的中位數**，
    # 與 canonical L_char（**單一最大建物**的 cross-flow y-extent，見 lchar_api）
    # 是不同的量。median 對多建物場景系統性小於 max，且未區分流向。
    # 目前 C-series 主力資料集不走此分支（cases.json 無 building_dims），
    # 僅 OSM 來源的 mask 會觸發。若日後要用 OSM 資料集訓練，必須先決定：
    #   (a) 改用 lchar_api 統一，或 (b) 明確記載此資料集的 Re 定義與 C-series 不同。
    # 在做出決定前，此處保留原行為但顯式標記 source 供下游辨識。
    bdims = (meta_entry or {}).get("building_dims")
    if bdims and "l_char_median_px" in bdims:
        mask_ctx["l_char"] = int(bdims["l_char_median_px"])
        mask_ctx["l_char_source"] = "osm_median"   # ≠ canonical（見上方警告）
        mask_ctx["l_char_max_px"] = int(bdims.get("l_char_max_px", bdims["l_char_median_px"]))
    else:
        mask_ctx["l_char"] = calc_l_char(
            png_path=mask_ctx["mask_path"],
            invert=sim_ctx["mask_invert"],
            nx=mask_ctx["nx"],
            ny=mask_ctx["ny"],
        )
        mask_ctx["l_char_source"] = "pixel_connected_component"
        mask_ctx["l_char_max_px"] = mask_ctx["l_char"]

    # --- Blockage ---
    metrics = compute_blockage_metrics_from_png(
        png_path=mask_ctx["mask_path"],
        invert=sim_ctx["mask_invert"],
        nx=mask_ctx["nx"],
        ny=mask_ctx["ny"],
        pad_right=mask_ctx["pad_right"],
        buffer=sim_ctx["blockage_buffer"],
    )
    mask_ctx["max_blockage"] = metrics["max_blockage"]


def calc_l_char(png_path: str, invert: bool, nx: int, ny: int) -> int:
    """從 PNG 計算特徵長度。委派至 public.geometry。"""
    return compute_characteristic_length_from_png(png_path, invert, nx, ny)


def calc_max_blockage(
    png_path: str,
    invert: bool,
    nx: int,
    ny: int,
    pad_right: int = 512,
    buffer: int = 128,
) -> float:
    """從 PNG 計算最大阻塞率。委派至 public.geometry。"""
    metrics = compute_blockage_metrics_from_png(
        png_path, invert, nx, ny, pad_right, buffer
    )
    return metrics["max_blockage"]
