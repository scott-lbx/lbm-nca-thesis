"""
從 master_config.yaml 建構批次生成用的共用 context 字典。

Velocity-inlet 模式：nu 由 u_inlet + target Re 反推，不需要 nu_lb_list。
"""

from typing import Any, Dict, Optional


def build_sim_context(master_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """建構整個批次生成作業共用的模擬 context。

    Args:
        master_cfg: 完整的 master_config.yaml 解析結果。

    Returns:
        共用 context 字典，供 config_batch_gen 各函式使用。
    """
    settings = master_cfg["settings"]
    physics = master_cfg["physics_control"]
    physical_constants = master_cfg["physical_constants"]
    base_template = master_cfg["template"]

    project_name = settings["project_name"]
    project_dir = f"SimCases/{project_name}"

    # 跨 repo 資料集路徑：當 dataset_batch_name 存在時，整個批次存在 dataset_root 下
    dataset_batch_name = settings.get("dataset_batch_name")  # optional — TES per-case 模式無此欄位
    dataset_root       = settings.get("dataset_root", "CEDS_dataset")  # optional — legacy 預設值
    if dataset_batch_name:
        batch_dir = f"../dataset/{dataset_root}/{dataset_batch_name}"
        mask_dir = f"{batch_dir}/masks"
        mask_meta_dir = batch_dir
        config_output_dir = f"{batch_dir}/configs"
        data_save_root = batch_dir
    else:
        mask_dir = f"{project_dir}/masks"
        mask_meta_dir = project_dir
        config_output_dir = f"{project_dir}/configs"
        data_save_root = f"SimCases/{project_name}"

    target_reynolds_numbers = physics.get("target_reynolds_numbers") or []  # optional — 空白時 fallback 空列表
    target_reynolds_numbers = sorted(
        [float(value) for value in target_reynolds_numbers if float(value) > 0.0]
    )

    inlet_velocity = physical_constants["inlet_velocity_ms"]
    u_phys = inlet_velocity[0] if isinstance(inlet_velocity, list) else inlet_velocity

    domain_policy = master_cfg.get("domain_policy", {})  # optional — 舊 config 可能未定義
    export_roi_policy = master_cfg.get("export_roi_policy", {})  # optional — 舊 config 可能未定義
    raw_ext = master_cfg.get("domain_ext_pad", {})  # optional — 舊 config 無此 section
    domain_ext_pad = {
        "left":  int(raw_ext.get("left",  0)),
        "right": int(raw_ext.get("right", 0)),
        "top":   int(raw_ext.get("top",   0)),
        "bot":   int(raw_ext.get("bot",   0)),
    }

    # Per-Re CTU 覆寫表（key 為 int Re，value 為 {warmup, start_record, total}）
    raw_ctu_per_re = physics.get("ctu_per_re", {})  # optional — 未定義時使用全域 CTU 設定
    ctu_per_re: Dict[int, Dict[str, float]] = {
        int(re): {k: float(v) for k, v in entry.items()}
        for re, entry in raw_ctu_per_re.items()
    }

    return {
        "dataset_batch_name": dataset_batch_name,
        "rho_out": physics["rho_out"],
        "target_reynolds_numbers": target_reynolds_numbers,
        "warmup_passes": physics["warmup_passes"],
        "total_passes": physics["total_passes"],
        "start_record_passes": physics["start_record_passes"],
        "ctu_per_re": ctu_per_re,
        "saves_per_ctu": physics["saves_per_physical_second"],
        "c_smag": physics["smagorinsky_constant"],
        "U_phys": float(u_phys),
        "nu_air": float(physical_constants.get("kinematic_viscosity_air_m2_s", 1.5e-5)),  # optional — 舊 config 無此欄位，用空氣預設值
        "blockage_buffer": int(settings.get("blockage_buffer", 128)),  # optional — 預設 128px
        "mask_invert": bool(base_template.get("mask", {}).get("invert", False)),  # optional — 預設不反轉
        "project_name": project_name,
        "data_save_root": data_save_root,
        "output_dir": config_output_dir,
        "mask_dir": mask_dir,
        "mask_meta_dir": mask_meta_dir,
        "base_template": base_template,
        "physical_constants": physical_constants,
        "domain_ext_pad": domain_ext_pad,
        "domain_policy": {
            "upstream_min_lchar": float(domain_policy.get("upstream_min_lchar", 5.0)),
            "downstream_min_lchar": float(domain_policy.get("downstream_min_lchar", 10.0)),
            "lateral_min_lchar": float(domain_policy.get("lateral_min_lchar", 5.0)),
            "sponge_fraction": float(domain_policy.get("sponge_fraction", 0.4)),
            "buffer_px": int(domain_policy.get("buffer_px", 128)),
            "max_domain_pixels": int(domain_policy.get("max_domain_pixels", 15_000_000)),
            "enforce": bool(domain_policy.get("enforce", False)),
        },
        "export_roi_policy": {
            "mode": str(export_roi_policy.get("mode", "sponge_plus_buffer")),
            "save_resolution_height": int(
                export_roi_policy.get("save_resolution_height", 256)
            ),
        },
    }


def build_mask_context(
    mask_path: str,
    meta_entry: Dict[str, Any],
    sim_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """從 mask 路徑與 metadata 建構 per-mask context。

    Args:
        mask_path: 遮罩 PNG 檔案路徑。
        meta_entry: metadata.json 中對應的條目。
        sim_ctx: 共用 sim context（可選），含 domain_ext_pad；
                 當 metadata 無 ext_pad 時作為 fallback。

    Returns:
        mask context 字典，含 domain 尺寸、padding 等資訊。
    """
    import os

    mask_stem = os.path.splitext(os.path.basename(mask_path))[0]

    # ext_pad 優先順序：metadata（舊格式有值）> sim_ctx.domain_ext_pad > 0
    fallback = (sim_ctx or {}).get("domain_ext_pad", {})  # optional — sim_ctx 為 Optional 參數
    ext_pad_left  = int(meta_entry.get("ext_pad_left",  fallback.get("left",  0)))  # optional — 新格式由 sim_ctx 提供
    ext_pad_right = int(meta_entry.get("ext_pad_right", fallback.get("right", 0)))  # optional
    ext_pad_top   = int(meta_entry.get("ext_pad_top",   fallback.get("top",   0)))  # optional
    ext_pad_bot   = int(meta_entry.get("ext_pad_bot",   fallback.get("bot",   0)))  # optional

    return {
        "mask_path": mask_path,
        "mask_stem": mask_stem,
        # POI/mask 尺寸（= H5/mp4 輸出邊界）
        "nx": int(meta_entry["domain_W_total"]),
        "ny": int(meta_entry["domain_H_total"]),
        # mask 內部邊距（從 mask 邊緣到建物群）
        "pad_right": int(meta_entry["pad_right"]),
        "pad_top": int(meta_entry["pad_top"]),
        "pad_bot": int(meta_entry["pad_bot"]),
        "pad_left": int(meta_entry["pad_left"]),
        # 外部 sponge+buffer（solver 往外加的量）
        # 新格式：metadata 無此欄位，由 sim_ctx.domain_ext_pad 提供
        # 舊格式：metadata 有此欄位，直接使用
        "ext_pad_left":  ext_pad_left,
        "ext_pad_right": ext_pad_right,
        "ext_pad_top":   ext_pad_top,
        "ext_pad_bot":   ext_pad_bot,
        "l_char": None,
        "max_blockage": None,
        "min_distance_px": float(meta_entry.get("min_distance_px", 0)),  # optional — 舊 metadata 無此欄位
        "n_buildings": int(meta_entry.get("n_buildings", 0)),  # optional — 舊 metadata 無此欄位
    }
