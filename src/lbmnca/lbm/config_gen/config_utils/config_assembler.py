"""
從共用 context 與個案資料建構單一 case YAML config。
"""

import copy
import os

from lbmnca.lbm.lbm_mrt_les.public.geometry import compute_boundary_extent_metrics


def _format_case_suffix(case_result: dict) -> str:
    nu_lb = case_result["nu_lb"]
    nu_str = f"{nu_lb:.4f}".replace(".", "-")
    target_re = case_result.get("target_reynolds_number")  # optional — nu-only mode 無此欄位

    if target_re is None:
        return f"Nu{nu_str}"
    return f"ReT{int(round(target_re)):04d}_Nu{nu_str}"


def _evaluate_boundary_extent(mask_ctx: dict, sim_ctx: dict) -> dict:
    """計算邊界距離指標並與 domain_policy 比較，回傳 metadata + 警告。

    Args:
        mask_ctx: 含 l_char 的 mask context。
        sim_ctx: 含 domain_policy 的 sim context。

    Returns:
        包含 boundary_extent_px、boundary_extent_over_lchar、policy_warnings 的字典。
    """
    from lbmnca.lbm.lbm_mrt_les.public.mask import binary_mask_to_solver_mask, png_to_binary_mask

    l_char = mask_ctx["l_char"] or 1
    nx, ny = mask_ctx["nx"], mask_ctx["ny"]

    solid_hw = png_to_binary_mask(
        mask_ctx["mask_path"],
        invert=sim_ctx["mask_invert"],
        nx=nx,
        ny=ny,
    )
    solid_taichi = binary_mask_to_solver_mask(solid_hw)

    extent = compute_boundary_extent_metrics(
        solid_mask=solid_taichi,
        nx=nx,
        ny=ny,
        pad_left=mask_ctx["pad_left"],
        pad_right=mask_ctx["pad_right"],
        pad_top=mask_ctx["pad_top"],
        pad_bot=mask_ctx["pad_bot"],
        l_char_px=l_char,
    )

    # 與 domain_policy 比較
    policy = sim_ctx.get("domain_policy", {})  # optional — domain_policy 可能未定義
    warnings = []
    checks = [
        ("inlet", extent["inlet_over_lchar"], policy.get("upstream_min_lchar", 5.0)),    # optional — 使用設計預設值
        ("outlet", extent["outlet_over_lchar"], policy.get("downstream_min_lchar", 10.0)),  # optional
        ("top", extent["top_over_lchar"], policy.get("lateral_min_lchar", 5.0)),          # optional
        ("bottom", extent["bottom_over_lchar"], policy.get("lateral_min_lchar", 5.0)),    # optional
    ]
    for name, actual, required in checks:
        if actual < required:
            warnings.append(
                f"{name}: {actual:.2f} L_char < required {required:.1f} L_char"
            )

    return {
        "boundary_extent_px": {
            "inlet": extent["inlet_px"],
            "outlet": extent["outlet_px"],
            "top": extent["top_px"],
            "bottom": extent["bottom_px"],
        },
        "boundary_extent_over_lchar": {
            "inlet": round(extent["inlet_over_lchar"], 6),
            "outlet": round(extent["outlet_over_lchar"], 6),
            "top": round(extent["top_over_lchar"], 6),
            "bottom": round(extent["bottom_over_lchar"], 6),
        },
        "policy_warnings": warnings,
    }


def build_config(case_result: dict, mask_ctx: dict, sim_ctx: dict) -> tuple[dict, str]:
    """建構單一 case 的完整 YAML config 字典。

    Args:
        case_result: 個案計算結果（含 nu_lb, rho_in_case, steps 等）。
        mask_ctx: per-mask context（含 nx, ny, padding, l_char 等）。
        sim_ctx: 共用 sim context（含 policy、template 等）。

    Returns:
        (config_dict, output_yaml_path) 的 tuple。
    """
    config = copy.deepcopy(sim_ctx["base_template"])
    config["physical_constants"] = sim_ctx["physical_constants"]

    rho_in = case_result["rho_in_case"]
    nu_lb = case_result["nu_lb"]
    mask_stem = mask_ctx["mask_stem"]
    case_suffix = _format_case_suffix(case_result)
    sim_name = f"{mask_stem}_{case_suffix}"

    sim = config["simulation"]
    sim["name"] = sim_name
    sim["nu"] = float(f"{nu_lb:.6f}")
    sim["characteristic_length"] = float(mask_ctx["l_char"])
    sim["rho_in"] = float(rho_in)
    sim["rho_out"] = float(sim_ctx["rho_out"])
    sim["u_inlet"] = float(case_result["u_inlet"])
    sim["compute_step_size"] = case_result["interval"]
    sim["warmup_steps"] = case_result["warmup_steps"]
    sim["max_steps"] = case_result["max_steps"]
    sim["steps_per_ctu"] = case_result["steps_per_ctu"]
    sim["smagorinsky_constant"] = sim_ctx["c_smag"]
    # Solver domain = mask(POI) + external sponge+buffer (ext_pad per side)
    # ext_pad 來源優先順序：mask_ctx（已從 metadata 或 sim_ctx fallback 填入）
    ext_left  = mask_ctx["ext_pad_left"]
    ext_right = mask_ctx["ext_pad_right"]
    ext_top   = mask_ctx["ext_pad_top"]
    ext_bot   = mask_ctx["ext_pad_bot"]
    nx_solver = mask_ctx["nx"] + ext_left + ext_right
    ny_solver = mask_ctx["ny"] + ext_top  + ext_bot
    sim["nx"] = nx_solver
    sim["ny"] = ny_solver

    out = config["outputs"]
    out["project_name"] = sim_ctx["project_name"]
    out["data_save_root"] = sim_ctx["data_save_root"]
    out["target_rho_in"] = float(rho_in)
    out["start_record_step"] = case_result["start_record_step"]
    out["gui"]["interval_steps"] = case_result["interval"]
    out["video"]["interval_steps"] = case_result["interval"]
    out["video"]["filename"] = f"{sim_name}.mp4"
    out["dataset"]["interval_steps"] = case_result["interval"]
    out["dataset"].pop("folder", None)

    # --- export_roi_policy: 覆寫 save_resolution_height ---
    roi_policy = sim_ctx.get("export_roi_policy", {})  # optional — 未定義時用預設解析度
    out["dataset"]["save_resolution_height"] = roi_policy.get(
        "save_resolution_height", 256  # optional — 預設 256px
    )

    # --- domain_zones: sponge 計算（v3.0）---
    # 新格式：ext_pad 各邊 = sponge + buffer（各佔一半），sponge_X = buffer_X = ext_X // 2
    # 舊格式 fallback（ext_pad == 0）：mask = 全域，sponge 從 pad 和 sponge_fraction 推算。
    domain_zones = config.get("domain_zones", {})  # optional — 後續填入，初始可能為空
    if ext_left > 0:
        # v3.0：sponge = buffer = ext / 2（非對稱各邊獨立）
        domain_zones["sponge_in"]  = ext_left  // 2
        domain_zones["sponge_out"] = ext_right // 2
        domain_zones["sponge_top"] = ext_top   // 2
        domain_zones["sponge_bot"] = ext_bot   // 2
        domain_zones["buffer_in"]  = ext_left  // 2
        domain_zones["buffer_out"] = ext_right // 2
        domain_zones["buffer_top"] = ext_top   // 2
        domain_zones["buffer_bot"] = ext_bot   // 2
        domain_zones["buffer"] = 0  # deprecated；LBM_Writer fallback 用
    else:
        # 舊格式 fallback（mask = 全域，sponge 從 pad 和 sponge_fraction 推算）
        buffer = sim_ctx["domain_policy"]["buffer_px"]
        sf = sim_ctx.get("domain_policy", {}).get("sponge_fraction", 0.4)  # optional — 舊格式 fallback 預設值
        domain_zones["sponge_in"]  = max(1, int((mask_ctx["pad_left"]  - buffer) * sf))
        domain_zones["sponge_out"] = max(1, int((mask_ctx["pad_right"] - buffer) * sf))
        domain_zones["sponge_top"] = max(1, int((mask_ctx["pad_top"]   - buffer) * sf))
        domain_zones["sponge_bot"] = max(1, int((mask_ctx["pad_bot"]   - buffer) * sf))
        domain_zones["buffer"] = buffer
    domain_zones.pop("sponge_x", None)
    domain_zones.pop("sponge_y", None)
    domain_zones.pop("inlet_buffer", None)
    domain_zones.pop("sponge_inlet", None)
    config["domain_zones"] = domain_zones

    # --- mask embed offsets（供 mask_utils.create_mask 將 POI mask 嵌入 solver domain）---
    # embed_offset_col：從 solver domain 左邊界到 mask 左邊界的距離（image convention）
    # embed_offset_row：從 solver domain 上邊界到 mask 上邊界的距離（image convention）
    config["mask"]["embed_offset_col"] = ext_left   # = sponge_in + buffer_in
    config["mask"]["embed_offset_row"] = ext_top    # = sponge_top + buffer_top

    # Velocity inlet BC: value[0] = [u_inlet, 0.0]
    config["boundary_condition"]["type"] = [0, 2, 1, 2]
    config["boundary_condition"]["value"] = [
        [float(case_result["u_inlet"]), 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    config["mask"]["path"] = mask_ctx["mask_path"]

    # --- Phase 2: boundary_extent 指標 + policy 警告 ---
    extent_info = _evaluate_boundary_extent(mask_ctx, sim_ctx)

    config["generation_metadata"] = {
        "selection_mode": case_result.get("re_selection_mode"),              # optional — nu-only mode 無
        "target_reynolds_number": case_result.get("target_reynolds_number"),  # optional — nu-only mode 無
        "estimated_reynolds_number": case_result.get("estimated_reynolds_number"),   # optional
        "re_relative_error_to_target": case_result.get("re_relative_error_to_target"),  # optional
        "boundary_extent_px": extent_info["boundary_extent_px"],
        "boundary_extent_over_lchar": extent_info["boundary_extent_over_lchar"],
        "domain_policy_warnings": extent_info["policy_warnings"],
    }

    if extent_info["policy_warnings"]:
        print(f"  [Warning] {sim_name}: boundary insufficient:")
        for w in extent_info["policy_warnings"]:
            print(f"    - {w}")

    config_filename = f"{mask_stem}_cfg_{case_suffix}.yaml"
    output_path = os.path.join(sim_ctx["output_dir"], config_filename)

    case_result["config_filename"] = config_filename
    case_result["sim_name"] = sim_name
    return config, output_path
