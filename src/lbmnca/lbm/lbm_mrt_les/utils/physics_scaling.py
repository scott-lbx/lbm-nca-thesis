"""lattice → physical unit 換算工具。

從 config 與 lattice 模擬結果計算真實物理量（長度、速度、時間尺度、Re）。
"""

from typing import Dict


def calculate_physical_params(config: Dict, lattice_metadata: Dict) -> Dict:
    """
    Calculates real-world physical parameters based on lattice results and scaling constants.
    
    Args:
        config: The configuration dictionary for the specific case.
        lattice_metadata: A dictionary of raw outputs from the lattice simulation run.
        
    Returns:
        A dictionary containing the calculated physical parameters (e.g., cell size in meters,
        time step in seconds, physical Reynolds number).
    """
    phys_const = config.get("physical_constants", {})  # optional — TES config 不含此欄位

    # Extract raw lattice data from the simulation results
    u_lb = lattice_metadata.get("u_inlet_lattice_lu", 0)       # optional — system boundary
    nu_lb = lattice_metadata.get("nu_lattice_lu", 0)            # optional — system boundary
    L_lb = lattice_metadata.get("l_char_lattice_px", 0)         # optional — system boundary

    # Extract physical constants defined in the master config
    U_phys_raw = phys_const.get("inlet_velocity_ms", 0)         # optional — CEDS era only
    # inlet_velocity_ms 在 master_config.yaml 中可能是 list [min, max]
    U_phys = U_phys_raw[0] if isinstance(U_phys_raw, list) and len(U_phys_raw) > 0 else U_phys_raw
    nu_phys = phys_const.get("kinematic_viscosity_air_m2_s", 0)  # optional — CEDS era only
    
    # --- Calculate Conversion Scales ---
    # Velocity Scale: Connects lattice speed to real-world speed (m/s)
    velocity_scale = U_phys / u_lb if u_lb > 1e-9 else 0
    
    # Length Scale (dx): Based on the definition of kinematic viscosity, which scales with velocity and length.
    # nu_phys = velocity_scale * length_scale * nu_lb
    length_scale = nu_phys / (velocity_scale * nu_lb) if (velocity_scale * nu_lb) > 1e-9 else 0
    dx_phys = length_scale  # The size of one lattice cell in meters.
    
    # Time Scale (dt): How many real-world seconds pass in one lattice time step.
    # time_scale = length_scale / velocity_scale
    time_scale = dx_phys / velocity_scale if velocity_scale > 1e-9 else 0
    dt_phys = time_scale

    # --- Calculate Final Physical Values ---
    # Convert the characteristic length from pixels to meters
    L_phys = L_lb * dx_phys
    
    # Re-calculate the Reynolds number using only physical units to verify consistency
    calculated_re = (U_phys * L_phys) / nu_phys if nu_phys > 1e-9 else 0
    
    # Calculate simulation time properties
    steps_per_phys_sec = 1.0 / dt_phys if dt_phys > 1e-9 else 0
    total_time_s = lattice_metadata.get("total_steps_executed", 0) * dt_phys  # optional — system boundary

    return {
        "reynolds_number_target": config.get("outputs", {}).get("target_re"),  # optional — CEDS era only
        "reynolds_number_calculated": calculated_re,
        "characteristic_length_m": L_phys,
        "inlet_velocity_ms": U_phys,
        "kinematic_viscosity_air_m2_s": nu_phys,
        "cell_size_m": dx_phys,
        "time_step_s": dt_phys,
        "steps_per_physical_second": steps_per_phys_sec,
        "total_simulation_time_s": total_time_s,
    }
