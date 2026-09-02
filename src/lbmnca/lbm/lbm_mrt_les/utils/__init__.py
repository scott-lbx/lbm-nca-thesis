# Expose all functions to maintain backward compatibility
# while allowing for modular file structure.

from .config_utils import load_config, get_zone_config, save_case_metadata
from .physics_utils import (
    print_reynolds_info,
    # calculate_characteristic_length 已於 2026-07-26 移除（語義與 canonical L_char
    # 不同的死碼）。L_char 請用 physics_proxy.lchar_api。
    calculate_simulation_time_scale,
    get_simulation_strategy,
    compute_coefficients,
    fit_sine_wave,
)
from ..visualization.viz_utils import (
    plot_mask,
    calcu_gui_size,
    apply_resize,
    draw_zone_overlay,
)
from .mask_utils import create_mask
from .system_utils import force_clean_cache, get_random_png_path
