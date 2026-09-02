"""模擬產物的驗證層。

`case_physics` 是每個 case 跑完後的**唯一收錄閘門**——PASS 才進 dataset，
FAIL 一律作廢重生成（2026-07-29 使用者裁決）。
"""

from .case_physics import Check, Report, validate_h5

__all__ = ["Check", "Report", "validate_h5"]
