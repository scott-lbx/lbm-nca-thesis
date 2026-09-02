"""TES-NCA 評估模組入口 — 匯出指標函式與 rollout 工具。

Layer 1（物理精度）：trajectory_mae、mass_conservation_residual、ns_momentum_residual_simplified
  ★對照觀察項：mass_conservation_residual_flux（通量式散度，對 grid-scale 敏感；
    既有 Sobel/8 版對 2Δ 波長零響應）。★該對照僅在
    ``spatial_ops.interior_core_mask`` 核心區成立——開放域的域邊界/壁面讀數含 O(|j|)
    邊界處理人工項，勿把 ``*_full_*`` 讀數當成 grid-scale 敏感度。
    ★``*_full_*``（全張量平均）與 ``*_core_*``（遮罩平均）分母不同、**不可相除**；
    分母定義見 ``metrics`` 模組 docstring 的「量測區域與分母」段。
Layer 2（軌跡結構）：compute_all_metrics（``include_flux_divergence=False`` 預設，行為不變）
視覺評估：compute_visual_gate_metrics
視覺品質指數：compute_vqi（``vqi`` 子套件）。刻度逐案由該案真值導出，
  ``+1`` 為真值、``−1`` 為零場，不需要事先載入任何錨點檔。
  一次呼叫吃模型的直接輸出，反正規化與頻譜等前處理全在內部完成。
"""
from __future__ import annotations

from .metrics import (
    compute_all_metrics,
    mass_conservation_residual,
    mass_conservation_residual_flux,
    ns_momentum_residual_simplified,
    trajectory_mae,
)
from .visual_metrics import compute_visual_gate_metrics
from .vqi import VQIScore, compute_vqi

__all__ = [
    "trajectory_mae",
    "mass_conservation_residual",
    "mass_conservation_residual_flux",
    "ns_momentum_residual_simplified",
    "compute_all_metrics",
    "compute_visual_gate_metrics",
    "compute_vqi",
    "VQIScore",
]
