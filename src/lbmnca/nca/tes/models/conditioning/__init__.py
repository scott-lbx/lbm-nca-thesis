"""conditioning 子模組：條件編碼器集合（FiLM 調製參數生成）。

三種編碼器依用途分工：
  TESConditionEncoder:   物理條件向量（p_vec + dt）→ FiLM (γ_p, β_p)
  GVecEncoder:           幾何先驗向量（G_vec 14D）→ FiLM (γ_g, β_g)，per-rollout 計算一次
  MeanFlowEncoder:       mean_flow 場 [B,9,H,W] → 空間 FiLM (γ_m, β_m)，turb_module 用
  ScaleConditionEncoder: 尺度條件特徵（H-series）→ 加法式 FiLM（選用）
  build_scale_features:  建構 SCALE_FEAT_DIM=4 尺度特徵向量工具函式
"""
from __future__ import annotations

from .cond_encoder import TESConditionEncoder
from .gvec_encoder import GVecEncoder
from .mean_flow_encoder import MeanFlowEncoder
from .scale_encoder import SCALE_FEAT_DIM, ScaleConditionEncoder, build_scale_features

__all__ = [
    "TESConditionEncoder",
    "GVecEncoder",
    "MeanFlowEncoder",
    "ScaleConditionEncoder",
    "build_scale_features",
    "SCALE_FEAT_DIM",
]
