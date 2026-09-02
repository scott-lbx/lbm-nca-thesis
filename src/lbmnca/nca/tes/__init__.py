"""TES-NCA 根模組 — Transient Evolution Surrogate Neural Cellular Automata。

學習 LBM 動力算子 T: (f_t, G, P) → f_{t+1}，以 NCA 代理瞬態流場演進。

狀態向量 [B, 13, H, W]：
    C0-C3  : static  — SDF_norm, upstream_norm, X_norm, Y_norm
    C4-C12 : physics — 9-ch MRT moments (Mphy_lattice_norm)

    無 hidden state（純 Markov），與 CEDS-NCA 的 29ch 狀態不同。

條件向量 cond_17D [B, 17]（正規化後）：
    [0:14]  : G_vec_B_norm — 14D 幾何條件
    [14:17] : P_vec_B_norm — 3D 物理條件 (Re, U_inlet, L_char)
"""
from __future__ import annotations
