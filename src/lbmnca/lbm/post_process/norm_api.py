"""
TES Zarr 正規化公開 API（tes_v3）。

供 lbmnca.nca DataLoader 使用，提供 snapshot 與 static 的正向／反向轉換。

tes_v3 設計摘要：
  Snapshot channels (9ch MRT):
    forward : out  = tanh((raw - M_REF) / S_combined)
    inverse : raw  = atanh(clip(out, ±TANH_CLIP)) * S_combined + M_REF
    norm_groups 保證 momentum[3,4,5,6] / stress[7,8] 共享 rescale A

  Static channels (4ch):
    C0 SDF      : tanh(SDF_h5s / (L_char_px / H5_DOWNSAMPLE))  — per-case，零點=邊界
    C1 upstream : (upstream - mean) / std                        — 全域 z-score
    C2 pos_x    : raw [0,1], inlet=0, outlet=1                  — 不需反轉
    C3 pos_y    : raw [0,1], bottom=0, top=1                    — 不需反轉

使用方式（lbmnca.nca DataLoader）:
    from lbmnca.lbm.post_process.norm_api import SnapshotNorm, StaticNorm

    # 讀 stats JSON → 建 normalizer
    snap_norm = SnapshotNorm.from_stats(stats_dict)
    stat_norm = StaticNorm.from_stats(stats_dict)

    # forward（zarr 已儲存 forward 結果，DataLoader 直接讀取即可）
    # inverse（推論後還原物理量）
    raw = snap_norm.inverse(model_output)          # [T, 9, H, W] → raw MRT
    sdf_raw = stat_norm.sdf_inverse(sdf_norm, l_char_px)   # → SDF in H5s px
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ── 常數 ──────────────────────────────────────────────────────────────────────
H5_DOWNSAMPLE: int = 3          # ROI(1920×1080) → H5s(640×360)
TANH_CLIP: float = 0.9999       # atanh(±1) = ±inf 防護
NUM_SNAPSHOT_CH: int = 9        # MRT D2Q9 channel 數
NUM_STATIC_CH: int = 4          # SDF, upstream, pos_x, pos_y


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot Normalizer
# ══════════════════════════════════════════════════════════════════════════════


class SnapshotNorm:
    """MRT snapshot 正規化器（tes_v3: tanh + group-shared A）。

    zarr 中 snapshots 已儲存 forward 結果（float16），DataLoader 直接讀取。
    此 class 主要用於：
      1. 推論後 inverse 還原物理量
      2. 診斷用 forward 驗算

    Attributes:
        m_ref: 靜止參考態 (9,) float32。
        s_combined: Chapman-Enskog × group-shared A，(9,) float32。
    """

    def __init__(self, m_ref: np.ndarray, s_combined: np.ndarray) -> None:
        self.m_ref = np.asarray(m_ref, dtype=np.float32)
        self.s_combined = np.asarray(s_combined, dtype=np.float32)

    @classmethod
    def from_stats(cls, stats: dict[str, Any]) -> "SnapshotNorm":
        """從 {dataset}_stats.json dict 建立 normalizer。

        Args:
            stats: json.load 後的 dict，version 需為 "tes_v3"。

        Returns:
            SnapshotNorm instance。
        """
        pn = stats["physics_norm"]
        return cls(
            m_ref=np.array(pn["M_REF"], dtype=np.float32),
            s_combined=np.array(pn["S_combined"], dtype=np.float32),
        )

    def _bc(self, arr: np.ndarray) -> np.ndarray:
        """將 (9,) broadcast 為與 arr 相容的形狀。"""
        if arr.ndim == 3:   # (9, H, W)
            return arr.reshape(NUM_SNAPSHOT_CH, 1, 1)
        if arr.ndim == 4:   # (T, 9, H, W)
            return arr.reshape(1, NUM_SNAPSHOT_CH, 1, 1)
        raise ValueError(f"不支援 ndim={arr.ndim}，預期 3 或 4")

    def forward(self, raw: np.ndarray) -> np.ndarray:
        """raw MRT → tanh normalized ∈ (-1, 1)。

        Args:
            raw: (9, H, W) 或 (T, 9, H, W) float32。

        Returns:
            normalized float32，同形狀。
        """
        s = self._bc(self.s_combined)
        m = self._bc(self.m_ref)
        return np.tanh((raw.astype(np.float32) - m) / s).astype(np.float32)

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        """tanh normalized → raw MRT。

        Args:
            normalized: (9, H, W) 或 (T, 9, H, W)，值域 ∈ (-1, 1)。

        Returns:
            raw MRT float32，同形狀。
        """
        s = self._bc(self.s_combined)
        m = self._bc(self.m_ref)
        clipped = np.clip(normalized.astype(np.float32), -TANH_CLIP, TANH_CLIP)
        return (np.arctanh(clipped) * s + m).astype(np.float32)

    def to_velocity(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """raw MRT → (ux, uy) 速度場。

        CH_JX=3, CH_JY=5。ρ ≈ 1 在 LBM 格子單位。

        Args:
            raw: (..., 9, H, W) float32。

        Returns:
            (ux, uy)，各 (..., H, W) float32。
        """
        rho = np.maximum(raw[..., 0:1, :, :], 1e-6)
        ux = raw[..., 3:4, :, :] / rho
        uy = raw[..., 5:6, :, :] / rho
        return ux.squeeze(-3), uy.squeeze(-3)


# ══════════════════════════════════════════════════════════════════════════════
# Static Normalizer
# ══════════════════════════════════════════════════════════════════════════════


class StaticNorm:
    """Static 特徵 (4ch) 正規化器（tes_v3）。

    各通道策略（儲存在 zarr 已是 normalized）：
      C0 SDF      : tanh(SDF_h5s / L_char_h5s)  per-case
      C1 upstream : z-score (mean, std 儲存在 stats JSON)
      C2 pos_x    : raw [0,1]
      C3 pos_y    : raw [0,1]

    Attributes:
        upstream_mean: float。
        upstream_std: float。
    """

    def __init__(self, upstream_mean: float, upstream_std: float) -> None:
        self.upstream_mean = float(upstream_mean)
        self.upstream_std = float(upstream_std)

    @classmethod
    def from_stats(cls, stats: dict[str, Any]) -> "StaticNorm":
        """從 {dataset}_stats.json dict 建立 normalizer。"""
        sn = stats["static_norm"]
        return cls(
            upstream_mean=float(sn["upstream"]["mean"]),
            upstream_std=float(sn["upstream"]["std"]),
        )

    # ── C0 SDF ────────────────────────────────────────────────────────────────

    @staticmethod
    def sdf_forward(sdf_h5s: np.ndarray, l_char_px: float) -> np.ndarray:
        """SDF (H5s px 單位) → tanh normalized。

        Args:
            sdf_h5s: (H, W) float32，正=流體，負=固體。
            l_char_px: 特徵長度（原始解析度 px，e.g. 150）。

        Returns:
            tanh normalized (H, W) ∈ (-1, 1)，0 仍對應牆面邊界。
        """
        l_char_h5s = l_char_px / H5_DOWNSAMPLE
        return np.tanh(sdf_h5s.astype(np.float32) / l_char_h5s).astype(np.float32)

    @staticmethod
    def sdf_inverse(sdf_norm: np.ndarray, l_char_px: float) -> np.ndarray:
        """tanh normalized SDF → 原始 SDF (H5s px 單位)。

        Args:
            sdf_norm: (H, W) ∈ (-1, 1)。
            l_char_px: 特徵長度（原始解析度 px）。

        Returns:
            SDF in H5s px，正=流體，負=固體。
        """
        l_char_h5s = l_char_px / H5_DOWNSAMPLE
        clipped = np.clip(sdf_norm.astype(np.float32), -TANH_CLIP, TANH_CLIP)
        return (np.arctanh(clipped) * l_char_h5s).astype(np.float32)

    # ── C1 upstream ───────────────────────────────────────────────────────────

    def upstream_forward(self, upstream: np.ndarray) -> np.ndarray:
        """upstream [0,1] → z-score normalized。"""
        return ((upstream.astype(np.float32) - self.upstream_mean) / self.upstream_std).astype(np.float32)

    def upstream_inverse(self, upstream_norm: np.ndarray) -> np.ndarray:
        """z-score normalized → upstream [0,1]。"""
        return (upstream_norm.astype(np.float32) * self.upstream_std + self.upstream_mean).astype(np.float32)

    # ── C2/C3 pos: no-op（raw [0,1] 不需反轉）────────────────────────────────

    @staticmethod
    def pos_forward(pos: np.ndarray) -> np.ndarray:
        """pos [0,1] → 不變（已正規化）。"""
        return pos.astype(np.float32)

    @staticmethod
    def pos_inverse(pos_norm: np.ndarray) -> np.ndarray:
        """pos [0,1] → 不變（已是原始值）。"""
        return pos_norm.astype(np.float32)

    # ── 整批 4ch ──────────────────────────────────────────────────────────────

    def inverse_batch(self, static_norm: np.ndarray, l_char_px: float) -> np.ndarray:
        """全部 4 通道一次反正規化。

        Args:
            static_norm: (4, H, W) float32，zarr 讀取值。
            l_char_px: 特徵長度（原始解析度 px）。

        Returns:
            (4, H, W) float32，各通道還原為物理單位。
        """
        c0 = self.sdf_inverse(static_norm[0], l_char_px)
        c1 = self.upstream_inverse(static_norm[1])
        c2 = self.pos_inverse(static_norm[2])
        c3 = self.pos_inverse(static_norm[3])
        return np.stack([c0, c1, c2, c3], axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# G_vec / P_vec Normalizer（raw → z-score，供 CondNormalizer 使用）
# ══════════════════════════════════════════════════════════════════════════════


class CondNorm:
    """G_vec / P_vec 條件向量正規化器。

    zarr 儲存 raw 值，DataLoader 在 runtime 做 z-score。

    Attributes:
        g_mean: (14,) float32。
        g_std:  (14,) float32。
        p_mean: (3,) float32。
        p_std:  (3,) float32。
    """

    def __init__(
        self,
        g_mean: np.ndarray,
        g_std: np.ndarray,
        p_mean: np.ndarray,
        p_std: np.ndarray,
    ) -> None:
        self.g_mean = np.asarray(g_mean, dtype=np.float32)
        self.g_std  = np.asarray(g_std,  dtype=np.float32)
        self.p_mean = np.asarray(p_mean, dtype=np.float32)
        self.p_std  = np.asarray(p_std,  dtype=np.float32)

    @classmethod
    def from_stats(cls, stats: dict[str, Any]) -> "CondNorm":
        """從 {dataset}_stats.json dict 建立 normalizer。"""
        cs = stats["cond_stats"]
        return cls(
            g_mean=np.array(cs["g_vec"]["mean"], dtype=np.float32),
            g_std =np.array(cs["g_vec"]["std"],  dtype=np.float32),
            p_mean=np.array(cs["p_vec"]["mean"], dtype=np.float32),
            p_std =np.array(cs["p_vec"]["std"],  dtype=np.float32),
        )

    def normalize_gvec(self, g_vec: np.ndarray) -> np.ndarray:
        """raw G_vec (14,) → z-score (14,)。"""
        return ((g_vec.astype(np.float32) - self.g_mean) / self.g_std).astype(np.float32)

    def normalize_pvec(self, p_vec: np.ndarray) -> np.ndarray:
        """raw P_vec (3,) → z-score (3,)。"""
        return ((p_vec.astype(np.float32) - self.p_mean) / self.p_std).astype(np.float32)

    def denormalize_gvec(self, g_norm: np.ndarray) -> np.ndarray:
        """z-score G_vec → raw。"""
        return (g_norm.astype(np.float32) * self.g_std + self.g_mean).astype(np.float32)

    def denormalize_pvec(self, p_norm: np.ndarray) -> np.ndarray:
        """z-score P_vec → raw。"""
        return (p_norm.astype(np.float32) * self.p_std + self.p_mean).astype(np.float32)
