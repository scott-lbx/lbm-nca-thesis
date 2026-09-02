"""
FieldNormalizer — GPU-tensor 正規化器
=====================================
支援三種 MRT 9ch 正規化模式（自動偵測）：

  1. TES-v3 tanh（當前標準，stats 含 "snapshot_norm" + "physics_norm"）：
     norm: ñ = tanh((m - M_REF) / S_combined)
     denorm: m = atanh(clip(ñ, ±0.9999)) × S_combined + M_REF
     is_tanh=True

  2. Physics-informed linear（v2，stats 含 "physics_norm" 但無 "snapshot_norm"）：
     norm: ñ = (m - M_REF) / S_combined
     denorm: m = ñ × S_combined + M_REF
     is_physics_informed=True

  3. Z-score（legacy，stats 含 "mean_field"）：
     norm: ñ = (m - μ) / σ
     denorm: m = ñ × σ + μ

SDF v3: method=tanh_lchar（per-case, threshold=0.0）→ mean=0, std=1 (identity passthrough)
pos_x/pos_y v3: method=raw_01 → mean=0, std=1 (identity passthrough)
所有 buffer 隨 .to(device) 移動，支援全程 GPU 運算。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


# ── helpers ───────────────────────────────────────────────────────────────────


def _safe_std(std: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np.where(np.abs(std) < eps, 1.0, std).astype(np.float32)


def _load_gs(path: Union[str, Path]) -> dict:
    # 未指定 encoding 時 Windows 走 cp950，含非 ASCII 的 stats 檔會直接
    #   UnicodeDecodeError。這類「只在某些檔案上炸」的缺陷會被「常用檔
    #   剛好全 ASCII」長期掩蓋。
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


# ── FieldNormalizer ───────────────────────────────────────────────────────────


class FieldNormalizer(nn.Module):
    """
    GPU-tensor normalizer for all GC-NCA field types.

    自動偵測 physics-informed 或 z-score 模式。
    所有統計量以 nn.Buffer 存儲，隨 .to(device) 自動移動到目標裝置。
    """

    def __init__(
        self,
        mf_mean:   np.ndarray,   # (9,) — z-score μ 或 physics M_REF
        mf_std:    np.ndarray,   # (9,) — z-score σ 或 physics S_combined
        up_mean:   np.ndarray,   # (9,) — u_prime offset (z-score μ 或 zeros)
        up_std:    np.ndarray,   # (9,) — u_prime scale (z-score σ 或 physics S)
        sdf_mean:  float,
        sdf_std:   float,
        px_mean:   float,
        px_std:    float,
        py_mean:   float,
        py_std:    float,
        cond_mean: np.ndarray,   # (N,) cond_field per-dim (7D legacy / 10D new)
        cond_std:  np.ndarray,   # (N,) cond_field per-dim
        is_physics_informed: bool = False,
        is_tanh: bool = False,
        upstream_mean: float = 0.0,
        upstream_std: float = 1.0,
        # 原本這四個參數為具體預設值（0.14 / 0.016 / 0.3333 / 0.01）。
        # 預設值就是**靜默 fallback** —— 違反「絕對禁止帶 fallback 的 config 存取」規則。
        # 數值恰好等於 SSOT 時看不出問題，但顯示範圍改變後，任何未經
        # `from_stats_dict` 的路徑會靜默沿用舊值。
        #
        # 改為 `None`＝未注入：**建構仍允許**（測試不需要顯示範圍），
        # 但**讀取對應 property 時直接 raise**（見下方 `_require_vis`）。
        # 真正的風險是「讀到過期的預設」，不是「建構時沒傳」，故守在讀取端。
        vis_vel_max: Optional[float] = None,
        vis_vor_range: Optional[float] = None,
        vis_p_center: Optional[float] = None,
        vis_p_range: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.is_tanh = is_tanh
        self.is_physics_informed = is_physics_informed

        def _v(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.astype(np.float32))

        self.register_buffer("_mf_mean",   _v(mf_mean))
        self.register_buffer("_mf_std",    _v(_safe_std(mf_std)))
        self.register_buffer("_up_mean",   _v(up_mean))
        self.register_buffer("_up_std",    _v(_safe_std(up_std)))

        self.register_buffer("_sdf_mean",  torch.tensor(float(sdf_mean)))
        self.register_buffer("_sdf_std",   torch.tensor(max(abs(float(sdf_std)), 1e-8)))
        self.register_buffer("_upstream_mean", torch.tensor(float(upstream_mean)))
        self.register_buffer("_upstream_std",  torch.tensor(max(abs(float(upstream_std)), 1e-8)))
        self.register_buffer("_px_mean",   torch.tensor(float(px_mean)))
        self.register_buffer("_px_std",    torch.tensor(max(abs(float(px_std)),  1e-8)))
        self.register_buffer("_py_mean",   torch.tensor(float(py_mean)))
        self.register_buffer("_py_std",    torch.tensor(max(abs(float(py_std)),  1e-8)))

        self.register_buffer("_cond_mean", _v(cond_mean))
        self.register_buffer("_cond_std",  _v(_safe_std(cond_std)))

        # ── vis_range（colormap SSOT，denorm/physical space，與 norm 正交）──
        _nan = float("nan")
        self.register_buffer("_vis_vel_max",
                             torch.tensor(_nan if vis_vel_max is None else float(vis_vel_max)))
        self.register_buffer("_vis_vor_range",
                             torch.tensor(_nan if vis_vor_range is None else float(vis_vor_range)))
        self.register_buffer("_vis_p_center",
                             torch.tensor(_nan if vis_p_center is None else float(vis_p_center)))
        self.register_buffer("_vis_p_range",
                             torch.tensor(_nan if vis_p_range is None else float(vis_p_range)))

    # ── vis_range properties（colormap 固定範圍，三階段共用單一真相）──────────

    @staticmethod
    def _require_vis(value: torch.Tensor, name: str) -> float:
        """讀取顯示範圍；未注入（NaN）即 raise。

        ★ 這裡刻意**不給 fallback**。
        顯示範圍的單一真相是 ``stats.json`` 的 ``vis_range``，
        由 :meth:`from_stats_dict` 注入。拿不到就必須大聲失敗 ——
        靜默沿用一個過期常數會讓所有審查圖與視覺指標在錯誤尺度上作答，
        而且**看不出來**。
        """
        v = float(value)
        if v != v:  # NaN
            raise RuntimeError(
                f"FieldNormalizer.{name}：vis_range 未注入。"
                " 顯示範圍的 SSOT 是 stats.json 的 vis_range，"
                " 請用 FieldNormalizer.from_stats_dict()/from_file() 建構，"
                " ⛔ 不得硬編或使用預設值。"
            )
        return v

    @property
    def vel_vis_max(self) -> float:
        """速度量值 colormap 上界 [0, vel_vis_max]（lattice units）。"""
        return self._require_vis(self._vis_vel_max, "vel_vis_max")

    @property
    def vor_vis_range(self) -> float:
        """渦度 colormap 對稱範圍 [-vor_vis_range, +vor_vis_range]。"""
        return self._require_vis(self._vis_vor_range, "vor_vis_range")

    @property
    def p_vis_center(self) -> float:
        """壓力 colormap 中心（rho/3）。"""
        return self._require_vis(self._vis_p_center, "p_vis_center")

    @property
    def p_vis_range(self) -> float:
        """壓力 colormap 半寬。"""
        return self._require_vis(self._vis_p_range, "p_vis_range")

    # ── factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, stats_path: Union[str, Path]) -> "FieldNormalizer":
        """從 tes_v3 global_stats.json 建立 FieldNormalizer。必須含 snapshot_norm 鍵。"""
        gs = _load_gs(stats_path)
        return cls.from_stats_dict(gs)

    @classmethod
    def from_stats_dict(cls, stats: dict) -> "FieldNormalizer":
        """從已載入記憶體的 stats dict 建立 FieldNormalizer。

        自動偵測格式：
          - tes_v3（snapshot_norm + physics_norm）：tanh 模式
          - GC-NCA legacy（mean_field + u_prime）：z-score 模式
        """
        if "snapshot_norm" not in stats:
            if "mean_field" in stats:
                return cls._from_gc_nca_legacy_stats(stats)
            raise ValueError(
                "FieldNormalizer requires tes_v3 stats (snapshot_norm key missing). "
                f"Found keys: {list(stats.keys())}"
            )

        pn = stats["physics_norm"]
        mf_mean = np.array(pn["M_REF"], dtype=np.float32)
        mf_std  = np.array(pn["S_combined"], dtype=np.float32)
        up_mean = np.zeros(9, dtype=np.float32)
        # u_prime（擾動）尺度。舊 schema 用 Chapman-Enskog 的 `S`；凍結 schema 沒有
        # 這一層 —— `S` 在舊公式裡與 group rescale 相乘後**完全抵消**（同一 group 內
        # `S` 為常數），它從來沒有影響過任何場數值，故凍結時不再輸出。
        # ★ `norm_uprime` / `denorm_uprime` 在 mean-only 範疇下無消費端
        #   （擾動場屬 turb，已出範圍）。此處讓它與狀態同尺度，語義單一。
        up_std  = np.array(pn.get("S", pn["S_combined"]), dtype=np.float32)

        cond_mean = np.zeros(1, dtype=np.float32)
        cond_std  = np.ones(1, dtype=np.float32)

        sn = stats["static_norm"]

        sdf_e = sn["sdf"]
        if sdf_e.get("method") != "tanh_lchar":
            raise ValueError(f"Expected static_norm.sdf.method=tanh_lchar, got: {sdf_e}")
        sdf_mean, sdf_std = 0.0, 1.0

        upstream_mean = float(sn["upstream"]["mean"])
        upstream_std  = float(sn["upstream"]["std"])

        px_e = sn["pos_x"]
        if px_e.get("method") != "raw_01":
            raise ValueError(f"Expected static_norm.pos_x.method=raw_01, got: {px_e}")
        px_mean, px_std = 0.0, 1.0

        py_e = sn["pos_y"]
        if py_e.get("method") != "raw_01":
            raise ValueError(f"Expected static_norm.pos_y.method=raw_01, got: {py_e}")
        py_mean, py_std = 0.0, 1.0

        # vis_range（strict：缺鍵直接 KeyError，禁 fallback）
        vr = stats["vis_range"]

        return cls(
            mf_mean=mf_mean,
            mf_std=mf_std,
            up_mean=up_mean,
            up_std=up_std,
            sdf_mean=sdf_mean,
            sdf_std=sdf_std,
            px_mean=px_mean,
            px_std=px_std,
            py_mean=py_mean,
            py_std=py_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
            is_physics_informed=True,
            is_tanh=True,
            upstream_mean=upstream_mean,
            upstream_std=upstream_std,
            vis_vel_max=float(vr["vel_max"]),
            vis_vor_range=float(vr["vor_range"]),
            vis_p_center=float(vr["p_center"]),
            vis_p_range=float(vr["p_range"]),
        )

    @classmethod
    def _from_gc_nca_legacy_stats(cls, stats: dict) -> "FieldNormalizer":
        """GC-NCA legacy global_stats.json（mean_field + u_prime + ...）→ z-score 模式。"""
        def _arr(key: str, sub: str) -> np.ndarray:
            return np.array(stats[key][sub], dtype=np.float32)

        def _scalar(key: str, sub: str) -> float:
            v = stats[key][sub]
            return float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)

        mf_mean   = _arr("mean_field", "mean")
        mf_std    = _arr("mean_field", "std")
        up_mean   = _arr("u_prime", "mean")
        up_std    = _arr("u_prime", "std")
        sdf_mean  = _scalar("sdf", "mean")
        sdf_std   = max(abs(_scalar("sdf", "std")), 1e-8)
        up_obs_m  = _scalar("upstream_obstacle_sum", "mean")
        up_obs_s  = max(abs(_scalar("upstream_obstacle_sum", "std")), 1e-8)
        px_mean   = _scalar("pos_x", "mean")
        px_std    = max(abs(_scalar("pos_x", "std")), 1e-8)
        py_mean   = _scalar("pos_y", "mean")
        py_std    = max(abs(_scalar("pos_y", "std")), 1e-8)
        cond_mean = _arr("cond_field", "mean")
        cond_std  = _arr("cond_field", "std")

        return cls(
            mf_mean=mf_mean,
            mf_std=mf_std,
            up_mean=up_mean,
            up_std=up_std,
            sdf_mean=sdf_mean,
            sdf_std=sdf_std,
            px_mean=px_mean,
            px_std=px_std,
            py_mean=py_mean,
            py_std=py_std,
            cond_mean=cond_mean,
            cond_std=cond_std,
            is_physics_informed=False,
            is_tanh=False,
            upstream_mean=up_obs_m,
            upstream_std=up_obs_s,
        )

    # ── 內部 broadcast ────────────────────────────────────────────────────

    @staticmethod
    def _norm_ch(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Per-channel z-score. x: [C,H,W] or [B,C,H,W]."""
        # 確保裝置一致
        mean = mean.to(x.device)
        std = std.to(x.device)
        if x.ndim == 3:
            return (x - mean[:, None, None]) / std[:, None, None]
        return (x - mean[None, :, None, None]) / std[None, :, None, None]

    @staticmethod
    def _denorm_ch(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        # 確保裝置一致
        mean = mean.to(x.device)
        std = std.to(x.device)
        if x.ndim == 3:
            return x * std[:, None, None] + mean[:, None, None]
        return x * std[None, :, None, None] + mean[None, :, None, None]

    # ── mean_field  (Mphy_lattice_norm ↔ Mphy_lattice_raw) ───────────────

    def norm_mphy(self, x: torch.Tensor) -> torch.Tensor:
        """[B,9,H,W] or [9,H,W]  raw → norm"""
        if self.is_tanh:
            m = self._mf_mean.to(x.device)
            s = self._mf_std.to(x.device)
            if x.ndim == 3:
                return torch.tanh((x - m[:, None, None]) / s[:, None, None])
            return torch.tanh((x - m[None, :, None, None]) / s[None, :, None, None])
        return self._norm_ch(x, self._mf_mean, self._mf_std)

    def denorm_mphy(self, x: torch.Tensor) -> torch.Tensor:
        """[B,9,H,W] or [9,H,W]  norm → raw（tanh 模式為 atanh 反變換）。

        ★梯度病態警告：`is_tanh` 時本函式為
        `atanh(clamp(x, ±0.9999))·s + m`，其導數 `1/(1−x²)` 在 |x|→1 時發散，
        而 clamp 使 |x|>0.9999 的區域導數**恆為 0**。後果是：任何建立在
        raw-space 上的物理殘差軟約束（散度、動量殘差）在**場飽和時失去梯度**——
        亦即最需要約束的失效模式（能量泵浦、場爆掉）恰恰是它管不到的。
        引用「以 raw-space 物理殘差施加軟約束」時必須帶此限定。
        （canonical 的散度懲罰權重為 0、該路徑未啟用，故此缺陷目前無實際影響。）
        """
        if self.is_tanh:
            m = self._mf_mean.to(x.device)
            s = self._mf_std.to(x.device)
            xc = x.clamp(-0.9999, 0.9999)
            if x.ndim == 3:
                return torch.atanh(xc) * s[:, None, None] + m[:, None, None]
            return torch.atanh(xc) * s[None, :, None, None] + m[None, :, None, None]
        return self._denorm_ch(x, self._mf_mean, self._mf_std)

    # ── u_prime ───────────────────────────────────────────────────────────

    def norm_uprime(self, x: torch.Tensor) -> torch.Tensor:
        """[B,9,H,W] or [9,H,W]  raw → norm"""
        return self._norm_ch(x, self._up_mean, self._up_std)

    def denorm_uprime(self, x: torch.Tensor) -> torch.Tensor:
        """[B,9,H,W] or [9,H,W]  norm → raw"""
        return self._denorm_ch(x, self._up_mean, self._up_std)

    # ── SDF ───────────────────────────────────────────────────────────────

    def norm_sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Any shape  raw → norm"""
        return (x - self._sdf_mean.to(x.device)) / self._sdf_std.to(x.device)

    def denorm_sdf(self, x: torch.Tensor) -> torch.Tensor:
        """Any shape  norm → raw"""
        return x * self._sdf_std.to(x.device) + self._sdf_mean.to(x.device)

    def norm_upstream(self, x: torch.Tensor) -> torch.Tensor:
        """Any shape raw upstream mask/distance proxy → norm"""
        return (x - self._upstream_mean.to(x.device)) / self._upstream_std.to(x.device)

    def denorm_upstream(self, x: torch.Tensor) -> torch.Tensor:
        """Any shape norm upstream mask/distance proxy → raw"""
        return x * self._upstream_std.to(x.device) + self._upstream_mean.to(x.device)

    # ── Position embeddings ───────────────────────────────────────────────

    def norm_pos_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._px_mean.to(x.device)) / self._px_std.to(x.device)

    def denorm_pos_x(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._px_std.to(x.device) + self._px_mean.to(x.device)

    def norm_pos_y(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._py_mean.to(x.device)) / self._py_std.to(x.device)

    def denorm_pos_y(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._py_std.to(x.device) + self._py_mean.to(x.device)

    # ── Conditioning vector (N-dim: 7D legacy / 10D new) ────────────────

    def norm_cond(self, x: torch.Tensor) -> torch.Tensor:
        """[B,N] or [N]  raw → norm"""
        mean = self._cond_mean.to(x.device)
        std = self._cond_std.to(x.device)
        if x.ndim == 1:
            return (x - mean) / std
        return (x - mean[None, :]) / std[None, :]

    def denorm_cond(self, x: torch.Tensor) -> torch.Tensor:
        """[B,N] or [N]  norm → raw"""
        mean = self._cond_mean.to(x.device)
        std = self._cond_std.to(x.device)
        if x.ndim == 1:
            return x * std + mean
        return x * std[None, :] + mean[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """預設 forward 等同 norm_mphy。"""
        return self.norm_mphy(x)
