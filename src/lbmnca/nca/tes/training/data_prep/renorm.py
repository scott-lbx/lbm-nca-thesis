"""Source-local 至 dataset-global 正規化空間轉換子模組。

本模組提供 DataRenormalizer，負責將多 zarr source 的 local-normalized 張量
轉至 runtime dataset-global 正規化空間，從 TESTrainer._renorm_source_to_global、
_renorm_ic_mphy、_renorm_ic_mphy_traj 提取。
"""

from __future__ import annotations

import torch

from ..context import TrainerContext


class DataRenormalizer:
    """Source-local 至 dataset-global 正規化空間轉換器。

    多 zarr local stats 不一致時必須逐 source_stats_idx 反正規化回 raw
    再以 global normalizer 重新正規化；單 dataset（無 source_field_normalizers
    或無 source_stats_idx）為 no-op 直接回傳原張量。

    Args:
        ctx: TrainerContext 共享狀態容器。
    """

    def __init__(self, ctx: TrainerContext) -> None:
        """初始化 DataRenormalizer。

        Args:
            ctx: TrainerContext 共享狀態容器。
        """
        self.ctx = ctx

    def _get_field_norm(self) -> object:
        """延遲載入 FieldNormalizer（首次呼叫時從 stats_path 建立並快取至 ctx）。

        Returns:
            FieldNormalizer 實例。
        """
        if self.ctx.field_normalizer is None:
            from ....utils.FieldNormalizer import FieldNormalizer

            self.ctx.field_normalizer = FieldNormalizer.from_file(
                self.ctx.cfg.stats_path
            )
        return self.ctx.field_normalizer

    def renorm_source_to_global(
        self,
        static: torch.Tensor,
        ic_physics: torch.Tensor,
        target_traj: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """將 source-local 正規化張量轉至 runtime dataset-global 正規化空間。

        多 zarr local stats 不一致時必須（逐 source_stats_idx 反正規化回 raw 再以
        global normalizer 重新正規化）；單 dataset（無 source_field_normalizers 或
        無 source_stats_idx）為 no-op 直接回傳原張量。

        Args:
            static: 靜態通道張量 [B, 4, H, W]。
            ic_physics: IC physics 張量 [B, 9, H, W]。
            target_traj: 目標軌跡張量 [B, K, 9, H, W]。
            batch: DataLoader batch dict（讀取 source_stats_idx）。

        Returns:
            (static, ic_physics, target_traj) 轉至 global norm 空間後的版本。
        """
        ctx = self.ctx
        if not ctx.source_field_normalizers:
            return static, ic_physics, target_traj
        source_idx = batch.get("source_stats_idx")
        if source_idx is None:
            return static, ic_physics, target_traj

        global_norm = self._get_field_norm()
        source_idx = source_idx.to(ctx.device, non_blocking=True)
        B, K, C, H, W = target_traj.shape
        static_out = static.clone()
        ic_out = ic_physics.clone()
        target_out = target_traj.clone()

        for idx in torch.unique(source_idx).tolist():
            idx_int = int(idx)
            src_norm = ctx.source_field_normalizers[idx_int]
            mask = source_idx == idx_int
            if not bool(mask.any()):
                continue

            ic_out[mask] = global_norm.norm_mphy(src_norm.denorm_mphy(ic_physics[mask]))

            tgt_src = target_traj[mask].reshape(-1, C, H, W)
            tgt_raw = src_norm.denorm_mphy(tgt_src)
            target_out[mask] = global_norm.norm_mphy(tgt_raw).reshape(-1, K, C, H, W)

            st = static[mask]
            st_out = st.clone()
            st_out[:, 0] = global_norm.norm_sdf(src_norm.denorm_sdf(st[:, 0]))
            st_out[:, 1] = global_norm.norm_upstream(src_norm.denorm_upstream(st[:, 1]))
            st_out[:, 2] = global_norm.norm_pos_x(src_norm.denorm_pos_x(st[:, 2]))
            st_out[:, 3] = global_norm.norm_pos_y(src_norm.denorm_pos_y(st[:, 3]))
            static_out[mask] = st_out

        return static_out, ic_out, target_out

    def renorm_ic_mphy(
        self,
        ic_raw: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """將單一 Mphy IC 幀 renorm 至 dataset-global 正規化空間。

        供 RP delta-loss 參考幀（t_start + r_rp）使用——該幀不屬於
        target_traj 但須共用 global norm 策略。

        Args:
            ic_raw: 單幀 Mphy 張量 [B, 9, H, W]（source-local norm）。
            batch: DataLoader batch dict（讀取 source_stats_idx）。

        Returns:
            轉至 global norm 空間後的單幀張量。
        """
        ctx = self.ctx
        if not ctx.source_field_normalizers:
            return ic_raw
        source_idx = batch.get("source_stats_idx")
        if source_idx is None:
            return ic_raw
        global_norm = self._get_field_norm()
        source_idx = source_idx.to(ctx.device, non_blocking=True)
        ic_out = ic_raw.clone()
        for idx in torch.unique(source_idx).tolist():
            idx_int = int(idx)
            src_norm = ctx.source_field_normalizers[idx_int]
            mask = source_idx == idx_int
            if not bool(mask.any()):
                continue
            ic_out[mask] = global_norm.norm_mphy(src_norm.denorm_mphy(ic_raw[mask]))
        return ic_out

    def renorm_ic_mphy_traj(
        self,
        traj: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """對多幀 trajectory [B, K, 9, H, W] 套用 source→global renorm（複用單幀路徑）。

        多 source 時逐幀 renorm；單 source（無 source_field_normalizers）為 identity。

        Args:
            traj: 多幀 Mphy 軌跡張量 [B, K, 9, H, W]（source-local norm）。
            batch: DataLoader batch dict（讀取 source_stats_idx）。

        Returns:
            轉至 global norm 空間後的多幀軌跡張量 [B, K, 9, H, W]。
        """
        if not self.ctx.source_field_normalizers:
            return traj
        B, K, C, H, W = traj.shape
        out = traj.clone()
        for kk in range(K):
            out[:, kk] = self.renorm_ic_mphy(traj[:, kk], batch)
        return out
