"""POD-based 結構指標模組：計算 TES 速度場的 Proper Orthogonal Decomposition 相關損失與評估指標。

提供 POD basis 的擬合、存取、係數計算，以及可微分 POD 損失函式（``pod_coeff_loss``）
和評估用指標計算（``compute_pod_metrics``）。所有速度場輸入皆需先反正規化再轉換為
固定範圍 [0,1] 的速度大小。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...utils.FieldNormalizer import FieldNormalizer

RHO_CH: int = 0
JX_CH: int = 3
JY_CH: int = 5
RHO_EPS: float = 1e-6
# vel_max = stats.json vis_range SSOT，經 normalizer.vel_vis_max 取得（見 denorm_vel_mag_01）


def denorm_vel_mag_01(
    x_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
) -> Tensor:
    """反正規化後計算固定範圍遮蔽速度大小，輸出 shape ``[B, H, W]``。

    Args:
        x_norm: 正規化的 physics 場，shape ``[B, 9, H, W]``。
        fluid_mask: 流體遮罩，shape ``[B, 1, H, W]``（float）。
        normalizer: 正規化/反正規化橋接器。

    Returns:
        clamp 至 [0, 1] 並乘上 fluid_mask 的速度大小張量，shape ``[B, H, W]``。
    """
    raw = normalizer.denorm_mphy(x_norm.float())
    rho = raw[:, RHO_CH].clamp_min(RHO_EPS)
    vel_mag = (
        (raw[:, JX_CH] / rho).square()
        + (raw[:, JY_CH] / rho).square()
        + 1e-8
    ).sqrt()
    fm = fluid_mask.squeeze(1).float()
    return (vel_mag / normalizer.vel_vis_max).clamp(0.0, 1.0) * fm


def fit_pod_basis(
    fields_01: Tensor,
    n_modes: int = 16,
) -> dict[str, np.ndarray]:
    """從訓練目標場擬合 POD basis。

    使用 SVD 分解中心化場矩陣，取前 ``n_modes`` 個右奇異向量作為 POD 模態。

    Args:
        fields_01: 速度大小場序列，shape ``[N, H, W]``，值域 [0, 1]。
        n_modes: 要保留的 POD 模態數。

    Returns:
        包含以下鍵的字典：
        - ``mean``: 均值場，shape ``[H, W]``，float32。
        - ``modes``: POD 模態，shape ``[n_modes, H, W]``，float32。
        - ``singular_values``: 奇異值陣列，float32。
        - ``explained_variance``: 各模態解釋變異比例，float32。
        - ``n_samples``: 樣本數，int32。

    Raises:
        ValueError: 若 fields_01 維度不為 3，或樣本數少於 2。
    """
    if fields_01.dim() != 3:
        raise ValueError(f"fields_01 must be [N,H,W], got {tuple(fields_01.shape)}")
    x = fields_01.detach().cpu().float()
    n, h, w = x.shape
    if n < 2:
        raise ValueError("Need at least 2 fields to fit POD basis")
    mat = x.reshape(n, h * w)
    mean = mat.mean(dim=0, keepdim=True)
    centered = mat - mean
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    modes = vh[: min(n_modes, vh.shape[0])]
    var = s.square()
    explained = var / var.sum().clamp_min(1e-12)
    return {
        "mean": mean.reshape(h, w).numpy().astype("float32"),
        "modes": modes.reshape(modes.shape[0], h, w).numpy().astype("float32"),
        "singular_values": s.cpu().numpy().astype("float32"),
        "explained_variance": explained.cpu().numpy().astype("float32"),
        "n_samples": np.asarray([n], dtype=np.int32),
    }


def save_pod_basis(
    path: str | Path,
    basis: dict[str, np.ndarray],
    meta: dict[str, Any] | None = None,
) -> None:
    """將 POD basis 壓縮儲存至 .npz 檔案。

    Args:
        path: 目標 .npz 路徑；父目錄不存在時自動建立。
        basis: 由 ``fit_pod_basis`` 回傳的 basis 字典。
        meta: 附加 metadata（鍵值對），以 ``meta_`` 為前綴存入 .npz。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(basis)
    if meta:
        for key, value in meta.items():
            payload[f"meta_{key}"] = np.asarray([str(value)])
    np.savez_compressed(path, **payload)


def load_pod_basis(
    path: str | Path,
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """從 .npz 檔案載入 POD basis 並轉為 PyTorch Tensor。

    Args:
        path: 由 ``save_pod_basis`` 儲存的 .npz 路徑。
        device: 目標裝置；None 表示不移動（保留 CPU）。

    Returns:
        包含 ``mean``、``modes``、``explained_variance``、``singular_values``
        的 float32 Tensor 字典。
    """
    data = np.load(Path(path), allow_pickle=False)
    dev = torch.device(device) if device is not None else None
    return {
        "mean": torch.from_numpy(data["mean"]).to(dev).float(),
        "modes": torch.from_numpy(data["modes"]).to(dev).float(),
        "explained_variance": torch.from_numpy(data["explained_variance"]).to(dev).float(),
        "singular_values": torch.from_numpy(data["singular_values"]).to(dev).float(),
    }


def _coeffs(
    field_01: Tensor,
    basis: dict[str, Tensor],
    n_modes: int | None = None,
) -> Tensor:
    """計算場相對於 POD basis 的係數向量。

    Args:
        field_01: 速度大小場，shape ``[B, H, W]``，值域 [0, 1]。
        basis: 由 ``load_pod_basis`` 載入的 basis 字典。
        n_modes: 使用前 n 個模態；None 表示使用全部。

    Returns:
        POD 係數矩陣，shape ``[B, n_modes]``。
    """
    b, h, w = field_01.shape
    mean = basis["mean"].to(field_01.device, dtype=field_01.dtype).reshape(1, h * w)
    modes = basis["modes"].to(field_01.device, dtype=field_01.dtype)
    if n_modes is not None:
        modes = modes[:n_modes]
    modes_flat = modes.reshape(modes.shape[0], h * w)
    centered = field_01.reshape(b, h * w) - mean
    return centered @ modes_flat.t()


@torch.inference_mode()
def compute_pod_metrics(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    basis: dict[str, Tensor],
    n_modes: int | None = None,
) -> dict[str, float]:
    """計算預測場與目標場之間的 POD 結構指標（推論模式，無梯度）。

    Args:
        pred_norm: 模型預測的正規化 physics 場，shape ``[B, 9, H, W]``。
        target_norm: 目標正規化 physics 場，shape ``[B, 9, H, W]``。
        fluid_mask: 流體遮罩，shape ``[B, 1, H, W]``。
        normalizer: 反正規化橋接器。
        basis: 已載入的 POD basis 字典。
        n_modes: 限制使用的模態數；None 表示使用全部。

    Returns:
        包含以下指標的字典：
        - ``PODCoeff_MAE``: POD 係數平均絕對誤差。
        - ``PODCoeff_RelL2``: POD 係數相對 L2 誤差。
        - ``PODCoeff_R2``: POD 係數 R² 值。
        - ``PODCoeff_Cosine``: POD 係數餘弦相似度均值。
        - ``PODRecon_MAE``: POD 重建場 MAE（fluid 區）。
    """
    pred_f = denorm_vel_mag_01(pred_norm, fluid_mask, normalizer)
    tgt_f = denorm_vel_mag_01(target_norm, fluid_mask, normalizer)
    cp = _coeffs(pred_f, basis, n_modes)
    ct = _coeffs(tgt_f, basis, n_modes)
    diff = cp - ct
    denom = ct.square().sum().clamp_min(1e-8)
    rel_l2 = diff.square().sum().sqrt() / denom.sqrt()
    r2 = 1.0 - diff.square().sum() / denom
    cosine = F.cosine_similarity(cp, ct, dim=1, eps=1e-8).mean()

    modes = basis["modes"].to(pred_f.device, dtype=pred_f.dtype)
    if n_modes is not None:
        modes = modes[:n_modes]
    b, h, w = pred_f.shape
    mean = basis["mean"].to(pred_f.device, dtype=pred_f.dtype).reshape(1, h * w)
    modes_flat = modes.reshape(modes.shape[0], h * w)
    rp = (mean + cp @ modes_flat).reshape(b, h, w)
    rt = (mean + ct @ modes_flat).reshape(b, h, w)
    fm = fluid_mask.squeeze(1).float()
    recon_mae = ((rp - rt).abs() * fm).sum() / fm.sum().clamp_min(1.0)

    return {
        "PODCoeff_MAE": float(diff.abs().mean().item()),
        "PODCoeff_RelL2": float(rel_l2.item()),
        "PODCoeff_R2": float(r2.item()),
        "PODCoeff_Cosine": float(cosine.item()),
        "PODRecon_MAE": float(recon_mae.item()),
    }


def pod_coeff_loss(
    pred_norm: Tensor,
    target_norm: Tensor,
    fluid_mask: Tensor,
    normalizer: FieldNormalizer,
    basis: dict[str, Tensor],
    n_modes: int | None = None,
) -> Tensor:
    """可微分的最終步 POD 係數損失。

    僅取 trajectory 最後一步（``pred_norm[:, -1]``）計算 POD 係數 L1 差，
    並以目標係數尺度正規化，使損失權重在不同 POD basis 之間具有可比性。

    Args:
        pred_norm: 模型預測軌跡，shape ``[B, T, 9, H, W]``。
        target_norm: 目標軌跡，shape ``[B, T, 9, H, W]``。
        fluid_mask: 流體遮罩，shape ``[B, 1, H, W]``。
        normalizer: 反正規化橋接器。
        basis: 已載入的 POD basis 字典。
        n_modes: 限制使用的模態數；None 表示使用全部。

    Returns:
        標量損失 Tensor（已正規化 L1 POD 係數誤差）。
    """
    pred_f = denorm_vel_mag_01(pred_norm[:, -1].float(), fluid_mask, normalizer)
    tgt_f = denorm_vel_mag_01(target_norm[:, -1].float(), fluid_mask, normalizer)
    cp = _coeffs(pred_f, basis, n_modes)
    ct = _coeffs(tgt_f, basis, n_modes)
    scale = ct.detach().abs().mean().clamp_min(1e-6)
    return (cp - ct).abs().mean() / scale
