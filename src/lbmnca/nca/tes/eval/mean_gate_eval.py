"""mean_module gate 定量評估器。

本模組提供純 metric 函式，由 trainer val dispatch（`_validate_split_mean`）與
`decompose_inference` 共用，涵蓋 mean gate 規格要求的 4 項定量判準：

1. fluid-區 MSE / relative-L2 vs ``mean_flow_post_conv``（`mean_gate_metrics`）
2. SSIM（target = mean，jx 通道，對齊 `mean_flow_loss` 的通道慣例）
3. rollout 收斂曲線：KE 兩窗判準（借 `long_rollout._two_window_ke`，
   solver 式 |Δmean|/mean；mean fixed-point 收斂 → ratio→0）
4. self-rollout recovery：k→2k 續推 MSE drift（≤0 = 自我修正/穩定）

所有輸入皆為 dataset-global norm 空間（呼叫端負責 `_renorm_source_to_global`，
否則 gate 數值在混合 norm 空間失真）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# 包內重用既有實作：_ke_norm / _two_window_ke 為 long_rollout
# 模組級工具，_ssim_loss_2d 為 loss 共用 SSIM kernel（已鎖 jx 通道慣例）。
from ..inference.protocol import (
    NOMINAL_DELTA_CTU_PER_STEP as _NOMINAL_DELTA_CTU_PER_STEP,
)
from ..loss.loss_terms_v2 import JX_CH, _ssim_loss_2d
from .ke_convergence import ke_norm as _ke_norm, two_window_ke as _two_window_ke

__all__ = [
    "ke_curve_convergence",
    "mean_gate_metrics",
    "mean_gate_pool_terms",
    "self_rollout_drift",
]

_EPS: float = 1e-8


def mean_gate_metrics(
    pred_phys: Tensor,
    target_mean: Tensor,
    fluid_mask: Tensor,
) -> dict[str, float]:
    """§6.1 逐 batch 定量指標：fluid MSE、relative-L2、jx-SSIM。

    所有輸入皆須為 dataset-global norm 空間（tanh 正規化）。

    Args:
        pred_phys: 模型預測的正規化 physics 場，shape ``[B, 9, H, W]``。
        target_mean: 目標 mean_flow_post_conv（norm 空間），shape ``[B, 9, H, W]``。
        fluid_mask: 流體遮罩，shape ``[B, 1, H, W]``，float。

    Returns:
        包含以下指標的字典：
        - ``mean_mse``: fluid 像素 9ch MSE。
        - ``mean_rel_l2``: ||pred−tgt||₂/||tgt||₂（fluid 區）。
        - ``mean_jx_ssim``: SSIM 值（越高越好；通道 = JX_CH=3，與 ``mean_flow_loss`` 一致）。
    """
    pred_f = pred_phys.float()
    tgt_f = target_mean.float()
    mask = fluid_mask.float().expand_as(pred_f)

    diff_sq = (pred_f - tgt_f).square() * mask
    n_fluid = mask.sum().clamp_min(1.0)
    mse = diff_sq.sum() / n_fluid

    tgt_sq = tgt_f.square() * mask
    rel_l2 = torch.sqrt(diff_sq.sum()) / torch.sqrt(tgt_sq.sum()).clamp_min(_EPS)

    # tanh-norm 值域 ~[−1,1] → [0,1]（同 mean_flow_loss 慣例）
    jx_p = ((pred_f[:, JX_CH : JX_CH + 1] + 1.0) * 0.5).clamp(0.0, 1.0)
    jx_t = ((tgt_f[:, JX_CH : JX_CH + 1] + 1.0) * 0.5).clamp(0.0, 1.0)
    ssim_val: float = 1.0 - float(_ssim_loss_2d(jx_p, jx_t).item())

    return {
        "mean_mse": float(mse.item()),
        "mean_rel_l2": float(rel_l2.item()),
        "mean_jx_ssim": ssim_val,
    }


def mean_gate_pool_terms(
    pred_phys: torch.Tensor,
    target_mean: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> dict[str, float]:
    """回傳跨 batch **精確池化**所需的原始累加量（與 ``mean_gate_metrics`` 分離）。

    為何需要：``mean_rel_l2`` 是 ``sqrt(Σd²)/sqrt(Σt²)`` 的比值，跨 batch 取算術平均
    （無論加不加權）都不等於整個 split 的 rel_L2；``mean_mse`` 的分母是 ``n_fluid``
    而非 batch size，逐 batch 平均在各 batch 流體格數不同時亦不精確。聚合端可用：

    * split 的 mse    = ``Σ_b sq_err / Σ_b n_fluid``
    * split 的 rel_l2 = ``sqrt(Σ_b sq_err) / sqrt(Σ_b sq_tgt)``

    ★為何不直接塞進 ``mean_gate_metrics`` 的回傳：混進三個 ``_pool_*`` 鍵的話，
    內部累加量會沿著「metrics dict → 聚合 → 論文證據 JSON」這條路徑洩漏成指標
    欄位。分成獨立函式即無此風險，呼叫端只在需要精確池化時多呼叫一次。

    會把整個 dict 寫進證據 JSON 的聚合路徑，寫的是聚合後 ``aggregate_val_metrics``
    的輸出而非本函式的原始累加量，「不得把內部量混進指標 dict」這條約束因此仍然
    成立，只是守備範圍在聚合層而不是呼叫端。鍵集合契約由
    ``tests/nca/tes/test_val_aggregation.py`` 把關。

    Args:
        pred_phys: ``[B, 9, H, W]`` 預測時均場（norm 空間）。
        target_mean: ``[B, 9, H, W]`` 目標時均場（norm 空間）。
        fluid_mask: ``[B, 1, H, W]`` 流體遮罩。

    Returns:
        ``{"sq_err": Σ(pred−tgt)²·mask, "sq_tgt": Σtgt²·mask, "n_fluid": Σmask}``。
        ``n_fluid`` 是**真實**格數、不 clamp（全固體 batch 回 0.0）——與
        ``mean_gate_metrics`` 內部為避免除零而用的 ``clamp_min(1.0)`` 不同。
    """
    pred_f = pred_phys.float()
    tgt_f = target_mean.float()
    mask = fluid_mask.float().expand_as(pred_f)
    return {
        "sq_err": float(((pred_f - tgt_f).square() * mask).sum().item()),
        "sq_tgt": float((tgt_f.square() * mask).sum().item()),
        # ★池化量回**真實** n_fluid，不 clamp。
        #   clamp_min(1.0) 是 metrics 那條為了避免除零而設的，套到累加量上會讓
        #   全固體 batch 貢獻一個幽靈 n_fluid=1.0 進 Σn_fluid 分母——分子是 0、
        #   分母憑空 +1，整個 split 的 mean_mse 被無聲稀釋。真實 batch 不會全固體，
        #   但不能靜默假設不會發生。
        "n_fluid": float(mask.sum().item()),
    }


def ke_curve_convergence(
    ke_curve: list[float] | np.ndarray,
    *,
    ctu_per_step: float = _NOMINAL_DELTA_CTU_PER_STEP,
    window_ctu: float | None = None,
    eps: float = 0.05,
) -> dict[str, float]:
    """rollout KE 曲線收斂判準（§6.1 第 3 項）。

    mean_module 為 fixed-point 任務：KE 曲線應趨平。沿用 solver 式兩窗
    判準 `_two_window_ke`；步軸以 ``ctu_per_step`` 映射為 pseudo-CTU。
    ★該常數的單一真相為 ``tes.inference.protocol.NOMINAL_DELTA_CTU_PER_STEP``
    （舊預設 0.2 已移除）。

    Args:
        ke_curve: 逐步 norm 空間 KE（`_ke_norm` 序列）。
        ctu_per_step: 每 NCA 步對應 CTU（預設＝三態協定常數；``window_ctu=None``
            時窗長取 span/4，此常數會被約分掉，不影響判定）。
        window_ctu: 半窗大小；None → span/4（兩窗恰用曲線後半，
            短 val rollout 也可判）。
        eps: 收斂門檻 |Δmean|/mean。

    Returns:
        包含以下指標的字典：
        - ``ke_converged``: 收斂旗標（0.0 或 1.0）。
        - ``ke_ratio``: 兩窗均相對差。
        - ``ke_final``: 末步 KE 值。
    """
    n = len(ke_curve)
    if n < 4:
        return {
            "ke_converged": 0.0,
            "ke_ratio": float("nan"),
            "ke_final": float(ke_curve[-1]) if n else float("nan"),
        }
    ctu_axis = np.arange(n, dtype=np.float64) * ctu_per_step
    if window_ctu is None:
        window_ctu = float(ctu_axis[-1]) / 4.0
    conv, ratio = _two_window_ke(
        list(ke_curve), list(ctu_axis), window_ctu=window_ctu, eps=eps
    )
    return {
        "ke_converged": conv,
        "ke_ratio": ratio,
        "ke_final": float(ke_curve[-1]),
    }


def self_rollout_drift(mse_at_k: float, mse_at_2k: float) -> dict[str, float]:
    """self-rollout recovery（§6.1 第 4 項）：k→2k 續推 MSE drift。

    drift = mse@2k − mse@k。≤0 → 續推不偏離 mean（自我修正/穩定）；
    >0 → 模型在自身軌道上漂離 fixed-point，gate 應視為風險信號。

    Args:
        mse_at_k: 在 k 步時的 MSE 值。
        mse_at_2k: 在 2k 步時的 MSE 值。

    Returns:
        包含以下指標的字典：
        - ``mean_mse_2k``: 2k 步時的 MSE。
        - ``mean_drift_2k``: mse@2k − mse@k 的漂移量。
    """
    return {
        "mean_mse_2k": float(mse_at_2k),
        "mean_drift_2k": float(mse_at_2k - mse_at_k),
    }


# 供 _ke_norm 的 re-export（trainer/_validate_split_mean 與 decompose_inference
# 逐步累積 KE 曲線時使用，避免直接依賴 long_rollout 私名）。
ke_norm = _ke_norm
