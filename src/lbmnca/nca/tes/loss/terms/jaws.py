"""JAWS：Spatially-Adaptive Jacobian Regularization（paper-faithful MAP 版）。

依據 Nie & Suzuki, *JAWS: Enhancing Long-term Rollout of Neural PDE Solvers via
Spatially-Adaptive Jacobian Regularization*（arXiv:2603.05538v2, 2026）§3.3–§3.4 實作。

★與既有 ``temporal_ke.jacobian_amp_loss``（V8/V9）的差異：
  - V8/V9 用 **有限差分** ``‖f(x+ε)−f(x)‖/‖ε‖`` 的放大率 + **啟發式**複雜度權重
    ``w=1/(1+λ_c·|∇f|)``（``_local_complexity``，非可學）。
  - 本檔是 **paper-faithful**：把 operator learning 重構成帶 heteroscedastic
    uncertainty 的 MAP estimation，正規化強度由 **可學的逐點容忍度圖** s1(x)/s2(x)
    自動調制（由輔助 head 輸出，非啟發式），且 ‖J‖²_F 用 **Hutchinson + VJP** 估計。

JAWS 目標函式（論文 Eq. 6，逐點 x 加總）::

    L_JAWS = Σ_x [ ½ e^{−s1} ‖u − û‖²        ← Adaptive Reconstruction
                 + ½ e^{−s2} ‖J(x)‖²_F        ← Adaptive Regularization
                 + ½ (s1 + s2)                 ← Complexity Penalty        ]

  - s1(x)：重建（aleatoric）log-variance。難擬合區 s1↑ → 降本地 loss 權重，不過擬合噪音。
  - s2(x)：穩定性容忍 log-variance。震波/尖梯度 s2↑ → 放鬆 Jacobian 懲罰保高頻；
           平滑區 s2↓ → 強制 ‖J‖→0 真收縮。
  - ½(s1+s2)：complexity penalty，防止 s1,s2→∞ 的 trivial solution。

‖J‖²_F 的 Hutchinson 估計（論文 Eq. 7–8）::

    ‖J‖²_F = Tr(JᵀJ) = E_v[ ‖Jᵀv‖² ],  v ~ Rademacher(±1)
    Jᵀv = ∇_u( Mθ(u) · v )            ← VJP，一次反傳即得

  逐點保留空間結構：``Jᵀv`` 與輸入同 shape ``[B, pd, H, W]``，對 channel 平方和 →
  ``[B, H, W]`` 即逐點 ‖J(x)‖²_F 估計，直接對齊逐點 s2(x)。

接線注意（論文 Eq. 9–10，非本檔職責，由 trainer/step 處理）：
  JAWS 只加在 base transition step（k=1），後續 Pushforward unroll **不** 加 Jacobian
  懲罰（避免 s1/s2 被混沌時間噪音污染），且進 Pushforward 前對 state 做 stop_gradient。

本檔僅提供純函式（接受已算好的張量 / 可微 callable），不存取 config、不持有 model，
與 ``temporal_ke`` 同風格。trainer 端接線範本見本檔末 docstring 註解。
"""

from __future__ import annotations

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# 模組層級常數
# ---------------------------------------------------------------------------

_S_CLAMP: float = 10.0  # log-variance 夾值（防 exp(±s) 溢位；對應論文 log-var 參數化）
_FLUID_EPS: float = 1.0  # fluid 像素數下限（防全固體 batch 除零）


# ---------------------------------------------------------------------------
# Hutchinson 隨機探針
# ---------------------------------------------------------------------------


def rademacher_like(x: Tensor) -> Tensor:
    """產生與 ``x`` 同 shape/dtype/device 的 Rademacher 探針（元素 ±1 等機率）。

    Rademacher 在所有 unbiased trace estimator 中對固定矩陣方差最小（論文 §3.4）。

    Args:
        x: 參考張量（決定 shape/dtype/device）。

    Returns:
        與 ``x`` 同規格的 ±1 張量。
    """
    return torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2.0 - 1.0


def hutchinson_jac_fro_sq(
    f_x: Tensor,  # [B, pd, H, W] 單步輸出 f(x)（須由 x 帶 graph 算出）
    x: Tensor,  # [B, pd, H, W] 輸入葉節點（requires_grad=True）
    n_probes: int = 1,
    *,
    create_graph: bool = True,
) -> Tensor:
    """逐點 ‖J(x)‖²_F 的 Hutchinson + VJP 估計（論文 Eq. 7–8）。

    以 ``Jᵀv = ∇_x( f(x) · v )`` 經 VJP 一次反傳取得（``v`` 為 Rademacher 探針），
    對 channel 維平方和保留空間結構 → 逐點 ``[B, H, W]`` 估計。
    多探針時取平均（降方差；論文指出單探針靠 Adam 跨步隱式平滑即可，預設 1）。

    ``create_graph=True`` 使正規化梯度可回流至模型參數（訓練用）；單元測試取 False。
    估計在 fp32 下做平方歸約以維持數值穩定（VJP 本身沿 ``f_x`` 既有 dtype 之 graph）。

    Args:
        f_x: ``[B, pd, H, W]`` 單步輸出，必須由 ``x`` 經可微 forward 算得。
        x: ``[B, pd, H, W]`` 輸入葉節點，``requires_grad=True`` 且為 ``f_x`` 的 graph 來源。
        n_probes: Rademacher 探針數（≥1）。
        create_graph: 是否建立二階 graph（訓練 True／單測 False）。

    Returns:
        ``[B, H, W]`` 逐點 ‖J(x)‖²_F 估計（fp32，非負）。
    """
    acc = torch.zeros(x.shape[0], *x.shape[2:], device=x.device, dtype=torch.float32)
    for i in range(n_probes):
        v = rademacher_like(f_x)
        # VJP：Jᵀv = ∇_x( <f(x), v> )；retain 供多探針重用同一 forward graph
        (jtv,) = torch.autograd.grad(
            outputs=f_x,
            inputs=x,
            grad_outputs=v,
            retain_graph=(create_graph or i < n_probes - 1),
            create_graph=create_graph,
        )  # [B, pd, H, W]
        acc = acc + jtv.float().square().sum(dim=1)  # [B, H, W]
    return acc / float(n_probes)


# ---------------------------------------------------------------------------
# JAWS MAP 目標（論文 Eq. 6）
# ---------------------------------------------------------------------------


def jaws_map_objective(
    pred: Tensor,  # [B, pd, H, W] 單步輸出 f(x)（norm physics）
    target: Tensor,  # [B, pd, H, W] 目標（norm physics）
    jac_fro_sq: Tensor,  # [B, H, W] 逐點 ‖J(x)‖²_F（hutchinson_jac_fro_sq 輸出）
    s1_map: Tensor,  # [B, 1, H, W] 或 [B, H, W] 重建 log-variance（aux head）
    s2_map: Tensor,  # [B, 1, H, W] 或 [B, H, W] 穩定 log-variance（aux head）
    fluid_mask: Tensor,  # [B, 1, H, W] float，流體=1 固體=0
    *,
    s_clamp: float = _S_CLAMP,
    return_components: bool = False,
):
    """JAWS 三項 MAP 目標（論文 Eq. 6），逐點計算後在 fluid 像素平均。

    逐點::

        recon(x)      = ½ e^{−s1} · Σ_ch (pred − target)²
        reg(x)        = ½ e^{−s2} · ‖J(x)‖²_F
        complexity(x) = ½ (s1 + s2)

    s1/s2 先夾至 ``[−s_clamp, s_clamp]``（log-var 參數化的數值保護）。
    僅在 fluid 像素平均（與 codebase masked-loss 慣例一致），grad 同時回流至
    模型參數（經 pred 與 jac_fro_sq）與 aux head（經 s1/s2）。

    Args:
        pred: ``[B, pd, H, W]`` 單步輸出 f(x)（norm physics）。
        target: ``[B, pd, H, W]`` 目標（norm physics）。
        jac_fro_sq: ``[B, H, W]`` 逐點 ‖J(x)‖²_F。
        s1_map: 重建 log-variance，``[B, 1, H, W]`` 或 ``[B, H, W]``。
        s2_map: 穩定 log-variance，``[B, 1, H, W]`` 或 ``[B, H, W]``。
        fluid_mask: ``[B, 1, H, W]`` float，流體=1 固體=0。
        s_clamp: log-variance 對稱夾值。
        return_components: True 時加回 (recon, reg, complexity) 三項各自的 scalar
            （僅供診斷/log，不進 LossTotal）。

    Returns:
        scalar Tensor（fluid-masked 平均）；若 ``return_components``，回
        ``(total, {"recon", "reg", "complexity"})``。
    """
    m = fluid_mask.squeeze(1).float()  # [B, H, W]
    s1 = s1_map.squeeze(1) if s1_map.dim() == 4 else s1_map
    s2 = s2_map.squeeze(1) if s2_map.dim() == 4 else s2_map
    s1 = s1.clamp(-s_clamp, s_clamp)  # [B, H, W]
    s2 = s2.clamp(-s_clamp, s_clamp)

    sq_err = (pred - target).square().sum(dim=1)  # [B, H, W]（channel L2²）
    recon_px = 0.5 * torch.exp(-s1) * sq_err  # [B, H, W]
    reg_px = 0.5 * torch.exp(-s2) * jac_fro_sq  # [B, H, W]
    comp_px = 0.5 * (s1 + s2)  # [B, H, W]

    denom = m.sum().clamp_min(_FLUID_EPS)

    def _masked_mean(px: Tensor) -> Tensor:
        return (px * m).sum() / denom

    recon = _masked_mean(recon_px)
    reg = _masked_mean(reg_px)
    complexity = _masked_mean(comp_px)
    total = recon + reg + complexity

    if return_components:
        return total, {"recon": recon, "reg": reg, "complexity": complexity}
    return total


# ---------------------------------------------------------------------------
# trainer 端接線範本（僅文件；實作放 training/regularization/regularizer.py）
# ---------------------------------------------------------------------------
#
# 對齊既有 Regularizer._compute_jacobian，但改 VJP + 可學 s1/s2。範例（base step k=1）：
#
#   sd, pd = cfg.model.static_dim, cfg.model.physics_dim
#   phys = state[:, sd:sd + pd].detach().requires_grad_(True)   # x：可微葉節點
#   state_in = torch.cat([state[:, :sd], phys, state[:, sd + pd:]], dim=1) \
#              if cfg.model.hidden_dim > 0 else \
#              torch.cat([state[:, :sd], phys], dim=1)
#   _, traj = model(state_in, cond, mask, 1, return_trajectory=True, **kw)
#   f_x = traj[-1]                                              # [B, pd, H, W]
#   jac = hutchinson_jac_fro_sq(f_x, phys, n_probes=cfg.jaws_n_probes)  # [B, H, W]
#   s1, s2 = model.jaws_head(...)        # aux head 輸出兩張 log-var map（zero-init，見 §7.3）
#   loss = jaws_map_objective(f_x, target_step1, jac, s1, s2, mask)
#
# 接線需另外（本檔不含，依 README §7.4 / §7.3）：
#   1. aux head：在現行主架構 operator_gated 加輕量 conv head 輸出 s1/s2（zero-init）。
#   2. config：jaws_map_enabled / jaws_n_probes / jaws_map_weight 寫進 dataclass + yaml。
#   3. registry：本目標需 model VJP + aux head，**不**走 _LossContext registry；
#      比照 jacobian_amp_loss 由 Regularizer 呼叫（README §7.4 例外，registry 僅限
#      Callable[[_LossContext], Tensor] 純張量 term）。
