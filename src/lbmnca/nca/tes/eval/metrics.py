"""TES 評估指標 — 軌跡 MAE、質量守恆殘差、NS 動量殘差。

下述「兩層評估架構」中的 Layer 1 在**現行 active 路徑上零呼叫**——`compute_all_metrics`
與 `mass_conservation_residual` 全 repo 僅存定義，無任何呼叫端。它已由
`tes/eval/physics_gate.py`（六項時均場物理判準、門檻前置註冊）以更適合本任務的形式
取代。本檔函式仍可作為單元計算使用，但**不要據此宣稱論文有 Layer 1 物理精度評估層**。

兩層評估架構（依方法論文件，歷史）：
    Layer 1 — 物理精度：
        trajectory_mae: 逐步 MAE vs LBM
        mass_conservation_residual: ∂jx/∂x + ∂jy/∂y
        ns_momentum_residual_simplified: ∂j/∂t + (j·∇)j + ∇p

    Layer 2 — 軌跡結構：
        compute_all_metrics: 彙整上列指標（依旗標與 k 值條件性納入，非無條件全含）

★量測區域與分母：上列四個指標函式
（:func:`trajectory_mae`、:func:`mass_conservation_residual`、
:func:`ns_momentum_residual_simplified`、:func:`mass_conservation_residual_flux`）
都先把固體格乘 ``fluid_mask`` 歸零，但**歸零之後的平均分母並非都是 ``n_fluid``**：
前三者與第四者的 ``*_full_*`` 取**全張量平均**，分母＝**攤平後的 ``N×H×W``**
（5D 輸入 ``[B, k, 9, H, W]`` 時 ``N = B×k``；4D 輸入時 ``N = B``；含已歸零的固體格）；
:func:`trajectory_mae` 的 ``mae_per_step`` **另含 9 個通道維**（分母 ``B×9×H×W``／步），
:func:`ns_momentum_residual_simplified` 的 Args 契約只收 4D 且函式體無 ``dim()==5``
分支，故其 ``N`` 就是 ``B``。
四者之中走遮罩平均（分母 ``n_core``、經 :func:`_masked_mean_max`）的只有
:func:`mass_conservation_residual_flux` 回傳 6 個 key 中的 4 個 ``*_core_*``。
故固體面積佔比會系統性影響全張量平均那一組、不影響 ``*_core_*``；
跨 case 比較請優先用 ``*_core_*``。

★依賴備註：本模組 import ``..utils.spatial_ops``，而該模組在 module level
即 ``from ...utils.FieldNormalizer import FieldNormalizer``。故 ``import tes.eval.metrics``
會連帶載入 :class:`~lbmnca.nca.utils.FieldNormalizer.FieldNormalizer`（實測無循環
import）。輕量環境若只想取本模組的純張量指標，仍須滿足該依賴。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.spatial_ops import divergence_flux, interior_core_mask

_RHO_IDX: int = 0
_JX_IDX: int = 3
_JY_IDX: int = 5
_EPS: float = 1e-8


@torch.inference_mode()
def trajectory_mae(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> dict[str, float]:
    """計算預測軌跡與 LBM 軌跡的逐步 MAE。

    Args:
        pred_traj: [B, k, 9, H, W] NCA 預測軌跡。
        target_traj: [B, k, 9, H, W] LBM 目標軌跡。
        fluid_mask: [B, 1, H, W] float，流體=1。

    Returns:
        dict：
            "mae_mean"     — 所有步的平均 MAE（scalar）
            "mae_per_step" — [k] 各步 MAE
            "jx_mae"       — jx 通道平均 MAE
            "jy_mae"       — jy 通道平均 MAE
            "rho_mae"      — rho 通道平均 MAE
    """
    mask = fluid_mask.unsqueeze(1)                       # [B, 1, 1, H, W]
    diff = (pred_traj - target_traj).abs() * mask        # [B, k, 9, H, W]

    mae_per_step: torch.Tensor = diff.mean(dim=(0, 2, 3, 4))   # [k]
    mae_mean: torch.Tensor = mae_per_step.mean()

    jx_mae: torch.Tensor = diff[:, :, _JX_IDX].mean()
    jy_mae: torch.Tensor = diff[:, :, _JY_IDX].mean()
    rho_mae: torch.Tensor = diff[:, :, _RHO_IDX].mean()

    return {
        "mae_mean": mae_mean.item(),
        "mae_per_step": mae_per_step.tolist(),
        "jx_mae": jx_mae.item(),
        "jy_mae": jy_mae.item(),
        "rho_mae": rho_mae.item(),
    }


@torch.inference_mode()
def mass_conservation_residual(
    physics: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> dict[str, float]:
    """計算質量守恆（連續方程）殘差。

    殘差：|∂jx/∂x + ∂jy/∂y|，在**正規化** Mphy 空間計算（不 denorm）。

    ★量測慣例。本 repo 內另有數個 divj 家族實作，各版空間／差分核／遮罩／量綱不同，
    **讀數切勿混用、不可跨版套同一組門檻**：
      - 本函式 — **norm 空間**、Sobel/8 核、完整 ``fluid_mask``、絕對量。
        Sobel 橫向平滑核 ``[1,2,1]`` 使其對 Nyquist 調變**額外**歸零（比純中央差分更鈍）。
      - ``tes.loss.terms.phys_residual.phys_mass_divj_absmean`` — **raw/denorm 空間**、
        ``spatial_ops.ddx/ddy`` 中央差分、完整 ``fluid_mask``、絕對量（此 loss 現況未啟用）。
      - ``tes.eval.physics_gate.divj_over_lbm_floor``（內部工具 ``_divj_norm_on``）—
        六項物理判準之一。**denorm raw Mphy 動量空間**（``_vphy_np`` 的 ``jx``／``jy`` 出自
        ``normalizer.denorm_mphy``，非 ``toVphy``）、``np.gradient``（先對整場算再取區域）、
        相對量，且**再除以同 case GT 的同法讀數**回報倍數；區域有三個變體
        ``divj_over_lbm_floor_{eroded,raw,wall_band}``（``wall_band`` 為
        ``binary_dilation(solid, divj_wall_band_px) & fluid``，只跟量體走、不含域外框）。
      - 本 docstring 引入的**通量式**對照家族 ``tes.utils.spatial_ops.divergence_flux``
        與 :func:`mass_conservation_residual_flux`（見下）。
      以上除通量式家族外**全屬中央差分家族**（頻率響應在 2Δ 波長增益為 0）。對
      grid-scale（≤4 px）結構的敏感度對照算子即通量式家族；該對照只在
      ``interior_core_mask`` 核心區內成立（開放域的域邊界/壁面會給出 O(|j|) 人工項，
      見 :func:`mass_conservation_residual_flux`）。

    Args:
        physics: [B, 9, H, W] 或 [B, k, 9, H, W] 預測物理場。
        fluid_mask: [B, 1, H, W] float。

    Returns:
        dict：
            "mass_residual_mean" — 平均殘差，**全張量平均**（固體格已乘 ``fluid_mask``
                歸零，分母仍為攤平後的 ``N×H×W``、非 ``n_fluid``；5D 輸入時
                ``N = B×k``，4D 時 ``N = B``）
            "mass_residual_max"  — 最大殘差（固體已歸零，故等同流體區最大值）
    """
    if physics.dim() == 5:
        B, K, C, H, W = physics.shape
        physics = physics.reshape(B * K, C, H, W)
        flat_mask = fluid_mask.unsqueeze(1).expand(B, K, 1, H, W).reshape(B * K, 1, H, W)
    else:
        flat_mask = fluid_mask

    jx: torch.Tensor = physics[:, _JX_IDX]   # [B, H, W]
    jy: torch.Tensor = physics[:, _JY_IDX]

    djx_dx = _sobel_x(jx)
    djy_dy = _sobel_y(jy)
    divergence = (djx_dx + djy_dy) * flat_mask.squeeze(1)

    return {
        "mass_residual_mean": divergence.abs().mean().item(),
        "mass_residual_max": divergence.abs().amax().item(),
    }


@torch.inference_mode()
def mass_conservation_residual_flux(
    physics: torch.Tensor,
    fluid_mask: torch.Tensor,
    core_erode_iters: int = 2,
) -> dict[str, float]:
    """質量守恆殘差的 **grid-scale 敏感對照版**。

    與 :func:`mass_conservation_residual` 的差異是散度算子：本函式用
    ``spatial_ops.divergence_flux``（通量式前向/後向差分，核 ``[-1, 1]``，2Δ 波長增益 2，
    且在固/流界面有定義），而非 Sobel/8（中央差分家族，2Δ 波長增益 0）。

    ★**只有 ``*_core_*`` 這一對可以相除解讀為「中央差分對 grid-scale 結構的低估倍率」**。
    ``divergence_flux`` 把域外一律當固體，故在有 inlet/outlet 的開放域上，域邊界與壁面格
    會回報 O(|j|) 的單面通量項——那是邊界處理慣例、不是離散化差異。本函式的
    ``mass_residual_sobel_*`` 用 Sobel/8，其 ``F.conv2d(padding=1)`` 零填補在域邊界
    等效於域外 ``j=0``，與通量式注入同一種單面項，故 Sobel 與通量式在**域邊界／壁面**上
    的讀數不構成有意義的低估倍率對照；本函式因此只在**核心區**回報 Sobel 讀數，核心區外
    的 Sobel 讀數不予回報。

    ★**同一 dict 內有兩種正規化分母，勿混用**：``*_core_*`` 為**遮罩平均**
    （分母 = 核心格數 ``n_core``）；``*_full_*`` 為**全張量平均**（分母 = 攤平後的
    ``N×H×W``，5D 輸入時 ``N = B×k``，含已歸零的固體格）。保留 ``*_full_*`` 這個行為
    是為了與 :func:`mass_conservation_residual` 的 ``mass_residual_mean`` 同慣例、
    兩者可直接對照。``*_full_*`` **不等於**流體遮罩平均——兩者差一個 ``n_fluid/n_total``
    係數，該係數隨建物面積佔比變動。要跨 case 比較請用 ``*_core_*``。

    仍在**正規化** Mphy 空間計算，以便與 :func:`mass_conservation_residual` 對照。
    本函式為新增觀察項，**不取代**任何既有指標；``compute_all_metrics`` 預設不呼叫它。
    本 repo 內尚無任何呼叫端消費其輸出（除測試 ``tests/nca/tes/test_divergence_flux.py``）。

    Args:
        physics: [B, 9, H, W] 或 [B, k, 9, H, W] 預測物理場。
        fluid_mask: [B, 1, H, W] float，流體=1 固體=0。
        core_erode_iters: 核心遮罩腐蝕次數（px），傳給
            ``spatial_ops.interior_core_mask``；``<=0`` 時核心＝全 fluid（此時
            ``*_core_*`` 讀數同樣含邊界人工項，比值失去意義）。

    Returns:
        dict（★三對 key 的分母不同，見上方「兩種正規化分母」段）：
            "mass_residual_flux_core_mean" / "_max" — 通量式散度，核心區**遮罩平均**
                （分母 ``n_core``；本 dict 內唯一可與 ``mass_residual_sobel_core_*``
                相除的讀數）
            "mass_residual_sobel_core_mean" / "_max" — Sobel/8 散度，**同一**核心區、
                同樣為遮罩平均（分母 ``n_core``）
            "mass_residual_flux_full_mean" / "_max" — 通量式散度，**全張量平均**
                （固體格已由 :func:`~tes.utils.spatial_ops.divergence_flux` 歸零，
                分母 ``N×H×W`` 而非 ``n_fluid``；與 :func:`mass_conservation_residual`
                的 ``mass_residual_mean`` 同慣例）。含域邊界與壁面的 O(|j|) 通量項，
                僅供追溯，勿單獨引用，亦不可與 ``*_core_*`` 相除。
        ``*_core_*`` 的 ``_max`` 取核心遮罩內最大值；``*_full_*`` 的 ``_max`` 取全張量
        ``amax``（固體已歸零，故等同全流體最大值）。
        核心遮罩為空（幾何過度阻塞或 ``core_erode_iters`` 過大）時，``*_core_*`` 四個值
        回 ``nan``（不以 0 冒充）。
    """
    if physics.dim() == 5:
        B, K, C, H, W = physics.shape
        physics = physics.reshape(B * K, C, H, W)
        flat_mask = fluid_mask.unsqueeze(1).expand(B, K, 1, H, W).reshape(B * K, 1, H, W)
    else:
        flat_mask = fluid_mask

    jx = physics[:, _JX_IDX]
    jy = physics[:, _JY_IDX]
    fluid_2d = flat_mask.squeeze(1)
    core_2d = interior_core_mask(flat_mask, core_erode_iters).squeeze(1)

    div_flux = divergence_flux(jx, jy, fluid_2d).abs()
    div_sobel = ((_sobel_x(jx) + _sobel_y(jy)) * fluid_2d).abs()

    flux_core_mean, flux_core_max = _masked_mean_max(div_flux, core_2d)
    sobel_core_mean, sobel_core_max = _masked_mean_max(div_sobel, core_2d)

    return {
        "mass_residual_flux_core_mean": flux_core_mean,
        "mass_residual_flux_core_max": flux_core_max,
        "mass_residual_sobel_core_mean": sobel_core_mean,
        "mass_residual_sobel_core_max": sobel_core_max,
        "mass_residual_flux_full_mean": div_flux.mean().item(),
        "mass_residual_flux_full_max": div_flux.amax().item(),
    }


@torch.inference_mode()
def ns_momentum_residual_simplified(
    pred_step1: torch.Tensor,
    ic_physics: torch.Tensor,
    fluid_mask: torch.Tensor,
) -> dict[str, float]:
    """計算簡化 NS 動量方程殘差（k=1 時使用）。

    殘差：∂j/∂t + (j·∇)j + ∇p（忽略黏性項）。

    Args:
        pred_step1: [B, 9, H, W] k=1 預測物理場。
        ic_physics: [B, 9, H, W] 物理初值。
        fluid_mask: [B, 1, H, W] float。

    Returns:
        dict：
            "ns_residual_x_mean" — x 方向殘差均值
            "ns_residual_y_mean" — y 方向殘差均值
            "ns_residual_total"  — 總殘差（|res_x| + |res_y| 均值）
    """
    fm: torch.Tensor = fluid_mask.squeeze(1)

    rho: torch.Tensor = pred_step1[:, _RHO_IDX] + _EPS
    jx: torch.Tensor = pred_step1[:, _JX_IDX]
    jy: torch.Tensor = pred_step1[:, _JY_IDX]
    ux = jx / rho
    uy = jy / rho
    p = rho / 3.0

    djx_dt = jx - ic_physics[:, _JX_IDX]
    djy_dt = jy - ic_physics[:, _JY_IDX]

    adv_x = ux * _sobel_x(jx) + uy * _sobel_y(jx)
    adv_y = ux * _sobel_x(jy) + uy * _sobel_y(jy)

    res_x = (djx_dt + adv_x + _sobel_x(p)) * fm
    res_y = (djy_dt + adv_y + _sobel_y(p)) * fm

    return {
        "ns_residual_x_mean": res_x.abs().mean().item(),
        "ns_residual_y_mean": res_y.abs().mean().item(),
        "ns_residual_total": (res_x.abs() + res_y.abs()).mean().item(),
    }


@torch.inference_mode()
def compute_all_metrics(
    pred_traj: torch.Tensor,
    target_traj: torch.Tensor,
    ic_physics: torch.Tensor,
    fluid_mask: torch.Tensor,
    include_flux_divergence: bool = False,
) -> dict[str, float]:
    """彙整評估指標（★條件性納入，非無條件全含）。

    恆納入 :func:`trajectory_mae` 與 :func:`mass_conservation_residual`；
    :func:`mass_conservation_residual_flux` 僅在 ``include_flux_divergence=True`` 時納入；
    :func:`ns_momentum_residual_simplified` 僅在 ``pred_traj.shape[1] == 1``（k=1）時納入。

    Args:
        pred_traj: [B, k, 9, H, W]。
        target_traj: [B, k, 9, H, W]。
        ic_physics: [B, 9, H, W]。
        fluid_mask: [B, 1, H, W]。
        include_flux_divergence: True → 額外附上
            :func:`mass_conservation_residual_flux` 的六個 key（grid-scale 敏感對照）。
            預設 ``False`` = 與既有輸出逐 key 相同，既有讀數與判準零影響。

    Returns:
        dict（扁平化），key 集合依上述三個條件而異。
    """
    metrics: dict[str, float] = {}

    traj_m = trajectory_mae(pred_traj, target_traj, fluid_mask)
    metrics.update(traj_m)

    mass_m = mass_conservation_residual(pred_traj, fluid_mask)
    metrics.update(mass_m)

    if include_flux_divergence:
        metrics.update(mass_conservation_residual_flux(pred_traj, fluid_mask))

    if pred_traj.shape[1] == 1:
        ns_m = ns_momentum_residual_simplified(
            pred_traj[:, 0], ic_physics, fluid_mask
        )
        metrics.update(ns_m)

    return metrics


# ------------------------------------------------------------------
# 內部工具：遮罩統計
# ------------------------------------------------------------------

def _masked_mean_max(field: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """在遮罩上取 (mean, max)；遮罩全空回 ``(nan, nan)``。

    空遮罩刻意不回 0：0 會被誤讀成「殘差為零」，nan 才能讓下游看出「此讀數不存在」。

    Args:
        field: 任意形狀的非負純量場（呼叫端已取 abs）。
        mask: 與 ``field`` 同形狀的 float 遮罩（選取=1）。

    Returns:
        ``(mean, max)`` 兩個 Python float。
    """
    sel = mask > 0.5
    if not bool(sel.any()):
        return float("nan"), float("nan")
    picked = field[sel]
    return picked.mean().item(), picked.amax().item()


# ------------------------------------------------------------------
# 內部工具：Sobel 梯度（無 buffer，直接計算）
# ------------------------------------------------------------------

def _sobel_x(field: torch.Tensor) -> torch.Tensor:
    """Sobel x 梯度，輸入 [B, H, W]，輸出 [B, H, W]。

    Args:
        field: [B, H, W] 純量場。

    Returns:
        [B, H, W] x 方向梯度估計。
    """
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=field.device, dtype=field.dtype
    ).unsqueeze(0) / 8.0
    return F.conv2d(field.unsqueeze(1), kx, padding=1).squeeze(1)


def _sobel_y(field: torch.Tensor) -> torch.Tensor:
    """Sobel y 梯度，輸入 [B, H, W]，輸出 [B, H, W]。

    Args:
        field: [B, H, W] 純量場。

    Returns:
        [B, H, W] y 方向梯度估計。
    """
    ky = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=field.device, dtype=field.dtype
    ).unsqueeze(0) / 8.0
    return F.conv2d(field.unsqueeze(1), ky, padding=1).squeeze(1)
