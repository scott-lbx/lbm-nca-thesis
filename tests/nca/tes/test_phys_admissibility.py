"""物理可容許性投影的構造性保證與恆等性。

投影存在的理由見 `src/tes/models/transport/admissibility.py` 的模組 docstring：
state 存在 tanh 正規化的像裡而更新是無界加法，密度通道無任何護欄。

本檔驗兩件事，兩者都是「宣稱」而非「觀察」，故寫成斷言：

  1. **構造性**：任意極端輸入（含已漂出 tanh 像的 state）投影後，
     流體區的 ρ 恆為正、|u| 恆不超過 mach_max·c_s。
  2. **恆等性**：未違反約束的輸入逐位不變。這是把投影掛上既有 ckpt
     而不動搖既有讀數的前提。
"""
from __future__ import annotations

import math

import pytest
import torch

from lbmnca.nca.tes.config import TESModelConfig
from lbmnca.nca.tes.models.transport.admissibility import (
    LATTICE_SOUND_SPEED, PhysicsAdmissibility,
)

# tanh v3 正規化常數（四份 dataset stats 實測一致，見 physics_norm.M_REF / S_combined）
M_REF = [1.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
S_COMB = [0.99999996, 1.99999993, 1.99999993, 0.13427734, 0.13427734,
          0.13427734, 0.13427734, 0.01748657, 0.01748657]
RHO_MIN, RHO_MAX, MACH_MAX = 0.02, 2.24, 0.95
_RHO, _JX, _JY = 0, 3, 5


def _cfg(enabled: bool = True) -> TESModelConfig:
    return TESModelConfig(
        state_dim=21, static_dim=4, physics_dim=9, hidden_dim=8, cond_dim=17,
        phys_adm_enabled=enabled, phys_adm_rho_min=RHO_MIN, phys_adm_rho_max=RHO_MAX,
        phys_adm_mach_max=MACH_MAX, phys_adm_m_ref=M_REF, phys_adm_s_combined=S_COMB,
    )


def _denorm(n: torch.Tensor, ch: int) -> torch.Tensor:
    return torch.atanh(n.clamp(-0.9999, 0.9999)) * S_COMB[ch] + M_REF[ch]


def _macro(phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rho = _denorm(phys[:, _RHO], _RHO)
    jx, jy = _denorm(phys[:, _JX], _JX), _denorm(phys[:, _JY], _JY)
    return rho, torch.sqrt(jx ** 2 + jy ** 2) / rho


@pytest.fixture()
def fluid_mask() -> torch.Tensor:
    fm = torch.ones(2, 1, 24, 32)
    fm[:, :, :4, :] = 0.0        # 固體帶：投影不得依賴它，也不得在該處產生非零值
    return fm


def test_positivity_and_mach_under_extreme_state(fluid_mask: torch.Tensor) -> None:
    """任意極端 state（含漂出 tanh 像者）投影後仍在可容許集合內。"""
    torch.manual_seed(0)
    adm = PhysicsAdmissibility(_cfg())
    # 值域刻意跨出 (-1,1)：這正是實測發散時 state 所在的位置（ρ 讀出 clip 底 −3.95）
    phys = torch.empty(2, 9, 24, 32).uniform_(-4.0, 4.0)
    out = adm(phys, fluid_mask)

    m = fluid_mask[:, 0] > 0.5
    rho, u = _macro(out)
    assert torch.isfinite(out).all()
    assert float(rho[m].min()) >= RHO_MIN - 1e-6, f"ρ 最小 {float(rho[m].min())}"
    assert float(rho[m].max()) <= RHO_MAX + 1e-6
    # 容差 1e-4：投影把縮幅後的 j 寫回 norm 空間，測試再 atanh 讀回來，
    # 這一趟 fp32 來回在 0.55 量級上帶約 1e-5 的誤差。mach_max<1 的餘裕正是為此。
    assert float(u[m].max()) <= MACH_MAX * LATTICE_SOUND_SPEED + 1e-4
    # VQI 的兩條物理有效性判準因而 by construction 成立
    assert float(rho[m].min()) > 0.0
    assert float(u[m].max()) < LATTICE_SOUND_SPEED
    # 固體區歸零不受影響
    assert float(out[:, :, :4, :].abs().max()) == 0.0


def test_identity_when_admissible(fluid_mask: torch.Tensor) -> None:
    """未違反約束時逐位不變——否則不能掛在既有 ckpt 上。"""
    torch.manual_seed(1)
    adm = PhysicsAdmissibility(_cfg())
    # GT 流體區實測值域（S0-2）：ρ∈[0.327,1.038]、|u|max=0.145，全在盒內
    phys = torch.zeros(2, 9, 24, 32)
    phys[:, _RHO] = math.tanh((1.0 - M_REF[_RHO]) / S_COMB[_RHO])
    phys[:, _JX] = math.tanh(0.10 / S_COMB[_JX])
    phys[:, _JY] = math.tanh(-0.05 / S_COMB[_JY])
    phys[:, 1] = 0.3
    phys = phys * fluid_mask

    out = adm(phys, fluid_mask)
    assert torch.equal(out, phys), "可容許輸入必須逐位不變"


def test_disabled_is_identity(fluid_mask: torch.Tensor) -> None:
    """停用時對任何輸入皆為恆等（既有 config 零行為變更）。"""
    torch.manual_seed(2)
    adm = PhysicsAdmissibility(_cfg(enabled=False))
    phys = torch.empty(2, 9, 24, 32).uniform_(-4.0, 4.0)
    assert torch.equal(adm(phys, fluid_mask), phys)


def test_momentum_direction_preserved(fluid_mask: torch.Tensor) -> None:
    """馬赫投影只縮幅、不轉向。"""
    adm = PhysicsAdmissibility(_cfg())
    phys = torch.zeros(1, 9, 8, 8)
    phys[:, _RHO] = math.tanh((0.5 - M_REF[_RHO]) / S_COMB[_RHO])
    phys[:, _JX] = math.tanh(0.9 / S_COMB[_JX])      # 遠超 mach 上限
    phys[:, _JY] = math.tanh(0.3 / S_COMB[_JY])
    out = adm(phys, torch.ones(1, 1, 8, 8))

    jx0, jy0 = _denorm(phys[:, _JX], _JX), _denorm(phys[:, _JY], _JY)
    jx1, jy1 = _denorm(out[:, _JX], _JX), _denorm(out[:, _JY], _JY)
    cos = (jx0 * jx1 + jy0 * jy1) / (torch.sqrt(jx0 ** 2 + jy0 ** 2)
                                     * torch.sqrt(jx1 ** 2 + jy1 ** 2))
    assert float(cos.min()) > 1.0 - 1e-5
    assert float(torch.sqrt(jx1 ** 2 + jy1 ** 2).max()) < float(torch.sqrt(jx0 ** 2 + jy0 ** 2).max())


def test_gradients_finite_with_zero_momentum_region(fluid_mask: torch.Tensor) -> None:
    """梯度必須有限——固體區 jx=jy=0 曾讓 sqrt 的反向導數變 ∞ 而毀掉整批訓練。

    實測（`runs/LBM_NCA-8/main_20260814_1950`，已作廢）：未 detach 縮放係數時
    96 個訓練步的有限 `grad_norm` 為 **0 筆**、AMP scale 崩到 8e-25。
    本測試就是那次事故的回歸守衛，不可刪。
    """
    torch.manual_seed(3)
    adm = PhysicsAdmissibility(_cfg())
    phys = torch.empty(2, 9, 24, 32).uniform_(-3.0, 3.0)
    phys = phys * fluid_mask          # 固體區精確為 0 → sqrt(0) 的反向即在此
    phys.requires_grad_(True)

    out = adm(phys, fluid_mask)
    out.square().mean().backward()

    assert phys.grad is not None
    assert torch.isfinite(phys.grad).all(), (
        f"非有限梯度 {int((~torch.isfinite(phys.grad)).sum())} 個")


def test_config_rejects_incomplete_setup() -> None:
    """啟用但常數不全時立即報錯，不得靜默退化為恆等。"""
    with pytest.raises(ValueError):
        TESModelConfig(phys_adm_enabled=True, phys_adm_rho_min=0.02,
                       phys_adm_rho_max=2.24, phys_adm_mach_max=0.95,
                       phys_adm_m_ref=[1.0], phys_adm_s_combined=S_COMB)
    with pytest.raises(ValueError):
        TESModelConfig(phys_adm_enabled=True, phys_adm_rho_min=0.0,
                       phys_adm_rho_max=2.24, phys_adm_mach_max=0.95,
                       phys_adm_m_ref=M_REF, phys_adm_s_combined=S_COMB)
