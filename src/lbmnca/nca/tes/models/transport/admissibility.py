"""物理可容許性投影（PhysicsAdmissibility）。

**這個算子存在的理由**

NCA 的 state 存放在 tanh 正規化的**像**裡，合法值域為 $(-1,1)$；
反正規化為 ``atanh(clip(ñ, ±0.9999))·S + M_REF``（`FieldNormalizer.denorm_mphy`）。
但 `MarkovNCACell` 的更新是**無界加法**（``dynamic = state + delta``），
值域為 $\\mathbb{R}$。既有的三道結構性護欄——ψ-curl 動量、KE rescale／ceiling、
hidden 的 tanh 界——**全部只覆蓋 jx/jy 與 hidden**，密度通道沒有任何機制
把它留在像內。

實測後果：長推進段固體邊界旁的像素 $\\tilde\\rho$ 漂出 $-1$，反正規化讀出 clip 底
$1-\\operatorname{atanh}(0.9999)=-3.9517$（回報值 −3.9516，吻合到小數第四位），
速度 $u=j/\\rho$ 在 $\\rho$ 穿越零點時發散。k=50 起九案有一案失效，k=100 有五案。

本模組把兩條**物理上不可違反**的條件變成構造性保證：

| 條件 | 施加方式 |
|---|---|
| $\\rho \\in [\\rho_{\\min}, \\rho_{\\max}]$，$\\rho_{\\min}>0$ | norm 空間硬夾（tanh 單調，夾 norm ≡ 夾 raw）|
| $\\lvert u\\rvert = \\lvert j\\rvert/\\rho \\le M\\,c_s$ | 逐像素等比縮 (jx, jy)，不改方向 |

**兩者在區間內部逐位恆等。** 未違反的像素連浮點來回都不做（`torch.where` 取原值），
故本算子對健康場的輸出**逐位不變**——這是它能掛在既有 ckpt 上而不動搖既有讀數的前提，
不是近似說法。實測 k≤25 逐案 rel_l2 與關閉時完全相同。

**三態同一算子。** 訓練、評估、部署走同一段程式碼，不提供任何 inference-only 分支。
常數由 config 攜帶（`TESModelConfig.phys_adm_*`），註冊為 buffer 後隨 ckpt 序列化，
故任何載入該 ckpt 的路徑都拿得到同一組常數。

**不做的事**：本算子不修正場的正確性。它是保險絲，不是修正器——
密度長期貼在 $\\rho_{\\min}$ 代表模型本身在對抗約束，那要靠訓練解決，
監控欄位 `n_rho_floor` / `n_mach_capped` 就是為了讓這件事無法被蓋掉。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ...config import TESModelConfig

#: 晶格聲速。LBM 低馬赫近似的適用邊界，與 `eval/vqi/terms.py::LATTICE_SOUND_SPEED` 同值。
LATTICE_SOUND_SPEED: float = 1.0 / math.sqrt(3.0)

#: physics block 內的通道索引（`src/lbmnca/nca/utils/FieldConverter.py` 的 MPHY_*_IDX）。
_RHO, _JX, _JY = 0, 3, 5


class PhysicsAdmissibility(nn.Module):
    """把 physics block 投影回物理可容許集合。停用時為恆等映射。

    Attributes:
        enabled: 是否啟用。False 時 `forward` 直接回傳輸入。
        last_counts: 上一次呼叫的觸界計數（診斷用，detached int）。
    """

    def __init__(self, cfg: TESModelConfig) -> None:
        """由 model config 建立投影算子。

        Args:
            cfg: TES 模型設定。需有 `phys_adm_enabled` 等欄位；啟用時
                `phys_adm_m_ref` / `phys_adm_s_combined` 必須為長度 9 的序列
                （由 config 產生腳本從 dataset stats 寫入）。
        """
        super().__init__()
        self.enabled: bool = bool(cfg.phys_adm_enabled)
        self.last_counts: dict[str, int] = {}
        if not self.enabled:
            return
        m_ref = torch.tensor(list(cfg.phys_adm_m_ref), dtype=torch.float32)
        s_comb = torch.tensor(list(cfg.phys_adm_s_combined), dtype=torch.float32)
        self.register_buffer("_m_ref", m_ref)
        self.register_buffer("_s_comb", s_comb)
        # norm 空間的 rho 界。tanh 單調遞增 → 夾 norm 等價於夾 raw，且省一次 atanh。
        self._n_lo: float = math.tanh((float(cfg.phys_adm_rho_min) - float(m_ref[_RHO]))
                                      / float(s_comb[_RHO]))
        self._n_hi: float = math.tanh((float(cfg.phys_adm_rho_max) - float(m_ref[_RHO]))
                                      / float(s_comb[_RHO]))
        self._mach_max: float = float(cfg.phys_adm_mach_max)
        self._rho_min: float = float(cfg.phys_adm_rho_min)

    # ------------------------------------------------------------------
    def _denorm(self, n: torch.Tensor, ch: int) -> torch.Tensor:
        """單通道 norm → raw。與 `FieldNormalizer.denorm_mphy` 同式（含 ±0.9999 clip）。"""
        return torch.atanh(n.clamp(-0.9999, 0.9999)) * self._s_comb[ch] + self._m_ref[ch]

    # ------------------------------------------------------------------
    def forward(
        self,
        phys: torch.Tensor,
        fluid_mask: torch.Tensor,
        collect_counts: bool = False,
    ) -> torch.Tensor:
        """投影 physics block。

        Args:
            phys: physics 通道，形狀 [B, physics_dim, H, W]，norm 空間。
            fluid_mask: 流體遮罩 [B, 1, H, W]，固體為 0。
            collect_counts: 是否統計觸界像素數並寫入 `last_counts`（診斷用，
                含 GPU→CPU 同步，訓練主迴圈預設關閉）。

        Returns:
            投影後的 physics，形狀與 dtype 同輸入。未違反的像素逐位不變。
        """
        if not self.enabled:
            return phys
        with torch.no_grad():
            proj = self._project(phys, fluid_mask, collect_counts)
        # ★straight-through：forward 取投影值，backward 為恆等映射。
        #
        # 記憶體：投影本身若進計算圖，k=24 的 rollout 會為每個 outer step 存下
        #   atanh／tanh／where／cat 的中間張量，實測 peak +2.79GB（8170→10961MB）。
        #   batch 5 原本就吃到顯存的八成，多這一塊就讓 allocator 開始抖動，
        #   每步耗時隨 k 從 1.6s 爬到 3.5s。改走 straight-through 後
        #   peak 8168MB＝**與投影關閉時相同**，而 forward 數值一模一樣。
        #
        # 梯度：這不只是省記憶體，語義上也更對。硬夾的真導數在界外為 **0**，
        #   代表**違規的像素永遠收不到梯度、也就永遠學不會不違規**。
        #   straight-through 讓損失在投影後的場上算出的梯度原樣傳回投影前的值，
        #   模型才有機會學到「別把密度推到界外」。
        #
        # 恆等性：未違反時 proj 與 phys 逐位相同 → delta 為 0 → 輸出不變。
        return phys + (proj - phys.detach())

    # ------------------------------------------------------------------
    def _project(
        self,
        phys: torch.Tensor,
        fluid_mask: torch.Tensor,
        collect_counts: bool,
    ) -> torch.Tensor:
        """實際的投影運算（呼叫端已置於 `no_grad` 中）。

        Args:
            phys: physics 通道 [B, physics_dim, H, W]，norm 空間。
            fluid_mask: 流體遮罩 [B, 1, H, W]。
            collect_counts: 是否統計觸界像素數（含 GPU→CPU 同步）。

        Returns:
            投影後的 physics，形狀與 dtype 同輸入。
        """
        dt = phys.dtype
        # 只取需要的三個通道轉 fp32：atanh 在 |x|→1 附近病態，而 AMP 下 phys 可能是 fp16。
        # 其餘六個通道原樣帶過，不做多餘的整批轉型。
        with torch.autocast(device_type=phys.device.type, enabled=False):
            rho_n = phys[:, _RHO:_RHO + 1].float().clamp(self._n_lo, self._n_hi)
            jx_n0 = phys[:, _JX:_JX + 1].float()
            jy_n0 = phys[:, _JY:_JY + 1].float()

            rho = self._denorm(rho_n, _RHO).clamp_min(self._rho_min)
            jx = self._denorm(jx_n0, _JX)
            jy = self._denorm(jy_n0, _JY)
            jmag = torch.sqrt(jx * jx + jy * jy)
            cap = self._mach_max * LATTICE_SOUND_SPEED * rho
            scale = torch.clamp(cap / jmag.clamp_min(1e-12), max=1.0)
            capped = scale < 1.0
            # ★未觸界者取原值而非重算：denorm→renorm 來回即使 scale=1 也帶浮點誤差，
            #   每步注入一次會累積，讓「不介入時逐位恆等」的宣稱不成立（實測 k=6 已偏 1e-4）。
            jx_n = torch.where(capped,
                               torch.tanh((jx * scale - self._m_ref[_JX]) / self._s_comb[_JX]),
                               jx_n0)
            jy_n = torch.where(capped,
                               torch.tanh((jy * scale - self._m_ref[_JY]) / self._s_comb[_JY]),
                               jy_n0)

            if collect_counts:
                fm = fluid_mask > 0.5
                self.last_counts = {
                    "n_rho_floor": int(((phys[:, _RHO:_RHO + 1] < self._n_lo) & fm).sum()),
                    "n_rho_ceil": int(((phys[:, _RHO:_RHO + 1] > self._n_hi) & fm).sum()),
                    "n_mach_capped": int((capped & fm).sum()),
                    "n_fluid": int(fm.sum()),
                }

        out = torch.cat([
            rho_n.to(dt), phys[:, _RHO + 1:_JX],
            jx_n.to(dt), phys[:, _JX + 1:_JY],
            jy_n.to(dt), phys[:, _JY + 1:],
        ], dim=1)
        return out * fluid_mask.to(dt)


__all__ = ["PhysicsAdmissibility", "LATTICE_SOUND_SPEED"]
