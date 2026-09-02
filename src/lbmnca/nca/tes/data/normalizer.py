"""TESCondNormalizer — cond_17D / cond_5D 條件向量正規化。"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# cond_5D（M1A）cumulative CTU 正規化上限
# cold trajectory 最長約 100 CTU；log1p(100) ≈ 4.615
_CTU_LOG_MAX: float = math.log1p(100.0)
# delta_CTU 正規化上限：Staircase schedule 最大步長 = 2.0 CTU
_DELTA_CTU_LOG_MAX: float = math.log1p(5.0)  # 稍留餘量

# 時間條件化未接線時的替代常數。
# 這兩個值決定 cond[17]／cond[18] 在 canonical 下的實際內容：
#   delta_ctu=1.0  → log1p(1.0)/log1p(5.0) = 0.3868
#   cumul_ctu=0.0  → log1p(0.0)/log1p(100) = 0.0
# 亦即 cond_dim=19 在 canonical mean 線上實質為 17D＋兩個常數維。
# 原本這兩個值是寫死在 build_cond 內的裸字面量，看不出它們會流進條件向量。
# 會實際消費這兩維的 cond_dim（17/3 不含時間維，傳 None 無害）


class TESCondNormalizer:
    """G_vec_B (14D) 與 P_vec_B (3D) 的 z-score 正規化器。

    統計量從 global_stats.json 的 `cond_stats` 欄位載入，
    或由資料集直接計算（`from_dataset`）。

    正規化格式：
        ñ = (x - μ) / (σ + ε)

    輸出 cond_17D：
        [0:14]  G_vec_B_norm
        [14:17] P_vec_B_norm

    Args:
        g_mean: [14] G_vec 均值。
        g_std:  [14] G_vec 標準差。
        p_mean: [3]  P_vec 均值（Re, U_inlet, L_char）。
        p_std:  [3]  P_vec 標準差。
        eps: 避免除以零的小量。
    """

    # time-fallback 一次性警告旗標（避免每個 batch 洗版）。

    def __init__(
        self,
        g_mean: np.ndarray,
        g_std: np.ndarray,
        p_mean: np.ndarray,
        p_std: np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        self._g_mean = torch.tensor(g_mean, dtype=torch.float32)
        self._g_std = torch.tensor(g_std, dtype=torch.float32)
        self._p_mean = torch.tensor(p_mean, dtype=torch.float32)
        self._p_std = torch.tensor(p_std, dtype=torch.float32)
        self._eps = eps

    # ------------------------------------------------------------------
    # 建構方式
    # ------------------------------------------------------------------

    @classmethod
    def from_stats_json(cls, stats_path: str | Path) -> TESCondNormalizer:
        """從 global_stats.json 的 `cond_stats` 欄位建構。

        JSON 格式：
            {
              "cond_stats": {
                "g_vec": {"mean": [...14...], "std": [...14...]},
                "p_vec": {"mean": [...3...],  "std": [...3...]}
              }
            }

        Args:
            stats_path: global_stats.json 路徑。

        Returns:
            TESCondNormalizer 實例。

        Raises:
            KeyError: JSON 缺少 `cond_stats` 欄位。
        """
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        cond = stats["cond_stats"]
        return cls(
            g_mean=np.array(cond["g_vec"]["mean"], dtype=np.float32),
            g_std=np.array(cond["g_vec"]["std"], dtype=np.float32),
            p_mean=np.array(cond["p_vec"]["mean"], dtype=np.float32),
            p_std=np.array(cond["p_vec"]["std"], dtype=np.float32),
        )

    @classmethod
    def from_stats_dict(cls, stats: dict[str, dict]) -> TESCondNormalizer:
        """從已載入記憶體的 stats dict 建構。

        Args:
            stats: 包含 ``cond_stats`` 鍵的統計量字典，格式與
                ``from_stats_json`` 所讀取的 JSON 結構相同。

        Returns:
            TESCondNormalizer 實例。

        Raises:
            KeyError: ``stats`` 缺少 ``cond_stats`` 欄位。
        """
        cond = stats["cond_stats"]
        return cls(
            g_mean=np.array(cond["g_vec"]["mean"], dtype=np.float32),
            g_std=np.array(cond["g_vec"]["std"], dtype=np.float32),
            p_mean=np.array(cond["p_vec"]["mean"], dtype=np.float32),
            p_std=np.array(cond["p_vec"]["std"], dtype=np.float32),
        )

    @classmethod
    def from_arrays(
        cls,
        g_vecs: np.ndarray,
        p_vecs: np.ndarray,
        eps: float = 1e-8,
    ) -> TESCondNormalizer:
        """從原始 g_vec / p_vec 陣列計算統計量。

        Args:
            g_vecs: [N, 14] 所有案例的 G_vec。
            p_vecs: [N, 3]  所有案例的 P_vec。
            eps: 避免除以零。

        Returns:
            TESCondNormalizer 實例。
        """
        return cls(
            g_mean=g_vecs.mean(axis=0),
            g_std=g_vecs.std(axis=0) + eps,
            p_mean=p_vecs.mean(axis=0),
            p_std=p_vecs.std(axis=0) + eps,
            eps=eps,
        )

    # ------------------------------------------------------------------
    # 正規化 / 反正規化
    # ------------------------------------------------------------------

    #: 退化維判定門檻。訓練資料中樣本標準差低於此值者視為常數維。
    #: 實測 B4/C2/F3 三維在全部 670 個 canonical 節點上 std 精確為 0，
    #: stats.json 內存的是 tes_h5_to_zarr.py 加的 eps（9.99999994e-09）。
    DEGENERATE_STD: float = 1e-6

    def _zscore_guarded(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """z-score 正規化；退化維（訓練期為常數）一律輸出 0，不做除法。

        g 側與 p 側共用同一實作，確保兩者行為不會分岔。運算順序為先算完整
        z-score、再以 ``torch.where`` 覆寫退化維，使 g 側維持逐位元不變。

        Args:
            x: ``[..., D]`` 原始條件向量。
            mean: ``[D]`` 訓練期均值。
            std: ``[D]`` 訓練期標準差。

        Returns:
            ``[..., D]`` 正規化結果；``std <= DEGENERATE_STD`` 的維度為 0。
        """
        device = x.device
        mean = mean.to(device)
        std = std.to(device)
        out = (x - mean) / (std + self._eps)
        degenerate = (std <= self.DEGENERATE_STD)
        if bool(degenerate.any()):
            out = torch.where(degenerate.expand_as(out),
                              torch.zeros_like(out), out)
        return out

    def normalize_g(self, g_vec: torch.Tensor) -> torch.Tensor:
        """正規化 G_vec_B。

        退化維（訓練期為常數）一律輸出 0，不做除法。

        G_vec 的 B4_max_enclosed_fluid_area_ratio、C2_blockage_ratio、
        F3_outlet_reachability 三維在全部 670 個 canonical 節點上樣本標準差
        **精確為 0**（分別恆為 0 / 0 / 1）——這是遮罩生成慣例的副產物
        （建物限於 inner domain、不觸入口、無封閉口袋），不是幾何本質。

        原本 ``(x - μ) / (std + eps)`` 的有效除數為 2e-8（stats 端加一次 eps、
        此處再加一次）。訓練期 x ≡ μ 故輸出恆 0，**缺陷不可能在訓練期暴露**；
        但部署端只要使用者畫出中庭（B4>0）、貼入口量體（C2>0）或橫幅牆體
        （F3<1）——恰恰是行人風環境最關切的都市形態——正規化即輸出
        **5×10⁵ ~ 5×10⁶ 量級**的條件分量，而 TESConditionEncoder 僅
        Linear→SiLU→Linear、無 clamp，regime_features 的守門也只檢查
        L_char 與 BCR，全鏈無任何 OOD 防護。

        本修正**對訓練數值完全等價**：訓練樣本在這些維上 x − μ ≡ 0，
        原式輸出 0，新式亦輸出 0（逐位元相同），故**既有 checkpoint 不需重訓**。
        改變的只有部署端遇到非常數輸入時的行為：由爆炸改為安全的 0
        （等同「模型從未見過這個訊號，故不注入資訊」）。

        Args:
            g_vec: [..., 14] 原始幾何條件向量。

        Returns:
            [..., 14] z-score 正規化後的 G_vec；退化維為 0。
        """
        return self._zscore_guarded(g_vec, self._g_mean, self._g_std)

    def normalize_p(self, p_vec: torch.Tensor) -> torch.Tensor:
        """正規化 P_vec_B（Re, U_inlet, L_char）。

        退化維（訓練期為常數）一律輸出 0，不做除法。

        U_inlet 在全部 814 個生產 config 與 670 個 canonical node 上恆為 0.05
        （位元完全相同，``0x3d4ccccd``），樣本標準差為 0；stats 內存的
        1e-8 ~ 1.3e-7 是生成端 eps（``tes_h5_to_zarr.py:1106``）與 float32
        殘差。無守衛時部署端只要給出非 0.05 的 U_inlet，即產生 1e4 ~ 1e6
        量級的條件分量（實測 U=0.06 -> +73579，rollout 場崩壞 5448%），
        且 ``regime_features.py`` 的 config fallback 曾可靜默供給 0.06。

        ⚠ 與 ``normalize_g`` 不同，本修正**不是逐位元等價**：訓練期該欄的值
        是常數 −0.05482094（canonical 合併 stats；純粹是 float32 表示殘差
        被浮點除數放大而來），改為 0 會使既有 checkpoint 的輸出有微小位移。
        此代價可接受：該常數本來就隨載入哪份 stats 而變動、對應位移量級相近，
        歸零同時消滅了這個 stats 來源相依的變異，使條件值不再取決於載入哪份統計。

        ★ 範圍限制：歸零等於明確宣告「模型無法對入口速度做出反應」。這不是
        守衛造成的損失 —— 訓練集 814/814 全部 u_inlet=0.05，模型本來就沒學過
        這個自由度；守衛只是把既有的無能力顯性化，而非輸出垃圾。

        Args:
            p_vec: [..., 3] 原始物理條件向量。

        Returns:
            [..., 3] z-score 正規化後的 P_vec；退化維為 0。
        """
        return self._zscore_guarded(p_vec, self._p_mean, self._p_std)

    def build_cond17d(
        self,
        g_vec: torch.Tensor,
        p_vec: torch.Tensor,
    ) -> torch.Tensor:
        """組合並正規化 cond_17D。

        Args:
            g_vec: [B, 14] 原始 G_vec_B。
            p_vec: [B, 3]  原始 P_vec_B。

        Returns:
            cond_norm: [B, 17] 正規化後的條件向量。
        """
        g_norm = self.normalize_g(g_vec)
        p_norm = self.normalize_p(p_vec)
        return torch.cat([g_norm, p_norm], dim=-1)

    def normalize_cumulative_ctu(self, cumulative_ctu: torch.Tensor) -> torch.Tensor:
        """正規化 cumulative CTU：log1p / log1p(100)，映射至 [0, 1]。

        Args:
            cumulative_ctu: [...] float，原始累積 CTU 值。

        Returns:
            [...] 正規化值，約在 [0, 1]（超過 100 CTU 會超出 1.0 但可接受）。
        """
        return torch.log1p(cumulative_ctu.float()) / _CTU_LOG_MAX

    def normalize_delta_ctu(self, delta_ctu: torch.Tensor) -> torch.Tensor:
        """正規化 delta CTU：log1p / log1p(5)，映射至 [0, 1]。

        Staircase 典型值：0.2 → 0.103，1.0 → 0.387，2.0 → 0.613。

        Args:
            delta_ctu: [...] float，原始 delta CTU（步長，單位 CTU）。

        Returns:
            [...] 正規化值。
        """
        return torch.log1p(delta_ctu.float().clamp_min(0.0)) / _DELTA_CTU_LOG_MAX




    # ------------------------------------------------------------------
    # cond_dim 統一分派
    # ------------------------------------------------------------------


    def build_cond(
        self,
        cond_dim: int,
        g_vec: torch.Tensor,
        p_vec: torch.Tensor,
    ) -> torch.Tensor:
        """建構條件向量。**只支援 17D**。

        ``cond_17D = [G_vec_norm(14), P_vec_norm(3)]``

        mean flow 是**時均場**，時間維（``delta_ctu`` / ``cumulative_ctu``）在語義上
        不該存在於條件向量中，故 cond_dim 固定為 17。canonical mean 線的訓練與部署
        都不傳這兩個時間值（``inference/protocol.py`` 明文把「三態都以
        delta_ctu=None 建條件向量」定為協定）。

        cond_dim=19（含時間維）的既有 ckpt 與本架構的 encoder 輸入維度不同，
        不可直接載入，須從頭訓練。

        Args:
            cond_dim: 必須為 17（保留參數是為了讓呼叫端顯式宣告期望維度，
                寫錯即 raise，而不是靜默送出別的維度）。
            g_vec: ``[B, 14]`` 原始 G_vec_B。
            p_vec: ``[B, 3]``  原始 P_vec_B。

        Returns:
            ``[B, 17]`` 正規化後的條件向量。

        Raises:
            ValueError: ``cond_dim != 17``。
        """
        if cond_dim != 17:
            raise ValueError(
                f"cond_dim 必須為 17，得到 {cond_dim}。含時間維的 4/5/19 與過渡用的 "
                "3/18 已於 2026-08-05 移除——mean flow 是時均場，時間條件在語義上不成立。"
            )
        return self.build_cond17d(g_vec, p_vec)

    # ------------------------------------------------------------------
    # 裝置搬移
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> TESCondNormalizer:
        """將內部統計張量搬至指定裝置（in-place，回傳 self 方便鏈式呼叫）。

        Args:
            device: 目標裝置，例如 ``"cuda"``、``"cpu"``、``torch.device("cuda:0")``。

        Returns:
            self，以支援鏈式呼叫。
        """
        self._g_mean = self._g_mean.to(device)
        self._g_std = self._g_std.to(device)
        self._p_mean = self._p_mean.to(device)
        self._p_std = self._p_std.to(device)
        return self

    def to_stats_dict(self) -> dict[str, dict[str, list[float]]]:
        """匯出統計量為 dict（用於存入 global_stats.json）。

        Returns:
            dict，符合 from_stats_json 期望的格式。
        """
        return {
            "g_vec": {
                "mean": self._g_mean.tolist(),
                "std": self._g_std.tolist(),
            },
            "p_vec": {
                "mean": self._p_mean.tolist(),
                "std": self._p_std.tolist(),
            },
        }
