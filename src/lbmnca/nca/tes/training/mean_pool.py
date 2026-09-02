"""mean_pool.py — Mean Flow 模組專用 Experience Pool。

mean_module 目標是**固定點均場**（child_mean），中間步無 GT。
本 pool 每個 slot 存「模型自身 rollout 後的 state」+ 該 case 的
(child_mean target / g_vec / p_vec)，cross-iteration 累積真實偏離分布。
state 含 static 通道（state[:, :sd]）。

slot 組成（每次 sample 動態指派，非固定 slot range）：
  - seed   ：重置為 init_state(static, parent_mean) → 練習「從頭收斂」
  - rolling：前次 rollout 結果 → 練習「從中間偏離態繼續收斂」
  - corrupt：全場加高斯噪聲 state → 練習「從損壞態自修復」
  - damage ：挖掉連續 rect 區域（physics 通道）→ Growing-NCA 局部自組織重建
            （與 corrupt 並存；corrupt 練全域去噪、damage 練局部靠鄰域重生）

pushforward 由 pool 的累積偏離提供，故訓練端
grad_tail = min(K, cap) 即可全 BPTT 短窗，不需額外 no_grad preroll。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MeanPoolSample:
    """sample() 回傳：B 個 slot 的 IC state + per-case target/cond。"""

    slot_ids: torch.Tensor  # [B] long
    state: torch.Tensor  # [B, C, H, W] IC（含 static + physics）
    target: torch.Tensor  # [B, pd, H, W] child_mean
    parent: torch.Tensor  # [B, pd, H, W] parent_mean（wake 加權 |target−parent| 用）
    g_vec: torch.Tensor  # [B, g]
    p_vec: torch.Tensor  # [B, p]
    case_ids: list[
        str | None
    ]  # 對應 B 個 slot 的 case_id（writeback 後鏈式接力用）


class MeanPool:
    """Mean Flow Experience Pool（GPU tensor 池）。"""

    def __init__(
        self,
        size: int,
        sd: int,
        pd: int,
        device: torch.device,
        seed_frac: float = 0.15,
        corrupt_frac: float = 0.10,
        corrupt_sigma: float = 0.2,
        damage_frac: float = 0.0,
        damage_mode: str = "rect",
        damage_size_frac: float = 0.2,
        damage_fill: str = "zero",
    ) -> None:
        self.N = int(size)
        self.sd = int(sd)
        self.pd = int(pd)
        self.device = device
        self.seed_frac = float(seed_frac)
        self.corrupt_frac = float(corrupt_frac)
        self.corrupt_sigma = float(corrupt_sigma)
        self.damage_frac = float(damage_frac)
        self.damage_mode = str(damage_mode)
        self.damage_size_frac = float(damage_size_frac)
        self.damage_fill = str(damage_fill)
        self._damage_count = 0

        self._states: torch.Tensor | None = None  # [N, C, H, W]
        self._targets: torch.Tensor | None = None  # [N, pd, H, W]
        self._parent: torch.Tensor | None = (
            None  # [N, pd, H, W]（wake 權重來源＝真實 parent）
        )
        # ★seed 重置的 IC 必須與該臂的消融條件一致，不能用真實 parent。
        #   `_parent`（wake 加權來源）統一為真實 parent；若 refill 也拿它灌
        #   seed IC，seed 重置（依 seed_frac 比例）會把真實 parent 當 IC 注入
        #   **全部消融臂**：abl_no 的「完全沒有暖起點」不成立，far/rand 也會有
        #   樣本等同 abl_use。一個欄位兼兩職是根因，故拆成兩欄。
        self._seed_ic: torch.Tensor | None = None  # [N, pd, H, W]（各臂覆寫後的 IC）
        self._gvec: torch.Tensor | None = None
        self._pvec: torch.Tensor | None = None
        self._filled = torch.zeros(self.N, dtype=torch.bool, device=device)
        self._ptr = 0  # round-robin refill 指標
        self._reset_count = 0
        self._sample_count = 0
        # per-slot case_id（None=未指派，refill 時填入）。跨案例鏈式接力用
        # （chain_writeback 需知道哪個 slot 目前住著哪個 case，才能找到 child 在池中的位置）。
        self._case_ids: list[str | None] = [None] * self.N
        self._chain_count = 0

    # ------------------------------------------------------------------

    def _alloc(self, C: int, H: int, W: int, gd: int, pdv: int) -> None:
        self._states = torch.zeros(self.N, C, H, W, device=self.device)
        self._targets = torch.zeros(self.N, self.pd, H, W, device=self.device)
        self._parent = torch.zeros(self.N, self.pd, H, W, device=self.device)
        self._seed_ic = torch.zeros(self.N, self.pd, H, W, device=self.device)
        self._gvec = torch.zeros(self.N, gd, device=self.device)
        self._pvec = torch.zeros(self.N, pdv, device=self.device)

    @property
    def ready(self) -> bool:
        return bool(self._filled.any())

    def fill_count(self) -> int:
        return int(self._filled.sum().item())

    # ------------------------------------------------------------------

    @torch.no_grad()
    def refill(
        self,
        state0: torch.Tensor,  # [B, C, H, W] init_state(static, parent_mean)
        target: torch.Tensor,  # [B, pd, H, W] child_mean
        parent: torch.Tensor,  # [B, pd, H, W] 真實 parent（wake 加權用，**不作 seed IC**）
        seed_ic: torch.Tensor,  # [B, pd, H, W] 各臂覆寫後的 IC（seed re-init 用）
        g_vec: torch.Tensor,  # [B, g]
        p_vec: torch.Tensor,  # [B, p]
        case_ids: (
            list[str] | None
        ) = None,  # 對應 B 個 case 的 zarr key（None=不追蹤）
    ) -> None:
        """以新 batch 的 case 灌入/刷新 pool slot（round-robin），引入新案例多樣性。"""
        B, C, H, W = state0.shape
        if self._states is None:
            self._alloc(C, H, W, g_vec.shape[1], p_vec.shape[1])
        for i in range(B):
            j = self._ptr % self.N
            self._states[j] = state0[i].detach()
            self._targets[j] = target[i].detach()
            self._parent[j] = parent[i].detach()
            self._seed_ic[j] = seed_ic[i].detach()
            self._gvec[j] = g_vec[i].detach()
            self._pvec[j] = p_vec[i].detach()
            self._filled[j] = True
            self._case_ids[j] = case_ids[i] if case_ids is not None else None
            self._ptr += 1

    @torch.no_grad()
    def sample(self, B: int, model) -> MeanPoolSample:
        """抽 B 個 slot；部分重置為 seed（從 parent 重收斂）、部分全場加噪 corrupt、
        部分挖 rect damage（局部重建），其餘為 rolling（從偏離態續收斂）。

        Args:
            B: batch size。
            model: 提供 init_state(static, parent) 重建 seed IC。
        """
        filled_idx = torch.nonzero(self._filled, as_tuple=False).squeeze(1)
        # 不足 B 時允許重複抽
        replace = filled_idx.numel() < B
        sel = (
            filled_idx[torch.randint(filled_idx.numel(), (B,), device=self.device)]
            if replace
            else filled_idx[torch.randperm(filled_idx.numel(), device=self.device)[:B]]
        )

        state = self._states[sel].clone()
        target = self._targets[sel].clone()
        parent = self._parent[sel].clone()
        seed_ic = self._seed_ic[sel].clone()
        gvec = self._gvec[sel].clone()
        pvec = self._pvec[sel].clone()

        n_seed = int(round(B * self.seed_frac))
        n_corrupt = int(round(B * self.corrupt_frac))
        n_damage = int(round(B * self.damage_frac))
        sd, pd = self.sd, self.pd

        # slot 前綴分區：[0:n_seed]=seed、[n_seed:+n_corrupt]=corrupt、
        # [+n_damage]=damage、其餘=rolling。三者互斥（config 已驗證 frac 和 <=1）。
        # seed：前 n_seed 個重置為 init_state(static, seed_ic)（從頭收斂）
        # ★用 seed_ic 而非 parent —— 後者是真實 parent，拿它當 IC 會讓消融臂
        #   在 pool 路徑上取得它本該被移除的暖起點（見 __init__ 的說明）。
        if n_seed > 0:
            _static = state[:n_seed, :sd]
            state[:n_seed] = model.init_state(_static, seed_ic[:n_seed])
            self._reset_count += n_seed
        # corrupt：接下來 n_corrupt 個對 physics 通道全場加噪（去噪自修復壓力）
        if n_corrupt > 0:
            lo, hi = n_seed, n_seed + n_corrupt
            noise = torch.randn_like(state[lo:hi, sd : sd + pd]) * self.corrupt_sigma
            state[lo:hi, sd : sd + pd] = state[lo:hi, sd : sd + pd] + noise
        # damage：再接 n_damage 個挖連續 rect 區域（Growing-NCA 局部自組織重建）
        if n_damage > 0:
            lo, hi = n_seed + n_corrupt, n_seed + n_corrupt + n_damage
            self._apply_rect_damage(state, lo, hi, sd, pd)
            self._damage_count += n_damage

        self._sample_count += 1
        sel_case_ids = [self._case_ids[int(s)] for s in sel.tolist()]
        return MeanPoolSample(
            slot_ids=sel,
            state=state,
            target=target,
            parent=parent,
            g_vec=gvec,
            p_vec=pvec,
            case_ids=sel_case_ids,
        )

    @torch.no_grad()
    def _apply_rect_damage(
        self, state: torch.Tensor, lo: int, hi: int, sd: int, pd: int
    ) -> None:
        """對 state[lo:hi] 的 physics 通道挖隨機連續 rect 區域 → Growing-NCA 局部破壞。

        每個 slot 獨立隨機位置。fill='zero' 硬挖（norm 空間置 0，經典 Growing NCA）；
        'noise' 則以 corrupt_sigma 高斯覆蓋。solid cell 於 rollout 由 fluid_mask 重罩，
        故實質等同僅破壞 fluid。static 通道（:sd）不動。

        Args:
            state: [B, C, H, W] pool sample state（in-place 修改）。
            lo/hi: 要破壞的 slot 索引區間 [lo, hi)。
            sd/pd: static / physics 通道數。
        """
        _, _, H, W = state.shape
        dh = max(1, int(round(H * self.damage_size_frac)))
        dw = max(1, int(round(W * self.damage_size_frac)))
        for i in range(lo, hi):
            top = int(torch.randint(0, max(1, H - dh + 1), (1,)).item())
            left = int(torch.randint(0, max(1, W - dw + 1), (1,)).item())
            region = state[i, sd : sd + pd, top : top + dh, left : left + dw]
            if self.damage_fill == "zero":
                region.zero_()
            else:
                region.copy_(torch.randn_like(region) * self.corrupt_sigma)

    @torch.no_grad()
    def writeback(self, slot_ids: torch.Tensor, final_state: torch.Tensor) -> None:
        """rollout 後把最終 state 寫回 pool（detached），累積偏離分布。"""
        self._states[slot_ids] = final_state.detach()

    @torch.no_grad()
    def chain_writeback(
        self,
        parent_case_ids: list[str | None],
        final_state: torch.Tensor,
        model: torch.nn.Module,
        child_map: dict[str, list[str]],
        frac: float,
    ) -> None:
        """跨案例鏈式接力。

        以機率 ``frac`` 把剛完成 rollout 的 parent case 輸出，接成池中 **同時在場**的
        child case 之新 IC——補「訓練期 IC 恆為 GT 父案例」與「評測期 IC 為 model 自身
        鏈式輸出」的系統性落差。只在 child 恰好也在池中時觸發（機率性、非保證覆蓋）；
        child 自己的 target/g_vec/p_vec 不動（本就存在該 slot），只換 physics IC。

        Args:
            parent_case_ids: 剛 writeback 的各 row 對應 case_id（來自 sample() 回傳的
                ``MeanPoolSample.case_ids``），長度需與 ``final_state`` 的 batch 維一致。
            final_state: 對應各 row 的 model 最終輸出（含 static+physics），
                形狀 [B, C, H, W]，未 detach 亦可（本函式內 detach）。
            model: 提供 ``init_state(static, mean_physics)`` 建構 child IC 的模型實例。
            child_map: ``{parent_case_id: [child_case_id, ...]}``。
            frac: 每個 parent row 觸發鏈式接力的機率，0=停用（呼叫端應先短路跳過）。
        """
        if frac <= 0.0 or not child_map:
            return
        case_to_slot = {c: i for i, c in enumerate(self._case_ids) if c is not None}
        sd, pd = self.sd, self.pd
        fs = final_state.detach()
        for i, pcid in enumerate(parent_case_ids):
            if pcid is None or pcid not in child_map:
                continue
            if float(torch.rand(1).item()) >= frac:
                continue
            for ccid in child_map[pcid]:
                cslot = case_to_slot.get(ccid)
                if cslot is None or not bool(self._filled[cslot]):
                    continue
                child_static = self._states[cslot : cslot + 1, :sd].to(fs.device)
                parent_physics = fs[i : i + 1, sd : sd + pd]
                new_ic = model.init_state(child_static, parent_physics)
                self._states[cslot] = new_ic[0].detach().to(self.device)
                self._chain_count += 1

    @torch.no_grad()
    def peek(self, first_n: int = 8, last_n: int = 8) -> dict | None:
        """取首 first_n / 末 last_n slot 的 physics 場（vis 用，監控 pool 偏離分布）。

        Returns:
            dict(first=[n,pd,H,W], last=[n,pd,H,W], target_first/last 同) 或 None（未配置）。
        """
        if self._states is None:
            return None
        fn = min(first_n, self.N)
        ln = min(last_n, self.N)
        return {
            "first": self._states[:fn].clone(),  # [n, C, H, W] full state（含 static）
            "last": self._states[self.N - ln :].clone(),
        }

    def stats(self) -> dict:
        return {
            "pool_fill": self.fill_count(),
            "pool_reset_count": self._reset_count,
            "pool_sample_count": self._sample_count,
            "pool_chain_count": self._chain_count,
        }


__all__ = ["MeanPool", "MeanPoolSample"]
