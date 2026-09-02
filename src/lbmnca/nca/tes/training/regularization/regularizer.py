"""訓練正則化項（KE 兩窗穩定性 Dy-C/E + Jacobian 壓縮 Dy-V8/V9）。

本模組將 TESTrainer 中的正則化邏輯封裝為獨立子元件，
降低 trainer.py 的複雜度，並讓正則化策略可獨立測試。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ...inference import protocol
from ...loss.loss_terms_v2 import (
    domain_mean_ke,
    jacobian_amp_loss,
    two_window_ke_loss,
)

if TYPE_CHECKING:
    from ..context import TrainerContext


class Regularizer:
    """訓練正則化項容器（KE 兩窗穩定性 Dy-C/E + Jacobian 壓縮 Dy-V8/V9）。

    封裝 TESTrainer._temporal_ke_loss 與 _jacobian_reg_loss 的完整邏輯。
    所有方法保留原始算法行為；當對應 config 停用時回傳 None，
    呼叫方可直接檢查 None 決定是否加入 total loss。

    Attributes:
        ctx: TrainerContext — 所有 Trainer 子元件共用的狀態容器。
    """

    def __init__(self, ctx: "TrainerContext") -> None:
        """初始化正則化子元件。

        Args:
            ctx: TrainerContext 共用狀態容器。
        """
        self.ctx = ctx
        # 可觀測性：計算閘門觸發次數 / hinge 實際非零次數 /
        # 最近一次量到的放大率與 raw hinge 值。三者皆與 log 取樣的 step 奇偶無關
        # （若記錄點 step 恆為奇數、計算閘門卻要求偶數，該項將整訓練期不可觀測）。
        self.jac_gate_calls: int = 0
        self.jac_active_calls: int = 0
        self.last_jac_ratio: float = float("nan")
        self.last_jac_raw: float = float("nan")

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    def temporal_ke(
        self,
        state: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        g_vec_norm: Optional[torch.Tensor],
        cumul_ctu: Optional[torch.Tensor],
        l_char: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Dy-C/E：L_temporal（solver 2CTU KE 統計穩態）。

        從監督起點 ``state`` 做長 rollout，逐步單步推進並增量算
        domain-mean KE（不 stack 整段 [B,R,9,H,W]，省顯存）：
        preroll ``R-G`` 步 ``no_grad``（detached）+ grad_tail ``G`` 步帶梯度，
        收集 ke_t [B,R] → ``two_window_ke_loss``。

        當 ``cfg.temporal.enabled=False``，或當前 step 不在 interval 週期
        時，回傳 ``None``（由呼叫方決定是否加入 total loss）。

        Args:
            state: 監督起點 state tensor，shape [B, state_dim, H, W]。
            cond: 正規化條件向量，shape [B, cond_dim]。
            mask: fluid mask，shape [B, 1, H, W]。
            g_vec_norm: 正規化幾何向量，shape [B, g_dim] 或 None。
            cumul_ctu: 監督起點的累積 CTU，shape [B] 或 None。
            l_char: 特徵長度（pixel 單位），shape [B] 或 None。

        Returns:
            scalar Tensor（grad 可回流至 grad_tail 段），或 None（停用時）。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        tc = cfg.temporal
        if not tc.enabled:
            return None
        if ctx.step % int(tc.interval) != 0:
            return None

        return self._compute_temporal_ke(
            state, cond, mask, g_vec_norm, cumul_ctu, l_char
        )

    def jacobian(
        self,
        state: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        g_vec_norm: Optional[torch.Tensor],
        delta_ctu: Optional[torch.Tensor],
        prior_ctx: Optional[dict],
        fx_ref: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Dy-V8/V9：單步 Jacobian 放大率懲罰（真收縮）。

        對 ``state`` 的 physics 加擾動 ε，比較 ``f(x+ε)`` 與 ``f(x)``
        的放大率 → ``jacobian_amp_loss``（全域 relu(ratio−γ) 或 JAWS per-pixel）。

        當 ``cfg.jac_reg_weight == 0.0`` 時回傳 ``None``。

        記憶體優化：``fx_ref`` 給定時重用既算好的 f(x)（省一次 forward），
        梯度仍經 f(x+ε) 回流至參數，收縮懲罰有效。

        Args:
            state: 監督起點 state tensor，shape [B, state_dim, H, W]。
            cond: 正規化條件向量，shape [B, cond_dim]。
            mask: fluid mask，shape [B, 1, H, W]。
            g_vec_norm: 正規化幾何向量，shape [B, g_dim] 或 None。
            delta_ctu: 第一步 delta CTU，shape [B, 1] 或 None。
            prior_ctx: operator-gated arch 的 runtime metadata dict，或 None。
            fx_ref: 已計算的 f(x) reference（可重用，省 forward），或 None。

        Returns:
            scalar Tensor，或 None（jac_reg_weight=0 時）。
        """
        cfg = self.ctx.cfg
        if float(cfg.jac_reg_weight) == 0.0:
            return None
        return self._compute_jacobian(
            state, cond, mask, g_vec_norm, delta_ctu, prior_ctx, fx_ref
        )

    @staticmethod
    def _jac_ratio(
        f,
        x: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """單步 Jacobian 放大率 ``ratio = ‖f(x+ε) − f(x)‖ / ‖ε‖``（全張量範數）。

        純函式（接受任意 callable ``f``），供 Jacobian reg 數學單元測試
        （例如 ``f = lambda x: 1.5*x`` 應得 ratio≈1.5）。
        實際訓練的 per-sample / JAWS per-pixel 版本另由
        ``loss_terms_v2.jacobian_amp_loss`` 提供。

        Args:
            f: 單步映射 callable（x → f(x)，同 shape）。
            x: 輸入張量。
            eps: 擾動張量（同 shape）。

        Returns:
            scalar Tensor：全域放大率。
        """
        num = (f(x + eps) - f(x)).flatten().norm()
        den = eps.flatten().norm().clamp_min(1e-12)
        return num / den

    # ------------------------------------------------------------------
    # 私有實作
    # ------------------------------------------------------------------

    def _compute_temporal_ke(
        self,
        state: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        g_vec_norm: Optional[torch.Tensor],
        cumul_start_ctu: Optional[torch.Tensor],
        l_char_px: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """_temporal_ke_loss 的完整實作（temporal.enabled 已在外部確認）。

        Args:
            state: 監督起點 state，shape [B, state_dim, H, W]。
            cond: 正規化條件向量。
            mask: fluid mask。
            g_vec_norm: 正規化幾何向量，或 None。
            cumul_start_ctu: 監督起點累積 CTU [B]，或 None。
            l_char_px: 特徵長度 [B]，或 None。

        Returns:
            scalar Tensor（two_window_ke_loss 輸出）。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        model = ctx.model
        tc = cfg.temporal
        R = int(tc.r_total)
        G = min(int(tc.grad_tail), R)
        pre = R - G
        delta = float(tc.delta)
        B = state.shape[0]
        dev = state.device

        # 判斷模型是否需要 per-step delta_ctu / prior_ctx
        uses_dctu: bool = self._uses_per_step_delta_ctu()
        wants_prior: bool = bool(getattr(model, "wants_prior_ctx", False))
        fnorm = ctx.field_normalizer
        prior_vel = self._get_prior_vel_norm() if wants_prior else None

        def _one_step(
            st: torch.Tensor, j: int
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """單步推進；回傳 (new_state, phys[B,9,H,W])。j = step index（0-based）。"""
            dt = (
                torch.full((B, 1), delta, device=dev, dtype=torch.float32)
                if uses_dctu
                else None
            )
            kw: dict = {"delta_ctu_seq": dt, "g_vec_norm": g_vec_norm}
            if (
                wants_prior
                and cumul_start_ctu is not None
                and l_char_px is not None
            ):
                cc = cumul_start_ctu.view(B, 1) + float(j) * delta  # [B,1]
                kw["prior_ctx"] = {
                    "current_ctu_raw_seq": cc,
                    "l_char_px": l_char_px,
                    "prior_vel_norm": prior_vel,
                }
            new_state, traj = model(
                st, cond, mask, 1, return_trajectory=True,
                ke_anchor=protocol.resolve_ke_anchor(cfg, st, mask), **kw
            )
            return new_state, traj[-1]

        ke_list: list[torch.Tensor] = []

        # preroll（no_grad，逐步丟棄 frame，只留 KE）
        with torch.no_grad():
            for j in range(pre):
                state, phys = _one_step(state, j)
                ke_list.append(
                    domain_mean_ke(phys, mask, fnorm).detach()
                )
            state = state.detach()

        # grad tail（梯度回流；僅 G 步圖）
        for jj in range(G):
            state, phys = _one_step(state, pre + jj)
            ke_list.append(domain_mean_ke(phys, mask, fnorm))

        ke_t = torch.stack(ke_list, dim=1)  # [B, R]

        # ctu_seq：每步 END ctu（raw）= cumul_start + (i+1)·Δ
        i_end = torch.arange(1, R + 1, device=dev, dtype=torch.float32)  # [R]
        if cumul_start_ctu is not None:
            ctu_seq = (
                cumul_start_ctu.view(B, 1) + i_end.view(1, R) * delta
            )
        else:
            ctu_seq = i_end.view(1, R).expand(B, R) * delta

        return two_window_ke_loss(
            ke_t, ctu_seq, window_ctu=tc.window_ctu, eps=tc.eps
        )

    def _compute_jacobian(
        self,
        state: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        g_vec_norm: Optional[torch.Tensor],
        dt_first: Optional[torch.Tensor],
        prior_ctx: Optional[dict],
        fx_ref: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """_jacobian_reg_loss 的完整實作（jac_reg_weight > 0 已在外部確認）。

        Args:
            state: 監督起點 state，shape [B, state_dim, H, W]。
            cond: 正規化條件向量。
            mask: fluid mask。
            g_vec_norm: 正規化幾何向量，或 None。
            dt_first: 第一步 delta CTU [B,1]，或 None。
            prior_ctx: operator-gated arch 的 runtime metadata dict，或 None。
            fx_ref: 已計算的 f(x) reference（可重用，省 forward），或 None。

        Returns:
            scalar Tensor（jacobian_amp_loss 輸出）。
        """
        ctx = self.ctx
        cfg = ctx.cfg
        model = ctx.model

        sd: int = cfg.model.static_dim
        pd: int = cfg.model.physics_dim
        phys = state[:, sd : sd + pd]
        eps = float(cfg.jac_eps) * torch.randn_like(phys) * mask
        if bool(cfg.jac_band_probe):   # band-shell 定向 jac probe
            # 各向同性隨機 eps 與 band-shell 最壞方向重疊≈0 → jac_reg 盲於 band gain。
            # 下方 DoG（avg_pool 31 與 9）的實際通帶約為 ~9–31 px；box 濾波在固體
            #   邊界與域邊界會引入偏誤（未做 normalized convolution）。canonical
            #   設定為 jac_band_probe=false，本分支預設不啟用。
            # DoG band-pass eps → 探測方向落在中波長子空間 → jacobian_amp_loss 罰 band 方向增益。
            import torch.nn.functional as _Fb
            _lo = _Fb.avg_pool2d(eps, 31, stride=1, padding=15)   # >~31px（大尺度）
            _hi = _Fb.avg_pool2d(eps, 9, stride=1, padding=4)     # >~9px（中+大）
            _band = (_hi - _lo)                                    # 留 ~9-31px 中波長帶
            _rms0 = eps.detach().pow(2).mean().sqrt()
            _rmsb = _band.detach().pow(2).mean().sqrt() + 1e-8
            eps = (_band * (_rms0 / _rmsb)) * mask                 # renorm 保 jac_eps 尺度

        # 組裝加擾動後的 state（保留 static + 所有 extra channels：hidden or turb m̄+∇m̄）
        # 用 shape（而非 hidden_dim>0）判斷是否有額外通道：extra-channel 架構下
        # hidden=0 但 state 仍可能有額外通道，hidden_dim>0 這個條件對這類架構不成立。
        state_eps = torch.cat(
            [state[:, :sd], phys + eps, state[:, sd + pd :]], dim=1
        )  # state[:, sd+pd:] 為空時 cat 無 side-effect（沒有額外通道時安全略過）

        kw: dict = {"delta_ctu_seq": dt_first, "g_vec_norm": g_vec_norm}
        if prior_ctx is not None:
            kw["prior_ctx"] = prior_ctx

        # Jacobian 探針量的必須是**部署實際跑的那個算子**，故 ceiling
        # 修正器也要在探針內生效；錨點由三態協定從各自起點產生（禁靜默 bypass）。
        kw_x = dict(kw, ke_anchor=protocol.resolve_ke_anchor(cfg, state, mask))
        kw_xe = dict(kw, ke_anchor=protocol.resolve_ke_anchor(cfg, state_eps, mask))

        # f(x)：重用 fx_ref（省一次 forward）或自算
        if fx_ref is None:
            _, tx = model(
                state, cond, mask, 1, return_trajectory=True, **kw_x
            )
            fx = tx[-1]
        else:
            fx = fx_ref  # detached reference，省一次 forward

        # f(x+ε)：含梯度 forward
        _, txe = model(
            state_eps, cond, mask, 1, return_trajectory=True, **kw_xe
        )

        raw = jacobian_amp_loss(
            fx,
            txe[-1],
            eps,
            mask,
            gamma=float(cfg.jac_gamma),
            gamma_low=float(cfg.jac_gamma_low),
            jaws_enabled=bool(cfg.jaws_enabled),
            jaws_lambda_c=float(cfg.jaws_lambda_c),
        )

        # 此處是四個可觀測性欄位（jac_gate_calls / jac_active_calls /
        # last_jac_ratio / last_jac_raw）的寫入點。需要這些欄位的原因：本項是
        # hinge `relu(ratio−γ)+relu(γ_low−ratio)`，值為 0 有兩種完全不同的意思
        # ——「約束已滿足（算子確實收縮）」與「該步根本沒算」。只看 `jac_reg`
        # 這一欄無法區分，而論文對收縮性的歸因正繫於此。
        with torch.no_grad():
            m = mask
            num = ((txe[-1] - fx) * m).flatten(1).norm(dim=1)
            den = (eps * m).flatten(1).norm(dim=1).clamp_min(1e-8)
            self.jac_gate_calls += 1
            self.last_jac_ratio = float((num / den).mean())
            self.last_jac_raw = float(raw)
            if self.last_jac_raw > 0.0:
                self.jac_active_calls += 1
        return raw

    # ------------------------------------------------------------------
    # 私有工具方法（從 trainer 移植的 helper，避免循環依賴）
    # ------------------------------------------------------------------

    def _uses_per_step_delta_ctu(self) -> bool:
        """判斷 model.forward 是否應接收 per-step delta_ctu_seq。

        time_conditioned=True，或（arch=h_multirate 且 cond_dim=19）時為 True。
        與 TESTrainer._uses_per_step_delta_ctu 邏輯完全一致。

        Returns:
            bool：True 表示訓練步需組裝並傳遞 delta_ctu_seq。
        """
        cfg = self.ctx.cfg
        return bool(
            cfg.time_conditioned
            or (
                cfg.model.arch == "h_multirate"
                and cfg.model.cond_dim == 19
            )
        )

    def _get_prior_vel_norm(self) -> torch.Tensor:
        """HF-1d：均勻入口 momentum 的 normalized 值（tanh 正規化，cache）。

        raw jx_inlet = rho(≈1)·U_inlet(0.05)、jy_inlet = 0；
        以 physics normalizer tanh((raw−M_REF)/S_combined) 轉至 model state 空間。
        結果在第一次呼叫時計算並快取。

        Returns:
            Tensor shape [2]：[jx_inlet_norm, jy_inlet_norm]。
        """
        cached = getattr(self, "_prior_vel_norm_cache", None)
        if cached is None:
            # loss_fn._get_normalizer() 回傳 MRT physics normalizer（含 _mf_mean/_mf_std）
            nz = self.ctx.loss_fn._get_normalizer()
            mref, scomb = nz._mf_mean, nz._mf_std  # [9]
            u_inlet = 0.05                          # lattice U_inlet（P_vec[1] 固定）
            jx_n = torch.tanh((u_inlet - mref[3]) / scomb[3])
            jy_n = torch.tanh((0.0 - mref[5]) / scomb[5])
            cached = torch.stack([jx_n, jy_n]).to(self.ctx.device).float()
            self._prior_vel_norm_cache = cached
        return cached
