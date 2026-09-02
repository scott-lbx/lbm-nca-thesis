"""lbmnca.nca.tes.monitoring — mean / turb rollout 監控組裝層。

mean / turb 是兩條獨立 pipeline，**共用** roll 核心（``eval.mean_rollout_vis.build_rollout_seq_cols``），
**不同** 在 (a) compose_fn（turb 多 ū+u′）、(b) converged 判定、(c) 時空繪圖函式。

本套件為「組裝層」：把共用核心（roll + masked metric + 組欄）與 module-專屬繪圖
（mean 多 Re 序列 / turb ū+u′ + 時空 panel）接起來，供 async 監控腳本與（未來）
``MeanTrainer`` 的 ``save_monitor_vis`` hook 呼叫。

★additive：本套件不取代既有 ``trainer._rollout_vis_mean`` / ``_rollout_vis_turb``，
亦不改 ``eval/`` 既有 API；僅提供新的對稱組裝入口。

對外：
    - ``mean_vis.build_mean_sample`` / ``mean_vis.render_mean_vis``
"""
from __future__ import annotations

from lbmnca.nca.tes.monitoring import mean_vis

__all__ = ["mean_vis"]
