"""SimulationSession：持有整個模擬會話的狀態。"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SimPhase(Enum):
    IDLE = "idle"
    COLD_START = "cold_start"
    WARM_START = "warm_start"
    # LC-decompose 兩階段（backend=decompose；PLAN §5 handoff）
    MEAN_CONVERGING = "mean_converging"  # Phase 1：mean_module rollout 至 KE 兩窗收斂
    TURB_LIVE = "turb_live"  # Phase 2：turb 無限 rollout（無幀上限，滑窗緩衝）
    COMPLETED = "completed"


@dataclass
class SimulationSession:
    """整個模擬會話的狀態容器。

    此 dataclass 由 main.py 建立後傳給 run_ggui()。
    GGUI 主循環直接讀寫這些欄位。
    WebSocket commands 透過 CommandQueue 更新這些欄位。
    """

    # ── Simulation phase ─────────────────────────────────────────
    sim_phase: SimPhase = SimPhase.IDLE
    generation_head: int = 0  # NCA 已生成幀數

    # ── Playback state ────────────────────────────────────────────
    playback_mode: str = "live"  # "live" | "seek" | "slow_mo"
    playback_head: float = 0.0  # 顯示位置（小數，用於插值）
    playback_speed: float = 1.0  # 播放速度係數
    total_frames: int = 0  # buffer 中總幀數

    # ── IBFV / PlaybackDirector（smooth_transition Phase A）──────────
    # director_rate：playback clock 速率（<1 慢放）；由 [ / ] 鍵調整，
    # 主迴圈每幀套用到 PlaybackDirector.rate。預設 None = 用 config 初值。
    director_rate: float | None = None

    # ── 顯示層 inlet 邊界修正（純 render-layer，不回饋模型/warm chain）────
    # inlet_fix_width：col0 起算的固定值填補寬度（px），由 -/= 鍵調整。
    # feather 寬度由 window._border_fix() 依 inlet_fix_width 自動導出，不單獨存狀態。
    # ★2026-07-23：10px 不夠寬。實測 vorticity 逐 pixel 掃描（C1-1_Cold_M300C300_c0018）
    # 顯示真正的尖峰落在 col≈31-32（inlet_fix_width=10+feather=5=15 之外），即該尖峰
    # 不是 _border_fix 自己的接縫，是模型 rollout 近 inlet 邊界本身殘留的窄欄偽影
    # （原始資料就有，不是填補/羽化造成）。10px 沒蓋到，改 34px（+feather 17px＝
    # 覆蓋到 col 51，含餘裕）。見 docs/audit/0723/IBFV_STREAMLINE_INLET_FIX_LOG_2026-07-23.md。
    inlet_fix_width: int = 34

    # ── Geometry + physics ────────────────────────────────────────
    wind_angle: float = 0.0  # 使用者設定的風向角度（度）
    re_level: int = 2  # 1=150, 2=300, 3=500
    active_polygons: list[list[list[float]]] | None = (
        None  # GGUI 記憶體中的多邊形（歸一化 [0, 1] 座標）
    )
    drag_state: dict[str, Any] | None = (
        None  # 滑鼠拖曳狀態 {"idx": int, "last_x": float, "last_y": float}
    )

    # ── Viz ──────────────────────────────────────────────────────
    viz_mode: str = (
        "velocity"  # "velocity"|"vorticity"|"pressure"|"streamline"|"comfort"
    )

    # ── Version history ───────────────────────────────────────────
    versions: list[dict[str, Any]] = field(default_factory=list)
    max_versions: int = 10  # 最多保留 10 個版本

    # ── Runtime objects ──────────────────────────────────────────
    backend: Any = None  # src.integration.model_backend.ModelBackend
    trajectory: Any = None  # src.simulation.trajectory.TrajectoryBuffer
    config: Any = None  # app config dict
    guard: Any = None  # src.simulation.convergence_guard.ConvergenceGuard
    tier: int = 0  # current degradation tier (0 best .. 3 fallback)

    # ── 警示（分布守門 / G_vec fallback 等；GGUI overlay 顯示）─────
    warnings: list[str] = field(default_factory=list)

    def save_version(self, label: str = "") -> int:
        """儲存當前版本快照（幾何 + phase metadata）。

        回傳版本 index（0-based）。
        若超過 max_versions，移除最舊的版本。
        """
        version = {
            "index": len(self.versions),
            "label": label or f"Version {len(self.versions) + 1}",
            "wind_angle": self.wind_angle,
            "re_level": self.re_level,
            "total_frames": self.total_frames,
            "generation_head": self.generation_head,
        }
        self.versions.append(version)
        if len(self.versions) > self.max_versions:
            self.versions.pop(0)
        return version["index"]

    def get_re_value(self) -> int:
        """將 re_level (1/2/3) 轉為實際 Re 數 (150/300/500)。"""
        mapping = {1: 150, 2: 300, 3: 500}
        return mapping[self.re_level]
