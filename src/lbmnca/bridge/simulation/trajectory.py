"""
DVR Trajectory Buffer。

儲存 NCA 推論產生的物理場幀序列（PyTorch CUDA tensor）。
支援即時 push（live 模式）和任意幀 get（seek 模式）。
"""
from __future__ import annotations
import torch


class TrajectoryBuffer:
    """NCA 物理幀的滑窗環形緩衝區（PyTorch CUDA tensor）。

    buffer 槽位 = 9ch MRT physics，shape=(9, H, W)。所有操作均在 GPU 上，
    不走 CPU。

    滑窗語意（UI-04：TURB_LIVE 無限 rollout）：
        - 容量 ``capacity``（config simulation.live_window_frames，預設 256；
          9×360×640 fp32 ×256 ≈ 2.1 GB VRAM）。
        - ``count`` 為已 push 總幀數（無上界）；超過容量後最舊幀被覆蓋，
          可取回的視窗 = 最近 ``n_frames`` 幀。
        - ``get_frame(rel_idx)`` 以**窗內相對索引** [0, n_frames-1] 取幀
          （0=窗內最舊、n_frames-1=最新）——timeline/playback 介面不變。
    """

    MAX_FRAMES_COLD: int = 25  # cold-start Fibonacci 快照數（≈ 30 CTU）
    MAX_FRAMES_WARM: int = 22  # warm-start Fibonacci 快照數（≈ 5 CTU）

    def __init__(
        self, H: int = 360, W: int = 640, device: str = "cuda",
        capacity: int | None = None,
    ) -> None:
        self.H = H
        self.W = W
        self.device = device
        cap = capacity if capacity else max(self.MAX_FRAMES_COLD, self.MAX_FRAMES_WARM)
        # 預分配最大容量，避免動態 realloc
        self.buffer = torch.zeros(
            cap, 9, H, W,
            dtype=torch.float32,
            device=device,
        )
        self.count: int = 0   # 已 push 的幀數（不限上界，用 % 取模）
        self.mode: str = "cold"  # "cold" | "warm" | "live"

    def push(self, physics_9ch: torch.Tensor) -> int:
        """存入一幀 physics tensor，回傳存入的 buffer index。

        physics_9ch: [9, H, W] CUDA tensor（NCA step_one() 的輸出）。
        使用 detach() 防止梯度圖保留。
        """
        idx = self.count % len(self.buffer)
        self.buffer[idx] = physics_9ch.detach()
        self.count += 1
        return idx

    def get_frame(self, rel_idx: int) -> torch.Tensor:
        """取窗內第 rel_idx 幀（clamp 至 [0, n_frames-1]），CUDA tensor 供 DLPack。

        rel_idx=0 為窗內最舊保留幀、n_frames-1 為最新；count ≤ capacity 時
        與絕對索引一致（舊行為不變）。
        """
        n = self.n_frames
        if n == 0:
            return self.buffer[0]
        rel = max(0, min(rel_idx, n - 1))
        abs_idx = self.count - n + rel
        return self.buffer[abs_idx % len(self.buffer)]

    def reset(self, mode: str = "cold") -> None:
        """重置 buffer（新分析開始前呼叫）。不清零 GPU 記憶體（push 會覆蓋）。"""
        self.count = 0
        self.mode = mode

    @property
    def n_frames(self) -> int:
        """目前 buffer 中有效幀數。"""
        return min(self.count, len(self.buffer))

    @property
    def is_empty(self) -> bool:
        return self.count == 0

    @property
    def max_frames(self) -> int:
        return self.MAX_FRAMES_COLD if self.mode == "cold" else self.MAX_FRAMES_WARM
