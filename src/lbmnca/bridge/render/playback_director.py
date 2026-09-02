"""Playback Director — 時間重映射（compute clock 與 playback clock 分離）。

設計動機（compute clock vs playback clock）
-------------------------------------------
NCA 推論產生「物理幀」的節奏（**compute clock**）由模型一步一步推進決定：
mean 收斂可能 8~200 步、turb live 則無限 rollout。但「畫面播放」的節奏
（**playback clock**）應由觀感需求驅動——慢放收斂過程、迴圈一個 turb cycle、
換幾何時 crossfade。兩個時鐘**不該耦合**：compute 端只管 ``push`` 新幀進序列，
playback 端用浮點 ``playhead`` 在序列上自由前進／回放／插值。

本類別把 :func:`src.render.interpolate.lerp_frames` 的「兩幀線性插值」語意泛化成
**連續時間軸上的取樣**：``playhead`` 落在 ``[floor, ceil]`` 之間時，輸出
``lerp(buf[floor], buf[ceil], frac)``。``lerp_frames`` 本身是 GPU Taichi kernel
（吃 ``ti.template()`` field、寫入 out field）；本 Director 為**純 Python**，不呼叫
``ti.init``、不直接依賴 Taichi——而是把實際插值委派給注入的 ``lerp_fn``，
預設為 torch/numpy 通用的逐元素 lerp。主迴圈若要走 GPU，可把包裝過的
``lerp_frames`` 當 ``lerp_fn`` 注入（見模組末用法範例）。

用途對應（EXECUTION_PLAN §2.5 / §3.2）
--------------------------------------
- **mean 收斂**：有限序列 + ``rate < 1`` 慢放 + ``loop_mode="none"``，播完 ``hold``
  在最後一幀（KE 兩窗收斂敘事可慢慢看）。
- **turb live**：對一個穩定 cycle 設 ``loop_mode="loop"``（或 ``"pingpong"``）無接縫迴圈。
- **換幾何**：``interrupt(new_seq)`` 觸發 crossfade，從舊序列當前輸出淡入新序列。
"""
from __future__ import annotations

import math
from typing import Any, Callable, List, Optional

Frame = Any  # torch.Tensor 或 numpy.ndarray，shape 慣例 (9, H, W)
LerpFn = Callable[[Frame, Frame, float], Frame]


def _default_lerp(fa: Frame, fb: Frame, alpha: float) -> Frame:
    """預設逐元素線性插值（泛化 ``lerp_frames`` 的 CPU/GPU-tensor 版本）。

    語意與 :func:`src.render.interpolate.lerp_frames` 完全一致：
    ``alpha=0.0`` 回 ``fa``、``alpha=1.0`` 回 ``fb``。對 torch.Tensor 與
    numpy.ndarray 皆有效（兩者皆支援 ``*`` 與 ``+`` 廣播）。

    Args:
        fa: 前一幀（shape (9, H, W)）。
        fb: 後一幀（同 shape）。
        alpha: 插值權重，已假設落在 [0.0, 1.0]。

    Returns:
        插值後的新幀（與輸入同型別／同 device）。
    """
    return fa * (1.0 - alpha) + fb * alpha


def _ease(t: float, mode: str) -> float:
    """對正規化進度 ``t∈[0,1]`` 套用 easing 曲線。

    Args:
        t: 線性進度（0.0~1.0）。
        mode: ``"linear"`` | ``"ease-in"`` | ``"ease-out"`` | ``"ease-in-out"``。

    Returns:
        重映射後的進度（仍落在 [0,1]）。

    Raises:
        ValueError: ``mode`` 非支援值。
    """
    if mode == "linear":
        return t
    if mode == "ease-in":
        return t * t
    if mode == "ease-out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if mode == "ease-in-out":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    raise ValueError(f"未知的 easing 模式：{mode!r}")


class PlaybackDirector:
    """時間重映射播放器：在物理幀序列上以浮點 playhead 取樣並插值。

    把離散的 compute-clock 幀序列重映射到連續的 playback-clock 時間軸。
    序列以內部 list 持有（``push`` 累加），``playhead`` 為浮點索引，
    ``advance(dt)`` 依 ``rate`` 前進，``current()`` 回傳 playhead 兩側幀的
    lerp 結果。實際 lerp 委派給 ``lerp_fn``（預設純 torch/numpy；可注入
    GPU 版 ``lerp_frames``）。

    Attributes:
        rate: 播放速率（playback 幀／compute 幀）。``<1`` 慢放、``1`` 原速、
            ``>1`` 快放。
        easing: easing 模式，作用於**單一相鄰幀區間內**的 ``frac``，
            令幀間過渡有加減速感。
        loop_mode: ``"none"``（播完 hold 末幀）｜``"loop"``（回到頭）｜
            ``"pingpong"``（往返）。
        crossfade_frames: ``interrupt`` 時跨多少「playback 步」完成新舊序列淡入。
    """

    def __init__(
        self,
        rate: float = 1.0,
        easing: str = "linear",
        loop_mode: str = "none",
        crossfade_frames: float = 8.0,
        lerp_fn: Optional[LerpFn] = None,
    ) -> None:
        """初始化 Director。

        Args:
            rate: 播放速率，必須 ``> 0``。``<1`` 慢放（mean 收斂用）。
            easing: ``"linear"`` | ``"ease-in"`` | ``"ease-out"`` | ``"ease-in-out"``。
            loop_mode: ``"none"`` | ``"loop"`` | ``"pingpong"``。
            crossfade_frames: ``interrupt`` crossfade 的 playback 步長，``> 0``。
            lerp_fn: 兩幀插值函式，簽名 ``(fa, fb, alpha) -> frame``；
                預設 :func:`_default_lerp`（純 torch/numpy）。主迴圈可注入
                包裝過的 GPU :func:`src.render.interpolate.lerp_frames`。

        Raises:
            ValueError: ``rate`` 或 ``crossfade_frames`` 非正、``loop_mode``
                非支援值。
        """
        if rate <= 0.0:
            raise ValueError(f"rate 必須 > 0，收到 {rate}")
        if crossfade_frames <= 0.0:
            raise ValueError(f"crossfade_frames 必須 > 0，收到 {crossfade_frames}")
        if loop_mode not in ("none", "loop", "pingpong"):
            raise ValueError(f"未知的 loop_mode：{loop_mode!r}")
        # easing 在 _ease 內驗證（first current() 時才觸發；提前驗證較友善）
        _ease(0.0, easing)

        self.rate: float = rate
        self.easing: str = easing
        self.loop_mode: str = loop_mode
        self.crossfade_frames: float = crossfade_frames
        self._lerp_fn: LerpFn = lerp_fn if lerp_fn is not None else _default_lerp

        self._buf: List[Frame] = []
        self.playhead: float = 0.0
        self._direction: int = 1  # pingpong 用：+1 前進、-1 後退
        self._finished: bool = False  # loop_mode="none" 播完末幀後為 True

        # crossfade 狀態（interrupt 後啟用）
        self._fade_from: Optional[Frame] = None  # 舊序列當下輸出（凍結快照）
        self._fade_progress: float = 0.0  # 0→1，達 1 結束 crossfade

    # ------------------------------------------------------------------ #
    # 序列維護（compute clock 端呼叫）
    # ------------------------------------------------------------------ #
    def push(self, frame: Frame) -> int:
        """把一個 compute-clock 物理幀附加到序列尾端。

        compute 端（主迴圈每步推論後）呼叫，與 playback 節奏無關。

        Args:
            frame: 物理場幀（shape (9, H, W) 的 torch/numpy field）。
                呼叫端負責 ``detach``（與 ``TrajectoryBuffer.push`` 相同約定）。

        Returns:
            附加後的序列長度。
        """
        self._buf.append(frame)
        return len(self._buf)

    def clear(self) -> None:
        """清空序列並重置 playhead（新分析開始前呼叫）。"""
        self._buf.clear()
        self.playhead = 0.0
        self._direction = 1
        self._finished = False
        self._fade_from = None
        self._fade_progress = 0.0

    # ------------------------------------------------------------------ #
    # 時間推進（playback clock 端呼叫）
    # ------------------------------------------------------------------ #
    def advance(self, dt: float = 1.0) -> None:
        """以 ``rate`` 推進 playback clock。

        ``dt`` 為一個 playback tick（通常主迴圈每幀傳 ``1.0``，或傳實際
        frame time 做時基無關播放）。playhead 前進量 = ``dt * rate``，
        再依 ``loop_mode`` 環繞／反彈／停在末幀。crossfade 進行中同步推進。

        Args:
            dt: playback tick 長度（預設 1.0）。
        """
        if self._fade_from is not None:
            self._fade_progress += dt / self.crossfade_frames
            if self._fade_progress >= 1.0:
                self._fade_from = None
                self._fade_progress = 0.0

        n = len(self._buf)
        if n <= 1:
            self.playhead = 0.0
            return

        last = float(n - 1)
        self.playhead += dt * self.rate * self._direction

        if self.loop_mode == "none":
            if self.playhead >= last:
                self.playhead = last
                self._finished = True  # hold 末幀
            elif self.playhead < 0.0:
                self.playhead = 0.0
        elif self.loop_mode == "loop":
            # 環繞回頭（保留小數相位）
            period = last  # 幀 0..last，loop 週期 = last（last 與 0 視為接縫）
            self.playhead = self.playhead % period if period > 0.0 else 0.0
        elif self.loop_mode == "pingpong":
            # 在 [0, last] 之間反彈
            if self.playhead > last:
                self.playhead = last - (self.playhead - last)
                self._direction = -1
            elif self.playhead < 0.0:
                self.playhead = -self.playhead
                self._direction = 1

    def seek(self, playhead: float) -> None:
        """直接設定 playhead（scrub／timeline 拖曳用）。

        Args:
            playhead: 目標浮點索引；自動 clamp 至 ``[0, n_frames-1]``。
        """
        n = len(self._buf)
        if n == 0:
            self.playhead = 0.0
            return
        self.playhead = max(0.0, min(playhead, float(n - 1)))
        self._finished = self.loop_mode == "none" and self.playhead >= float(n - 1)

    # ------------------------------------------------------------------ #
    # 取樣（render 端呼叫）
    # ------------------------------------------------------------------ #
    def current(self) -> Optional[Frame]:
        """回傳 playhead 當前位置的插值幀（泛化 ``lerp_frames``）。

        取 ``floor``/``ceil`` 兩側幀，以 easing 後的 ``frac`` 委派 ``lerp_fn`` 插值。
        若 crossfade 進行中，再把結果與凍結的舊序列輸出做二次 lerp。

        Returns:
            插值後物理幀（與序列元素同型別／device）；序列為空時回 ``None``。
        """
        n = len(self._buf)
        if n == 0:
            return None
        if n == 1:
            sampled = self._buf[0]
        else:
            lo = int(math.floor(self.playhead))
            lo = max(0, min(lo, n - 1))
            hi = min(lo + 1, n - 1)
            frac = self.playhead - float(lo)
            frac = min(max(frac, 0.0), 1.0)
            if hi == lo or frac == 0.0:
                sampled = self._buf[lo]
            else:
                sampled = self._lerp_fn(self._buf[lo], self._buf[hi], _ease(frac, self.easing))

        if self._fade_from is not None:
            # crossfade：舊輸出 → 新序列輸出
            alpha = min(max(self._fade_progress, 0.0), 1.0)
            return self._lerp_fn(self._fade_from, sampled, alpha)
        return sampled

    # ------------------------------------------------------------------ #
    # 中斷／換幾何（crossfade）
    # ------------------------------------------------------------------ #
    def interrupt(self, new_seq: Optional[List[Frame]] = None) -> None:
        """中斷當前播放、切換到新序列，並啟動 crossfade。

        換幾何時呼叫：凍結舊序列當前輸出作為淡出來源，替換序列、重置
        playhead，後續 ``current()`` 會在 ``crossfade_frames`` 個 playback 步內
        從舊畫面淡入新序列起點，避免硬切。

        Args:
            new_seq: 新物理幀序列；``None`` 表示清空（淡出至空）。
                傳入後 Director 直接持有該 list（呼叫端勿再原地修改）。
        """
        snapshot = self.current()  # 凍結舊輸出（可能為 None）
        self._buf = list(new_seq) if new_seq is not None else []
        self.playhead = 0.0
        self._direction = 1
        self._finished = False
        if snapshot is not None and self._buf:
            self._fade_from = snapshot
            self._fade_progress = 0.0
        else:
            self._fade_from = None
            self._fade_progress = 0.0

    # ------------------------------------------------------------------ #
    # 狀態查詢
    # ------------------------------------------------------------------ #
    @property
    def n_frames(self) -> int:
        """目前序列中的幀數。"""
        return len(self._buf)

    @property
    def is_empty(self) -> bool:
        """序列是否為空。"""
        return len(self._buf) == 0

    @property
    def is_finished(self) -> bool:
        """``loop_mode="none"`` 是否已播完並 hold 在末幀。"""
        return self._finished

    @property
    def is_crossfading(self) -> bool:
        """是否正在 crossfade（``interrupt`` 後尚未淡入完成）。"""
        return self._fade_from is not None
