"""FrameProfiler — per-stage timing for the GGUI loop.

Wrap each render stage with `with prof.stage("name"):` and call `prof.tick()`
once per frame. Every `report_every` frames it prints rolling-average ms per
stage plus FPS (flushed, so it shows up live in redirected logs).

Set sync=True to torch.cuda.synchronize + ti.sync inside each stage for
accurate GPU attribution (adds overhead — diagnosis only). sync=False measures
wall-clock kernel-launch time (cheap, production-safe).
"""
import time
from contextlib import contextmanager


class _Null:
    """No-op profiler when profiling disabled (zero overhead)."""

    @contextmanager
    def stage(self, name):
        yield

    def tick(self):
        pass


class FrameProfiler:
    def __init__(self, report_every: int = 30, sync: bool = False) -> None:
        self.report_every = max(1, report_every)
        self.sync = sync
        self._acc: dict[str, float] = {}
        self._order: list[str] = []
        self._frame = 0
        self._t_report = time.perf_counter()
        self._sync_fn = self._make_sync()

    def _make_sync(self):
        if not self.sync:
            return None
        try:
            import torch
            import taichi as ti

            def _s():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ti.sync()
            return _s
        except Exception:
            return None

    @contextmanager
    def stage(self, name: str):
        if name not in self._acc:
            self._acc[name] = 0.0
            self._order.append(name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._sync_fn is not None:
                self._sync_fn()
            self._acc[name] += (time.perf_counter() - t0) * 1000.0

    def tick(self) -> None:
        self._frame += 1
        if self._frame % self.report_every != 0:
            return
        now = time.perf_counter()
        wall = now - self._t_report
        fps = self.report_every / wall if wall > 0 else 0.0
        parts = [f"{n}={self._acc[n] / self.report_every:.2f}" for n in self._order]
        total = sum(self._acc.values()) / self.report_every
        print(f"[prof] {fps:5.1f} fps | total {total:5.2f}ms | " + " ".join(parts),
              flush=True)
        for n in self._acc:
            self._acc[n] = 0.0
        self._t_report = now


def make_profiler(config: dict) -> "FrameProfiler | _Null":
    """Build profiler from config['render']: profile (bool), profile_sync (bool)."""
    r = config.get("render", {}) if config else {}
    if not r.get("profile", False):
        return _Null()
    return FrameProfiler(
        report_every=int(r.get("profile_every", 30)),
        sync=bool(r.get("profile_sync", False)),
    )
