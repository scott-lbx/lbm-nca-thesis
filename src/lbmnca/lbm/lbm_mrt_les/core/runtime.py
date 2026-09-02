import taichi as ti


_RUNTIME_INITIALIZED = False


def ensure_taichi_runtime(
    arch=ti.gpu,
    device_memory_fraction: float = 0.8,
    log_level=ti.INFO,
) -> None:
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    ti.init(
        arch=arch,
        device_memory_fraction=device_memory_fraction,
        log_level=log_level,
        offline_cache=True,
    )
    _RUNTIME_INITIALIZED = True


def reset_taichi_runtime() -> None:
    global _RUNTIME_INITIALIZED
    if not _RUNTIME_INITIALIZED:
        return
    ti.reset()
    _RUNTIME_INITIALIZED = False
