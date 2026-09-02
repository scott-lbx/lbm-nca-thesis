"""通量式散度算子 `divergence_flux` 與核心遮罩 `interior_core_mask` 的純計算單元測試。

全部為 CPU 純張量運算，不依賴 GPU / 真 model / dataset（DEEP_AUDIT R5 round-2）。
守住的性質：
    - 解析上無散度的均勻流：**核心區**讀數為 0（通量式與 Sobel 皆然）；
      而**全域**讀數不為 0——那是開放域邊界處理慣例產生的人工項，不是離散化差異。
    - 2Δ（Nyquist）checkerboard：通量式核心區增益為 2、中央差分（Sobel/8）為 0。
    - torch 實作與後向差分定義（numpy 參考）逐點等值。
    - ★torch 實作與 docstring 宣稱等價的**兩個出處**逐點等值：
      ``docs/audit/0714/probes/pb_helmholtz_project.py::divergence_backward`` 與
      ``the probe tooling (not shipped): warmchain_stability_probe.py::divergence_flux_np``
      （後者為 round-5 補；先前只守住前者）。
    - 形狀不一致直接 ``ValueError``（禁靜默廣播）。
    - ``compute_all_metrics`` 預設不輸出任何 flux key（既有判準零影響）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lbmnca.nca.tes.eval.metrics import (
    compute_all_metrics,
    mass_conservation_residual_flux,
)
from lbmnca.nca.tes.utils.spatial_ops import divergence_flux, interior_core_mask

_H, _W = 64, 96
_AMP = 0.01


def _fluid_with_building() -> torch.Tensor:
    """回傳 ``[1, 1, H, W]`` 流體遮罩，中間挖一塊矩形固體。"""
    fluid = torch.ones(1, 1, _H, _W)
    fluid[:, :, 20:40, 30:50] = 0.0
    return fluid


def _uniform_flow(fluid: torch.Tensor) -> torch.Tensor:
    """回傳 ``[1, 9, H, W]`` 物理場：jx 均勻 0.1、其餘為 0（解析上 div ≡ 0）。"""
    physics = torch.zeros(1, 9, _H, _W)
    physics[:, 3] = 0.1 * fluid[:, 0]
    return physics


def test_uniform_flow_core_is_zero_full_is_not() -> None:
    """無散度均勻流：核心區兩算子皆 0；全域讀數非 0（＝邊界人工項）。"""
    fluid = _fluid_with_building()
    out = mass_conservation_residual_flux(_uniform_flow(fluid), fluid)
    assert out["mass_residual_flux_core_mean"] == pytest.approx(0.0, abs=1e-12)
    assert out["mass_residual_sobel_core_mean"] == pytest.approx(0.0, abs=1e-12)
    # 全域讀數必須明顯非零，否則本測試就守不住「不可用 full 版解讀低估倍率」這件事
    assert out["mass_residual_flux_full_mean"] > 1e-4
    # 最大值恰為單格 |j|（域邊界整條入流動量被當成殘差）
    assert out["mass_residual_flux_full_max"] == pytest.approx(0.1, rel=1e-6)


def test_nyquist_checkerboard_gain() -> None:
    """2Δ checkerboard：通量式核心增益 = 2×amp、中央差分（Sobel/8）為 0。"""
    yy, xx = torch.meshgrid(torch.arange(_H), torch.arange(_W), indexing="ij")
    physics = torch.zeros(1, 9, _H, _W)
    physics[:, 3] = _AMP * ((-1.0) ** (xx + yy).float())
    out = mass_conservation_residual_flux(physics, torch.ones(1, 1, _H, _W))
    assert out["mass_residual_flux_core_mean"] == pytest.approx(2.0 * _AMP, rel=1e-6)
    assert out["mass_residual_sobel_core_mean"] == pytest.approx(0.0, abs=1e-9)


def test_matches_backward_difference_reference() -> None:
    """torch 實作與後向差分參考定義（numpy）逐點等值。"""
    gen = torch.Generator().manual_seed(0)
    fluid = _fluid_with_building()[0, 0]
    jx = torch.randn(_H, _W, generator=gen).double()
    jy = torch.randn(_H, _W, generator=gen).double()
    got = divergence_flux(jx, jy, fluid.double()).numpy()

    f = fluid.numpy().astype(bool)
    jx_n, jy_n = jx.numpy(), jy.numpy()
    fw = np.zeros_like(f); fw[:, 1:] = f[:, 1:] & f[:, :-1]
    fs = np.zeros_like(f); fs[1:, :] = f[1:, :] & f[:-1, :]
    jx_w = np.zeros_like(jx_n); jx_w[:, 1:] = jx_n[:, :-1]
    jy_s = np.zeros_like(jy_n); jy_s[1:, :] = jy_n[:-1, :]
    ref = np.zeros_like(jx_n)
    ref[f] = (np.where(fw, jx_n - jx_w, jx_n) + np.where(fs, jy_n - jy_s, jy_n))[f]
    assert np.abs(got - ref).max() == pytest.approx(0.0, abs=1e-15)


_ORIGIN_PROBE: Path = (
    WORKSPACE_ROOT / "docs" / "audit" / "0714" / "probes" / "pb_helmholtz_project.py"
)
_WARMCHAIN_PROBE: Path = (
    WORKSPACE_ROOT / "lbmnca.bridge" / "tools" / "probes" / "warmchain_stability_probe.py"
)


def _load_np_function(path: Path, name: str) -> Callable[..., np.ndarray]:
    """從指定檔案**只**抽出單一函式並編譯，不 import 該模組。

    為什麼不直接 import：兩支探針在 module level 都會改 ``sys.path``、載入
    ``torch``/``zarr``/``scipy``、並牽出整條 lbmnca.bridge backend 鏈。本函式改以
    AST 取出單一 ``FunctionDef`` 後 ``exec`` 於只含 ``np`` 的隔離命名空間，
    故無任何 import 副作用。

    Args:
        path: 原始出處檔案路徑。
        name: 要抽出的頂層函式名。

    Returns:
        該函式物件（簽章 ``(jx, jy, fluid) -> np.ndarray``）。

    Raises:
        ValueError: 原始檔內找不到該函式定義（＝出處已改名或移除，宣稱需重寫）。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            namespace: dict[str, object] = {"np": np}
            exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
            return namespace[name]  # type: ignore[return-value]
    raise ValueError(
        f"{path} 內找不到 {name}——"
        "spatial_ops.divergence_flux docstring 的跨 repo 等價宣稱已失實，請改寫。"
    )


@pytest.mark.skipif(
    not _ORIGIN_PROBE.exists(),
    reason=f"原始出處不在此 checkout：{_ORIGIN_PROBE}",
)
def test_matches_origin_probe_definition() -> None:
    """torch ``divergence_flux`` 與**原始出處**的 ``divergence_backward`` 逐點等值。

    守的是 ``spatial_ops.divergence_flux`` docstring 裡「數值上與
    ``pb_helmholtz_project.py::divergence_backward`` 一致」這條跨 repo 宣稱本身，
    而非測試檔內就地重寫的參考實作（後者只能守自己）。
    """
    origin = _load_np_function(_ORIGIN_PROBE, "divergence_backward")
    gen = torch.Generator().manual_seed(20260728)
    fluid = _fluid_with_building()[0, 0]
    jx = torch.randn(_H, _W, generator=gen).double()
    jy = torch.randn(_H, _W, generator=gen).double()

    got = divergence_flux(jx, jy, fluid.double()).numpy()
    ref = origin(jx.numpy(), jy.numpy(), fluid.numpy().astype(bool))

    assert np.abs(got - ref).max() == pytest.approx(0.0, abs=1e-15)


@pytest.mark.skipif(
    not _WARMCHAIN_PROBE.exists(),
    reason=f"跨 repo 探針不在此 checkout：{_WARMCHAIN_PROBE}",
)
def test_matches_warmchain_probe_definition() -> None:
    """torch ``divergence_flux`` 與 03 端探針 ``divergence_flux_np`` 逐點等值。

    守的是 ``spatial_ops.divergence_flux`` docstring 裡等價宣稱的**另一半**
    （round-5：round-4 只守住 ``pb_helmholtz_project::divergence_backward``，
    跨 repo 的 numpy 版仍無測試守）。
    """
    probe = _load_np_function(_WARMCHAIN_PROBE, "divergence_flux_np")
    gen = torch.Generator().manual_seed(20260728)
    fluid = _fluid_with_building()[0, 0]
    jx = torch.randn(_H, _W, generator=gen).double()
    jy = torch.randn(_H, _W, generator=gen).double()

    got = divergence_flux(jx, jy, fluid.double()).numpy()
    ref = probe(jx.numpy(), jy.numpy(), fluid.numpy().astype(bool))

    assert np.abs(got - ref).max() == pytest.approx(0.0, abs=1e-15)


def test_shape_mismatch_raises() -> None:
    """形狀不一致必須報錯，不得靜默廣播成 ``[B, B, H, W]``。"""
    with pytest.raises(ValueError):
        divergence_flux(torch.randn(2, 8, 8), torch.randn(2, 8, 8), torch.ones(2, 1, 8, 8))


def test_interior_core_mask_properties() -> None:
    """核心遮罩：為 fluid 子集、域邊界全 0、``erode_iters<=0`` 為 identity。"""
    fluid = _fluid_with_building()
    core = interior_core_mask(fluid, 2)
    assert core.shape == fluid.shape
    assert int(((core > 0.5) & (fluid < 0.5)).sum()) == 0
    assert float(core[:, :, 0, :].sum()) == 0.0
    assert float(core[:, :, -1, :].sum()) == 0.0
    assert float(core[:, :, :, 0].sum()) == 0.0
    assert float(core[:, :, :, -1].sum()) == 0.0
    assert torch.equal(interior_core_mask(fluid, 0), fluid)


def test_empty_core_returns_nan() -> None:
    """核心遮罩為空時回 nan（不以 0 冒充「殘差為零」）。"""
    out = mass_conservation_residual_flux(
        torch.zeros(1, 9, 4, 4), torch.ones(1, 1, 4, 4), core_erode_iters=3
    )
    assert np.isnan(out["mass_residual_flux_core_mean"])
    assert np.isnan(out["mass_residual_sobel_core_max"])


def test_compute_all_metrics_default_has_no_flux_keys() -> None:
    """向後相容守衛：預設不輸出任何 flux/sobel 對照 key。"""
    fluid = _fluid_with_building()
    traj = torch.zeros(1, 2, 9, _H, _W)
    ic = torch.zeros(1, 9, _H, _W)
    default_keys = set(compute_all_metrics(traj, traj, ic, fluid))
    assert not any("flux" in k or "sobel" in k for k in default_keys)
    enabled = compute_all_metrics(traj, traj, ic, fluid, include_flux_divergence=True)
    assert set(enabled) - default_keys == {
        "mass_residual_flux_core_mean",
        "mass_residual_flux_core_max",
        "mass_residual_sobel_core_mean",
        "mass_residual_sobel_core_max",
        "mass_residual_flux_full_mean",
        "mass_residual_flux_full_max",
    }
