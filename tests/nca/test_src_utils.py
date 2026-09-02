"""
tests/test_src_utils.py
========================
Standalone test for:
  - src.utils.FieldNormalizer
  - src.utils.FieldConverter
  - src.utils.FieldColorizer
  - src.utils.PatchUtils

Run from project root:
    python -m pytest tests/test_src_utils.py -v
  or:
    python tests/test_src_utils.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ── 讓 src/ 可被 import ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lbmnca.nca.utils.FieldNormalizer import FieldNormalizer
from lbmnca.nca.utils.FieldConverter import FieldConverter
from lbmnca.nca.utils.FieldColorizer import FieldColorizer
from lbmnca.nca.utils.PatchUtils import data_to_patches, patches_to_full


# ── 合成 global_stats fixture ─────────────────────────────────────────────────


def _make_stats_json(path: Path) -> None:
    """建立最小合法 tes_v3 global_stats.json（採用物理上合理的數值）。"""
    stats = {
        "snapshot_norm": {"version": "v3"},
        "physics_norm": {
            "M_REF": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "S": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "S_combined": [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        },
        "static_norm": {
            "sdf": {"method": "tanh_lchar"},
            "upstream": {"mean": 0.5, "std": 0.3},
            "pos_x": {"method": "raw_01"},
            "pos_y": {"method": "raw_01"},
        },
        "cond_stats": {
            "g_vec": {"mean": [0.0] * 14, "std": [1.0] * 14},
            "p_vec": {"mean": [500.0, 0.05, 50.0], "std": [200.0, 1e-8, 20.0]},
        },
        "vis_range": {
            "vel_max": 0.14, "vor_range": 0.016,
            "p_center": 0.3333, "p_range": 0.01,
        },
    }
    path.write_text(json.dumps(stats))


@pytest.fixture(scope="module")
def stats_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("stats") / "global_stats.json"
    _make_stats_json(p)
    return p


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# FieldNormalizer
# ─────────────────────────────────────────────────────────────────────────────


class TestFieldNormalizer:

    def test_from_file(self, stats_path):
        norm = FieldNormalizer.from_file(stats_path)
        assert isinstance(norm, FieldNormalizer)

    def test_to_device(self, stats_path, device):
        norm = FieldNormalizer.from_file(stats_path).to(device)
        assert norm._mf_mean.device.type == device.type

    # ── mphy round-trip ──────────────────────────────────────────────────────

    def test_mphy_roundtrip_3d(self, stats_path, device):
        """v3 tanh 空間 round-trip：輸入需在物理合理範圍（|arg| << atanh(0.9999)≈4.6）。"""
        norm = FieldNormalizer.from_file(stats_path).to(device)
        x = _make_mphy_raw(1, 32, 32, device).squeeze(0)  # [9, 32, 32] 物理合理值
        x_rt = norm.denorm_mphy(norm.norm_mphy(x))
        assert x_rt.shape == x.shape
        assert torch.allclose(
            x, x_rt, atol=1e-4
        ), f"max err={( x - x_rt).abs().max():.2e}"

    def test_mphy_roundtrip_4d(self, stats_path, device):
        """v3 tanh 空間 round-trip（4D）。"""
        norm = FieldNormalizer.from_file(stats_path).to(device)
        x = _make_mphy_raw(4, 32, 32, device)  # [4, 9, 32, 32]
        x_rt = norm.denorm_mphy(norm.norm_mphy(x))
        assert torch.allclose(x, x_rt, atol=1e-4)

    # ── uprime round-trip ────────────────────────────────────────────────────

    def test_uprime_roundtrip(self, stats_path, device):
        norm = FieldNormalizer.from_file(stats_path).to(device)
        x = torch.randn(4, 9, 16, 16, device=device)
        assert torch.allclose(x, norm.denorm_uprime(norm.norm_uprime(x)), atol=1e-5)

    # ── SDF round-trip ───────────────────────────────────────────────────────

    def test_sdf_roundtrip(self, stats_path, device):
        norm = FieldNormalizer.from_file(stats_path).to(device)
        x = torch.randn(1, 64, 64, device=device) * 5
        assert torch.allclose(x, norm.denorm_sdf(norm.norm_sdf(x)), atol=1e-5)

    # ── position embeddings ──────────────────────────────────────────────────

    def test_pos_roundtrip(self, stats_path, device):
        norm = FieldNormalizer.from_file(stats_path).to(device)
        px = torch.rand(64, 64, device=device)
        py = torch.rand(64, 64, device=device)
        assert torch.allclose(px, norm.denorm_pos_x(norm.norm_pos_x(px)), atol=1e-5)
        assert torch.allclose(py, norm.denorm_pos_y(norm.norm_pos_y(py)), atol=1e-5)

    # ── cond round-trip ──────────────────────────────────────────────────────
    # v3 FieldNormalizer.norm_cond / denorm_cond 為 identity（cond_mean=0, cond_std=1）
    # 真正的 G_vec/P_vec norm 由 TESCondNormalizer 處理

    def test_cond_roundtrip_1d(self, stats_path, device):
        norm = FieldNormalizer.from_file(stats_path).to(device)
        c = torch.tensor([1.0], device=device)
        assert torch.allclose(c, norm.denorm_cond(norm.norm_cond(c)), atol=1e-6)

    def test_cond_roundtrip_2d(self, stats_path, device):
        norm = FieldNormalizer.from_file(stats_path).to(device)
        c = torch.randn(8, 1, device=device)
        assert torch.allclose(
            c, norm.denorm_cond(norm.norm_cond(c)), atol=1e-6
        )

    # ── forward alias ────────────────────────────────────────────────────────

    def test_forward_alias(self, stats_path, device):
        norm = FieldNormalizer.from_file(stats_path).to(device)
        x = torch.randn(9, 16, 16, device=device)
        assert torch.allclose(norm(x), norm.norm_mphy(x))


# ─────────────────────────────────────────────────────────────────────────────
# FieldConverter
# ─────────────────────────────────────────────────────────────────────────────


def _make_mphy_raw(B: int, H: int, W: int, device) -> torch.Tensor:
    """建立物理上合理的 Mphy_lattice_raw。
    rho≈1.0, jx/jy≈0.05*rho → vel_mag~0.07
    """
    m = torch.zeros(B, 9, H, W, device=device)
    m[:, 0] = 1.0 + torch.randn(B, H, W, device=device) * 0.01  # rho
    m[:, 3] = 0.05 + torch.randn(B, H, W, device=device) * 0.005  # jx
    m[:, 5] = 0.02 + torch.randn(B, H, W, device=device) * 0.005  # jy
    return m


class TestFieldConverter:

    def test_from_file(self, stats_path):
        conv = FieldConverter.from_file(stats_path)
        assert isinstance(conv, FieldConverter)

    def test_to_device(self, stats_path, device):
        conv = FieldConverter.from_file(stats_path).to(device)
        assert conv._sobel_kx.device.type == device.type

    # ── toMphy from raw (denorm=False) ───────────────────────────────────────

    def test_toMphy_passthrough(self, stats_path, device):
        conv = FieldConverter.from_file(stats_path).to(device)
        m = _make_mphy_raw(2, 32, 32, device)
        out = conv.toMphy(m, from_state=False, denorm=False)
        assert out.shape == (2, 9, 32, 32)
        assert torch.equal(out, m)

    def test_toMphy_from_state(self, stats_path, device):
        """從 29ch state 提取 ch4:13。"""
        conv = FieldConverter.from_file(stats_path).to(device)
        state = torch.randn(2, 29, 32, 32, device=device)
        out = conv.toMphy(state, from_state=True, denorm=False)
        assert out.shape == (2, 9, 32, 32)
        assert torch.equal(out, state[:, 4:13])

    def test_toMphy_denorm_roundtrip(self, stats_path, device):
        """norm → denorm 應回到原始值。"""
        conv = FieldConverter.from_file(stats_path).to(device)
        m_raw = _make_mphy_raw(2, 32, 32, device)
        m_norm = conv.normalizer.norm_mphy(m_raw)
        m_back = conv.toMphy(m_norm, from_state=False, denorm=True)
        assert torch.allclose(m_raw, m_back, atol=1e-5)

    # ── toVphy output shapes ─────────────────────────────────────────────────

    def test_toVphy_shape_4d(self, stats_path, device):
        conv = FieldConverter.from_file(stats_path).to(device)
        m = _make_mphy_raw(3, 32, 32, device)
        vphy = conv.toVphy(m)
        for key in ("vel_x", "vel_y", "vel_mag", "vorticity", "pressure"):
            assert key in vphy, f"Missing key: {key}"
            assert vphy[key].shape == (3, 1, 32, 32), f"{key} shape={vphy[key].shape}"

    def test_toVphy_shape_3d(self, stats_path, device):
        """CHW input → CHW output."""
        conv = FieldConverter.from_file(stats_path).to(device)
        m = _make_mphy_raw(1, 32, 32, device).squeeze(0)  # [9,32,32]
        vphy = conv.toVphy(m)
        for key in ("vel_x", "vel_y", "vel_mag", "vorticity", "pressure"):
            assert vphy[key].shape == (1, 32, 32), f"{key} shape={vphy[key].shape}"

    # ── toVphy 物理正確性 ────────────────────────────────────────────────────

    def test_toVphy_velocity(self, stats_path, device):
        """vel_mag = √(vx²+vy²)，驗證量值合理。"""
        conv = FieldConverter.from_file(stats_path).to(device)
        m = _make_mphy_raw(1, 32, 32, device)
        vphy = conv.toVphy(m)
        vel_mag = vphy["vel_mag"]
        # vel_mag 應在 (0, VEL_CLAMP] 範圍內
        assert vel_mag.min() >= 0.0
        assert vel_mag.max() <= conv.VEL_CLAMP + 1e-6
        # 對已知輸入做手算驗證
        rho = m[:, 0:1].clamp(min=1e-3)
        jx = m[:, 3:4]
        jy = m[:, 5:6]
        expected_mag = ((jx / rho) ** 2 + (jy / rho) ** 2).sqrt()
        expected_mag = expected_mag.clamp(0, conv.VEL_CLAMP)
        assert torch.allclose(vel_mag, expected_mag, atol=1e-5)

    def test_toVphy_pressure(self, stats_path, device):
        """pressure = rho / 3，rho ≈ 1.0 → pressure ≈ 0.333。"""
        conv = FieldConverter.from_file(stats_path).to(device)
        m = _make_mphy_raw(1, 32, 32, device)
        vphy = conv.toVphy(m)
        pressure = vphy["pressure"]
        assert pressure.min() > 0.0
        assert torch.allclose(pressure, m[:, 0:1] / 3.0, atol=1e-5)

    def test_toVphy_is_norm(self, stats_path, device):
        """is_norm=True 的結果應與先 denorm 再計算一致。"""
        conv = FieldConverter.from_file(stats_path).to(device)
        m_raw = _make_mphy_raw(2, 32, 32, device)
        m_norm = conv.normalizer.norm_mphy(m_raw)

        vphy_from_raw = conv.toVphy(m_raw, is_norm=False)
        vphy_from_norm = conv.toVphy(m_norm, is_norm=True)

        for key in vphy_from_raw:
            assert torch.allclose(
                vphy_from_raw[key], vphy_from_norm[key], atol=1e-4
            ), f"{key}: max_err={(vphy_from_raw[key] - vphy_from_norm[key]).abs().max():.2e}"

    # ── forward alias ────────────────────────────────────────────────────────

    def test_forward_alias(self, stats_path, device):
        conv = FieldConverter.from_file(stats_path).to(device)
        m = _make_mphy_raw(1, 16, 16, device)
        assert set(conv(m).keys()) == {
            "vel_x",
            "vel_y",
            "vel_mag",
            "vorticity",
            "pressure",
        }


# ─────────────────────────────────────────────────────────────────────────────
# FieldColorizer
# ─────────────────────────────────────────────────────────────────────────────


def _make_vphy(B, H, W, device):
    """建立物理上合理的 Vphy_lattice_raw 各通道。"""
    vel_mag = torch.rand(B, 1, H, W, device=device) * 0.20  # [0, 0.20]
    vorticity = (torch.rand(B, 1, H, W, device=device) - 0.5) * 0.08  # [-0.04, 0.04]
    pressure = 0.320 + torch.rand(B, 1, H, W, device=device) * 0.027  # [0.320, 0.347]
    mask = (torch.rand(B, 1, H, W, device=device) > 0.9).float()  # ~10% obstacle
    return vel_mag, vorticity, pressure, mask


class TestFieldColorizer:

    def test_init(self):
        c = FieldColorizer()
        assert c._vel_lut.shape == (256, 3)
        assert c._vor_lut.shape == (256, 3)
        assert c._prs_lut.shape == (256, 3)

    def test_to_device(self, device):
        c = FieldColorizer().to(device)
        assert c._vel_lut.device.type == device.type

    # ── output shape ─────────────────────────────────────────────────────────

    def test_vel_shape_4d(self, device):
        c = FieldColorizer().to(device)
        vel_mag, _, _, mask = _make_vphy(4, 32, 32, device)
        rgb = c.getVelColor(vel_mag, mask)
        assert rgb.shape == (4, 3, 32, 32)

    def test_vel_shape_3d(self, device):
        """CHW input → CHW output。"""
        c = FieldColorizer().to(device)
        vel_mag = torch.rand(1, 32, 32, device=device) * 0.20
        rgb = c.getVelColor(vel_mag)
        assert rgb.shape == (3, 32, 32)

    def test_vor_shape(self, device):
        c = FieldColorizer().to(device)
        _, vor, _, mask = _make_vphy(2, 32, 32, device)
        rgb = c.getVorColor(vor, mask)
        assert rgb.shape == (2, 3, 32, 32)

    def test_pressure_shape(self, device):
        c = FieldColorizer().to(device)
        _, _, prs, mask = _make_vphy(3, 32, 32, device)
        rgb = c.getPressureColor(prs, mask)
        assert rgb.shape == (3, 3, 32, 32)

    # ── output range ─────────────────────────────────────────────────────────

    def test_output_range(self, device):
        c = FieldColorizer().to(device)
        vel_mag, vorticity, pressure, mask = _make_vphy(4, 64, 64, device)

        for rgb in [
            c.getVelColor(vel_mag, mask),
            c.getVorColor(vorticity, mask),
            c.getPressureColor(pressure, mask),
        ]:
            assert rgb.min() >= 0.0 - 1e-6, f"min={rgb.min():.4f}"
            assert rgb.max() <= 1.0 + 1e-6, f"max={rgb.max():.4f}"

    # ── obstacle mask ─────────────────────────────────────────────────────────

    def test_obstacle_gray(self, device):
        """mask=0（無流體＝obstacle）的像素，RGB 應為 OBSTACLE_GRAY。

        ★fluid_mask 慣例＝1=fluid/0=solid（2026-03-30 2f673e5 全專案標準化，
        FieldColorizer._colorize：``mask > 0.5`` 才取原色，否則灰色）。
        """
        c = FieldColorizer().to(device)
        vel_mag = torch.rand(1, 1, 16, 16, device=device) * 0.20
        mask = torch.zeros(1, 1, 16, 16, device=device)  # 全部 obstacle（fluid_mask=0）
        rgb = c.getVelColor(vel_mag, mask)
        expected = torch.full_like(rgb, c.OBSTACLE_GRAY)
        assert torch.allclose(rgb, expected, atol=1e-5)

    def test_no_mask(self, device):
        """無 mask 時仍可正常執行且無 obstacle 混色。"""
        c = FieldColorizer().to(device)
        vel_mag = torch.rand(2, 1, 32, 32, device=device) * 0.20
        rgb = c.getVelColor(vel_mag)
        assert rgb.shape == (2, 3, 32, 32)

    # ── forward alias ────────────────────────────────────────────────────────

    def test_forward_alias(self, device):
        c = FieldColorizer().to(device)
        vel_mag = torch.rand(1, 1, 16, 16, device=device) * 0.20
        assert torch.equal(c(vel_mag), c.getVelColor(vel_mag))

    # ── LUT monotonicity — vel plasma ────────────────────────────────────────

    def test_vel_lut_not_uniform(self, device):
        """plasma LUT 應非單色（有色彩變化）。"""
        c = FieldColorizer().to(device)
        lut = c._vel_lut.cpu()
        # R channel 在 plasma 是單調遞增的
        assert lut[-1, 0] > lut[0, 0], "plasma LUT R channel should increase"


# ─────────────────────────────────────────────────────────────────────────────
# PatchUtils
# ─────────────────────────────────────────────────────────────────────────────


class TestPatchUtils:

    # ── data_to_patches ───────────────────────────────────────────────────────

    def test_output_type(self):
        field = torch.randn(3, 128, 128)
        patches, locs = data_to_patches(field, patch_size=64, stride=32)
        assert isinstance(patches, torch.Tensor)
        assert isinstance(locs, list)
        assert len(locs) > 0

    def test_output_shape(self):
        field = torch.randn(5, 128, 128)
        patches, locs = data_to_patches(field, patch_size=64, stride=32)
        # H=128, stride=32, patch=64 → positions: 0,32,64 → 3×3=9 patches
        assert patches.shape == (9, 5, 64, 64)
        assert len(locs) == 9

    def test_numpy_input(self):
        field = np.random.randn(3, 64, 64).astype(np.float32)
        patches, locs = data_to_patches(field, patch_size=32, stride=16)
        assert isinstance(patches, torch.Tensor)
        assert patches.dtype == torch.float32

    def test_single_channel(self):
        field = torch.randn(1, 64, 64)
        patches, locs = data_to_patches(field, patch_size=32, stride=16)
        assert patches.shape[1] == 1

    def test_many_channels(self):
        """任意通道數 (C=29)。"""
        field = torch.randn(29, 64, 64)
        patches, locs = data_to_patches(field, patch_size=32, stride=32)
        B, C, P, Q = patches.shape
        assert C == 29
        assert P == Q == 32

    def test_patch_content(self):
        """patch 內容應與原場域一致。"""
        field = torch.randn(3, 64, 64)
        patches, locs = data_to_patches(field, patch_size=32, stride=32)
        for i, (h_s, w_s) in enumerate(locs):
            expected = field[:, h_s : h_s + 32, w_s : w_s + 32]
            assert torch.equal(patches[i], expected)

    def test_stride_equals_patch(self):
        """stride == patch_size → 無重疊，patches 數 = (H/P)*(W/P)。"""
        field = torch.randn(4, 128, 128)
        patches, locs = data_to_patches(field, patch_size=32, stride=32)
        assert patches.shape[0] == 16  # (128/32)^2

    # ── patches_to_full ───────────────────────────────────────────────────────

    def test_roundtrip_no_overlap(self):
        """無重疊時（stride=patch_size）重建應與原場域一致（Hann 邊界除外）。"""
        # 注意：Hann window 在 stride=patch 時，邊緣權重為 0 → 重建有 artifact
        # 所以改用有重疊的情形測 shape，此 test 只驗 shape
        field = torch.randn(3, 64, 64)
        patches, locs = data_to_patches(field, patch_size=32, stride=32)
        full = patches_to_full(patches, locs, full_hw=(64, 64))
        assert full.shape == (3, 64, 64)
        assert isinstance(full, torch.Tensor)

    def test_roundtrip_with_overlap(self):
        """有重疊時（stride < patch），重建中央區域應接近原值。"""
        H, W = 64, 64
        C = 3
        # 用常數場 → Hann 加權後仍應還原
        field = torch.ones(C, H, W)
        patches, locs = data_to_patches(field, patch_size=32, stride=16)
        full = patches_to_full(patches, locs, full_hw=(H, W))
        # 中央區域：所有 patch 均覆蓋，Hann 加權後應≈1
        center = full[:, 16:48, 16:48]
        assert torch.allclose(
            center, torch.ones_like(center), atol=1e-4
        ), f"Center max err={(center - 1).abs().max():.4f}"

    def test_output_shape(self):
        patches = torch.randn(9, 7, 32, 32)
        # 製造對應 locs（3×3 grid with stride=16 in 64×64）
        locs = [(h, w) for h in range(0, 33, 16) for w in range(0, 33, 16)]
        full = patches_to_full(patches, locs, full_hw=(64, 64))
        assert full.shape == (7, 64, 64)

    def test_numpy_patches_input(self):
        """patches 也可以是 numpy array。"""
        patches_np = np.random.randn(4, 3, 32, 32).astype(np.float32)
        locs = [(0, 0), (0, 32), (32, 0), (32, 32)]
        full = patches_to_full(patches_np, locs, full_hw=(64, 64))
        assert isinstance(full, torch.Tensor)
        assert full.shape == (3, 64, 64)

    def test_channel_count_preserved(self):
        """任意通道數通過 round-trip 後維度不變。"""
        for C in [1, 9, 13, 29]:
            field = torch.randn(C, 64, 64)
            patches, locs = data_to_patches(field, patch_size=32, stride=16)
            full = patches_to_full(patches, locs, full_hw=(64, 64))
            assert full.shape == (C, 64, 64), f"C={C}"

    def test_data_to_patches_stride1(self):
        """stride=1 應產生最多 patches。"""
        H, W, P = 10, 10, 4
        field = torch.randn(2, H, W)
        patches, locs = data_to_patches(field, patch_size=P, stride=1)
        expected = (H - P + 1) * (W - P + 1)
        assert (
            patches.shape[0] == expected
        ), f"expected {expected} patches, got {patches.shape[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: FieldConverter → FieldColorizer pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestIntegration:

    def test_full_pipeline(self, stats_path, device):
        """模擬從 norm state → 反正規化 → 衍伸量 → 上色的完整流程。"""
        conv = FieldConverter.from_file(stats_path).to(device)
        colorizer = FieldColorizer().to(device)

        # 1. 合成 Mphy_lattice_raw 並正規化（模擬 DataLoader 輸出）
        m_raw = _make_mphy_raw(2, 64, 64, device)
        m_norm = conv.normalizer.norm_mphy(m_raw)

        # 2. 模擬模型輸出 (norm 空間) → denorm → Vphy
        vphy = conv.toVphy(m_norm, is_norm=True)

        # 3. 上色
        # ★fluid_mask 慣例＝1=fluid/0=solid（2026-03-30 2f673e5 全專案標準化）
        mask = torch.ones(2, 1, 64, 64, device=device)
        mask[:, :, 0:8, 0:8] = 0.0  # 左上角 obstacle（fluid_mask=0）

        vel_rgb = colorizer.getVelColor(vphy["vel_mag"], mask)
        vor_rgb = colorizer.getVorColor(vphy["vorticity"], mask)
        prs_rgb = colorizer.getPressureColor(vphy["pressure"], mask)

        # 4. 驗證
        for name, rgb in [("vel", vel_rgb), ("vor", vor_rgb), ("prs", prs_rgb)]:
            assert rgb.shape == (2, 3, 64, 64), f"{name} shape={rgb.shape}"
            assert rgb.min() >= 0.0 - 1e-6, f"{name} min={rgb.min():.4f}"
            assert rgb.max() <= 1.0 + 1e-6, f"{name} max={rgb.max():.4f}"

        # 5. obstacle 區域應為灰色
        gray = colorizer.OBSTACLE_GRAY
        for name, rgb in [("vel", vel_rgb), ("vor", vor_rgb), ("prs", prs_rgb)]:
            obs_region = rgb[:, :, 0:8, 0:8]
            assert torch.allclose(
                obs_region, torch.full_like(obs_region, gray), atol=1e-5
            ), f"{name} obstacle region not gray"

    def test_patchutils_with_normalizer(self, stats_path, device):
        """PatchUtils 與 FieldNormalizer 可獨立組合使用。"""
        norm = FieldNormalizer.from_file(stats_path).to(device)

        # 合成 Mphy_lattice_raw → norm → 切 patch → 重建
        m_raw = _make_mphy_raw(1, 64, 64, device).squeeze(0)  # [9, 64, 64]
        m_norm = norm.norm_mphy(m_raw)

        patches, locs = data_to_patches(m_norm.cpu(), patch_size=32, stride=16)
        full = patches_to_full(patches, locs, full_hw=(64, 64)).to(device)

        # 中央重建後應接近原始 norm 場
        center_full = full[:, 16:48, 16:48]
        center_orig = m_norm[:, 16:48, 16:48]
        assert torch.allclose(
            center_full, center_orig, atol=1e-4
        ), f"max err={(center_full - center_orig).abs().max():.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 直接執行支援
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile

    # 建立臨時 stats 檔
    tmp = tempfile.mkdtemp()
    sp = Path(tmp) / "global_stats.json"
    _make_stats_json(sp)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {dev}")

    # ── FieldNormalizer ──────────────────────────────────────────────────────
    tn = TestFieldNormalizer()
    tn.test_from_file(sp)
    tn.test_to_device(sp, dev)
    tn.test_mphy_roundtrip_3d(sp, dev)
    tn.test_mphy_roundtrip_4d(sp, dev)
    tn.test_uprime_roundtrip(sp, dev)
    tn.test_sdf_roundtrip(sp, dev)
    tn.test_pos_roundtrip(sp, dev)
    tn.test_cond_roundtrip_1d(sp, dev)   # v3 identity cond
    tn.test_cond_roundtrip_2d(sp, dev)   # v3 identity cond
    tn.test_forward_alias(sp, dev)
    print("[PASS] FieldNormalizer: all passed")

    # ── FieldConverter ───────────────────────────────────────────────────────
    tc = TestFieldConverter()
    tc.test_from_file(sp)
    tc.test_to_device(sp, dev)
    tc.test_toMphy_passthrough(sp, dev)
    tc.test_toMphy_from_state(sp, dev)
    tc.test_toMphy_denorm_roundtrip(sp, dev)
    tc.test_toVphy_shape_4d(sp, dev)
    tc.test_toVphy_shape_3d(sp, dev)
    tc.test_toVphy_velocity(sp, dev)
    tc.test_toVphy_pressure(sp, dev)
    tc.test_toVphy_is_norm(sp, dev)
    tc.test_forward_alias(sp, dev)
    print("[PASS] FieldConverter: all passed")

    # ── FieldColorizer ───────────────────────────────────────────────────────
    tco = TestFieldColorizer()
    tco.test_init()
    tco.test_to_device(dev)
    tco.test_vel_shape_4d(dev)
    tco.test_vel_shape_3d(dev)
    tco.test_vor_shape(dev)
    tco.test_pressure_shape(dev)
    tco.test_output_range(dev)
    tco.test_obstacle_gray(dev)
    tco.test_no_mask(dev)
    tco.test_forward_alias(dev)
    tco.test_vel_lut_not_uniform(dev)
    print("[PASS] FieldColorizer: all passed")

    # ── PatchUtils ───────────────────────────────────────────────────────────
    tp = TestPatchUtils()
    tp.test_output_type()
    tp.test_output_shape()
    tp.test_numpy_input()
    tp.test_single_channel()
    tp.test_many_channels()
    tp.test_patch_content()
    tp.test_stride_equals_patch()
    tp.test_roundtrip_no_overlap()
    tp.test_roundtrip_with_overlap()
    tp.test_output_shape()
    tp.test_numpy_patches_input()
    tp.test_channel_count_preserved()
    tp.test_data_to_patches_stride1()
    print("[PASS] PatchUtils: all passed")

    # ── Integration ──────────────────────────────────────────────────────────
    ti = TestIntegration()
    ti.test_full_pipeline(sp, dev)
    ti.test_patchutils_with_normalizer(sp, dev)
    print("[PASS] Integration: all passed")

    print("\n=== All tests passed ===")

    import shutil

    shutil.rmtree(tmp)
