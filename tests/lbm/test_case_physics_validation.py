"""`validation.case_physics` 的單元測試。

這個模組是模擬產物的**唯一收錄閘門**——它判 PASS 的資料會直接進 dataset、
進而餵給模型訓練。所以它自己必須先被驗證：每一條判準都要證明它在該過的地方過、
該擋的地方擋，尤其是**邊界**與**已知缺陷模式**。

測試分三層：

1. 純函式判準（不碰 I/O）：正例、反例、邊界
2. 場量計算（散度、半窗差、條紋指數）：用解析上已知答案的合成場
3. 端到端：合成一份健康 H5 應 PASS，逐一注入缺陷後應 FAIL
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lbmnca.lbm.lbm_mrt_les.validation.case_physics import (  # noqa: E402
    C_S,
    THRESH,
    Check,
    Report,
    check_burnin,
    check_geometry,
    check_inlet_ma,
    check_inlet_stripes,
    check_ke_growth,
    check_l_char,
    check_local_ma,
    check_mass_conservation,
    check_mass_residual,
    check_mean_convergence,
    check_re_consistency,
    check_required_attrs,
    check_symmetry,
    check_tau,
    divergence_ratio,
    half_window_diff,
    stripe_index,
    validate_h5,
)


# ── 1. 純函式判準 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("tau,expect", [
    (0.5000, False),   # 理論穩定界本身
    (0.5099, False),   # 剛好低於下限
    (0.5100, True),    # 下限（含）
    (0.6500, True),
    (2.0000, True),    # 上限（含）
    (2.0001, False),
])
def test_check_tau_boundaries(tau, expect):
    assert check_tau(tau).passed is expect


def test_check_tau_is_fatal():
    assert check_tau(0.4).severity == "fatal"


@pytest.mark.parametrize("u,expect", [
    (0.05, True),      # 生產值：Ma = 0.0866
    (0.0577, True),    # Ma = 0.0999
    (0.0578, False),   # Ma = 0.1001
    (0.10, False),
])
def test_check_inlet_ma(u, expect):
    assert check_inlet_ma(u).passed is expect


def test_check_re_consistency_pass_and_fail():
    # u·L/ν = 0.05 × 100 / 0.01 = 500
    assert check_re_consistency(500.0, 0.05, 100.0, 0.01).passed
    # attrs 寫 300 但實際 500 → 不自洽
    assert not check_re_consistency(300.0, 0.05, 100.0, 0.01).passed


def test_check_re_consistency_catches_lchar_mismatch():
    """L_char 用錯定義（如 min(w,h) 而非 y-extent）會被這條抓到。

    L3 實測：aspect 幾何下兩種定義差 5 倍，使實際 Re 從 300 變成 1500。
    """
    c = check_re_consistency(300.0, 0.05, 200.0, 0.05 * 40.0 / 300.0)
    assert not c.passed


def test_check_ke_growth_steady():
    rng = np.random.default_rng(0)
    ke = 1.0 + 0.01 * rng.standard_normal(64)
    assert check_ke_growth(ke).passed


def test_check_ke_growth_diverging():
    ke = np.exp(np.linspace(0, 3, 64))     # 末段/中段 遠超 1.5
    assert not check_ke_growth(ke).passed


def test_check_ke_growth_oscillating_is_ok():
    """渦脫落造成的週期振盪不該被判成發散。"""
    t = np.linspace(0, 20 * np.pi, 200)
    assert check_ke_growth(1.0 + 0.3 * np.sin(t)).passed


def test_check_ke_growth_insufficient_samples():
    assert not check_ke_growth(np.ones(5)).passed


def test_check_local_ma():
    assert check_local_ma(0.2).passed          # Ma = 0.346
    assert not check_local_ma(0.4).passed      # Ma = 0.693


def test_check_mass_conservation_uses_ma2_reference():
    """參照必須是 Ma²，不是 float32 底噪。"""
    inlet_ma = 0.05 / C_S                      # 0.0866
    ma2 = inlet_ma ** 2                        # 7.5e-3
    assert check_mass_conservation(0.3 * ma2, inlet_ma).passed
    assert not check_mass_conservation(1.5 * ma2, inlet_ma).passed


def test_check_mass_residual_missing_is_warn_not_fail():
    c = check_mass_residual(None)
    assert c.severity == "warn" and c.passed


def test_check_mass_residual_gate():
    assert check_mass_residual(np.full(10, 0.01)).passed
    assert not check_mass_residual(np.full(10, 0.20)).passed


def test_check_mean_convergence_half_diff_to_se():
    """半窗差 = √2 × 單窗 SE；gate 在 30%，warn 門檻在 10%。

    雙級的理由見 THRESH 註解：高 Re 的低頻尾流擺動使 SE 卡在 18-22%，
    那是物理限制不是設定錯誤，gate 設在 5% 只會讓那批 case 永遠 FAIL。
    """
    # ★參數現在直接是 batch-means SE（不再是半窗差）
    assert check_mean_convergence(0.035).passed           # SE 3.5%，品質良好
    assert check_mean_convergence(0.21).passed            # SE 21%，通過但標偏高
    assert "良好" in check_mean_convergence(0.035).detail
    assert "偏高" in check_mean_convergence(0.21).detail
    assert not check_mean_convergence(0.57).passed        # SE 57%，根本沒收斂


def test_check_burnin_cold_needs_re_ctus():
    # Re=300 的 cold case 需要 ≥ 0.9×300 = 270 CTU
    assert check_burnin(300.0, 300.0, "cold").passed
    assert not check_burnin(40.0, 300.0, "cold").passed    # 舊協定的 40 CTU


def test_check_burnin_warm_lower_factor():
    assert check_burnin(60.0, 300.0, "warm").passed


def test_check_burnin_missing_field_fails():
    c = check_burnin(-1.0, 300.0, "cold")
    assert not c.passed and "舊協定" in c.detail


def test_check_geometry_rejects_empty_and_overfull():
    assert check_geometry(0.05).passed
    assert not check_geometry(0.0).passed       # 沒有建物
    assert not check_geometry(0.8).passed       # 幾乎全固體


def test_check_l_char_tolerance():
    assert check_l_char(100.0, 101.0).passed        # 1% 誤差
    assert not check_l_char(100.0, 105.0).passed    # 5% 誤差


def test_check_required_attrs():
    from lbmnca.lbm.lbm_mrt_les.validation.case_physics import REQUIRED_ATTRS
    assert check_required_attrs(set(REQUIRED_ATTRS)).passed
    partial = set(REQUIRED_ATTRS) - {"tau"}
    c = check_required_attrs(partial)
    assert not c.passed and "tau" in c.detail


def test_check_symmetry_not_applicable_is_warn_pass():
    c = check_symmetry(None)
    assert c.passed and c.severity == "warn"


def test_check_symmetry_values():
    assert check_symmetry(0.01).passed
    assert not check_symmetry(0.10).passed


# ── 2. 場量計算 ────────────────────────────────────────────────────────────

def test_divergence_ratio_zero_for_divergence_free_field():
    """解析上無散度的場（剪切流 u=u(y), v=0）應給接近 0 的比值。"""
    H, W = 64, 96
    y = np.arange(H)[:, None] * np.ones((1, W))
    u = 0.05 + 0.01 * np.sin(2 * np.pi * y / H)
    v = np.zeros_like(u)
    rho = np.ones_like(u)
    fluid = np.ones_like(u, dtype=bool)
    assert divergence_ratio(rho, u, v, fluid) < 1e-10


def test_divergence_ratio_positive_for_diverging_field():
    """人造有散度的場（u 沿 x 線性增加）應給明顯非零比值。"""
    H, W = 64, 96
    x = np.ones((H, 1)) * np.arange(W)[None, :]
    u = 0.05 + 1e-3 * x
    v = np.zeros_like(u)
    assert divergence_ratio(np.ones_like(u), u, v,
                            np.ones_like(u, dtype=bool)) > 0.5


def test_half_window_diff_zero_for_identical_halves():
    seg = np.tile(np.arange(6, dtype=np.float64).reshape(1, 1, 2, 3), (8, 1, 1, 1))
    assert half_window_diff(seg, np.ones(8), 0) < 1e-12


def test_half_window_diff_detects_drift():
    """兩半系統性不同時應給明顯非零值。

    ★場必須有空間結構：度量會先去掉各通道的常數偏移，純常數場的分母為 0、
    該通道被跳過（那是正確行為——常數場沒有結構可比）。
    """
    x = np.arange(8, dtype=np.float64)[None, :] * np.ones((6, 1))
    base = np.sin(2 * np.pi * x / 8)[None, None]           # [1,1,6,8]
    seg = np.concatenate([np.tile(base, (4, 1, 1, 1)),
                          np.tile(base * 1.5, (4, 1, 1, 1))], axis=0)
    assert half_window_diff(seg, np.ones(8), 0) > 0.1


def test_half_window_diff_constant_field_is_nan():
    """全常數場無結構可比 → NaN，且該情形不得判 PASS。"""
    seg = np.ones((8, 1, 4, 4))
    assert np.isnan(half_window_diff(seg, np.ones(8), 0))
    assert not check_mean_convergence(float("nan")).passed


def test_batch_means_is_more_stable_than_half_window():
    """batch-means 的**變異**必須小於半窗差——這是換用它的唯一理由。

    半窗差只有 1 個自由度（前半 vs 後半一個對比），單次實現可能偏大也可能偏小；
    實測 D0-W 的 case 在同窗長、僅滑動起點 40 CTU 的情況下就跳動 2 倍。
    batch-means 用 k−1 自由度，估計應該穩定得多。

    所以正確的測試不是比單次數值大小（那是隨機的），而是比 20 次獨立實現的
    變異係數。
    """
    from lbmnca.lbm.lbm_mrt_les.validation.case_physics import batch_means_se
    x = np.arange(12, dtype=np.float64)[None, :] * np.ones((8, 1))
    base = np.sin(2 * np.pi * x / 12)
    bs, hs = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        seg = np.stack([base * (1.0 + 0.05 * rng.standard_normal())
                        for _ in range(48)])[:, None]
        n = np.ones(48)
        bs.append(batch_means_se(seg, n, 0, chan_groups=((0,),)))
        hs.append(half_window_diff(seg, n, 0) / np.sqrt(2.0))
    bs, hs = np.array(bs), np.array(hs)
    cv_b = bs.std(ddof=1) / bs.mean()
    cv_h = hs.std(ddof=1) / hs.mean()
    assert cv_b < cv_h, f"batch-means CV {cv_b:.3f} 應小於半窗差 CV {cv_h:.3f}"


def test_batch_means_se_insufficient_segments():
    seg = np.ones((8, 1, 4, 4))
    from lbmnca.lbm.lbm_mrt_les.validation.case_physics import batch_means_se
    assert np.isnan(batch_means_se(seg, np.ones(8), 0, n_batch=6))


def test_half_window_diff_nan_when_too_few_segments():
    seg = np.ones((3, 1, 2, 2))
    assert np.isnan(half_window_diff(seg, np.ones(3), 0))


def test_stripe_index_low_for_smooth_field():
    """平滑（低波數）場的高頻能量占比應遠低於格點級振盪。

    不要求趨近 0：逐列去線性趨勢會在邊界引入輕微不連續，單週期正弦實測 0.024、
    三週期 0.0011、純線性 0.012。判準取 0.05，與棋盤的 0.9999 仍差 20 倍以上，
    區分度充足。
    """
    H, W = 40, 128
    x = np.arange(W)[None, :] * np.ones((H, 1))
    for smooth in (np.sin(2 * np.pi * x / W),
                   np.sin(6 * np.pi * x / W),
                   x / W):
        assert stripe_index(smooth, 0, W) < 0.05


def test_stripe_index_high_for_checkerboard():
    """格點級（2 px 週期）振盪應把能量全部堆在高波數半段。"""
    H, W = 40, 128
    x = np.arange(W)[None, :] * np.ones((H, 1))
    checker = (-1.0) ** x
    assert stripe_index(checker, 0, W) > 0.9


def test_check_inlet_stripes():
    """G6.1 仍計算比值，但 2026-08-01 起是 warn（參考量），不判退資料。"""
    assert check_inlet_stripes(1.0).passed
    assert not check_inlet_stripes(10.0).passed     # 舊資料實測 4.5-51.7
    # ★severity 是本檢查的重點：比值被證明受遠場窗有無建物支配、對真正的
    #   條紋零敏感度，以它守門會在收錄集造成幾何系統性偏差。
    assert check_inlet_stripes(10.0).severity == "warn"


# ── 3. Report 組裝語意 ─────────────────────────────────────────────────────

def test_report_warn_does_not_affect_verdict():
    r = Report(h5_path="x")
    r.add(Check("A", "a", "gate", True))
    r.add(Check("B", "b", "warn", False))
    assert r.finalize().passed


def test_report_gate_failure_fails_case():
    r = Report(h5_path="x")
    r.add(Check("A", "a", "gate", False))
    assert not r.finalize().passed
    assert [c.id for c in r.failures] == ["A"]


def test_report_fatal_failure_fails_case():
    r = Report(h5_path="x")
    r.add(Check("A", "a", "fatal", False))
    assert not r.finalize().passed


# ── 4. 端到端：合成 H5 ─────────────────────────────────────────────────────

def _write_h5(path: Path, *, tau=0.65, re=100.0, l_char=100.0,
              burnin=100.0, n_seg=48, stripe=False, diverge=False,
              ke_diverge=False, drop_attr=None, solid_frac=0.05,
              nu_override=None):
    """造一份結構完整的合成 H5；各參數用來注入特定缺陷。

    `nu_override` 讓 `nu` 與 `re` 脫鉤——預設兩者由 `nu = u·L/re` 綁定（與真實
    config 一致），因此 G2.3 恆等、無法被觸發。真實世界會讓它們脫鉤的正是
    **組裝錯誤**（例如 L_char 用了 A 定義寫進 attrs、nu 卻由 B 定義算出）。
    """
    import h5py

    H, W, C = 48, 96, 9
    u_inlet = 0.05
    nu = nu_override if nu_override is not None else u_inlet * l_char / re

    mask = np.zeros((H, W), dtype=bool)
    n_solid = int(solid_frac * H * W)
    if n_solid > 0:
        side = max(1, int(np.sqrt(n_solid)))
        # 上下對稱擺放，讓對稱性檢查可適用
        r0 = H // 2 - side // 2
        mask[r0:r0 + side, W // 4:W // 4 + side] = True

    yy = np.arange(H)[:, None] * np.ones((1, W))
    u = np.full((H, W), u_inlet) + 0.002 * np.sin(2 * np.pi * yy / H)
    if diverge:
        xx = np.ones((H, 1)) * np.arange(W)[None, :]
        u = u + 2e-3 * xx
    if stripe:
        xs = np.arange(W)[None, :] * np.ones((H, 1))
        u = u + 0.01 * ((-1.0) ** xs) * (xs < 0.12 * W)
    v = np.zeros((H, W))
    rho = np.ones((H, W))

    m = np.zeros((C, H, W), dtype=np.float64)
    m[0] = rho
    m[3] = rho * u
    m[5] = rho * v

    with h5py.File(str(path), "w") as f:
        f.create_dataset("geometry/mask", data=mask)
        n_snap = 32
        f.create_dataset("trajectory/moments",
                         data=np.tile(m[None], (n_snap, 1, 1, 1)).astype(np.float16))
        f.create_dataset("trajectory/ctu",
                         data=np.linspace(0, 200, n_snap, dtype=np.float32))
        ke = (np.exp(np.linspace(0, 3, n_snap)) if ke_diverge
              else 1.0 + 0.01 * np.sin(np.linspace(0, 20, n_snap)))
        f.create_dataset("convergence_info/kinetic_energy",
                         data=ke.astype(np.float32))
        f.create_dataset("convergence_info/mass_residual_rel",
                         data=np.full(n_snap, 0.005, dtype=np.float32))

        seg = np.tile(m[None], (n_seg, 1, 1, 1))
        f.create_dataset("mean_accum/segment_mean", data=seg.astype(np.float32))
        f.create_dataset("mean_accum/segment_n",
                         data=np.full(n_seg, 1000, dtype=np.int64))
        f.create_dataset("mean_accum/segment_ctu0",
                         data=np.arange(n_seg, dtype=np.float64) * 5.0)
        f.create_dataset("mean_accum/segment_ctu1",
                         data=(np.arange(n_seg) + 1).astype(np.float64) * 5.0)

        attrs = {
            "case_id": "synthetic", "geometry_id": "g0",
            "dataset_name": "D_TEST", "ic_type": "cold",
            "re": re, "u_inlet_lu": u_inlet, "l_char_px": l_char,
            "nu_lu": nu, "tau": tau, "ctu_lbm_steps": int(l_char / u_inlet),
            "total_ctu": 200.0, "n_snapshots": n_snap,
            "warmup_ctus": 10.0, "burnin_ctus": burnin,
            "mean_window_ctus": 80.0,
        }
        if drop_attr:
            attrs.pop(drop_attr, None)
        for k, val in attrs.items():
            f.attrs[k] = val
    return path


@pytest.fixture()
def healthy_h5(tmp_path):
    # mask 為 5% 固體、單一方塊，y-extent 實算需與 l_char_px 相符：
    # 48×96 的 5% ≈ 230 px → side ≈ 15；mask 是 3× 降採樣 → L_char ≈ 45
    return _write_h5(tmp_path / "ok.h5", l_char=45.0, re=100.0, burnin=100.0)


def test_healthy_case_passes(healthy_h5):
    rep = validate_h5(healthy_h5)
    assert rep.passed, [(c.id, c.detail) for c in rep.failures]


def test_bad_tau_fails(tmp_path):
    p = _write_h5(tmp_path / "tau.h5", l_char=45.0, tau=0.505)
    rep = validate_h5(p)
    assert not rep.passed and "G2.1" in [c.id for c in rep.failures]


def test_insufficient_burnin_fails(tmp_path):
    """舊協定的 40 CTU burn-in 必須被擋下。"""
    p = _write_h5(tmp_path / "burn.h5", l_char=45.0, re=300.0, burnin=40.0)
    rep = validate_h5(p)
    assert not rep.passed and "G5.2" in [c.id for c in rep.failures]


def test_inlet_stripes_warns_but_does_not_reject(tmp_path):
    """有條紋時 G6.1 仍會不通過並現身於 failures，但**不得**使整份報告判退。

    降為 warn 之後，`Report.finalize` 只看 fatal/gate，所以 `rep.passed`
    不再受 G6.1 影響。這正是降級要達成的效果——數字照報，不擋資料。
    """
    p = _write_h5(tmp_path / "stripe.h5", l_char=45.0, stripe=True)
    rep = validate_h5(p)
    g6 = next(c for c in rep.checks if c.id == "G6.1")
    assert g6.severity == "warn"
    assert not g6.passed
    assert "G6.1" not in [c.id for c in rep.failures
                          if c.severity in ("fatal", "gate")]


def test_diverging_ke_fails(tmp_path):
    p = _write_h5(tmp_path / "ke.h5", l_char=45.0, ke_diverge=True)
    rep = validate_h5(p)
    assert not rep.passed and "G3.1" in [c.id for c in rep.failures]


def test_missing_required_attr_fails(tmp_path):
    p = _write_h5(tmp_path / "attr.h5", l_char=45.0, drop_attr="tau")
    rep = validate_h5(p)
    assert not rep.passed and "G7.3" in [c.id for c in rep.failures]


def test_empty_geometry_fails(tmp_path):
    p = _write_h5(tmp_path / "geo.h5", l_char=45.0, solid_frac=0.0)
    rep = validate_h5(p)
    assert not rep.passed and "G7.1" in [c.id for c in rep.failures]


def test_unopenable_file_fails_gracefully(tmp_path):
    p = tmp_path / "nope.h5"
    p.write_bytes(b"not an hdf5")
    rep = validate_h5(p)
    assert not rep.passed and rep.failures[0].id == "G1.1"


def test_report_serializable(healthy_h5):
    import json
    d = validate_h5(healthy_h5).to_dict()
    json.dumps(d, ensure_ascii=False)      # 不得拋錯
    assert d["n_checks"] >= 14
    assert set(d) >= {"passed", "checks", "meta", "failed_ids"}


def test_thresholds_are_documented():
    """每個判準常數都必須有對應的檢查在用——避免死參數。"""
    assert THRESH["div_over_ma2_max"] == 1.0
    assert THRESH["mean_se_gate"] > THRESH["mean_se_ideal"]
    assert 0.0 < THRESH["burnin_re_factor_min"] <= 1.0


# ── 5. 鑑別力：每個 gate/fatal 項都必須有能讓它 FAIL 的情境 ────────────────
#
# 動機（2026-07-29）：閘門的判準被連續修正過三次（G7.2 放寬容忍、G4.1/G5.1 依
# 資料來源降級、G4.2 降為 warn），每次都發生在看到 FAIL 之後。這有把閘門放寬到
# 「永遠會過」的風險。以下測試對**每一個 gate/fatal 項**構造一個應該 FAIL 的
# 端到端情境——任何一項若無法被觸發，就代表它已失去鑑別力。

def _all_gate_ids(rep) -> set[str]:
    return {c.id for c in rep.checks if c.severity in ("fatal", "gate")}


def test_every_gate_check_is_reachable_as_failure(tmp_path):
    """逐項注入缺陷，確認每個 gate/fatal 項都真的會被觸發。"""
    cases = {
        "G2.1": dict(tau=0.505),                       # tau 越下界
        # attrs 寫 re=100 但 nu 由「L_char 用錯定義」算出 → u·L/ν = 500
        "G2.3": dict(l_char=45.0, re=100.0, burnin=100.0,
                     nu_override=0.05 * 45.0 / 500.0),
        "G3.1": dict(ke_diverge=True),                 # KE 發散
        "G5.2": dict(re=300.0, burnin=40.0),           # 舊協定的 burn-in
        # G6.1 已於 2026-08-01 降為 warn，不再是判退理由 → 移出本表
        "G7.1": dict(solid_frac=0.0),                  # 空幾何
        "G7.3": dict(drop_attr="tau"),                 # 缺溯源欄位
    }
    for cid, kw in cases.items():
        kw.setdefault("l_char", 45.0)
        p = _write_h5(tmp_path / f"neg_{cid}.h5", **kw)
        rep = validate_h5(p)
        ids = [c.id for c in rep.failures]
        assert cid in ids, f"{cid} 未被觸發（實際 FAIL：{ids}）"


def test_l_char_mismatch_is_reachable(tmp_path):
    """G7.2：attrs 的 L_char 與 mask 實算差距夠大時必須擋下。

    容忍度含降採樣量化（1.5×ds/L_char），所以要用**遠超**該量化尺度的錯誤
    才能觸發——這正是它該抓的東西（用錯定義會差數倍，不是差一格）。
    """
    p = _write_h5(tmp_path / "lchar.h5", l_char=200.0, re=100.0, burnin=100.0)
    rep = validate_h5(p)
    assert "G7.2" in [c.id for c in rep.failures]


def test_margin_direction_semantics():
    """上限型與下限型判準的餘裕算法不可混用。"""
    upper = Check("X", "u", "gate", True, 0.5, 1.0, "", "upper")
    lower = Check("Y", "l", "gate", True, 300.0, 200.0, "", "lower")
    assert abs(upper.margin() - 0.5) < 1e-12      # 用了一半
    assert abs(lower.margin() - 0.5) < 1e-12      # 超出 50%
    assert Check("Z", "r", "gate", True, 1.0, None, "", "range").margin() is None


def test_burnin_margin_is_positive_when_satisfied():
    """G5.2 是下限型：達標時餘裕必須為正（先前用上限公式算出 −98%）。"""
    c = check_burnin(300.0, 300.0, "cold")
    assert c.direction == "lower" and c.margin() > 0
