import numpy as np
import pytest

from yabitsu import elevation as el


def _synthetic():
    s = np.arange(0.0, 1000.0)
    true = 500 + 0.06 * s
    rng = np.random.default_rng(0)
    z = true + 0.4 * rng.standard_normal(len(s)) + 0.3 * np.sin(s / 7)
    z[300] += 12.0            # スパイク
    z[600:640] = np.nan       # 欠損
    bridge = np.zeros(len(s), bool); bridge[800:820] = True
    z[800:820] -= 15.0        # 橋の下の谷を拾った値
    return s, true, z, bridge


def test_valid_mask_excludes_nan_and_bridge_with_margin():
    s, _, z, bridge = _synthetic()
    v = el.valid_mask(z, bridge, 1.0, margin_m=5.0)
    assert not v[600:640].any() and not v[795:825].any() and v[100]


def test_fit_profile_meets_tolerance_and_removes_spike():
    s, true, z, bridge = _synthetic()
    valid = el.valid_mask(z, bridge, 1.0)
    zf, zref, sigma = el.fit_profile(s, z, valid, 1.0, tol=1.5)
    assert not np.isnan(zf).any()
    assert np.abs(zf - zref)[valid].max() <= 1.5
    assert abs(zf[300] - true[300]) < 1.0
    assert np.abs(zf[800:820] - true[800:820]).max() < 1.0  # 橋は前後から補間


def test_profile_stats_grade():
    s = np.arange(0.0, 500.0); z = 100 + 0.08 * s
    st = el.profile_stats(s, z, z, np.ones(len(s), bool), 1.0)
    assert st["max_grade_pct"] == pytest.approx(8.0, abs=0.01) and st["rise_m"] == pytest.approx(0.08 * 499)
    assert st["pct_within_tol"] == pytest.approx(100.0)


def test_plots_are_written(tmp_path):
    s, _, z, bridge = _synthetic()
    valid = el.valid_mask(z, bridge, 1.0)
    zf, zref, _ = el.fit_profile(s, z, valid, 1.0)
    el.plot_profile(tmp_path / "p.png", s, zref, zf, valid, "t")
    assert (tmp_path / "p.png").stat().st_size > 1000
