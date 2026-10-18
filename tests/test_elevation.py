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


def _fixed_case():
    s = np.arange(0.0, 3001.0); z_raw = 100 + 0.08 * s + 3 * np.sin(s / 40.0); valid = np.ones_like(s, bool)
    return s, z_raw, valid


def test_fixed_sections_keep_their_heights_when_a_new_section_is_added():
    """受け入れ済みの区間（先頭の fixed 個）は、上に区間を足しても高さがまったく変わらない（S4 を峠の先に足すとき）。"""
    s, z_raw, valid = _fixed_case()
    old, _, _ = el.fit_sections(s[:2001], z_raw[:2001], valid[:2001], 1.0, [(0, 1000), (1000, 2000)])
    new, _, _ = el.fit_sections(s, z_raw, valid, 1.0, [(0, 1000), (1000, 2000), (2000, 3000)], fixed=2)
    assert np.allclose(new[:2001], old, atol=1e-9)


def test_new_section_meets_the_fixed_end_without_a_step_or_kink():
    s, z_raw, valid = _fixed_case()
    z, _, _ = el.fit_sections(s, z_raw, valid, 1.0, [(0, 1000), (1000, 2000), (2000, 3000)], fixed=2)
    assert abs(z[2001] - z[2000]) < 0.12                       # 1m で 8% の上り＋α：段差がない
    g0 = (z[2000] - z[1990]) / 10.0; g1 = (z[2010] - z[2000]) / 10.0
    assert abs(g1 - g0) <= 0.02                                 # 勾配が折れない（今のつなぎ目と同じ基準：test_join_has_no_grade_jump）


def _ravines():
    """5% の下り坂。道が沢を渡る所（3か所）だけ、標高データが谷底（6m 下、幅 20m）を拾っている。"""
    s = np.arange(0.0, 3001.0); true = 700 - 0.05 * s
    rng = np.random.default_rng(1)
    z = true + 0.3 * rng.standard_normal(len(s))
    for c in (800.0, 1500.0, 2300.0):
        z -= 6.0 * np.exp(-((s - c) / 8.0) ** 2)
    return s, true, z


def test_short_dips_in_the_dem_do_not_become_dips_in_the_road():
    """S4 は標高データの谷（沢を渡る所）に引っぱられて、ジェットコースターのような上り下りになっていた。"""
    s, true, z = _ravines()
    zf, _, _ = el.fit_sections(s, z, np.ones_like(s, bool), 1.0, [(0, 1000), (1000, 3000)], fixed=1, robust=True)
    k = 20; bump = np.abs(zf[2 * k:] - 2 * zf[k:-k] + zf[:-2 * k])[1000:]
    assert bump.max() < 1.0                              # 20m ごとの曲がり（上り下りの急な変わり方）が小さい
    assert np.abs(zf - true)[1000:].max() < 1.5           # 本当の坂からは 1.5m 以内


def test_fixed_sections_are_untouched_by_the_robust_fit():
    s, true, z = _ravines()
    a, _, _ = el.fit_sections(s, z, np.ones_like(s, bool), 1.0, [(0, 1000), (1000, 3000)], fixed=1)
    b, _, _ = el.fit_sections(s, z, np.ones_like(s, bool), 1.0, [(0, 1000), (1000, 3000)], fixed=1, robust=True)
    assert np.array_equal(a[:1001], b[:1001])


def test_short_humps_in_the_dem_are_ignored_too():
    """道のすぐ横の土手を拾って、標高データが短く盛り上がる所（S4 1780m）も外す。"""
    s, true, z = _ravines()
    z = z + 3.0 * np.exp(-((s - 1900.0) / 12.0) ** 2)
    zf, _, _ = el.fit_sections(s, z, np.ones_like(s, bool), 1.0, [(0, 1000), (1000, 3000)], fixed=1, robust=True)
    assert np.abs(zf - true)[1800:2000].max() < 0.8


def test_adding_the_next_section_keeps_every_attached_section_identical():
    """S5 を足しても S4（つないだ区間）の高さは1mmも変わらない（つなぐ区間は1つずつ、前から順につなぐ）。"""
    s, true, z = _ravines()
    s2 = np.arange(0.0, 4001.0); z2 = np.r_[z, z[-1] - 0.05 * np.arange(1.0, 1001.0)]
    v = np.ones_like(s2, bool)
    ab, _, _ = el.fit_sections(s2[:3001], z2[:3001], v[:3001], 1.0, [(0, 1000), (1000, 3000)], fixed=1, robust=True)
    abc, _, _ = el.fit_sections(s2, z2, v, 1.0, [(0, 1000), (1000, 3000), (3000, 4000)], fixed=1, robust=True)
    assert np.array_equal(abc[:3001], ab)
