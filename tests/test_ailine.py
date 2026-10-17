import numpy as np
import pytest

from yabitsu import ailine
from yabitsu.acdata import blender_to_ac
from yabitsu.config import AC_TRACKS, ProtectedPathError
from tests.test_meshgen import _circle, _straight


def _write_read(tmp_path, sec, direction=1, **kw):
    p = ailine.write_fast_lane(tmp_path / "ai" / "fast_lane.ai", sec, direction, **kw)
    return ailine.read_fast_lane(p)


def test_points_follow_the_centreline_in_ac_axes(tmp_path):
    sec = _straight(101)
    a = _write_read(tmp_path, sec)
    assert a["version"] == 7 and len(a["pos"]) == 101
    assert np.array_equal(a["id"], np.arange(101))
    assert np.allclose(a["pos"][10], blender_to_ac(10.0, 0.0, 100.0))
    assert np.allclose(a["length"], np.arange(101.0))


def test_extra_fields_on_a_straight(tmp_path):
    sec = _straight(101); sec.hw_left[:] = 2.0; sec.hw_right[:] = 4.0; sec.slope[:] = 0.0
    e = _write_read(tmp_path, sec)["extra"]
    assert np.allclose(e["side_left"], 2.0) and np.allclose(e["side_right"], 4.0)
    assert np.allclose(e["forward"][:-1], [1, 0, 0])         # 東向き = AC の +X
    assert np.allclose(e["normal"], [0, 1, 0]) and np.allclose(e["grade"], 0.0)
    assert np.allclose(e["seg_length"][:-1], 1.0)
    assert np.all(e["radius"] > 1e5)


def test_reverse_direction_swaps_order_and_sides(tmp_path):
    sec = _straight(101); sec.hw_left[:] = 2.0; sec.hw_right[:] = 4.0
    a = _write_read(tmp_path, sec, direction=-1)
    assert np.allclose(a["pos"][0], blender_to_ac(100.0, 0.0, 100.0))
    assert np.allclose(a["extra"]["side_left"], 4.0) and np.allclose(a["extra"]["side_right"], 2.0)
    assert np.allclose(a["extra"]["forward"][:-1], [-1, 0, 0])


def test_turn_sign_and_radius_match_the_kunos_convention(tmp_path):
    sec = _circle(50.0)  # 左カーブ
    a = _write_read(tmp_path, sec); e = a["extra"]
    f = e["forward"]
    turn = np.cross(f[:-2], f[1:-1])[:, 1]  # 実在の AI ラインで確かめた：dir = sign(fwd_i × fwd_i+1 の Y)
    mid = slice(20, 180)
    assert np.all(e["dir"][mid] == np.sign(turn[mid]))
    assert np.all(np.sign(e["lat_g"][mid]) == e["dir"][mid])
    assert np.median(e["radius"][mid]) == pytest.approx(50.0, rel=0.03)


def test_banked_surface_normal_leans_away_from_the_high_side(tmp_path):
    sec = _straight(51); sec.slope[:] = 0.05  # 左（北）が高い
    e = _write_read(tmp_path, sec)["extra"]
    left_ac = np.array(blender_to_ac(0.0, 1.0, 0.0))
    assert np.all(e["normal"] @ left_ac < 0)
    assert np.allclose(np.linalg.norm(e["normal"], axis=1), 1.0)
    assert np.allclose(np.sum(e["normal"][:-1] * e["forward"][:-1], 1), 0.0, atol=1e-6)


def test_speed_profile_respects_corner_and_braking_limits():
    k = np.r_[np.zeros(300), np.full(100, 1 / 30.0), np.zeros(300)]
    v, gas, brake = ailine.speed_profile(k, 1.0, v_max=25.0, a_lat=6.0, a_acc=2.5, a_brake=6.0)
    assert v.max() == pytest.approx(25.0)
    assert np.all(v[300:400] <= np.sqrt(6.0 * 30.0) + 1e-6)
    dv2 = np.diff(v ** 2) / 2
    assert dv2.max() <= 2.5 + 1e-6 and dv2.min() >= -6.0 - 1e-6
    assert brake[250:300].max() > 0 and brake[:200].max() == 0
    assert gas[:50].min() == 1.0 and np.all((gas >= 0) & (gas <= 1) & (brake >= 0) & (brake <= 1))


def test_lookup_grid_lists_nearest_points_per_10m_cell(tmp_path):
    sec = _circle(80.0)
    a = _write_read(tmp_path, sec); g = a["grid"]
    xz = a["pos"][:, [0, 2]]
    assert g["min"][0] == pytest.approx(xz[:, 0].min() - 350, abs=1e-3)
    assert g["max"][2] == pytest.approx(xz[:, 1].max() + 350, abs=1e-3)
    assert g["cols"] == int((g["max"][0] - g["min"][0]) // 10) and g["rows"] == int((g["max"][2] - g["min"][2]) // 10)
    for (i, j) in [(0, 0), (40, 37), (g["cols"] - 1, g["rows"] - 1)]:
        cell = g["cells"][i][j]
        c = np.array([g["min"][0] + (i + 0.5) * 10, g["min"][2] + (j + 0.5) * 10])
        d = np.linalg.norm(xz - c, axis=1)
        assert len(cell) == 10 and cell[0] == int(np.argmin(d))
        assert np.all(np.diff(d[cell]) >= -1e-4)


def test_refuses_protected_folder():
    with pytest.raises(ProtectedPathError):
        ailine.write_fast_lane(AC_TRACKS / "nanohanadai" / "ai" / "fast_lane.ai", _straight(11), 1)
