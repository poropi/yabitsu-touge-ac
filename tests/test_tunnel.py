import numpy as np
import pytest

from yabitsu import tunnel
from tests.test_meshgen import _straight
from tests.test_side_roads import _way


def test_tunnel_range_on_the_main_road():
    sec = _straight()
    ws = [_way([[100, 0], [160, 0]], id=7, highway="primary", ref="70", tunnel="yes", layer="-1", name="吹風トンネル"),
          _way([[20, 0], [40, 0]], id=8, highway="primary", ref="70", tunnel="yes")]
    t = sorted(tunnel.from_osm(ws, sec), key=lambda q: q.s0)
    assert [q.osm_id for q in t] == [8, 7]
    assert t[1].s0 == pytest.approx(100.0, abs=0.6) and t[1].s1 == pytest.approx(160.0, abs=0.6) and t[1].name == "吹風トンネル"
    assert t[0].name == ""


def test_tunnels_off_the_course_are_ignored():
    sec = _straight()
    ws = [_way([[100, 50], [160, 50]], id=9, highway="primary", tunnel="yes")]
    assert tunnel.from_osm(ws, sec) == []


def test_row_mask_covers_tunnel_and_margin():
    t = [tunnel.Tunnel(1, "", 100.0, 160.0)]
    s = np.arange(0.0, 201.0)
    m = tunnel.row_mask(s, t, margin_m=5.0)
    assert m[95] and m[165] and not m[94] and not m[166] and m.sum() == 71


def _mountain():
    """直線の本線（高さ 100）。x=100〜160 がトンネル。山は x=90〜170 で 125m、まわりは 100m。"""
    from yabitsu import terrain as tr
    sec = _straight()
    t = [tunnel.Tunnel(7, "吹風トンネル", 100.0, 160.0)]
    xs = np.arange(0.0, 201.0); ys = np.arange(-40.0, 41.0); X, Y = np.meshgrid(xs, ys)
    nat = np.where((X > 90) & (X < 170), 125.0, 100.0)
    keep = tunnel.row_mask(sec.s, t, margin_m=0.0)
    Z, _ = tr.carve(X, Y, nat, sec, keep_rows=keep)
    return sec, t, xs, ys, Z


def test_terrain_is_not_cut_inside_the_tunnel():
    sec, t, xs, ys, Z = _mountain()
    assert Z[np.searchsorted(ys, 0.0), np.searchsorted(xs, 130.0)] == pytest.approx(125.0)        # 道の上の山は残る
    assert Z[np.searchsorted(ys, 0.0), np.searchsorted(xs, 50.0)] < 100.0                          # トンネルの外は今まで通り


def test_terrain_stays_above_the_tunnel_roof():
    from yabitsu import terrain as tr
    sec = _straight(); t = [tunnel.Tunnel(7, "", 100.0, 160.0)]
    X, Y = np.meshgrid(np.arange(100.0, 161.0), np.arange(-5.0, 6.0))
    Z, _ = tr.carve(X, Y, np.full(X.shape, 103.0), sec, keep_rows=tunnel.row_mask(sec.s, t, 0.0))   # 土かぶりが薄い山
    roof = 100.0 + tunnel.bore_height(sec, 130) + tr.TUNNEL_COVER_M
    assert Z.min() >= roof - 1e-6


def test_tunnel_wall_encloses_the_road():
    from yabitsu.crosssection import GUTTER_WIDTH
    from yabitsu.meshgen import face_normals
    sec, t, xs, ys, Z = _mountain()
    ms = {m[0]: m for m in tunnel.meshes(sec, t, {}, lambda x, y: np.full(np.shape(x), 125.0))}
    wall = ms["1WALL_tunnel_7_bore"]
    v, f = wall[1], wall[2]
    mid = np.abs(v[:, 0] - 130.0) < 1.5
    assert v[mid, 1].min() <= -(sec.hw_right[130] + GUTTER_WIDTH) + 1e-6 and v[mid, 1].max() >= sec.hw_left[130] + GUTTER_WIDTH - 1e-6
    assert v[mid, 2].max() - 100.0 >= 4.7
    c = v[f].mean(1); n = face_normals(v, f)
    inward = ((-c[:, 1]) * n[:, 1] + (100.0 + 2.5 - c[:, 2]) * n[:, 2]) > 0
    assert inward.mean() > 0.95


def test_portal_has_no_gap_between_terrain_and_portal_face():
    from yabitsu import terrain as tr
    sec, t, xs, ys, Z = _mountain()
    from scipy.interpolate import RegularGridInterpolator
    g = RegularGridInterpolator((ys, xs), Z)
    zf = lambda x, y: g(np.c_[np.ravel(y), np.ravel(x)]).reshape(np.shape(x))
    ms = {m[0]: m for m in tunnel.meshes(sec, t, {}, zf)}
    v = ms["tunnel_7_portal_a"][1]
    assert np.abs(v[:, 0] - 100.0).max() < 1.5
    for y in (-8.0, 8.0):                       # 道の横：坑口の面は、外の削った地形から中の山の高さまでを覆う
        near = v[np.abs(v[:, 1] - y) < 2.0]
        lo, hi = float(zf(np.array([99.0]), np.array([y]))[0]), float(zf(np.array([101.0]), np.array([y]))[0])
        assert near[:, 2].min() <= min(lo, hi) + 1e-6 and near[:, 2].max() >= max(lo, hi) - 1e-6


def test_ground_does_not_poke_out_in_front_of_the_portal():
    """地形は 2m の格子なので、山を残す所の境が坑口ぴったりだと、坑口の前の道に山の斜面がかぶる。境は坑口より奥にする。"""
    from scipy.interpolate import RegularGridInterpolator
    from yabitsu import terrain as tr
    sec = _straight(); t = [tunnel.Tunnel(7, "", 100.0, 160.0)]
    xs = np.arange(0.0, 201.0, 2.0) + 0.7; ys = np.arange(-20.0, 21.0, 2.0); X, Y = np.meshgrid(xs, ys)
    Z, _ = tr.carve(X, Y, np.where((X > 80) & (X < 180), 130.0, 100.0), sec, keep_rows=tunnel.carve_mask(sec.s, t))
    g = RegularGridInterpolator((ys, xs), Z)
    for x in (99.0, 99.8, 160.2, 161.0):
        assert g([[0.0, x]])[0] < 100.0, x
