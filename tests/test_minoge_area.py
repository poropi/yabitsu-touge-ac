import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from yabitsu import geo, site
from yabitsu import minoge_area as ma
from yabitsu import start_lot
from tests.test_meshgen import _sec_from_xy


def _ll(x, y):
    lat, lon = geo.from_local(x, y)
    return [float(lat), float(lon)]


def _sec():
    x = np.arange(201.0)
    sec = _sec_from_xy(x, np.zeros_like(x))
    sec.z[:] = 100.0 + 0.05 * x
    return sec


def _house(x, y, size=(10.0, 7.0)):
    return {"center": _ll(x, y), "axis_deg": 0.0, "size": list(size), "roof_h": 1.5, "eaves": 0.4}


SPEC = {
    "turnaround": {"points": [_ll(*p) for p in [(60, 4), (90, 4), (90, 25), (60, 25)]]},
    "buildings": [_house(40.0, 20.0), _house(120.0, -20.0, (6.0, 5.0)), _house(150.0, 1.0)],  # 3軒目は道にかかる
    "shrine": {"center": _ll(100.0, 30.0), "axis_deg": 90.0, "size": [4.0, 3.0], "roof_h": 1.6, "eaves": 0.5},
    "monuments": [{"center": _ll(95.0, 6.0), "size": [0.8, 0.5, 1.8]}],
    "pits": [_ll(65.0 + 6 * i, 14.0) + [90.0] for i in range(4)],  # 向きは方位（北=0、東=90）
}


def _ground(x, y):
    return 100.0 + 0.2 * np.asarray(x, float)


def _site():
    return ma.build(SPEC, _sec(), _ground)


def test_site_helpers():
    sec = _sec()
    plane = site.fit_plane(sec, (100.0, 0.0))
    assert abs(site.plane_z(plane, 120.0, 5.0) - 106.0) < 0.05
    v, f, uv = site.draped(Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]), lambda x, y: np.full(np.shape(x), 7.0))
    assert np.allclose(v[:, 2], 7.0) and f.shape[1] == 3
    assert np.allclose(site.ll_to_xy([_ll(3.0, 4.0)]), [[3.0, 4.0]], atol=1e-6)


def test_house_sits_on_its_own_ground():
    s = _site()
    pad = next(p for p in s.pads() if p["name"] == "minoge_house_1")
    assert pad["z"] == pytest.approx(_ground(120.0, -20.0))
    walls = next(m for m in s.meshes() if m[0] == "1WALL_s2_houses")[1]
    near = walls[np.hypot(walls[:, 0] - 120.0, walls[:, 1] + 20.0) < 5.0]
    assert near[:, 2].min() == pytest.approx(pad["z"] - 0.3)


def test_houses_do_not_touch_the_road():
    s = _site()
    names = [p["name"] for p in s.pads() if p["name"].startswith("minoge_house")]
    assert names == ["minoge_house_0", "minoge_house_1"]


def test_small_house_is_one_storey():
    s = _site()
    walls = next(m for m in s.meshes() if m[0] == "1WALL_s2_houses")[1]
    big = walls[np.hypot(walls[:, 0] - 40.0, walls[:, 1] - 20.0) < 7.0]
    small = walls[np.hypot(walls[:, 0] - 120.0, walls[:, 1] + 20.0) < 5.0]
    assert np.ptp(big[:, 2]) == pytest.approx(5.5 + 0.3) and np.ptp(small[:, 2]) == pytest.approx(3.0 + 0.3)


def test_turnaround_follows_road_plane_and_carries_pits():
    s = _site()
    pad = next(p for p in s.pads() if p["name"] == "minoge_turnaround")
    v = next(m for m in s.meshes() if m[0] == "1ROAD_s2_minoge_turnaround")[1]
    assert np.allclose(v[:, 2], site.plane_z(pad["plane"], v[:, 0], v[:, 1]))
    assert len(pad["pit_points"]) == 4
    x, y, z, h = pad["pit_points"][0]
    assert (x, y) == pytest.approx((65.0, 14.0), abs=1e-4)
    assert abs(math.remainder(h, 2 * math.pi)) < 0.01  # 方位 90°（東）＝ +x 向き
    assert z == pytest.approx(site.plane_z(pad["plane"], x, y))


def test_turnaround_does_not_overlap_the_road():
    s = _site()
    road = start_lot.road_formation(_sec(), s_min=0.0, s_max=200.0)
    pad = next(p for p in s.pads() if p["name"] == "minoge_turnaround")
    assert Polygon(pad["polygon"]).intersection(road).area < 0.5


def test_meshes_have_expected_names():
    s = _site()
    out = {m[0]: m for m in s.meshes()}
    for n in ("1ROAD_s2_minoge_turnaround", "1WALL_s2_houses", "s2_houses_gables", "s2_houses_roof_brown",
              "s2_houses_roof_grey", "1WALL_s2_shrine", "s2_shrine_roof", "1WALL_s2_monuments"):
        assert n in out, n
    assert all(m[5] for m in out.values())


def test_houses_on_the_turnaround_or_shrine_are_skipped():
    spec = {**SPEC, "buildings": SPEC["buildings"] + [_house(75.0, 15.0, (5.0, 4.0)), _house(100.0, 30.0, (4.0, 3.0))]}
    s = ma.build(spec, _sec(), _ground)
    assert [p["name"] for p in s.pads() if p["name"].startswith("minoge_house")] == ["minoge_house_0", "minoge_house_1"]


def test_footprint_box_keeps_area_of_l_shaped_outline():
    l_shape = Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 20), (0, 20)])   # 面積 256
    c, axis_deg, (L, W) = site.footprint_box(l_shape)
    assert L * W == pytest.approx(l_shape.area, rel=1e-6) and L >= W
    rect = Polygon([(0, 0), (12, 0), (12, 5), (0, 5)])
    c, axis_deg, (L, W) = site.footprint_box(rect)
    assert (L, W) == pytest.approx((12, 5)) and abs(math.remainder(math.radians(axis_deg), math.pi)) < 1e-6
