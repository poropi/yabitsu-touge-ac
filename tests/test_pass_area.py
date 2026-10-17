import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from yabitsu import geo
from yabitsu import meshgen as mg
from yabitsu import pass_area as pa
from yabitsu import s1_layouts as lay
from yabitsu import start_lot
from yabitsu import terrain as tr
from tests.test_meshgen import _sec_from_xy

# 道は y=0 を +x へ（x=0..200）。わき道は x=60 で分かれて北へふくらみ、x=100 で戻る
LANE = [[100.0, 0.0], [95.0, 10.0], [80.0, 16.0], [65.0, 10.0], [60.0, 0.0]]


def _ll(xy):
    lat, lon = geo.from_local(*np.asarray(xy, float).T)
    return np.c_[lat, lon].tolist()


SPEC = {
    "lane": {"points": _ll(LANE), "width": 4.5},
    "pits": {"front": _ll([[70.0, 12.0]])[0], "spacing": 6.0, "lateral": 1.3},
    "plaza": {"points": _ll([[70.0, 20.0], [95.0, 20.0], [95.0, 30.0], [70.0, 30.0]])},
    "shelter": {"center": _ll([[82.0, 25.0]])[0], "axis_deg": 0.0, "size": [9.0, 4.5], "wall_h": 2.6, "roof_h": 1.4,
                "eaves": 0.6},
    "island_curb": 0.15,
}


def _sec():
    x = np.arange(201.0)
    sec = _sec_from_xy(x, np.zeros_like(x))
    sec.z[:] = 100.0 + 0.02 * x  # 2% の上り
    return sec


def _site():
    return pa.build(SPEC, _sec())


def test_plane_follows_the_road():
    site = _site()
    assert abs(site.z(50.0, 0.0) - 101.0) < 0.05 and abs(site.z(150.0, 5.0) - 103.0) < 0.05


def test_lane_island_and_plaza_do_not_overlap_the_road_or_each_other():
    sec = _sec(); site = pa.build(SPEC, sec)
    road = start_lot.road_formation(sec, s_min=0.0, s_max=sec.s[-1])
    for name in ("lane", "island", "plaza"):
        assert site.polys[name].intersection(road).area < 0.5, name
    assert site.polys["island"].intersection(site.polys["lane"]).area < 0.5
    assert site.polys["plaza"].intersection(site.polys["lane"]).area < 0.5
    assert site.polys["island"].area > 50  # 道とわき道の間の島


def test_pits_on_lane_facing_the_exit():
    site = _site()
    pits = site.pits()
    lane = LineString(LANE)
    assert len(pits) == 4 and [d.name for d in pits] == [f"AC_PIT_{i}" for i in range(4)]
    assert all(lane.distance(Point(d.location[:2])) < 1.5 for d in pits)
    # 出口は、道の下り側（s が小さい方、x=60）の合流口
    exit_dir = np.subtract([60.0, 0.0], [70.0, 12.0])
    h = pits[0].heading
    assert np.dot([math.cos(h), math.sin(h)], exit_dir) > 0


def test_meshes_and_pads():
    site = _site()
    names = {m[0]: m for m in site.meshes()}
    for n in ("1ROAD_s1_pass_lane", "1GRASS_s1_pass_island", "s1_pass_curb", "1ROAD_s1_pass_plaza", "1WALL_s1_pass_shelter",
              "s1_pass_shelter_roof", "s1_pass_shelter_gables"):
        assert n in names, n
    v, f = names["1ROAD_s1_pass_lane"][1], names["1ROAD_s1_pass_lane"][2]
    assert (mg.face_normals(v, f)[:, 2] > 0).all()
    assert all("plane" in p for p in site.pads())


def test_layout_marks_move_top_positions():
    sec = _sec()
    objs = {d.name: d for d in lay.layout_objects(sec, [], -1, top_s=150.0, down_grid_s=170.0)}
    assert abs(objs["AC_START_0"].location[0] - 170.0) < 0.5
    assert abs(objs["AC_START_3"].location[0] - (170.0 + 24.0)) < 0.5
    assert abs(objs["AC_OPEN_START_L"].location[0] - 100.0) < 0.5
    up = {d.name: d for d in lay.layout_objects(sec, [], 1, top_s=150.0)}
    assert abs(up["AC_OPEN_FINISH_L"].location[0] - 120.0) < 0.5


def test_pits_from_site_pad_win_over_parking():
    sec = _sec(); site = pa.build(SPEC, sec)
    parking = {"name": "P", "role": "end", "polygon": [[150, 20], [180, 20], [180, 50], [150, 50], [150, 20]], "z": 100.0}
    pits = [d for d in lay.layout_objects(sec, [parking] + site.pads(), -1) if d.name.startswith("AC_PIT")]
    lane = LineString(LANE)
    assert all(lane.distance(Point(d.location[:2])) < 1.5 for d in pits)


def test_carve_uses_plane_pads():
    X, Y = np.meshgrid(np.arange(0.0, 20.0, 2.0), np.arange(30.0, 50.0, 2.0))
    sec = _sec()
    pad = {"polygon": [[0, 30], [20, 30], [20, 50], [0, 50], [0, 30]], "z": 0.0, "plane": [0.0, 0.0, 200.0, 0.5, 0.0]}
    Z, _ = tr.carve(X, Y, np.full(X.shape, 500.0), sec, [pad])
    inside = (X > 1) & (X < 19) & (Y > 31) & (Y < 49)
    assert np.allclose(Z[inside], 200.0 + 0.5 * X[inside] - tr.UNDER_DEPTH)
