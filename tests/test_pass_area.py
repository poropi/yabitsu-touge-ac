import pytest
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from yabitsu import geo
from yabitsu import meshgen as mg
from yabitsu import pass_area as pa
from yabitsu import layouts as lay
from yabitsu import start_lot
from yabitsu import terrain as tr
from yabitsu.crosssection import surface_height as cs_surface
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


def test_layout_pits_from_pass_lane():
    sec = _sec(); site = pa.build(SPEC, sec)
    parking = {"name": "P", "role": "end", "polygon": [[150, 20], [180, 20], [180, 50], [150, 50], [150, 20]], "z": 100.0}
    layout = lay.Layout("t", "t", "s1", -1, _ll([[190.0, 0.0]])[0], _ll([[198.0, 0.0]])[0], _ll([[180.0, 0.0]])[0],
                        _ll([[10.0, 0.0]])[0], "pass_lane")
    pits = [d for d in lay.layout_objects(sec, [parking] + site.pads(), layout, {"s1": (0.0, 200.0)})
            if d.name.startswith("AC_PIT")]
    lane = LineString(LANE)
    assert all(lane.distance(Point(d.location[:2])) < 1.5 for d in pits)


def test_carve_uses_plane_pads():
    X, Y = np.meshgrid(np.arange(0.0, 20.0, 2.0), np.arange(30.0, 50.0, 2.0))
    sec = _sec()
    pad = {"polygon": [[0, 30], [20, 30], [20, 50], [0, 50], [0, 30]], "z": 0.0, "plane": [0.0, 0.0, 200.0, 0.5, 0.0]}
    Z, _ = tr.carve(X, Y, np.full(X.shape, 500.0), sec, [pad])
    inside = (X > 1) & (X < 19) & (Y > 31) & (Y < 49)
    assert np.allclose(Z[inside], 200.0 + 0.5 * X[inside] - tr.UNDER_DEPTH)


def test_island_fence_and_wooden_shelter():
    """島の縁の内側に杭とロープ、休憩所は指定の材質（板張り・白い腰壁・屋根の色）。"""
    spec = {**SPEC, "island_fence": {"inset": 0.35, "spacing": 1.5, "height": 0.5},
            "shelter": {**SPEC["shelter"], "wall": "wood", "roof": "roof_blue", "base_h": 0.6}}
    site = pa.build(spec, _sec())
    ms = {m[0]: m for m in site.meshes()}
    fence = ms["s1_pass_island_fence"]
    assert fence[4] == "wood"
    v = fence[1]
    from shapely.geometry import Point
    assert all(site.polys["island"].buffer(0.2).contains(Point(p[:2])) for p in v)
    assert ms["1WALL_s1_pass_shelter"][4] == "wood" and ms["s1_pass_shelter_roof"][4] == "roof_blue"
    assert ms["s1_pass_shelter_base"][4] == "wall_white"


def test_lane_meets_the_road_edge_without_a_step():
    """本線からわき道へ入る所に段差を作らない：道の縁では道の縁の高さ、縁から離れると平面の高さ。"""
    sec = _sec(); sec.slope[:] = 0.06  # 片勾配で、道の縁を平面より高くする
    site = pa.build(SPEC, sec)
    v = next(m[1] for m in site.meshes() if m[0] == "1ROAD_s1_pass_lane")
    edge = 3.0 + pa.GUTTER_WIDTH
    near = np.abs(v[:, 1] - edge) < 0.05
    assert near.sum() >= 2
    assert np.allclose(v[near, 2], sec.edge_z(1)[np.rint(v[near, 0]).astype(int)], atol=0.02)
    far = v[:, 1] > edge + pa.EDGE_BLEND_M + 0.5
    assert far.any() and np.allclose(v[far, 2], site.z(v[far, 0], v[far, 1]), atol=1e-6)


def test_sidewalk_without_curb_is_flush():
    spec = dict(SPEC, sidewalk={"points": _ll([[70.0, 30.0], [95.0, 30.0], [95.0, 33.0], [70.0, 33.0]]), "curb": 0.0})
    names = [m[0] for m in pa.build(spec, _sec()).meshes()]
    assert "1ROAD_s1_pass_sidewalk" in names and "s1_pass_sidewalk_curb" not in names


def test_island_shape_sets_the_island_and_paves_the_rest():
    """島の外形を決めたら、島はその内側だけ。わき道と本線の間の残りはわき道の舗装になる（すき間も重なりもない）。"""
    sec = _sec(); base = pa.build(SPEC, sec)
    shape = Polygon([[70.0, 4.0], [85.0, 4.0], [85.0, 9.0], [70.0, 9.0]])
    site = pa.build(dict(SPEC, island_shape={"points": _ll(np.asarray(shape.exterior.coords))}), sec)
    isl, lane = site.polys["island"], site.polys["lane"]
    assert abs(isl.area - shape.intersection(base.polys["island"]).area) < 0.5
    assert isl.intersection(lane).area < 0.05
    both = isl.union(lane).buffer(0.01)
    assert both.geom_type == "Polygon" and not list(both.interiors)
    assert isl.area + lane.area >= base.polys["island"].area + base.polys["lane"].area - 0.5


def test_island_shape_is_not_cut_at_the_road_edge():
    """ストリートビューの縁石が本線の縁より内側に来るときも、その形のまま作り、本線の路面の上に載せる。"""
    sec = _sec(); sec.slope[:] = 0.06
    shape = Polygon([[70.0, 2.0], [85.0, 2.0], [85.0, 9.0], [70.0, 9.0]])   # 道の縁（3.6 m）より 1.6 m 内側から
    site = pa.build(dict(SPEC, island_shape={"points": _ll(np.asarray(shape.exterior.coords))}), sec)
    assert site.polys["island"].bounds[1] < 2.1
    v = next(m[1] for m in site.meshes() if m[0] == "1GRASS_s1_pass_island")
    on_road = v[:, 1] < 3.0
    i = np.rint(v[on_road, 0]).astype(int)
    road_z = cs_surface(v[on_road, 1], sec.z[i], sec.slope[i], sec.crown[i])
    assert on_road.any() and np.allclose(v[on_road, 2], road_z + SPEC["island_curb"], atol=0.03)


def test_lane_shape_sets_the_lane_around_the_island():
    """わき道の外形を決めたら、わき道＝外形から島と本線を除いた所。広場はわき道と重ならない。"""
    sec = _sec()
    outline = Polygon([[62.0, 4.0], [98.0, 4.0], [98.0, 22.0], [62.0, 22.0]])
    shape = Polygon([[72.0, 5.0], [88.0, 5.0], [88.0, 14.0], [72.0, 14.0]])
    site = pa.build(dict(SPEC, island_shape={"points": _ll(np.asarray(shape.exterior.coords))},
                         lane_shape={"points": _ll(np.asarray(outline.exterior.coords))}), sec)
    isl, lane = site.polys["island"], site.polys["lane"]
    assert abs(isl.area - shape.area) < 0.5
    assert abs(lane.area - (outline.area - shape.area)) < 1.0
    assert site.polys["plaza"].intersection(lane).area < 0.05


def test_mouths_cover_where_the_lane_and_island_touch_the_road():
    """わき道と島が本線に接している区間（その側の側溝をくぼませない）。わき道は道の左（北）で x=60〜100 に接する。"""
    (side, s0, s1), = _site().mouths()
    assert side == 1 and 55 < s0 < 66 and 94 < s1 < 105


def test_gutter_is_left_out_at_the_mouth_and_road_continues():
    """わき道の入口の区間は側溝を作らず、道の縁の高さの路面を側溝の幅だけ延ばす。"""
    sec = _sec(); sl = slice(0, 11)
    seg = np.zeros(10, bool); seg[3:7] = True
    n_all = len(mg.gutter(sec, sl, 1)[1]); n_cut = len(mg.gutter(sec, sl, 1, skip=seg)[1])
    assert n_all - n_cut == 4 * 3
    v, f, _ = mg.mouth_strip(sec, sl, 1, seg)
    assert len(f) == 4
    used = v[np.unique(f)]
    assert np.allclose(used[:, 2], sec.edge_z(1)[np.rint(used[:, 0]).astype(int)], atol=1e-6)


def test_flat_lot_beside_the_road_is_level_just_above_the_road_edge():
    """道の脇の台（売店の前）：道と重なる所は除き、上面は平ら、高さは道の縁＋dz。地形を削る pad にもなる。"""
    sec = _sec(); sec.slope[:] = 0.0
    lot = Polygon([[20.0, 2.0], [40.0, 2.0], [40.0, 15.0], [20.0, 15.0]])
    site = pa.build(dict(SPEC, flats=[{"name": "shop_lot", "dz": 0.25, "points": _ll(np.asarray(lot.exterior.coords))}]), sec)
    (name, g, z, edge), = site.flats
    assert edge == "concrete"
    assert g.bounds[1] > 3.6 - 1e-6                     # 道（側溝まで）と重ならない
    assert z == pytest.approx(float(sec.edge_z(1)[int(round(g.centroid.x))]) + 0.25, abs=0.03)   # 台の真ん中の横の道の縁＋dz
    v = next(m[1] for m in site.meshes() if m[0] == "1ROAD_s1_pass_shop_lot")
    assert np.ptp(v[:, 2]) < 1e-9
    assert any(p["name"] == "pass_shop_lot" and p["z"] == z and p["buildings_ok"] for p in site.pads())   # 台の上には建物が建つ


PARK = {"lot": _ll([[120.0, -2.0], [160.0, -2.0], [160.0, -30.0], [120.0, -30.0]]),
        "beds": [{"name": "b", "points": _ll([[124.0, -10.0], [130.0, -10.0], [130.0, -16.0], [124.0, -16.0]])}],
        "stalls": [{"name": "row", "origin": _ll([[135.0, -12.0]])[0], "toward": _ll([[150.0, -12.0]])[0], "n": 6,
                    "side": "right", "face": "out", "occupied": [1, 3]}],
        "lines": [_ll([[121.0, -6.0], [126.0, -6.0]])]}


def _park_site():
    return pa.build({**SPEC, "parking": PARK}, _sec())


def test_parking_is_a_drivable_surface_off_the_road_with_the_bed_cut_out():
    sec = _sec(); site = pa.build({**SPEC, "parking": PARK}, sec)
    road = start_lot.road_formation(sec, s_min=0.0, s_max=sec.s[-1])
    lot = site.polys["parking"]
    assert lot.intersection(road).area < 0.5 and lot.area > 800
    assert not lot.contains(Point(127.0, -13.0))           # 植え込みは駐車場から抜く
    ms = {m[0]: m for m in site.meshes()}
    assert ms["1ROAD_s1_pass_parking"][4] == "asphalt"
    assert ms["1GRASS_s1_pass_bed_b"][4] == "ground" and "s1_pass_bed_b_curb" in ms
    assert ms["LINES_s1_pass_parking"][4] == "line_white"
    assert any(p["name"] == "pass_parking" and "plane" in p for p in site.pads())
    assert any(p["name"] == "pass_bed_b" for p in site.pads())


def test_parked_cars_only_in_occupied_stalls_facing_the_aisle_with_hidden_collision_boxes():
    site = _park_site()
    cars = site.parked_cars()
    assert [(n, i) for n, i, *_ in cars] == [("row", 1), ("row", 3)]
    for _, i, x, y, z, h in cars:
        assert abs(x - (135.0 + 2.5 * (i + 0.5))) < 0.05 and abs(y - (-14.5)) < 0.05
        assert abs(math.sin(h) - 1.0) < 1e-6               # 通路（列の北の線）の側、+y を向く
    ms = {m[0]: m for m in site.meshes()}
    box = ms["1WALL_s1_pass_parked_cars"]
    assert box[5] is False and len(box[2]) == 2 * 5        # 見えない箱 2 台分（側面 4＋天面）


def test_parking_height_is_level_across_and_follows_the_road_grade():
    site = _park_site()
    assert abs(site.park_z(140.0, -20.0) - site.park_z(140.0, -28.0)) < 0.03
    assert abs((site.park_z(155.0, -20.0) - site.park_z(125.0, -20.0)) - 0.6) < 0.05   # 道の 2% の上り


def test_two_buildings_each_with_their_own_roof():
    b0 = {**SPEC["shelter"], "roof": "roof_red"}
    b1 = {**SPEC["shelter"], "center": _ll([[88.0, 25.0]])[0], "roof": "roof_blue", "size": [5.0, 4.0]}
    spec = {k: v for k, v in SPEC.items() if k != "shelter"}
    ms = {m[0]: m for m in pa.build({**spec, "buildings": [b0, b1]}, _sec()).meshes()}
    assert ms["s1_pass_shelter_roof"][4] == "roof_red" and ms["s1_pass_shelter1_roof"][4] == "roof_blue"
    assert "1WALL_s1_pass_shelter1" in ms


def test_top_z_puts_things_on_the_face_they_stand_on():
    site = _park_site()
    lane = site.polys["lane"].representative_point()
    assert site.top_z(lane.x, lane.y)[0] == pytest.approx(float(site.surface_z(lane.x, lane.y)), abs=1e-6)   # わき道の上
    bed = site.top_z(127.0, -13.0)[0]                                                             # 植え込み：縁石の高さ分だけ上
    assert bed == pytest.approx(float(site.park_z(127.0, -13.0)) + 0.15, abs=1e-6)
    assert site.top_z(140.0, -20.0)[0] == pytest.approx(float(site.park_z(140.0, -20.0)), abs=1e-6)   # 駐車場
    assert np.isnan(site.top_z(140.0, -60.0)[0])                                                  # どの面からも遠い


def test_toilet_front_picture_lies_on_the_south_wall():
    """トイレの正面の絵は南の壁と同じ向き（壁と交差して半分が埋もれない）で、壁より高くない（屋根の上に白い帯が出ない）。"""
    import json
    from yabitsu import config
    spec = json.loads((config.DATA_MANUAL / "s1_pass.json").read_text(encoding="utf-8"))
    signs = json.loads((config.DATA_MANUAL / "s1_objects.json").read_text(encoding="utf-8"))["pole_signs"]
    toilet = next(b for b in spec["buildings"] if b.get("roof") == "roof_blue")
    front = next(s for s in signs if s.get("style") == "toilet_front")
    assert abs((front["facing_deg"] - (180.0 - toilet["axis_deg"]) + 180) % 360 - 180) < 0.5   # axis_deg（東から左回り）の南の面の方位
    assert front["h_m"] <= toilet["wall_h"] - 0.3
