import numpy as np
import pytest
from shapely.geometry import LineString, Point

from yabitsu import geo, town as tw, town_objects as to
from yabitsu.meshgen import face_normals
from tests.test_town import _auto_town, _ground, _node, _way


def _ll(x, y):
    lat, lon = geo.from_local(float(x), float(y))
    return [float(lat), float(lon)]


def test_drivable_area_is_cut_at_closures():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"})])
    d = t.drivable()
    assert d.contains(Point(10.0, 0.0))                    # 本線
    assert d.contains(Point(0.0, -15.0))                   # 横断歩道の手前の腕
    assert not d.contains(Point(0.0, -35.0))               # 封鎖の外
    assert not d.contains(Point(0.0, 30.0))                # 北の腕も封鎖の外


def test_boundary_is_closed_except_main_road_mouth():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"})])
    d = t.drivable()
    walls = t.boundary_walls()
    blocks = [LineString(tw._xy([c["a"], c["b"]])) for c in t.closures]
    covered = walls + blocks
    from shapely.ops import unary_union
    open_part = d.exterior.difference(unary_union([g.buffer(0.1) for g in covered]))
    from yabitsu.crosssection import GUTTER_WIDTH
    mouth = 2 * (sec.hw_left[0] + GUTTER_WIDTH)
    assert open_part.length == pytest.approx(mouth, abs=0.7)   # 口の両わき 0.3m は壁を止める（車は通れない）
    c = open_part.centroid
    assert c.x == pytest.approx(t.main_skip[1], abs=1.5)


def test_barricade_spans_closure_line():
    g = _ground()
    a, b = np.array([-3.0, -25.0]), np.array([3.0, -25.0])
    parts = to.barricade_meshes(a, b, g.z, front=np.array([0.0, 1.0]))
    names = [p[0] for p in parts]
    assert any(n.startswith("barricade") for n in names) and any(n.startswith("cone") for n in names)
    bar = next(p for p in parts if p[0].startswith("barricade"))
    v = bar[1]
    assert v[:, 0].min() == pytest.approx(-3.0, abs=0.3) and v[:, 0].max() == pytest.approx(3.0, abs=0.3)
    cones = next(p for p in parts if p[0].startswith("cone"))[1]
    xs = np.unique(np.round(cones[np.isclose(cones[:, 2], cones[:, 2].max())][:, 0], 1))
    assert len(xs) >= 4 and np.allclose(np.diff(xs), to.CONE_EVERY, atol=0.2)
    assert cones[:, 1].max() < -25.0                       # コーンは柵の奥（front の反対側）


def test_block_walls_are_hidden_and_collidable():
    g = _ground()
    parts = to.wall_meshes([LineString([(0, 0), (10, 0)])], g.z)
    assert all(n.startswith("1WALL_town_block_") and rend is False for n, v, f, uv, mat, rend in parts)
    v = parts[0][1]
    assert v[:, 2].max() == pytest.approx(100.0 + to.WALL_H, abs=1e-6)


def test_signal_faces_given_direction():
    g = _ground()
    parts = to.signal_meshes([{"at": _ll(0, 0), "facing_deg": 90.0, "kind": "vehicle", "arm_m": 4.0, "height_m": 5.5}], g.z)
    head = next(p for p in parts if p[4] == "signal_head")
    v, f = head[1], head[2]
    n = face_normals(v, f)
    lit = n[:, 0] > 0.99
    assert lit.any()                                        # 表の面が東（+x）を向く


def test_auto_signals_for_each_stop_line_and_crosswalk_end():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"}),
                         _node([0, -24], 2, highway="traffic_signals")])
    sig = t.signals()
    assert sum(1 for s in sig if s["kind"] == "vehicle") == 1
    assert sum(1 for s in sig if s["kind"] == "pedestrian") == 2
    veh = next(s for s in sig if s["kind"] == "vehicle")
    assert abs(((veh["facing_deg"] - 180.0) + 180) % 360 - 180) < 10   # 北へ向かう車（交差点へ）に向けて南向き


def test_town_object_meshes_have_unique_names_and_barricades_face_drivable_side():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"}),
                         _node([0, -24], 2, highway="traffic_signals")])
    ms = t.object_meshes()
    names = [m[0] for m in ms]
    assert len(names) == len(set(names))
    assert sum(n.startswith("barricade_") and not n.endswith("_legs") for n in names) == len(t.closures)
    south = next(k for k, c in enumerate(t.closures) if tw._xy([c["a"]])[0][1] < -5)
    cones = next(m for m in ms if m[0] == f"cone_{south:03d}")[1]
    assert cones[:, 1].mean() < -25.0          # 南の柵のコーンは南（封鎖した道）の側。交差点に飛び出さない


def test_hoarding_is_tall_white_panel_along_points():
    g = _ground()
    (name, v, f, uv, mat, rend), = to.object_meshes([{"kind": "hoarding", "points": [_ll(0, 0), _ll(20, 0)], "height_m": 3.0}], g.z)
    assert mat == "hoarding" and v[:, 2].max() == pytest.approx(103.0) and np.ptp(v[:, 0]) == pytest.approx(20.0)


def test_town_pits_are_inside_drivable_area():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"})])
    t.spec["pits"] = [_ll(20.0 + 6 * i, 1.5) + [90.0] for i in range(4)]
    (pad,) = t.pads()
    assert pad["name"] == "nagoki_pits" and len(pad["pit_points"]) == 4
    d = t.drivable()
    assert all(d.contains(Point(x, y)) for x, y, z, h in pad["pit_points"])
    assert abs(pad["pit_points"][0][3] - 0.0) < 1e-6        # 方位 90°（東）＝ +x の向き


def test_road_ground_follows_nearest_road_height():
    from tests.test_meshgen import _straight
    sec = _straight(201); sec.z[:] = 100.0 + 0.1 * sec.s
    z = to.road_ground(sec)
    assert float(z(np.array([50.0]), np.array([6.0]))[0]) == pytest.approx(105.0, abs=0.11)


def test_buildings_inside_a_drivable_lot_are_not_swallowed():
    from shapely.geometry import box
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"})])
    lot = box(10, 8, 40, 30); shop = box(18, 14, 30, 24)
    t.spec["drivable_extra"] = [{"points": [list(_ll(x, y)) for x, y in lot.exterior.coords]}]
    assert t.paved_avoid().intersection(shop).area < 0.01        # 建物を消すのに使う範囲に、店の所は入らない


def test_pole_sign_has_pole_and_two_faced_panel_with_its_own_texture(tmp_path):
    g = _ground()
    signs = [{"text": "セブン\nイレブン", "bg": "#FFFFFF", "fg": "#008C45", "at": _ll(0, 0), "facing_deg": 0.0,
              "w_m": 1.2, "h_m": 2.4, "z_m": 3.5, "pole_h": 6.0}]
    ms = to.pole_sign_meshes(signs, g.z)
    mats = {m[4] for m in ms}
    assert "pole_grey" in mats and "polesign_0" in mats
    panel = next(m for m in ms if m[4] == "polesign_0")[1]
    assert panel[:, 2].min() == pytest.approx(103.5) and panel[:, 2].max() == pytest.approx(105.9)
    paths = to.write_pole_sign_textures(signs, tmp_path)
    assert (tmp_path / "polesign_0.png").exists() and paths["polesign_0"] == "textures/polesign_0.png"


def test_cones_stand_behind_the_barricade_away_from_the_road():
    g = _ground()
    parts = to.barricade_meshes(np.array([-3.0, -25.0]), np.array([3.0, -25.0]), g.z, front=np.array([0.0, 1.0]))
    cones = next(p for p in parts if p[0].startswith("cone"))[1]
    assert cones[:, 1].max() < -25.0 + 0.01                 # front（+y＝道の側）の反対へ


def test_bus_stops_standing_on_the_road_are_moved_just_outside_it():
    """OSM のバス停の点が路面の中にあるとき（3480m は車線の中）、道の縁＋側溝の外へずらす。道の外の物はそのまま。"""
    from tests.test_meshgen import _straight
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()                                   # 本線は y=0 に沿って +x 向き、片側 3.6m
    hw = float(sec.hw_left[0])
    objs = [{"kind": "bus_stop", "points": [_ll(50.0, -2.0)]}, {"kind": "bus_stop", "points": [_ll(80.0, 7.0)]},
            {"kind": "stele", "points": [_ll(60.0, 1.0)]}]
    out = to.off_road(objs, sec)
    (x0, y0), (x1, y1), (x2, y2) = [geo.to_local(*o["points"][0]) for o in out]
    assert x0 == pytest.approx(50.0, abs=0.05) and y0 == pytest.approx(-(hw + GUTTER_WIDTH + to.OFF_ROAD_M), abs=0.05)
    assert (x1, y1) == pytest.approx((80.0, 7.0), abs=1e-6) and (x2, y2) == pytest.approx((60.0, 1.0), abs=1e-6)


def test_curve_mirror_and_road_signs():
    """カーブミラー（オレンジの柱＋丸い鏡）と道路標識（速度 30・警戒の黄色いひし形）を、柱の上に道の方へ向けて立てる。"""
    flat = lambda x, y: np.full(np.shape(x), 100.0)
    objs = [{"kind": "mirror", "points": [_ll(10.0, 5.0)], "facing_deg": 180.0},
            {"kind": "road_sign", "sign": "speed30", "points": [_ll(20.0, 5.0)], "facing_deg": 180.0},
            {"kind": "road_sign", "sign": "warning", "points": [_ll(30.0, 5.0)], "facing_deg": 180.0}]
    ms = {m[4]: m for m in to.object_meshes(objs, flat, prefix="s4_obj")}
    assert {"pole_orange", "mirror", "sign_speed30", "sign_warning"} <= set(ms)
    assert ms["mirror"][1][:, 2].max() > 102.0 and ms["sign_speed30"][1][:, 2].min() > 101.5


def test_pole_signs_with_a_prefix_do_not_clash_with_the_town_signs(tmp_path):
    flat = lambda x, y: np.full(np.shape(x), 100.0)
    s = [{"text": "札掛", "bg": "#3A2E26", "fg": "#FFFFFF", "at": _ll(5.0, 5.0), "facing_deg": 0.0}]
    a = {m[0] for m in to.pole_sign_meshes(s, flat)}; b = {m[0] for m in to.pole_sign_meshes(s, flat, prefix="s4")}
    assert not a & b
    assert set(to.write_pole_sign_textures(s, tmp_path, prefix="s4")) == {m[4] for m in to.pole_sign_meshes(s, flat, prefix="s4") if "sign" in m[4] and m[4] != "pole_grey"}


def test_retaining_wall_object_is_collidable_concrete_with_fence_on_top():
    """交差点の角の擁壁（宮ヶ瀬北原）：当たり判定のあるコンクリートの壁と、上の金網のフェンス"""
    g = _ground()
    parts = to.object_meshes([{"kind": "retaining_wall", "points": [_ll(0, 0), _ll(20, 0)], "height_m": 2.5, "fence_m": 1.5}], g.z)
    wall = next(p for p in parts if p[4] == "concrete")
    assert wall[0].startswith("1WALL_") and wall[5] is True
    assert wall[1][:, 2].max() == pytest.approx(102.5) and np.ptp(wall[1][:, 0]) == pytest.approx(20.0, abs=0.5)
    fence = next(p for p in parts if p[4] == "railing_bars")
    assert fence[1][:, 2].min() == pytest.approx(102.5) and fence[1][:, 2].max() == pytest.approx(104.0)


def test_vehicle_signal_arm_length_can_be_set_in_the_spec():
    """宮ヶ瀬北原の信号は柱の横に灯器（腕が短い）。spec の vehicle_arm_m で自動の信号の腕の長さを決める"""
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"}),
                         _node([0, -24], 2, highway="traffic_signals")])
    t.spec["vehicle_arm_m"] = 1.2
    veh = next(s for s in t.signals() if s["kind"] == "vehicle")
    assert veh["arm_m"] == pytest.approx(1.2)


def test_pole_sign_materials_carry_the_junction_name():
    """2 つ目の交差点の立て看板の絵が、名古木の polesign_0 とぶつからない"""
    sec, t = _auto_town()
    t.prefix = "kitahara"
    t.spec["pole_signs"] = [{"text": "64", "at": _ll(5, 5)}]
    mats = [m[4] for m in t.object_meshes()]
    assert "kitahara_polesign_0" in mats and "polesign_0" not in mats


def test_pole_sign_without_pole_is_just_the_board():
    """信号の腕に付ける交差点名の板：柱も当たり判定もない"""
    g = _ground()
    parts = to.pole_sign_meshes([{"text": "宮ヶ瀬北原", "at": _ll(0, 0), "no_pole": True, "z_m": 5.0, "h_m": 0.45}], g.z)
    assert [p[0] for p in parts] == ["town_polesign_0"]
    assert parts[0][1][:, 2].min() == pytest.approx(105.0)


def test_signal_name_plates_hang_on_the_vehicle_signal_arms():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"}),
                         _node([0, -24], 2, highway="traffic_signals")])
    t.spec["signal_plate"] = {"text": "宮ヶ瀬北原\nMiyagase Kitahara"}
    veh = [s for s in t.signals() if s["kind"] == "vehicle"]
    plates = [s for s in t.pole_signs() if s.get("no_pole")]
    assert len(plates) == len(veh) == 1
    p, v = tw._xy([plates[0]["at"]])[0], tw._xy([veh[0]["at"]])[0]
    assert np.linalg.norm(p - v) == pytest.approx(min(1.0, 0.4 * veh[0]["arm_m"]), abs=0.05)
    assert plates[0]["z_m"] == pytest.approx(veh[0]["height_m"] - 0.45)
    assert plates[0]["facing_deg"] == pytest.approx(veh[0]["facing_deg"])


def test_town_retaining_wall_height_is_measured_from_the_sidewalk_not_the_hill_behind():
    """擁壁は歩道の外に立つ。裏の地面（山）が高くても、壁の高さは歩道の上から（宮ヶ瀬北原の鳥居原側で壁が高すぎた）"""
    from tests.test_town import _ground, _town
    g = _ground(lambda X, Y: 100.0 + np.maximum(0.0, -X - 4.0) * 5.0)   # 西の歩道の外 0.5m から山
    spec = {"roads": {"7": {"width_m": 4.0, "sidewalk_l": 1.5, "sidewalk_r": 1.5}}, "corner_radius_m": 6.0,
            "closures": [], "markings": [], "auto": True,
            "objects": [{"kind": "retaining_wall", "points": [_ll(-4.6, -14.0), _ll(-4.6, -8.0)], "height_m": 2.5}]}
    sec, t = _town(spec, ways=[_way([[0, -60], [0, 60]], wid=7)], ground=g)
    wall = next(m for m in t.object_meshes() if m[0] == "1WALL_town_obj_retaining")
    assert wall[1][:, 2].max() == pytest.approx(100.0 + tw.CURB_H + 2.5, abs=0.05)


def test_second_junction_pits_are_named_after_the_junction():
    """宮ヶ瀬北原のピットが名古木の nagoki_pits と同じ名前にならない（レイアウトはピットを名前で選ぶ）"""
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"})])
    t.spec["pits"] = [_ll(20.0 + 6 * i, 1.5) + [90.0] for i in range(4)]
    t.prefix = "kitahara"
    assert t.pads()[0]["name"] == "kitahara_pits"
