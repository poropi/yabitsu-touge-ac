import numpy as np
import pytest
from shapely.geometry import Point

from yabitsu import geo, town as tw
from yabitsu.crosssection import GUTTER_WIDTH, surface_height
from yabitsu.meshgen import face_normals
from yabitsu.town_ground import TownGround
from tests.test_meshgen import _straight


def _way(xy, wid=1, **tags):
    lat, lon = geo.from_local(*np.asarray(xy, float).T)
    return {"type": "way", "id": wid, "tags": {"highway": "residential", **tags},
            "geometry": [{"lat": float(a), "lon": float(b)} for a, b in zip(lat, lon)]}


def _ground(f=lambda X, Y: np.full_like(X, 100.0)):
    xs = np.arange(-120.0, 121.0); ys = np.arange(-120.0, 121.0)
    X, Y = np.meshgrid(xs, ys)
    return TownGround(xs, ys, f(X, Y))


def _town(spec=None, ways=None, ground=None, radius=50.0):
    sec = _straight(201)   # 本線：交差点の中心 (0,0) から +x へ、幅 6、高さ 100
    ways = ways if ways is not None else [_way([[0, -60], [0, 60]], wid=7)]   # 横道：y 方向、幅 4
    spec = spec or {"roads": {}, "corner_radius_m": 6.0, "closures": [], "markings": []}
    return sec, tw.Town(spec, ways, sec, ground or _ground(), (0.0, 0.0), radius)


def test_carriageway_rounds_corners():
    sec, t = _town()
    assert t.carriageway.contains(Point(2.0 + 1.4, 3.0 + GUTTER_WIDTH + 1.4))   # 角の内側（角から 約2m）
    assert not t.carriageway.contains(Point(2.0 + 8.0, 3.0 + GUTTER_WIDTH + 8.0))


def test_sidewalk_runs_beside_carriageway():
    sec, t = _town({"roads": {"7": {"width_m": 4.0, "sidewalk_l": 2.0, "sidewalk_r": 2.0}}, "corner_radius_m": 6.0,
                    "closures": [], "markings": []})
    assert t.sidewalk.contains(Point(2.0 + 1.0, -30.0))
    assert not t.sidewalk.contains(Point(2.0 - 0.5, -30.0))


def test_footways_are_not_roads():
    sec, t = _town(ways=[_way([[0, -60], [0, 60]], wid=7), _way([[30, 5], [30, 40]], wid=8, highway="footway")])
    assert not t.carriageway.contains(Point(30.0, 20.0))


def test_surfaces_follow_town_ground():
    g = _ground(lambda X, Y: 100.0 + 0.1 * X)
    sec, t = _town({"roads": {"7": {"width_m": 4.0, "sidewalk_l": 2.0, "sidewalk_r": 2.0}}, "corner_radius_m": 6.0,
                    "closures": [], "markings": []}, ground=g)
    for name, v, f, uv, mat, rend in t.meshes():
        if name.startswith("1ROAD_town_carriageway"):
            assert np.allclose(v[:, 2], g.z(v[:, 0], v[:, 1]), atol=1e-6)
            assert (face_normals(v, f)[:, 2] > 0).all()
        if name.startswith("1ROAD_town_sidewalk"):
            assert np.allclose(v[:, 2], g.z(v[:, 0], v[:, 1]) + tw.CURB_H, atol=1e-6)


def test_no_vertex_far_from_ground_inside_cells():
    g = _ground(lambda X, Y: 100.0 + 0.004 * (X ** 2 + Y ** 2))
    sec, t = _town(ground=g)
    for name, v, f, uv, mat, rend in t.meshes():
        if name.startswith("1ROAD_town_carriageway"):
            c = v[f].mean(1)
            assert np.abs(c[:, 2] - g.z(c[:, 0], c[:, 1])).max() <= 0.03


def test_main_skip_ends_where_road_leaves_circle():
    sec, t = _town(radius=50.0)
    assert t.main_skip[0] == 0.0 and t.main_skip[1] == pytest.approx(51.0, abs=2.0)


def test_markings_sit_on_surface_and_bars_parallel():
    a, b = geo.from_local(np.array([-2.0, 2.0]), np.array([-20.0, -20.0]))
    spec = {"roads": {}, "corner_radius_m": 6.0, "closures": [],
            "markings": [{"kind": "crosswalk", "points": [[float(a[0]), float(b[0])], [float(a[1]), float(b[1])]]}]}
    sec, t = _town(spec)
    polys = [p for p, mat in tw.marking_polygons(spec)]
    assert polys and all(abs(np.ptp(np.asarray(p.exterior.coords)[:, 1]) - tw.CROSSWALK_W) < 0.01 for p in polys)
    assert all(np.ptp(np.asarray(p.exterior.coords)[:, 0]) == pytest.approx(tw.BAR_W, abs=0.01) for p in polys)
    lines = [m for m in t.meshes() if m[0].startswith("LINES_town")]
    v = lines[0][1]
    assert np.allclose(v[:, 2], 100.0 + tw.MARK_LIFT, atol=1e-6)


def test_add_road_skip_range_makes_no_ribbon():
    from yabitsu import assembly
    from yabitsu.manifest import Manifest
    import tempfile
    sec = _straight(201)
    with tempfile.TemporaryDirectory() as d:
        m = Manifest(d)
        assembly.add_road(m, sec, skip=(0.0, 50.0))
        for name in m.meshes:
            v = np.load(f"{d}/{m.meshes[name]['file']}")["verts"]
            assert v[:, 0].min() >= 50.0 - 1e-6, name


def test_join_town_to_ribbon_has_no_step():
    from yabitsu import town_ground
    from tests.test_town_ground import IDENT, _Dem
    sec = _straight(201)
    g = town_ground.build(sec, _Dem(lambda x, y: np.full_like(x, 98.0)), (0.0, 0.0), radius=120.0, unproject=IDENT)
    sec, t = _town(radius=50.0, ground=g)
    s_exit = t.main_skip[1]
    for name, v, f, uv, mat, rend in t.meshes():
        if name.startswith("1ROAD_town_carriageway"):
            m = (np.abs(v[:, 0] - s_exit) <= 1.0) & (np.abs(v[:, 1]) <= 3.0)
            assert m.any()
            i = int(round(s_exit))
            want = surface_height(v[m, 1], sec.z[i], sec.slope[i], sec.crown[i])
            assert np.abs(v[m, 2] - want).max() <= 0.02


def _node(xy, nid, **tags):
    lat, lon = geo.from_local(float(xy[0]), float(xy[1]))
    return {"type": "node", "id": nid, "lat": float(lat), "lon": float(lon), "tags": tags}


def _auto_town(extra=(), radius=50.0):
    ways = [_way([[0, -60], [0, 60]], wid=7), *extra]
    spec = {"roads": {"7": {"width_m": 4.0, "sidewalk_l": 1.5, "sidewalk_r": 1.5}}, "corner_radius_m": 6.0,
            "closures": [], "markings": [], "auto": True}
    return _town(spec, ways=ways, radius=radius)


def test_auto_crosswalk_spans_road_at_osm_crossing():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"})])
    cw = [m for m in t.markings if m["kind"] == "crosswalk"]
    assert len(cw) == 1
    a, b = town_xy = tw._xy(cw[0]["points"])
    assert abs(a[1] + 20) < 0.3 and abs(b[1] + 20) < 0.3
    assert abs(abs(a[0] - b[0]) - 4.0) < 0.3


def test_auto_closure_beyond_crosswalk_on_side_arm():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"})])
    south = [c for c in t.closures if tw._xy([c["a"]])[0][1] < -5]
    assert len(south) == 1
    (ax, ay), (bx, by) = tw._xy([south[0]["a"], south[0]["b"]])
    assert ay == pytest.approx(-20 - tw.CROSSWALK_W / 2 - tw.CLOSURE_GAP_M, abs=0.5)
    assert min(ax, bx) <= -(2.0 + 1.5) and max(ax, bx) >= 2.0 + 1.5


def test_side_street_on_main_gets_closure_at_mouth():
    sec, t = _auto_town([_way([[100, 3.6], [100, 40]], wid=9)], radius=150.0)
    mouth = [c for c in t.closures if abs(tw._xy([c["a"]])[0][0] - 100) < 5]
    assert len(mouth) == 1
    (ax, ay), (bx, by) = tw._xy([mouth[0]["a"], mouth[0]["b"]])
    assert 3.6 < ay < 3.6 + 3.0 and abs(ay - by) < 0.3


def test_signalised_crossing_gets_stop_line_on_approach_side():
    sec, t = _auto_town([_node([0, -20], 1, highway="crossing", **{"crossing:markings": "zebra"}),
                         _node([0, -24], 2, highway="traffic_signals")])
    sl = [m for m in t.markings if m["kind"] == "stop_line"]
    assert len(sl) == 1
    (ax, ay), (bx, by) = tw._xy(sl[0]["points"])
    assert ay == pytest.approx(-20 - tw.CROSSWALK_W / 2 - tw.STOP_GAP_M, abs=0.3)
    # 交差点へ向かう車（+y 向き）の左側＝ -x 側の半分
    assert max(ax, bx) <= 0.05 and min(ax, bx) == pytest.approx(-2.0, abs=0.2)


def test_auto_lines_on_major_roads_stop_short_of_the_junction():
    sec, t = _auto_town([_way([[-60, -10], [-5, -10]], wid=11, highway="trunk")], radius=80.0)
    centre = [m for m in t.markings if m["kind"] == "centre_line" and m.get("road") == "11"]
    edges = [m for m in t.markings if m["kind"] == "lane_line" and m.get("road") == "11"]
    assert centre and len(edges) == 2 and centre[0]["color"] == "orange"
    xs = tw._xy(centre[0]["points"])[:, 0]
    assert xs.max() <= -(tw.R_BOX - 1.0) + 1e-6 or np.hypot(xs.max(), -10) >= tw.R_BOX - 1.0
    assert not [m for m in t.markings if m.get("road") == "7" and m["kind"] == "centre_line"]   # 住宅地の道には引かない
    main = [m for m in t.markings if m.get("road") == "main" and m["kind"] == "centre_line"]
    assert main and tw._xy(main[0]["points"])[:, 0].min() >= tw.R_BOX - 1.0


def test_arm_that_passes_beside_the_centre_is_still_closed():
    # 中心から 18m 離れて通る連絡路（R_ARM の円に届かない）
    sec, t = _auto_town([_way([[-40, -18], [40, -18]], wid=12, highway="secondary_link")])
    arm = [c for c in t.closures if c["name"] == "arm_12"]
    assert len(arm) == 2


def test_edge_lines_do_not_run_inside_other_roads_and_links_get_none():
    sec, t = _auto_town([_way([[-60, -10], [-5, -10]], wid=11, highway="trunk"),
                         _way([[-60, -14], [-5, -10]], wid=12, highway="trunk"),       # 同じ道の別の線（重なる）
                         _way([[-50, -30], [-10, -12]], wid=13, highway="secondary_link")], radius=80.0)
    assert not [m for m in t.markings if m.get("road") == "13"]
    from shapely.geometry import LineString
    other = tw._way_line(_way([[-60, -14], [-5, -10]], wid=12)).buffer(tw.ROAD_DEFAULTS["trunk"][0] / 2 - 0.4)
    for m in t.markings:
        if m.get("road") == "11" and m["kind"] == "lane_line":
            assert LineString(tw._xy(m["points"])).intersection(other).length < 0.5


def test_lots_and_their_lines_are_drawn_on_the_ground_outside_the_road():
    from shapely.geometry import box
    sec, t = _town()
    lot = box(20, 10, 40, 30); line = box(25, 12, 25.15, 17)
    ll = lambda g: [list(geo.from_local(x, y)) for x, y in g.exterior.coords]
    ms = t.lot_meshes({"lots": [ll(lot)], "lines": [ll(line)]})
    names = [m[0] for m in ms]
    assert any(n.startswith("town_lot_") for n in names) and any(n.startswith("LINES_town_lot_") for n in names)
    v = next(m for m in ms if m[0].startswith("town_lot_"))[1]
    assert np.allclose(v[:, 2], 100.0 + tw.LOT_LIFT, atol=1e-6)
    lv = next(m for m in ms if m[0].startswith("LINES_town_lot_"))[1]
    assert lv[:, 2].min() > 100.0 + tw.LOT_LIFT


def test_add_road_skips_several_ranges():
    from yabitsu import assembly
    from yabitsu.manifest import Manifest
    import tempfile
    sec = _straight(201)
    with tempfile.TemporaryDirectory() as d:
        m = Manifest(d)
        assembly.add_road(m, sec, skip=[(0.0, 50.0), (100.0, 140.0)])
        xs = np.concatenate([np.load(f"{d}/{m.meshes[n]['file']}")["verts"][:, 0] for n in m.meshes if n.startswith("1ROAD_s1_") and "gutter" not in n])
        assert xs.min() >= 50.0 - 1e-6 and not ((xs > 100.5) & (xs < 139.5)).any()
        assert xs.max() == pytest.approx(200.0)


def test_main_sidewalk_follows_the_road_width_inside_the_town_only():
    """本線の歩道は、町の中のその場所の道幅の外に、決めた幅で付く（コースのどこか遠くの広い所＝バス停の切り込みに引っぱられない）。"""
    sec = _straight(201)
    sec.hw_left[150:] = 7.75; sec.hw_right[150:] = 7.75          # 町の外（遠く）に広い所
    spec = {"roads": {"main": {"sidewalk_l": 2.0, "sidewalk_r": 2.0}}, "corner_radius_m": 6.0, "closures": [], "markings": []}
    t = tw.Town(spec, [_way([[0, -60], [0, 60]], wid=7)], sec, _ground(), (0.0, 0.0), 50.0)
    edge = float(sec.hw_left[30]) + GUTTER_WIDTH
    assert t.sidewalk.contains(Point(30.0, edge + 1.0))
    assert not t.sidewalk.contains(Point(30.0, edge + 2.0 + 0.5))
