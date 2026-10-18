import numpy as np
import pytest
from shapely.geometry import LineString, Point

from yabitsu import terrain as tr, water
from yabitsu.start_lot import road_formation
from tests.test_meshgen import _straight
from tests.test_side_roads import _Flat, _way


def _scene(bridge=False):
    """本線は y=0 に沿って +x 向き（高さ 100）。沢が北（y=+90）から南（y=-90）へ、x=100 で本線を横切って流れる。地面は北が高い。"""
    sec = _straight()
    if bridge:
        sec.bridge[95:106] = True
    ways = [_way([[100, 90], [100, -90]], id=7, highway=None, waterway="stream")]
    for w in ways:
        w["tags"].pop("highway")
    return sec, ways, _Flat(97.0, gy=0.05)


def test_water_surface_only_goes_down_stream():
    sec, ways, smp = _scene()
    st, = water.from_osm(ways, sec, smp)
    assert np.all(np.diff(st.z) <= 1e-9)


def test_water_is_never_drawn_on_a_road():
    sec, ways, smp = _scene()
    form = road_formation(sec, 0.0, float(sec.s[-1]))
    ms = water.meshes(water.from_osm(ways, sec, smp), form)
    assert ms
    near = form.buffer(0.5)
    for name, v, f, uv, mat, r in ms:
        assert not any(near.contains(Point(x, y)) for x, y in v[:, :2]), name


def test_water_passes_under_a_bridge_below_the_deck():
    sec, ways, smp = _scene(bridge=True)
    st, = water.from_osm(ways, sec, smp)
    t = st.line.project(Point(100, 0))
    assert st.z_at(t) <= float(sec.z[100]) - 1.5
    form = road_formation(sec, 0.0, float(sec.s[-1]))
    v = np.concatenate([m[1] for m in water.meshes([st], water.road_area(sec, form)) if m[4] == "water"])
    assert any(form.contains(Point(x, y)) for x, y in v[:, :2])          # 橋の下には水面が見える


def test_channel_keeps_terrain_below_the_water():
    sec, ways, smp = _scene()
    st, = water.from_osm(ways, sec, smp)
    X = np.array([[100.0]]); Y = np.array([[60.0]])
    Z, _ = tr.carve(X, Y, np.full_like(X, 200.0), sec, channels=[st.channel()])
    assert Z[0, 0] <= st.z_at(st.line.project(Point(100, 60))) - 0.2


def test_falls_stand_on_the_stream():
    sec, ways, smp = _scene()
    weir = _way([[97, 50], [103, 50]], id=8, highway=None, waterway="weir")
    weir["tags"].pop("highway")
    streams = water.from_osm(ways + [weir], sec, smp)
    falls = water.falls_from_osm(ways + [weir], streams)
    assert len(falls) == 1
    ms = water.feature_meshes(falls, [])
    c = np.concatenate([m[1] for m in ms])[:, :2].mean(0)
    assert streams[0].line.distance(Point(*c)) <= 3.0


class _Step(_Flat):
    """北ほど高い地面。ただし y < 20 の所だけ 3m 高い（下流の way の始まりが、上流の way の終わりより高くなる）。"""
    def sample(self, lat, lon):
        from yabitsu import geo
        x, y = geo.to_local(np.asarray(lat), np.asarray(lon))
        return 90.0 + 0.05 * np.asarray(y) + np.where(np.asarray(y) <= 20.5, 3.0, 0.0)


def test_water_keeps_going_down_where_one_way_joins_the_next():
    """OSM では1本の川が何本もの way に分かれる。下流の way の水面が、上流の way の終わりより高く始まらない。"""
    sec = _straight()
    ws = [_way([[60, 90], [60, 20]], id=1, highway=None, waterway="stream"), _way([[60, 20], [60, -90]], id=2, highway=None, waterway="stream")]
    for w in ws:
        w["tags"].pop("highway")
    a, b = sorted(water.from_osm(ws, sec, _Step(90.0)), key=lambda s: -s.line.coords[0][1])
    assert b.z[0] <= a.z[-1] + 1e-9 and np.all(np.diff(b.z) <= 1e-9)


def _lake_rel():
    """外周 100x100 の正方形を2本の way に分けた湖（x 20〜120, y 20〜120）と、中の島（inner）。"""
    from yabitsu import geo
    def node(i, x, y):
        la, lo = geo.from_local(x, y); return {"type": "node", "id": i, "lat": float(la), "lon": float(lo)}
    ns = [node(1, 20, 20), node(2, 120, 20), node(3, 120, 120), node(4, 20, 120), node(5, 60, 60), node(6, 70, 60), node(7, 70, 70), node(8, 60, 70)]
    ws = [{"type": "way", "id": 11, "nodes": [1, 2, 3]}, {"type": "way", "id": 12, "nodes": [3, 4, 1]},
          {"type": "way", "id": 13, "nodes": [5, 6, 7, 8, 5]}]
    rel = {"type": "relation", "id": 99, "tags": {"natural": "water", "name": "宮ヶ瀬湖"},
           "members": [{"type": "way", "ref": 11, "role": "outer"}, {"type": "way", "ref": 12, "role": "outer"},
                       {"type": "way", "ref": 13, "role": "inner"}]}
    return ns + ws + [rel]


def test_lake_polygon_from_relation():
    poly = water.lake_from_osm(_lake_rel())
    assert poly.area == pytest.approx(100 * 100 - 10 * 10, rel=0.01)
    assert not poly.contains(Point(65, 65)) and poly.contains(Point(30, 30))


def test_lake_is_below_every_road_and_bridge():
    sec = _straight()   # 本線 y=0、高さ 100
    poly = water.lake_from_osm(_lake_rel()).union(Point(100, 0).buffer(15))   # 湖が道にかかる
    form = road_formation(sec, 0.0, float(sec.s[-1]))
    ms = water.lake_meshes(poly, 95.0, form)
    v = np.concatenate([m[1] for m in ms])
    assert np.allclose(v[:, 2], 95.0 + water.LIFT) and not any(form.buffer(0.5).contains(Point(x, y)) for x, y in v[:, :2])


def test_lake_and_streams_meet_without_a_step():
    """湖に入る川は、湖の中では描かず、水面は湖の水位より下がらない（段差なくつながる）。"""
    sec, ways, smp = _scene()
    lake = Point(100, -80).buffer(25)        # 川の終わり（下流）が湖に入る
    st, = water.attach_to_lake(water.from_osm(ways, sec, smp), lake, 99.0)
    assert not st.line.intersects(lake.buffer(-0.5)) and st.z.min() >= 99.0 - 1e-9


def test_terrain_under_the_lake_is_below_the_water():
    from yabitsu import terrain as tr
    sec = _straight()
    X, Y = np.meshgrid(np.arange(80.0, 121.0, 5.0), np.arange(30.0, 61.0, 5.0))
    Z, _ = tr.carve(X, Y, np.full(X.shape, 95.0), sec, lake=(Point(100, 45).buffer(30), 95.0))
    assert Z.max() <= 95.0 - tr.LAKE_DEPTH + 1e-6


def test_manual_falls_stand_on_the_ground_and_face_the_road():
    sec = _straight()
    from yabitsu import geo
    la, lo = geo.from_local(100.0, 10.0)
    fl = water.manual_falls([{"at": [la, lo], "height_m": 4.0, "width_m": 2.0, "facing_deg": None}], sec, lambda x, y: np.full(np.shape(x), 101.0))
    assert len(fl) == 1 and fl[0]["z_bottom"] == pytest.approx(101.0) and fl[0]["z_top"] == pytest.approx(105.0)
    assert fl[0]["down"][1] < -0.9            # 道（南）の方へ落ちる
    ms = water.feature_meshes(fl, [])
    assert any(m[4] == "water_fall" for m in ms)
