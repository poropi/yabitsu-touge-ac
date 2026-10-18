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
