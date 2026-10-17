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
    assert cones[:, 1].max() > -25.0                       # コーンは柵の手前（front の側）


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
    assert cones[:, 1].mean() > -25.0          # 南の柵のコーンは北（交差点）の側
