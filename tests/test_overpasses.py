import numpy as np
import pytest

from yabitsu import geo, overpasses as op
from yabitsu.crosssection import GUTTER_WIDTH
from tests.test_meshgen import _straight
from tests.test_side_roads import _Flat, _way


def test_overpass_crosses_above_main_road():
    sec = _straight()
    o, = op.from_osm([_way([[100, -80], [100, 80]], id=9, highway="motorway", bridge="yes", layer="1", lanes="2")],
                     sec, _Flat(95.0))
    assert o.s_cross == pytest.approx(100.0, abs=0.5)
    deck = o.meshes()[0]
    v = deck[1]
    z_road = float(max(sec.edge_z(1)[100], sec.edge_z(-1)[100]))
    assert v[:, 2].min() == pytest.approx(z_road + op.CLEARANCE_M, abs=0.01)   # 桁の下面＝路面＋クリアランス
    piers = next(m for m in o.meshes() if m[0].startswith("1WALL_"))
    pv = piers[1]
    assert np.abs(pv[:, 1]).min() >= sec.hw_left[100] + GUTTER_WIDTH + 1.0 - 1e-6
    assert pv[:, 2].min() <= 95.0   # 柱は地面から立つ


def test_ground_level_crossing_is_not_an_overpass():
    sec = _straight()
    assert op.from_osm([_way([[100, -80], [100, 80]], highway="primary")], sec, _Flat()) == []


def test_override_sets_clearance():
    sec = _straight()
    o, = op.from_osm([_way([[100, -80], [100, 80]], id=9, highway="motorway", bridge="yes", layer="1")], sec, _Flat(95.0),
                     overrides={9: {"clearance_m": 8.0}})
    assert o.clearance == 8.0


def test_deck_faces_point_outward():
    from yabitsu.meshgen import face_normals
    sec = _straight()
    o, = op.from_osm([_way([[100, -80], [100, 80]], id=9, highway="motorway", bridge="yes", layer="1")], sec, _Flat(95.0))
    name, v, f, uv, mat, rend = o.meshes()[0]
    n = face_normals(v, f); c = v[f].mean(1)
    bottom = np.isclose(c[:, 2], o.z_under)
    top = np.isclose(c[:, 2], o.z_under + op.GIRDER_DEPTH + op.SLAB) & (np.abs(c[:, 0] - 100) < o.width / 2 - 0.5)
    assert (n[bottom, 2] < -0.99).all() and (n[top, 2] > 0.99).all()


def test_bridge_on_the_course_itself_is_not_an_overpass():
    """県道70号そのものの橋（S4 の札掛橋など、OSM では layer=1）は、道の上を渡る橋ではない（空に橋が浮いていた）。"""
    sec = _straight()
    ws = [_way([[90, 0], [110, 0]], id=5, highway="primary", ref="70", bridge="yes", layer="1"),
          _way([[150, 0.5], [170, 0.3]], id=6, highway="primary", bridge="yes", layer="1")]   # 名前・ref のない本線の橋（線に沿う）
    assert op.from_osm(ws, sec, _Flat(95.0)) == []
