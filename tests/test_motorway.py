import numpy as np
import pytest
from shapely.geometry import Point

from yabitsu import motorway as mw
from yabitsu.crosssection import GUTTER_WIDTH
from tests.test_meshgen import _straight
from tests.test_side_roads import _Flat, _way


def _net(ground=95.0):
    """本線（x 軸、高さ 100）を、x=100 で南北に渡る高速道路。南は y=-150、北は y=+150 でトンネルに入る。"""
    sec = _straight()
    ways = [_way([[100, -400], [100, -150]], id=1, highway="motorway", tunnel="yes", layer="-1", oneway="yes"),
            _way([[100, -150], [100, -60]], id=2, highway="motorway", oneway="yes", lanes="2"),
            _way([[100, -60], [100, 60]], id=3, highway="motorway", bridge="yes", layer="2", oneway="yes", lanes="2"),
            _way([[100, 60], [100, 150]], id=4, highway="motorway", oneway="yes", lanes="2"),
            _way([[100, 150], [100, 400]], id=5, highway="motorway", tunnel="yes", layer="-1", oneway="yes")]
    return sec, ways, mw.from_osm(ways, sec, _Flat(ground), overrides={3: {"clearance_m": 7.0, "width_m": 12.0}})


def test_chain_runs_from_portal_to_portal():
    sec, ways, (m,) = _net()
    assert m.way_ids == {2, 3, 4}
    a, b = np.array(m.axis.coords[0]), np.array(m.axis.coords[-1])
    assert np.allclose(a, [100, -150], atol=0.01) and np.allclose(b, [100, 150], atol=0.01)
    assert m.portals == (True, True)


def test_height_meets_ground_at_portals_and_clears_main_road():
    sec, ways, (m,) = _net()
    assert m.z_top(0.0) == pytest.approx(95.0, abs=0.01) and m.z_top(m.axis.length) == pytest.approx(95.0, abs=0.01)
    z_road = float(max(sec.edge_z(1)[100], sec.edge_z(-1)[100]))
    assert m.z_under(m.t_cross) == pytest.approx(z_road + 7.0, abs=0.01)


def test_piers_only_where_high_and_outside_main_road():
    sec, ways, (m,) = _net()
    piers = next(x for x in m.meshes() if x[0].startswith("1WALL_mw_"))[1]
    assert np.abs(piers[:, 1]).min() >= sec.hw_left[100] + GUTTER_WIDTH + 1.0 - 1e-6
    for x, y, z in piers[np.isclose(piers[:, 2], piers[:, 2].max(), atol=3.0)]:
        t = m.axis.project(Point(x, y))
        assert m.z_top(t) - 95.0 > mw.VIADUCT_MIN_H - 0.5


def test_pads_carve_terrain_only_where_low():
    sec, ways, (m,) = _net()
    pads = m.pads()
    assert pads and all(p["kind"] == "motorway" and len(p["plane"]) == 5 for p in pads)
    for p in pads:
        c = np.asarray(p["polygon"]).mean(0)
        assert m.z_top(m.axis.project(Point(*c))) - 95.0 <= mw.VIADUCT_MIN_H + 0.5


def test_portal_faces_have_dark_openings():
    sec, ways, (m,) = _net()
    mats = {x[4] for x in m.meshes()}
    assert "tunnel_dark" in mats and "concrete" in mats


def test_piers_stay_off_the_sidewalks_too():
    """本線に歩道（2.5m）がある所では、柱は歩道の外 PIER_CLEAR から（s≈2700 の新東名の柱が歩道の上にあった。指摘 8）。"""
    sec = _straight()
    ways = _net()[1]
    walk = np.full(sec.n, 2.5)
    m, = mw.from_osm(ways, sec, _Flat(95.0), overrides={3: {"clearance_m": 7.0, "width_m": 12.0}}, walks=(walk, walk))
    piers = next(x for x in m.meshes() if x[0].startswith("1WALL_mw_"))[1]
    assert np.abs(piers[:, 1]).min() >= sec.hw_left[100] + GUTTER_WIDTH + 2.5 + 1.0 - 1e-6
