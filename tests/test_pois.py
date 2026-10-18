import numpy as np
import pytest

from yabitsu import pois
from tests.test_meshgen import _straight


def _node(x, y, nid=1, **tags):
    from yabitsu import geo
    lat, lon = geo.from_local(float(x), float(y))
    return {"type": "node", "id": nid, "lat": float(lat), "lon": float(lon), "tags": tags}


FLAT = lambda x, y: np.full(np.shape(x), 100.0)


def test_small_objects_are_placed_at_osm_points_facing_the_road():
    sec = _straight()
    el = [_node(50, 6, 1, amenity="vending_machine"), _node(80, -6, 2, amenity="post_box"),
          _node(120, 7, 3, historic="memorial", name="石碑"), _node(150, 30, 4, amenity="shelter")]
    items = pois.place(el, sec, skip=[])
    assert [i["kind"] for i in items] == ["vending_machine", "post_box", "stele", "shelter"]
    vm = items[0]
    assert abs(((vm["facing_deg"] - 180.0) + 180) % 360 - 180) < 10      # 道（南）の方を向く
    ms = pois.meshes(items, FLAT)
    mats = {m[4] for m in ms}
    assert {"vending", "post_red", "stone"} <= mats


def test_objects_near_existing_ones_are_skipped():
    sec = _straight()
    el = [_node(50, 6, 1, historic="memorial")]
    assert pois.place(el, sec, skip=[(50.5, 6.2)]) == []


def test_shop_sign_goes_on_nearest_building_edge_facing_road():
    from yabitsu import buildings as bd, geo
    sec = _straight()
    ll = lambda pts: [[float(a), float(b)] for a, b in zip(*geo.from_local(*np.asarray(pts, float).T))]
    b = bd.Building.from_spec({"id": "x", "footprint": ll([[40, 8], [60, 8], [60, 18], [40, 18]]), "storeys": 2, "height_m": 6,
                               "roof": {"type": "flat", "color": "#808080"}, "wall": "#FFFFFF"})
    n = pois.add_shop_signs([_node(50, 12, 7, shop="deli", name="丹沢大山五右衛門")], [b], sec)
    assert n == 1 and b.signs[0]["text"] == "丹沢大山五右衛門" and b.signs[0]["edge"] == bd.edge_facing(b, 180.0)


def test_point_on_the_road_is_moved_just_outside_it():
    """道のまん中にある OSM の点（S2 の案内板 2600687268）は、道の縁＋側溝の外へずらして置く（指摘 7）。"""
    from yabitsu import town_objects
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()
    hw = float(sec.hw_left[0])
    it, = pois.place([_node(50.0, 0.1, tourism="information", information="guidepost")], sec, [])
    x, y = it["xy"]
    assert x == pytest.approx(50.0, abs=0.05) and y == pytest.approx(hw + GUTTER_WIDTH + town_objects.OFF_ROAD_M, abs=0.05)
