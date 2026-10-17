import json

import pytest

from yabitsu import osm


def test_falls_back_to_next_endpoint(tmp_path):
    calls = []
    def post(url, q):
        calls.append(url)
        return (502, "busy") if url == "A" else (200, '{"elements": []}')
    d = osm.fetch_overpass("q", tmp_path / "c.json", post=post, endpoints=["A", "B"], retries_per_endpoint=1, wait=0)
    assert d == {"elements": []} and calls == ["A", "B"]


def test_uses_cache(tmp_path):
    (tmp_path / "c.json").write_text('{"elements": [1]}', encoding="utf-8")
    d = osm.fetch_overpass("q", tmp_path / "c.json", post=lambda u, q: pytest.fail("should not fetch"))
    assert d == {"elements": [1]}


def test_all_endpoints_fail(tmp_path):
    with pytest.raises(RuntimeError):
        osm.fetch_overpass("q", tmp_path / "c.json", post=lambda u, q: (504, ""), endpoints=["A", "B"],
                           retries_per_endpoint=2, wait=0)
    assert not (tmp_path / "c.json").exists()


def _way(coords, **tags):
    return {"type": "way", "tags": {"ref": "70", "highway": "primary", **tags},
            "geometry": [{"lat": a, "lon": b} for a, b in coords]}


def test_shortest_route_and_segment_tags():
    long = _way([(35.0, 139.0), (35.0, 139.01), (35.01, 139.01)], lanes="2")
    short = _way([(35.0, 139.0), (35.01, 139.01)], lanes="1", bridge="yes")
    r = osm.shortest_route([long, short], (35.0, 139.0), (35.01, 139.01))
    assert len(r.lat) == 2 and r.seg_lanes == [1] and r.seg_bridge == [True]


def test_route_follows_chain_in_order():
    a = _way([(35.0, 139.0), (35.0, 139.001)], lanes="2")
    b = _way([(35.0, 139.001), (35.0, 139.002), (35.0, 139.003)], lanes="1")
    r = osm.shortest_route([b, a], (35.0, 139.0), (35.0, 139.003))
    assert r.lon == pytest.approx([139.0, 139.001, 139.002, 139.003]) and r.seg_lanes == [2, 1, 1]


def test_parse_lanes():
    assert osm.parse_lanes("2") == 2 and osm.parse_lanes("1;2") == 1 and osm.parse_lanes(None) == 0 and osm.parse_lanes("x") == 0


def test_route_json_round_trip(tmp_path):
    r = osm.Route([35.0, 35.1], [139.0, 139.1], [2], [False])
    r.to_json(tmp_path / "r.json")
    assert osm.Route.from_json(tmp_path / "r.json") == r


def test_parking_polygons():
    el = [{"type": "way", "tags": {"amenity": "parking", "name": "P"},
           "geometry": [{"lat": 1, "lon": 2}, {"lat": 1.1, "lon": 2}, {"lat": 1.1, "lon": 2.1}, {"lat": 1, "lon": 2}]}]
    p = osm.parking_polygons(el)
    assert p[0]["name"] == "P" and len(p[0]["lat"]) == 4


def test_route70_ways_keeps_named_route_and_drops_side_streets():
    a = _way([(35.0, 139.0), (35.0, 139.001)])
    b = {"type": "way", "tags": {"highway": "primary", "ref": "県道70号線", "name": "秦野清川線"},
         "geometry": [{"lat": 35.0, "lon": 139.001}, {"lat": 35.0, "lon": 139.002}]}
    c = {"type": "way", "tags": {"highway": "residential"}, "geometry": [{"lat": 35.0, "lon": 139.0}, {"lat": 35.001, "lon": 139.0}]}
    d = {"type": "way", "tags": {"highway": "trunk", "ref": "246", "name": "国道246号"}, "geometry": []}
    assert osm.route70_ways([a, b, c, d]) == [a, b]


def test_shortest_route_on_route70_ways_does_not_cut_through_side_streets():
    # 県道70号：(0,0)→(0,0.002)→(0.001,0.002)（遠回り）。住宅地の近道：(0,0)→(0.001,0.002)
    r70 = _way([(35.0, 139.0), (35.0, 139.002), (35.001, 139.002)])
    side = {"type": "way", "tags": {"highway": "residential"},
            "geometry": [{"lat": 35.0, "lon": 139.0}, {"lat": 35.001, "lon": 139.002}]}
    r = osm.shortest_route(osm.route70_ways([r70, side]), (35.0, 139.0), (35.001, 139.002))
    assert len(r.lat) == 3
