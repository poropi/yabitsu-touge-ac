import numpy as np
import pytest

from yabitsu import geo, site, terayama_area as ta
from yabitsu.crosssection import GUTTER_WIDTH
from tests.test_meshgen import _straight


def _ll(x, y):
    lat, lon = geo.from_local(float(x), float(y))
    return [float(lat), float(lon)]


def _spec(sec):
    e = sec.hw_left[0] + GUTTER_WIDTH       # 道は x 方向（東）へまっすぐ、西＝左（L）は +y
    return {
        "plaza": {"points": [_ll(80, e - 0.3), _ll(95, e - 0.3), _ll(95, e + 8), _ll(80, e + 8)], "anchor_ll": _ll(88, e), "slope": 0.05},
        "lane": {"points": [_ll(100, e - 0.3), _ll(104, e - 0.3), _ll(104, e + 20), _ll(100, e + 20)], "anchor_ll": _ll(102, e), "slope": 0.10},
        "toilet": {"footprint": [_ll(84, e + 12), _ll(94, e + 12), _ll(94, e + 16), _ll(84, e + 16)], "axis_deg": 0.0, "size": [10.0, 4.0],
                   "wall_h": 2.5, "roof_rise": 1.3, "eaves": 0.9, "base_dz": 0.3},
        "no_edge_lines": [{"from_ll": _ll(70, 0), "to_ll": _ll(120, 0), "side": "L"}],
    }


def test_planes_meet_the_road_edge_and_rise_away_from_it():
    sec = _straight(201)
    sec.z = 100.0 + 0.02 * sec.s               # 道は 2 % で上る
    s = ta.build(_spec(sec), sec)
    e = sec.hw_left[0] + GUTTER_WIDTH
    for x in (82.0, 90.0):                     # 縁にそって段がない
        assert site.plane_z(s.plaza_plane, x, e) == pytest.approx(float(sec.edge_z(1)[int(x)]), abs=0.01)
    assert site.plane_z(s.plaza_plane, 88, e + 8) - site.plane_z(s.plaza_plane, 88, e) == pytest.approx(0.4, abs=0.01)
    assert s.mouths() == [(1, pytest.approx(100.0, abs=1.0), pytest.approx(104.0, abs=1.0))]
    assert s.no_lines() == [(1, pytest.approx(70.0), pytest.approx(120.0))]
    names = {m[0]: m for m in s.meshes()}
    assert {"1ROAD_terayama_lane", "1GRASS_terayama_plaza", "1WALL_terayama_toilet", "terayama_toilet_roof", "terayama_toilet_front"} <= set(names)
    roof = names["terayama_toilet_roof"][1]
    assert roof[:, 2].max() == pytest.approx(s.tz + 2.5 + 1.3)
    assert {p["name"] for p in s.pads()} == {"terayama_plaza", "terayama_lane", "terayama_toilet"}
    area = s.area()                             # OSM の小道を重ねない範囲：広場・わき道・トイレの台を含む
    assert area.contains(s.plaza.representative_point()) and area.contains(s.lane.representative_point())
    assert area.contains(__import__("shapely").geometry.Point(*s.tf.o))


def test_add_road_drops_edge_lines_only_where_asked():
    from yabitsu import assembly
    sec = _straight(201)

    class M:
        def __init__(self): self.meshes = {}
        def add_mesh(self, g, name, v, f, uv, mat, renderable=True): self.meshes[name] = (v, f)
    full, bare = M(), M()
    assembly.add_road(full, sec)
    assembly.add_road(bare, sec, no_lines=[(1, 50.0, 150.0)])
    def face_x(m, tag):   # 白線の面の真ん中の x（面だけを数える。頂点は残ることがある）
        return np.concatenate([v[np.asarray(f)].mean(1)[:, 0] for n, (v, f) in m.meshes.items() if n.startswith(f"LINES_s1_edge{tag}")])
    xs = face_x(bare, "L")
    assert not ((xs > 51) & (xs < 149)).any() and len(face_x(full, "L")) > len(xs)
    assert len(face_x(full, "R")) == len(face_x(bare, "R"))
