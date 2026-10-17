import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from yabitsu import geo, side_roads as sr
from yabitsu.crosssection import GUTTER_WIDTH
from tests.test_meshgen import _straight


def _way(xy, **tags):
    lat, lon = geo.from_local(*np.asarray(xy, float).T)
    return {"type": "way", "id": tags.pop("id", 1), "tags": {"highway": "residential", **tags},
            "geometry": [{"lat": float(a), "lon": float(b)} for a, b in zip(lat, lon)]}


class _Flat:
    def __init__(self, z=100.0, gy=0.0):
        self.z, self.gy = z, gy

    def sample(self, lat, lon):
        x, y = geo.to_local(np.asarray(lat), np.asarray(lon))
        return self.z + self.gy * np.asarray(y)


def test_stub_starts_at_gutter_edge_and_is_blocked():
    sec = _straight()   # 道は x 軸にそって 0〜200、幅 6（hw 3）
    st, = sr.stubs([_way([[100, 0], [100, 80]])], sec, _Flat(), keep_out=[])
    edge = sec.hw_left[100] + GUTTER_WIDTH
    assert st.surface.bounds[1] == pytest.approx(edge, abs=0.05)               # 路面の始まり
    assert st.surface.bounds[3] == pytest.approx(edge + sr.STUB_LENGTH_M, abs=0.5)
    (x0, y0), (x1, y1) = st.mouth.coords
    assert y0 == pytest.approx(edge + sr.MOUTH_SETBACK_M) and y1 == pytest.approx(y0)
    assert abs(x1 - x0) == pytest.approx(sr.DEFAULT_WIDTH["residential"] + 2 * sr.MOUTH_OVERHANG_M)


def test_stub_height_goes_from_road_edge_to_ground():
    sec = _straight()   # 道の高さ 100、地面は y 方向に 0.1 の上り
    st, = sr.stubs([_way([[100, 0], [100, 80]])], sec, _Flat(100.0, 0.1), keep_out=[])
    v, f, uv = st.mesh()
    near = v[v[:, 1] < sec.hw_left[100] + GUTTER_WIDTH + 0.1]
    far = v[v[:, 1] > v[:, 1].max() - 0.1]
    assert np.allclose(near[:, 2], sec.edge_z(1)[100], atol=0.02)
    assert np.allclose(far[:, 2], 100.0 + 0.1 * far[:, 1], atol=0.3)


def test_stub_skips_keep_out_and_route70_and_footways():
    sec = _straight()
    lot = Polygon([(40, 4), (60, 4), (60, 30), (40, 30)])
    ways = [_way([[50, 0], [50, 60]], id=2), _way([[150, 0], [150, 60]], id=3, ref="70"),
            _way([[120, 0], [120, 60]], id=4, highway="footway")]
    assert sr.stubs(ways, sec, _Flat(), keep_out=[lot]) == []


def test_stub_before_s_min_is_left_to_town():
    sec = _straight()
    assert sr.stubs([_way([[30, 0], [30, 60]])], sec, _Flat(), keep_out=[], s_min=50.0) == []


def test_stub_pads_are_tilted_planes_matching_the_mesh():
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 80]])], sec, _Flat(100.0, 0.1), keep_out=[])
    pads = st.pads()
    assert all(p["kind"] == "stub" and len(p["plane"]) == 5 for p in pads)
    v, f, uv = st.mesh()
    from yabitsu.site import plane_z
    for p in pads:
        inside = [q for q in v if Polygon(p["polygon"]).buffer(0.01).contains(Point(q[0], q[1]))]
        assert all(abs(plane_z(p["plane"], q[0], q[1]) - q[2]) < 0.02 for q in inside)


def test_town_exit_s():
    sec = _straight()
    assert sr.town_exit_s(sec, (0.0, 0.0), 50.0) == pytest.approx(51.0, abs=1.0)
