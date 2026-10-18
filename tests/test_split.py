import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from yabitsu import geo, split
from tests.test_meshgen import _sec_from_xy
from tests.test_side_roads import _Flat, _way


def _scene():
    """南（y<0）から来る2車線の道が、y=0 で西（x=-3）と東（x=+3）の1車線に分かれ、y=60 でまた合わさる。中に島。"""
    y = np.arange(-100.0, 161.0)
    x = np.where((y > 0) & (y < 60), -3.0 * np.sin(np.pi * np.clip(y, 0, 60) / 60), 0.0)   # 中心線＝西の車線
    sec = _sec_from_xy(x, y)
    ll = lambda pts: pts
    ways = [_way([[0, -100], [0, 0]], id=1, ref="70", lanes="2"),
            _way([[0, 0], [-3, 15], [-3, 45], [0, 60]], id=2, ref="70", lanes="1", oneway="yes"),
            _way([[0, 60], [3, 45], [3, 15], [0, 0]], id=3, ref="70", oneway="yes"),
            _way([[0, 60], [0, 160]], id=4, ref="70", lanes="2")]
    island = [list(geo.from_local(x, y)) for x, y in [(-0.8, 14), (0.8, 14), (0.8, 46), (-0.8, 46)]]
    spec = {"widths": {"1": 7.0, "2": 3.6, "3": 3.6, "4": 7.0}, "island": island, "red": ["3"], "objects": []}
    return sec, ways, spec


def test_paved_area_follows_every_map_line_and_has_no_gap():
    sec, ways, spec = _scene()
    z = split.Split(spec, ways, sec, _Flat(100.0))
    for p in [(0, -10), (2.5, -10), (-2.5, -10), (-3, 30), (3, 30), (0, 2), (2.0, 5)]:
        assert z.carriageway.contains(Point(*p)), p
    assert not z.carriageway.contains(Point(0, 30))          # 島の所は路面でない
    assert not z.carriageway.contains(Point(6.5, 30))         # 東の車線の外は路面でない


def test_main_ribbon_skipped_inside_the_split_only():
    sec, ways, spec = _scene()
    z = split.Split(spec, ways, sec, _Flat(100.0))
    s0, s1 = z.main_skip
    assert s0 < 100.0 - 0 and s1 > 160.0 and s0 > 40 and s1 < 230     # 分かれる所の手前から、合わさる所の先まで


def test_surface_meets_the_ribbon_without_a_step():
    sec, ways, spec = _scene()
    sec.z[:] = 100.0 + 0.05 * sec.s
    z = split.Split(spec, ways, sec, _Flat(95.0))
    s0 = z.main_skip[0]
    i = int(np.argmin(np.abs(sec.s - s0)))
    from yabitsu.crosssection import surface_height
    for lat in (-2.0, 0.0, 2.0):
        want = surface_height(lat, sec.z[i], sec.slope[i], sec.crown[i])
        assert float(z.ground.z(np.array([sec.x[i] - lat]), np.array([sec.y[i]]))[0]) == pytest.approx(want, abs=0.03)


def test_red_lane_and_lines():
    sec, ways, spec = _scene()
    z = split.Split(spec, ways, sec, _Flat(100.0))
    ms = {m[0]: m for m in z.meshes()}
    red = [m for m in ms.values() if m[4] == "paint_red"]
    assert red and all(np.all(m[1][:, 0] > -0.01) and m[1][:, 0].mean() > 1.5 for m in red)   # 赤は東の車線だけ
    orange = [m for m in ms.values() if m[4] == "line_orange"]
    assert orange and all(np.all((m[1][:, 1] < 15) | (m[1][:, 1] > 45)) for m in orange)   # 中央線は島の所にない
    assert any(m[4] == "line_white" for m in ms.values())


def test_split_surface_reaches_the_rows_where_the_main_ribbon_starts_again():
    """窓の端と本線の帯の間に草のすき間を作らない（神社の島の北の端 3484m）。"""
    sec, ways, spec = _scene()
    z = split.Split(spec, ways, sec, _Flat(100.0))
    s0, s1 = z.main_skip
    i0 = int(np.searchsorted(sec.s, s0, side="right")) - 1     # 帯が終わる行
    i1 = int(np.searchsorted(sec.s, s1, side="left"))          # 帯がまた始まる行
    for s in (sec.s[i0] + 0.2, sec.s[i1] - 0.2):
        i = int(np.argmin(np.abs(sec.s - s)))
        assert z.carriageway.contains(Point(sec.x[i], sec.y[i] + (s - sec.s[i]))), s


def test_road_keeps_its_full_width_where_the_two_lanes_come_together():
    """1車線の2本が合わさる所の手前でも、路面は2車線の幅いっぱい（3455〜3463m で細くなっていた）。"""
    sec, ways, spec = _scene()
    z = split.Split(spec, ways, sec, _Flat(100.0))
    for x in (-3.0, 3.0):
        assert z.carriageway.contains(Point(x, 58.0)) and z.carriageway.contains(Point(x, 2.0)), x
    assert not z.carriageway.contains(Point(4.6, 58.0))


def test_ground_override_keeps_under_the_main_road_beyond_the_window_ends():
    """島のまわりで地形を路面の高さに合わせるとき、窓の先の本線の帯の下も「舗装」に入れて、地形を路面の下へ下げる
    （3484m の先で、帯の上に草の帯が出ていた）。"""
    sec, ways, spec = _scene()
    z = split.Split(spec, ways, sec, _Flat(100.0))
    _, region, paved = z.town_override()
    for s in (z.main_skip[0] - 2.0, z.main_skip[1] + 2.0):
        i = int(np.argmin(np.abs(sec.s - s)))
        for lat in (-3.0, 0.0, 3.0):
            p = Point(sec.x[i] - lat * np.sin(sec.heading[i]), sec.y[i] + lat * np.cos(sec.heading[i]))
            if region.contains(p):
                assert paved.contains(p), (s, lat)
