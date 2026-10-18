import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box

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


def test_crosswalk_on_stub_lies_on_its_surface_and_bars_follow_the_road():
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 80]])], sec, _Flat(100.0, 0.1), keep_out=[])
    v, f, uv = st.crosswalk_mesh(np.array([100.0, 9.0]))
    from yabitsu.site import plane_z
    assert v[:, 1].min() == pytest.approx(9.0 - sr.CW_LEN / 2, abs=0.05) and v[:, 1].max() == pytest.approx(9.0 + sr.CW_LEN / 2, abs=0.05)
    assert v[:, 0].min() >= 100.0 - st.width / 2 and v[:, 0].max() <= 100.0 + st.width / 2
    for q in f:
        p = v[q]
        assert np.ptp(p[:, 1]) > np.ptp(p[:, 0])                  # 帯は道の向き（y）に長い
    ex = st.mesh()[0]
    assert np.all(v[:, 2] > np.interp(v[:, 1], ex[:, 1], ex[:, 2]) - 0.05)


def test_mouth_moves_out_of_excluded_area():
    sec = _straight()
    lane = box(90, 3.6, 130, 9.0)          # 下りの車線（本線の右どなりの路面）
    st, = sr.stubs([_way([[100, 0], [100, 60]])], sec, _Flat(), keep_out=[])
    sr.move_mouth_out(st, lane)
    (x0, y0), (x1, y1) = st.mouth.coords
    assert min(y0, y1) >= 9.0 - 1e-6                     # 入口の壁と柵は、下りの車線の外へ
    assert st.surface.intersection(lane).area < 0.01


def test_barricade_only_on_real_branch_roads():
    sec = _straight()
    road = _way([[100, 0], [100, 80]], id=11)                              # 住宅地の道（80m）
    alley = _way([[60, 0], [60, 40]], id=12, highway="service", service="alley")
    drive = _way([[140, 0], [140, 20]], id=13)                            # 20m で行き止まりの道（家の前）
    st = {s.osm_id: s for s in sr.stubs([road, alley, drive], sec, _Flat(), keep_out=[])}
    ways = {w["id"]: w for w in (road, alley, drive)}
    assert sr.needs_barricade(st[11], ways) and not sr.needs_barricade(st[12], ways) and not sr.needs_barricade(st[13], ways)


def test_mouth_is_set_back_from_road_and_sidewalk():
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 60]])], sec, _Flat(), keep_out=[])
    walk = box(0, 3.6, 200, 3.6 + 2.0)                       # 本線の左の歩道（幅 2m）
    sr.set_back_mouth(st, sec, [walk])
    (x0, y0), (x1, y1) = st.mouth.coords
    assert min(y0, y1) >= 3.6 + 2.0 + sr.CLEAR_M - 1e-6
    assert st.surface.contains(Point(100, 4.5))              # 路面は本線から切れずにつながる


def test_branch_gaps_on_the_main_road():
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 60]])], sec, _Flat(), keep_out=[])
    gaps = sr.branch_gaps([st], sec)
    (s0, s1), = gaps["L"]
    assert s0 <= 100 - st.width / 2 - 1.0 + 0.3 and s1 >= 100 + st.width / 2 + 1.0 - 0.3 and gaps["R"] == []


def test_branch_gap_is_only_the_opening_even_when_the_branch_turns_back_along_the_road():
    """入口から少し入って本線ぞいに戻っていく分かれ道でも、壁を切るのは入口の開いている所だけ（b12：798m 左）。"""
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [98, 7], [60, 21]])], sec, _Flat(), keep_out=[])
    sr.set_back_mouth(st, sec, [box(0, 3.6, 200, 3.6 + 1.5)])
    (s0, s1), = sr.branch_gaps([st], sec)["L"]
    assert 93.0 <= s0 and s1 <= 104.0 and s1 - s0 >= st.width


def test_barricade_stands_behind_the_crosswalk_on_the_branch():
    """分かれ道の入口に横断歩道があるときは、柵は横断歩道の後ろ（b16：1541m 右）。"""
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 60]])], sec, _Flat(), keep_out=[])
    cw = st.crosswalk_area((100, 7.0))
    assert cw.contains(Point(100, 7.0 + sr.CW_LEN / 2 - 0.1))
    sr.set_back_mouth(st, sec, [cw])
    (x0, y0), (x1, y1) = st.mouth.coords
    assert min(y0, y1) >= 7.0 + sr.CW_LEN / 2 + sr.CLEAR_M - 1e-6


def test_barricade_goes_behind_a_crosswalk_set_further_into_the_branch():
    """横断歩道が入口より奥（本線から離れた所）にあっても、柵はその後ろ（b17：1583m 左）。"""
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 60]])], sec, _Flat(), keep_out=[])
    sr.set_back_mouth(st, sec, [st.crosswalk_area((100, 12.0))])
    (x0, y0), (x1, y1) = st.mouth.coords
    assert min(y0, y1) >= 12.0 + sr.CW_LEN / 2 + sr.CLEAR_M - 1e-6


def test_branch_gap_uses_the_course_distance_not_the_map_length():
    """本線の s は地図上の長さとずれる（S1 の 9900m で 38m）。壁を切る範囲は、入口に一番近い本線の行の s で測る。"""
    sec = _straight()
    sec.s = sec.s * 1.05                     # 坂のぶん s が地図上の長さより長い
    st, = sr.stubs([_way([[100, 0], [100, 60]])], sec, _Flat(), keep_out=[])
    (s0, s1), = sr.branch_gaps([st], sec)["L"]
    assert (s0 + s1) / 2 == pytest.approx(105.0, abs=0.6)


def test_hidden_wall_follows_the_branch_surface_where_the_mouth_was_set_back():
    """入口を奥へ下げた所が下り坂でも、見えない壁の下に すき間をあけない（壁の高さは入口の所の路面から。指摘 6）。"""
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 60]])], sec, _Flat(100.0, gy=-0.1), keep_out=[])
    sr.set_back_mouth(st, sec, [st.crosswalk_area((100, 12.0))])
    v, f, _ = st.wall()
    for x, y in st.mouth.coords:
        z = st.z_at(x, y)
        near = v[np.hypot(v[:, 0] - x, v[:, 1] - y) < 1e-6]
        assert near[:, 2].min() <= z - 0.4 and near[:, 2].max() >= z + 2.5
    zs = st.ground_z(np.array([100.0, 100.0]), np.array([20.0, 30.0]))
    assert zs[0] == pytest.approx(st.z_at(100.0, 20.0)) and zs[1] == pytest.approx(st.z_at(100.0, 30.0))


def test_long_branch_is_built_to_its_length_with_a_wall_at_the_far_end():
    """S4 の分かれ道は、入口から 200m まで道として作り、奥の終わりに見えない壁（柵を外せば奥まで走れる形）。"""
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 300]])], sec, _Flat(), keep_out=[], length_m=200.0)
    assert st.axis.length == pytest.approx(200.0, abs=1.0)
    v, f, _ = st.end_wall()
    end = np.array(st.axis.coords[-1])
    assert np.abs(v[:, 1] - end[1]).max() < 0.5 and v[:, 0].min() < 100 - st.width / 2 and v[:, 0].max() > 100 + st.width / 2
    assert v[:, 2].max() - v[:, 2].min() > 3.0


def test_long_branch_follows_the_next_way_to_its_length():
    """OSM の道は途中で別の way に分かれていることが多い。つながる車道をたどって 200m まで作る。"""
    sec = _straight()
    ways = [_way([[100, 0], [100, 80]], id=1), _way([[100, 80], [100, 300]], id=2)]
    st, = sr.stubs(ways, sec, _Flat(), keep_out=[], length_m=200.0)
    assert st.axis.length == pytest.approx(200.0, abs=1.0)


def test_long_branch_stops_before_touching_the_main_road_again():
    """入口から出て本線へまた戻ってくる道は、本線の路面（＋側溝＋CLEAR_M）に入る手前で終える。"""
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 40], [160, 40], [160, 0]])], sec, _Flat(), keep_out=[], length_m=200.0)
    from yabitsu.start_lot import road_formation
    form = road_formation(sec, 0.0, float(sec.s[-1]))
    assert form.distance(Point(st.axis.coords[-1])) >= st.width / 2 + sr.CLEAR_M
    assert st.surface.distance(form.difference(form.buffer(-0.01)).buffer(0)) < 1.0   # 入口は本線とつながる
    assert st.axis.length > 80


def test_only_long_branches_get_a_far_end_wall_mesh():
    sec = _straight()
    long_, = sr.stubs([_way([[100, 0], [100, 300]])], sec, _Flat(), keep_out=[], length_m=200.0)
    short, = sr.stubs([_way([[50, 0], [50, 300]], id=2)], sec, _Flat(), keep_out=[])
    names = [m[0] for m in sr.stub_meshes([long_, short])]
    assert names.count("1WALL_side_end_000") == 1 and not any(n.endswith("end_001") for n in names)
    assert next(m for m in sr.stub_meshes([long_]) if m[0] == "1WALL_side_end_000")[5] is False   # 見えない壁


def test_barricade_decision_uses_the_built_branch_length():
    """OSM の1本目の way が短くても、つながる道をたどって長く作った分かれ道には柵を置く（S4 1630m）。"""
    sec = _straight()
    ways = [_way([[100, 0], [100, 20]], id=1, highway="unclassified"), _way([[100, 20], [100, 300]], id=2, highway="unclassified")]
    st, = sr.stubs(ways, sec, _Flat(), keep_out=[], length_m=200.0)
    assert sr.needs_barricade(st, {w["id"]: w for w in ways})


def test_two_ends_of_a_loop_road_do_not_overlap():
    """本線から出てまた本線へ戻る1本の旧道は、両方の入口から分かれ道として作られる。2本目は1本目の路面の手前で終える（S4 1630m・1672m）。"""
    sec = _straight()
    ways = [_way([[100, 0], [100, 40], [130, 40]], id=1, highway="unclassified"),     # OSM では旧道が2本の way に分かれている
            _way([[160, 0], [160, 40], [130, 40]], id=2, highway="unclassified")]
    sts = sr.stubs(ways, sec, _Flat(), keep_out=[], length_m=200.0)
    assert sts and all(p.surface.intersection(q.surface).area < 1.0 for i, p in enumerate(sts) for q in sts[i + 1:])


class _Pit(_Flat):
    def sample(self, lat, lon):
        x, y = geo.to_local(np.asarray(lat), np.asarray(lon))
        z = np.full(np.shape(x), self.z, float)
        return np.where(np.hypot(np.asarray(x) - 100, np.asarray(y) - 64.2) < 3, self.z - 4.0, z)   # 標高データの1点の穴（t=60 の点）


def test_long_branch_has_no_pit_from_one_bad_elevation_sample():
    """長い分かれ道（S4）の高さは、標高データの1点だけの穴や出っぱりに引っぱられない（S4 1672m の 40m に 4m の穴があった）。"""
    sec = _straight()
    st, = sr.stubs([_way([[100, 0], [100, 300]])], sec, _Pit(), keep_out=[], length_m=200.0)
    i = int(np.argmin(np.abs(st.t_knots - 60)))
    assert abs(st.z_knots[i] - 100.0) < 1.0


def test_branch_running_alongside_the_road_before_leaving_is_built_in_full():
    """入口からしばらく本線ぞいを走ってから離れていく林道（S5 1012m の林道唐沢線）は、途中で切らずに作る。"""
    sec = _straight()
    st, = sr.stubs([_way([[60, 0], [63, 7.0], [100, 7.0], [130, 60], [150, 200]], id=3, highway="unclassified")],
                   sec, _Flat(), keep_out=[], length_m=200.0)
    assert st.axis.length > 150
