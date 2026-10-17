import numpy as np
import pytest

from yabitsu import crosssection as cs
from yabitsu import terrain as tr
from tests.test_meshgen import _straight


def _pts(sec, lats, s=100.0):
    i = int(s)
    X = np.array([[sec.x[i] + 0 * l for l in lats]]); Y = np.array([[l for l in lats]])
    return X, Y


def test_fill_slope_on_embankment():
    sec = _straight()  # 路面の高さ 100m、幅 6m、側溝 0.6m
    X, Y = _pts(sec, [3.6 + 2.0, 120.0])  # 120m 先では盛り土の斜面（1:1）が地面に届いている
    Z, E = tr.carve(X, Y, np.zeros_like(X), sec)
    z_edge = cs.surface_height(3.0, 100.0, sec.slope[100], sec.crown[100])
    assert Z[0, 0] == pytest.approx(z_edge - 2.0 * tr.FILL_SLOPE) and Z[0, 1] == pytest.approx(0.0)
    assert E[0, 0] == pytest.approx(2.0)


def test_cut_slope_and_wall_slope():
    sec = _straight()
    X, Y = _pts(sec, [3.6 + 2.0])
    Z, _ = tr.carve(X, Y, np.full_like(X, 200.0), sec)
    z_edge = cs.surface_height(3.0, 100.0, sec.slope[100], sec.crown[100])
    assert Z[0, 0] == pytest.approx(z_edge + 2.0 * tr.CUT_SLOPE)
    Zw, _ = tr.carve(X, Y, np.full_like(X, 200.0), sec, cut_left=np.full(sec.n, 3.0), cut_right=np.full(sec.n, 2.0))
    assert Zw[0, 0] == pytest.approx(z_edge + 2.0 * 3.0)


def test_under_road_is_below_surface():
    sec = _straight()
    X, Y = _pts(sec, [0.0, 2.9, -3.3])
    Z, E = tr.carve(X, Y, np.full_like(X, 300.0), sec)
    surf = cs.surface_height(np.array([0.0, 2.9, -3.3]).clip(-3, 3), 100.0, sec.slope[100], sec.crown[100])
    assert (E == 0).all() and (Z[0] <= surf - tr.UNDER_DEPTH + 1e-9).all()


def test_beyond_section_end_is_natural():
    sec = _straight()
    X = np.array([[sec.x[-1] + 80.0]]); Y = np.array([[0.0]])
    Z, _ = tr.carve(X, Y, np.full_like(X, 120.0), sec)
    assert Z[0, 0] == pytest.approx(120.0)


def test_pad_is_flattened():
    sec = _straight()
    pad = {"polygon": [[90, 20], [110, 20], [110, 40], [90, 40], [90, 20]], "z": 105.0}
    X = np.array([[100.0]]); Y = np.array([[30.0]])
    Z, E = tr.carve(X, Y, np.full_like(X, 50.0), sec, pads=[pad])
    assert Z[0, 0] == pytest.approx(105.0 - tr.UNDER_DEPTH) and E[0, 0] == 0


def test_grid_mesh_chunks_respect_keep_and_vertex_limit():
    xs = np.arange(0.0, 301.0, 1.0); ys = np.arange(0.0, 201.0, 1.0)
    Z = np.zeros((len(ys), len(xs)))
    keep = np.ones((len(ys) - 1, len(xs) - 1), bool); keep[:, :100] = False
    chunks = tr.grid_mesh_chunks(xs, ys, Z, keep, block=100)
    assert sum(len(f) for _, f, _ in chunks) == keep.sum()
    assert all(len(v) <= 65535 for v, _, _ in chunks)
    v, f, _ = chunks[0]
    from yabitsu.meshgen import face_normals
    assert (face_normals(v, f)[:, 2] > 0).all()


def test_no_teeth_across_formation_edge_on_cut_side():
    # 道の端の内側と外側の格子点を結ぶ三角形が、側溝の外側の縁より上に出ないこと
    sec = _straight()
    X, Y = _pts(sec, [3.0, 4.6])
    Z, _ = tr.carve(X, Y, np.full_like(X, 200.0), sec)
    z_edge = cs.surface_height(3.0, 100.0, sec.slope[100], sec.crown[100])
    z_at_form = Z[0, 0] + (Z[0, 1] - Z[0, 0]) * (3.6 - 3.0) / (4.6 - 3.0)
    assert z_at_form <= z_edge + 0.02


def test_nan_filled_with_nearest_value():
    Z = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
    out = tr.fill_nan_nearest(Z)
    assert not np.isnan(out).any() and out[0, 2] in (2.0, 6.0) and out[1, 1] in (2.0, 4.0, 6.0)


class _FlatSampler:
    def sample(self, lat, lon):
        return np.full(np.shape(lat), 100.0)


def test_near_grid_reaches_extra_areas_far_from_the_road():
    from tests.test_meshgen import _straight
    sec = _straight(201)
    far_spot = [[100, 140], [110, 140], [110, 150], [100, 150], [100, 140]]  # 道から140m以上
    plain = tr.build_near(sec, _FlatSampler(), [])
    near = tr.build_near(sec, _FlatSampler(), [], extra=[far_spot])
    far = tr.build_far(sec, _FlatSampler(), extra=[far_spot])
    assert plain["ys"].max() < 140
    i = np.searchsorted(near["xs"], 105); j = np.searchsorted(near["ys"], 145)
    assert near["D"][j, i] <= tr.NEAR_BAND  # 細かい地形が張られる
    ci = np.searchsorted(far["xs"], 105) - 1; cj = np.searchsorted(far["ys"], 145) - 1
    assert far["Dc"][cj, ci] < tr.FAR_HOLE  # 粗い地形には穴があく（重ならない）


def test_sidewalk_widens_the_formation():
    sec = _straight()
    X, Y = _pts(sec, [3.6 + 1.0, 3.6 + 2.0 + 2.0])   # 歩道（幅 2）の上と、歩道の外側から 2m
    walk = np.full(sec.n, 2.0)
    Z, E = tr.carve(X, Y, np.full_like(X, 200.0), sec, side_extra_left=walk)
    z_edge = cs.surface_height(3.0, 100.0, sec.slope[100], sec.crown[100])
    assert Z[0, 0] < z_edge + 0.15                          # 歩道の下に地形がかぶらない
    assert Z[0, 1] == pytest.approx(z_edge + 2.0 * tr.CUT_SLOPE)   # 切り土の斜面は歩道の外側から始まる


def _town_case():
    from shapely.geometry import Point
    from yabitsu.town_ground import TownGround
    from tests.test_meshgen import _straight
    sec = _straight(201)
    xs = np.arange(-50.0, 251.0); ys = np.arange(-150.0, 151.0)
    X, Y = np.meshgrid(xs, ys)
    g = TownGround(xs, ys, 97.0 + 0.01 * X)            # 町の地面（道とは別の高さ）
    road = Point(100, 0).buffer(3.6)                    # 走れる範囲（の代わり）
    return sec, g, Point(100, 0).buffer(40.0), road


def test_terrain_inside_town_follows_town_ground():
    sec, g, circle, road = _town_case()
    near = tr.build_near(sec, _FlatSampler(), [], town=(g, circle, road))
    i = np.searchsorted(near["xs"], 100); j = np.searchsorted(near["ys"], 25)
    assert near["Z"][j, i] == pytest.approx(97.0 + 0.01 * near["xs"][i], abs=0.01)     # 道の外：町の地面
    j0 = np.searchsorted(near["ys"], 0)
    assert near["Z"][j0, i] == pytest.approx(97.0 + 0.01 * near["xs"][i] - tr.UNDER_DEPTH, abs=0.01)   # 道の下


def test_terrain_blends_outside_town_circle():
    sec, g, circle, road = _town_case()
    plain = tr.build_near(sec, _FlatSampler(), [])
    near = tr.build_near(sec, _FlatSampler(), [], town=(g, circle, road))
    i = np.searchsorted(near["xs"], 100); j = np.searchsorted(near["ys"], 40 + tr.TOWN_BLEND_M + 2)
    assert near["Z"][j, i] == pytest.approx(plain["Z"][j, i], abs=1e-6)


def test_pad_reach_is_limited():
    sec = _straight()
    pad = {"polygon": [[100, 40], [110, 40], [110, 50], [100, 50], [100, 40]], "z": 300.0}   # 地面より 200m 高い
    X, Y = _pts(sec, [40.0 + tr.PAD_REACH_M + 15.0])     # 土台から PAD_REACH_M より遠い点
    Z0, _ = tr.carve(X, Y, np.full_like(X, 150.0), sec)
    Z1, _ = tr.carve(X, Y, np.full_like(X, 150.0), sec, pads=[pad])
    assert np.allclose(Z0, Z1)


def test_mid_band_overlaps_near_and_reaches_200m():
    from tests.test_meshgen import _straight
    sec = _straight(201)
    mid = tr.build_mid(sec, _FlatSampler(), [])
    D = mid["D"]
    keep = tr.mid_keep(D)
    i = np.searchsorted(mid["xs"], 100)
    ys = mid["ys"][:-1][keep[:, i]]
    assert ys.max() >= 195.0 and np.abs(ys).min() <= tr.NEAR_BAND - 3.0      # 200m まで、細かい地形の下にもぐり込む
    assert np.allclose(mid["Z"][np.abs(mid["ys"] - 150).argmin(), i], 100.0 - tr.MID_DROP)
    assert tr.FAR_HOLE >= tr.MID_BAND - 10.0
