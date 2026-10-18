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


def test_house_plot_next_to_the_road_does_not_lift_the_ground_over_the_road():
    """道のすぐ横の高い家の土台（S3 4297m、蓑毛の家 3.3m 上）が、道の路面の上に地形を持ち上げない。"""
    sec = _straight()
    road = float(sec.z[0]); hw = float(sec.hw_left[0])
    pad = {"polygon": [[90, -hw - 0.3], [110, -hw - 0.3], [110, -20], [90, -20], [90, -hw - 0.3]], "z": road + 3.3}
    X = np.array([[100.0, 100.0, 100.0]]); Y = np.array([[-(hw - 0.3), -(hw + cs.GUTTER_WIDTH + 0.5), -15.0]])
    Z, _ = tr.carve(X, Y, np.full_like(X, road), sec, pads=[pad])
    assert Z[0, 0] < road                                               # 路面の下
    assert Z[0, 1] <= road + tr.CUT_SLOPE * 0.5 + 1e-6                   # 道の縁からの切り土の斜面より上に出ない
    assert Z[0, 2] == pytest.approx(road + 3.3 - tr.UNDER_DEPTH)          # 道から離れた所は土台の高さのまま


def test_ground_stays_under_the_main_road_where_the_town_ground_blends_out():
    """町（や島）の地面へなめらかに移す帯の中でも、本線の帯の下の地形は路面より UNDER_DEPTH 下（202m・3330m・3487m）。"""
    from yabitsu.town_ground import TownGround
    sec, _, circle, road = _town_case()
    xs = np.arange(-50.0, 251.0); ys = np.arange(-150.0, 151.0)
    g = TownGround(xs, ys, np.full((len(ys), len(xs)), float(sec.z[0])))   # 町の地面＝路面の高さ
    near = tr.build_near(sec, _FlatSampler(), [], town=(g, circle, road))
    plain = tr.build_near(sec, _FlatSampler(), [])
    i = np.searchsorted(near["xs"], 142.0)
    for y in (0.0, 2.0, -2.0):                    # 格子の点で、路面（片側 3.0m）の中、円のすぐ外
        j = np.searchsorted(near["ys"], y); assert near["ys"][j] == y
        assert near["Z"][j, i] == pytest.approx(plain["Z"][j, i]), y   # 本線の下は削ったままの高さ（町の地面へ持ち上げない）


def test_ground_stays_under_the_main_road_inside_the_town_where_the_town_paving_stops_short():
    """町の円の中でも、町の舗装の多角形が届いていない本線の帯の下は、削ったままの高さ（S5 29700m、熊野橋の先の北原の入口）。"""
    from yabitsu.town_ground import TownGround
    sec, _, circle, road = _town_case()
    xs = np.arange(-50.0, 251.0); ys = np.arange(-150.0, 151.0)
    g = TownGround(xs, ys, np.full((len(ys), len(xs)), float(sec.z[0])))   # 町の地面＝路面の高さ
    near = tr.build_near(sec, _FlatSampler(), [], town=(g, circle, road))   # 舗装は (100, 0) の半径 3.6m だけ
    plain = tr.build_near(sec, _FlatSampler(), [])
    i = np.searchsorted(near["xs"], 130.0)
    for y in (0.0, 2.0, -2.0):                    # 円の中、舗装の外、本線の路面の中
        j = np.searchsorted(near["ys"], y)
        assert near["Z"][j, i] == pytest.approx(plain["Z"][j, i]), y


def test_road_clear_counts_places_where_ground_comes_up_to_the_road():
    """track_check の road_clear：本線（と分かれ道）の路面から 10cm 以内に地形が来る所の数。0 で ok。"""
    sec = _straight()
    xs = np.arange(-10.0, 211.0); ys = np.arange(-30.0, 31.0); X, Y = np.meshgrid(xs, ys)
    Z = np.full(X.shape, float(sec.z[0]) - 1.0)
    near = {"xs": xs, "ys": ys, "Z": Z}
    assert tr.road_clear_count(sec, near) == 0
    Z[(X > 50) & (X < 60) & (np.abs(Y) < 2)] = float(sec.z[0]) + 0.3          # 道の上に土
    assert tr.road_clear_count(sec, near) > 0


def test_road_clear_skips_tunnel_rows():
    sec = _straight()
    xs = np.arange(-10.0, 211.0); ys = np.arange(-30.0, 31.0); X, Y = np.meshgrid(xs, ys)
    Z = np.where((X > 50) & (X < 60), float(sec.z[0]) + 30.0, float(sec.z[0]) - 1.0)    # 山（トンネル）
    near = {"xs": xs, "ys": ys, "Z": Z}
    skip = (sec.s > 49) & (sec.s < 61)
    assert tr.road_clear_count(sec, near) > 0 and tr.road_clear_count(sec, near, skip_rows=skip) == 0


def test_road_clear_edge_points_count_only_ground_above_the_surface():
    """路面の端（側溝のきわ）は、2m の地形の格子が切り土の斜面とつながる所。路面より上に出たときだけ数える。"""
    sec = _straight()
    xs = np.arange(-10.0, 211.0, 0.5); ys = np.arange(-30.0, 31.0, 0.5); X, Y = np.meshgrid(xs, ys)
    Z = np.full(X.shape, float(sec.z[0]) - 1.0)
    edge = float(sec.hw_left[0]) - 0.2
    zs = float(cs.surface_height(edge, float(sec.z[55]), sec.slope[55], sec.crown[55]))
    band = (np.abs(Y - edge) < 0.3) & (X > 50) & (X < 60)
    Z[band] = zs - 0.05                                                               # 端で 5cm 下
    assert tr.road_clear_count(sec, {"xs": xs, "ys": ys, "Z": Z}) == 0
    Z[band] = zs + 0.05                                                               # 端で 5cm 上
    assert tr.road_clear_count(sec, {"xs": xs, "ys": ys, "Z": Z}) > 0


def test_far_terrain_inside_the_lake_is_below_the_water():
    from shapely.geometry import Point
    sec = _straight()
    lake = Point(100, 600).buffer(300)
    far = tr.build_far(sec, _FlatSampler(), lake=(lake, 100.0))
    X, Y = np.meshgrid(far["xs"], far["ys"])
    import shapely
    inside = shapely.contains_xy(lake, X, Y)
    assert inside.any() and far["Z"][inside].max() <= 100.0 - tr.LAKE_DEPTH + 1e-6


def test_terrain_faces_never_cross_the_tunnel_bore():
    """トンネルの出入口で、残した山（天井＋土かぶり）と削った地面をつなぐ斜めの面が、筒の中（車の高さ）を横切らない（S5 2454m・10686m でクラッシュ）"""
    from tests.test_meshgen import _straight
    from yabitsu import tunnel as tn
    sec = _straight(201)
    tun = [tn.Tunnel(1, "", 80.0, 130.0)]
    near = tr.build_near(sec, _FlatSampler(), [], keep_rows=tn.carve_mask(sec.s, tun),
                         bore_rows=tn.row_mask(sec.s, tun, margin_m=tr.BORE_MARGIN_M))
    xs, ys, Z, H = near["xs"], near["ys"], near["Z"], near["H"]
    road, roof = 100.0, 100.0 + tn.bore_height(sec, 100)
    c = np.stack([Z[:-1, :-1], Z[1:, :-1], Z[:-1, 1:], Z[1:, 1:]])
    cx = (xs[:-1] + xs[1:]) / 2; cy = (ys[:-1] + ys[1:]) / 2
    box = (np.abs(cy)[:, None] <= 3.0) & ((cx >= 76) & (cx <= 134))[None, :]
    crossing = (c.max(0) > road + 0.3) & (c.min(0) < roof - 0.2)
    assert not (box & crossing & ~H).any()
    assert (box & H).any()                                               # 出入口の斜めの面は描かない
    j, i = np.searchsorted(cy, 0.0), np.searchsorted(cx, 105.0)
    assert not H[j, i] and c[:, j, i].min() >= roof                       # トンネルの中ほどの山は残る


def test_road_clear_counts_ground_at_car_height_inside_a_tunnel():
    """トンネルの中でも、地形が路面から天井までの間（車の高さ）に来たら数える。描かない面（H）の中は数えない"""
    sec = _straight()
    xs = np.arange(-10.0, 211.0); ys = np.arange(-30.0, 31.0); X, Y = np.meshgrid(xs, ys)
    Z = np.where((X > 50) & (X < 60), float(sec.z[0]) + 1.0, float(sec.z[0]) - 1.0)    # 筒の中に土
    skip = (sec.s > 49) & (sec.s < 61)
    near = {"xs": xs, "ys": ys, "Z": Z}
    assert tr.road_clear_count(sec, near, skip_rows=skip) > 0
    near["H"] = np.zeros((len(ys) - 1, len(xs) - 1), bool); near["H"][:, (xs[:-1] >= 48) & (xs[:-1] < 62)] = True
    assert tr.road_clear_count(sec, near, skip_rows=skip) == 0


def test_no_fill_under_a_bridge_keeps_nearby_branch_pads_filled():
    """大きな橋の下で谷を盛らない処理が、橋の近くの分かれ道の土台の盛り土まで消さない（S5 9844m で分かれ道が 3m 浮いた）"""
    sec = _straight()
    pad = {"polygon": [[90, 10], [110, 10], [110, 20], [90, 20], [90, 10]], "z": 100.0}
    X, Y = _pts(sec, [15.0])
    nf = np.ones(sec.n, bool)                                            # 全部の行が橋
    Z, _ = tr.carve(X, Y, np.full_like(X, 80.0), sec, pads=[pad], no_fill_rows=nf)   # 谷（80m）
    assert Z[0, 0] == pytest.approx(100.0 - tr.UNDER_DEPTH)


def test_tunnel_keep_rows_do_not_raise_ground_in_the_lake():
    """トンネルの山を残す処理が、湖の中の地形を水位より上に戻さない（吹風トンネルの横で湖に陸の壁）"""
    from shapely.geometry import box
    sec = _straight()
    X, Y = _pts(sec, [30.0])
    keep = np.ones(sec.n, bool)
    Z, _ = tr.carve(X, Y, np.full_like(X, 120.0), sec, keep_rows=keep, lake=(box(0, 20, 200, 60), 90.0))
    assert Z[0, 0] <= 90.0 - tr.LAKE_DEPTH + 1e-9


def test_town_ground_stays_below_the_lake():
    """町の範囲の地面（宮ヶ瀬北原）も、湖の中は水位より下"""
    from shapely.geometry import box
    sec, g, circle, road = _town_case()
    near = tr.build_near(sec, _FlatSampler(), [], town=(g, circle, road), lake=(box(110, 15, 130, 30), 90.0))
    X, Y = np.meshgrid(near["xs"], near["ys"])
    inl = (X > 111) & (X < 129) & (Y > 16) & (Y < 29)
    assert near["Z"][inl].max() <= 90.0 - tr.LAKE_DEPTH + 1e-9


def test_tunnel_holes_leave_cells_in_front_of_the_portal():
    """坑口より外の地形の面（切り土の斜面）は穴にしない（後ろに何もなく、向こうが透けて見える）。筒の中にかかる面だけ"""
    sec = _straight()
    xs = np.arange(60.0, 151.0, 2.0); ys = np.arange(-10.0, 11.0, 2.0)
    Z = np.full((len(ys), len(xs)), 100.0 - 1.0)
    from yabitsu import tunnel as tn
    j0, j1 = np.searchsorted(ys, 0.0), np.searchsorted(ys, 2.0)
    for x in (78.0, 82.0):                                   # 坑口（80m）のすぐ手前（78〜80m）と、2m 奥に、車の高さを横切る面
        i = np.searchsorted(xs, x); Z[j0:j1 + 1, i:i + 2] = 102.0; Z[j0, i] = 99.0
    rows = tn.row_mask(sec.s, [tn.Tunnel(1, "", 80.0, 130.0)], margin_m=tr.BORE_MARGIN_M)   # track_06 と同じ
    H = tr.tunnel_holes(sec, xs, ys, Z, rows)
    assert not H[j0, np.searchsorted(xs, 78.0)] and H[j0, np.searchsorted(xs, 82.0)]


def test_ground_just_outside_the_lake_reaches_the_water_edge():
    """湖の岸のすぐ外（SHORE_M）の地面が水面より低いと、水面の縁が浮いて見える（宮ヶ瀬湖の岸の半分）。水面の高さまで上げる"""
    from shapely.geometry import box
    sec = _straight()
    X, Y = _pts(sec, [31.0, 50.0])                       # 岸の 3m 外と、湖の中
    Z, _ = tr.carve(X, Y, np.full_like(X, 80.0), sec, lake=(box(0, 34, 200, 60), 90.0))
    assert Z[0, 0] >= 90.0 and Z[0, 1] <= 90.0 - tr.LAKE_DEPTH
