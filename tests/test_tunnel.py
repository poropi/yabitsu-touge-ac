import numpy as np
import pytest

from yabitsu import tunnel
from tests.test_meshgen import _straight
from tests.test_side_roads import _way


def test_tunnel_range_on_the_main_road():
    sec = _straight()
    ws = [_way([[100, 0], [160, 0]], id=7, highway="primary", ref="70", tunnel="yes", layer="-1", name="吹風トンネル"),
          _way([[20, 0], [40, 0]], id=8, highway="primary", ref="70", tunnel="yes")]
    t = sorted(tunnel.from_osm(ws, sec), key=lambda q: q.s0)
    assert [q.osm_id for q in t] == [8, 7]
    assert t[1].s0 == pytest.approx(100.0, abs=0.6) and t[1].s1 == pytest.approx(160.0, abs=0.6) and t[1].name == "吹風トンネル"
    assert t[0].name == ""


def test_tunnels_off_the_course_are_ignored():
    sec = _straight()
    ws = [_way([[100, 50], [160, 50]], id=9, highway="primary", tunnel="yes")]
    assert tunnel.from_osm(ws, sec) == []


def test_row_mask_covers_tunnel_and_margin():
    t = [tunnel.Tunnel(1, "", 100.0, 160.0)]
    s = np.arange(0.0, 201.0)
    m = tunnel.row_mask(s, t, margin_m=5.0)
    assert m[95] and m[165] and not m[94] and not m[166] and m.sum() == 71


def _mountain():
    """直線の本線（高さ 100）。x=100〜160 がトンネル。山は x=90〜170 で 125m、まわりは 100m。"""
    from yabitsu import terrain as tr
    sec = _straight()
    t = [tunnel.Tunnel(7, "吹風トンネル", 100.0, 160.0)]
    xs = np.arange(0.0, 201.0); ys = np.arange(-40.0, 41.0); X, Y = np.meshgrid(xs, ys)
    nat = np.where((X > 90) & (X < 170), 125.0, 100.0)
    keep = tunnel.row_mask(sec.s, t, margin_m=0.0)
    Z, _ = tr.carve(X, Y, nat, sec, keep_rows=keep)
    return sec, t, xs, ys, Z


def test_terrain_is_not_cut_inside_the_tunnel():
    sec, t, xs, ys, Z = _mountain()
    assert Z[np.searchsorted(ys, 0.0), np.searchsorted(xs, 130.0)] == pytest.approx(125.0)        # 道の上の山は残る
    assert Z[np.searchsorted(ys, 0.0), np.searchsorted(xs, 50.0)] < 100.0                          # トンネルの外は今まで通り


def test_terrain_stays_above_the_tunnel_roof():
    from yabitsu import terrain as tr
    sec = _straight(); t = [tunnel.Tunnel(7, "", 100.0, 160.0)]
    X, Y = np.meshgrid(np.arange(100.0, 161.0), np.arange(-5.0, 6.0))
    Z, _ = tr.carve(X, Y, np.full(X.shape, 103.0), sec, keep_rows=tunnel.row_mask(sec.s, t, 0.0))   # 土かぶりが薄い山
    roof = 100.0 + tunnel.bore_height(sec, 130) + tr.TUNNEL_COVER_M
    assert Z.min() >= roof - 1e-6


def test_tunnel_wall_encloses_the_road():
    from yabitsu.crosssection import GUTTER_WIDTH
    from yabitsu.meshgen import face_normals
    sec, t, xs, ys, Z = _mountain()
    ms = {m[0]: m for m in tunnel.meshes(sec, t, {}, lambda x, y: np.full(np.shape(x), 125.0))}
    wall = ms["1WALL_tunnel_7_bore"]
    v, f = wall[1], wall[2]
    mid = np.abs(v[:, 0] - 130.0) < 1.5
    assert v[mid, 1].min() <= -(sec.hw_right[130] + GUTTER_WIDTH) + 1e-6 and v[mid, 1].max() >= sec.hw_left[130] + GUTTER_WIDTH - 1e-6
    assert v[mid, 2].max() - 100.0 >= 4.7
    c = v[f].mean(1); n = face_normals(v, f)
    inward = ((-c[:, 1]) * n[:, 1] + (100.0 + 2.5 - c[:, 2]) * n[:, 2]) > 0
    assert inward.mean() > 0.95


def test_portal_wall_covers_the_cells_cut_over_the_bore():
    """坑口の壁は、筒にかかる地形の面を抜く幅（筒の外へ terrain.NEAR_CELL）より外まであり、横の部分の下の端は外の地形より下。
    左右に 12 m 広げた大きな壁はストリートビューに無い（2026-11-01）ので、筒の外へ PORTAL_SIDE まで。"""
    from yabitsu import terrain as tr
    from yabitsu.crosssection import GUTTER_WIDTH
    sec, t, xs, ys, Z = _mountain()
    from scipy.interpolate import RegularGridInterpolator
    g = RegularGridInterpolator((ys, xs), Z)
    zf = lambda x, y: g(np.c_[np.ravel(y), np.ravel(x)]).reshape(np.shape(x))
    v = {m[0]: m for m in tunnel.meshes(sec, t, {}, zf)}["tunnel_7_portal_a"][1]
    assert np.abs(v[:, 0] - 100.0).max() < 1.5
    l, r = sec.hw_left[100] + GUTTER_WIDTH, sec.hw_right[100] + GUTTER_WIDTH
    assert v[:, 1].max() >= l + tr.NEAR_CELL + 0.5 and v[:, 1].min() <= -(r + tr.NEAR_CELL + 0.5)
    assert v[:, 1].max() <= l + tunnel.PORTAL_SIDE + 1e-6 and v[:, 1].min() >= -(r + tunnel.PORTAL_SIDE) - 1e-6
    for y in (v[:, 1].min(), v[:, 1].max()):
        col = v[np.abs(v[:, 1] - y) < 1e-6]
        assert col[:, 2].min() <= float(zf(np.array([99.0]), np.array([y]))[0]) + 1e-6


def test_ground_does_not_poke_out_in_front_of_the_portal():
    """地形は 2m の格子なので、山を残す所の境が坑口ぴったりだと、坑口の前の道に山の斜面がかぶる。境は坑口より奥にする。"""
    from scipy.interpolate import RegularGridInterpolator
    from yabitsu import terrain as tr
    sec = _straight(); t = [tunnel.Tunnel(7, "", 100.0, 160.0)]
    xs = np.arange(0.0, 201.0, 2.0) + 0.7; ys = np.arange(-20.0, 21.0, 2.0); X, Y = np.meshgrid(xs, ys)
    Z, _ = tr.carve(X, Y, np.where((X > 80) & (X < 180), 130.0, 100.0), sec, keep_rows=tunnel.carve_mask(sec.s, t))
    g = RegularGridInterpolator((ys, xs), Z)
    for x in (99.0, 99.8, 160.2, 161.0):
        assert g([[0.0, x]])[0] < 100.0, x


def test_backfill_closes_the_cut_cells_between_the_wall_and_the_mountain():
    """坑口と山のあいだの山肌は筒にかかる面を描かない（terrain.tunnel_holes）。壁の天端はアーチの縁のすぐ上なので、その上を盛り土で
    ふさぐ：盛り土は壁の天端から始まり、筒の上では坑口から 3〜7 m 奥の山（一番高い所）より低くならない（2026-10-31 にゲームで、壁の
    上から山の中が鏡のように見えた。前は向きを逆にして 4 m 手前の削った地面を測っていた）。"""
    from scipy.interpolate import RegularGridInterpolator
    from yabitsu import terrain as tr
    sec = _straight(); t = [tunnel.Tunnel(7, "", 100.0, 160.0)]
    xs = np.arange(0.0, 201.0, 2.0) + 0.7; ys = np.arange(-30.0, 31.0, 2.0); X, Y = np.meshgrid(xs, ys)
    nat = 100.0 + np.clip(np.minimum(X - 95.0, 165.0 - X), 0.0, None) * 1.5     # 坑口から奥へ登る山
    Z, _ = tr.carve(X, Y, nat, sec, keep_rows=tunnel.carve_mask(sec.s, t))
    g = RegularGridInterpolator((ys, xs), Z)
    zf = lambda x, y: g(np.c_[np.ravel(y), np.ravel(x)]).reshape(np.shape(x))
    ms = {m[0]: m for m in tunnel.meshes(sec, t, {}, zf)}
    for tag, x0, inward in (("a", 100.0, 1.0), ("b", 160.0, -1.0)):
        wall = ms[f"tunnel_7_portal_{tag}"][1]; bf = ms[f"1GRASS_s1_terrain_tunnel_7_{tag}"][1]
        top = wall[:, 2].max()
        assert top <= 100.0 + tunnel.bore_height(sec, 130) + tunnel.RIM_W + tunnel.HEAD_ABOVE + 1e-6   # 写真の坑門の高さ
        for y in (-4.0, 0.0, 4.0):                 # 筒の上（道の幅の中）
            behind = max(float(zf(np.array([x0 + inward * k]), np.array([y]))[0]) for k in tunnel.PORTAL_IN_DEPTHS)
            for dep in (1.0, 2.0, 3.0, 5.0, 7.0):
                col = bf[(np.abs(bf[:, 1] - y) < 0.6) & (np.abs((bf[:, 0] - x0) * inward - dep) < 1.6)]
                assert len(col) and col[:, 2].max() >= min(behind, top) - 1e-6, (tag, y, dep)
            mid = bf[(np.abs(bf[:, 1] - y) < 0.6) & ((bf[:, 0] - x0) * inward > 2.0) & ((bf[:, 0] - x0) * inward < 7.6)]
            assert mid[:, 2].min() >= behind - 1e-6, (tag, y, mid[:, 2].min(), behind)
            front = bf[(np.abs(bf[:, 1] - y) < 0.6) & (np.abs((bf[:, 0] - x0) * inward - tunnel.HEAD_T) < 1e-3)]
            assert np.allclose(front[:, 2], top), (tag, y)


def test_osm_tunnel_ends_can_be_shifted_to_match_street_view(monkeypatch):
    """OSM の範囲がストリートビューとずれるトンネル（S5 4091 m：入口が 20 m 手前・出口が 5 m 奥）は、s5_tunnels.json の
    s0_shift_m・s1_shift_m で端をずらす（2026-11-01）。"""
    sec = _straight()
    ws = [_way([[100, 0], [160, 0]], id=7, highway="primary", ref="70", tunnel="yes")]
    monkeypatch.setattr(tunnel, "_shifts", lambda: {"7": {"s0_shift_m": 20.0, "s1_shift_m": -5.0}})
    t = tunnel.from_osm(ws, sec)[0]
    assert t.s0 == pytest.approx(120.0, abs=0.6) and t.s1 == pytest.approx(155.0, abs=0.6)
