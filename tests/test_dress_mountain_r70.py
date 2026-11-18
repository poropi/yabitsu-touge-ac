"""S4・S5 の仕上げの層で、Codex の部品（r70_guardrail・r70_utility_pole）を道に沿って置く（yabitsu/dress_mountain.py）。"""
import numpy as np
import pytest

from yabitsu import config, track
from yabitsu import dress_mountain as dm
from yabitsu.crosssection import Section
from yabitsu.features import Feature

SEC = config.DATA_PROCESSED / "track_section.json"
pytestmark = pytest.mark.skipif(not SEC.exists(), reason="中心線が無い")


def _sec():
    return Section.load_json(SEC)


def _part(x0=0.0, x1=4.0):
    """+X に x0〜x1、道の側（−Y）に y＝−0.2 の板と、外（+Y）に y＝+0.1 の点、高さ 0〜0.8 の小さな部品。"""
    v = np.array([[x0, -0.2, 0.0], [x1, -0.2, 0.0], [x1, -0.2, 0.8], [x0, 0.1, 0.8]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]]), np.zeros((4, 2))


def _lat_d(sec, p):
    """点 → (いちばん近い行, 中心線の左を正とした横の距離)。"""
    i = int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))
    h = sec.heading[i]
    return i, -(p[0] - sec.x[i]) * np.sin(h) + (p[1] - sec.y[i]) * np.cos(h)


@pytest.mark.parametrize("side", ["L", "R"])
def test_rail_faces_the_road_and_follows_the_line(side):
    """道の側（−Y）の点は基準線より道に近く、外（+Y）の点は遠い。左右どちらの柵でも同じ。高さは縁の路面から。"""
    sec = _sec()
    s0 = float(track.load_bounds()["s4"][0]) + 1000.0
    f = Feature(s0, s0 + 37.0, side, "guardrail", 0.8, "")
    out = dm.r70_guardrail_meshes(sec, f, {"span": _part(), "end": _part(0.0, 0.6)})
    spans = [o for o in out if o[3] == "span"]
    n, L = dm.r70_rail_layout(37.0)
    assert n == 9 and abs(L * n - 37.0) < 1e-9
    v = spans[0][0]
    rows, sd, lat, ze = dm.guardrail_line(sec, f)
    for k in range(0, len(v), 4):
        i_road, d_road = _lat_d(sec, v[k])        # y＝−0.2（道の側）
        i_out, d_out = _lat_d(sec, v[k + 3])      # y＝+0.1（外）
        assert abs(d_road) < abs(d_out)
        assert np.sign(d_road) == sd
    zmin = v[:, 2].min()
    assert abs(zmin - ze.min()) < 0.6           # 縁の路面の高さ（坂の分のずれだけ）


def test_rail_spans_cover_the_whole_line_without_gaps():
    """区切りの始めと終わりが基準線の両端にそろう（端数は全体に配る）。"""
    sec = _sec()
    s0 = float(track.load_bounds()["s5"][0]) + 2000.0
    f = Feature(s0, s0 + 22.0, "L", "guardrail", 0.8, "")
    out = dm.r70_guardrail_meshes(sec, f, {"span": _part(), "end": _part(0.0, 0.6)})
    v = np.concatenate([o[0] for o in out if o[3] == "span"])
    rows, *_ = dm.guardrail_line(sec, f)
    S = [_lat_d(sec, p)[0] for p in v[:, :2]]
    assert min(sec.s[S]) <= sec.s[rows[0]] + 0.5 and max(sec.s[S]) >= sec.s[rows[-1]] - 0.5


def test_pole_crossarm_is_square_to_the_wires():
    """電柱の腕金（部品の +X）は、隣の柱を結ぶ向きと直角。根元は縁の高さから 0.3 m 埋める。"""
    arm = (np.array([[-0.8, 0, 9.5], [0.8, 0, 9.5], [0, 0, 0]], float), np.array([[0, 1, 2]]), np.zeros((3, 2)))
    poles = [(0.0, 0.0, 100.0), (30.0, 0.0, 101.0), (60.0, 0.0, 102.0)]
    v, f, uv = dm.r70_pole_mesh(poles, arm)
    a = v[3:5]                       # 2 本目の腕金の両端
    assert np.allclose(a[:, 0], 30.0) and np.isclose(abs(a[1, 1] - a[0, 1]), 1.6)
    assert np.isclose(v[5, 2], 101.0 - 0.3) and len(f) == 3


@pytest.mark.parametrize("side", ["L", "R"])
def test_rail_faces_keep_their_front_toward_the_road(side):
    """面の表（三角形の巻き方の法線）も道の方を向く。進む向きの右の柵を +s へつなぐと、部品が鏡写しになって表裏が逆になる。"""
    sec = _sec()
    s0 = float(track.load_bounds()["s4"][0]) + 1500.0
    f = Feature(s0, s0 + 16.0, side, "guardrail", 0.8, "")
    out = dm.r70_guardrail_meshes(sec, f, {"span": _part(), "end": _part(0.0, 0.6)})
    v, fc = out[0][0], out[0][1]
    for t in fc[::2]:                      # 各区切りの道の側の三角形（0, 1, 2）
        p0, p1, p2 = v[t]
        n = np.cross(p1 - p0, p2 - p0)[:2]
        i, d = _lat_d(sec, p0)
        h = sec.heading[i]
        to_road = -np.sign(d) * np.array([-np.sin(h), np.cos(h)])
        assert n @ to_road > 0


def test_cover_scale_is_the_smallest_that_covers_the_rim():
    """トンネルの坑口：正面の外形（幅 2・高さ 2 の四角）が点を覆う、いちばん小さい倍率。覆っていれば 1、高さ 3 の点には約 1.5 倍。"""
    v = np.array([[-1, 0, 0], [1, 0, 0], [1, 0, 2], [-1, 0, 2]], float)
    part = (v, np.array([[0, 1, 2], [0, 2, 3]]), np.zeros((4, 2)))
    assert dm.r70_cover_scale(part, [[0.5, 1.0]]) == 1.0
    k = dm.r70_cover_scale(part, [[0.0, 3.0], [0.9, 0.5]], margin=0.0)
    assert 1.49 < k < 1.52


def test_whole_bridge_deck_follows_the_road_gradient():
    """Codex の橋の全体の部品（赤いアーチ橋・青宇治橋）は、床（部品の Z＝0.7）が橋の長さのどこでも路面の高さに来る。
    水平のまま置くと、勾配のある橋で床の横桁が路面から出たり、床が沈んだりした（2026-11-01）。"""
    sec = _sec()
    i = int(np.argmin(np.abs(sec.s - 28782.9)))   # 赤いアーチ橋を置く所（長さ 100 m、路面が 2.8 m 上がる）
    assert abs(sec.z[np.argmin(np.abs(sec.s - 28832.9))] - sec.z[np.argmin(np.abs(sec.s - 28732.9))]) > 2.0
    ys = np.arange(-50.0, 50.1, 5.0)
    pts = np.c_[np.zeros_like(ys), ys, np.full_like(ys, 0.7)]
    v = dm.r70_whole_bridge_verts(sec, i, pts)
    for p in v:
        j, d = _lat_d(sec, p)
        assert abs(d) < 0.5 and abs(p[2] - sec.z[j]) < 0.05, (p, sec.z[j])


def _plate_part():
    """名板の札 3 枚（幅 0.46 m・間 0.18 m、y＝−0.488、z 5.62〜6.08、uv は 8〜398 px を 3 等分・v 325〜451 px）と、坑門の面の三角形 1 枚。"""
    v, uv, f = [], [], []
    for k in range(3):
        x0 = (k - 1) * 0.64 - 0.23
        u0, u1 = (8 + k * 130) / 1024, (8 + (k + 1) * 130) / 1024
        b = len(v)
        v += [[x0, -0.488, 5.62], [x0 + 0.46, -0.488, 5.62], [x0 + 0.46, -0.488, 6.08], [x0, -0.488, 6.08]]
        uv += [[u0, 451 / 1024], [u1, 451 / 1024], [u1, 325 / 1024], [u0, 325 / 1024]]
        f += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    b = len(v); v += [[-3, -0.436, 0], [3, -0.436, 0], [0, -0.436, 6.8]]; uv += [[0.9, 0.9]] * 3; f += [[b, b + 1, b + 2]]
    return np.array(v, float), np.array(f), np.array(uv, float)


def test_nameplate_tiles_are_respaced_to_the_photo():
    """吹風トンネル出口の名板：写真（R70-0992_b）は札が約 0.55 m 角・0.81 m おき。札の中心の高さと並びの真ん中は変えない。坑門の面は動かさない。"""
    v, f, uv = _plate_part()
    v2 = dm.r70_respace_plate(v, f, uv, size_m=0.55, pitch_m=0.81)
    tiles = [v2[4 * k:4 * k + 4] for k in range(3)]
    cx = [t[:, 0].mean() for t in tiles]
    assert np.allclose(np.diff(cx), 0.81) and abs(np.mean(cx)) < 1e-9
    assert all(np.ptp(t[:, 0]) == pytest.approx(0.55) and np.ptp(t[:, 2]) == pytest.approx(0.55) for t in tiles)
    assert all(t[:, 2].mean() == pytest.approx(5.85) for t in tiles)
    assert np.array_equal(v2[12:], v[12:])


def test_nameplate_texture_has_dark_ink_on_slate_tiles():
    """札の絵：地は坑門の面の色 × (0.65, 0.75, 0.91)、字は × (0.29, 0.36, 0.48)（写真の札・字・下の壁の色の比）。札の外の絵は変えない。"""
    from PIL import Image
    img = Image.new("RGB", (1024, 1024), (110, 115, 106))
    out = np.asarray(dm.r70_plate_texture(img, "吹風", cells=[(8, 325, 138, 451), (138, 325, 268, 451)], face_rgb=(110, 115, 106)), float)
    cell = out[325:451, 8:138].reshape(-1, 3)
    L = cell.mean(1)
    assert np.allclose(cell[L >= np.percentile(L, 90)].mean(0), np.array([110, 115, 106]) * [0.65, 0.75, 0.91], atol=3)
    assert np.allclose(cell[L <= np.percentile(L, 3)].mean(0), np.array([110, 115, 106]) * [0.29, 0.36, 0.48], atol=6)
    assert (out[600:700, 600:700] == [110, 115, 106]).all()


def test_portal_setback_puts_the_face_just_in_front_of_the_wall():
    """坑口の正面（部品の y＝face_y）を、今の坑口の壁の正面（縁の真ん中から外へ wall_out m）の 3 cm 手前に。外は部品の −Y。"""
    q = np.array([[0, -0.436 * 1.13, 3.0], [0, 1.364 * 1.13, 3.0], [0, -1.364 * 1.13, 1.0]])
    q2 = dm.r70_portal_setback(q, face_y=-0.436 * 1.13, wall_out=-0.12)
    assert -q2[0, 1] == pytest.approx(-0.12 + 0.03)
    assert np.allclose(q2[:, 1] - q[:, 1], q2[0, 1] - q[0, 1]) and np.array_equal(q2[:, [0, 2]], q[:, [0, 2]])


def test_wall_normal_follows_the_wall_not_the_tunnel_chord():
    """坑口の壁（横に長い板）の正面の向き。トンネルの両端を結ぶ向き（out_hint）から 10° ずれていても、壁の面に直角で、out_hint と同じ側を向く。"""
    a = np.radians(10.0); lat = np.array([np.cos(a), np.sin(a)]); nrm = np.array([lat[1], -lat[0]])
    pts = np.array([s * lat + t * nrm for s in np.linspace(-8, 8, 9) for t in (0.0, -0.6)]) + [100.0, 50.0]
    o = dm.r70_wall_normal(pts, out_hint=np.array([0.0, -1.0]))
    assert np.allclose(o, nrm) or np.allclose(o, -nrm)
    assert o @ np.array([0.0, -1.0]) > 0
