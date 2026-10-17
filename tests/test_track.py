import numpy as np
import pytest

from yabitsu import elevation, geo, track
from yabitsu.centerline import build_centerline
from yabitsu.config import SectionDef
from yabitsu.osm import Route


def _route(xy, lanes=1):
    lat, lon = geo.from_local(*np.asarray(xy, float).T)
    n = len(xy)
    return Route(list(map(float, lat)), list(map(float, lon)), [lanes] * (n - 1), [False] * (n - 1))


def _two_routes():
    lower = _route([[0, 0], [100, 0], [200, 0]])
    upper = _route([[200, 0], [300, 0], [400, 0]], lanes=2)
    return lower, upper


def test_join_routes_drops_duplicate_joint_and_reports_it():
    lower, upper = _two_routes()
    r, joints = track.join_routes([lower, upper])
    assert len(r.lat) == 5 and len(r.seg_lanes) == 4 and r.seg_lanes == [1, 1, 2, 2]
    assert joints == [(upper.lat[0], upper.lon[0])]


def test_join_routes_rejects_gap():
    lower = _route([[0, 0], [100, 0], [200, 0]])
    upper = _route([[205, 0], [300, 0]])
    with pytest.raises(ValueError):
        track.join_routes([lower, upper])


def test_section_bounds_cover_track_without_gaps():
    r, joints = track.join_routes(list(_two_routes()))
    cl = build_centerline(r)
    defs = [SectionDef("s2", (0, 0), (0, 0)), SectionDef("s1", (0, 0), (0, 0))]
    b = track.section_bounds(cl, defs, joints)
    assert b["s2"][0] == 0.0 and b["s1"][1] == cl.s[-1] and b["s2"][1] == b["s1"][0]
    assert abs(b["s1"][0] - 200.0) < 1.0


def test_load_section_rows_shifts_by_section_start(tmp_path):
    (tmp_path / "s2_widths.csv").write_text("s_from,s_to,width_m,lanes,note\n0,100,5.0,1,a\n100,999,6.0,1,b\n", encoding="utf-8")
    (tmp_path / "s1_widths.csv").write_text("s_from,s_to,width_m,lanes,note\n0,50,7.0,2,c\n", encoding="utf-8")
    rows = track.load_section_rows("widths", {"s2": (0.0, 300.0), "s1": (300.0, 500.0)}, tmp_path)
    got = [(r["s_from"], r["s_to"], r["width_m"]) for r in rows]
    assert got == [(0.0, 100.0, "5.0"), (100.0, 300.0, "6.0"), (300.0, 350.0, "7.0")]


def _fit_inputs():
    s = np.arange(0.0, 800.0, 1.0)
    rng = np.random.default_rng(0)
    z = np.where(s < 400, 300 + 0.05 * s, 320 + 0.08 * (s - 400)) + rng.normal(0, 0.3, len(s))
    return s, z, np.ones(len(s), bool)


def test_fit_sections_keeps_upper_section_exact():
    s, z_raw, valid = _fit_inputs()
    z, z_ref, sigmas = elevation.fit_sections(s, z_raw, valid, 1.0, [(0.0, 400.0), (400.0, 799.0)])
    m = s >= 400
    z_up, _, _ = elevation.fit_profile(s[m], z_raw[m], valid[m], 1.0)
    assert np.allclose(z[m], z_up) and len(sigmas) == 2


def test_fit_sections_upper_sections_do_not_move_when_a_section_is_added_below():
    # 下に区間を足すと通しの s の格子が区間の始点に対してずれる（ここでは 0.37m）。高さは区間の始点からの
    # 格子で測って合わせるので、上の区間の高さは足す前と同じ（道の上の同じ場所どうしで比べる）
    rng = np.random.default_rng(3)
    bumps = rng.normal(0, 1.5, 900)
    terrain = lambda d: 100 + 0.06 * d + np.interp(d, np.arange(900.0), bumps)  # d = 上の区間の始点からの距離
    alone = np.arange(0.0, 800.0)
    z0, _, _ = elevation.fit_sections(alone, terrain(alone), np.ones_like(alone, bool), 1.0, [(0.0, 799.0)],
                                      sample=terrain)
    off = 300.37
    s = np.append(np.arange(0.0, off + 799.0), off + 799.0)  # 中心線の格子と同じく、最後は道の終わりちょうど
    z1, _, _ = elevation.fit_sections(s, terrain(s - off), np.ones_like(s, bool), 1.0, [(0.0, off), (off, s[-1])],
                                      sample=lambda q: terrain(q - off))
    m = s >= off + 30
    assert np.abs(z1[m] - np.interp(s[m] - off, alone, z0)).max() <= 0.01


def test_fit_sections_joins_value_and_slope():
    s, z_raw, valid = _fit_inputs()
    z, _, _ = elevation.fit_sections(s, z_raw, valid, 1.0, [(0.0, 400.0), (400.0, 799.0)])
    assert abs(z[400] - z[399]) < 0.1
    g_below = (z[400] - z[390]) / 10; g_above = (z[410] - z[400]) / 10
    assert abs(g_below - g_above) < 0.01


def test_section_view_has_local_distance():
    r, joints = track.join_routes(list(_two_routes()))
    cl = build_centerline(r)
    defs = [SectionDef("s2", (0, 0), (0, 0)), SectionDef("s1", (0, 0), (0, 0))]
    b = track.section_bounds(cl, defs, joints)
    v = track.section_view(cl, b, "s1")
    assert v.s[0] == 0.0 and abs(v.s[-1] - (b["s1"][1] - b["s1"][0])) < 1e-9
    assert np.allclose(v.x, cl.x[cl.s >= b["s1"][0] - 1e-9]) and len(v.heading) == len(v.s)


def test_load_bounds_roundtrip(tmp_path):
    np.savez(tmp_path / "track_profile.npz", s=np.arange(3.0), bounds=np.array({"s2": (0.0, 1.0), "s1": (1.0, 2.0)}, dtype=object))
    assert track.load_bounds(tmp_path) == {"s2": (0.0, 1.0), "s1": (1.0, 2.0)}


# --- 実データのつなぎ目（S2 を足したあと）。処理済みデータに2区間以上あるときだけ動く ---
from yabitsu import config  # noqa: E402

_PROF = config.DATA_PROCESSED / "track_profile.npz"
_CL = config.DATA_PROCESSED / "track_centerline.npz"


def _real():
    from yabitsu.centerline import Centerline
    pr = np.load(_PROF, allow_pickle=True)
    return Centerline.load(_CL), pr, pr["bounds"].item()


_joined = pytest.mark.skipif(not (_PROF.exists() and _CL.exists() and len(np.load(_PROF, allow_pickle=True)["bounds"].item()) > 1)
                             if _PROF.exists() else True, reason="2区間の処理済みデータがまだない")


@_joined
def test_join_has_no_kink():
    cl, pr, bounds = _real()
    for (_, b) in list(bounds.values())[:-1]:
        m = (cl.s >= b - 20) & (cl.s <= b + 20)
        # 折れ＝曲がり方（1m ごとの向きの変化）が急に変わること。ヘアピンの急カーブそのものは折れではない
        assert np.abs(np.diff(np.diff(cl.heading[m]))).max() <= 0.02


@_joined
def test_join_has_no_grade_jump():
    cl, pr, bounds = _real()
    s, z = pr["s"], pr["z"]
    for (_, b) in list(bounds.values())[:-1]:
        i = int(np.argmin(np.abs(s - b)))
        below = (z[i] - z[i - 10]) / (s[i] - s[i - 10]); above = (z[i + 10] - z[i]) / (s[i + 10] - s[i])
        assert abs(below - above) <= 0.02


@_joined
def test_join_profile_close_to_dem():
    cl, pr, bounds = _real()
    s = pr["s"]
    for (_, b) in list(bounds.values())[:-1]:
        m = (s >= b - 60) & (s <= b + 60) & pr["valid"]
        assert np.abs(pr["z"] - pr["z_ref"])[m].max() <= 1.5


@_joined
def test_s3_starts_at_nagoki():
    cl, pr, bounds = _real()
    if "s3" not in bounds:
        pytest.skip("S3 がまだない")
    x, y = geo.to_local(*config.NAGOKI)
    assert np.hypot(cl.x[0] - float(x), cl.y[0] - float(y)) <= 3.0
