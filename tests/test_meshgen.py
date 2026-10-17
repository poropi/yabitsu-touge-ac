import numpy as np
import pytest

from yabitsu import crosssection as cs
from yabitsu import meshgen as mg


def _sec_from_xy(x, y, hw=3.0, lanes=2):
    from yabitsu.centerline import heading_curvature
    xy = np.c_[x, y]
    h, k = heading_curvature(xy, 1.0)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    n = len(s)
    slope, crown = cs.cross_slope(k, np.full(n, lanes), 1.0)
    return cs.Section(s, x, y, np.full(n, 100.0), h, k, np.full(n, lanes, np.int32), np.zeros(n, bool),
                      np.full(n, hw), np.full(n, hw), slope, crown)


def _straight(n=201):
    t = np.arange(float(n)); return _sec_from_xy(t, np.zeros(n))


def _circle(R, n=200):
    a = np.linspace(0, np.pi, n); return _sec_from_xy(R * np.sin(a), R * (1 - np.cos(a)))


def test_ribbon_faces_point_up():
    sec = _straight()
    v, f, uv = mg.road_surface(sec, slice(0, 201))
    assert len(v) == 201 * 5 and len(f) == 200 * 4
    assert (mg.face_normals(v, f)[:, 2] > 0).all()


def test_road_width():
    sec = _straight()
    v, f, uv = mg.road_surface(sec, slice(0, 201))
    row0 = v[:5]
    assert np.hypot(*(row0[-1, :2] - row0[0, :2])) == pytest.approx(6.0)
    assert row0[2, 2] == pytest.approx(100.0) and row0[0, 2] == pytest.approx(100 - 0.06)


def test_hairpin_r10_does_not_fold():
    sec = _circle(10.0)
    v, f, _ = mg.road_surface(sec, slice(0, sec.n))
    assert len(mg.folded_faces(v, f)) == 0


def test_hairpin_tighter_than_half_width_is_detected():
    sec = _circle(2.5)
    v, f, _ = mg.road_surface(sec, slice(0, sec.n))
    assert len(mg.folded_faces(v, f)) > 0


def test_gutter_profile_and_order():
    sec = _straight()
    v, f, uv = mg.gutter(sec, slice(0, 201), 1)
    row0 = v[:4]
    assert row0[0, 1] == pytest.approx(3.0) and row0[2, 1] == pytest.approx(3.6)
    assert row0[1, 2] == pytest.approx(row0[0, 2] - cs.GUTTER_DEPTH)
    vr, fr, _ = mg.gutter(sec, slice(0, 201), -1)
    assert np.all(np.diff(vr[:4, 1]) >= 0)  # 右側も lat が増える順


def test_dashed_line_has_gaps():
    sec = _straight()
    v, f, uv = mg.compact(*mg.line(sec, slice(0, 201), np.zeros(201), dash=(5.0, 5.0)))
    assert 90 <= len(f) <= 110


def test_line_row_mask():
    sec = _straight()
    mask = np.zeros(200, bool); mask[:50] = True
    v, f, uv = mg.compact(*mg.line(sec, slice(0, 201), np.zeros(201), row_mask=mask))
    assert len(f) == 50


def test_left_wall_faces_road_and_right_wall_too():
    sec = _straight()
    n = 201; L = np.full(n, 4.0)
    vl, fl, _ = mg.wall_ribbon(sec, slice(0, n), 1, L, L, np.full(n, 100.0), np.full(n, 101.0))
    vr, fr, _ = mg.wall_ribbon(sec, slice(0, n), -1, L, L, np.full(n, 100.0), np.full(n, 101.0))
    assert (mg.face_normals(vl, fl)[:, 1] < 0).all() and (mg.face_normals(vr, fr)[:, 1] > 0).all()


def test_split_rows():
    assert mg.split_rows(401, 200) == [(0, 200), (200, 400)]
    assert mg.split_rows(450, 200) == [(0, 200), (200, 400), (400, 449)]
