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
