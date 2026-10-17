import numpy as np
import pytest

from yabitsu import centerline as cl
from yabitsu.osm import Route


def _arc(R=50.0, chord=10.0, total=3.0):
    a = np.arange(0, total, chord / R)
    return np.c_[R * np.sin(a), R * (1 - np.cos(a))]  # 原点から +X 向きに出て左へ曲がる


def test_resample_uniform():
    s, xy, step = cl.resample_polyline(np.array([[0.0, 0.0], [10.0, 0.0]]), 1.0)
    assert len(s) == 11 and step == pytest.approx(1.0) and np.allclose(np.diff(xy[:, 0]), 1.0)


def test_curvature_of_left_arc():
    s, xy, step = cl.resample_polyline(_arc(), 1.0)
    xy = cl.smooth_xy(xy, 4.0, step)
    _, k = cl.heading_curvature(xy, step)
    mid = k[len(k) // 4: -len(k) // 4]
    assert mid.mean() == pytest.approx(1 / 50, rel=0.05)


def test_smoothing_moves_little():
    s, xy, step = cl.resample_polyline(_arc(), 1.0)
    sm = cl.smooth_xy(xy, 4.0, step)
    mid = slice(len(xy) // 4, -len(xy) // 4)
    r = np.hypot(sm[mid, 0], sm[mid, 1] - 50)
    assert np.abs(r - 50).max() < 0.5


def test_apply_offsets_left_is_positive():
    s, xy, step = cl.resample_polyline(np.array([[0.0, 0.0], [100.0, 0.0]]), 1.0)
    h, _ = cl.heading_curvature(xy, step)
    out = cl.apply_offsets(s, xy, h, np.array([[0.0, 2.0], [100.0, 2.0]]))
    assert np.allclose(out[:, 1], 2.0)


def test_missing_offsets_csv_is_empty(tmp_path):
    assert cl.load_offsets_csv(tmp_path / "none.csv").shape == (0, 2)


def test_offsets_csv(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("s_m,lateral_m,note\n10,1.5,a\n200,-0.5,\n", encoding="utf-8")
    assert np.allclose(cl.load_offsets_csv(p), [[10, 1.5], [200, -0.5]])


def test_seg_value_at():
    v = cl.seg_value_at(np.array([0.0, 10.0, 20.0]), np.array([1, 2]), np.array([0.0, 9.9, 10.0, 20.0]))
    assert list(v) == [1, 1, 2, 2]


def test_build_centerline_carries_attributes(tmp_path):
    route = Route([0.0, 0.0, 0.0], [0.0, 50.0, 100.0], [1, 2], [False, True])
    c = cl.build_centerline(route, project=lambda lat, lon: (np.asarray(lon, float), np.asarray(lat, float)))
    assert c.s[-1] == pytest.approx(100, abs=0.5)
    assert c.lanes[10] == 1 and c.lanes[90] == 2 and not c.bridge[10] and c.bridge[90]
    c.save(tmp_path / "c.npz")
    c2 = cl.Centerline.load(tmp_path / "c.npz")
    assert np.allclose(c2.x, c.x) and c2.step == pytest.approx(c.step)
