import numpy as np
import pytest

from yabitsu import pavements as pv
from yabitsu.crosssection import surface_height
from yabitsu.meshgen import face_normals
from tests.test_meshgen import _straight


def _mesh(kind, a=10.0, b=28.0):
    sec = _straight(201)
    (name, v, f, uv, mat), = pv.meshes(sec, pv.load_rows([{"s_from": a, "s_to": b, "kind": kind, "note": ""}]))
    return sec, name, v, f, mat


def test_red_stripes_count_and_width():
    sec, name, v, f, mat = _mesh("red_stripes")
    assert mat == "paint_red" and name == "LINES_track_pave_000"
    assert len(f) == 10 * 4                                 # 18m の範囲、1.8m おきに10本（横に4面ずつ）
    assert v[:, 1].max() == pytest.approx(sec.hw_left[0]) and v[:, 1].min() == pytest.approx(-sec.hw_right[0])
    widths = [np.ptp(v[q][:, 0]) for q in f]
    assert np.allclose(widths, 0.6)


def test_crosswalk_bars_parallel_to_road():
    sec, name, v, f, mat = _mesh("crosswalk", 50.0, 54.0)
    assert mat == "line_white"
    for q in f:
        p = v[q]
        assert np.ptp(p[:, 0]) == pytest.approx(4.0) and np.ptp(p[:, 1]) == pytest.approx(0.45)
    assert v[:, 1].max() <= sec.hw_left[0] - 0.5 + 1e-9


def test_concrete_circles_cover_range():
    sec, name, v, f, mat = _mesh("concrete_circles", 0.0, 40.0)
    assert mat == "concrete_circles" and v[:, 0].min() == pytest.approx(0.0) and v[:, 0].max() == pytest.approx(40.0)


@pytest.mark.parametrize("kind", ["red_stripes", "crosswalk", "concrete_circles"])
def test_pavement_faces_point_up_and_sit_above_road(kind):
    sec, name, v, f, mat = _mesh(kind)
    assert (face_normals(v, f)[:, 2] > 0).all()
    z_road = surface_height(v[:, 1], 100.0, np.interp(v[:, 0], sec.s, sec.slope), np.interp(v[:, 0], sec.s, sec.crown))
    assert np.all((v[:, 2] - z_road > 0.004) & (v[:, 2] - z_road < 0.006))


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        pv.load_rows([{"s_from": 0, "s_to": 1, "kind": "zebra", "note": ""}])


def test_red_stripes_follow_the_road_crown():
    # 路面は中央が高い（片勾配・横断勾配）。帯が端から端まで1枚の平面だと、中央が路面の下に隠れる
    sec, name, v, f, mat = _mesh("red_stripes")
    assert np.any(np.abs(v[:, 1]) < 1e-9)          # 中央にも頂点がある
    assert len(v) == 10 * 2 * 5                     # 帯ごとに 長さ2 × 横5 点


def test_stop_line_covers_left_half_only():
    sec, name, v, f, mat = _mesh("stop_line", 50.0, 50.45)
    assert mat == "line_white"
    assert v[:, 1].min() == pytest.approx(0.0) and v[:, 1].max() == pytest.approx(sec.hw_left[0])
    assert np.ptp(v[:, 0]) == pytest.approx(0.45)


def test_stop_line_down_covers_right_half():
    sec, name, v, f, mat = _mesh("stop_line_down", 50.0, 50.45)
    assert v[:, 1].max() == pytest.approx(0.0) and v[:, 1].min() == pytest.approx(-sec.hw_right[0])


def test_red_half_down_paints_the_right_half():
    sec, name, v, f, mat = _mesh("red_half_down", 50.0, 70.0)
    assert mat == "paint_red" and v[:, 1].max() == pytest.approx(0.0) and v[:, 1].min() == pytest.approx(-sec.hw_right[0])
