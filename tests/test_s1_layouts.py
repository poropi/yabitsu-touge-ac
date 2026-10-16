import math

import numpy as np
import pytest

from yabitsu import meshgen, s1_layouts as lay
from tests.test_meshgen import _straight

NAMES = {f"AC_START_{i}" for i in range(4)} | {f"AC_PIT_{i}" for i in range(4)} | {
    "AC_HOTLAP_START_0", "AC_OPEN_START_L", "AC_OPEN_START_R", "AC_OPEN_FINISH_L", "AC_OPEN_FINISH_R"}


def _by(objs):
    return {d.name: d for d in objs}


@pytest.mark.parametrize("direction", [1, -1])
def test_layout_names_and_order(direction):
    sec = _straight(1001)
    d = _by(lay.layout_objects(sec, [], direction))
    assert set(d) == NAMES
    along = lambda n: d[n].location[0] * direction
    assert along("AC_HOTLAP_START_0") < along("AC_START_0") < along("AC_OPEN_START_L") < along("AC_OPEN_FINISH_L")
    expected = 0.0 if direction > 0 else math.pi
    assert abs(math.remainder(d["AC_START_0"].heading - expected, 2 * math.pi)) < 1e-6


def test_pits_use_pad_when_it_fits():
    sec = _straight(1001)
    pad = {"name": "P", "role": "start", "polygon": [[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]], "z": 100.0}
    d = _by(lay.layout_objects(sec, [pad], 1))
    assert all(10 < d[f"AC_PIT_{i}"].location[0] < 40 and 10 < d[f"AC_PIT_{i}"].location[1] < 40 for i in range(4))


def test_pits_fall_back_to_road_without_pad():
    sec = _straight(1001)
    d = _by(lay.layout_objects(sec, [], -1))
    assert all(abs(d[f"AC_PIT_{i}"].location[1]) < 3.0 for i in range(4))


def test_cameras_cover_track_in_order():
    sec = _straight(1001)
    for direction in (1, -1):
        cams = lay.cameras(sec, direction)
        ins = [c.in_point for c in cams]
        assert ins == sorted(ins) and ins[0] == 0.0 and cams[-1].out_point == 1.0


def test_pad_mesh_triangles_point_up():
    v, f, uv = meshgen.pad_mesh([[0, 0], [10, 0], [10, 5], [0, 5], [0, 0]], 50.0)
    assert f.shape[1] == 3 and (meshgen.face_normals(v, f)[:, 2] > 0).all() and np.allclose(v[:, 2], 50.0)
