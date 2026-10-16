import math
import numpy as np
import pytest
from yabitsu import acobjects as ao

S = np.arange(0.0, 201.0)
STRAIGHT = {"s": S, "x": S.copy(), "y": np.zeros_like(S), "z": np.full_like(S, 10.0), "heading": np.zeros_like(S)}


def test_start_grid_forward():
    g = ao.start_grid(STRAIGHT, s0=50, direction=1, n=4, spacing=8, stagger=1.2)
    assert [d.name for d in g] == ["AC_START_0", "AC_START_1", "AC_START_2", "AC_START_3"]
    assert [round(d.location[0]) for d in g] == [50, 42, 34, 26]
    assert [round(d.location[1], 1) for d in g] == [1.2, -1.2, 1.2, -1.2]
    assert all(abs(d.heading) < 1e-9 for d in g)
    assert g[0].location[2] == pytest.approx(10.0 + ao.DUMMY_Z_OFFSET)


def test_start_grid_reverse():
    g = ao.start_grid(STRAIGHT, s0=150, direction=-1, n=2, spacing=8, stagger=1.2)
    assert [round(d.location[0]) for d in g] == [150, 158]
    assert g[0].heading == pytest.approx(math.pi)
    assert g[0].location[1] == pytest.approx(-1.2)  # 下り向きの左側 = -y


def test_timing_gate_sides():
    fwd = ao.timing_gate(STRAIGHT, 60, 1, 4.0, "START")
    assert [d.name for d in fwd] == ["AC_OPEN_START_L", "AC_OPEN_START_R"]
    assert fwd[0].location[1] == pytest.approx(4.0) and fwd[1].location[1] == pytest.approx(-4.0)
    rev = ao.timing_gate(STRAIGHT, 60, -1, 4.0, "FINISH")
    assert rev[0].name == "AC_OPEN_FINISH_L" and rev[0].location[1] == pytest.approx(-4.0)


def test_heading_interpolates_across_wrap():
    path = {"s": np.array([0.0, 2.0]), "x": np.zeros(2), "y": np.zeros(2), "z": np.zeros(2),
            "heading": np.array([math.pi - 0.1, -math.pi + 0.1])}
    _, _, _, h = ao.frame_at(path, 1.0)
    assert abs(abs(h) - math.pi) < 1e-6


def test_pit_boxes_side_by_side():
    p = ao.pit_boxes((0.0, 0.0), 5.0, 0.0, n=3, spacing=4.0)
    assert [d.name for d in p] == ["AC_PIT_0", "AC_PIT_1", "AC_PIT_2"]
    ys = [d.location[1] for d in p]
    assert ys == pytest.approx([4.0, 0.0, -4.0]) and all(d.location[0] == pytest.approx(0) for d in p)


def test_rotation_uses_offset():
    d = ao.Dummy("AC_PIT_0", (0, 0, 0), 1.0)
    assert d.rotation_z == pytest.approx(1.0 + ao.DUMMY_YAW_OFFSET)
