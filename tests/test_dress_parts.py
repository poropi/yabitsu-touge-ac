import numpy as np

from yabitsu import dress_parts
from yabitsu.crosssection import Section


def _sec(n=300):
    s = np.arange(n, dtype=float)
    z = np.full(n, 10.0)
    return Section(s, s.copy(), np.zeros(n), z, np.zeros(n), np.zeros(n), np.full(n, 2, np.int32),
                   np.zeros(n, bool), np.full(n, 3.0), np.full(n, 3.0), np.zeros(n), np.zeros(n))


class _Keep:
    """道路（|y| < 4）と、x=100 付近の建物を植えない範囲とする。"""
    def blocked(self, xy):
        xy = np.asarray(xy)
        return (np.abs(xy[:, 1]) < 4.0) | ((np.abs(xy[:, 0] - 100) < 3) & (xy[:, 1] > 0))


def test_decals_float_above_road():
    sec = _sec()
    items = dress_parts.road_decals(sec, [(0, 200, "manhole", 50.0), (0, 200, "grating", 25.0)], seed=1)
    assert items
    for v, f, uv, tex in items:
        assert np.allclose(v[:, 2], 10.02, atol=0.005), tex


def test_poles_not_on_road():
    sec = _sec()
    poles = dress_parts.utility_poles(sec, 0, 250, "L", 38.0, _Keep(), np.zeros(sec.n))
    assert len(poles) >= 5
    p = np.array(poles)
    assert not _Keep().blocked(p[:, :2]).any() and (p[:, 1] > 0).all()


def test_fence_follows_side():
    sec = _sec()
    v, f, uv = dress_parts.fence_mesh(sec, 50, 120, "R", "wbeam", np.zeros(sec.n))
    assert (v[:, 1] < -3.0).all()
    assert v[:, 0].min() >= 49 and v[:, 0].max() <= 121


def test_wires_sag_between_poles():
    poles = [(0.0, 5.0, 10.0), (40.0, 5.0, 10.0)]
    v, f, uv = dress_parts.wire_mesh(poles, n_wires=1, top=11.0, sag=0.8)
    mid = v[np.abs(v[:, 0] - 20) < 1.5]
    end = v[np.abs(v[:, 0]) < 0.5]
    assert mid[:, 2].mean() < end[:, 2].mean() - 0.5


def test_barrel_and_mirror_stand_on_ground():
    v, f, uv = dress_parts.barrel_mesh((10.0, 5.0, 100.0))
    assert abs(v[:, 2].min() - 100.0) < 1e-6 and 0.8 < v[:, 2].max() - 100.0 < 1.0
    assert np.hypot(v[:, 0] - 10, v[:, 1] - 5).max() <= 0.31
    pole, disc = dress_parts.mirror_meshes((0.0, 0.0, 50.0), facing_deg=90.0)
    assert pole[0][:, 2].max() > 52.5 and disc[0][:, 2].mean() > 52.0
