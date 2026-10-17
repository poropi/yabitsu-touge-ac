import numpy as np
import pytest

from yabitsu import town_ground as tg
from yabitsu.crosssection import GUTTER_WIDTH, surface_height
from tests.test_meshgen import _straight


class _Dem:
    """標高（ローカル座標で直接）。unproject を恒等にして使う。"""
    def __init__(self, f):
        self.f = f

    def sample(self, x, y):
        return self.f(np.asarray(x, float), np.asarray(y, float))


IDENT = lambda x, y: (x, y)


def _ground(f, sec=None, **kw):
    sec = sec or _straight()   # 道は x=0〜200、y=0、高さ 100、幅 6
    return sec, tg.build(sec, _Dem(f), (100.0, 0.0), radius=60.0, unproject=IDENT, **kw)


def test_ground_equals_road_surface_on_main_road():
    sec, g = _ground(lambda x, y: np.full_like(x, 95.0))
    for lat in np.linspace(-3.0, 3.0, 13):
        for s in (70.0, 100.0, 130.0):
            i = int(s)
            want = surface_height(lat, sec.z[i], sec.slope[i], sec.crown[i])
            assert float(g.z(s, lat)) == pytest.approx(want, abs=0.01)


def test_ground_blends_to_smoothed_dem():
    sec, g = _ground(lambda x, y: np.full_like(x, 95.0), blend_m=8.0)
    edge = sec.hw_left[100] + GUTTER_WIDTH
    assert float(g.z(100.0, edge + 8.5)) == pytest.approx(95.0, abs=0.01)
    assert float(g.z(100.0, -(edge + 20.0))) == pytest.approx(95.0, abs=0.01)
    mid = float(g.z(100.0, edge + 4.0))
    assert 95.0 < mid < float(sec.edge_z(1)[100])


def test_ground_is_smooth():
    rng = np.random.default_rng(1)
    bumps = rng.normal(0, 1.5, (200, 200))
    f = lambda x, y: 95.0 + bumps[np.clip(np.round(y).astype(int) + 100, 0, 199), np.clip(np.round(x).astype(int), 0, 199)]
    sec, g = _ground(f)
    ys = np.arange(15.0, 50.0)
    z = g.z(np.full_like(ys, 100.0), ys)
    assert np.abs(np.diff(z)).max() <= 0.25


def test_save_load_roundtrip(tmp_path):
    sec, g = _ground(lambda x, y: np.full_like(x, 95.0))
    g.save(tmp_path / "g.npz")
    h = tg.TownGround.load(tmp_path / "g.npz")
    assert np.array_equal(h.Z, g.Z) and float(h.z(100.0, 30.0)) == float(g.z(100.0, 30.0))
