import numpy as np
import pytest

from yabitsu import trees as tr


def _near():
    xs = np.arange(0.0, 401.0, 2.0); ys = np.arange(-120.0, 121.0, 2.0)
    X, Y = np.meshgrid(xs, ys)
    return {"xs": xs, "ys": ys, "Z": 100 + 0.1 * X, "E": np.maximum(np.abs(Y) - 3.6, 0), "D": np.abs(Y)}


def test_scatter_keeps_clear_of_road_and_inside_band():
    t = tr.scatter(_near(), density=1 / 40)
    assert len(t) > 0
    assert (np.abs(t[:, 1]) >= 3.6 + 3.0 - 1.0).all() and (np.abs(t[:, 1]) <= 105 + 1.0).all()


def test_scatter_density():
    t = tr.scatter(_near(), density=1 / 40)
    area = 400 * 2 * (105 - 6.6)
    assert len(t) == pytest.approx(area / 40, rel=0.2)


def test_tree_base_on_terrain():
    t = tr.scatter(_near())
    assert np.allclose(t[:, 2], 100 + 0.1 * t[:, 0], atol=0.05)


def test_scatter_avoids_pads():
    pad = {"polygon": [[100, 20], [200, 20], [200, 60], [100, 60], [100, 20]], "z": 100}
    t = tr.scatter(_near(), pads=[pad])
    inside = (t[:, 0] > 100) & (t[:, 0] < 200) & (t[:, 1] > 20) & (t[:, 1] < 60)
    assert not inside.any()


def test_cards_shape():
    t = np.array([[0.0, 0.0, 10.0, 12.0], [5.0, 5.0, 11.0, 9.0]])
    v, f, uv = tr.cards(t)
    assert v.shape == (16, 3) and f.shape == (4, 4) and v[:, 2].max() == pytest.approx(10 + 12)
