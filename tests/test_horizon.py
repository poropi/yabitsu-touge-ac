import numpy as np

from yabitsu import horizon as hz
from yabitsu.meshgen import face_normals


def _far():
    return {"xs": np.arange(-1000.0, 1001.0, 20.0), "ys": np.arange(-2000.0, 2001.0, 20.0)}


def test_horizon_surrounds_the_far_terrain_without_covering_it():
    far = _far()
    xs, ys = hz.grid(far, half=5000.0, cell=100.0)
    keep = hz.face_keep(xs, ys, far)
    cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    # 遠い地形の長方形の真ん中には作らない、外には作る
    assert not keep[np.searchsorted(cy, 0.0), np.searchsorted(cx, 0.0)]
    assert keep[np.searchsorted(cy, 4000.0), np.searchsorted(cx, 0.0)] and keep[np.searchsorted(cy, 0.0), np.searchsorted(cx, 3000.0)]
    # 縁は 1 マス重ねる（すき間なし）
    assert keep[np.searchsorted(cy, 1950.0) - 0, np.searchsorted(cx, 0.0)]


def test_sea_is_zero_and_everything_is_dropped():
    xs, ys = np.array([0.0, 100.0]), np.array([0.0, 100.0])
    Z = hz.heights(xs, ys, lambda X, Y: np.where(X > 50, np.nan, 120.0))
    assert np.allclose(Z[:, 0], 120.0 - hz.DROP) and np.allclose(Z[:, 1], -hz.DROP)


def test_meshes_face_up_with_photo_uv():
    far = _far()
    xs, ys = hz.grid(far, half=5000.0, cell=100.0)
    Z = hz.heights(xs, ys, lambda X, Y: np.zeros_like(X))
    ms = hz.meshes(xs, ys, Z, hz.face_keep(xs, ys, far))
    assert ms and all(name.startswith("HORIZON_") for name, *_ in ms)
    for name, v, f, uv in ms:
        assert (face_normals(v, f)[:, 2] > 0).all()
        assert uv.min() >= 0 and uv.max() <= 1
        south = v[:, 1].argmin()
        assert uv[south, 1] == uv[:, 1].min()          # v=0 が南（画像の下）
