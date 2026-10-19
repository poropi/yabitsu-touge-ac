import numpy as np
from PIL import Image

from yabitsu import dress_tex


def test_height_to_normal_flat():
    n = dress_tex.height_to_normal(np.full((16, 16), 0.5), strength=2.0)
    assert np.allclose(n[..., 2], 1.0) and np.allclose(n[..., :2], 0.0)


def test_height_to_normal_slope_points_downhill():
    g = np.tile(np.linspace(0, 1, 32), (32, 1))         # 右へ高くなる
    n = dress_tex.height_to_normal(g, strength=4.0)
    assert (n[8:-8, 8:-8, 0] < 0).all()


def test_make_tileable_edges():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:96]
    a = gaussian_filter(rng.normal(0, 1, (64, 96)), 3) * 40 + xx * 1.5 + yy * 1.5   # 端どうしが違うなめらかな画像
    a = np.repeat(((a - a.min()) / (a.max() - a.min()) * 255)[..., None], 3, 2)
    t = np.asarray(dress_tex.make_tileable(Image.fromarray(a.astype(np.uint8))), float)
    assert np.abs(t[:, 0] - t[:, -1]).mean() < 6 and np.abs(t[0] - t[-1]).mean() < 6


def test_circle_alpha():
    img = dress_tex.circle_alpha(Image.new("RGB", (64, 64), (90, 90, 90)))
    a = np.asarray(img)[..., 3]
    assert a[32, 32] == 255 and a[0, 0] == 0
