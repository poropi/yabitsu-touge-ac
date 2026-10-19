import numpy as np

from yabitsu import forestmap

CFG = forestmap.DEFAULT_CLASSIFY


def _patch(rng, color, noise, n=40):
    return np.clip(np.array(color, float) + rng.normal(0, noise, (n, n, 3)), 0, 255)


def test_classify_synthetic():
    rng = np.random.default_rng(0)
    conifer = _patch(rng, (78, 97, 103), 12)      # 暗い青緑（杉・檜。空中写真の実測）
    broad = _patch(rng, (98, 118, 110), 12)       # やや明るく黄緑寄り（広葉樹）
    field = _patch(rng, (139, 156, 129), 6)       # 伐採地・畑（明るい）
    road = _patch(rng, (163, 152, 137), 6)        # 裸地（明るい茶）
    rgb = np.concatenate([conifer, broad, field, road], 1).astype(np.uint8)   # 横に 4 区画
    cls = forestmap.classify(rgb, px_size=1.0, cfg={**CFG, "cell_m": 4.0, "smooth_cells": 1})
    w = cls.shape[1] // 4
    mid = lambda k: np.bincount(cls[2:-2, k * w + 2:(k + 1) * w - 2].ravel(), minlength=4).argmax()
    assert [mid(k) for k in range(4)] == [1, 2, 0, 0]


def test_forestmap_at_lookup():
    cls = np.array([[1, 2], [0, 1]], np.uint8)     # 行 0 が北
    fm = forestmap.ForestMap(x0=0.0, y1=8.0, cell=4.0, cls=cls)
    got = fm.at(np.array([[2.0, 6.0], [6.0, 6.0], [2.0, 2.0], [6.0, 2.0], [-1.0, 2.0], [2.0, 9.0]]))
    assert got.tolist() == [1, 2, 0, 1, 0, 0]
