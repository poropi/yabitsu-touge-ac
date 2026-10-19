import numpy as np

from yabitsu import forestmap

CFG = forestmap.DEFAULT_CLASSIFY


def _patch(rng, color, noise, n=40):
    return np.clip(np.array(color, float) + rng.normal(0, noise, (n, n, 3)), 0, 255)


def _classes(patches):
    rgb = np.concatenate(patches, 1).astype(np.uint8)          # 横に並べる
    cls = forestmap.classify(rgb, px_size=1.0, cfg={**CFG, "cell_m": 4.0, "smooth_cells": 1})
    w = cls.shape[1] // len(patches)
    return [np.bincount(cls[2:-2, k * w + 2:(k + 1) * w - 2].ravel(), minlength=4).argmax() for k in range(len(patches))]


def test_classify_summer():
    rng = np.random.default_rng(0)
    conifer = _patch(rng, (74, 92, 100), 30)       # 暗い青緑・ざらざら（杉・檜。夏の写真の実測に近い）
    broad = _patch(rng, (102, 122, 112), 30)       # 少し明るい・ざらざら（広葉樹）
    field = _patch(rng, (139, 156, 129), 4)        # 伐採地・畑（なめらか）
    bare = _patch(rng, (163, 152, 137), 5)         # 裸地（なめらか）
    assert _classes([conifer, broad, field, bare]) == [1, 2, 0, 0]


def test_classify_winter_deciduous_is_forest():
    rng = np.random.default_rng(1)
    teal = _patch(rng, (110, 150, 145), 35)        # 冬の写真：青緑の植林
    brown = _patch(rng, (160, 140, 118), 35)       # 冬の写真：葉の落ちた広葉樹（茶色・ざらざら）
    clearing = _patch(rng, (175, 160, 140), 5)     # 伐採地（なめらか）
    assert _classes([teal, brown, clearing]) == [1, 2, 0]


def test_clear_polygons():
    fm = forestmap.ForestMap(x0=0.0, y1=40.0, cell=4.0, cls=np.ones((10, 10), np.uint8))
    forestmap.clear_polygons(fm, [np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])])
    assert fm.at(np.array([[10.0, 10.0], [30.0, 30.0]])).tolist() == [0, 1]


def test_forestmap_at_lookup():
    cls = np.array([[1, 2], [0, 1]], np.uint8)     # 行 0 が北
    fm = forestmap.ForestMap(x0=0.0, y1=8.0, cell=4.0, cls=cls)
    got = fm.at(np.array([[2.0, 6.0], [6.0, 6.0], [2.0, 2.0], [6.0, 2.0], [-1.0, 2.0], [2.0, 9.0]]))
    assert got.tolist() == [1, 2, 0, 1, 0, 0]
