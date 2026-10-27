import pytest
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


def test_dense_smooth_canopy_is_forest():
    rng = np.random.default_rng(2)
    rough = _patch(rng, (74, 92, 100), 30)         # ふつうの森（ざらざら）
    dense = _patch(rng, (70, 90, 92), 5)           # 葉が詰まってなめらかに写る広葉樹（周りより暗い）
    field = _patch(rng, (139, 156, 129), 4)        # 伐採地（なめらかで明るい）
    assert [c > 0 for c in _classes([rough, dense, field])] == [True, True, False]


def test_settlement_keeps_only_dark_green():
    """家の近く（集落）では、暗い緑だけを森にする（田んぼ・畑・屋根の並びはざらざらでも森にしない）。"""
    rng = np.random.default_rng(3)
    forest = _patch(rng, (80, 98, 104), 25)          # 実測：明るさ 0.39、R/G 0.85
    field = _patch(rng, (150, 152, 145), 25)         # 実測：明るさ 0.57、R/G 0.99（ざらざら）
    rgb = np.concatenate([forest, field], 1).astype(np.uint8)       # 80×40 px（1 m/px）
    fm = forestmap.ForestMap(x0=0.0, y1=40.0, cell=4.0, cls=np.ones((10, 20), np.uint8))
    house = [np.array([[60.0, 18.0], [64.0, 18.0], [64.0, 22.0], [60.0, 22.0]])]
    mask = forestmap.settlement_mask(fm, house, [], radius_m=100.0)
    forestmap.settlement_filter(fm, rgb, 1.0, mask, CFG)
    assert fm.at(np.array([[10.0, 20.0], [70.0, 20.0]])).tolist() == [1, 0]


def test_settlement_mask_radius():
    fm = forestmap.ForestMap(x0=0.0, y1=400.0, cell=4.0, cls=np.ones((100, 100), np.uint8))
    house = [np.array([[200.0, 200.0], [204.0, 200.0], [204.0, 204.0], [200.0, 204.0]])]
    m = forestmap.settlement_mask(fm, house, [], radius_m=40.0)
    r = lambda x, y: m[int((400 - y) // 4), int(x // 4)]
    assert r(202.0, 230.0) and not r(202.0, 260.0)


def test_settlement_drops_speckles():
    """集落の中の 1 マスだけの森（家の影）は消す。まとまった林は残す。"""
    cls = np.zeros((20, 20), np.uint8)
    cls[3, 3] = 1                       # 影の点
    cls[10:16, 10:16] = 1               # 6×6 マスの林
    fm = forestmap.ForestMap(x0=0.0, y1=80.0, cell=4.0, cls=cls)
    rgb = np.full((80, 80, 3), (80, 98, 104), np.uint8)           # 全部「暗い緑」
    forestmap.settlement_filter(fm, rgb, 1.0, np.ones((20, 20), bool), CFG)
    assert fm.cls[3, 3] == 0 and fm.cls[12, 12] == 1


def test_brown_near_houses_is_farmland():
    """家から farm_m 以内の茶色い所（耕した畑・刈った田）は、ざらざらでも森にしない（S3 3400 の神社の西の畑が森になっていた）。
    遠い山の茶色（冬の写真の落葉樹）は森のまま。"""
    rng = np.random.default_rng(4)
    brown = _patch(rng, (150, 128, 108), 30, n=40)
    rgb = np.concatenate([brown] * 10, 1).astype(np.uint8)            # 40×400 px（1 m/px）
    fm = forestmap.ForestMap(x0=0.0, y1=40.0, cell=4.0, cls=np.full((10, 100), 2, np.uint8))
    house = [np.array([[0.0, 18.0], [4.0, 18.0], [4.0, 22.0], [0.0, 22.0]])]
    forestmap.farm_filter(fm, rgb, 1.0, house, [], {**CFG, "farm_m": 150.0})
    assert fm.at(np.array([[60.0, 20.0], [380.0, 20.0]])).tolist() == [0, 2]


def test_restore_above_water_brings_back_forest_on_land_inside_a_lake_outline():
    """OSM の湖の輪郭が陸の斜面まで含んでいた（北原の南東の丘）：水面＋3 m より高く、写真で森のマスだけ森に戻す。"""
    fm = forestmap.ForestMap(x0=0.0, y1=40.0, cell=4.0, cls=np.ones((10, 10), np.uint8))
    raw = fm.cls.copy(); raw[:, 9] = 0                                         # 東の端の列は写真でも森でない
    lake = [np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]])]
    forestmap.clear_polygons(fm, lake)
    zf = lambda xy: np.where(np.asarray(xy)[:, 0] < 20.0, 284.0, 300.0)       # 西半分は水面の高さ、東半分は陸
    n = forestmap.restore_above_water(fm, raw, lake, zf, 285.0)
    got = fm.at(np.array([[10.0, 10.0], [30.0, 10.0], [38.0, 10.0]]))
    assert got.tolist() == [0, 1, 0] and n == 10 * 4                          # 東半分（x 20〜36）の 4 列だけ


def test_mountain_forest_greens_steep_slopes_but_keeps_bare_rock_villages_and_flat_fields():
    """山の斜面は森：急な斜面の開けた土地のうち、写真で森（集落で外した茶色い広葉樹）か緑っぽい所を森に。ベージュの崩れ・家のまわり・平らな畑は残す。"""
    cls = np.zeros((4, 4), np.uint8)
    fm = forestmap.ForestMap(x0=0.0, y1=16.0, cell=4.0, cls=cls)
    raw = np.zeros((4, 4), np.uint8); raw[0, 0] = 2                                # (0,0)：写真では広葉樹（集落の判定で外した）
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[...] = (90, 110, 90)                                                      # 緑っぽい（夏の森）
    rgb[0:2, 2:4] = (200, 185, 160)                                               # (0,1)：ベージュのむき出しの土
    keep = np.zeros((4, 4), bool); keep[0, 2] = True                              # (0,2)：家のまわり
    slope = np.full((4, 4), 30.0, np.float32); slope[0, 3] = 3.0                  # (0,3)：平らな畑
    n = forestmap.mountain_forest(fm, raw, rgb, 2.0, keep, slope)
    assert fm.cls[0].tolist() == [2, 0, 0, 0]
    assert (fm.cls[1:] == 2).all() and n == 1 + 12


def test_slope_map_measures_degrees_from_the_ground():
    fm = forestmap.ForestMap(x0=0.0, y1=40.0, cell=4.0, cls=np.zeros((10, 10), np.uint8))
    s = forestmap.slope_map(fm, lambda xy: np.asarray(xy)[:, 0] * 0.5)            # 東へ 1 m で 0.5 m 上る ＝ 26.6°
    assert s[5, 5] == pytest.approx(26.57, abs=0.05)


def test_settlement_keeps_textured_light_green_as_garden():
    """集落で森にしない明るい緑のうち、緑っぽい所（G>B）は庭木（4）、灰色の屋根の並びは何も無し（名古木の道ぞい、2026-10-28）。"""
    cls = np.ones((10, 20), np.uint8)
    fm = forestmap.ForestMap(x0=0.0, y1=40.0, cell=4.0, cls=cls)
    rgb = np.zeros((40, 80, 3), np.uint8)
    rgb[:, :40] = (150, 175, 110)       # 明るい緑（庭木・生け垣）：明るさ 0.57 は settle_lum_max より明るい
    rgb[:, 40:] = (140, 140, 145)       # 灰色の屋根
    forestmap.settlement_filter(fm, rgb, 1.0, np.ones((10, 20), bool), CFG)
    assert fm.at(np.array([[20.0, 20.0], [60.0, 20.0]])).tolist() == [4, 0]
