from types import SimpleNamespace

import numpy as np
import pytest

from yabitsu import forest_ground as fg
from yabitsu import forest_place as fp
from yabitsu.forestmap import ForestMap
from tests.test_forest_ground import _manifest, _square

MODELS = {
    "sugi_a": SimpleNamespace(height=20.0, crown_base=0.5, crown_radius=2.5),
    "hinoki_a": SimpleNamespace(height=16.0, crown_base=0.4, crown_radius=3.0),
    "broadleaf_a": SimpleNamespace(height=14.0, crown_base=0.2, crown_radius=6.0),
    "shrub_a": SimpleNamespace(height=1.8, crown_base=0.0, crown_radius=1.2),
    "fern_a": SimpleNamespace(height=0.4, crown_base=0.0, crown_radius=0.9),
}
SPECIES = {"sugi": ["sugi_a"], "hinoki": ["hinoki_a"], "broadleaf": ["broadleaf_a"], "shrub": ["shrub_a"], "fern": ["fern_a"]}
CFG = {**fp.DEFAULT_PLACE,
       "tiers": [{"max_d": 40, "spacing": {"conifer": 4.0, "broadleaf": 6.0}, "size": [0.9, 1.1]},
                 {"max_d": 1e9, "spacing": {"conifer": 12.0, "broadleaf": 14.0}, "size": [1.0, 1.3]}],
       "blend_m": 5.0}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """400 m 四方の平らな地面、y=−3〜3 の道路。北半分は針葉樹、南は x<−150 だけ広葉樹、それ以外は森なし。"""
    tmp = tmp_path_factory.mktemp("w")
    road = _square(-200, -3, 200, 3, 10.0)
    m = _manifest(tmp, {"1GRASS_s1_terrain_000": (*_square(-200, -200, 200, 200, 10.0), "ground"),
                        "1ROAD_s1_000": (*road, "asphalt")})
    ground = fg.load_ground(m)
    keep = fg.load_keepout(m, {"cell": 0.5, "coarse_cell": 2.0})
    cls = np.zeros((100, 100), np.uint8)
    cls[:50] = 1                       # 北半分（行が北から）
    cls[50:, :12] = 2                  # 南西の一角は広葉樹
    fm = ForestMap(x0=-200.0, y1=200.0, cell=4.0, cls=cls)
    return fm, ground, keep


def _place(world, seed=1, area=None):
    fm, ground, keep = world
    return fp.place(area, fm, ground, keep, MODELS, CFG, seed=seed, species=SPECIES)


def test_none_in_keepout(world):
    t = _place(world)
    assert len(t) > 100
    assert not world[2].blocked(np.c_[t.x, t.y]).any()
    assert np.isfinite(t.z).all()


def test_z_on_ground(world):
    t = _place(world)
    assert np.abs(t.z - world[1].z(np.c_[t.x, t.y])).max() <= 0.3


def test_no_low_crown_over_road(world):
    t = _place(world)
    trees = np.array([not p.startswith(("shrub", "fern")) for p in t.pattern])
    d = world[2].road_distance(np.c_[t.x, t.y])
    for i in np.flatnonzero(trees):
        m = MODELS[t.pattern[i]]
        if d[i] < m.crown_radius * t.scale[i]:
            assert t.z[i] + m.crown_base * m.height * t.scale[i] >= 10.0 + 5.0


def test_density_by_tier(world):
    t = _place(world)
    con = np.array([p.startswith(("sugi", "hinoki")) for p in t.pattern])
    near = con & (t.y > 10) & (t.y < 35)
    far = con & (t.y > 100) & (t.y < 125)
    assert near.sum() > 3 * far.sum() > 0


def test_deterministic(world):
    a, b = _place(world, seed=3), _place(world, seed=3)
    assert np.array_equal(a.x, b.x) and list(a.pattern) == list(b.pattern)


def test_open_class_empty(world):
    t = _place(world)
    assert not ((t.y < -10) & (t.x > -140)).any()


def test_understory_only_near_road(world):
    t = _place(world)
    under = np.array([p.startswith(("shrub", "fern")) for p in t.pattern])
    assert under.any()
    assert world[2].road_distance(np.c_[t.x, t.y])[under].max() <= CFG["understory"]["max_d"] + 1


def test_area_limits(world):
    from shapely.geometry import box
    t = _place(world, area=box(-50, 0, 50, 100))
    assert len(t) and (t.x >= -50).all() and (t.x <= 50).all() and (t.y >= 0).all()


def test_tiers_follow_course_not_osm(world):
    fm, ground, keep = world
    keep.add_lines([np.array([[-200.0, 150.0], [200.0, 150.0]])], [4.0])    # 北の森の中の OSM の道
    t = fp.place(None, fm, ground, keep, MODELS, CFG, seed=1, species=SPECIES)
    con = np.array([p.startswith(("sugi", "hinoki")) for p in t.pattern])
    near_osm = con & (t.y > 155) & (t.y < 180)
    near_course = con & (t.y > 10) & (t.y < 35)
    assert near_course.sum() > 3 * near_osm.sum()


def _one(x, y, z, pattern, scale=1.0):
    t = np.zeros(1, fp.DTYPE)
    t[0] = (x, y, z, pattern, 0.0, scale)
    return t


def test_sightline_clears_overhang_but_keeps_inner_forest():
    # 半径 40 m の左カーブ（中心 (0, 40)）を走る。樹冠が目の高さにある杉を、道からの距離を変えて置く
    a = np.linspace(-np.pi / 2, 0.2, 120)
    cl = np.c_[40 * np.cos(a), 40 + 40 * np.sin(a)]
    z = np.zeros(len(cl))
    at = lambda r, zz: _one(r * np.cos(-0.7), 40 + r * np.sin(-0.7), zz, "sugi_a")
    t = np.concatenate([at(36.0, -12.0),    # 内側の路肩のすぐ外：樹冠が道の上にはみ出す → 取る
                        at(43.0, -12.0),    # 外側（谷側）から樹冠が道にはみ出す → 取る
                        at(26.0, -12.0),    # 内側の森の奥 → 残す
                        at(36.0, 30.0)]     # 樹冠が視線より上 → 残す
                       ).view(np.recarray)
    keep = fp.sightline_keep(t, MODELS, cl, z, fp.DEFAULT_SIGHT)
    assert keep.tolist() == [False, False, True, True]


def test_sightline_keeps_understory():
    cl = np.c_[np.linspace(0, 100, 50), np.zeros(50)]
    t = _one(50.0, 4.0, 0.0, "fern_a").view(np.recarray)
    assert fp.sightline_keep(t, MODELS, cl, np.zeros(50), fp.DEFAULT_SIGHT).tolist() == [True]


def test_overhang_uses_road_surface_not_ground_under_it(tmp_path):
    """道路の真下の地形は削ってある。張り出しは路面（1ROAD）の高さで判定すること。"""
    from tests.test_forest_ground import _manifest, _square
    road = _square(-100, -3, 100, 3, 6.0)                    # 盛り土の上の道路（路面 6 m）
    m = _manifest(tmp_path, {"1GRASS_s1_terrain_000": (*_square(-100, -100, 100, 100, 0.0), "ground"),
                             "1ROAD_s1_000": (*road, "asphalt")})
    ground, keep = fg.load_ground(m), fg.load_keepout(m, {"cell": 0.5, "coarse_cell": 2.0})
    surface = fg.load_ground(m, prefixes=("1ROAD_",))
    fm = ForestMap(x0=-100.0, y1=100.0, cell=4.0, cls=np.ones((50, 50), np.uint8))
    cfg = {**CFG, "tiers": [{"max_d": 1e9, "spacing": {"conifer": 3.0, "broadleaf": 3.0}, "size": [1.0, 1.0]}],
           "understory": {**CFG["understory"], "max_d": 0.0}}
    models = {**MODELS, "sugi_a": SimpleNamespace(height=20.0, crown_base=0.5, crown_radius=4.0)}
    t = fp.place(None, fm, ground, keep, models, cfg, seed=1, species={**SPECIES, "hinoki": ["sugi_a"]},
                 road_surface=surface)
    d = keep.road_distance(np.c_[t.x, t.y])
    near = d < 4.0
    assert not (near & (t.z + 10.0 < 6.0 + 5.0)).any()
