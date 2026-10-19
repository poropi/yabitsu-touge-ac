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
