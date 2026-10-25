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


def test_young_broadleaves_fill_conifer_understory_without_low_road_overhang(world):
    fm, ground, keep = world
    young = {"max_d": 25.0, "spacing": 6.0, "p": 0.55, "size": [0.35, 0.55]}
    cfg = {**CFG, "young_broadleaf": young}
    t = fp.place(None, fm, ground, keep, MODELS, cfg, species=SPECIES)
    small = np.char.startswith(t.pattern, "broadleaf") & (t.scale < 0.6)
    assert small.sum() > 25
    xy = np.c_[t.x, t.y][small]
    assert (fm.at(xy) == 1).all()  # 既存の広葉樹林や空き地には重ねない
    assert keep.course_distance(xy).max() <= young["max_d"]
    assert not keep.blocked(xy).any()
    assert (keep.road_distance(xy) >= MODELS["broadleaf_a"].crown_radius * t.scale[small]).all()
    # 上層の植林を置き換えず、同じ木を残す。
    baseline = fp.place(None, fm, ground, keep, MODELS, CFG, species=SPECIES)
    for pattern in ("sugi_a", "hinoki_a"):
        assert np.array_equal(t.x[t.pattern == pattern], baseline.x[baseline.pattern == pattern])


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


def test_view_clear_drops_tall_trees_below_viewpoint():
    """展望の場所（駐車場）より下の斜面の木で、梢が展望の高さを超えるものは植えない（谷の眺めが開けている）。"""
    t = np.zeros(4, fp.DTYPE)
    t[0] = (0.0, 30.0, 90.0, "sugi_a", 0, 1.0)      # 下の斜面・梢 110 > 100 → 取る
    t[1] = (0.0, 30.0, 70.0, "sugi_a", 0, 1.0)      # 下の斜面・梢 90 < 100 → 残す
    t[2] = (0.0, 30.0, 105.0, "sugi_a", 0, 1.0)     # 上の斜面 → 残す
    t[3] = (0.0, 300.0, 90.0, "sugi_a", 0, 1.0)     # 遠い → 残す
    keep = fp.view_clear_keep(t.view(np.recarray), MODELS, [{"center": [0.0, 0.0], "radius": 100.0, "z": 100.0, "margin": 0.0}])
    assert keep.tolist() == [False, True, True, True]


def test_species_zone_forces_broadleaf(world):
    """樹種を決める場所（species_zones）：中では森の判定に関わらず広葉樹と低木を植える（菜の花台のまわりの雑木）。"""
    fm, ground, keep = world
    zones = [{"center": [-120.0, -100.0], "radius": 40.0, "species": "broadleaf", "spacing": 5.0}]
    t = fp.place(None, fm, ground, keep, MODELS, CFG, seed=1, species=SPECIES, zones=zones)
    d = np.hypot(t.x + 120, t.y + 100)
    inz = d < 35
    assert inz.sum() > 20
    assert all(p.startswith(("broadleaf", "shrub", "fern")) for p in np.asarray(t.pattern)[inz])


def test_species_zone_with_fixed_trees_and_understory_inside_a_polygon(world):
    """島：木は決めた位置（支柱の位置）に、下草（シダ）は多角形の中だけに植える。"""
    from shapely.geometry import box
    fm, ground, keep = world
    island = box(-130.0, -110.0, -110.0, -90.0)
    pts = [[-125.0, -105.0], [-115.0, -95.0]]
    zones = [{"center": [-120.0, -100.0], "radius": 30.0, "species": "broadleaf", "spacing": 5.0, "points": pts,
              "shrub_species": "fern", "shrub_spacing": 1.0, "inside": island}]
    t = fp.place(None, fm, ground, keep, MODELS, CFG, seed=1, species=SPECIES, zones=zones)
    pat = np.asarray(t.pattern); inz = np.hypot(t.x + 120, t.y + 100) < 30
    trees = inz & np.char.startswith(pat, "broadleaf")
    assert sorted(zip(np.round(t.x[trees], 3), np.round(t.y[trees], 3))) == sorted(map(tuple, pts))
    ferns = inz & np.char.startswith(pat, "fern")
    assert ferns.sum() > 100
    assert ((t.x[ferns] >= -130) & (t.x[ferns] <= -110) & (t.y[ferns] >= -110) & (t.y[ferns] <= -90)).all()


def test_later_zones_keep_the_island_trees_and_stay_out_of_it(world):
    """島の若木は、あとに処理する重なった場所に消されない。その場所の木も島の中には植えない。"""
    from shapely.geometry import box
    fm, ground, keep = world
    island = box(-130.0, -110.0, -110.0, -90.0)
    pts = [[-125.0, -105.0], [-115.0, -95.0]]
    zones = [{"center": [-120.0, -100.0], "radius": 15.0, "species": "broadleaf", "spacing": 5.0, "points": pts,
              "shrub_species": "fern", "shrub_spacing": 1.0, "inside": island},
             {"center": [-115.0, -100.0], "radius": 30.0, "species": "broadleaf", "spacing": 3.0}]
    t = fp.place(None, fm, ground, keep, MODELS, CFG, seed=1, species=SPECIES, zones=zones)
    pat = np.asarray(t.pattern)
    inside = (t.x > -130) & (t.x < -110) & (t.y > -110) & (t.y < -90)
    trees = inside & np.char.startswith(pat, "broadleaf")
    assert sorted(zip(np.round(t.x[trees], 3), np.round(t.y[trees], 3))) == sorted(map(tuple, pts))
    assert not (inside & np.char.startswith(pat, "shrub")).any()


def test_zone_cleared_by_game_objects_only_ignores_map_paths(tmp_path):
    """clear: "game"：地図の小道の線（OSM）が通っていても、ゲームの道から離れていれば木を植える（峠の斜面）。"""
    road = _square(-200, -3, 200, 3, 10.0)
    m = _manifest(tmp_path, {"1GRASS_s1_terrain_000": (*_square(-200, -200, 200, 200, 10.0), "ground"),
                             "1ROAD_s1_000": (*road, "asphalt")})
    ground = fg.load_ground(m)
    fm = ForestMap(x0=-200.0, y1=200.0, cell=4.0, cls=np.zeros((100, 100), np.uint8))
    zone = {"center": [0.0, 60.0], "radius": 20.0, "species": "broadleaf", "spacing": 3.0, "shrub_spacing": 50.0}
    counts = []
    for extra in ({}, {"clear": "game", "road_clear": 1.5}):
        keep = fg.load_keepout(m, {"cell": 0.5, "coarse_cell": 2.0})
        keep.add_lines([np.array([[x, 60.0 + 7 * np.sin(x / 3)] for x in np.arange(-25, 25, 1.0)])], [2.0])   # 斜面をうねる小道
        t = fp.place(None, fm, ground, keep, MODELS, CFG, seed=1, species=SPECIES, zones=[{**zone, **extra}])
        counts.append(int((np.char.startswith(np.asarray(t.pattern), "broadleaf") & (np.hypot(t.x, t.y - 60) < 20)).sum()))
    assert counts[1] >= counts[0] + 25   # 小道の線で消えていた木（約 4 割）が残る


def test_fill_bare_plants_only_where_nothing_grows(world):
    """むき出しの地面を埋める：木の無い所にだけ植え、道・open の円には植えない。"""
    fm, ground, keep = world
    t0 = fp.place(None, fm, ground, keep, MODELS, CFG, seed=1, species=SPECIES)
    zone = {"center": [100.0, -100.0], "radius": 30.0, "spacing": 3.0, "road_clear": 2.5, "open": [[100.0, -100.0, 8.0]]}
    add = fp.fill_bare(t0, ground, keep, SPECIES, zone)
    assert len(add) > 50
    assert (np.hypot(add.x - 100, add.y + 100) >= 8.0 - 1.0).all()            # open の中は草地のまま（茂みのずらし分だけ許す）
    assert (np.abs(add.y) > 3.0).all()                                        # 道の上には植えない
    again = fp.fill_bare(np.concatenate([np.asarray(t0), np.asarray(add)]).view(np.recarray), ground, keep, SPECIES, zone)
    assert len(again) < len(add) * 0.1                                        # 埋めた後はほとんど足さない
