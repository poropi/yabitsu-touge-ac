from types import SimpleNamespace

import numpy as np
import shapely
from scipy.spatial import cKDTree

from scripts.check_valley_scope import roadside_forest_changes
from yabitsu import roadside_cover
from yabitsu.forest_place import DTYPE


def route():
    s = np.arange(41.)
    return roadside_cover.Road(dict(s=s, x=s, y=s*0, z=s*0, hw_left=s*0+3, hw_right=s*0+2,
                                    bridge=np.zeros(len(s), bool), heading=s*0))


def test_pavement_width_and_station():
    road = route()
    p = np.array([[8.25, 7.], [8.25, -7.], [8., 1.]])
    pos = road.locate(p)
    np.testing.assert_allclose(pos.s, [8.25, 8.25, 8.])
    np.testing.assert_allclose(road.edge_distance(p), [4., 5., 0.])
    np.testing.assert_allclose(pos.towards, [[0., -7.], [0., 7.], [0., -1.]])


def test_cover_on_steep_open_ground_respects_masks_and_manual_stems():
    road = route()
    p = np.array([[x, y] for x in range(3, 38) for y in range(5, 15)], float)
    exclusion = shapely.box(13.5, 4., 19.5, 16.)
    ground = SimpleNamespace(z=lambda xy: .4 * xy[:, 1])
    fm = SimpleNamespace(at=lambda xy: np.zeros(len(xy), dtype=int))
    spec = {
        "seed": 56, "min_edge_m": 1.5, "max_edge_m": 12, "min_slope_deg": 15,
        "cover_radius_m": 1.5, "jitter_m": .24, "manual_clear_m": 1., "shrub_spacing_m": 2.5,
        "young_spacing_m": 5.5, "tall_min_edge_m": 3., "overhang_clear_m": 4.5,
    }
    manual = np.array([[9., 6.], [9., 7.]])
    table = np.empty(0, dtype=DTYPE).view(np.recarray)
    paths = {"fern": ["fern_a"], "grass": ["grass_a"], "shrub": ["shrub_a"], "sapling": ["sapling_a"]}
    models = {kind[0]: SimpleNamespace(height=.5, crown_radius=.6, crown_base=0.) for kind in paths.values()}
    parts = dict(road=road, fm=fm, ground=ground, exclusion=exclusion, models=models,
                 species=paths, manual_xy=manual, cfg=spec, place_cfg={}, sight={})
    first = roadside_cover.plant(p, table, **parts)
    second = roadside_cover.plant(p, table, **parts)
    np.testing.assert_array_equal(first, second)
    xy = np.c_[first.x, first.y]
    assert not shapely.intersects_xy(exclusion, *xy.T).any()
    assert (cKDTree(manual).query(xy)[0] >= 1).all()
    assert (road.edge_distance(xy) >= 1.5).all()
    cover = np.char.startswith(first.pattern, "fern") | np.char.startswith(first.pattern, "grass")
    wanted = p[~shapely.intersects_xy(exclusion, *p.T)]
    assert cKDTree(xy[cover]).query(wanted)[0].max() <= 1.5


def roadside_patterns(rows):
    from scripts.roadside_forest import pilot_patterns

    return pilot_patterns(rows, route().locate(np.c_[rows.x, rows.y]).towards, rotate_by_name=True)


def test_scope_preserves_non_model_bytes_and_detects_a_move(tmp_path):
    from scripts.align_s5_8960_trees import _line

    old, new = tmp_path / "old", tmp_path / "new"
    base = np.array([(8., 7., 2., "sugi_a", 120., 1.),
                     (25., 8., 2., "sugi_b", 45., 1.)], dtype=DTYPE).view(np.recarray)
    changed = base.astype([(n, "U32" if n == "pattern" else base.dtype[n]) for n in base.dtype.names]).view(np.recarray)
    changed.pattern[0] = "mt_sugi_edge_a_r07"
    assert roadside_patterns(base).pattern[0] == changed.pattern[0]
    for directory, rows in ((old, base), (new, changed)):
        d = directory / "extension/trees"
        d.mkdir(parents=True)
        np.save(directory / "trees.npy", rows)
        (d / "forest_0_0.txt").write_bytes(b"configure: test\r\n" + b"\r\n".join(map(_line, rows)) + b"\r\n")
    assert roadside_forest_changes(old, new, road=route(), span=(0, 20)) == []
    p = new / "extension/trees/forest_0_0.txt"
    p.write_bytes(p.read_bytes().replace(b"angle = 45", b"angle = 46"))
    assert roadside_forest_changes(old, new, road=route(), span=(0, 20))
    p.write_bytes(p.read_bytes().replace(b"angle = 46", b"angle = 45"))
    changed.y[0] += 1
    np.save(new / "trees.npy", changed)
    assert roadside_forest_changes(old, new, road=route(), span=(0, 20))


def test_overhang_height_is_measured_from_the_highest_road_surface():
    s = np.arange(50.)
    road = roadside_cover.Road(dict(s=s, x=s, y=s*0, z=s*0, hw_left=s*0+3, hw_right=s*0+2,
                                    bridge=np.zeros(len(s), bool), heading=s*0, slope=s*0+.08, crown=s*0))
    trees = np.array([(8., 4.5, 0., "sugi_a", 0., 1.),
                      (12., 4.5, .4, "sugi_a", 0., 1.)], dtype=DTYPE).view(np.recarray)
    models = {"sugi_a": SimpleNamespace(height=10., crown_radius=2., crown_base=.45)}
    result = roadside_cover.safe_trees(trees, road, models, {"overhang_clear_m": 4.5}, {})
    np.testing.assert_array_equal(result, [False, True])


def test_low_plant_stays_clear_after_game_rounding():
    road = route()
    shrubs = np.array([(8., 4.51, 0., "shrub_a", 0., .54),
                       (12., 10., 0., "mt_shrub_a", 0., .50)], dtype=DTYPE).view(np.recarray)
    models = {p: SimpleNamespace(crown_radius=3.) for p in shrubs.pattern}
    roadside_cover.limit_ground_crown(shrubs, road, models)
    np.testing.assert_allclose(shrubs.scale, [.49, .50])


def test_steep_garden_colour_is_not_a_settlement_footprint():
    road = route()
    xy = np.array([[8., 6.], [20., 6.], [28., 6.]])
    fm = SimpleNamespace(at=lambda xy: np.full(len(xy), 4))
    ground = SimpleNamespace(z=lambda xy: np.where(xy[:,0] < 24, .4 * xy[:,1], 0))
    exclusion = shapely.box(17, 2, 23, 12)
    cfg = {"min_edge_m": 1.5, "max_edge_m": 12, "min_slope_deg": 15}
    eligible, _, _ = roadside_cover.eligible(xy, road, fm, ground, exclusion, cfg)
    np.testing.assert_array_equal(eligible, [True, False, False])


def test_scope_rejects_additions_on_pavement_even_at_allowed_station(tmp_path):
    from scripts.align_s5_8960_trees import _line

    old, new = tmp_path / "old", tmp_path / "new"
    rows = np.array([(12., 7., 0., "sugi_a", 0., 1.), (8., 1., 0., "fern_a", 0., 1.)],
                    dtype=DTYPE).view(np.recarray)
    for folder, table in ((old, rows[:1]), (new, rows)):
        d = folder / "extension/trees"
        d.mkdir(parents=True)
        np.save(folder / "trees.npy", table)
        (d / "forest_0_0.txt").write_bytes(b"\n".join(map(_line, table)) + b"\n")
    assert roadside_forest_changes(old, new, road=route(), span=(0, 20))


def test_pilot_retains_the_manually_planted_models():
    from scripts.roadside_forest import pilot_patterns

    old = np.array([(12., 7., 0., "broadleaf_a", 30., 1.), (16., 7., 0., "sugi_b", 90., 1.)],
                   dtype=DTYPE).view(np.recarray)
    xy = np.c_[old.x, old.y]
    facing = route().locate(xy).towards
    new = pilot_patterns(old, facing, rotate_by_name=True, manual_xy=np.array([[12.005, 7.]]))
    assert new.pattern[0] == "broadleaf_a"
    assert new.pattern[1] == "mt_sugi_edge_b_r08"
