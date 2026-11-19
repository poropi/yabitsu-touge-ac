"""旧橋の路肩からの距離・床と柵の接続、および現橋Cの袖の張り出しを測る。"""
import numpy as np
import pytest
from yabitsu import config, kn5, small_bridges, terrain, track
from yabitsu.crosssection import GUTTER_WIDTH, Section
from yabitsu.valley_fit import Surfaces


@pytest.fixture(scope="module")
def bridges():
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    spans = small_bridges.spans(sec, track.load_bounds(), small_bridges.load())
    # 構造物の接地を確かめる実データ。更新後も控えを読む。
    base = config.PROJECT_ROOT / "export/compare/route70/work/before_34/track_terrain_near.npz"
    if not base.exists():
        pytest.skip("依頼34前の控えを使用")
    with np.load(base) as d:
        near = dict(d)
    ground = terrain.terrain_mesh_z(near, near)
    parts = {name: (v, kn5._triangles(f), mat) for name, v, f, uv, mat, vis in
             small_bridges.support_meshes(sec, spans, ground)}
    return sec, spans, parts, ground


def test_old_bridge_left_separation_and_structure(bridges):
    sec, spans, parts, ground = bridges
    names = {n: (v, f, mat) for n, (v, f, mat) in parts.items() if n.startswith("smallspan_16_old_road_")}
    assert names, "Bの山側、現橋の外に旧橋の床と欄干が必要"
    rails = [(v, f) for n, (v, f, mat) in names.items() if mat == "valley_rusted_steel"]
    assert len(rails) == 2
    floor, triangles, material = names["smallspan_16_old_road_floor"]
    assert material == "gravel_grey"
    slab = Surfaces(floor[triangles])
    for (v, f), low, high in zip(rails, (5.4, 10.2), (7.0, 13.6)):
        frame = terrain.road_frame(sec, *v[:, :2].T)
        assert np.min(frame["lat"]) >= low and np.max(frame["lat"]) <= high
        above = v[:, 2] - slab.height(v[:, :2])
        assert np.isfinite(above).all()
        assert abs(above.min()) < .015 and .60 <= above.max() <= .74
    foot = [v for n, (v, _, _) in names.items() if "_abutment_" in n]
    assert len(foot) == 2
    for v in foot:
        low = v[v[:, 2] == v[:, 2].min()]
        assert (ground(*low[:, :2].T) - low[:, 2]).min() >= .12
    i, j = spans[16]
    center = np.array([sec.x[(i+j)//2], sec.y[(i+j)//2]])
    assert not np.isfinite(slab.height(center[None])).any()


def test_c_end_right_crown_follows_the_gutter_without_a_shelf(bridges):
    sec, spans, parts, _ = bridges
    name = "smallspan_25_approach_end_right"
    v, _, _ = parts[name]
    frame = terrain.road_frame(sec, *v[:, :2].T)
    top = v[:, 2] > sec.edge_z(-1)[frame["idx"]] - .18
    overhang = -frame["lat"][top] - sec.hw_right[frame["idx"][top]] - GUTTER_WIDTH
    assert overhang.max() <= .06


def test_old_road_approaches_reach_both_sides_of_existing_terrain(bridges):
    sec, spans, parts, ground = bridges
    for end in ("start", "end"):
        v, _, _ = parts["smallspan_16_old_road_bank_" + end]
        frame = terrain.road_frame(sec, *v[:, :2].T)
        for shoulder in (frame["lat"] < 4.9, frame["lat"] > 14.0):
            assert shoulder.sum()
            assert (v[shoulder, 2] - ground(*v[shoulder, :2].T)).max() < 0


def test_old_slab_has_a_bearing_on_each_bank(bridges):
    import shapely
    _, _, parts, _ = bridges
    deck, tri, _ = parts["smallspan_16_old_road_floor"]
    footprint = shapely.union_all(shapely.polygons(deck[tri, :2]))
    for end in ("start", "end"):
        pier, _, _ = parts["smallspan_16_old_road_abutment_" + end]
        bearing = shapely.convex_hull(shapely.multipoints(pier[:, :2])).intersection(footprint)
        assert bearing.area >= .90  # 約5.5mの肩幅に16cm以上の桁掛かり


def test_c_wing_foot_and_gutter_are_separate(bridges):
    from yabitsu import valley_fit
    sec, spans, parts, ground = bridges
    v, f, _ = parts["smallspan_25_approach_end_right"]
    wall = v[f]
    feet = valley_fit._edge_points(valley_fit._lower_edges(wall))
    assert (feet[:, 2] - ground(*feet[:, :2].T)).max() < 0
    path = config.PROJECT_ROOT / "export/compare/route70/work/before_34/track/meshes/1ROAD_s1_gutterR_134.npz"
    with np.load(path) as src:
        all_gutter = src["verts"][kn5._triangles(src["faces"])]
    nearby = np.linalg.norm(all_gutter.mean(axis=1)[:, :2] - v[0, :2], axis=1) < 12
    assert not valley_fit.overlap_checks([("gutter", all_gutter[nearby]), ("wing", wall)])
