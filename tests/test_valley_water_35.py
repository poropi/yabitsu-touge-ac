"""控えの水・川原と掘削後の三角形を照合する。"""
import json
from collections import Counter

import numpy as np
import pytest
import shapely

from scripts import track_06_terrain
from yabitsu import config, features, gsi, kn5, nagoki, side_roads, start_lot, terrain, track, tunnel, valleys, water
from yabitsu.crosssection import Section
from yabitsu.valley_fit import Surfaces


SAVED = config.PROJECT_ROOT / "export/compare/route70/work/before_34"
PROCESSED = config.DATA_PROCESSED
pytestmark = pytest.mark.skipif(not (SAVED / "track_terrain_near.npz").is_file(),
                               reason="依頼34前の近景と水面NPZを使用する")


@pytest.fixture(scope="module")
def river_inputs():
    with np.load(SAVED / "track_terrain_near.npz") as src:
        before = dict(src)
    after = {**before, "Z": before["Z"].copy()}
    sec = Section.load_json(PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    # 依頼35の実際の入力 data/manual/valleys.json（依頼38着手前）。
    rows = valleys.load(config.PROJECT_ROOT / "tests/fixtures/valleys_round1.json")
    sampler = gsi.default_sampler(gsi.TileCache(offline=True))
    pads = json.loads((PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    all_pads = pads + start_lot.flats_for(config.DATA_MANUAL / "s1_start_lot.json", pads)
    feats = features.load_features_rows(track.load_section_rows("features", bounds))
    _, terrain_features = nagoki.track_stubs(sec, all_pads, feats, sampler=sampler)
    keep = tunnel.carve_mask(sec.s, tunnel.from_osm(side_roads.load_elements(), sec))
    track_06_terrain.update_valley_ground(sec, after, rows, bounds, sampler,
                                          track_06_terrain.terrain_features(terrain_features, bounds),
                                          pads=all_pads, keep_rows=keep)
    stubs, _ = nagoki.track_stubs(sec, pads, feats, sampler=sampler)
    road = water.road_area(sec, start_lot.road_formation(sec, 0., float(sec.s[-1])), stubs)
    streams, _, _ = water.load(sec, sampler)
    ground = terrain.terrain_mesh_z(after, before)
    levels = water.valley_surface_levels(sec, bounds, rows, streams, ground)
    return streams, road, levels, ground, after, valleys.changed_ground(before, after)


def baseline(name):
    with np.load(SAVED / "track" / "meshes" / f"{name}.npz") as data:
        return data["verts"], data["faces"], data["uvs"]


@pytest.fixture(scope="module")
def new_meshes(river_inputs):
    streams, road, levels, ground, near, dug = river_inputs
    normals = {}
    parts = []
    keep_out = road.buffer(.5)
    for index, stream in enumerate(streams):
        if stream.osm_id not in (water.A_CREEK, 90334744):
            continue
        old_water, old_bed = ([baseline(f"{n}_{index:03d}_00")] for n in ("water", "riverbed"))
        parts.extend(water._scoped_stream(index, stream, old_water, old_bed, levels.get(stream.osm_id),
                                         levels, ground, near, dug.polygon, normals, keep_out, dug.old.height))
    return {n: (v, f, uv) for n, v, f, uv, _, _ in parts}, normals


def triangles(mesh):
    v, f, _ = mesh
    return v[kn5._triangles(f)]


def signatures(mesh, normals):
    v, f, uv = mesh
    t = kn5._triangles(f)
    normals = kn5._normals(v @ kn5.C.T, t) if normals is None else normals
    attributes = np.c_[v, uv.astype(np.float32), normals.astype(np.float32)][t]
    return Counter(min(np.roll(tri, n, axis=0).tobytes() for n in range(3)) for tri in attributes)


def test_a_creek_keeps_water_and_protects_dry_triangles(river_inputs, new_meshes):
    _, _, _, ground, _, dug = river_inputs
    changed_area = dug.polygon
    mesh, normals = new_meshes
    for old, new in zip(baseline("water_022_00"), mesh["water_022_00"]):
        np.testing.assert_array_equal(old, new)

    name = "riverbed_022_00"
    v, f, uv = baseline(name)
    tris = kn5._triangles(f)
    cut = shapely.area(shapely.intersection(changed_area, shapely.polygons(v[tris, :2]))) > 1e-9
    assert cut.sum() == 36
    old_normals = kn5._normals(v @ kn5.C.T, tris)
    assert signatures((v, tris[~cut], uv), old_normals) <= signatures(mesh[name], normals.get(name))

    old = v[tris]
    new = triangles(mesh[name])
    patch = new[shapely.covers(changed_area.buffer(1e-7), shapely.polygons(new[..., :2]))]
    boundary = changed_area.boundary.intersection(water._wet_footprint(old))
    lines = [line for line in shapely.get_parts(boundary) if line.geom_type == "LineString"]
    xy = np.concatenate([shapely.get_coordinates(shapely.line_interpolate_point(
        line, np.linspace(0., line.length, max(2, int(line.length / .05))))) for line in lines])
    old_edge, new_edge = Surfaces(old).height(xy), Surfaces(patch).height(xy)
    both = np.isfinite(old_edge) & np.isfinite(new_edge)
    assert both.sum() > 250
    assert np.max(np.abs(old_edge[both] - new_edge[both])) <= 1e-6
    p = np.concatenate([patch.reshape(-1, 3), patch.mean(axis=1)])
    earlier_gap = Surfaces(old).height(p) - dug.old.height(p[:, :2])
    later_gap = p[:, 2] - ground(*p[:, :2].T)
    assert np.max(later_gap - earlier_gap) <= 1e-6

    samples = v[tris[cut]].mean(axis=1)
    inside = shapely.contains_xy(changed_area.buffer(-.25), *samples[:, :2].T)
    samples = samples[inside]
    actual = Surfaces(triangles(mesh[name])).height(samples)
    assert len(samples)
    assert np.max(actual - ground(*samples[:, :2].T)) <= .10


def test_b_river_bank_meets_the_drawn_ground(river_inputs, new_meshes):
    _, _, _, ground, _, dug = river_inputs
    changed_area = dug.polygon
    mesh, normals = new_meshes
    for name in ("water_012_00", "riverbed_012_00"):
        v, f, uv = baseline(name)
        tris = kn5._triangles(f)
        outside = shapely.area(shapely.intersection(changed_area, shapely.polygons(v[tris, :2]))) <= 1e-9
        old_normals = kn5._normals(v @ kn5.C.T, tris)
        assert signatures((v, tris[outside], uv), old_normals) <= signatures(mesh[name], normals.get(name))
    retained_bed = triangles(mesh["riverbed_012_00"])
    retained_areas = shapely.area(shapely.intersection(changed_area, shapely.polygons(retained_bed[..., :2])))
    assert np.max(retained_areas) <= 1e-9
    new_bed = np.concatenate([triangles(part) for name, part in mesh.items() if name.startswith("riverbed_012_valley_")])
    assert np.min(shapely.area(shapely.polygons(new_bed[..., :2]))) > 1e-9
    existing = triangles(mesh["water_012_00"])
    retained_water = shapely.area(shapely.intersection(changed_area, shapely.polygons(existing[..., :2])))
    assert np.max(retained_water) <= 1e-9
    fresh = np.concatenate([triangles(part) for name, part in mesh.items() if name.startswith("water_012_valley_")])
    bank = water.shore_samples(fresh, existing)
    assert len(bank) > 1000
    assert np.max(np.abs(bank[:, 2] - ground(*bank[:, :2].T))) <= .05
    edge = water.shore_samples(fresh)
    previous = Surfaces(existing).height(edge)
    joins = np.isfinite(previous)
    assert joins.any()
    assert np.max(np.abs(edge[joins, 2] - previous[joins])) <= .01


def test_scope_runs_on_real_water_parts(river_inputs, new_meshes, tmp_path):
    from scripts.check_valley_scope import Scope, mesh_data
    from yabitsu.manifest import Manifest

    _, _, _, _, after, _ = river_inputs
    with np.load(SAVED / "track_terrain_near.npz") as source:
        before = dict(source)
    sec = Section.load_json(PROCESSED / "track_section.json")
    scope = Scope(sec, track.load_bounds(), before, after)
    root = Manifest(tmp_path)
    meshes, normals = new_meshes
    for name, (v, f, uv) in meshes.items():
        root.add_mesh("yabitsu_s4", name, v, f, uv, "water", normals=normals.get(name))
        updated = mesh_data(root.root, root.meshes[name])
        path = f"meshes/{name}.npz"
        if (SAVED / "track" / path).is_file():
            saved = mesh_data(SAVED / "track", {"file": path})
            scope.geometry("track", "yabitsu_s4", name, saved, updated)
        else:
            scope.triangles("track", "yabitsu_s4", name, updated, "added")
    assert not scope.issues

    # 実物の範囲外の水面を1頂点だけ動かすと、同じ判定が失敗を返す。
    v, f, uv = meshes["water_022_00"]
    moved = v.copy()
    far = np.flatnonzero(~scope.dug.covers(moved[:, :2]))[0]
    moved[far, 2] -= .01
    root.add_mesh("yabitsu_s4", "mutated", moved, f, uv, "water")
    scope.geometry("track", "yabitsu_s4", "water_022_00",
                   mesh_data(SAVED / "track", {"file": "meshes/water_022_00.npz"}),
                   mesh_data(root.root, root.meshes["mutated"]))
    assert any(issue["change"] == "position" for issue in scope.issues)
