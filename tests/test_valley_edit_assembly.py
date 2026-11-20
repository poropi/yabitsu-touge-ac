import json

import numpy as np
import pytest
from PIL import Image

from scripts.track_build_manifest import card_adjustments
from yabitsu import terrain, valleys, valley_fit, valley_surface
from yabitsu.manifest import Manifest


def base_surface():
    grid = {"xs": np.arange(4.) * 2, "ys": np.arange(4.) * 2,
            "Z": np.ones((4, 4)), "D": np.zeros((4, 4)), "H": np.zeros((3, 3), bool)}
    vertices = np.array([[2, 2, 1], [4, 2, 1], [4, 4, 1], [2, 4, 1], [3, 3, .9]])
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    near = {**grid, "valley_cells": np.array([[1, 1]]), "valley_vertices": vertices,
            "valley_faces": faces, "valley_bed": np.zeros(len(faces), bool),
            "valley_gravel": np.zeros(len(faces), bool),
            "valley_coarse_remainder": np.full((4, 3), -1),
            "valley_source_faces": np.arange(4)}
    return grid, near


def edit_directory(root, near, *, changed=True, stale=False, part_names=()):
    from yabitsu import valley_edits

    path = root / "A-s4-1724"
    path.mkdir(parents=True)
    faces = near["valley_faces"]
    vertices = near["valley_vertices"].copy()
    vertices[-1, 2] -= .5
    material = "valley_edit_A-s4-1724_trial"
    materials = np.full(len(faces), "valley_forest_floor", dtype="<U40")
    materials[0] = material
    uv = vertices[faces, :2] / 5
    uv[0] = [[0, 0], [1, 0], [0, 1]]
    np.savez(path / "terrain.npz", vertices=vertices, faces=faces[::-1], uv=uv[::-1],
             materials=materials[::-1], face_kind=np.full(len(faces), "bank"),
             vertex_ids=np.arange(len(vertices)), original_face_ids=np.arange(len(faces))[::-1],
             source_faces=near["valley_source_faces"][::-1],
             coarse_remainder=near["valley_coarse_remainder"][::-1])
    (path / "edit.json").write_text(json.dumps({
        "fingerprint": {"terrain": "stale" if stale else valley_edits.surface_digest(near)},
        "terrain_faces": list(range(len(faces))), "terrain_changed": changed, "changed_parts": list(part_names),
    }))
    textures = path / "textures"
    textures.mkdir()
    Image.new("RGB", (2, 2), "brown").save(textures / f"{material}.png")
    return path, material, uv[0]


def test_missing_edit_keeps_all_arrays_and_meshes_identical(tmp_path):
    from yabitsu import valley_edits

    _, near = base_surface()
    expected = tuple(valley_surface.mesh_parts(near))
    arrays = {k: v.copy() for k, v in near.items()}
    assert not valley_edits.apply_terrain(near, valley_edits.load(tmp_path))
    assert near.keys() == arrays.keys()
    assert all(np.array_equal(v, arrays[k]) for k, v in near.items())
    for old, new in zip(expected, valley_surface.mesh_parts(near), strict=True):
        for a, b in zip(old, new, strict=True):
            np.testing.assert_array_equal(a, b)


def test_edit_lowers_ground_card_forest_rail_and_collision_and_uses_baked_uv(tmp_path):
    from scripts.align_valley_trees import plan_forest_changes
    from yabitsu import dress_mountain, features, kn5, small_bridges, valley_edits
    from yabitsu.crosssection import Section

    base, near = base_surface()
    original_z = terrain.terrain_mesh_z(near, base)(3., 3.)
    original_ground = terrain.terrain_mesh_z(near, base)
    chain = np.array([[2., 3., 1.1], [3., 3., 1.1], [4., 3., 1.1]])
    original_foot = valleys.fit_foot_chain(chain, valleys.changed_ground(base, near), .1)
    _, material, uv = edit_directory(tmp_path / "edits", near)
    edits = valley_edits.load(tmp_path / "edits")
    assert valley_edits.apply_terrain(near, edits)
    assert terrain.terrain_z(near, base)(3., 3.) == pytest.approx(original_z - .5)
    assert terrain.terrain_mesh_z(near, base)(3., 3.) == pytest.approx(original_z - .5)
    dug = valleys.changed_ground(base, near)
    no_water = valley_fit.Surfaces(np.empty((0, 3, 3)))
    root = np.array([[3., 3., original_z]])
    heights, remove = card_adjustments(root, dug, [], (no_water, no_water))
    assert heights[0] == pytest.approx(original_z - .5)
    assert not remove[0]
    trees = np.rec.fromarrays(root.T, names="x,y,z")
    plan = plan_forest_changes(trees, dug, [], (no_water, no_water))
    assert len(plan) == 1 and plan[0].new_ground == pytest.approx(original_z - .5)
    foot = valleys.fit_foot_chain(chain, dug, .1)
    np.testing.assert_allclose(foot, original_foot - .5)
    np.testing.assert_array_equal(near["valley_source_faces"], np.arange(4))
    s = np.linspace(1., 5., 41)
    zero, one = np.zeros_like(s), np.ones_like(s)
    sec = Section(s, s, zero.copy(), 2 * one, zero, zero, one.astype(int), zero.astype(bool),
                  one, one, zero, zero)
    rail = features.Feature(1., 5., "R", "guardrail", .7, "")
    _, _, offset, _ = dress_mountain.guardrail_line(sec, rail)
    sec.y[:] = 3. + offset[20] + .46
    old_support = list(small_bridges.rail_support_meshes(sec, rail, (20, 21), original_ground))[0]
    new_ground = terrain.terrain_mesh_z(near, base)
    new_support = list(small_bridges.rail_support_meshes(sec, rail, (20, 21), new_ground))[0]
    assert new_support[1][:, 2].min() == pytest.approx(old_support[1][:, 2].min() - .5)

    manifest = Manifest(tmp_path / "manifest")
    manifest.add_material("valley_forest_floor", color=[40, 30, 20])
    valley_edits.register_materials(manifest, edits)
    for n, verts, faces, uvs, mat, normals in valley_surface.mesh_parts(near):
        manifest.add_mesh("test", n, verts, faces, uvs, mat, normals=normals)
    manifest.save()
    name = next(n for n, row in manifest.meshes.items() if row["material"] == material)
    assert name.startswith("1GRASS_")
    with np.load(manifest.root / manifest.meshes[name]["file"]) as exported:
        np.testing.assert_array_equal(exported["uvs"], uv)
        assert exported["verts"][:, 2].min() == pytest.approx(original_z - .5)
    kn5_path = tmp_path / "trial.kn5"
    kn5.write_from_manifest(manifest.root / "manifest.json", "test", kn5_path)
    assert f"{material}.dds" in kn5.read_kn5(kn5_path)["textures"]


def test_stale_geometry_rejected_before_changing_arrays(tmp_path):
    from yabitsu import valley_edits

    _, near = base_surface()
    edit_directory(tmp_path, near, stale=True)
    old = near["valley_vertices"].copy()
    with pytest.raises(ValueError, match="指紋"):
        valley_edits.apply_terrain(near, valley_edits.load(tmp_path))
    np.testing.assert_array_equal(near["valley_vertices"], old)


def test_part_replacement_keeps_name_group_order_and_visibility(tmp_path):
    from yabitsu import valley_edits

    _, near = base_surface()
    name = "1WALL_track_smallspan_02_abutment"
    path, material, _ = edit_directory(tmp_path / "edit", near, changed=False, part_names=[name])
    parts = path / "parts"
    parts.mkdir()
    v = near["valley_vertices"][:3]
    f = np.array([[0, 1, 2]])
    uv = v[:, :2] / 5
    norm = np.tile([0., 1., 0.], (3, 1))
    np.savez(parts / f"{name}.npz", vertices=v + [0, 0, -.2], faces=f, uv=uv,
             normals=norm, material=material)
    manifest = Manifest(tmp_path / "track")
    manifest.add_material("ground", color=[20, 20, 20])
    manifest.add_mesh("s4", name, v, f, uv, "ground", False)
    manifest.add_mesh("s4", "other", v, f, uv, "ground")
    sites = tmp_path / "sites"
    record_source(sites, name, manifest, path)
    edits = valley_edits.load(tmp_path / "edit")
    valley_edits.register_materials(manifest, edits)
    valley_edits.replace_parts(manifest, edits, sites_root=sites)
    assert list(manifest.meshes) == [name, "other"]
    assert manifest.groups["s4"]["meshes"] == [name, "other"]
    assert manifest.meshes[name]["renderable"] is False
    assert manifest.meshes[name]["material"] == material
    assert manifest.materials["ground"]["color"] == [20, 20, 20]
    with np.load(manifest.root / manifest.meshes[name]["file"]) as exported:
        np.testing.assert_array_equal(exported["verts"], v + [0, 0, -.2])


def test_parts_only_edit_checks_the_rebuilt_skeleton(tmp_path):
    from yabitsu import valley_edits

    _, near = base_surface()
    edit_directory(tmp_path, near, changed=False, stale=True, part_names=("smallspan_02_deck",))
    with pytest.raises(ValueError, match="指紋"):
        valley_edits.apply_terrain(near, valley_edits.load(tmp_path))


def record_source(sites, name, manifest, path):
    from yabitsu import valley_edits

    target = sites / path.name
    target.mkdir(parents=True)
    entry = manifest.meshes[name]
    archive = manifest.root / entry["file"]
    with np.load(archive) as mesh:
        np.savez(target / "source.npz", part_0_vertices=mesh["verts"], part_0_faces=mesh["faces"],
                 part_0_uv=mesh["uvs"], part_0_normals=mesh.get("normals", np.empty((0, 3))))
    spec_path = path / "edit.json"
    spec = json.loads(spec_path.read_text())
    spec["fingerprint"]["parts"] = {name: valley_edits.file_digest(archive)}
    spec["fingerprint"]["part_materials"] = {name: entry["material"]}
    (target / "site.json").write_text(json.dumps({
        "site": path.name,
        "fingerprint": {k: v for k, v in spec["fingerprint"].items() if k != "part_materials"},
        "parts": [{"name": name, "key": "part_0", "material": entry["material"]}],
    }))
    spec["source_sha256"] = valley_edits.file_digest(target / "source.npz")
    spec_path.write_text(json.dumps(spec))


@pytest.mark.parametrize("changed_array", ["verts", "faces", "uvs", "normals"])
def test_part_source_checks_fresh_arrays_without_old_manifest(tmp_path, changed_array):
    from yabitsu import valley_edits

    _, near = base_surface()
    name = "smallspan_02_deck"
    path, material, _ = edit_directory(tmp_path / "edits", near, changed=False, part_names=[name])
    baseline = Manifest(tmp_path / "track")
    v = near["valley_vertices"][:3]
    f = np.array([[0, 1, 2]])
    normals = np.repeat([[0., 1., 0.]], 3, axis=0)
    baseline.add_mesh("s4", name, v, f, v[:, :2], "ground", normals=normals)
    sites = tmp_path / "sites"
    record_source(sites, name, baseline, path)
    edits = valley_edits.load(tmp_path / "edits")
    replacements = {name: {name: (v + [0, 0, -.2], f, v[:, :2], material, None)}}
    source = baseline.root / baseline.meshes[name]["file"]
    with np.load(source) as packed:
        arrays = dict(packed)
    np.savez(source, **arrays)
    valley_edits.check_part_sources(baseline, edits, replacements, sites_root=sites)
    arrays[changed_array] = arrays[changed_array].copy()
    arrays[changed_array].flat[0] += 1
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match=f"{name}.*{changed_array}"):
        valley_edits.check_part_sources(baseline, edits, replacements, sites_root=sites)


def test_water_bake_splits_diffuse_material_but_keeps_both_meshes_for_trees(tmp_path):
    from yabitsu import valley_edits

    _, near = base_surface()
    name = "water_022_valley_00"
    path, material, _ = edit_directory(tmp_path / "edits", near, changed=False, part_names=[name])
    parts = path / "parts"
    parts.mkdir()
    v, f = near["valley_vertices"][:4], np.array([[0, 1, 2], [0, 2, 3]])
    uv = v[:, :2]
    np.savez(parts / f"{name}.npz", vertices=v, faces=f, uv=uv, normals=np.empty((0, 3)),
             material="water", materials=["water", material])
    manifest = Manifest(tmp_path / "track")
    manifest.add_material("water")
    manifest.add_mesh("s4", name, v, f, uv, "water")
    sites = tmp_path / "sites"
    record_source(sites, name, manifest, path)
    edits = valley_edits.load(tmp_path / "edits")
    replacements = valley_edits.part_edits(edits)
    valley_edits.register_materials(manifest, edits)
    valley_edits.replace_parts(manifest, edits, replacements=replacements, sites_root=sites)
    manifest.save()
    assert manifest.groups["s4"]["meshes"] == [name, name + "_edit_trial"]
    assert manifest.meshes[name]["material"] == "water"
    assert manifest.meshes[name + "_edit_trial"]["material"] == material
    np.testing.assert_array_equal(valley_fit.river_surface(manifest.root).height(np.array([[3.8, 3.], [2.2, 3.]])),
                                  [1, 1])
    final = valley_edits.replace_water([(name, v, f, uv, "water", True)], replacements)
    assert [row[0] for row in final] == [name, name + "_edit_trial"]


def test_assembly_refuses_in_progress_import(tmp_path):
    from yabitsu import valley_edits

    (tmp_path / ".B-s4-6354.importing").mkdir()
    with pytest.raises(ValueError, match="書き戻し"):
        valley_edits.load(tmp_path)


def test_bridge_clearance_reads_the_material_fragments_with_their_parent():
    from yabitsu.crosssection import Section

    s = np.array([0., 1., 2.])
    zero, one = np.zeros_like(s), np.ones_like(s)
    sec = Section(s, s, zero, 3 * one, zero, zero, one.astype(int), zero.astype(bool),
                  one, one, zero, zero)
    v = np.array([[0., -2., 2.], [0., 2., 2.], [2., 2., 2.], [2., -2., 2.]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    name = "smallspan_02_deck"
    source = valley_fit.Mesh(name, v, f)
    primary = valley_fit.Mesh(name, v, f[:1])
    baked = valley_fit.Mesh(name + "_edit_trial", v, f[1:], parent=name)
    ground = lambda x, y: np.zeros_like(x)
    a = valley_fit.bridge_checks([source], sec, {2: (0, 2)}, ground)
    b = valley_fit.bridge_checks([primary, baked], sec, {2: (0, 2)}, ground)
    assert not any(x.category == "deck_missing" for x in a)
    assert a == b


def test_part_edit_requires_exported_source_archive(tmp_path):
    from yabitsu import valley_edits

    _, near = base_surface()
    name = "smallspan_02_deck"
    path, _, _ = edit_directory(tmp_path / "edits", near, changed=False, part_names=[name])
    manifest = Manifest(tmp_path / "track")
    manifest.add_mesh("s4", name, near["valley_vertices"][:3], [[0, 1, 2]], np.zeros((3, 2)), "ground")
    sites = tmp_path / "sites"
    record_source(sites, name, manifest, path)
    (sites / path.name / "source.npz").unlink()
    with pytest.raises(ValueError, match="source.npz"):
        valley_edits.check_part_sources(manifest, valley_edits.load(tmp_path / "edits"),
                                       {name: {}}, sites_root=sites)


def test_terrain_material_split_keeps_the_procedural_group(tmp_path):
    from yabitsu import valley_edits

    _, near = base_surface()
    edit_directory(tmp_path / "edits", near)
    edits = valley_edits.load(tmp_path / "edits")
    valley_edits.apply_terrain(near, edits)
    manifest = Manifest(tmp_path / "track")
    for name, verts, faces, uv, material, normals in valley_surface.mesh_parts(near):
        manifest.add_mesh("s4", name, verts, faces, uv, material, normals=normals)
    name = "1GRASS_valley_A_000_bank"
    with np.load(manifest.root / manifest.meshes[name]["file"]) as mesh:
        assert mesh["verts"][:, 1].mean() > 3.05
    manifest.split_group("s4", "s5", lambda xy: xy[1] > 3.05,
                         centers=valley_edits.terrain_centers(near))
    assert name in manifest.groups["s4"]["meshes"]
