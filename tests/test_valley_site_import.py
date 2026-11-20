import os

import numpy as np
import pytest


@pytest.fixture
def terrain():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 1, 0]], float)
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    source = dict(
        vertices=vertices + [4000.00000021, 5000, 300],
        faces=faces,
        uv=vertices[faces, :2].astype(np.float32),
        materials=np.array(["ground"] * 4),
        face_kind=np.array(["bank"] * 4),
        vertex_ids=np.arange(11, 16),
        original_face_ids=np.arange(21, 25),
        source_faces=np.arange(31, 35),
        coarse_remainder=np.full((4, 3), -1),
    )
    baseline = dict(
        vertices=vertices.astype(np.float32), faces=faces.copy(),
        uv=source["uv"].copy(), materials=source["materials"].copy(),
        kind=np.zeros(4, int), vertex_ids=np.arange(11, 16), face_ids=np.arange(21, 25),
        source_faces=source["source_faces"].copy(), coarse_remainder=source["coarse_remainder"].copy(),
    )
    site = dict(origin=[4000, 5000, 300], lock=[11, 12, 13, 14], keep_z={}, radius=60)
    return source, baseline, {key: value.copy() for key, value in baseline.items()}, site


def test_no_edit_preserves_original_precision(terrain):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    result = validate_terrain(source, baseline, current, site)
    assert result.keys() == source.keys()
    for key in source:
        np.testing.assert_array_equal(result[key], source[key], err_msg=key)


def test_millimeter_snap_and_interior_height_edit(terrain):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    current["vertices"][0, 2] += 0.0009
    current["vertices"][4, 2] -= 0.3
    result = validate_terrain(source, baseline, current, site)
    np.testing.assert_array_equal(result["vertices"][:4], source["vertices"][:4])
    assert result["vertices"][4, 2] == pytest.approx(299.7)
    assert result["vertices"][4, 0] == source["vertices"][4, 0]


def test_face_reorder_preserves_source_and_coarse_flags(terrain):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    source["coarse_remainder"][0] = [1, 2, 3]
    site["lock"].append(15)
    for key in ("faces", "uv", "materials", "kind", "face_ids", "source_faces", "coarse_remainder"):
        current[key] = current[key][::-1]
    result = validate_terrain(source, baseline, current, site)
    np.testing.assert_array_equal(result["source_faces"], [34, 33, 32, 31])
    np.testing.assert_array_equal(result["coarse_remainder"][-1], [1, 2, 3])


@pytest.mark.parametrize(("change", "error"), [
    ("lock", "LOCK"), ("keep", "KEEP_Z"), ("flip", "winding"), ("area", "area"),
    ("hole", "boundary"), ("duplicate", "Duplicate"), ("scope", "scope"),
    ("uv", "UV"),
])
def test_rejects_invalid_terrain(terrain, change, error):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    if change == "lock":
        current["vertices"][0, 2] += 0.0011
    elif change == "keep":
        site["keep_z"] = {"15": "shoulder"}
        current["vertices"][4, 2] += 0.0051
    elif change == "flip":
        current["faces"][0] = current["faces"][0, ::-1]
    elif change == "area":
        current["vertices"][4] = [1, 0, 0]
    elif change == "hole":
        for key in ("faces", "uv", "materials", "kind", "face_ids", "source_faces", "coarse_remainder"):
            current[key] = current[key][1:]
    elif change == "duplicate":
        current["faces"][1] = current["faces"][0]
    elif change == "scope":
        current["vertices"][4] = [2.1, 1, 0]
    elif change == "uv":
        current.pop("uv")
    with pytest.raises(ValueError, match=error):
        validate_terrain(source, baseline, current, site)


def test_requires_registered_bake_and_preserves_shared_material(tmp_path):
    from PIL import Image
    from yabitsu.valley_site_import import validate_materials

    baseline = {"ground": {"signature": "base"}}
    assert validate_materials({"ground"}, baseline, baseline, "B", tmp_path) == {}
    with pytest.raises(ValueError, match="ground"):
        validate_materials({"ground"}, baseline, {"ground": {"signature": "edited"}}, "B", tmp_path)
    with pytest.raises(ValueError, match="other"):
        validate_materials({"other"}, baseline, {"other": {"texture": "other.png"}}, "B", tmp_path)
    key = "valley_edit_B_bed"
    with pytest.raises(ValueError, match=key):
        validate_materials({key}, baseline, {key: {"texture": "bed.png"}}, "B", tmp_path)
    Image.new("RGB", (32, 32)).save(tmp_path / "bed.png")
    assert validate_materials({key}, baseline, {key: {"texture": "bed.png"}}, "B", tmp_path) == {
        key: tmp_path / "bed.png",
    }


@pytest.fixture
def blend_site(tmp_path, terrain):
    import json
    from pathlib import Path
    import subprocess

    from PIL import Image
    from yabitsu import config

    if not config.BLENDER_EXE.is_file():
        pytest.skip("Blender is not installed")
    root = Path(__file__).resolve().parents[1]
    folder = tmp_path / "site"
    folder.mkdir()
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    Image.new("RGB", (8, 8), (120, 90, 60)).save(folder / "ground.png")
    source, _, _, site = terrain
    face_ids = source.pop("original_face_ids")
    np.savez(folder / "source.npz", **{"terrain_" + key: value for key, value in source.items()},
             terrain_face_ids=face_ids)
    meta = dict(site, site="B-s4-6354", parts=[], fingerprint={"terrain": "source"})
    (folder / "site.json").write_text(json.dumps(meta), encoding="utf-8")
    script = folder / "fixture.py"
    script.write_text(f"""
import bpy
import sys
sys.path.insert(0, {str(root / "blender_scripts")!r})
from valley_site_io import capture
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
collections = {{name: bpy.data.collections.new(name) for name in ('EDIT_terrain', 'LOCK_context', 'REF_cameras')}}
for collection in collections.values():
    scene.collection.children.link(collection)
mesh = bpy.data.meshes.new('terrain')
mesh.from_pydata([(0,0,0),(2,0,0),(2,2,0),(0,2,0),(1,1,0)], [], [[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
material = bpy.data.materials.new('ground')
material.use_nodes = True
image = bpy.data.images.load({str(folder / "ground.png")!r})
image.pack()
node = material.node_tree.nodes.new('ShaderNodeTexImage')
node.image = image
mesh.materials.append(material)
uv = mesh.uv_layers.new()
for loop in mesh.loops:
    uv.data[loop.index].uv = mesh.vertices[loop.vertex_index].co[:2]
for name, domain, values in [('valley_vertex_id','POINT', range(12,17)), ('valley_face_id','FACE',range(22,26)), ('valley_kind','FACE',[0]*4)]:
    mesh.attributes.new(name, 'INT', domain).data.foreach_set('value', list(values))
obj = bpy.data.objects.new('valley_terrain', mesh)
collections['EDIT_terrain'].objects.link(obj)
ctx = bpy.data.objects.new('locked', mesh.copy())
collections['LOCK_context'].objects.link(ctx)
capture(scene, {str(folder / "baseline_scene.npz")!r})
bpy.ops.wm.save_as_mainfile(filepath={str(folder / "site.blend")!r})
image.filepath = '//ground.png'
capture(scene, {str(folder / "baseline_scene.npz")!r})
bpy.ops.wm.save_as_mainfile(filepath={str(folder / "site.blend")!r})
bpy.ops.wm.save_as_mainfile(filepath={str(relocated / "site.blend")!r})
ctx.location.z += 1
bpy.ops.wm.save_as_mainfile(filepath={str(folder / "invalid.blend")!r})
ctx.location.z -= 1
layer = bpy.context.view_layer.layer_collection.children['REF_cameras']
layer.exclude = True
bpy.ops.wm.save_as_mainfile(filepath={str(folder / "excluded.blend")!r})
layer.exclude = False
edited = material.copy()
edited.name = 'valley_edit_B-s4-6354_ground'
nodes = edited.node_tree.nodes
links = edited.node_tree.links
image_node = next(node for node in nodes if node.type == 'TEX_IMAGE')
principled = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
gamma = nodes.new('ShaderNodeGamma')
links.new(image_node.outputs['Color'], gamma.inputs['Color'])
links.new(gamma.outputs['Color'], principled.inputs['Base Color'])
mesh.materials.append(edited)
mesh.polygons[0].material_index = 1
bpy.ops.wm.save_as_mainfile(filepath={str(folder / "unbaked.blend")!r})
""", encoding="utf-8")
    subprocess.run([str(config.BLENDER_EXE), "-b", "--python-exit-code", "1", "--python", str(script)],
                   check=True, capture_output=True, text=True)
    return folder, source, relocated


@pytest.mark.skipif(os.environ.get("YABITSU_RUN_BLENDER_TESTS") != "1",
                    reason="Blender を使うため YABITSU_RUN_BLENDER_TESTS=1 が必要")
def test_headless_import_preserves_arrays_and_rejects_changed_context(blend_site, tmp_path):
    from hashlib import sha256
    import json
    import subprocess
    import sys

    folder, source, relocated = blend_site
    edits = tmp_path / "edits"
    command = [sys.executable, "-m", "scripts.valley_site_import", "B-s4-6354",
               "--site-dir", str(folder), "--edits-root", str(edits)]
    good = subprocess.run(command, capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr
    result = edits / "B-s4-6354"
    before = (result / "edit.json").read_bytes()
    assert json.loads(before)["terrain_changed"] is False
    with np.load(result / "terrain.npz") as data:
        for key in source:
            np.testing.assert_array_equal(data[key], source[key], err_msg=key)
    moved = subprocess.run(command + ["--blend", str(relocated / "site.blend")],
                           capture_output=True, text=True)
    assert moved.returncode == 0, moved.stdout + moved.stderr
    before = (result / "edit.json").read_bytes()
    assert json.loads(before)["blend_sha256"] == sha256((relocated / "site.blend").read_bytes()).hexdigest()
    bad = subprocess.run(command + ["--blend", str(folder / "invalid.blend")],
                         capture_output=True, text=True)
    assert bad.returncode != 0
    assert "LOCK_context" in bad.stderr
    assert (result / "edit.json").read_bytes() == before
    excluded = subprocess.run(command + ["--blend", str(folder / "excluded.blend")],
                              capture_output=True, text=True)
    assert excluded.returncode != 0
    assert "REF_cameras" in excluded.stderr
    unbaked = subprocess.run(command + ["--blend", str(folder / "unbaked.blend")],
                             capture_output=True, text=True)
    assert unbaked.returncode != 0
    assert "baked PNG" in unbaked.stderr


def part_surfaces(vertices, faces):
    source = dict(vertices=np.asarray(vertices, float), faces=np.asarray(faces), uv=np.zeros((len(vertices), 2)),
                  normals=np.empty((0, 3)), material=np.asarray("ground"), vertex_ids=np.arange(len(vertices)))
    current = dict(vertices=source["vertices"].copy(), faces=source["faces"].copy(),
                   uv=np.zeros((len(faces), 3, 2)), normals=np.tile([0, 0, 1.0], (len(faces), 3, 1)),
                   materials=np.full(len(faces), "ground"), vertex_ids=source["vertex_ids"].copy(),
                   face_ids=np.arange(len(faces)))
    return source, {key: a.copy() for key, a in current.items()}, current


def test_part_with_preexisting_unused_vertices_can_be_edited():
    from yabitsu.valley_site_import import _validate_part

    source, baseline, current = part_surfaces(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 0]], [[0, 1, 2]],
    )
    current["vertices"][2, 2] += .1
    result = _validate_part(source, baseline, current, {"origin": [0, 0, 0], "radius": 60}, {})
    assert result["vertices"][2, 2] == pytest.approx(.1)
    assert np.array_equal(result["vertices"][3], [2, 2, 0])


def test_part_corner_split_is_rejected_as_a_hole():
    from yabitsu.valley_site_import import _validate_part

    source, baseline, current = part_surfaces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2], [3, 4, 5]],
    )
    current["vertices"][3, 2] += .1
    with pytest.raises(ValueError, match="boundary"):
        _validate_part(source, baseline, current, {"origin": [0, 0, 0], "radius": 60}, {})


def test_part_can_bake_inside_while_retaining_protected_face():
    from yabitsu.valley_site_import import _validate_part

    source, baseline, current = part_surfaces(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 2], [0, 2, 3]],
    )
    current["materials"] = np.array(["ground", "valley_edit_B-s4-6354_bed"])
    part = {"lock": [0, 1, 2], "protected_faces": [0]}
    site = {"site": "B-s4-6354", "origin": [0, 0, 0], "radius": 60}
    result = _validate_part(source, baseline, current, site, part)
    np.testing.assert_array_equal(result["materials"], ["ground", "valley_edit_B-s4-6354_bed"])
    current["materials"] = result["materials"][::-1]
    with pytest.raises(ValueError, match="Protected part faces changed"):
        _validate_part(source, baseline, current, site, part)


def test_rebuilt_closed_part_rejects_inward_faces():
    from yabitsu.valley_site_import import _validate_part

    source, baseline, current = part_surfaces(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    )
    current["face_ids"][:] = -1
    current["faces"] = current["faces"][:, ::-1]
    with pytest.raises(ValueError, match="winding"):
        _validate_part(source, baseline, current, {"origin": [0, 0, 0], "radius": 60}, {})


def test_coarse_remainder_normal_is_fixed(terrain):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    source["coarse_remainder"][0] = [1, 2, 3]
    site["lock"].append(15)
    normal = np.tile([0, 0, 1.0], (4, 3, 1))
    baseline["normals"] = normal.copy()
    source["normals"] = normal.copy()
    current["normals"] = normal.copy()
    current["normals"][0, 0] *= -1
    with pytest.raises(ValueError, match="coarse remainder"):
        validate_terrain(source, baseline, current, site)


def test_coarse_remainder_accepts_a_cyclic_corner_order(terrain):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    source["coarse_remainder"][0] = [1, 2, 3]
    site["lock"].append(15)
    current["faces"][0] = np.roll(current["faces"][0], -1, axis=0)
    current["uv"][0] = np.roll(current["uv"][0], -1, axis=0)
    result = validate_terrain(source, baseline, current, site)
    np.testing.assert_array_equal(result["faces"], source["faces"])
    np.testing.assert_array_equal(result["uv"], source["uv"])
    np.testing.assert_array_equal(result["coarse_remainder"], source["coarse_remainder"])


@pytest.mark.parametrize("material", ["retex:mt_terrain", "water", "concrete", "valley_edit_C_test"])
def test_terrain_rejects_other_materials_on_a_face(terrain, material):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    current["materials"] = current["materials"].astype("<U40")
    current["materials"][2] = material
    with pytest.raises(ValueError, match=f"face 23.*{material}"):
        validate_terrain(source, baseline, current, {**site, "site": "B"})


def test_part_rejects_material_of_another_part():
    from yabitsu.valley_site_import import _validate_part

    source, baseline, current = part_surfaces(
        np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0.]]), np.array([[0, 1, 2]])
    )
    current["materials"] = np.array(["water"])
    with pytest.raises(ValueError, match="face 0.*water"):
        _validate_part(source, baseline, current,
                       dict(site="B", origin=[0, 0, 0], radius=5), dict(name="wall"))


@pytest.mark.parametrize("horizontal", [(0.0052, 0), (0.004, 0.004)])
def test_keep_z_limits_horizontal_distance(terrain, horizontal):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    site["keep_z"] = {"15": "shoulder"}
    current["vertices"][4, :2] += horizontal
    with pytest.raises(ValueError, match="KEEP_Z"):
        validate_terrain(source, baseline, current, site)


def test_keep_z_allows_small_horizontal_move_and_restores_height(terrain):
    from yabitsu.valley_site_import import validate_terrain

    source, baseline, current, site = terrain
    site["keep_z"] = {"15": "shoulder"}
    current["vertices"][4] += [0.004, 0, 0.004]
    result = validate_terrain(source, baseline, current, site)
    expected = source["vertices"][4] + [0.004, 0, 0]
    np.testing.assert_allclose(result["vertices"][4], expected, atol=1e-7, rtol=0)
