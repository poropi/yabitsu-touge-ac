"""谷のシーンを作る。scripts/valley_site_export.py が Blender 内で呼び出す。"""

import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from valley_site_io import capture
from valley_site_light import lighting, localize_images


def collection(scene, name):
    node = bpy.data.collections.new(name)
    scene.collection.children.link(node)
    return node


def image_file(path):
    image = bpy.data.images.load(path, check_existing=True)
    if not image.packed_file:
        image.pack()
    return image


def material(spec, emission):
    mat = bpy.data.materials.new(spec["name"])
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = .08 if spec["name"] == "water" else .85
    bsdf.inputs["Specular IOR Level"].default_value = .1
    bsdf.inputs["Emission Strength"].default_value = emission
    if spec["texture"]:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = image_file(spec["texture"])
        tex.location = (-400, 130)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        if spec.get("alpha"):
            nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            mat.surface_render_method = "DITHERED"
    elif spec.get("color"):
        bsdf.inputs["Base Color"].default_value = (*spec["color"][:3], 1.)
        bsdf.inputs["Emission Color"].default_value = bsdf.inputs["Base Color"].default_value
    mat.use_backface_culling = not spec.get("two_sided")
    mat["valley_alpha"] = bool(spec.get("alpha"))
    return mat


def integer_attribute(mesh, name, domain, values):
    mesh.attributes.new(name, "INT", domain).data.foreach_set("value", values.astype(np.int32))


def mesh_object(coll, name, vertices, faces, uv, normals):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.uv_layers.new(name="UVMap").uv.foreach_set("vector", uv.astype(np.float32).ravel())
    # Game normals are defined per mesh vertex, and the combined ground has
    # its original independent corner normals where materials or UVs change.
    mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    mesh.normals_split_custom_set(normals.reshape(-1, 3).tolist())
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    coll.objects.link(ob)
    return ob


def vertex_group(ob, name, indices):
    group = ob.vertex_groups.new(name=name)
    if len(indices):
        group.add(indices.tolist(), 1.0, "REPLACE")


def reference_cameras(cameras, photos, scene, views, origin):
    for view in views:
        camera = bpy.data.cameras.new(view["camera"])
        ob = bpy.data.objects.new(view["camera"], camera)
        cameras.objects.link(ob)
        ob.location = Vector(np.asarray(view["position"]) - origin)
        ob.rotation_euler = Vector(view["direction"]).to_track_quat("-Z", "Y").to_euler()
        camera.sensor_fit = "VERTICAL"
        camera.angle_y = math.radians(view["fov"])
        camera.clip_start, camera.clip_end = .05, 3000
        camera.passepartout_alpha = 1
        if view["photo"]:
            image = image_file(view["photo"])
            background = camera.background_images.new()
            background.image = image
            background.display_depth = "FRONT"
            background.alpha = .5
            background.frame_method = "CROP"
            camera.show_background_images = True
            ref = bpy.data.objects.new(f"Photo {Path(view['photo']).stem}", None)
            ref.empty_display_type = "IMAGE"
            ref.data = image
            ref.hide_viewport = True
            ref.hide_render = True
            ref["camera"] = ob.name
            photos.objects.link(ref)
        if scene.camera is None:
            scene.camera = ob


def reference_trees(coll, rows, emission):
    # Saved positions and scales describe CSP items; silhouettes are local
    # guides because the CSP tree renderer isn't part of this scene.
    foliage = material(dict(name="CSP proxy foliage", texture=None, color=[.055, .12, .045], two_sided=True), emission)
    bark = material(dict(name="CSP proxy bark", texture=None, color=[.09, .062, .039], two_sided=True), emission)
    shapes = {}
    sizes = {
        "sugi": (18., 2.2), "hinoki": (16., 2.2), "broadleaf": (10., 3.5),
        "shrub": (1.6, .9), "fern": (.6, .5), "grass": (.4, .3),
    }
    for species, (height, radius) in sizes.items():
        # Build each shape once and share its mesh between all instances.
        stem_r = min(.22, radius * .12)
        stem_h = height * .34
        bpy.ops.mesh.primitive_cone_add(vertices=7, radius1=radius, radius2=radius * (
            .24 if species == "broadleaf" else .035), depth=height * .72,
            location=(0., 0., height * .64))
        crown = bpy.context.object
        crown.data.materials.append(foliage)
        bpy.ops.mesh.primitive_cone_add(vertices=5, radius1=stem_r, radius2=stem_r * .6,
                                        depth=stem_h, location=(0., 0., stem_h / 2))
        stem = bpy.context.object
        stem.data.materials.append(bark)
        for ob in (crown, stem):
            # Primitive vertices are local to their own centre.
            ob.data.transform(ob.matrix_world)
        shapes[species] = (crown.data, stem.data)
        bpy.data.objects.remove(crown, do_unlink=True)
        bpy.data.objects.remove(stem, do_unlink=True)

    for row in rows:
        species = row["pattern"].split("_", 1)[0]
        for shape in shapes.get(species, shapes["broadleaf"]):
            tree = bpy.data.objects.new(f"CSP {row['pattern']}", shape)
            tree.location = row["loc"]
            tree.scale = (row["scale"],) * 3
            tree.rotation_euler.z = math.radians(row["angle"])
            coll.objects.link(tree)


def build(workspace):
    data = json.loads((workspace / "site.json").read_text(encoding="utf-8"))
    render = json.loads((workspace / "render.json").read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.
    scene["valley_site"] = data["site"]
    collections = {name: collection(scene, name) for name in
                   ("EDIT_terrain", "EDIT_parts", "LOCK_context", "REF_trees", "REF_cameras", "REF_photo")}
    for name, coll in collections.items():
        coll.hide_select = name.startswith(("LOCK_", "REF_"))
    emission = data["light"]["eevee_emission"]
    materials = {name: material(spec, emission) for name, spec in render["materials"].items()}
    with np.load(workspace / "render.npz", allow_pickle=False) as geom, np.load(
        workspace / "source.npz", allow_pickle=False
    ) as source:
        ob = mesh_object(collections["EDIT_terrain"], data["terrain_name"],
                         *(geom["terrain_" + key] for key in ("vertices", "faces", "uv", "normals")))
        mesh = ob.data
        ids = {name: k for k, name in enumerate(dict.fromkeys(geom["terrain_materials"]))}
        for name in ids:
            mesh.materials.append(materials[name])
        mesh.polygons.foreach_set("material_index", [ids[name] for name in geom["terrain_materials"]])
        integer_attribute(mesh, "valley_vertex_id", "POINT", source["terrain_vertex_ids"] + 1)
        integer_attribute(mesh, "valley_face_id", "FACE", source["terrain_face_ids"] + 1)
        integer_attribute(mesh, "valley_kind", "FACE", source["terrain_face_kind"])
        integer_attribute(mesh, "valley_source_faces", "FACE", source["terrain_source_faces"])
        for suffix, values in zip(("r", "c", "face"), source["terrain_coarse_remainder"].T):
            integer_attribute(mesh, "valley_coarse_remainder_" + suffix, "FACE", values)
        vertex_group(ob, "LOCK", geom["terrain_lock"])
        vertex_group(ob, "KEEP_Z", geom["terrain_keep_z"])

        for row in render["objects"]:
            key = row["key"]
            ob = mesh_object(collections[row["collection"]], row["name"],
                             *(geom[key + suffix] for suffix in ("_v", "_f", "_uv", "_n")))
            ob.data.materials.append(materials[row["material"]])
            if row["collection"] == "EDIT_parts":
                integer_attribute(ob.data, "valley_vertex_id", "POINT", geom[key + "_ids"])
                integer_attribute(ob.data, "valley_face_id", "FACE", geom[key + "_face_ids"])
                vertex_group(ob, "LOCK", geom[key + "_lock"])

    reference_trees(collections["REF_trees"], render["trees"], data["light"]["eevee_tree_emission"])
    reference_cameras(collections["REF_cameras"], collections["REF_photo"], scene, data["views"], data["origin"])
    lighting(scene, collections["LOCK_context"], data["light"])
    localize_images()
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x, scene.render.resolution_y = 768, 480
    scene.render.image_settings.file_format = "PNG"
    # Start in the first photo camera. Camera backgrounds remain viewport-only.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                view = area.spaces.active
                view.region_3d.view_perspective = "CAMERA"
                view.clip_start = .05
                view.clip_end = 3000.
    bpy.context.preferences.filepaths.save_version = 0
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(workspace / "site.blend"), relative_remap=False)
    capture(scene, workspace / "baseline_scene.npz")


if __name__ == "__main__":
    build(Path(sys.argv[sys.argv.index("--") + 1]).resolve())
