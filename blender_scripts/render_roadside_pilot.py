"""Render identical views with the current forest and the roadside pilot forest."""

from configparser import ConfigParser
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import bpy
from mathutils import Vector
import numpy as np
import OpenImageIO as oiio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

from valley_site_export import material
from valley_site_light import lighting
from yabitsu.csptree import read_mesh


AOV = "RoadsideGround"
GRASS_AOV = "GrassFXProxy"
OLD_TREES = ROOT / "export/compare/route70/work/supervisor_41.fmEVx6/deliver/forest/extension/trees"
MOUNTAIN = ROOT / "assets/landmarks/trees_mountain/trees_mountain.blend"


def ground_aov(objects, grass_materials=()):
    """Record world X/Y and ground visibility using the same depth test as color."""
    aov = bpy.context.view_layer.aovs.add()
    aov.name, aov.type = AOV, "COLOR"
    if grass_materials:
        grass_aov = bpy.context.view_layer.aovs.add()
        grass_aov.name, grass_aov.type = GRASS_AOV, "VALUE"
    tagged = {}
    for obj in objects:
        for slot in obj.material_slots:
            original = slot.material
            if original not in tagged:
                clone = original.copy()
                tree = clone.node_tree
                world = tree.nodes.new("ShaderNodeNewGeometry")
                coords = tree.nodes.new("ShaderNodeSeparateXYZ")
                channels = tree.nodes.new("ShaderNodeCombineColor")
                output = tree.nodes.new("ShaderNodeOutputAOV")
                output.aov_name = AOV
                tree.links.new(world.outputs["Position"], coords.inputs[0])
                tree.links.new(coords.outputs["X"], channels.inputs[0])
                tree.links.new(coords.outputs["Y"], channels.inputs[1])
                channels.inputs[2].default_value = 1
                tree.links.new(channels.outputs[0], output.inputs["Color"])
                if original in grass_materials:
                    grass_output = tree.nodes.new("ShaderNodeOutputAOV")
                    grass_output.aov_name = GRASS_AOV
                    grass_output.inputs["Value"].default_value = 1.
                tagged[original] = clone
            slot.material = tagged[original]


def render_with_ground_aov(scene, prefix):
    settings = scene.render.image_settings
    settings.media_type = "MULTI_LAYER_IMAGE"
    settings.color_depth = "32"
    exr = prefix.with_suffix(".exr")
    scene.render.filepath = str(exr)
    bpy.ops.render.render(write_still=True)
    source = oiio.ImageInput.open(str(exr))
    ground_found = False
    try:
        index = 0
        while source.seek_subimage(index, 0):
            names = source.spec().channelnames
            rgb = [next((i for i, name in enumerate(names) if name.lower().endswith(f"{AOV}.{c}".lower())), None)
                   for c in "RGB"]
            if all(i is not None for i in rgb):
                lo, hi = min(rgb), max(rgb)
                pixels = source.read_image(index, 0, lo, hi + 1, oiio.FLOAT)
                np.save(prefix.with_suffix(".npy"), pixels[..., [i - lo for i in rgb]])
                ground_found = True
            for i, name in enumerate(names):
                if GRASS_AOV.lower() in name.lower().split("."):
                    pixels = source.read_image(index, 0, i, i + 1, oiio.FLOAT)
                    np.save(prefix.with_name(prefix.name + "_grass.npy"), pixels[..., 0])
                    break
            index += 1
        if not ground_found:
            raise ValueError(f"地形パスが見つかりません: {exr}")
    finally:
        source.close()
    exr.unlink()
    settings.media_type = "IMAGE"
    settings.file_format, settings.color_mode, settings.color_depth = "PNG", "RGB", "8"
    bpy.data.images["Render Result"].save_render(str(prefix.with_suffix(".png")), scene=scene)


def current_library(patterns, texture_dir, emission, local_packages):
    """Recreate CSP mesh silhouettes in Blender from the current tree packages."""
    materials = {}
    library = {}
    for pattern in sorted(patterns):
        filename = f"{pattern}.zip"
        path = local_packages / filename if local_packages and (local_packages / filename).is_file() else OLD_TREES / filename
        with ZipFile(path) as archive:
            settings = ConfigParser()
            settings.read_string(archive.read("tree.ini").decode())
            height = settings["BASIC"].getfloat("HEIGHT")
            half_width = settings["BASIC"].getfloat("WIDTH") / 2
            for lod in range(3):
                collection = bpy.data.collections.new(f"Current {pattern} LOD{lod}")
                library[pattern, lod] = collection
                textures, pieces = read_mesh(archive.read(settings[f"LOD_{lod}"]["MODEL"]))
                for number, piece in enumerate(pieces):
                    texture = textures[piece.diffuse][1]
                    if texture not in materials:
                        image_path = texture_dir / texture
                        image_path.write_bytes(archive.read(texture))
                        mat = material(dict(name=f"Current {texture}", texture=str(image_path), two_sided=True,
                                            alpha="_leaf" in texture), emission)
                        if "_leaf" in texture:
                            tree = mat.node_tree
                            shader = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
                            alpha = shader.inputs["Alpha"].links[0].from_socket
                            clip = tree.nodes.new("ShaderNodeMath")
                            clip.operation, clip.inputs[1].default_value = "GREATER_THAN", .4
                            tree.links.new(alpha, clip.inputs[0])
                            tree.links.new(clip.outputs[0], shader.inputs["Alpha"])
                        materials[texture] = mat
                    name = f"{collection.name} part {number}"
                    mesh = bpy.data.meshes.new(name)
                    xyz = piece.verts["p"][:, [0, 2, 1]] * (half_width, -half_width, height)
                    mesh.from_pydata(xyz, [], piece.indices.reshape(-1, 3))
                    mesh.materials.append(materials[texture])
                    uv = piece.verts["uv"].copy()
                    uv[:, 1] = 1 - uv[:, 1]
                    loops = np.empty(len(mesh.loops), dtype=np.int32)
                    mesh.loops.foreach_get("vertex_index", loops)
                    mesh.uv_layers.new().uv.foreach_set("vector", uv[loops].flatten())
                    normals = piece.verts["n"][:, [0, 2, 1]] * (1, -1, 1)
                    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
                    mesh.normals_split_custom_set_from_vertices(normals)
                    mesh.update()
                    collection.objects.link(bpy.data.objects.new(name, mesh))
    return library


def prototype_library(patterns, emission):
    model_names = {f"{name[3:-1]}{name[-1].upper()}_LOD{lod}": (name, lod)
                   for name in patterns for lod in range(2)}
    with bpy.data.libraries.load(str(MOUNTAIN), link=False) as (available, chosen):
        if not model_names.keys() <= set(available.collections):
            raise ValueError(f"試作のコレクションがありません: {model_names.keys() - set(available.collections)}")
        chosen.collections = list(model_names)
    library = {}
    materials = set()
    for collection in chosen.collections:
        library[model_names[collection.name]] = collection
        collection.hide_render = collection.hide_viewport = False
        for obj in collection.all_objects:
            obj.hide_render = obj.hide_viewport = False
            materials.update(mat for mat in obj.data.materials if mat)
    for mat in materials:
        tree = mat.node_tree
        for bsdf in (node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"):
            base = bsdf.inputs["Base Color"]
            if base.is_linked:
                tree.links.new(base.links[0].from_socket, bsdf.inputs["Emission Color"])
            else:
                bsdf.inputs["Emission Color"].default_value = base.default_value
            bsdf.inputs["Emission Strength"].default_value = emission
    return library


def place_trees(scene, rows, poses):
    trees = bpy.data.collections.new("Roadside forest")
    scene.collection.children.link(trees)
    for index, (row, pose) in enumerate(zip(rows, poses)):
        instance = bpy.data.objects.new(f"Tree {index:05d}", None)
        instance.instance_type = "COLLECTION"
        location = tuple(float(row[k]) for k in ("x", "y", "z"))
        instance.location = location
        instance.rotation_euler.z = math.radians(pose["angle"])
        instance.scale = (float(row["scale"]),) * 3
        trees.objects.link(instance)
    return list(trees.objects)


def road_height(point, roads):
    hits = []
    for ob in roads:
        origin = ob.matrix_world.inverted() @ Vector((point[0], point[1], point[2] + 8))
        hit, pos, _, _ = ob.ray_cast(origin, (0, 0, -1))
        if hit:
            hits.append((float((ob.matrix_world @ pos).z), ob.name))
    if not hits:
        raise ValueError(f"路面を特定できません {point}")
    return max(hits)


def grass_preview(ground, settings, retex):
    """対象面だけに解析用の緑とAOVを適用する。路面や河床には触れない。"""
    chosen = set(settings["meshes"])
    images = {name: bpy.data.images.load(path) for name, path in retex.items()}
    painted = set()
    clones = {}
    for obj in ground:
        tagged_faces = set(settings["valley_faces"]) if obj.name == "valley_terrain" else set()
        marked = obj.name in chosen
        for slot, original in enumerate(list(obj.data.materials)):
            key = re.match(r"retex:(mt_[a-z]+)(?:\.\d+)?$", original.name)
            color = images.get(key[1]) if key else None
            for grass in (False, True):
                indexes = [p for p in obj.data.polygons if p.material_index == slot and
                           (marked or p.index in tagged_faces) == grass]
                if not indexes or (not grass and color is None):
                    continue
                ident = original, grass
                if ident not in clones:
                    mat = original.copy()
                    clones[ident] = mat
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    tex = next(node for node in nodes if node.type == "TEX_IMAGE")
                    if color is not None:
                        tex.image = color
                    if grass:
                        coord = nodes.new("ShaderNodeNewGeometry")
                        noise = nodes.new("ShaderNodeTexNoise")
                        noise.inputs["Scale"].default_value = 3.5
                        ramp = nodes.new("ShaderNodeValToRGB")
                        ramp.color_ramp.elements[0].color = (.013, .035, .009, 1.)
                        ramp.color_ramp.elements[1].color = (.038, .105, .018, 1.)
                        mix = nodes.new("ShaderNodeMixRGB")
                        mix.inputs[0].default_value = .66
                        links.new(coord.outputs["Position"], noise.inputs["Vector"])
                        links.new(noise.outputs["Fac"], ramp.inputs[0])
                        links.new(tex.outputs["Color"], mix.inputs[1])
                        links.new(ramp.outputs[0], mix.inputs[2])
                        bsdf = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
                        links.new(mix.outputs[0], bsdf.inputs["Base Color"])
                        links.new(mix.outputs[0], bsdf.inputs["Emission Color"])
                        painted.add(mat)
                obj.data.materials.append(clones[ident])
                idx = len(obj.data.materials) - 1
                for p in indexes:
                    p.material_index = idx
    if not painted:
        raise ValueError("シーンのGrassFX対象面を確認できません")
    return painted


def render(site, out, mode):
    setup = json.loads((site / "setup.json").read_text())
    with np.load(site / f"{mode}.npz", allow_pickle=False) as forest:
        rows, poses = forest["rows"], forest["poses"]
    scene = bpy.context.scene
    context = bpy.data.collections["LOCK_context"]
    editable = bpy.data.collections["EDIT_parts"]
    lighting(scene, context, setup["lights"])
    bpy.data.collections["REF_trees"].hide_render = True
    ground = list(bpy.data.collections["EDIT_terrain"].objects)
    prefixes = ("1GRASS_valley_", "1GRASS_s1_terrain_", "mtsplit_s1_terrain_", "riverbed_")
    ground.extend(obj for obj in context.objects if obj.name.startswith(prefixes))
    grass_materials = grass_preview(ground, setup["grass"], setup["retex"]) if "grass" in setup else ()
    ground_aov(ground, grass_materials)
    roads = [obj for obj in context.objects if re.fullmatch(r"1ROAD_s1_\d+", obj.name)]
    roads.extend(obj for obj in editable.objects if obj.name.endswith("_deck"))

    camera = bpy.data.cameras.new("Roadside comparison")
    cam = bpy.data.objects.new(camera.name, camera)
    scene.collection.objects.link(cam)
    scene.camera = cam
    camera.sensor_fit = "HORIZONTAL"
    camera.clip_start, camera.clip_end = .035, 1500
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x, scene.render.resolution_y = setup["size"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.film_transparent = False

    old_rows = ~np.char.startswith(poses["model"], "mt_")
    old = set(poses["model"][old_rows])
    new = set(poses["model"][~old_rows])
    emission = setup["lights"]["eevee_tree_emission"]
    checks = []
    with TemporaryDirectory(prefix="roadside-textures-") as temp:
        packages = Path(setup["tree_packages"]) if "tree_packages" in setup else None
        library = current_library(old, Path(temp), emission, packages)
        if new:
            library.update(prototype_library(new, emission))
        instances = place_trees(scene, rows, poses)
        reference = setup["photo_height_reference"]
        photo_height = reference[2] - road_height(reference, roads)[0]
        positions = np.column_stack([rows["x"], rows["y"]])
        for view in setup["views"]:
            location = view["center"]
            floor, mesh = road_height(location, roads)
            height = view["height"] if view["height"] is not None else photo_height
            camera_z = location[2] if view["fixed_z"] else floor + height
            cam.location = (location[0], location[1], camera_z)
            cam.rotation_euler = Vector(view["direction"]).to_track_quat("-Z", "Y").to_euler()
            camera.lens = camera.sensor_width / (2 * math.tan(math.radians(view["h_fov"] / 2)))
            distance = np.linalg.norm(positions - np.asarray(location[:2]), axis=1)
            lods = np.where(old_rows, np.where(distance < 40, 0, np.where(distance < 110, 1, 2)), distance >= 40)
            for instance, model, lod in zip(instances, poses["model"], lods):
                instance.instance_collection = library[model, int(lod)]
            checks.append(dict(view=view["name"], station=view["station"], mode=mode, location=list(cam.location),
                               direction=list(Vector(view["direction"]).normalized()), fov_horizontal=view["h_fov"],
                               road_mesh=mesh, height_from_road=camera_z - floor, plants=len(rows),
                               old=int(old_rows.sum()), prototypes=int((~old_rows).sum())))
            print(json.dumps(checks[-1]), flush=True)
            render_with_ground_aov(scene, out / f"{view['name']}__{mode}")
    (out / f"cameras_{mode}.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    scene_dir, output, variant = sys.argv[sys.argv.index("--") + 1:]
    render(Path(scene_dir), Path(output), variant)
