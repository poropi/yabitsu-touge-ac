"""Render the cameras saved in a river-site blend, with and without proxy trees."""

import json
from pathlib import Path
import sys

import bpy
import numpy as np
import OpenImageIO as oiio


def cryptomatte_ids(path):
    """Read the dominant object and material from Eevee's visibility passes."""
    source = oiio.ImageInput.open(str(path))
    if not source:
        raise ValueError(f"描画のEXRを読めません: {path}")
    try:
        attributes = {prop.name: prop.value for prop in source.spec().extra_attribs}
        manifest = {}
        for key, name in attributes.items():
            if not key.startswith("cryptomatte/") or not key.endswith("/name"):
                continue
            hashes = json.loads(attributes[key.removesuffix("name") + "manifest"])
            manifest[name] = {}
            for item, value in hashes.items():
                bits = int(value, 16)
                if (bits >> 23) & 255 in (0, 255):
                    bits ^= 1 << 23
                manifest[name][bits] = item
        layers = {}
        index = 0
        while source.seek_subimage(index, 0):
            names = source.spec().channelnames
            for name in manifest:
                if name + "00.r" in names:
                    pixels = source.read_image(oiio.FLOAT)
                    hits = pixels[..., names.index(name + "00.r")].copy().view(np.uint32)
                    weights = pixels[..., names.index(name + "00.g")]
                    layers[name.rsplit(".", 1)[-1]] = np.where(weights >= .5, hits, 0), manifest[name]
            index += 1
        return layers["CryptoObject"], layers["CryptoMaterial"]
    finally:
        source.close()


def save_visibility(scene, exr, destination):
    (objects, object_names), (materials, material_names) = cryptomatte_ids(exr)
    pairs = objects.astype(np.uint64) << 32 | materials
    keys, ids = np.unique(pairs, return_inverse=True)
    labels = []
    for value in keys.tolist():
        name = object_names.get(value >> 32)
        material = material_names.get(value & 0xffffffff)
        obj = scene.objects.get(name) if name else None
        collections = sorted(item.name for item in obj.users_collection) if obj else []
        labels.append({"object": name, "material": material, "collections": collections})
    np.savez_compressed(destination, pixels=ids.reshape(pairs.shape).astype(np.int32),
                        labels=json.dumps(labels, ensure_ascii=False))


def render(request, destination):
    scene = bpy.context.scene
    trees = bpy.data.collections["REF_trees"]
    view_layer = bpy.context.view_layer
    view_layer.use_pass_cryptomatte_object = True
    view_layer.use_pass_cryptomatte_material = True
    view_layer.pass_cryptomatte_depth = 2
    scene.render.resolution_x, scene.render.resolution_y = request["frame_size"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.film_transparent = True
    image_settings = scene.render.image_settings

    for mode in request["modes"]:
        trees.hide_render = mode == "no_trees"
        for view in request["views"]:
            camera = scene.objects[view["camera"]]
            if camera.type != "CAMERA":
                raise ValueError(f"{camera.name} is not a camera")
            scene.camera = camera
            prefix = destination / f"{view['name']}__{mode}"
            exr = prefix.with_suffix(".exr")
            if bpy.app.version >= (5, 2, 0):
                image_settings.media_type = "MULTI_LAYER_IMAGE"
            else:
                image_settings.file_format = "OPEN_EXR_MULTILAYER"
            image_settings.color_depth = "32"
            scene.render.filepath = str(exr)
            bpy.ops.render.render(write_still=True)
            save_visibility(scene, exr, prefix.with_suffix(".npz"))
            exr.unlink()
            if bpy.app.version >= (5, 2, 0):
                image_settings.media_type = "IMAGE"
            image_settings.file_format = "PNG"
            image_settings.color_mode = "RGBA"
            image_settings.color_depth = "8"
            bpy.data.images["Render Result"].save_render(str(prefix.with_suffix(".png")), scene=scene)

    background = next(node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND")
    colour = background.inputs["Color"].default_value[:3]
    strength = background.inputs["Strength"].default_value
    scale = strength * 2 ** scene.view_settings.exposure

    def srgb(linear):
        return 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055

    sky = [round(255 * max(0, min(1, srgb(value * scale)))) for value in colour]
    (destination / "sky.json").write_text(json.dumps(sky), encoding="utf-8")


if __name__ == "__main__":
    request_path, out_dir = map(Path, sys.argv[sys.argv.index("--") + 1:])
    render(json.loads(request_path.read_text(encoding="utf-8")), out_dir)
