"""切り出しと既存シーンの照明更新で使う Blender 内の操作。"""

from hashlib import sha256
import math
from pathlib import PurePosixPath

import bpy
from mathutils import Vector


def lighting(scene, collection, settings):
    world = scene.world
    if world is None:
        world = scene.world = bpy.data.worlds.new("Game Sky")
        if bpy.app.version < (5, 0, 0):
            world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (*settings["world_color"], 1)
    background.inputs["Strength"].default_value = settings["world_strength"]
    sun = scene.objects.get("Game Sun")
    if sun is None:
        light = bpy.data.lights.new("Game Sun", "SUN")
        sun = bpy.data.objects.new("Game Sun", light)
        collection.objects.link(sun)
    sun.data.energy = settings["eevee_sun_energy"]
    sun.data.angle = math.radians(1.)
    azimuth = math.radians(settings["sun_azimuth_deg"])
    elevation = math.radians(settings["sun_elevation_deg"])
    direction = Vector((math.sin(azimuth) * math.cos(elevation),
                        math.cos(azimuth) * math.cos(elevation), math.sin(elevation)))
    sun.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    scene.view_settings.view_transform = settings["view_transform"]
    scene.view_settings.look = "None"
    scene.view_settings.exposure = settings["exposure"]
    scene.view_settings.gamma = 1.0
    scene.view_settings.use_curve_mapping = False


def localize_images():
    """Blender の「外へ出す」の各選び方が元のファイルを指さないようにする。"""
    for image in bpy.data.images:
        if image.source != "FILE":
            continue
        if image.library or image.is_dirty:
            raise ValueError(f"画像を安全に移せません。ローカル画像を保存してください: {image.name}")
        if not image.packed_file:
            image.pack()
        if not image.packed_file:
            raise ValueError(f"画像を同梱できません: {image.name}")
        path = PurePosixPath(image.filepath.replace("\\", "/"))
        if path.parent.as_posix() == "//textures":
            target = image.filepath
        else:
            content = sha256(image.packed_file.data).hexdigest()[:12]
            target = f"//textures/{path.stem}-{content}{path.suffix}"
        # Blender discards unused datablocks on save unless explicitly retained.
        if image.users == 0:
            image.use_fake_user = True
        image.filepath_raw = target
        for packed in image.packed_files:
            packed.filepath = target


def material_coefficients(settings):
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        strength = settings["eevee_tree_emission" if material.name in {"CSP proxy foliage", "CSP proxy bark"}
                            else "eevee_emission"]
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Emission Strength"].default_value = strength
