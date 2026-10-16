"""確認用：.blend を開き、区間データの指定位置からの眺めを Workbench で描画する。
使い方: blender -b FILE.blend --python render_views.py -- SECTION_JSON OUT_DIR s1 s2 ... [--top]
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
sec = json.loads(Path(argv[0]).read_text(encoding="utf-8")); out = Path(argv[1]); out.mkdir(parents=True, exist_ok=True)
marks = [a for a in argv[2:] if a != "--top"]; top = "--top" in argv

scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_WORKBENCH"
except TypeError as e:
    print("engine:", e)
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "TEXTURE"
scene.render.resolution_x, scene.render.resolution_y = 1280, 720
scene.world = scene.world or bpy.data.worlds.new("w")
for c in bpy.data.collections:
    if c.name == "UNDERLAY":
        c.hide_render = not top
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 24; cam_data.clip_end = 6000
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam


def at(s):
    S = sec["s"]; i = min(range(len(S)), key=lambda k: abs(S[k] - s))
    return i


for m in marks:
    i = at(float(m)); h = sec["heading"][i]
    x, y, z = sec["x"][i], sec["y"][i], sec["z"][i]
    eye = Vector((x - 12 * math.cos(h), y - 12 * math.sin(h), z + 4.0))
    j = at(float(m) + 25); target = Vector((sec["x"][j], sec["y"][j], sec["z"][j]))
    cam.location = eye; cam.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out / f"view_s{int(float(m)):04d}.png"); bpy.ops.render.render(write_still=True)
if top:
    xs, ys = sec["x"], sec["y"]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cam_data.type = "ORTHO"; cam_data.ortho_scale = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.15
    cam.location = (cx, cy, 2000); cam.rotation_euler = (0, 0, 0)
    scene.render.resolution_x, scene.render.resolution_y = 1000, 1400
    scene.render.filepath = str(out / "view_top.png"); bpy.ops.render.render(write_still=True)
