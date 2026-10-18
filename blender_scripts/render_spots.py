"""確認用：.blend を開き、決めた地点・向きからの眺めを Workbench で描画する（Street View と並べて見るため）。
使い方: blender -b FILE.blend --python render_spots.py -- SPOTS_JSON OUT_DIR
SPOTS_JSON = [{"name", "x", "y", "z", "yaw_deg"（方位、北=0）, "pitch_deg"（正＝下向き）, "fov_deg"}, ...] と {"top": {...}}"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler

argv = sys.argv[sys.argv.index("--") + 1:]
spec = json.loads(Path(argv[0]).read_text(encoding="utf-8")); out = Path(argv[1]); out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_WORKBENCH"
except TypeError as e:
    print("engine:", e)
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "TEXTURE"
scene.render.resolution_x, scene.render.resolution_y = 900, 600
for ob in bpy.data.objects:   # ゲームで見えない部品（見えない壁など）は描かない
    if ob.get("ac_renderable") is False or ob.get("ac_renderable") == 0:
        ob.hide_render = True


def _ground(x, y):
    """z を書かない地点：真上から下へ光線を当て、描く物の一番下（路面・地面。信号の腕などの上ではなく）の高さ"""
    from mathutils import Vector
    dg = bpy.context.evaluated_depsgraph_get(); z = 3000.0; low = 0.0
    while True:
        hit, loc, _, _, ob, _ = scene.ray_cast(dg, Vector((x, y, z)), Vector((0, 0, -1)))
        if not hit:
            return low
        if not ob.hide_render:
            low = loc.z
        z = loc.z - 0.01


cam_data = bpy.data.cameras.new("cam"); cam_data.clip_end = 6000
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam
for s in spec.get("spots", []):
    cam_data.type = "PERSP"
    cam_data.angle = math.radians(s.get("fov_deg", 100.0))
    cam.location = (s["x"], s["y"], s["z"] if s.get("z") is not None else _ground(s["x"], s["y"]) + s.get("eye_m", 1.7))
    yaw = math.radians(s["yaw_deg"]); pitch = math.radians(s.get("pitch_deg", 0.0))
    # Blender のカメラは -Z を向く。X 軸まわりに 90°−pitch 起こしてから、方位ぶん Z 軸まわりに回す（北=+Y）
    cam.rotation_euler = Euler((math.pi / 2 - pitch, 0.0, -yaw), "XYZ")
    scene.render.filepath = str(out / f"{s['name']}.png"); bpy.ops.render.render(write_still=True)
if "top" in spec:
    t = spec["top"]
    cam_data.type = "ORTHO"; cam_data.ortho_scale = t["size"]
    cam.location = (t["x"], t["y"], t["z"]); cam.rotation_euler = (0.0, 0.0, 0.0)
    scene.render.resolution_x, scene.render.resolution_y = 1100, 1100
    scene.render.filepath = str(out / "top.png"); bpy.ops.render.render(write_still=True)
