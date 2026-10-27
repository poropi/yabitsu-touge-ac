"""部品集（kit.py）で家 1 軒を建て、道からストリートビューの高さで撮る。
使い方: blender -b --factory-startup --python-exit-code 1 --python one_house.py -- <出力 png> [seed] [roof gable|hip]"""
import math, random, sys
import bpy
from mathutils import Matrix
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[sys.argv.index("--") + 1:]
OUT = args[0]; seed = int(args[1]) if len(args) > 1 else 3; roof = args[2] if len(args) > 2 else "hip"
bpy.ops.wm.read_factory_settings(use_empty=True)
import kit
sc = bpy.context.scene
rnd = random.Random(seed)

# 家：長さ 10.0 × 奥行き 7.3、正面は道（-y 側の道を向くように 180° 回す）。床は地面から 0.5 m
kit.build_house("house", 10.0, 7.3, rnd, roof=roof, storeys=2, wall_hex="#ECEAE4", roof_hex="#5A3C2C",
                origin=Matrix.Translation((0, 0, kit.FL)) @ Matrix.Rotation(math.pi, 4, "Z"))

# 地面：庭（土と草）、駐車場のコンクリート、歩道、道
def plane(name, m, x0, x1, y0, y1, z=0.0):
    G = kit.Mesh(); G.box(m, x0, x1, y0, y1, z - 0.1, z); return G.emit(name, bevel=0)
plane("yard", kit.mat("grass", "#5A6A3A", 0.95, tex="Concrete034", scale=0.3, tint=0.9), -300, 300, -12, 300, -0.02)
plane("park", kit.mat("concrete_pad", "#A8A69E", 0.9, tex="Concrete034", scale=0.4, tint=0.4), 5.5, 11, -9, -3.7, 0.0)
plane("walk", kit.mat("sidewalk", "#8C8C88", 0.9, tex="Concrete034", scale=0.5, tint=0.5), -20, 20, -11.0, -9.0, 0.05)
plane("road", kit.mat("asphalt", "#4A4A48", 0.9, tex="Concrete034", scale=0.25, tint=0.7), -300, 300, -300, -11.0, -0.1)
G = kit.Mesh(); G.box(kit.mat("curb", "#B0AEA8", 0.85), -20, 20, -11.15, -11.0, -0.1, 0.12); G.emit("curb")
# ブロック塀（敷地の前、駐車場の口は開ける）と生け垣
blk = kit.mat("block", "#827E74", 0.9, tex="Concrete034", scale=1.2, tint=0.6, grime=1.5)
G = kit.Mesh()
G.box(blk, -7, 5.3, -8.95, -8.8, 0, 1.0); G.box(kit.mat("cap", "#8A867C", 0.8), -7.02, 5.32, -8.98, -8.77, 1.0, 1.05)
jn = kit.mat("joint", "#6E6A62", 0.95)
for k in range(1, 5):                     # ブロックの目地：横 0.2 m・縦 0.4 m（段ごとに半分ずらす）
    G.box(jn, -7, 5.3, -8.962, -8.95, k * 0.2 - 0.006, k * 0.2 + 0.006)
for r in range(5):
    for k in range(32):
        x = -7 + k * 0.4 + (0.2 if r % 2 else 0.0)
        if x < 5.3: G.box(jn, x - 0.006, x + 0.006, -8.962, -8.95, r * 0.2, r * 0.2 + 0.2)
G.emit("wall", bevel=0)
rr = random.Random(5)
items = [("hedge", x, -8.2, 0.0, 1.0) for x in [-6.5 + 0.9 * k for k in range(7)]]
items += [("shrub", 3.0, -6.5, 0.0, 1.0), ("shrub", -5.5, -5.0, 0.0, 0.8), ("tree", -8.0, -6.0, 0.0, 1.0), ("tree", 4.2, -6.8, 0.0, 0.8)]
kit.plants("plants", items, rr)
# 白線
G = kit.Mesh(); G.box(kit.mat("line", "#E6E6E0", 0.6), -40, 40, -11.9, -11.75, -0.099, -0.09); G.emit("line")

# 光：太陽と空（Nishita）、Cycles
sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 3.5; sun.angle = math.radians(0.6)
so = bpy.data.objects.new("sun", sun); so.rotation_euler = (math.radians(40), 0, math.radians(150)); sc.collection.objects.link(so)
w = bpy.data.worlds.new("w"); sc.world = w; nt = w.node_tree; sky = nt.nodes.new("ShaderNodeTexSky")
for k, v in (("sky_type", "NISHITA"), ("sun_elevation", math.radians(50)), ("sun_rotation", math.radians(150)), ("sun_disc", False)):
    try: setattr(sky, k, v)
    except Exception: pass
nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"]); nt.nodes["Background"].inputs["Strength"].default_value = 0.25

cam = bpy.data.cameras.new("cam"); cam.lens = 28; cam.sensor_width = 36
co = bpy.data.objects.new("cam", cam); sc.collection.objects.link(co); sc.camera = co
co.location = (-9.5, -17.5, 2.4)
d = (0.5, 0.0, 3.6)
import mathutils
v = mathutils.Vector(d) - co.location; co.rotation_euler = v.to_track_quat("-Z", "Y").to_euler()
sc.render.engine = "CYCLES"; sc.cycles.samples = 96; sc.cycles.use_denoising = True
try:
    pr = bpy.context.preferences.addons["cycles"].preferences; pr.compute_device_type = "METAL"; pr.get_devices()
    for dv in pr.devices: dv.use = True
    sc.cycles.device = "GPU"
except Exception:
    pass
sc.render.resolution_x, sc.render.resolution_y = 1240, 800
sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast" if False else "None"
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT.replace(".png", ".blend"))
print("RENDERED", OUT)
