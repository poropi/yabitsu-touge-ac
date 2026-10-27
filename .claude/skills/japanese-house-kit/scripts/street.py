"""試作（町並み）：照合地点のまわりの家の位置に、部品集（kit.py）の家を建て、地面は空中写真の色から芝・舗装・土を見分けて素材で塗り、
道の両側に電柱と電線、家の前に生け垣・庭木を置いて、ストリートビューと同じ位置・向き・画角で撮る。
使い方: blender -b --factory-startup --python-exit-code 1 --python street.py -- <scene.json のフォルダ> <出力 png>"""
import json, math, os, random, sys
import bmesh, bpy, numpy as np
from mathutils import Matrix, Vector
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
args = sys.argv[sys.argv.index("--") + 1:]
D, OUT = args[0], args[1]
S = json.load(open(f"{D}/scene.json"))
bpy.ops.wm.read_factory_settings(use_empty=True)
import kit
sc = bpy.context.scene
R = S["R"]; g = np.array(S["grid"]); Z = np.array(S["Z"]); n = len(g)

# ---- 地面の高さ：格子をならす（段差の角がとがって見えた）----
from math import exp
k = np.array([exp(-(i * i) / 2.0) for i in range(-2, 3)]); k /= k.sum()
Zs = Z.copy()
for ax in (0, 1):
    Zs = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 2, mode="edge"), k, "valid"), ax, Zs)

# 道から 14 m 以内は道の高さへ寄せる（地形の格子の盛り上がりが道ぞいの土手に見えた。写真は平ら）
RPz = np.c_[S["road"]["x"], S["road"]["y"]]; RZ = np.array(S["road"]["z"])
GX, GY = np.meshgrid(g, g)
from math import inf
dmin = np.full(GX.shape, inf); zr = np.zeros(GX.shape)
for (x, y), z in zip(RPz, RZ):
    d = np.hypot(GX - x, GY - y); m = d < dmin; dmin[m] = d[m]; zr[m] = z
wgt = np.clip((dmin - 6.0) / 8.0, 0, 1)
Zs = zr * (1 - wgt) + Zs * wgt


def zat(x, y):
    i = np.clip((x + R) / 2.0, 0, n - 1.001); j = np.clip((y + R) / 2.0, 0, n - 1.001)
    i0, j0 = int(i), int(j); fi, fj = i - i0, j - j0
    return (Zs[j0, i0] * (1 - fi) * (1 - fj) + Zs[j0, i0 + 1] * fi * (1 - fj) + Zs[j0 + 1, i0] * (1 - fi) * fj + Zs[j0 + 1, i0 + 1] * fi * fj)

# ---- 地面の塗り分け：空中写真の色から芝（緑）・土（茶）・舗装（灰）。家の輪郭の内側は舗装扱い ----
# 塗り分けの絵 ground_mask.png（R＝芝、G＝土、それ以外は舗装）は ground_mask.py で作る（Blender の Python には PIL が無い）
me = bpy.data.meshes.new("ground"); bm = bmesh.new(); uvl = bm.loops.layers.uv.new()
vs = [[bm.verts.new((g[i], g[j], Zs[j][i] - 0.04)) for i in range(n)] for j in range(n)]
for j in range(n - 1):
    for i in range(n - 1):
        f = bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
        for l in f.loops:
            l[uvl].uv = ((l.vert.co.x + R) / (2 * R), (l.vert.co.y + R) / (2 * R))
bm.to_mesh(me); bm.free()
gm = bpy.data.materials.new("ground"); nt = gm.node_tree; bsdf = nt.nodes["Principled BSDF"]; bsdf.inputs["Specular IOR Level"].default_value = 0.1
bsdf.inputs["Roughness"].default_value = 0.95
tc = nt.nodes.new("ShaderNodeTexCoord"); uvn = nt.nodes.new("ShaderNodeUVMap")
mt = nt.nodes.new("ShaderNodeTexImage"); mt.image = bpy.data.images.load(f"{D}/ground_mask.png"); mt.image.colorspace_settings.name = "Non-Color"
sep = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(mt.outputs["Color"], sep.inputs["Color"])
def hkcol(name, size_m):
    """Codex で作った地面の絵（assets/house_kit/<name>.png）の色。地面は UV でなく位置で敷く（1 枚 size_m m）。"""
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1 / size_m,) * 3; nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    im = nt.nodes.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(str(kit.HK / f"{name}.png"), check_existing=True)
    nt.links.new(mp.outputs["Vector"], im.inputs["Vector"]); return im.outputs["Color"]
def texcol(asset, scale, tint_hex, k=0.8):
    mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (scale,) * 3; nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    im = nt.nodes.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(f"{kit.ACG}/{asset}/{asset}_1K-JPG_Color.jpg", check_existing=True)
    nt.links.new(mp.outputs["Vector"], im.inputs["Vector"])
    bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(im.outputs["Color"], bw.inputs["Color"])
    mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"; mx.inputs["Factor"].default_value = 1.0
    kit._sock(mx.inputs, "A_Color").default_value = kit._lin(tint_hex)
    ma = nt.nodes.new("ShaderNodeMath"); ma.operation = "MULTIPLY_ADD"; ma.inputs[1].default_value = k; ma.inputs[2].default_value = 1 - k / 2
    nt.links.new(bw.outputs["Val"], ma.inputs[0]); nt.links.new(ma.outputs["Value"], kit._sock(mx.inputs, "B_Color"))
    return kit._sock(mx.outputs, "Result_Color")
if (kit.HK / "lawn.png").exists():   # Codex の絵があればそれ（舗装は土間コンクリートと砂利を斑に混ぜる）
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.08; nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    cr = nt.nodes.new("ShaderNodeMapRange"); cr.inputs["From Min"].default_value = 0.48; cr.inputs["From Max"].default_value = 0.52; nt.links.new(nz.outputs["Fac"], cr.inputs["Value"])
    mp0 = nt.nodes.new("ShaderNodeMix"); mp0.data_type = "RGBA"; nt.links.new(cr.outputs["Result"], mp0.inputs["Factor"])
    nt.links.new(hkcol("driveway", 2.0), kit._sock(mp0.inputs, "A_Color")); nt.links.new(hkcol("gravel", 2.0), kit._sock(mp0.inputs, "B_Color"))
    paved = kit._sock(mp0.outputs, "Result_Color"); grasc = hkcol("lawn", 2.0); soilc = hkcol("garden_soil", 2.0)
else:
    paved = texcol("Concrete034", 0.3, "#6A6862", 0.6)   # 明るすぎて白く飛んだ
    grasc = texcol("Concrete034", 1.2, "#4E6232", 1.4)
    soilc = texcol("Concrete034", 0.5, "#6E5A44", 0.9)
m1 = nt.nodes.new("ShaderNodeMix"); m1.data_type = "RGBA"; nt.links.new(sep.outputs["Red"], m1.inputs["Factor"])
nt.links.new(paved, kit._sock(m1.inputs, "A_Color")); nt.links.new(grasc, kit._sock(m1.inputs, "B_Color"))
m2 = nt.nodes.new("ShaderNodeMix"); m2.data_type = "RGBA"; nt.links.new(sep.outputs["Green"], m2.inputs["Factor"])
nt.links.new(kit._sock(m1.outputs, "Result_Color"), kit._sock(m2.inputs, "A_Color")); nt.links.new(soilc, kit._sock(m2.inputs, "B_Color"))
nt.links.new(kit._sock(m2.outputs, "Result_Color"), bsdf.inputs["Base Color"])
og = bpy.data.objects.new("ground", me); og.data.materials.append(gm); sc.collection.objects.link(og)

# ---- 道：アスファルト、外側線・中央線、縁石 ----
rd = S["road"]; N = len(rd["x"])
def strip(name, m, off0, off1, dz):
    me = bpy.data.meshes.new(name); bm = bmesh.new(); prev = None
    for k in range(N - 1):
        dx, dy = rd["x"][k + 1] - rd["x"][k], rd["y"][k + 1] - rd["y"][k]; l = math.hypot(dx, dy) or 1; nx, ny = -dy / l, dx / l
        a, b_ = off0(k), off1(k)
        cur = (bm.verts.new((rd["x"][k] + nx * a, rd["y"][k] + ny * a, rd["z"][k] + dz)), bm.verts.new((rd["x"][k] + nx * b_, rd["y"][k] + ny * b_, rd["z"][k] + dz)))
        if prev:
            bm.faces.new((prev[0], prev[1], cur[1], cur[0]))
        prev = cur
    bm.to_mesh(me); bm.free(); o = bpy.data.objects.new(name, me); o.data.materials.append(m); sc.collection.objects.link(o)
road_m = kit.img_mat("asphalt", (4.0, 2.0), 0.9, 0.6) if (kit.HK / "asphalt.png").exists() else kit.mat("asphalt", "#4E4E4C", 0.9, tex="Concrete034", scale=0.25, tint=0.8, spec=0.1)
strip("road", road_m, lambda k: -rd["hr"][k] - 0.7, lambda k: rd["hl"][k] + 0.7, 0.03)
lw = kit.mat("line_w", "#E2E2DC", 0.6); ly = kit.mat("line_y", "#D8A030", 0.6)
strip("lineL", lw, lambda k: rd["hl"][k] - 0.25, lambda k: rd["hl"][k] - 0.1, 0.04)
strip("lineR", lw, lambda k: -rd["hr"][k] + 0.1, lambda k: -rd["hr"][k] + 0.25, 0.04)
strip("lineC", ly, lambda k: -0.075, lambda k: 0.075, 0.04)

# ---- ゲームの擁壁（石垣の絵）・塀・ガードレール ----
WM = {"ishigaki": kit.img_mat("ishigaki", (2.0, 2.0), 0.95, 1.5) if (kit.HK / "ishigaki.png").exists() else kit.mat("ishigaki", "#5A544C", 0.9, tex="Concrete034"),
      "block": kit.mat("block", "#827E74", 0.9, tex="Concrete034", scale=1.2, tint=0.6, grime=1.5),
      "rail": kit.mat("rail", "#D8D8D2", 0.35, 0.6)}
for w_ in S.get("walls", []):
    me = bpy.data.meshes.new(w_["name"]); me.from_pydata(w_["v"], [], w_["f"]); me.update()
    o = bpy.data.objects.new(w_["name"], me); o.data.materials.append(WM[w_["kind"]]); sc.collection.objects.link(o)


def road_frame(k):
    dx, dy = rd["x"][k + 1] - rd["x"][k], rd["y"][k + 1] - rd["y"][k]; l = math.hypot(dx, dy) or 1
    return np.array([rd["x"][k], rd["y"][k]]), np.array([dx / l, dy / l]), np.array([-dy / l, dx / l])

# ---- 家 ----
RP = np.c_[rd["x"], rd["y"]]
plant_items = []; rnd_p = random.Random(11)
HK_OBJ = os.environ.get("HK_OBJ")   # ゲーム用の家（game_preview.py の OBJ）で撮るとき
if HK_OBJ:
    bpy.ops.wm.obj_import(filepath=HK_OBJ, forward_axis="Y", up_axis="Z")
    for mm in bpy.data.materials:
        if mm.node_tree and "Principled BSDF" in mm.node_tree.nodes:
            bb = mm.node_tree.nodes["Principled BSDF"]; bb.inputs["Roughness"].default_value = 0.8; bb.inputs["Specular IOR Level"].default_value = 0.2
for h in ([] if HK_OBJ else S["houses"]):
    rnd = random.Random(sum(map(ord, h["id"])))
    if np.hypot(*(RP - np.array(h["c"])).T).min() < 3.0:
        continue
    a = math.radians(h["ang"]) + (0 if h["front"] > 0 else math.pi)
    floor = h["z_high"] + kit.FL; drop = h["z_high"] - h["z_low"]
    T = Matrix.Translation((h["c"][0], h["c"][1], floor)) @ Matrix.Rotation(a, 4, "Z")
    if h["area"] > 260 or h["flat"]:
        L, W = h["L"], h["W"]
        G = kit.Mesh(); P = kit.palette(rnd)
        G.box(P["base"], -L / 2, L / 2, -W / 2, W / 2, -kit.FL - drop - 0.3, 0)
        st = max(2, h["storeys"])
        for s in range(st):
            G.box(P["wall"], -L / 2, L / 2, -W / 2, W / 2, s * 3.0, (s + 1) * 3.0)
            for i in range(max(1, int(L / 3.6))):
                u = -L / 2 + (i + 0.5) * L / max(1, int(L / 3.6))
                kit.sash_window(G, P, Matrix.Translation((0, W / 2, s * 3.0)), u, 1.6, 0.9, 2.1, rnd, shutter=False)
        G.box(P["trim"], -L / 2 - 0.05, L / 2 + 0.05, -W / 2 - 0.05, W / 2 + 0.05, st * 3.0, st * 3.0 + 0.5)
        G.emit(h["id"], T)
        continue
    kit.build_house(h["id"], h["L"], h["W"], rnd, roof="hip" if rnd.random() < 0.4 else "gable", storeys=2 if h["area"] > 45 else 1,
                    ground_drop=drop, origin=T)
    # 家の前：生け垣かブロック塀、庭木
    fwd = np.array([math.cos(a + math.pi / 2), math.sin(a + math.pi / 2)]); side = np.array([math.cos(a), math.sin(a)])
    c = np.array(h["c"]); front = c + fwd * (h["W"] / 2 + rnd.uniform(1.5, 3.0))
    if rnd.random() < 0.65:
        for t in np.arange(-h["L"] / 2, h["L"] / 2 - 1.5, 0.9):
            p = front + side * t
            plant_items.append(("hedge", p[0], p[1], zat(*p), rnd.uniform(0.9, 1.1)))
    for _ in range(rnd.randint(1, 3)):
        p = c + fwd * (h["W"] / 2 + rnd.uniform(0.8, 2.5)) + side * rnd.uniform(-h["L"] / 2, h["L"] / 2)
        plant_items.append(("tree" if rnd.random() < 0.6 else "shrub", p[0], p[1], zat(*p), rnd.uniform(0.7, 1.1)))
# 空中写真の濃い緑（ground_mask.py の trees.json）に庭木・林の木。家の輪郭の近くと道の上は外す
import os
if os.path.exists(f"{D}/trees.json"):
    HC = np.array([h["c"] for h in S["houses"]]) if S["houses"] else np.zeros((0, 2))
    HR = np.array([max(h["L"], h["W"]) / 2 + 1.0 for h in S["houses"]]) if S["houses"] else np.zeros(0)
    for x, y, s in json.load(open(f"{D}/trees.json")):
        if np.hypot(*(RP - [x, y]).T).min() < 6.0:
            continue
        if len(HC) and (np.hypot(*(HC - [x, y]).T) < HR).any():
            continue
        plant_items.append(("tree" if rnd_p.random() < 0.7 else "shrub", x, y, zat(x, y), s))
if plant_items:
    kit.plants("plants", plant_items, rnd_p)

# ---- 電柱と電線（道の左右、約 33 m おき） ----
poleM = kit.mat("pole", "#B6B4AC", 0.8, tex="Concrete034", scale=1.0, tint=0.4)
wireM = kit.mat("wire", "#141414", 0.6)
G = kit.Mesh()
L_road = np.cumsum([0] + [math.hypot(rd["x"][k + 1] - rd["x"][k], rd["y"][k + 1] - rd["y"][k]) for k in range(N - 1)])
for side, off in ((1, 1.0), (-1, 1.0)):
    tops = []
    for s in np.arange(8.0 if side > 0 else 25.0, L_road[-1] - 2, 33.0):
        k = int(np.searchsorted(L_road, s)) - 1; k = max(0, min(k, N - 2))
        o, t, nn = road_frame(k)
        hw = rd["hl"][k] if side > 0 else rd["hr"][k]
        p = o + nn * side * (hw + 0.9 + off)
        z = zat(*p)
        G.cyl(poleM, (p[0], p[1], z - 0.5), (p[0], p[1], z + 12.0), 0.17, seg=10)
        for zz, wa in ((z + 11.4, 1.0), (z + 10.6, 0.8), (z + 8.2, 0.6)):
            G.box(poleM, -wa, wa, -0.05, 0.05, -0.05, 0.05, Matrix.Translation((p[0], p[1], zz)) @ Matrix.Rotation(math.atan2(nn[1], nn[0]), 4, "Z"))
        if random.Random(int(s)).random() < 0.4:   # 変圧器
            G.cyl(kit.mat("trans", "#8A8E8C", 0.5, 0.3), (p[0] + t[0] * 0.4, p[1] + t[1] * 0.4, z + 7.0), (p[0] + t[0] * 0.4, p[1] + t[1] * 0.4, z + 8.0), 0.3, seg=12)
        tops.append((p, nn * side, z))
    for (p0, n0, z0), (p1, n1, z1) in zip(tops[:-1], tops[1:]):
        for dz, lat in ((11.4, -0.8), (11.4, 0.0), (11.4, 0.8), (10.6, -0.6), (10.6, 0.6), (8.2, 0.0), (7.6, 0.3)):
            a0 = np.r_[p0 + n0 * lat, z0 + dz]; a1 = np.r_[p1 + n1 * lat, z1 + dz]
            d = np.linalg.norm(a1[:2] - a0[:2]); prev = a0
            for q in range(1, 11):
                tt = q / 10; cur = a0 + (a1 - a0) * tt; cur[2] -= 0.025 * d * 4 * tt * (1 - tt)
                G.cyl(wireM, prev, cur, 0.012 if dz > 9 else 0.02, seg=5); prev = cur
G.emit("poles", bevel=0)

# ---- 光と空、カメラ ----
sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 3.5; sun.angle = math.radians(0.6)
so = bpy.data.objects.new("sun", sun); so.rotation_euler = (math.radians(30), 0, math.radians(170)); sc.collection.objects.link(so)
w = bpy.data.worlds.new("w"); sc.world = w; wn = w.node_tree; sky = wn.nodes.new("ShaderNodeTexSky")
for kk, v in (("sky_type", "NISHITA"), ("sun_elevation", math.radians(60)), ("sun_rotation", math.radians(170)), ("sun_disc", False)):
    try: setattr(sky, kk, v)
    except Exception: pass
wn.links.new(sky.outputs["Color"], wn.nodes["Background"].inputs["Color"]); wn.nodes["Background"].inputs["Strength"].default_value = 0.25
cam = bpy.data.cameras.new("cam"); cam.lens = 18; cam.sensor_width = 36; cam.sensor_fit = "HORIZONTAL"; cam.clip_end = 3000
co = bpy.data.objects.new("cam", cam); sc.collection.objects.link(co); sc.camera = co
c = S["cam"]; co.location = (c["x"], c["y"], c["z"]); co.rotation_euler = (math.radians(90), 0, math.radians(-c["heading"]))
sc.render.engine = "CYCLES"; sc.cycles.samples = 64; sc.cycles.use_denoising = True
try:
    pr = bpy.context.preferences.addons["cycles"].preferences; pr.compute_device_type = "METAL"; pr.get_devices()
    for dv in pr.devices: dv.use = True
    sc.cycles.device = "GPU"
except Exception:
    pass
sc.render.resolution_x, sc.render.resolution_y = 1500, 1000
sc.view_settings.view_transform = "AgX"
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT.replace(".png", ".blend"))
print("RENDERED", OUT)
