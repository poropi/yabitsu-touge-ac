"""浅間神社の駐車場の石碑（Codex 制作、assets/landmarks/asama_monument/sengen_monument.blend）を、ゲームの部品にする。
コレクション「01 石碑・台座*」の物体を写して 1 つにまとめ（モディファイアは適用）、展開した UV へ元の物体の見た目
（手続き型の石肌・碑文・地衣類）を色だけ焼き付け、AO を掛ける。面に重ねた薄い板（本文の刻線・題字の文字・地衣類）は
形として持たず、下の石の面へ色として焼き込む（数千の小さな島で画像が埋まるのと、ゲームでのちらつきを避ける）。元の .blend は保存しない。
出力：<out>/monument.npz（v：m・+Z 上・正面 −Y、f：三角形、uv：Blender と同じ v=0 が画像の下。kn5 に書くときに 1−v にする）と <out>/monument.png。
使い方: blender -b sengen_monument.blend --python bake_asama_monument.py -- <out_dir> [解像度] [陰の強さ 0〜1、既定 0.3]"""
import sys
from pathlib import Path

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = Path(argv[0]); OUT.mkdir(parents=True, exist_ok=True)
RES = int(argv[1]) if len(argv) > 1 else 2048

scene = bpy.context.scene
col = next(c for c in scene.collection.children_recursive if c.name.startswith("01 "))
src = [o for o in col.all_objects if o.type in ("MESH", "FONT") and not o.hide_render]

bpy.ops.object.select_all(action="DESELECT")
for o in src:                                  # 文字は面に変える（焼き付けの元にも使う）
    if o.type == "FONT":
        o.select_set(True); bpy.context.view_layer.objects.active = o
        bpy.ops.object.convert(target="MESH")
        o.select_set(False)
src = [o for o in col.all_objects if o.type == "MESH" and not o.hide_render]

OVERLAY = ("本文", "額文字", "地衣類")          # 面に重ねた薄い板：焼き付けの元だけにする
body = [o for o in src if not any(k in o.name for k in OVERLAY)]

# まとめた写し（モディファイアを適用した形）
dg = bpy.context.evaluated_depsgraph_get()
parts = []
for o in body:
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    me.transform(o.matrix_world)
    t = bpy.data.objects.new("_part_" + o.name, me); scene.collection.objects.link(t); parts.append(t)
bpy.ops.object.select_all(action="DESELECT")
for t in parts:
    t.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
tgt = bpy.context.view_layer.objects.active; tgt.name = "monument_game"
tgt.data.materials.clear()
for uv in list(tgt.data.uv_layers):
    tgt.data.uv_layers.remove(uv)
tgt.data.uv_layers.new(name="bake")
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.002, area_weight=0.0, scale_to_bounds=False)
bpy.ops.uv.pack_islands(rotate=True, margin=0.002, shape_method="CONCAVE")
bpy.ops.object.mode_set(mode="OBJECT")

def bake_image(name):
    img = bpy.data.images.new(name, RES, RES, alpha=False, float_buffer=True)   # 線形のまま読む（8 ビットだと sRGB の二重変換になる）
    mat = bpy.data.materials.new("bake_" + name); mat.use_nodes = True
    n = mat.node_tree.nodes.new("ShaderNodeTexImage"); n.image = img
    mat.node_tree.nodes.active = n
    tgt.data.materials.clear(); tgt.data.materials.append(mat)
    return img

scene.render.engine = "CYCLES"; scene.cycles.device = "CPU"
bk = scene.render.bake
bk.margin = 8

# 色：元の物体から写しへ（選んだ物→アクティブ）
col_img = bake_image("color")
scene.cycles.samples = 8
bk.use_pass_direct = False; bk.use_pass_indirect = False; bk.use_pass_color = True
bk.use_selected_to_active = True; bk.cage_extrusion = 0.012; bk.max_ray_distance = 0.03
bpy.ops.object.select_all(action="DESELECT")
for o in src:
    o.select_set(True)
tgt.select_set(True); bpy.context.view_layer.objects.active = tgt
bpy.ops.object.bake(type="DIFFUSE")

# 陰（AO）：写しの形だけで
ao_img = bake_image("ao")
scene.cycles.samples = 96
bk.use_selected_to_active = False
bpy.ops.object.select_all(action="DESELECT"); tgt.select_set(True)
bpy.ops.object.bake(type="AO")

c = np.array(col_img.pixels[:], np.float32).reshape(RES, RES, 4)[..., :3]
a = np.array(ao_img.pixels[:], np.float32).reshape(RES, RES, 4)[..., :1]
AO_W = float(argv[2]) if len(argv) > 2 else 0.3   # 陰の強さ（0＝掛けない）。0.55 では木陰の駐車場で黒くつぶれた（2026-10-30）
lin = c * (1.0 - AO_W + AO_W * a)                # くぼみ・目地を少し暗く
enc = lambda x: np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055)
raw = bpy.data.images.new("monument_color", RES, RES, alpha=False)   # 陰を掛ける前の色（確認用）
raw.pixels = np.concatenate([enc(c), np.ones((RES, RES, 1), np.float32)], axis=2).ravel()
raw.filepath_raw = str(OUT / "monument_color.png"); raw.file_format = "PNG"; raw.save()
srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, 1), 1 / 2.4) - 0.055)
out = bpy.data.images.new("monument_out", RES, RES, alpha=False)
out.pixels = np.concatenate([srgb, np.ones((RES, RES, 1), np.float32)], axis=2).ravel()
out.filepath_raw = str(OUT / "monument.png"); out.file_format = "PNG"; out.save()

# 頂点は UV ごとに分ける（ゲームの部品は頂点ごとに 1 つの UV）
me = tgt.data
uvl = me.uv_layers["bake"].data
key, verts, uvs, faces = {}, [], [], []
for p in me.polygons:
    tri = []
    for li in p.loop_indices:
        vi = me.loops[li].vertex_index; u = tuple(round(x, 6) for x in uvl[li].uv)
        k = (vi, u)
        if k not in key:
            key[k] = len(verts); verts.append(tuple(me.vertices[vi].co)); uvs.append((u[0], u[1]))
        tri.append(key[k])
    faces.append(tri)
np.savez_compressed(OUT / "monument.npz", v=np.array(verts, np.float64), f=np.array(faces, np.int64), uv=np.array(uvs, np.float64))
print(f"MONUMENT {len(verts)} verts {len(faces)} tris -> {OUT}")
