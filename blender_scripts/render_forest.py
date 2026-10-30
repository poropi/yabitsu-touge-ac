"""森の確認画像を描く（scripts.forest_build --render から呼ばれる）。
使い方: blender -b --factory-startup --python blender_scripts/render_forest.py -- <出力フォルダ> <area>
出力フォルダの views.json（カメラ）・trees.npy（配置）・extension/trees/*.zip（木）と、manifest のメッシュ（コース）を使う。
環境変数 SAVE_BLEND=<パス> のときは描かずに、最初のカメラの範囲の場面へ全部のカメラ（名前＝views の name）を置き、
画像を詰めて .blend に保存する（Blender の画面で見せる用）。
"""
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from yabitsu import csptree  # noqa: E402

out = Path(sys.argv[sys.argv.index("--") + 1])
views = json.loads((out / "views.json").read_text(encoding="utf-8"))
SAMPLES = int(__import__("os").environ.get("RENDER_SAMPLES", "16"))
trees = np.load(out / "trees.npy")
MAN = ROOT / "data" / "processed" / "track"
manifest = json.loads((MAN / "manifest.json").read_text(encoding="utf-8"))
tmp = Path(tempfile.mkdtemp())
(out / "views").mkdir(exist_ok=True)

# manifest メッシュの範囲を先に調べておく
bboxes = {}
for name, e in manifest["meshes"].items():
    if e["material"] == "tree":
        continue
    v = np.load(MAN / e["file"])["verts"]
    bboxes[name] = (v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max())


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = SAMPLES   # 既定の 64 は 1 枚 2〜3 分かかる。比べる絵には 16 で足りる
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (0.55, 0.68, 0.85, 1); bg.inputs[1].default_value = 1.0
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.rotation_euler = (np.radians(40), np.radians(15), np.radians(140)); sun.data.energy = 4.0
    sun.data.angle = np.radians(2)
    return sc


def image_material(name, path, alpha=False, color=None):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.85
    if path and Path(path).exists():
        im = nt.nodes.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(str(path))
        nt.links.new(im.outputs[0], bsdf.inputs["Base Color"])
        if alpha:
            nt.links.new(im.outputs[1], bsdf.inputs["Alpha"])
            mat.use_backface_culling = False
    elif color:
        c = color.lstrip("#")
        bsdf.inputs["Base Color"].default_value = (*[int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)], 1)
    return mat


def mesh_object(name, v, f, uv, mat, coll):
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f.tolist())
    if uv is not None:
        layer = me.uv_layers.new()
        u = uv[np.asarray(f).ravel()]
        layer.data.foreach_set("uv", u.ravel())   # 部品データの uv はそのまま（kn5 は 1-v で書く＝AC と同じ見え方。2026-10-22 に 1-v をやめた）
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob)
    return ob


# 道路まわりの仕上げ（dress_build・dress_mountain の出力フォルダ）。DRESS_DIR に os.pathsep 区切りで複数書ける
DRESSES = [Path(p) for p in __import__("os").environ.get("DRESS_DIR", "").split(__import__("os").pathsep) if p]
RETEX = {n: (d, k) for d in DRESSES for n, k in json.loads((d / "retex.json").read_text(encoding="utf-8")).items()}
HIDDEN = {n for d in DRESSES for n in json.loads((d / "hidden.json").read_text(encoding="utf-8"))}


def load_dress(sc, win):
    """仕上げの部品（各フォルダの manifest）。"""
    x0, y0, x1, y1 = win
    for DRESS in DRESSES:
        dm = json.loads((DRESS / "manifest.json").read_text(encoding="utf-8"))
        for name, e in dm["meshes"].items():
            z = np.load(DRESS / e["file"])
            v = z["verts"]
            if v[:, 0].max() < x0 or v[:, 0].min() > x1 or v[:, 1].max() < y0 or v[:, 1].min() > y1:
                continue
            mspec = dm["materials"][e["material"]]
            tex = DRESS / mspec["texture"] if mspec.get("texture") else None
            col = mspec.get("color")
            hexcol = "#%02X%02X%02X" % tuple(int(c * 255) for c in col) if col else None
            mesh_object(name, v, z["faces"], z["uvs"], image_material(f"d_{DRESS.name}_" + e["material"], tex, mspec.get("alpha"), hexcol),
                        sc.collection)


def load_course(sc, win):
    x0, y0, x1, y1 = win
    for name, (a, b, c, d) in bboxes.items():
        if c < x0 or a > x1 or d < y0 or b > y1:
            continue
        e = manifest["meshes"][name]
        if not e.get("renderable", True) or name in HIDDEN:
            continue
        z = np.load(MAN / e["file"])
        m = manifest["materials"].get(e["material"], {})
        tex = MAN / m["texture"] if m.get("texture") else None
        mat = image_material(e["material"], tex, m.get("alpha"), m.get("color"))
        if name in RETEX:          # 仕上げの貼り替え（CSP の SHADER_REPLACEMENT と同じ画像）
            d, key = RETEX[name]
            mat = image_material("retex_" + key, d / "extension" / f"dress_{key}.dds")
        mesh_object(name, z["verts"], z["faces"], z["uvs"], mat, sc.collection)


def tree_objects(lod):
    """木の種類ごとのメッシュ（実寸）を作る。表示しないコレクションに入れ、インスタンスの元にする。"""
    coll = bpy.data.collections.new("tree_src")
    objs = {}
    for zp in sorted((out / "extension" / "trees").glob("*.zip")):
        z = zipfile.ZipFile(zp)
        ini = z.read("tree.ini").decode()
        H = float(ini.split("HEIGHT=")[1].split()[0]); W = float(ini.split("WIDTH=")[1].split()[0])
        tex, subs = csptree.read_mesh(z.read(f"LOD{lod}.mesh"))
        vs, fs, uvs, mats, base = [], [], [], [], 0
        me = bpy.data.meshes.new(zp.stem)
        allv, allf, alluv, mat_idx = [], [], [], []
        for k, s in enumerate(subs):
            p = s.verts["p"].astype(float)
            allv.append(np.c_[p[:, 0] * W / 2, -p[:, 2] * W / 2, p[:, 1] * H])
            f = s.indices.reshape(-1, 3).astype(np.int64)
            allf.append(f + base); alluv.append(s.verts["uv"]); mat_idx += [k] * len(f)
            base += len(p)
            name = tex[s.diffuse][1]
            path = tmp / name
            if not path.exists():
                path.write_bytes(z.read(name))
            me.materials.append(image_material(name, path, alpha=(k == 0 and "bark" not in name)))
        V, F, UV = np.concatenate(allv), np.concatenate(allf), np.concatenate(alluv)
        me.from_pydata(V.tolist(), [], F.tolist())
        layer = me.uv_layers.new()
        u = UV[F.ravel()]
        layer.data.foreach_set("uv", u.ravel())   # 部品データの uv はそのまま（kn5 は 1-v で書く＝AC と同じ見え方。2026-10-22 に 1-v をやめた）
        me.polygons.foreach_set("material_index", mat_idx)
        ob = bpy.data.objects.new(zp.stem, me); coll.objects.link(ob)
        objs[zp.stem] = ob
    return objs


def instance_trees(sc, objs, win):
    x0, y0, x1, y1 = win
    t = trees[(trees["x"] >= x0) & (trees["x"] <= x1) & (trees["y"] >= y0) & (trees["y"] <= y1)]
    for name, src in objs.items():
        sel = t[t["pattern"] == name]
        if not len(sel):
            continue
        me = bpy.data.meshes.new(f"pts_{name}")
        me.from_pydata(np.c_[sel["x"], sel["y"], sel["z"]].tolist(), [], [])
        a = me.attributes.new("rot", "FLOAT", "POINT"); a.data.foreach_set("value", np.radians(sel["angle_deg"]).astype(float))
        s = me.attributes.new("scl", "FLOAT", "POINT"); s.data.foreach_set("value", sel["scale"].astype(float))
        ob = bpy.data.objects.new(f"pts_{name}", me); sc.collection.objects.link(ob)
        gn = ob.modifiers.new("inst", "NODES")
        tree = bpy.data.node_groups.new(f"gn_{name}", "GeometryNodeTree")
        tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        n = tree.nodes
        gi, go = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
        iop = n.new("GeometryNodeInstanceOnPoints")
        oi = n.new("GeometryNodeObjectInfo"); oi.inputs[0].default_value = src
        ra = n.new("GeometryNodeInputNamedAttribute"); ra.data_type = "FLOAT"; ra.inputs[0].default_value = "rot"
        sa = n.new("GeometryNodeInputNamedAttribute"); sa.data_type = "FLOAT"; sa.inputs[0].default_value = "scl"
        comb = n.new("ShaderNodeCombineXYZ")
        L = tree.links
        L.new(gi.outputs[0], iop.inputs["Points"]); L.new(oi.outputs["Geometry"], iop.inputs["Instance"])
        L.new(ra.outputs["Attribute"], comb.inputs["Z"]); L.new(comb.outputs[0], iop.inputs["Rotation"])
        L.new(sa.outputs["Attribute"], iop.inputs["Scale"])
        L.new(iop.outputs[0], go.inputs[0])
        gn.node_group = tree
    return len(t)


def make_cam(sc, v):
    cam = bpy.data.objects.new(v["name"], bpy.data.cameras.new(v["name"])); sc.collection.objects.link(cam)
    cam.data.clip_end = v.get("clip_end", 20000)
    if v["type"] == "top":
        cx, cy, size = v["center"][0], v["center"][1], v["size"]
        cam.data.type = "ORTHO"; cam.data.ortho_scale = size
        cam.location = (cx, cy, v["z"] + 2000); cam.rotation_euler = (0, 0, 0)
        sc.render.resolution_x = sc.render.resolution_y = 1400
    else:
        eye, tgt = np.array(v["eye"]), np.array(v["target"])
        cam.location = eye.tolist()
        d = tgt - eye
        cam.rotation_euler = (np.arctan2(np.hypot(d[0], d[1]), -d[2]), 0, np.arctan2(d[1], d[0]) - np.pi / 2)
        cam.data.lens = v.get("lens", 24)
        sc.render.resolution_x, sc.render.resolution_y = v.get("res", (1600, 900))
    return cam


SAVE = __import__("os").environ.get("SAVE_BLEND")
for v in views:
    sc = reset()
    load_course(sc, v["window"])
    load_dress(sc, v["window"])
    objs = tree_objects(v.get("lod", 1))
    n = instance_trees(sc, objs, v.get("tree_window", v["window"]))   # 遠くを見る絵では、木だけ狭い範囲にできる
    cams = [make_cam(sc, w) for w in (views if SAVE else [v])]
    cam = cams[0]
    sc.camera = cam
    if SAVE:
        bpy.ops.file.pack_all()
        bpy.context.preferences.filepaths.file_preview_type = "NONE"   # .blend の保存でサムネイルを作らない（tests/test_blender_save_no_thumbnail.py）
        bpy.ops.wm.save_as_mainfile(filepath=SAVE)
        print("saved", SAVE, "trees", n, flush=True)
        break
    sc.render.filepath = str(out / "views" / f"{v['name']}.png")
    bpy.ops.render.render(write_still=True)
    print("rendered", v["name"], "trees", n, flush=True)
