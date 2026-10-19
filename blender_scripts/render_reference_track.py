"""参考コース（nanohanadai）の一部を描いて見るだけの確認用。データは私たちのコースに使わない。
使い方: blender -b --factory-startup --python blender_scripts/render_reference_track.py -- <refdir> <views.json> <outdir>
views.json: [{"name", "center":[x,z(AC)], "r", "eye":[x,y,z], "target":[x,y,z], "lens", "top": bool, "size"}]（AC 座標・y が上）"""
import json
import pickle
import sys
from pathlib import Path

import bpy
import numpy as np

ref, views_path, outdir = (Path(a) for a in sys.argv[sys.argv.index("--") + 1:])
meshes = pickle.load(open(ref / "meshes.pkl", "rb"))
views = json.loads(views_path.read_text(encoding="utf-8"))
outdir.mkdir(parents=True, exist_ok=True)


def material(tex):
    key = tex or "none"
    m = bpy.data.materials.get(key)
    if m:
        return m
    m = bpy.data.materials.new(key); m.use_nodes = True
    nt = m.node_tree; bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.9
    p = ref / "tex" / tex if tex else None
    if p and p.exists():
        try:
            im = nt.nodes.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(str(p))
            nt.links.new(im.outputs[0], bsdf.inputs["Base Color"]); nt.links.new(im.outputs[1], bsdf.inputs["Alpha"])
        except Exception:
            pass
    m.use_backface_culling = False
    return m


for v in views:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    try:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        sc.render.engine = "BLENDER_EEVEE"
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    next(n for n in w.node_tree.nodes if n.type == "BACKGROUND").inputs[0].default_value = (0.6, 0.72, 0.88, 1)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.rotation_euler = (0.7, 0.2, 2.4); sun.data.energy = 3.5
    cx, cz = v["center"]; r = v["r"]
    for k, m in enumerate(meshes):
        p = m["p"]
        if not np.isfinite(p).all() or len(p) > 200000:
            continue
        if np.hypot(p[:, 0] - cx, p[:, 2] - cz).min() > r:
            continue
        verts = np.c_[p[:, 0], -p[:, 2], p[:, 1]]          # AC (x, y上, z) → Blender (x, -z, y)
        f = m["idx"].reshape(-1, 3)
        me = bpy.data.meshes.new(f"m{k}"); me.from_pydata(verts.tolist(), [], f.tolist())
        uvl = me.uv_layers.new(); u = m["uv"][f.ravel()]
        uvl.data.foreach_set("uv", np.c_[u[:, 0], 1 - u[:, 1]].ravel())
        me.materials.append(material(m["tex"]))
        ob = bpy.data.objects.new(me.name, me); sc.collection.objects.link(ob)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
    cam.data.clip_end = 5000
    if v.get("top"):
        cam.data.type = "ORTHO"; cam.data.ortho_scale = v["size"]
        cam.location = (cx, -cz, v.get("zc", 600) + 300); cam.rotation_euler = (0, 0, 0)
        sc.render.resolution_x = sc.render.resolution_y = 1000
    else:
        e = np.array([v["eye"][0], -v["eye"][2], v["eye"][1]]); t = np.array([v["target"][0], -v["target"][2], v["target"][1]])
        d = t - e
        cam.location = e.tolist()
        cam.rotation_euler = (np.arctan2(np.hypot(d[0], d[1]), -d[2]), 0, np.arctan2(d[1], d[0]) - np.pi / 2)
        cam.data.lens = v.get("lens", 22)
        sc.render.resolution_x, sc.render.resolution_y = 900, 600
    sc.camera = cam
    sc.render.filepath = str(outdir / f"{v['name']}.png")
    bpy.ops.render.render(write_still=True)
    print("rendered", v["name"], flush=True)
