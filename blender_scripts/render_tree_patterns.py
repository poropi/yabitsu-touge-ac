"""木のパッケージ（zip）を並べて描画する確認用。
使い方: blender -b --factory-startup --python blender_scripts/render_tree_patterns.py -- <zip のフォルダ> <出力 png> [lod]
"""
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yabitsu import csptree  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = Path(argv[0]), Path(argv[1])
lod = int(argv[2]) if len(argv) > 2 else 0
tmp = Path(tempfile.mkdtemp())

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
try:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 2400, 900
world = bpy.data.worlds.new("w"); scene.world = world; world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs[0].default_value = (0.62, 0.74, 0.9, 1); bg.inputs[1].default_value = 0.9


def material(name, z, tex, alpha):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    p = tmp / tex; p.write_bytes(z.read(tex))
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nt = mat.node_tree; bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    im = nt.nodes.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(str(p))
    nt.links.new(im.outputs[0], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    if alpha:
        nt.links.new(im.outputs[1], bsdf.inputs["Alpha"])
        try:
            mat.blend_method = "CLIP"
        except Exception:
            pass
        mat.use_backface_culling = False
    return mat


x = 0.0
zips = sorted(src.glob("*.zip"))
for zp in zips:
    z = zipfile.ZipFile(zp)
    ini = z.read("tree.ini").decode()
    H = float(ini.split("HEIGHT=")[1].split()[0]); W = float(ini.split("WIDTH=")[1].split()[0])
    tex, subs = csptree.read_mesh(z.read(f"LOD{lod}.mesh"))
    x += W / 2 + 1.0
    for k, s in enumerate(subs):
        p = s.verts["p"].astype(float)
        v = np.c_[p[:, 0] * W / 2 + x, -p[:, 2] * W / 2, p[:, 1] * H]
        f = s.indices.reshape(-1, 3)
        me = bpy.data.meshes.new(f"{zp.stem}_{k}")
        me.from_pydata(v.tolist(), [], f.tolist())
        uvl = me.uv_layers.new()
        uv = s.verts["uv"][f.ravel()]
        uvl.data.foreach_set("uv", np.c_[uv[:, 0], 1 - uv[:, 1]].ravel())
        ob = bpy.data.objects.new(me.name, me); scene.collection.objects.link(ob)
        ob.data.materials.append(material(tex[s.diffuse][1], z, tex[s.diffuse][1], k == 0))
    x += W / 2
    print(zp.stem, "H", H, "W", W, "tris", sum(len(s.indices) // 3 for s in subs))

ground = bpy.data.meshes.new("g"); ground.from_pydata([(-5, -40, 0), (x + 5, -40, 0), (x + 5, 40, 0), (-5, 40, 0)], [], [(0, 1, 2, 3)])
g = bpy.data.objects.new("g", ground); scene.collection.objects.link(g)
gm = bpy.data.materials.new("gm"); gm.use_nodes = True
next(n for n in gm.node_tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"].default_value = (0.18, 0.15, 0.1, 1)
ground.materials.append(gm)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); scene.collection.objects.link(sun)
sun.rotation_euler = (0.9, 0.2, 0.6); sun.data.energy = 3.5
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); scene.collection.objects.link(cam)
cam.data.type = "ORTHO"; cam.data.ortho_scale = x + 4
cam.location = (x / 2, -100, 11); cam.rotation_euler = (np.radians(90), 0, 0)
scene.camera = cam
scene.render.filepath = str(out)
bpy.ops.render.render(write_still=True)
