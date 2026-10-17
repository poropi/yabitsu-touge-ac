import argparse
import json
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

FBX_SETTINGS = dict(  # Task 2 の試験で確かめる。変えたら docs/ac-conventions.md に記録する
    apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE", axis_forward="-Z", axis_up="Y",
    object_types={"MESH", "EMPTY"}, mesh_smooth_type="FACE", use_mesh_modifiers=True,
    path_mode="COPY", embed_textures=False, bake_space_transform=False)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--group", required=True)
    ap.add_argument("--blend"); ap.add_argument("--fbx")
    return ap.parse_args(argv)


def make_material(name, spec, root):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    if spec.get("texture"):
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str((root / spec["texture"]).resolve()))
        nt.links.new(tex.outputs[0], bsdf.inputs["Base Color"])
        if spec.get("alpha"):
            nt.links.new(tex.outputs[1], bsdf.inputs["Alpha"])
    elif spec.get("color"):
        bsdf.inputs["Base Color"].default_value = (*spec["color"], 1.0)
    mat.use_backface_culling = not spec.get("two_sided", False)
    return mat


def make_mesh(name, path, mat, coll):
    d = np.load(path)
    me = bpy.data.meshes.new(name)
    me.from_pydata(d["verts"].tolist(), [], d["faces"].tolist())
    me.validate()
    uvl = me.uv_layers.new(name="UVMap")
    vi = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    uvl.data.foreach_set("uv", d["uvs"][vi].astype(np.float32).ravel())
    me.materials.append(mat)
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def main():
    a = args()
    mpath = Path(a.manifest).resolve(); root = mpath.parent
    m = json.loads(mpath.read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"; scene.unit_settings.scale_length = 1.0
    # --group は「,」で区切って複数指定できる（確認用の描画で町の kn5 の部品もいっしょに見るため）
    groups = a.group.split(",")
    g = {"meshes": [n for gg in groups for n in m["groups"][gg]["meshes"]],
         "empties": [e for gg in groups for e in m["groups"][gg]["empties"]]}
    coll = bpy.data.collections.new(groups[0]); scene.collection.children.link(coll)
    mats = {}
    for name in g["meshes"]:
        info = m["meshes"][name]
        if info["material"] not in mats:
            mats[info["material"]] = make_material(info["material"], m["materials"][info["material"]], root)
        ob = make_mesh(name, root / info["file"], mats[info["material"]], coll)
        ob["ac_renderable"] = info["renderable"]
    for key in g["empties"]:
        e = m["empties"][key]
        ob = bpy.data.objects.new(e["name"], None)
        ob.empty_display_type = "ARROWS"; ob.empty_display_size = 2.0
        ob.location = e["location"]; ob.rotation_euler = (0.0, 0.0, e["rotation_z"])
        coll.objects.link(ob)
    names = [o.name for o in coll.objects]
    expected = [m["empties"][k]["name"] for k in g["empties"]] + g["meshes"]
    if sorted(names) != sorted(expected):
        raise RuntimeError(f"Blender が名前を変えました: {set(names) ^ set(expected)}")
    print(f"BUILT group={a.group} meshes={len(g['meshes'])} empties={len(g['empties'])}")
    if a.blend:
        Path(a.blend).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.blend).resolve()))
    if a.fbx:
        Path(a.fbx).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.fbx(filepath=str(Path(a.fbx).resolve()), use_selection=False, **FBX_SETTINGS)
        # ksEditor は FBX の隣の texture/ からテクスチャを探す（.fbm は見ない）
        tex_dir = Path(a.fbx).resolve().parent / "texture"; tex_dir.mkdir(exist_ok=True)
        for img in bpy.data.images:
            if img.filepath:
                shutil.copy2(bpy.path.abspath(img.filepath), tex_dir / Path(img.filepath).name)
        print(f"FBX {a.fbx}")


main()
