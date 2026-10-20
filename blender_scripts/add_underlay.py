import json
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
folder = Path(argv[0]).resolve(); save_as = argv[1] if len(argv) > 1 else None
coll = bpy.data.collections.new("UNDERLAY"); bpy.context.scene.collection.children.link(coll)
for item in json.loads((folder / "underlay.json").read_text(encoding="utf-8")):
    s = item["size_m"]
    me = bpy.data.meshes.new(item["file"])
    x0, y0, z = item["x0"], item["y0"], 900.0
    me.from_pydata([(x0, y0, z), (x0 + s, y0, z), (x0 + s, y0 + s, z), (x0, y0 + s, z)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(); uv.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    mat = bpy.data.materials.new(item["file"]); mat.use_nodes = True
    nt = mat.node_tree; bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(str(folder / item["file"]))
    nt.links.new(tex.outputs[0], bsdf.inputs["Base Color"]); me.materials.append(mat)
    ob = bpy.data.objects.new("UNDERLAY_" + item["file"], me); coll.objects.link(ob)
if save_as:
    # サンドボックス等でホームを取得できないとBlenderが文字化けしたサムネイル用フォルダを作るため、保存時のサムネイルを作らない。
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(save_as).resolve()))
