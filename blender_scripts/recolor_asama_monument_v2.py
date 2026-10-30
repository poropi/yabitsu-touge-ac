"""既存の浅間石碑の材質だけを直し、別名保存する。ベイクは実行しない。"""
import hashlib
import json
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "export/compare/asama"
SOURCE = ROOT / "assets/landmarks/asama_monument/sengen_monument.blend"
DEST = SOURCE.with_name("sengen_monument_color_v2.blend")
col = next(c for c in bpy.context.scene.collection.children_recursive if c.name.startswith("01 "))


def snapshot():
    records = {}
    dg = bpy.context.evaluated_depsgraph_get()
    for o in col.all_objects:
        item = {"type": o.type, "matrix": [list(v) for v in o.matrix_world],
                "collections": sorted(c.name for c in o.users_collection),
                "material_slots": len(o.material_slots), "hidden": o.hide_render}
        if o.type in {"MESH", "FONT"}:
            e = o.evaluated_get(dg)
            me = e.to_mesh()
            me.calc_loop_triangles()
            coords = repr([tuple(v.co) for v in me.vertices]).encode()
            faces = repr([tuple(p.vertices) for p in me.polygons]).encode()
            item.update(triangles=len(me.loop_triangles),
                        geometry_sha256=hashlib.sha256(coords + faces).hexdigest())
            e.to_mesh_clear()
        records[o.name] = item
    return records


before = snapshot()
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
stone = next(s.material for o in col.all_objects if o.name.startswith("石碑本体")
             for s in o.material_slots if s.material)
panel = bpy.data.materials["02 碑文の磨き面"]


def palette(mat, low, high):
    ramp = next(n.color_ramp for n in mat.node_tree.nodes if n.type == "VALTORGB")
    ramp.elements[0].color = (*low, 1)
    ramp.elements[-1].color = (*high, 1)


# 線形RGB。中間値をsRGBへ戻すと石肌は約110/94/87。
# ノイズの空間配置・凹凸・既存オブジェクトは保持する。
palette(stone, (.062, .045, .038), (.25, .18, .15))
palette(panel, (.070, .055, .045), (.23, .17, .135))
stain = next(n for n in panel.node_tree.nodes if n.type == "MIX_RGB")
stain.inputs[2].default_value = (.18, .132, .095, 1)

# 同じ石材だった台座だけ、彩度の低い灰褐色へ分ける。
base = stone.copy()
base.name = "01 台座の灰褐色 v2"
palette(base, (.058, .049, .042), (.21, .18, .155))
for o in col.all_objects:
    if o.name.startswith(("上段", "下段")):
        for slot in o.material_slots:
            if slot.material == stone:
                slot.material = base

after = snapshot()
assert before == after, "材質以外の構成が変化した"
bpy.context.preferences.filepaths.file_preview_type = "NONE"   # .blend の保存でサムネイルを作らない（tests/test_blender_save_no_thumbnail.py）
bpy.ops.wm.save_as_mainfile(filepath=str(DEST), copy=True)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
report = {"source": str(SOURCE.relative_to(ROOT)), "destination": str(DEST.relative_to(ROOT)),
          "source_sha256": source_hash, "geometry_unchanged": before == after,
          "collection": col.name, "objects": after,
          "evaluated_triangles_all": sum(x.get("triangles", 0) for x in after.values()),
          "body_triangles_without_overlays": sum(x.get("triangles", 0) for n, x in after.items()
              if not any(k in n for k in ("本文", "額文字", "地衣類"))),
          "bake_performed": False,
          "note": "色は材質値。写真の照明から反射率を一意には復元できない。新しい焼付け後の画像平均は未測定。"}
(OUT / "monument2_material_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
result = {k:v for k,v in report.items() if k != "objects"}
