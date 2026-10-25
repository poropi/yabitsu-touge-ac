"""Microsoft Rocketbox の人（FBX、両腕を広げた姿勢）を、腕を下ろした立ち姿で固めて npz に書き出す。
使い方: blender -b --factory-startup --python export_rocketbox.py -- <in.fbx> <out.npz>
出力: verts（ワールド座標、三角形の角ごとに1つ、Z が上、m）、faces、uv、mat（三角形ごとの材質番号）、mat_names。"""
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

src, out = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
obj = next(o for o in bpy.data.objects if o.type == "MESH")
for o in bpy.data.objects:
    if o.animation_data:
        o.animation_data.action = None
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
mw = arm.matrix_world
R = mw.to_quaternion()
# 上腕：体の脇から少し外・少し前へ下ろす。前腕：少しだけ曲げる（真っすぐな棒にしない）
# 骨の向きは「付け根 → 次の関節の付け根」で測る（FBX の骨の先端は腕の向きと合っていない）
for bone, child, down in (("UpperArm", "Forearm", Vector((0.17, 0.06, -1.0))), ("Forearm", "Hand", Vector((0.10, 0.18, -1.0)))):
    for side in ("L", "R"):
        bpy.context.view_layer.update()
        pb = arm.pose.bones[f"Bip01 {side} {bone}"]
        d = (mw @ arm.pose.bones[f"Bip01 {side} {child}"].head - mw @ pb.head).normalized()
        t = Vector((down.x * (1 if d.x > 0 else -1), down.y, down.z)).normalized()
        qa = R.inverted() @ d.rotation_difference(t) @ R
        M = pb.matrix.copy()
        pb.matrix = Matrix.Translation(M.to_translation()) @ qa.to_matrix().to_4x4() @ M.to_3x3().to_4x4()
bpy.context.view_layer.update()
bpy.ops.object.mode_set(mode="OBJECT")

dg = bpy.context.evaluated_depsgraph_get()
me = obj.evaluated_get(dg).to_mesh()
me.calc_loop_triangles()
m4 = np.array(obj.matrix_world)
co = np.array([v.co[:] for v in me.vertices]) @ m4[:3, :3].T + m4[:3, 3]
uvl = me.uv_layers.active.data
verts, uvs, mats = [], [], []
for t in me.loop_triangles:
    for li, vi in zip(t.loops, t.vertices):
        verts.append(co[vi]); uvs.append(uvl[li].uv[:])
    mats.append(t.material_index)
verts = np.array(verts, np.float32)
np.savez(out, verts=verts, faces=np.arange(len(verts), dtype=np.int32).reshape(-1, 3), uv=np.array(uvs, np.float32),
         mat=np.array(mats, np.int32), mat_names=np.array([m.name for m in obj.data.materials]))
print("EXPORTED", len(mats), "tris", verts.min(0), verts.max(0), [m.name for m in obj.data.materials])
