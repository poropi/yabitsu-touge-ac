"""保存済み植物blendの撮影環境で、NPZ/PNGだけから復元した実体を描画する。

対象の_v2.blendを開き、末尾引数にvinesまたはgrassを指定。blendは保存しない。
"""
from pathlib import Path
import sys
import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'blender_scripts'))
from build_asama_small_assets import mesh_object, material

kind = sys.argv[-1]
assert kind in {'vines', 'grass'}
asset = 'asama_wall_vines' if kind == 'vines' else 'asama_grass_tufts'
folder = ROOT / 'data/processed/landmarks' / asset
checks = []
for i, letter in enumerate('abc'):
    scene = list(bpy.data.scenes)[i] if kind == 'vines' else bpy.data.scenes[f'grass_{letter}']
    bpy.context.window.scene = scene
    original = next(o for o in scene.objects if o.type == 'MESH')
    original.hide_render = True
    col = bpy.data.collections.new(f'01 NPZ確認_{kind}_{letter}')
    scene.collection.children.link(col)
    with np.load(folder / f'{kind}_{letter}.npz') as data:
        v, f, uv = (data[k] for k in ['v', 'f', 'uv'])
        obj = mesh_object(f'{kind}_{letter}_npz', v.tolist(), f.tolist(),
                          [[uv[j].tolist() for j in face] for face in f], col,
                          material(folder / f'{kind}_{letter}.png', f'{kind}_{letter}_game', True))
    target = Vector((0, 1.3, 3.3)) if kind == 'vines' else Vector((0, 0, [.58, .81, 1.06][i] * .44))
    scene.camera.data.ortho_scale = 5.8 if kind == 'vines' else [0.66, 0.84, 1.06][i] * 1.3
    scene.camera.location = target + Vector((4, -8, 2.2) if kind == 'vines' else (1, -5, 1.4))
    scene.camera.rotation_euler = (target - scene.camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.cycles.samples = 10
    scene.render.resolution_x = scene.render.resolution_y = 500
    scene.render.filepath = str(ROOT / f'export/compare/asama/{kind}2_npz_{i}.png')
    bpy.ops.render.render(write_still=True)
    checks.append(scene.render.filepath)
result = {'ok': True, 'npz_renders': checks, 'source_saved': False}
