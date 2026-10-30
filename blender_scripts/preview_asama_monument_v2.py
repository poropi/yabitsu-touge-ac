"""単体の照明比較と六方向確認。読込元blendは保存しない。ベイクもしない。"""
import json
import math
from pathlib import Path
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'export/compare/asama'
scene = bpy.context.scene
is_new = Path(bpy.data.filepath).stem.endswith('color_v2')
col = next(c for c in scene.collection.children_recursive if c.name.startswith('01 '))
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 12
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = True
# 元の制作シーンと同一の照明・AgXで旧/新を比較する。
original_camera = scene.camera
scene.render.filepath = str(OUT / ('monument2_shaded_after.png' if is_new else 'monument2_shaded_before.png'))
bpy.ops.render.render(write_still=True)

if is_new:
    for o in scene.objects:
        if o.type in {'MESH', 'FONT', 'CURVE'} and o not in list(col.all_objects):
            o.hide_render = True
    center = Vector((0, 0, 1.65))
    dirs = {'front': (0,-1,0), 'left': (-1,0,0), 'right': (1,0,0),
            'back': (0,1,0), 'top': (0,0,1), 'persp': (.28,-1,.18)}
    cam = bpy.data.objects.new('_monument2_inspection', bpy.data.cameras.new('_monument2_inspection'))
    scene.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 3.85
    scene.camera = cam
    for name, d in dirs.items():
        cam.location = center + Vector(d).normalized() * 9
        cam.rotation_euler = (center - cam.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath = str(OUT / f'monument2_{name}.png')
        bpy.ops.render.render(write_still=True)

    # Base Colorの検査用。Emissionはこのプロセス内のみで、保存元の材質は変えない。
    mats = {s.material for o in col.all_objects for s in o.material_slots if s.material}
    for mat in mats:
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        output = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
        emission = nt.nodes.new('ShaderNodeEmission')
        if bsdf.inputs['Base Color'].is_linked:
            nt.links.new(bsdf.inputs['Base Color'].links[0].from_socket, emission.inputs['Color'])
        else:
            emission.inputs['Color'].default_value = bsdf.inputs['Base Color'].default_value
        nt.links.new(emission.outputs[0], output.inputs['Surface'])
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    cam.location = (0, -9, 1.65)
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath = str(OUT / 'monument2_basecolor_front.png')
    bpy.ops.render.render(write_still=True)
result = {'saved_source_unchanged': True, 'new': is_new, 'render_directory': str(OUT)}
