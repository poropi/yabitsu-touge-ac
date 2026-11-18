"""依頼15の比較図に使う色見本。Blender CLIで制作元を開いて実行する。"""
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
PART = 'red_arch_bridge_v2'
DEST = ROOT / 'export/compare/route70/work/fubuki15/bridge'
scene = bpy.context.scene
for collection in scene.collection.children:
    collection.hide_render = collection.name != PART

material = bpy.data.materials[PART]
nodes = material.node_tree.nodes
links = material.node_tree.links
texture = nodes.new('ShaderNodeTexImage')
texture.image = bpy.data.images.load(str(ROOT / 'assets/landmarks/r70_fubuki' / (PART + '.png')))
texture.image.colorspace_settings.name = 'Non-Color'
# 8ビットRGB値に指定倍率を掛ける見本。物理照明・ゲームの露出は含めない。
geo = nodes.new('ShaderNodeNewGeometry')
dot = nodes.new('ShaderNodeVectorMath')
dot.operation = 'DOT_PRODUCT'
dot.inputs[1].default_value = Vector((.78, -.08, .62)).normalized()
links.new(geo.outputs['Normal'], dot.inputs[0])
weight = nodes.new('ShaderNodeMath')
weight.operation = 'MULTIPLY'
weight.inputs[1].default_value = 1.72
weight.use_clamp = True
links.new(dot.outputs['Value'], weight.inputs[0])
gain = nodes.new('ShaderNodeMath')
gain.operation = 'MULTIPLY_ADD'
gain.inputs[1].default_value = .62
gain.inputs[2].default_value = 1.13
links.new(weight.outputs[0], gain.inputs[0])
emission = nodes.new('ShaderNodeEmission')
links.new(texture.outputs['Color'], emission.inputs['Color'])
links.new(gain.outputs[0], emission.inputs['Strength'])
links.new(emission.outputs[0], next(n for n in nodes if n.type == 'OUTPUT_MATERIAL').inputs['Surface'])

scene.render.engine = 'CYCLES'
scene.cycles.samples = 4
scene.cycles.device = 'CPU'
scene.cycles.use_denoising = False
scene.view_settings.view_transform = 'Raw'
scene.view_settings.look = 'None'
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new('見本用の背景')
scene.world.use_nodes = True
next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND').inputs['Color'].default_value = (.78, .80, .79, 1)
next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND').inputs['Strength'].default_value = 1.
# 単体に床は含まれないため、撮影中だけ路面相当の無地の面を置く。
bpy.ops.mesh.primitive_plane_add(size=1, location=(1.45, 0, .70))
deck = bpy.context.object
deck.scale = (11.7, 99., 1.)
neutral = bpy.data.materials.new('見本用の仮路面')
neutral.use_nodes = True
surface = neutral.node_tree.nodes.new('ShaderNodeEmission')
surface.inputs['Color'].default_value = (.37, .38, .37, 1)
neutral.node_tree.links.new(surface.outputs[0], next(n for n in neutral.node_tree.nodes if n.type == 'OUTPUT_MATERIAL').inputs['Surface'])
deck.data.materials.append(neutral)

camera_data = bpy.data.cameras.new('単体比較')
camera = bpy.data.objects.new('単体比較', camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera_data.lens = 24
camera_data.clip_start = .05
scene.render.resolution_x, scene.render.resolution_y = 840, 446
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
DEST.mkdir(parents=True, exist_ok=True)
for name, pos, target in [('forward', (-1.5, -50, 3.2), (-1., -3., 3.3)),
                          ('backward', (-1.5, 43, 3.2), (-.9, -2., 3.2))]:
    camera.location = pos
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(DEST / (name + '.png'))
    bpy.ops.render.render(write_still=True)
