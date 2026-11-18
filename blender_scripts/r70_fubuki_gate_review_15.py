"""保存済みの制作元を開いたBlenderから依頼15の部品を撮影・照合する。"""
import hashlib
import json
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'assets/landmarks/r70_fubuki'
WORK = ROOT/'export/compare/route70/work/fubuki15'
POST, WALL = 'gate_brick_post', 'gate_brick_wall_2m'
parts = json.loads((SRC/'asset.json').read_text())['parts']

# 依頼10と同じ面順で、制作元のUVとNPZを全件照合する。
for part in parts:
    name = part['name']
    col = bpy.data.collections[name]
    col.hide_render = name not in (POST,WALL)
    col.hide_viewport = col.hide_render
    checked = 0
    with np.load(SRC/(name+'.npz')) as data:
        v, f, uv = data['v'],data['f'],data['uv']
        assert np.isfinite(uv).all() and uv.min()>=0 and uv.max()<=1
        for obj in col.objects:
            assert obj.matrix_world == Matrix.Identity(4)
            mesh = obj.data
            faces = [i for c in part['components']
                     if c['name']==obj['component'] or name=='wall_top_concrete_2m'
                     for i in range(c['triangle_start'],c['triangle_start']+c['triangle_count'])]
            ids = f[faces].reshape(-1)
            assert np.allclose([tuple(mesh.vertices[l.vertex_index].co) for l in mesh.loops],v[ids],rtol=0,atol=4e-6)
            mapped = np.array([list(d.uv) for d in mesh.uv_layers.active.data])
            mapped[:,1] = 1-mapped[:,1]
            assert np.allclose(mapped,uv[ids],rtol=0,atol=1e-7)
            mat = mesh.materials[0]
            image = next(node.image for node in mat.node_tree.nodes if node.type=='TEX_IMAGE')
            assert image.packed_file.data == (SRC/(name+'.png')).read_bytes()
            assert tuple(image.size)==(1024,1024)
            checked += len(mesh.polygons)
        assert checked==len(f)==part['triangles']
        if name==WALL:
            assert set(v[:,0])=={0,2}
            face0 = np.unique(v[v[:,0]==0,1:],axis=0)
            face2 = np.unique(v[v[:,0]==2,1:],axis=0)
            assert np.array_equal(face0,face2)
        if name=='red_arch_bridge_v2':
            assert hashlib.sha256((SRC/(name+'.npz')).read_bytes()).hexdigest()=='010b64e005f76370dfd72f35d29240f49dee06ff5346d8b9f5450d0e1c0dd5b6'
    print('照合済み',name,checked)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 12
scene.cycles.device = 'CPU'
scene.view_settings.view_transform = 'Standard'
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.resolution_percentage = 100
world = bpy.data.worlds.new('比較用環境')
scene.world = world
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type=='BACKGROUND')
bg.inputs[0].default_value = (.80,.86,1,1)
bg.inputs[1].default_value = .46
light = bpy.data.lights.new('日光','SUN')
light.energy = 2.15
light.angle = .28
sun = bpy.data.objects.new('日光',light)
scene.collection.objects.link(sun)
sun.rotation_euler = Vector((.8,.8,-1.35)).to_track_quat('-Z','Y').to_euler()
cam = bpy.data.objects.new('比較用カメラ',bpy.data.cameras.new('カメラ'))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width = 36


def render(name,w,h):
    scene.render.resolution_x, scene.render.resolution_y = w,h
    scene.render.filepath = str(WORK/(name+'.png'))
    bpy.ops.render.render(write_still=True)


def ortho(name,pos,target,span,w,h):
    cam.data.type = 'ORTHO'
    cam.data.shift_x = cam.data.shift_y = 0
    cam.location = pos
    cam.rotation_euler = (Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.ortho_scale = span
    if name == 'joined':
        bpy.context.view_layer.update()
        rot = cam.rotation_euler.to_matrix()
        points = [rot.transposed() @ (obj.matrix_world @ Vector(pt)-cam.location)
                  for group in (POST,WALL) for obj in bpy.data.collections[group].objects
                  for pt in obj.bound_box]
        lo,hi = np.min(points,axis=0),np.max(points,axis=0)
        cam.location += rot @ Vector(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,0))
        cam.data.ortho_scale = max(hi[0]-lo[0], (hi[1]-lo[1])*w/h)*1.14
    render(name,w,h)


# R70-0971_fの正面4隅を使って推定した透視カメラ。元画像の同じ矩形を切り出す。
cam.location = (6.17683022,-7.38231478,2.00365513)
right = (.25781065,.96607939,.01497608)
up = (-.01569219,-.01131135,.99981289)
back = (.96606802,-.25799742,.01224371)
cam.rotation_euler = Matrix((right,up,back)).transposed().to_euler()
cam.data.lens = 800*36/350
cam.data.shift_x = (1371-800)/350
cam.data.shift_y = (500-579)/350
bpy.data.collections[WALL].hide_render = True
bg.inputs[1].default_value = .26
light.energy = 2.7
render('gate_photo_angle',700,676)
bg.inputs[1].default_value = .46
light.energy = 2.15

# 各方向は同じ原点のまま、コレクションの表示だけを切り替える。
for name, pos in [('front',(0,-8,1.25)),('right',(8,0,1.25)),('left',(-8,0,1.25)),
                  ('back',(0,8,1.25)),('top',(0,0,9))]:
    ortho(name,pos,(0,0,1.25),2.9,336,365)

wall_col = bpy.data.collections[WALL]
wall_col.hide_render = False
for obj in list(wall_col.objects):
    obj.location.x = .60
    second = obj.copy()
    wall_col.objects.link(second)
    second.location.x += 2
ortho('joined',(5.9,-10,5.45),(1.95,0,1.29),7.1,960,488)
