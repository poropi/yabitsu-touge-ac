"""Blender CLI。引数 source は制作元を保存、review は保存済みblendの検査と撮影。"""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'assets/landmarks/r70_fubuki'
OUT = ROOT / 'export/compare/route70/work/fubuki10'
INFO = json.loads((SRC/'asset.json').read_text())
BRIDGE = 'red_arch_bridge_v2'
WALL = 'wall_top_concrete_2m'
WIRE = 'slope_fence_wire_2m'
BAR = 'slope_fence_bar_2m'


def source():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    for p in INFO['parts']:
        name = p['name']
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bs = next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        bs.inputs['Roughness'].default_value = .89 if name==WALL else .57
        bs.inputs['Metallic'].default_value = .0 if name==WALL else .16
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(str(SRC/p['textures'][0]))
        tex.image.pack()
        tex.image.filepath = '//'+p['textures'][0]
        mat.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
        with np.load(SRC/(name+'.npz')) as d:
            v,f,uv = d['v'],d['f'],d['uv']
        components = defaultdict(list)
        for c in p['components']:
            # 白壁の小口も側面も1個の閉じたメッシュにする。
            label = '白壁' if name==WALL else c['name']
            start = c['triangle_start']
            components[label].extend(range(start,start+c['triangle_count']))
        for label, face_ids in components.items():
            indices = f[face_ids]
            coords, inverse = np.unique(v[indices].reshape(-1,3).round(8),axis=0,return_inverse=True)
            me = bpy.data.meshes.new(label)
            me.from_pydata(coords.tolist(),[],inverse.reshape(-1,3).tolist())
            me.update()
            me.materials.append(mat)
            layer = me.uv_layers.new(name='UVMap')
            mapped = uv[indices].reshape(-1,2).copy()
            mapped[:,1] = 1-mapped[:,1]
            layer.data.foreach_set('uv',mapped.astype('f').ravel())
            obj = bpy.data.objects.new(name+'.'+label,me)
            obj['export_part'] = name
            obj['component'] = label
            col.objects.link(obj)
        col.hide_viewport = col.hide_render = name!=BRIDGE
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                space = area.spaces.active
                space.region_3d.view_rotation = Vector((-1,.53,-.55)).to_track_quat('-Z','Y')
                space.region_3d.view_distance = 103
                space.region_3d.view_location = (1.45,0,3.8)
                space.shading.color_type = 'TEXTURE'
                space.overlay.show_floor = False
    bpy.context.preferences.filepaths.save_version = 0
    bpy.context.preferences.filepaths.file_preview_type = 'NONE'
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(SRC/'r70_fubuki.blend'))


def verify():
    results = {}
    for p in INFO['parts']:
        n = p['name']
        obs = list(bpy.data.collections[n].objects)
        with np.load(SRC/(n+'.npz')) as d:
            v,f,uv = d['v'],d['f'],d['uv']
        checked = 0
        for o in obs:
            assert tuple(o.location)==(0,0,0) and tuple(o.scale)==(1,1,1)
            mat = o.data.materials[0]
            texture = next(node for node in mat.node_tree.nodes if node.type=='TEX_IMAGE').image
            assert texture.packed_file and tuple(texture.size)==(1024,1024)
            ids = [k for c in p['components'] if n==WALL or c['name']==o['component']
                   for k in range(c['triangle_start'],c['triangle_start']+c['triangle_count'])]
            idx = f[ids]
            me = o.data
            actual = np.asarray([tuple(me.vertices[l.vertex_index].co) for l in me.loops])
            u = np.asarray([tuple(d.uv) for d in me.uv_layers.active.data])
            u[:,1] = 1-u[:,1]
            assert np.allclose(actual,v[idx].reshape(-1,3),rtol=0,atol=4e-6),(n,o.name,'頂点')
            assert np.allclose(u,uv[idx].reshape(-1,2),rtol=0,atol=1e-7),(n,o.name,'UV')
            checked += len(me.polygons)
        assert checked==p['triangles']
        results[n] = dict(triangles=checked,objects=len(obs),bounds_m=p['bounds_m'],
                          verified='全三角形の頂点・面順・UVをNPZと照合。画像をblend内に格納。')
    # 読み直した実座標で左右を個別検査。非対称拡幅・脚の取り違えを検出する。
    col = bpy.data.collections[BRIDGE]
    def points(label):
        o = next(o for o in col.objects if o['component']==label)
        return np.array([tuple(vertex.co) for vertex in o.data.vertices])
    curb = points('欄干の地覆')
    rib = points('床下へ降りるアーチリブ')
    splices = points('リブの継目板')
    full_rib = np.vstack([rib,splices])
    separation = []
    for side,center in [(-1,-4.4),(1,7.3)]:
        edge = curb[curb[:,0]*side>0,0]*side
        arch = full_rib[full_rib[:,0]*side>0,0]*side
        assert abs((edge.min()+edge.max())/2-abs(center))<1e-5
        clearance = arch.min()-edge.max()
        assert clearance>.28,(side, clearance)
        separation.append(round(float(clearance),4))
    band = points('脚位置の赤い短帯')
    locations = []
    for half in (-1,1):
        y = band[band[:,1]*half>0,1]
        locations.append([round(float(y.min()),3),round(float(y.max()),3)])
        assert abs(np.ptp(y)-5.1)<1e-5
    assert np.ptp(curb[:,1])==99 and np.allclose([curb[:,1].min(),curb[:,1].max()],[-49.5,49.5])
    assert rib[:,2].min()<points('灰色の床側桁')[:,2].min()
    results['bridge_checks'] = dict(rib_to_outside_curb_m=separation,band_ranges_local_y_m=locations,
        rib_min_z_m=round(float(rib[:,2].min()),4),deck_z_m=.7,road_center_x_m=0,rail_center_x_m=[-4.4,7.3])
    (OUT/'source_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')


def material(n, colour):
    mat = bpy.data.materials.new(n)
    bs = next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    bs.inputs['Base Color'].default_value = (*colour,1)
    bs.inputs['Roughness'].default_value = .92
    mat.diffuse_color = (*colour,1)
    return mat


def plane(name,x0,x1,y0,y1,z,mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(x0,y0,z),(x1,y0,z),(x1,y1,z),(x0,y1,z)],[],[(0,1,2,3)])
    o = bpy.data.objects.new(name,me)
    bpy.context.scene.collection.objects.link(o)
    me.materials.append(mat)
    return o


def review():
    verify()
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.device = 'CPU'
    s.cycles.samples = 24
    s.cycles.use_denoising = False
    s.render.threads_mode = 'FIXED'
    s.render.threads = 8
    s.view_settings.view_transform = 'AgX'
    s.render.film_transparent = True
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGBA'
    s.render.resolution_percentage = 100
    world = bpy.data.worlds.new('撮影用環境')
    s.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type=='BACKGROUND')
    bg.inputs[0].default_value = (.80,.84,.91,1)
    bg.inputs[1].default_value = .54
    ld = bpy.data.lights.new('撮影用日光','SUN')
    ld.energy = 2.8
    ld.angle = math.radians(7)
    lamp = bpy.data.objects.new(ld.name,ld)
    s.collection.objects.link(lamp)
    lamp.rotation_euler = (.45,-.8,-.32)
    cd = bpy.data.cameras.new('寸法比較用カメラ')
    cam = bpy.data.objects.new(cd.name,cd)
    s.collection.objects.link(cam)
    s.camera = cam
    cd.clip_start, cd.clip_end = .02, 500
    cd.sensor_fit = 'HORIZONTAL'
    cd.sensor_width = 36
    def render(name, pos, target, scale=None, lens=7.7, size=(1120,630)):
        cam.location = pos
        cam.rotation_euler = (Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()
        cd.type = 'ORTHO' if scale else 'PERSP'
        if scale: cd.ortho_scale = scale
        else: cd.lens = lens
        s.render.resolution_x,s.render.resolution_y = size
        s.render.filepath = str(OUT/(name+'.png'))
        bpy.ops.render.render(write_still=True)
    def show(*names):
        for p in INFO['parts']:
            col = bpy.data.collections[p['name']]
            col.hide_viewport = col.hide_render = col.name not in names
    def instance(n,offset):
        ob = bpy.data.objects.new('撮影用反復',None)
        ob.instance_type = 'COLLECTION'
        ob.instance_collection = bpy.data.collections[n]
        ob.location = offset
        s.collection.objects.link(ob)
        return ob
    def remove(obs):
        for ob in obs: bpy.data.objects.remove(ob,do_unlink=True)
    if 'directions' in sys.argv:
        # 同じ正投影カメラへ各部品の六方向を並べる。変更は撮影用シーンだけ。
        views = [(0,-1,0),(-1,0,0),(1,0,0),(0,1,0),(0,0,1),(1,-1,.58)]
        stage = []
        for p in INFO['parts']:
            col = bpy.data.collections[p['name']]
            objs = list(col.objects)
            col.hide_render = col.hide_viewport = True
            corners = [Vector(c) for o in objs for c in o.bound_box]
            for j,direction in enumerate(views):
                orient = Vector(direction).to_track_quat('Z','Y').inverted().to_matrix().to_4x4()
                if direction==(0,0,1) and p['name']==BRIDGE:
                    orient = Matrix.Rotation(math.pi/2,4,'Z') @ orient
                vs = np.array([tuple(orient@v) for v in corners])
                lo,hi = vs.min(0),vs.max(0)
                scale = min(.91/max(hi[0]-lo[0],.01),.72/max(hi[1]-lo[1],.01))
                row = INFO['parts'].index(p)
                transform = (Matrix.Translation((j-2.5,(2.5-row)*.86,0))
                             @ Matrix.Scale(scale,4) @ Matrix.Translation(Vector(-(hi+lo)/2)) @ orient)
                for o in objs:
                    clone = bpy.data.objects.new('確認用 '+o.name,o.data)
                    clone.matrix_world = transform
                    s.collection.objects.link(clone)
                    stage.append(clone)
        render('six_views',(0,0,45),(0,0,0),6,size=(1560,1342))
        remove(stage)
        return
    def repeat_panel(n, end):
        show(n,end)
        c = bpy.data.collections[end]
        # 支柱単体の原点を重ねず、撮影中だけ終端の位置へ移動。
        for o in c.objects: o.location.x=4
        extra = [instance(n,(2,0,0))]
        render(n,(4.6,-6.8,2.7),(2,0,.73),4.85,size=(1000,520))
        remove(extra)
        for o in c.objects: o.location.x=0
    repeat_panel(WIRE,'slope_fence_wire_end')
    repeat_panel(BAR,'slope_fence_bar_end')
    show(WALL)
    extra = [instance(WALL,(2*i,0,0)) for i in (1,2)]
    render('wall',(-1.5,-6,2.8),(2.8,0,.44),6.1,size=(1000,425))
    remove(extra)
    show(BRIDGE)
    render('bridge_axo',(91,-49,37),(1.45,0,3.2),99,size=(1600,550))
    render('bridge_side',(90,0,3.7),(1.45,0,3.7),109,size=(1600,300))
    render('bridge_plan',(1.45,0,125),(1.45,0,0),106,size=(750,750))
    render('bridge_below',(70,35,-28),(1.45,0,1),100,size=(1250,380))
    road = material('撮影用車道',(.10,.096,.092))
    walk = material('撮影用歩道',(.064,.056,.048))
    line = material('撮影用外側線',(.76,.76,.70))
    center = material('撮影用中央線',(.75,.48,.21))
    stage = [plane('仮路面',-4.56,7.45,-49.5,49.5,.7,road),
             plane('右の仮歩道',3.76,7.37,-49.5,49.5,.82,walk)]
    for x in (-3.55,3.55):
        stage.append(plane('外側線',x-.052,x+.052,-49.5,49.5,.705,line))
    stage.append(plane('中央線',-.056,.056,-49.5,49.5,.706,center))
    for key,y,heading,tilt in [('0975_f',-44,1,-.076),('0976_f',-19.4,1,.043),
                                ('0978_b',31,-1,-.018),('0978_f',31,1,-.076)]:
        z = 3.2
        render('view_'+key,(2.1,y,z),(2.1,y+heading*12,z+tilt*12),size=(960,600))
    remove(stage)


if __name__=='__main__':
    mode = sys.argv[sys.argv.index('--')+1]
    if mode=='source': source()
    elif mode in ('review','directions'): review()
    elif mode=='verify': verify()
    else: raise ValueError(mode)
