"""納品NPZ/PNGを再読込し、単体の全方向と指定写真方向を撮影する。"""
from pathlib import Path
import math
import json
import sys
import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
CMP=ROOT/'export/compare/nanohanadai'
OUT=ROOT/'data/processed/landmarks/nanohanadai_tower'


def setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene
    s.render.engine='CYCLES'
    s.cycles.device='CPU'
    s.cycles.samples=16
    s.cycles.use_denoising=True
    s.render.threads_mode='FIXED'
    s.render.threads=6
    s.view_settings.view_transform='AgX'
    s.render.image_settings.file_format='PNG'
    s.render.film_transparent=True
    w=bpy.data.worlds.new('撮影用の空')
    w.use_nodes=True
    w.node_tree.nodes.get('Background').inputs[0].default_value=(.7,.78,.9,1)
    w.node_tree.nodes.get('Background').inputs[1].default_value=.65
    s.world=w
    tex=bpy.data.images.load(str(OUT/'frame.png'))
    mat=bpy.data.materials.new('納品PNGの色')
    bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    bs.inputs['Roughness'].default_value=.85
    image=mat.node_tree.nodes.new('ShaderNodeTexImage')
    image.image=tex
    mat.node_tree.links.new(image.outputs['Color'],bs.inputs['Base Color'])
    for name in ('frame','stairs','deck','roof','base'):
        with np.load(OUT/(name+'.npz')) as d:
            v,f,u=d['v'],d['f'],d['uv']
        me=bpy.data.meshes.new(name)
        me.from_pydata(v,[],f)
        me.update()
        me.materials.append(mat)
        ob=bpy.data.objects.new(name,me)
        s.collection.objects.link(ob)
        layer=me.uv_layers.new(name='UVMap')
        for poly in me.polygons:
            for li in poly.loop_indices:
                uv=u[me.loops[li].vertex_index]
                layer.data[li].uv=(uv[0],1-uv[1])
    ld=bpy.data.lights.new('太陽','SUN')
    ld.energy=2.0
    ld.angle=math.radians(12)
    light=bpy.data.objects.new('太陽',ld)
    s.collection.objects.link(light)
    light.rotation_euler=(math.radians(23),math.radians(-28),math.radians(-35))
    cd=bpy.data.cameras.new('確認カメラ')
    cam=bpy.data.objects.new('確認カメラ',cd)
    s.collection.objects.link(cam)
    s.camera=cam
    return s,cam


def aim(cam,pos,target):
    cam.location=pos
    cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()


def render(mode='all'):
    s,cam=setup()
    views={
        'south':dict(pos=(0,-26,6.2),target=(0,0,6.0)),
        'southeast':dict(pos=(19,-19,6.2),target=(0,0,6.0)),
        'east':dict(pos=(26,0,6.2),target=(0,0,6.0)),
        'northeast_estimated':dict(pos=(19,19,6.2),target=(0,0,6.0)),
        'north_estimated':dict(pos=(0,26,6.2),target=(0,0,6.0)),
        'northwest_estimated':dict(pos=(-19,19,6.2),target=(0,0,6.0)),
        'west':dict(pos=(-26,0,6.2),target=(0,0,6.0)),
        'southwest':dict(pos=(-19,-19,6.2),target=(0,0,6.0)),
        'top':dict(pos=(0,0,35),target=(0,0,0),scale=11.2),
        'persp':dict(pos=(-18,-23,17),target=(0,0,6)),
        'deck_detail':dict(pos=(1.6,-2.1,10.85),target=(-2,1,10.8),perspective=True,lens=19),
        'stairs_detail':dict(pos=(.69,-1.66,4.9),target=(1.5,.5,5.5),perspective=True,lens=17),
        'roof_detail':dict(pos=(-6,-7,16),target=(0,0,12.4),scale=10),
        'base_detail':dict(pos=(7,-13,5),target=(0,-1.0,-.15),scale=10),
        'reference_yakei':dict(pos=(-23,-23,1.50),target=(0,0,6.0),perspective=True,lens=55),
        'reference_0rX1Rg_045':dict(pos=(-5.2,-8.6,2.7),target=(0,0,6.0),perspective=True,lens=20),
        'reference_0rXDBQ_345':dict(pos=(6.2,-21,2.4),target=(0,0,6.0),perspective=True,lens=27),
        'reference_4_znQE_015':dict(pos=(2.7,-12.8,.8),target=(0,0,6.0),perspective=True,lens=23),
    }
    if mode=='draft':
        views={k:views[k] for k in ['reference_yakei','top','stairs_detail']}
        s.cycles.samples=6
    if mode=='details':
        views={k:views[k] for k in ['deck_detail','stairs_detail']}
    poses={}
    for name,view in views.items():
        s.render.resolution_x=960
        s.render.resolution_y=1200
        if name in ('top','deck_detail','stairs_detail','roof_detail','base_detail'):
            s.render.resolution_x=s.render.resolution_y=1000
        if mode=='draft':
            s.render.resolution_x//=2
            s.render.resolution_y//=2
        s.render.resolution_percentage=100
        aim(cam,view['pos'],view['target'])
        cam.data.type='PERSP' if view.get('perspective') else 'ORTHO'
        cam.data.lens=view.get('lens',32)
        cam.data.sensor_width=36
        cam.data.ortho_scale=view.get('scale',16.0)
        cam.data.clip_start=.03
        cam.data.clip_end=300
        suffix='draft_'+name if mode=='draft' else name
        s.render.filepath=str(CMP/('nanohanadai_tower_'+suffix+'.png'))
        bpy.ops.render.render(write_still=True)
        poses[name]=view
    if mode=='all':
        # 航空写真では屋根と地面に視差があるため、屋根外形だけを別に重ねる。
        for name in ('frame','stairs','deck','base'):
            bpy.data.objects[name].hide_render=True
        aim(cam,(0,0,35),(0,0,0))
        cam.data.type='ORTHO'
        cam.data.ortho_scale=11.2
        s.render.resolution_x=s.render.resolution_y=1000
        s.render.filepath=str(CMP/'nanohanadai_tower_roof_top.png')
        bpy.ops.render.render(write_still=True)
    mode_prefix='' if mode=='all' else mode+'_'
    (CMP/('nanohanadai_tower_'+mode_prefix+'cameras.json')).write_text(json.dumps(poses,indent=2)+'\n')
    return dict(rendered=list(views))


if __name__=='__main__':
    render('draft' if '--draft' in sys.argv else 'all')
