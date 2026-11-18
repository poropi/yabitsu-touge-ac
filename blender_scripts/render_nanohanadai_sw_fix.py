"""生成済みの局所メッシュをBlenderで描く。全コースの組立・ゲーム導入は行わない。"""
from pathlib import Path
import json,sys,math
import bpy,numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'export/compare/nanohanadai';WORK=OUT/'work/sw_fix'
D=json.loads((WORK/'scene.json').read_text());spec=D['spec'];z=D['pad']['z'];axis=math.radians(spec['axis_deg']);U=np.array([math.cos(axis),math.sin(axis)]);N=np.array([-U[1],U[0]]);O=np.array(spec['origin'])
def xy(a,b):return O+a*U+b*N

def material(name,m):
 mat=bpy.data.materials.get(name)
 if mat:return mat
 mat=bpy.data.materials.new(name);mat.use_nodes=True;bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bs.inputs['Roughness'].default_value=.88
 if m.get('texture') and Path(m['texture']).exists():
  tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(m['texture']),check_existing=True);mat.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
  if m.get('alpha'):mat.node_tree.links.new(tex.outputs['Alpha'],bs.inputs['Alpha'])
 elif m.get('color'):
  col=m['color'];col=tuple(int(col.lstrip('#')[i:i+2],16)/255 for i in (0,2,4)) if isinstance(col,str) else col;bs.inputs['Base Color'].default_value=(*col[:3],1)
 return mat

def setup():
 bpy.ops.wm.read_factory_settings(use_empty=True);sc=bpy.context.scene;sc.render.engine='BLENDER_EEVEE';sc.eevee.taa_render_samples=16;sc.render.threads_mode='FIXED';sc.render.threads=4;sc.view_settings.view_transform='Standard'
 world=bpy.data.worlds.new('空');world.use_nodes=True;None
 next(n for n in world.node_tree.nodes if n.type=='BACKGROUND').inputs[0].default_value=(.55,.68,.85,1);next(n for n in world.node_tree.nodes if n.type=='BACKGROUND').inputs[1].default_value=.7;sc.world=world
 sun=bpy.data.objects.new('sun',bpy.data.lights.new('sun','SUN'));sc.collection.objects.link(sun);sun.rotation_euler=(math.radians(40),math.radians(15),math.radians(140));sun.data.energy=2.2;sun.data.angle=math.radians(2)
 mats={**D['old_materials'],**D['materials']}
 for e in D['old_meshes']+D['meshes']:
  d=np.load(e['file']);v=d['verts'];f=d['faces'];me=bpy.data.meshes.new(e['name']);me.from_pydata(v.tolist(),[],f.tolist());me.uv_layers.new().data.foreach_set('uv',d['uvs'][f.ravel()].ravel());me.materials.append(material(e['material'],mats[e['material']]))
  ob=bpy.data.objects.new(e['name'],me);sc.collection.objects.link(ob)
 return sc

def cam(sc,h):
 c=bpy.data.objects.new(f'sw_{h:03d}',bpy.data.cameras.new(f'sw_{h:03d}'));sc.collection.objects.link(c);pos=xy(-64.8442,3.8390);c.location=(*pos,z+2.5);r=math.radians(h-56.1);d=math.cos(r)*U-math.sin(r)*N
 pitch=math.radians(4.13);direction=Vector((d[0]*math.cos(pitch),d[1]*math.cos(pitch),-math.sin(pitch)));c.rotation_euler=direction.to_track_quat('-Z','Y').to_euler();c.data.sensor_fit='VERTICAL';c.data.sensor_height=32;c.data.lens=16;c.data.clip_end=20000;sc.camera=c;sc.render.resolution_x=1500;sc.render.resolution_y=1000;sc.render.resolution_percentage=100;return c

def run(views=(135,180,0),save=False):
 sc=setup()
 for h in views:cam(sc,h)
 if save:
  bpy.context.preferences.filepaths.file_preview_type = "NONE"
  bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'nanohanadai_sw_fix_scene.blend'))
 for h in views:
  sc.camera=bpy.data.objects[f'sw_{h:03d}'];sc.render.filepath=str(OUT/f'nanohanadai_sw_fix_sw_{h:03d}.png');bpy.ops.render.render(write_still=True)
 # 真上：航空写真と同じアフィン変換で照合するため、高解像度の北上正射影を保存。
 if save:
  c=bpy.data.objects.new('top',bpy.data.cameras.new('top'));sc.collection.objects.link(c);center=xy(-65,8);c.location=(*center,z+300);c.rotation_euler=(0,0,0);c.data.type='ORTHO';c.data.ortho_scale=100;sc.camera=c;sc.render.resolution_x=1600;sc.render.resolution_y=1600;sc.render.filepath=str(OUT/'nanohanadai_sw_fix_top.png');bpy.ops.render.render(write_still=True)
if __name__=='__main__':run()
