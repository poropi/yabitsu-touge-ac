"""依頼07の制作・色ベイク・上端v=0 NPZ書出し。写真素材は使わない。"""
from pathlib import Path
import json
import math
import shutil
import bpy
import numpy as np
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[1]
CMP=ROOT/'export/compare/pass'
CURRENT=None

def collection(name):
    global CURRENT
    c=bpy.data.collections.new(name);bpy.context.scene.collection.children.link(c);CURRENT=c
    return c

def move(o):
    for c in list(o.users_collection):c.objects.unlink(o)
    CURRENT.objects.link(o)
    return o

def mesh(name,verts,faces,mat):
    me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update()
    o=bpy.data.objects.new(name,me);CURRENT.objects.link(o)
    if mat:me.materials.append(mat)
    # 元の印刷画像はUVMapを明示的に参照。BakeUVとは別。
    if not me.uv_layers:me.uv_layers.new(name='UVMap')
    import bmesh
    bm=bmesh.new();bm.from_mesh(me)
    # 開いた面/両面葉/UV境界で分離されたNPZは、与えた巻き方向を保持する。
    if bm.edges and all(e.is_manifold for e in bm.edges):
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(me);bm.free()
    return o

def box(name,loc,size,mat,bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc)
    o=move(bpy.context.object);o.name=name;o.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if mat:o.data.materials.append(mat)
    if bevel:
        m=o.modifiers.new('角の摩耗','BEVEL');m.width=bevel;m.segments=1
        m=o.modifiers.new('面の法線','WEIGHTED_NORMAL');m.keep_sharp=True
    return o

def cylinder(name,loc,radius,depth,mat,vertices=12):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=radius,depth=depth,location=loc)
    o=move(bpy.context.object);o.name=name
    if mat:o.data.materials.append(mat)
    return o

def beam(name,a,b,width,mat,depth=None):
    a,b=Vector(a),Vector(b)
    o=box(name,(a+b)/2,(width,depth or width,(b-a).length),mat)
    o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler()
    return o

def pipe(name,a,b,radius,mat,vertices=8):
    a,b=Vector(a),Vector(b)
    o=cylinder(name,(a+b)/2,radius,(b-a).length,mat,vertices)
    o.rotation_euler=(b-a).to_track_quat('Z','Y').to_euler()
    return o

def proc(name,color,kind='stone',scale=5):
    m=bpy.data.materials.new(name);m.use_nodes=True
    n=m.node_tree.nodes;l=m.node_tree.links
    bs=next(x for x in n if x.type=='BSDF_PRINCIPLED');bs.inputs['Roughness'].default_value=.86
    def lin(x):return x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4
    color=[lin(float(c)/255 if c>1 else float(c)) for c in color[:3]]
    bs.inputs['Base Color'].default_value=(*color,1)
    m.diffuse_color=(*color,1)
    geo=n.new('ShaderNodeNewGeometry')
    mapping=n.new('ShaderNodeVectorMath');mapping.operation='MULTIPLY'
    mapping.inputs[1].default_value=(22,22,.4) if kind=='wood' else (1,1,1)
    l.new(geo.outputs['Position'],mapping.inputs[0])
    noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=scale;noise.inputs['Detail'].default_value=3.5
    l.new(mapping.outputs['Vector'],noise.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.15;ramp.color_ramp.elements[1].position=.85
    ramp.color_ramp.elements[0].color=(*(max(0,c*.65) for c in color),1)
    ramp.color_ramp.elements[1].color=(*(min(1,c*1.24+.005) for c in color),1)
    l.new(noise.outputs['Fac'],ramp.inputs[0]);l.new(ramp.outputs['Color'],bs.inputs['Base Color'])
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.20;bump.inputs['Distance'].default_value=.008 if kind=='stone' else .002
    l.new(noise.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs[0],bs.inputs['Normal'])
    return m

def image_material(name,path):
    m=bpy.data.materials.new(name);m.use_nodes=True
    bs=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bs.inputs['Roughness'].default_value=.82
    tex=m.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path),check_existing=True);tex.image.pack()
    uv=m.node_tree.nodes.new('ShaderNodeUVMap');uv.uv_map='UVMap'
    m.node_tree.links.new(uv.outputs[0],tex.inputs[0]);m.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
    return m

def front_panel(name,center,width,height,mat):
    x,y,z=center
    o=mesh(name,[(x-width/2,y,z-height/2),(x+width/2,y,z-height/2),(x+width/2,y,z+height/2),(x-width/2,y,z+height/2)],[(0,1,2,3)],mat)
    for li in o.data.polygons[0].loop_indices:
        co=o.data.vertices[o.data.loops[li].vertex_index].co
        o.data.uv_layers['UVMap'].data[li].uv=((co.x-x)/width+.5,(co.z-z)/height+.5)
    return o

def init_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene;s.unit_settings.system='METRIC';s.unit_settings.scale_length=1
    s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=16
    s.render.threads_mode='FIXED';s.render.threads=4
    s.view_settings.view_transform='Standard'
    w=bpy.data.worlds.new('空');w.use_nodes=True
    bg=next(n for n in w.node_tree.nodes if n.type=='BACKGROUND');bg.inputs[0].default_value=(.5,.6,.75,1);bg.inputs[1].default_value=.65;s.world=w
    return s

def evaluated_copy(objects,name):
    bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get();copies=[]
    for ob in objects:
        if ob.type not in ('MESH','FONT','CURVE'):continue
        me=bpy.data.meshes.new_from_object(ob.evaluated_get(dg),preserve_all_data_layers=True,depsgraph=dg)
        me.transform(ob.matrix_world)
        o=bpy.data.objects.new('_export_'+ob.name,me);bpy.context.scene.collection.objects.link(o);copies.append(o)
    bpy.ops.object.select_all(action='DESELECT')
    for o in copies:o.select_set(True)
    bpy.context.view_layer.objects.active=copies[0];bpy.ops.object.join();o=bpy.context.object;o.name=name
    return o

def bake_part(objects,out,name,res=1024):
    o=evaluated_copy(objects,name+'_baked');me=o.data
    # 画像ノード用UVMapは保持、ゲーム用のみ新規展開する。
    layer=me.uv_layers.new(name='BakeUV');me.uv_layers.active=layer;layer.active_render=True
    bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris();bpy.ops.uv.smart_project(angle_limit=1.1,island_margin=.008,area_weight=.2)
    bpy.ops.object.mode_set(mode='OBJECT')
    img=bpy.data.images.new(name+'_color',res,res,alpha=False);img.generated_color=(.32,.32,.29,1)
    for mat in me.materials:
        node=mat.node_tree.nodes.new('ShaderNodeTexImage');node.image=img;mat.node_tree.nodes.active=node
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=1
    s.render.bake.use_pass_direct=False;s.render.bake.use_pass_indirect=False;s.render.bake.use_pass_color=True
    s.render.bake.use_selected_to_active=False;s.render.bake.margin=8
    bpy.ops.object.bake(type='DIFFUSE')
    img.filepath_raw=str(out/(name+'.png'));img.file_format='PNG';img.save()
    me.calc_loop_triangles();uv=me.uv_layers['BakeUV'].data
    vs=[];us=[];fs=[];keys={}
    for t in me.loop_triangles:
        face=[]
        for li in t.loops:
            vi=me.loops[li].vertex_index;u=tuple(round(float(x),7) for x in uv[li].uv);normal=tuple(round(float(x),5) for x in t.normal);k=(vi,u,normal)
            if k not in keys:keys[k]=len(vs);vs.append(tuple(me.vertices[vi].co));us.append((u[0],1-u[1]))
            face.append(keys[k])
        fs.append(face)
    v=np.array(vs,np.float64);f=np.array(fs,np.int64);u=np.array(us,np.float64)
    np.savez_compressed(out/(name+'.npz'),v=v,f=f,uv=u)
    stats={'name':name,'triangles':len(f),'vertices':len(v),'dimensions_m':np.round(np.ptp(v,axis=0),4).tolist(),'bounds_m':[v.min(0).tolist(),v.max(0).tolist()]}
    bpy.data.objects.remove(o,do_unlink=True)
    return stats

def save_asset(asset_id,meta):
    src=ROOT/'assets/landmarks'/asset_id;out=ROOT/'data/processed/landmarks'/asset_id
    src.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    for im in bpy.data.images:
        if im.source=='FILE':im.pack()
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(src/(asset_id+'.blend')))
    stats=[]
    for name,objects in meta['parts'].items():
        stat=bake_part(objects,out,name,meta.get('resolution',1024))
        path=out/(name+'.npz')
        with np.load(path) as d:v,f,u=d['v'].copy(),d['f'],d['uv']
        offset=np.array([(v[:,0].min()+v[:,0].max())/2,(v[:,1].min()+v[:,1].max())/2,v[:,2].min()])
        v-=offset
        np.savez_compressed(path,v=v,f=f,uv=u)
        stat['assembly_offset_m']=offset.round(6).tolist()
        stat['bounds_m']=[v.min(0).tolist(),v.max(0).tolist()]
        stat.update(meta.get('part_info',{}).get(name,{}))
        stats.append(stat)
    info={k:v for k,v in meta.items() if k!='parts'}
    info.update(asset_id=asset_id,source=asset_id+'.blend',generator=meta.get('generator'),parts=stats,coordinates={'unit':'m','up':'+Z','front':'-Y','origin':'各NPZのXY外接矩形中央・最下点。建物の組立はassembly_offset_mを加算。制作元は共通組立座標。'},npz={'keys':['v','f','uv'],'uv':'v=0画像上端。Blender再読込時は1-v。','f':'triangles'},game_status='色焼付済みNPZ/PNG。配置・KN5・インストールは依頼範囲外。',source_image_usage='写真は比較専用。自作画像/手続き材質のみ使用。',bake='Cycles Diffuse COLORのみ。直接光/間接光は焼かない。',bake_docs='https://docs.blender.org/manual/en/latest/render/cycles/baking.html')
    (src/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print('ASSET_RESULT '+json.dumps(info,ensure_ascii=False),flush=True)
    return info
