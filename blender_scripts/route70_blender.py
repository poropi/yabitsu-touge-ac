"""ゲーム用データからBlender制作元を保存し、別プロセスで再読込・多方向描画。"""
from pathlib import Path
import json,sys,math
import numpy as np
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];CMP=ROOT/'export/compare/route70'

def material(name,path):
    mat=bpy.data.materials.new(name);nd=mat.node_tree.nodes;lk=mat.node_tree.links
    bs=next(n for n in nd if n.type=='BSDF_PRINCIPLED');bs.inputs['Roughness'].default_value=.7
    tex=nd.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path),check_existing=True);tex.image.pack();lk.new(tex.outputs['Color'],bs.inputs['Base Color']);lk.new(tex.outputs['Alpha'],bs.inputs['Alpha'])
    return mat

def make_blend(asset):
    bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;s.unit_settings.system='METRIC'
    src=ROOT/'assets/landmarks'/asset;out=ROOT/'data/processed/landmarks'/asset;info=json.loads((src/'asset.json').read_text())
    for pi,p in enumerate(info['parts']):
        n=p['name'];col=bpy.data.collections.new(n);s.collection.children.link(col)
        with np.load(out/(n+'.npz')) as d:v,f,uv=d['v'],d['f'],d['uv']
        mat=material(n,src/(n+'.png'))
        for co in p['components']:
            ff=f[co['triangle_start']:co['triangle_start']+co['triangle_count']]
            ids=np.unique(ff);remap={int(k):i for i,k in enumerate(ids)}
            me=bpy.data.meshes.new(co['name']);me.from_pydata(v[ids].tolist(),[],[[remap[int(i)] for i in q] for q in ff]);me.update();me.materials.append(mat)
            layer=me.uv_layers.new(name='UVMap')
            for poly in me.polygons:
                for li in poly.loop_indices:
                    u=uv[ids[me.loops[li].vertex_index]];layer.data[li].uv=(u[0],1-u[1])
            ob=bpy.data.objects.new(co['name'],me);col.objects.link(ob)
            ob['export_part']=n;ob['front']='-Y';ob['origin_note']=p['origin']
        col.hide_render=pi>0;col.hide_viewport=pi>0
    info['source_visibility']='各部品は同じローカル原点。最初のコレクション以外は非表示。部品を選んで表示切替する。'
    info['source_meshes']='印刷面/支柱/金具を命名した独立オブジェクト。NPZと同一三角形。'
    (src/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(src/(asset+'.blend')))
    return {'asset':asset,'parts':len(info['parts']),'objects':len(bpy.data.objects),'saved':str(src/(asset+'.blend'))}

def verify_source(asset):
    src=ROOT/'assets/landmarks'/asset;info=json.loads((src/'asset.json').read_text());report=[]
    for p in info['parts']:
        obs=[o for o in bpy.data.objects if o.get('export_part')==p['name']]
        assert len(obs)==len(p['components']),(p['name'],'構成数')
        assert all(len(o.data.vertices)>0 and len(o.data.polygons)>0 for o in obs),(p['name'],'空の構成部品')
        vs=np.array([tuple(o.matrix_world@v.co) for o in obs for v in o.data.vertices]);tris=sum(len(o.data.polygons) for o in obs)
        assert tris==p['triangles'] and np.allclose(np.ptp(vs,axis=0),p['dimensions_m'],atol=1e-5)
        assert all(o.data.materials and o.data.materials[0].node_tree for o in obs)
        report.append({'part':p['name'],'components':len(obs),'triangles':tris,'dimensions_m':np.ptp(vs,axis=0).tolist()})
    missing=[i.filepath for i in bpy.data.images if i.source=='FILE' and not i.packed_file and not Path(bpy.path.abspath(i.filepath)).exists()];assert not missing,missing
    result={'asset':asset,'passed':True,'missing_files':missing,'parts':report}
    (CMP/(asset+'_source_validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result

def render_parts(asset,names=None):
    info=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text());out=ROOT/'data/processed/landmarks'/asset
    bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=4;s.cycles.use_denoising=True;s.render.threads_mode='FIXED';s.render.threads=6
    s.render.resolution_x=320;s.render.resolution_y=320;s.render.resolution_percentage=100;s.view_settings.view_transform='Standard'
    w=bpy.data.worlds.new('確認用の空');s.world=w;w.use_nodes=True;bg=next(n for n in w.node_tree.nodes if n.type=='BACKGROUND');bg.inputs[0].default_value=(.65,.69,.74,1);bg.inputs[1].default_value=.8
    ld=bpy.data.lights.new('確認用の太陽','SUN');ld.energy=2.0;ld.angle=.3;sun=bpy.data.objects.new('確認用の太陽',ld);s.collection.objects.link(sun);sun.rotation_euler=(.5,-.6,-.3)
    cd=bpy.data.cameras.new('確認用');cd.type='ORTHO';cam=bpy.data.objects.new('確認用',cd);s.collection.objects.link(cam);s.camera=cam
    refdirs={'r70_curve_mirror':(-.35,-1,.04),'r70_utility_pole':(.2,-1,-.12),'r70_pylon':(1,-1,-.08),'r70_lights':(1,-.3,0),'r70_km_post':(.25,-1,-.04),'r70_delineator':(.4,-1,.12),'r70_road_signs':(-.4,-1,-.03),'r70_boards':(-.35,-1,.05),'r70_guardrail':(.9,-1,.12),'r70_fence':(.9,-1,.06),'r70_retaining_wall':(-1,-1,.3),'r70_drain':(.6,-1,1.5),'r70_construction':(.5,-1,.20),'r70_gate':(.4,-1,.15),'r70_misc':(.4,-1,.15)}
    views={'reference':refdirs[asset],'front':(0,-1,0),'left':(-1,0,0),'right':(1,0,0),'back_estimated':(0,1,0),'top':(0,0,1),'persp':(1,-1,.65)};done=[]
    for p in info['parts']:
        n=p['name']
        if names and n not in names:continue
        with np.load(out/(n+'.npz')) as d:v,f,u=d['v'],d['f'],d['uv']
        me=bpy.data.meshes.new(n);me.from_pydata(v.tolist(),[],f.tolist());me.update();ob=bpy.data.objects.new(n,me);s.collection.objects.link(ob);mat=material(n,out/(n+'.png'));me.materials.append(mat);ul=me.uv_layers.new(name='UVMap')
        for poly in me.polygons:
            for li in poly.loop_indices:uv=u[me.loops[li].vertex_index];ul.data[li].uv=(uv[0],1-uv[1])
        center=Vector((v.min(0)+v.max(0))/2);extent=max(np.ptp(v,axis=0))
        for vn,direction in views.items():
            vec=Vector(direction).normalized();cam.location=center+vec*(extent*3+1);cam.rotation_euler=(center-cam.location).to_track_quat('-Z','Y').to_euler()
            rot=cam.rotation_euler.to_matrix().transposed();projected=np.array([tuple(rot@(Vector(x)-center)) for x in v]);cd.ortho_scale=max(np.ptp(projected,axis=0)[:2])*1.18
            s.render.filepath=str(CMP/(asset+'_'+n+'_'+vn+'.png'));bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(ob,do_unlink=True);done.append(n)
    return {'asset':asset,'rendered':done}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--asset',required=True);ap.add_argument('--mode',choices=['build','verify','render'],default='build');arg=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    result=make_blend(arg.asset) if arg.mode=='build' else verify_source(arg.asset) if arg.mode=='verify' else render_parts(arg.asset)
    print(json.dumps(result,ensure_ascii=False))
