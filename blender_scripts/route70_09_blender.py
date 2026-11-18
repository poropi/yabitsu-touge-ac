"""依頼09の制作元保存・別プロセス検証・ゲームNPZ多方向描画。"""
from pathlib import Path
import sys,json,math
import numpy as np
import bpy
from mathutils import Vector
R=Path(__file__).resolve().parents[1];C=R/'export/compare/route70'
sys.path.insert(0,str(R/'blender_scripts'))
from route70_blender import make_blend,verify_source,material

def setup(size=320):
    bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene
    s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=3;s.cycles.use_denoising=True
    s.render.threads_mode='FIXED';s.render.threads=6;s.render.resolution_x=size;s.render.resolution_y=size;s.render.resolution_percentage=100
    s.render.image_settings.file_format='PNG';s.view_settings.view_transform='AgX'
    w=bpy.data.worlds.new('撮影用空');s.world=w;bg=w.node_tree.nodes.get('Background');bg.inputs[0].default_value=(.65,.7,.8,1);bg.inputs[1].default_value=.65
    ld=bpy.data.lights.new('撮影用太陽','SUN');ld.energy=2;ld.angle=.2;sun=bpy.data.objects.new(ld.name,ld);s.collection.objects.link(sun);sun.rotation_euler=(.4,-.6,-.3)
    cd=bpy.data.cameras.new('撮影用');cam=bpy.data.objects.new(cd.name,cd);s.collection.objects.link(cam);s.camera=cam
    return s,cam

def loadpart(asset,name,offset=(0,0,0),angle=0,scale=(1,1,1)):
    with np.load(R/'data/processed/landmarks'/asset/(name+'.npz')) as a:v,f,u=a['v'],a['f'],a['uv']
    co,si=math.cos(angle),math.sin(angle);rot=np.array([[co,-si,0],[si,co,0],[0,0,1]])
    v=(v*np.array(scale))@rot.T+offset
    me=bpy.data.meshes.new(name);me.from_pydata(v.tolist(),[],f.tolist());me.update()
    ob=bpy.data.objects.new(name,me);bpy.context.scene.collection.objects.link(ob)
    me.materials.append(material(name,R/'data/processed/landmarks'/asset/(name+'.png')))
    layer=me.uv_layers.new(name='UVMap')
    for face in me.polygons:
        for li in face.loop_indices:
            q=u[me.loops[li].vertex_index];layer.data[li].uv=(q[0],1-q[1])
    return ob,v

def frame(s,cam,vs,direction,path,perspective=False,camera=None,target=None):
    vv=np.concatenate(vs);center=Vector((vv.min(0)+vv.max(0))/2);ext=max(np.ptp(vv,axis=0))
    cam.location=Vector(camera) if camera else center+Vector(direction).normalized()*(ext*3+2)
    aim=Vector(target) if target else center
    cam.rotation_euler=(aim-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type='PERSP' if perspective else 'ORTHO';cam.data.lens=18;cam.data.sensor_fit='HORIZONTAL'
    if not perspective:
        rot=cam.rotation_euler.to_matrix().transposed();p=np.array([tuple(rot@(Vector(x)-center)) for x in vv]);cam.data.ortho_scale=max(np.ptp(p,axis=0)[:2])*1.18
    s.render.filepath=str(path);bpy.ops.render.render(write_still=True)

def render(asset,names=None):
    info=json.loads((R/'assets/landmarks'/asset/'asset.json').read_text());s,cam=setup(300)
    views={'front':(0,-1,0),'left':(-1,0,0),'right':(1,0,0),'back_estimated':(0,1,0),'top':(0,0,1),'persp':(1,-1,.65)}
    done=[]
    for p in info['parts']:
        if names and p['name'] not in names:continue
        ob,v=loadpart(asset,p['name'])
        for vn,d in views.items():frame(s,cam,[v],d,C/(asset+'_'+p['name']+'_'+vn+'.png'))
        bpy.data.objects.remove(ob,do_unlink=True);done.append(p['name'])
    return dict(asset=asset,rendered=done)

def bridges_reference():
    a='r70_bridges';info=json.loads((R/'assets/landmarks'/a/'asset.json').read_text());s,cam=setup(600);s.render.resolution_x=800;s.render.resolution_y=500
    data={p['name']:p for p in info['parts']};done=[]
    for i,l in enumerate(info['locations']):
        vs=[];obs=[];L=l['length_m'];width=8 if l['chainage_m']>9600 else 4.2
        full=l['left_rail'] if l['left_rail'] in ['aouji_bridge','red_arch_bridge'] else 'red_arch_bridge' if l['section']=='S5' and l['chainage_m']==10513 else None
        if full:
            ob,v=loadpart(a,full);vs.append(v);obs.append(ob)
            width=8.8 if full=='red_arch_bridge' else 4.4;L=data[full]['length_m'];zz=.7
        else:
            zz=0
            for side,key in [(-1,'left_rail'),(1,'right_rail')]:
                n=l[key];span=data[n]['repeat_length_m'];count=max(1,math.ceil(L/span));unit=L/count;LL=L
                for y in np.arange(count)*unit-LL/2+unit/2:
                    ob,v=loadpart(a,n,(side*width/2,float(y),0),math.pi/2,(unit/span,1,1));obs.append(ob);vs.append(v)
                if l['end_post']:
                    for y in [-LL/2,LL/2]:
                        ob,v=loadpart(a,l['end_post'],(side*width/2,y,0));obs.append(ob);vs.append(v)
            if l.get('right_outer_rail'):
                n=l['right_outer_rail'];span=data[n]['repeat_length_m']
                for y in np.arange(max(1,round(L/span)))*span-L/2+span/2:
                    ob,v=loadpart(a,n,(width/2+.65,float(y),0),math.pi/2);obs.append(ob);vs.append(v)
        # 撮影専用の仮路面。配布メッシュには含まない。
        me=bpy.data.meshes.new('確認専用の仮路面');me.from_pydata([(-width/2,-L/2-4,zz),(width/2,-L/2-4,zz),(width/2,L/2+4,zz),(-width/2,L/2+4,zz)],[],[(0,1,2,3)]);me.update()
        road=bpy.data.objects.new(me.name,me);s.collection.objects.link(road);obs.append(road)
        mat=bpy.data.materials.new('確認専用アスファルト');mat.diffuse_color=(.16,.17,.18,1)
        bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bs.inputs['Base Color'].default_value=(.12,.13,.14,1);bs.inputs['Roughness'].default_value=.9;me.materials.append(mat)
        # 写真がBACKなら宮ヶ瀬側から撮る。同じ側・視線方向、配置前なので遠近位置は推定。
        ref=next((r for r in info['references'] if r['oid'] in l['references']),None)
        backward=ref and '_b.' in ref['image'];sgn=1 if backward else -1
        camco=(0,sgn*(L/2+3),zz+2.4);aim=(0,sgn*(-L/2),zz+2.2)
        if l['section']=='S5' and l['chainage_m']==10588:camco=(0,30,zz+2.4)
        path=C/(f'r70_bridges_location_{l["section"]}_{l["chainage_m"]}_reference.png')
        frame(s,cam,vs,(0,-1,0),path,True,camco,aim)
        for ob in obs:bpy.data.objects.remove(ob,do_unlink=True)
        done.append(str(path))
    return done

def tunnel_references():
    a='r70_tunnels';info=json.loads((R/'assets/landmarks'/a/'asset.json').read_text());s,cam=setup(600);s.render.resolution_x=800;s.render.resolution_y=500;done=[]
    for l in info['locations']:
        for pn in [l['portal']]+([l['other_portal']] if l.get('other_portal') else []):
            obs=[];vs=[]
            ob,v=loadpart(a,pn);obs.append(ob);vs.append(v)
            ob,v=loadpart(a,l['lining'],(0,11.4,0),math.pi/2);obs.append(ob);vs.append(v)
            # 内部から撮られた0718だけ内側へカメラを置く。
            inside=pn=='s5_4091_portal'
            camera=(0,7.5,1.65) if inside else (0,-12,1.65)
            target=(0,-1,1.8) if inside else (0,4,2.2)
            path=C/(a+'_'+pn+'_reference.png');frame(s,cam,vs,(0,-1,0),path,True,camera,target)
            for ob in obs:bpy.data.objects.remove(ob,do_unlink=True)
            done.append(str(path))
    return done

def building_references(names=None):
    a='r70_buildings';info=json.loads((R/'assets/landmarks'/a/'asset.json').read_text());s,cam=setup(600);s.render.resolution_x=700;s.render.resolution_y=600;done=[]
    obs=json.loads((C/'work09/building_observations.json').read_text())
    for b in obs['buildings']:
        n=b['model_name']
        if names and n not in names:continue
        ob,v=loadpart(a,n)
        # 各写真が見せる面を合わせる。遮蔽物・周辺地形は単体制作の範囲外。
        for ri,ref in enumerate(b['references']):
            fn=Path(ref['image']).stem
            direction=(.7,-1,.2)
            if n=='s4_2080_brick_white':direction=(1,-.25,.06) if '0380' in fn else (.3,-1,.03)
            elif n=='s4_1550_water_hut':direction=(-.1,-1,.13) if '0359' in fn else (1,-.5,.13)
            elif n=='s4_2919_school':direction=(.65,-1,.03)
            elif n=='s5_11637_mokko_office':direction=(1,-.3,.1) if '1020' in fn else (.4,-1,.1)
            elif n=='s4_2120_cafe':direction=(-.3,-1,.08)
            elif n=='s5_7639_green_hut':direction=(.6,-1,1.0)
            path=C/(a+'_'+n+'_'+fn+'_reference.png');frame(s,cam,[v],direction,path)
            done.append(str(path))
        bpy.data.objects.remove(ob,do_unlink=True)
    return done

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['build','verify','render','bridge_refs']);ap.add_argument('asset',nargs='?',default='r70_bridges');args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    result=make_blend(args.asset) if args.mode=='build' else verify_source(args.asset) if args.mode=='verify' else bridges_reference() if args.mode=='bridge_refs' else render(args.asset)
    print(json.dumps(result,ensure_ascii=False))
