"""依頼04: 別プロセスで保存済みblend/ゲームNPZを再読込する。"""
import bpy,sys,json,math
from pathlib import Path
import numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'blender_scripts'))
import nanohanadai_common as C
CMP=C.CMP;SRC=ROOT/'assets/landmarks/nanohanadai_toilet';OUT=ROOT/'data/processed/landmarks/nanohanadai_toilet'
NAMES=['ramp_walls_sw_v2','ramp_rails_sw_v2','ramp_sw_approach']

def validate():
    import bmesh,hashlib
    meta=json.loads((SRC/'ramp_sw_approach.asset.json').read_text());checks=[]
    bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
    for stat in meta['parts']:
        name=stat['name']
        with np.load(OUT/(name+'.npz')) as d:
            assert set(d.files)=={'v','f','uv'};v,f,u=d['v'],d['f'],d['uv']
        assert v.shape[1]==3 and f.shape[1]==3 and u.shape==(len(v),2)
        assert np.isfinite(v).all() and np.isfinite(u).all()
        assert f.min()>=0 and f.max()<len(v) and len(v)<65535
        assert u.min()>=0 and u.max()<=1
        area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
        assert area.min()>1e-10,(name,area.min())
        assert len(f)==stat['triangles'] and np.allclose(np.ptp(v,axis=0),stat['dimensions_m'],atol=1e-4)
        if name=='ramp_sw_approach':assert len(f)<=3000
        col=next(c for c in bpy.data.collections if name in c.name)
        source_tri=0;vol=0;closed=0;bounds=[]
        for o in col.objects:
            me=bpy.data.meshes.new_from_object(o.evaluated_get(dg),depsgraph=dg);me.transform(o.matrix_world);me.calc_loop_triangles();source_tri+=len(me.loop_triangles)
            bounds.extend(tuple(x.co) for x in me.vertices)
            bm=bmesh.new();bm.from_mesh(me)
            assert all(e.is_manifold for e in bm.edges),(name,o.name,'open')
            closed+=1;vol+=bm.calc_volume(signed=False);bm.free();bpy.data.meshes.remove(me)
        assert source_tri==len(f),(name,source_tri,len(f))
        assert np.allclose(np.min(bounds,axis=0),v.min(0),atol=1e-5) and np.allclose(np.max(bounds,axis=0),v.max(0),atol=1e-5)
        if name!='ramp_sw_approach':
            oldname=name.replace('_sw_v2','')
            with np.load(OUT/(oldname+'.npz')) as d:oldv=d['v']
            a=np.unique(np.round(oldv[oldv[:,0]>=-1.5],5),axis=0);b=np.unique(np.round(v[v[:,0]>=-1.5],5),axis=0)
            assert a.shape==b.shape and np.array_equal(a,b),(name,'central/NE geometry changed')
        checks.append(dict(stat,source_triangles=source_tri,source_closed_objects=closed,source_volume_m3=vol,min_area_m2=float(area.min()),uv_range=[u.min(0).tolist(),u.max(0).tolist()]))
    # 共通境界で床のX/Y/Z、壁頂の高さが一致することを頂点/レイで確認。
    def load(name):
        with np.load(OUT/(name+'.npz')) as d:return d['v'],d['f']
    from mathutils.bvhtree import BVHTree
    def height(name,x,y):
        v,f=load(name);bvh=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
        hit=bvh.ray_cast(Vector((x,y,3)),Vector((0,0,-1)),4);return hit[0].z if hit[0] is not None else None
    seams=[]
    for y in [-4.80,-4.20,-3.50]:
        old=height('ramp_walls_sw_v2',-8.899,y);new=height('ramp_sw_approach',-8.901,y)
        assert old is not None and new is not None and abs(old-new)<.002,(y,old,new)
        seams.append({'y':y,'v2_z':old,'approach_z':new,'gap_m':abs(old-new)})
    originals=json.loads((CMP/'work/request04_before_hashes.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in originals.items())
    result={'passed':True,'blend_reopened':bpy.data.filepath,'parts':checks,'floor_seam_checks':seams,'source_originals_unchanged':True,'original_files_checked':len(originals),'game_integration':'依頼範囲外、未実施'}
    (CMP/'nanohanadai_toilet_sw_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def load_game(context=False):
    C.init_scene();col=C.collection('01 書き出したNPZの再読込')
    names=NAMES+(['walls','metal_roof','wood_trusses','roof_rails','stone_base'] if context else [])
    for name in names:
        with np.load(OUT/(name+'.npz')) as d:v,f,u=d['v'],d['f'],d['uv']
        mat=C.image_material(name,OUT/(name+'.png'));ob=C.mesh(name,v.tolist(),f.tolist(),mat)
        for p in ob.data.polygons:
            for li in p.loop_indices:
                uv=u[ob.data.loops[li].vertex_index];ob.data.uv_layers['UVMap'].data[li].uv=(uv[0],1-uv[1])
    return col

def stage():
    s=bpy.context.scene;col=C.collection('99 撮影専用')
    ground=C.proc('撮影の駐車場',(131,126,111),scale=8)
    C.box('撮影用舗装',(-5,-8,-.053),(5000,5000,.10),ground)
    ld=bpy.data.lights.new('柔らかい太陽','SUN');ld.energy=2.2;ld.angle=.2
    ob=bpy.data.objects.new('柔らかい太陽',ld);col.objects.link(ob);ob.rotation_euler=(.45,-.25,-.5)
    ld=bpy.data.lights.new('空の補助光','AREA');ld.energy=700;ld.size=12
    ob=bpy.data.objects.new('空の補助光',ld);col.objects.link(ob);ob.location=(-9,-8,12)
    cd=bpy.data.cameras.new('確認カメラ');cam=bpy.data.objects.new('確認カメラ',cd);col.objects.link(cam);s.camera=cam
    s.render.resolution_x=1000;s.render.resolution_y=700;s.render.resolution_percentage=100
    s.cycles.samples=12;s.cycles.use_denoising=True
    return s,cam

def render(mode):
    context=mode=='reference';objects=load_game(context);s,cam=stage()
    if context:
        s.render.resolution_x=1500;s.render.resolution_y=1000
        # 依頼の仮カメラ位置と指定方位。追加物以外の屋根などは既存NPZを表示用に読むだけ。
        cam.location=(-9.4,-9.8,2.5);cam.data.type='PERSP';cam.data.sensor_fit='VERTICAL';cam.data.lens=cam.data.sensor_height/2
        X=np.array([math.sin(math.radians(59)),math.cos(math.radians(59)),0]);Y=np.array([math.sin(math.radians(329)),math.cos(math.radians(329)),0])
        for heading in [0,315,270]:
            a=math.radians(heading);p=math.radians(-4.13);f=np.array([math.sin(a)*math.cos(p),math.cos(a)*math.cos(p),math.sin(p)]);direction=Vector((f@X,f@Y,f[2]))
            cam.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
            s.render.filepath=str(CMP/('nanohanadai_toilet_sw_reference_%03d.png'%heading));bpy.ops.render.render(write_still=True)
    else:
        focus=Vector((-12.15,-4.94,.38));cam.data.type='ORTHO';cam.data.ortho_scale=8.0
        views={'front':(0,-1,.03),'left':(-1,0,.03),'right':(1,0,.03),'back_estimated':(0,1,.03),'top':(0,0,1),'persp':(-1,-1,.75)}
        # 部品だけの方向確認。v2全体は画角外を許し、延長部を中心に置く。
        for name,pos in views.items():
            bpy.data.objects['撮影用舗装'].hide_render=name not in ['top','persp']
            for ob in objects.objects:
                if ob.name!='ramp_sw_approach':ob.hide_render=name!='persp'
            cam.location=focus+Vector(pos).normalized()*15;cam.rotation_euler=(focus-cam.location).to_track_quat('-Z','Y').to_euler()
            s.render.filepath=str(CMP/('nanohanadai_toilet_sw_'+name+'.png'));bpy.ops.render.render(write_still=True)
    return {'rendered':mode}
