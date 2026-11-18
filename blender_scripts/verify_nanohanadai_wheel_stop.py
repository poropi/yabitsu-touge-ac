"""輪止めの保存元・NPZを再読込し、6方向と写真方向を確認する。"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy,numpy as np,bmesh
from mathutils import Vector
import nanohanadai_common as C

def verify(render=True):
    src=C.ROOT/'assets/landmarks/nanohanadai_wheel_stop'
    path=C.ROOT/'data/processed/landmarks/nanohanadai_wheel_stop/wheel_stop'
    info=json.loads((src/'asset.json').read_text())
    # 呼出し元はこのblendを新規プロセスで読み込む。
    obs=[o for o in bpy.context.scene.objects if o.type=='MESH']
    assert len(obs)==1 and obs[0].name.startswith('輪止め_')
    bm=bmesh.new();bm.from_mesh(obs[0].data)
    volume=bm.calc_volume(signed=True);closed=all(e.is_manifold for e in bm.edges);bm.free()
    assert closed and .025<volume<.05,(closed,volume)
    with np.load(path.with_suffix('.npz')) as d:v,f,u=d['v'],d['f'],d['uv']
    assert set(np.load(path.with_suffix('.npz')).files)=={'v','f','uv'}
    assert len(f)==220 and len(f)<=300 and f.shape[1]==3 and u.shape==(len(v),2)
    assert np.isfinite(v).all() and np.isfinite(u).all() and f.min()>=0 and f.max()<len(v)
    assert 0<=u.min() and u.max()<=1 and np.allclose(np.ptp(v,axis=0),[1.65,.24,.12])
    assert np.allclose((v.min(0)+v.max(0))[:2]/2,[0,0]) and abs(v[:,2].min())<1e-8
    cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    area=np.linalg.norm(cross,axis=1)/2;assert area.min()>1e-10
    normals=np.zeros_like(v)
    for j in range(3):np.add.at(normals,f[:,j],cross)
    assert np.linalg.norm(normals,axis=1).min()>1e-10
    result={'passed':True,'blend_reopened':True,'source_objects':[o.name for o in obs],'source_closed':closed,'source_volume_m3':volume,'triangles':len(f),'vertices':len(v),'dimensions_m':np.ptp(v,axis=0).tolist(),'bounds_m':[v.min(0).tolist(),v.max(0).tolist()],'min_triangle_area_m2':float(area.min()),'uv_range':[u.min(0).tolist(),u.max(0).tolist()]}
    (C.CMP/'nanohanadai_wheel_stop_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if not render:return result
    C.init_scene();col=C.collection('01 NPZ再読込')
    mat=C.image_material('焼付済み輪止め',path.with_suffix('.png'));ob=C.mesh('wheel_stop',v.tolist(),f.tolist(),mat)
    for poly in ob.data.polygons:
        for li in poly.loop_indices:
            uv=u[ob.data.loops[li].vertex_index];ob.data.uv_layers['UVMap'].data[li].uv=(uv[0],1-uv[1])
    stage=C.collection('99 撮影専用')
    floor=C.proc('撮影床',(165,162,149),scale=25)
    ground=C.box('撮影用の地面',(0,0,-.035),(100,100,.06),floor)
    s=bpy.context.scene;s.cycles.samples=12;s.cycles.use_denoising=True
    s.render.resolution_x=900;s.render.resolution_y=450;s.render.resolution_percentage=100
    for name,pos,power,size in [('主光',(-3,-4,6),700,4),('補光',(3,2,4),350,4)]:
        ld=bpy.data.lights.new(name,'AREA');ld.energy=power;ld.size=size
        o=bpy.data.objects.new(name,ld);stage.objects.link(o);o.location=pos;o.rotation_euler=(-o.location).to_track_quat('-Z','Y').to_euler()
    cd=bpy.data.cameras.new('確認カメラ');cam=bpy.data.objects.new('確認カメラ',cd);stage.objects.link(cam);s.camera=cam
    views={'front':(0,-1,.025),'left':(-1,0,.025),'right':(1,0,.025),'back_estimated':(0,1,.025),'top':(0,0,1),'persp':(1,-1,.7),'reference_mid2_135':(-.25,1,.29),'reference_mid_135':(-.25,1,.24),'reference_mid_090':(-.86,.52,.20)}
    for name,pos in views.items():
        ground.hide_render=name in ['front','left','right','back_estimated']
        target=Vector((0,0,.06));cam.location=target+Vector(pos).normalized()*3
        cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cd.type='ORTHO';cd.ortho_scale=2.05 if name not in ['left','right'] else .6
        s.render.filepath=str(C.CMP/('nanohanadai_wheel_stop_'+name+'.png'));bpy.ops.render.render(write_still=True)
    return result

if __name__=='__main__':verify()
