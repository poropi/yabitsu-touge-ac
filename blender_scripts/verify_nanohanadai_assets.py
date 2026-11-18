"""NPZを再構築して多方向を描画。別プロセスで制作元blendも検査する。
blender -b --python-exit-code 1 --python blender_scripts/verify_nanohanadai_assets.py -- <asset-id> [render|source]
"""
import sys,json,math
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
import nanohanadai_common as C
ROOT=C.ROOT;CMP=C.CMP
LIMIT={'monument':15000,'park_sign':2000,'toilet':25000,'mirror':2000,'arrow_board':2000,'curb_a':300,'curb_b':300,'wood_box_bench':1000,'yellow_sign':50,'red_white_sign':50,'wood_bollard':200,'concrete_bench':1000,'l_curb':300,'checker_planter':1000,'white_planter':1000,'site_notice':2000}

EXPECTED={
 'nanohanadai_monument':{'monument'},'nanohanadai_park_sign':{'park_sign'},
 'nanohanadai_toilet':{'walls','metal_roof','wood_trusses','roof_rails','stone_base','ramp_walls','ramp_rails'},
 'nanohanadai_mirror':{'mirror','arrow_board'},'nanohanadai_curb_block':{'curb_a','curb_b','wood_box_bench'},
 'nanohanadai_fence_signs':{'yellow_sign','red_white_sign'},'nanohanadai_path_items':{'wood_bollard','concrete_bench','l_curb'},
 'nanohanadai_road_planters':{'checker_planter','white_planter'}}

def validate(asset):
    info=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text());checks=[]
    assert {p['name'] for p in info['parts']}==EXPECTED[asset],(asset,'missing expected parts')
    total=0
    for part in info['parts']:
        name=part['name'];path=ROOT/'data/processed/landmarks'/asset/(name+'.npz')
        with np.load(path) as d:v,f,u=d['v'],d['f'],d['uv']
        assert v.shape[1]==3 and f.shape[1]==3 and u.shape==(len(v),2),name
        assert np.isfinite(v).all() and np.isfinite(u).all(),name
        assert f.min()>=0 and f.max()<len(v) and u.min()>=-1e-5 and u.max()<=1.00001,name
        area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
        assert area.min()>1e-10,(name,'degenerate',area.min())
        assert len(v)<=65535,(name,'KN5 vertex limit')
        normals=np.zeros_like(v)
        cross=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        for j in range(3):np.add.at(normals,f[:,j],cross)
        lengths=np.linalg.norm(normals,axis=1)
        assert lengths.min()>1e-10,(name,'zero vertex normal',lengths.min())
        total+=len(f);limit=LIMIT.get(name,25000 if asset.endswith('toilet') else 2000)
        assert len(f)<=limit,(name,len(f),limit)
        assert part['triangles']==len(f)
        im=bpy.data.images.load(str(path.with_suffix('.png')),check_existing=True)
        p=np.array(im.pixels[:]).reshape(-1,4)
        assert p[:,:3].max()>.12 and np.ptp(p[:,:3])>.04,(name,'texture is empty')
        assert np.allclose(np.ptp(v,axis=0),part['dimensions_m'],atol=.0001),name
        checks.append({'part':name,'triangles':len(f),'dimensions_m':np.ptp(v,axis=0).round(4).tolist(),'min_triangle_area_m2':float(area.min()),'uv_range':[u.min(0).tolist(),u.max(0).tolist()],'texture_size':list(im.size),'texture_nonuniform':True,'vertices':len(v),'min_vertex_normal_length':float(lengths.min())})
    if asset.endswith('toilet'):assert total<=25000,total
    return info,checks

def load_game(asset,info):
    C.init_scene();col=C.collection('01 NPZ再読込')
    offsets=info.get('preview_offsets',{})
    for p in info['parts']:
        name=p['name'];path=ROOT/'data/processed/landmarks'/asset/name
        with np.load(path.with_suffix('.npz')) as d:v,f,u=d['v'].copy(),d['f'],d['uv']
        v+=np.array(offsets.get(name,[0,0,0]))
        mat=C.image_material(name,path.with_suffix('.png'));ob=C.mesh(name,v.tolist(),f.tolist(),mat)
        for poly in ob.data.polygons:
            for li in poly.loop_indices:
                uv=u[ob.data.loops[li].vertex_index];ob.data.uv_layers['UVMap'].data[li].uv=(uv[0],1-uv[1])
    return list(col.objects)

def render(asset,info):
    obs=load_game(asset,info);s=bpy.context.scene
    vv=np.array([tuple(o.matrix_world@v.co) for o in obs for v in o.data.vertices]);lo=vv.min(0);hi=vv.max(0);center=(lo+hi)/2
    extent=max(hi-lo);stage=C.collection('99 撮影専用')
    ground=C.proc('撮影床',(177,181,171),scale=2)
    C.mesh('撮影用地面',[(center[0]+x*extent*100,center[1]+y*extent*100,lo[2]-.02) for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]],[(0,1,2,3)],ground)
    for name,loc,power,size in [('大きい空の光',(-1,-2,3),1500,extent*1.8),('補助',(2,1,3),850,extent*2)]:
        ld=bpy.data.lights.new(name,'AREA');ld.energy=power*(extent/3)**2;ld.size=size
        o=bpy.data.objects.new(name,ld);stage.objects.link(o);o.location=Vector(center)+Vector(loc)*extent;o.rotation_euler=(Vector(center)-o.location).to_track_quat('-Z','Y').to_euler()
    cd=bpy.data.cameras.new('確認');cam=bpy.data.objects.new('確認',cd);stage.objects.link(cam);s.camera=cam
    cd.type='ORTHO';cd.ortho_scale=extent*1.4
    s.render.resolution_x=800;s.render.resolution_y=700;s.render.resolution_percentage=100
    s.cycles.samples=16;s.cycles.use_denoising=True
    views={'front':(0,-1,.05),'left':(-1,0,.05),'right':(1,0,.05),'back_estimated':(0,1,.05),'top':(0,0,1),'persp':(1,-1,.65)}
    # 写真と同じ物体の見える面（透視の位置合わせではなく同向き比較）。
    for n,pos in info.get('reference_views',{}).items():views[n]=pos
    for name,pos in views.items():
        focus=Vector(center);frame_extent=extent
        if asset.endswith('toilet') and name.startswith('reference'):
            focus=Vector((0,0,3.05));frame_extent=10.0
        cam.location=focus+Vector(pos).normalized()*frame_extent*2
        cam.rotation_euler=(focus-cam.location).to_track_quat('-Z','Y').to_euler()
        cd.type='ORTHO';cd.ortho_scale=frame_extent*1.35
        if name.startswith('reference'):
            cd.type='PERSP';cd.lens=52
        s.render.filepath=str(CMP/(asset+'_'+name+'.png'));bpy.ops.render.render(write_still=True)
    print('RENDER_DONE',asset,flush=True)

def source(asset,info):
    path=ROOT/'assets/landmarks'/asset/(asset+'.blend');bpy.ops.wm.open_mainfile(filepath=str(path))
    dg=bpy.context.evaluated_depsgraph_get();objects=[];bounds=[];printed_planes=0
    for o in bpy.context.scene.objects:
        if o.type!='MESH':continue
        ev=o.evaluated_get(dg);me=ev.to_mesh();me.calc_loop_triangles()
        co=np.array([tuple(o.matrix_world@v.co) for v in me.vertices]);assert len(co)>0 and np.isfinite(co).all(),o.name
        if np.ptp(co[:,1])<1e-6 and np.ptp(co[:,0])>.05 and np.ptp(co[:,2])>.05:
            if any(any(n.type=='TEX_IMAGE' and n.image and n.image.source=='FILE' for n in m.node_tree.nodes) for m in me.materials):
                assert all(poly.normal.y<-.99 for poly in me.polygons),(o.name,'printed front normal is not -Y')
                printed_planes+=1
        objects.append({'name':o.name,'triangles':len(me.loop_triangles)});bounds.extend(co.tolist());ev.to_mesh_clear()
    assert len(objects)>0
    assert sum(x['triangles'] for x in objects)==sum(p['triangles'] for p in info['parts']),(asset,'source/export triangle count')
    assert all(im.packed_file for im in bpy.data.images if im.source=='FILE'),'unpacked external image'
    return {'blend_reopened':True,'printed_front_planes_checked':printed_planes,'objects':objects,'collections':[c.name for c in bpy.context.scene.collection.children],'bounds_m':[np.min(bounds,0).tolist(),np.max(bounds,0).tolist()]}

if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:];asset=args[0];mode=args[1] if len(args)>1 else 'render'
    info,checks=validate(asset)
    if mode=='source':
        result={'asset_id':asset,'parts':checks,'source':source(asset,info),'passed':True}
        (CMP/(asset+'_validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    else:render(asset,info)
