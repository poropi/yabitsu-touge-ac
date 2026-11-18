"""依頼07: NPZ契約検査、別プロセスblend再読込、ゲーム用データの多方向描画。"""
from pathlib import Path
import sys,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CMP=ROOT/'export/compare/pass'
TOTAL_LIMIT={'pass_shop':20000,'pass_toilet':25000,'pass_lodge':15000,'pass_resthouse':15000}
PER_LIMIT={'pass_vending':1500,'pass_info_boards':1000,'pass_trail_stairs':500,'pass_benches':1000,'pass_trail_posts':300,'pass_bike_rack':400,'pass_bus_stop':1000,'pass_brown_signs':1000,'pass_utility_pole':2000}
PER_LIMIT.update(pass_barriers=400,pass_site_equipment=1000)
ROAD_LIMIT={'guardrail':1000,'wood_fence':500,'camera_standing':1000,'camera_pole_board':1000}
EXPECTED={'pass_shop':['shop_body','shop_sign','shop_annex','shop_stone_base','shop_steps'],'pass_vending':['vending_a','vending_b']}
EXPECTED.update({
 'pass_resthouse':['walls','roof','front_glazing','deck','foundation','rails','stairs','fittings'],
 'pass_toilet':['walls','metal_roof','wood_trusses','front_boards','built_in_bench','entrance_marks','west_clerestory','rear_shed'],
 'pass_lodge':['lodge_walls','lodge_roof','lodge_end_details'],
 'pass_info_boards':['large_white','small_weathered','parking_notice'],
 'pass_trail_stairs':['timber_step','rail_left','rail_right','wood_rail_left','wood_rail_right'],
 'pass_benches':['bench_low_plank','bench_box_short','bench_box_long'],
 'pass_trail_posts':['trail_white_post','trail_brown_directions','trail_blue_notice','trail_green_notice'],
 'pass_bike_rack':['cycle_rack_A'],
 'pass_bus_stop':['bus_round_blank','bus_round_blank_dark','bus_tall_timetable'],
 'pass_brown_signs':['south_tall_sign','south_wide_sign','lotn_small_sign','u1_black_direction'],
 'pass_road_items':['orange_pole','camera_pole_board','camera_standing','white_cylinder','guardrail','wood_fence','island_rope_fence','tree_tripod','white_stepping_stone','manhole'],
 'pass_utility_pole':['pole_shop','pole_north','pole_opposite_red'],
 'pass_barriers':['barrier_A_short','barrier_A_long'],
 'pass_site_equipment':['electric_cabinet_small','electric_cabinet_large','equipment_fence','island_sign','opposite_yellow_notice','small_roof_notice_post','green_column']})

def validate(asset):
    from PIL import Image
    info=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text());checks=[];total=0
    if asset in EXPECTED:assert {p['name'] for p in info['parts']}==set(EXPECTED[asset]),asset
    for p in info['parts']:
        name=p['name'];path=ROOT/'data/processed/landmarks'/asset/name
        with np.load(path.with_suffix('.npz')) as d:
            assert set(d.files)=={'v','f','uv'}
            v,f,u=d['v'],d['f'],d['uv']
        assert v.ndim==2 and v.shape[1]==3 and len(v)>2 and f.ndim==2 and f.shape[1]==3 and len(f)>0
        assert u.shape==(len(v),2) and np.isfinite(v).all() and np.isfinite(u).all()
        assert np.issubdtype(f.dtype,np.integer) and f.min()>=0 and f.max()<len(v)
        assert u.min()>=-1e-5 and u.max()<=1.00001
        assert len(v)<=65535
        a=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        area=np.linalg.norm(a,axis=1)/2;assert area.min()>1e-10,(asset,name,'degenerate',area.min())
        assert abs(v[:,2].min())<1e-5 and np.max(np.abs((v.min(0)+v.max(0))[:2]))<1e-5,(name,'origin')
        assert np.allclose(np.ptp(v,axis=0),p['dimensions_m'],atol=.00011)
        assert p['triangles']==len(f)
        limit=PER_LIMIT.get(asset,25000)
        if asset=='pass_road_items':limit=ROAD_LIMIT.get(name,300)
        assert len(f)<=limit,(asset,name,len(f),limit)
        im=Image.open(path.with_suffix('.png'));arr=np.asarray(im)
        assert im.mode in ('RGB','RGBA') and min(im.size)>=128
        assert arr[:,:,:3].max()>40 and np.ptp(arr[:,:,:3])>15,(asset,name,'empty color')
        checks.append({'name':name,'triangles':len(f),'vertices':len(v),'dimensions_m':np.ptp(v,axis=0).round(4).tolist(),'min_triangle_area_m2':float(area.min()),'image_mode':im.mode,'texture_size':list(im.size),'uv_range':[u.min(0).tolist(),u.max(0).tolist()],'origin_checked':True})
        total+=len(f)
    assert total<=TOTAL_LIMIT.get(asset,100000),(asset,total)
    result={'asset':asset,'passed':True,'triangles':total,'parts':checks}
    (CMP/(asset+'_npz_validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return info,result

def load_game(asset,info,assembly=True):
    import bpy
    import pass_common as C
    C.init_scene();col=C.collection('01 ゲーム用NPZ再読込')
    for p in info['parts']:
        path=ROOT/'data/processed/landmarks'/asset/p['name']
        with np.load(path.with_suffix('.npz')) as d:v,f,u=d['v'].copy(),d['f'],d['uv']
        offset=info.get('preview_offsets',{}).get(p['name'],p['assembly_offset_m'] if assembly else (0,0,0))
        v+=np.array(offset)
        mat=C.image_material(p['name'],path.with_suffix('.png'));o=C.mesh(p['name'],v.tolist(),f.tolist(),mat)
        for poly in o.data.polygons:
            for li in poly.loop_indices:
                uv=u[o.data.loops[li].vertex_index];o.data.uv_layers['UVMap'].data[li].uv=(uv[0],1-uv[1])
    return list(col.objects)

def render(asset,views=None,part_name=None):
    import bpy
    from mathutils import Vector
    import pass_common as C
    info=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text())
    if part_name:
        info['parts']=[p for p in info['parts'] if p['name']==part_name]
        assert info['parts'],part_name
        info['preview_offsets']={part_name:[0,0,0]}
        info['reference_views']={}
    obs=load_game(asset,info)
    if asset=='pass_shop' and not part_name:
        # 比較用だけの組合せ。自販機のNPZは別assetのまま保持する。
        for i,name in enumerate(['vending_a','vending_b']):
            path=ROOT/'data/processed/landmarks/pass_vending'/name
            with np.load(path.with_suffix('.npz')) as d:v,f,u=d['v'].copy(),d['f'],d['uv']
            v+=np.array([3.40+i*1.06,-1.42,.62])
            mat=C.image_material(name,path.with_suffix('.png'));o=C.mesh(name,v.tolist(),f.tolist(),mat)
            for poly in o.data.polygons:
                for li in poly.loop_indices:
                    uv=u[o.data.loops[li].vertex_index];o.data.uv_layers['UVMap'].data[li].uv=(uv[0],1-uv[1])
            obs.append(o)
    vv=np.array([tuple(o.matrix_world@v.co) for o in obs for v in o.data.vertices]);lo=vv.min(0);hi=vv.max(0);center=(lo+hi)/2;extent=max(hi-lo)
    s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=6 if part_name else 12;s.cycles.use_denoising=True
    stage=C.collection('99 撮影専用')
    ground=C.proc('撮影床',(170,174,164),scale=3)
    stage_floor=C.box('撮影床',(center[0],center[1],lo[2]-.08),(extent*20,extent*20,.10),ground)
    for name,pos,power,size in [('空',(-2,-3,5),1600,2.8),('補助',(3,1,4),950,3.0)]:
        ld=bpy.data.lights.new(name,'AREA');ld.energy=power*(extent/4)**2;ld.size=extent*size
        ob=bpy.data.objects.new(name,ld);stage.objects.link(ob);ob.location=Vector(center)+Vector(pos)*extent;ob.rotation_euler=(Vector(center)-ob.location).to_track_quat('-Z','Y').to_euler()
    cd=bpy.data.cameras.new('確認');cam=bpy.data.objects.new('確認',cd);stage.objects.link(cam);s.camera=cam
    s.render.resolution_x=420 if part_name else 840;s.render.resolution_y=360 if part_name else 720;s.render.resolution_percentage=100
    allviews={'front':(0,-1,0),'left':(-1,0,0),'right':(1,0,0),'back_estimated':(0,1,0),'top':(0,0,1),'persp':(1,-1,.65)}
    allviews.update(info.get('reference_views',{}))
    for name,pos in allviews.items():
        if views and name not in views:continue
        stage_floor.hide_render=name in ('front','left','right','back_estimated')
        if asset=='pass_toilet' and name.startswith('reference'):
            pos=(pos[0],pos[1],0)
        cam.location=Vector(center)+Vector(pos).normalized()*extent*(2.3 if name.startswith('reference') else 8.0);cam.rotation_euler=(Vector(center)-cam.location).to_track_quat('-Z','Y').to_euler()
        if name.startswith('reference') and asset in ('pass_shop','pass_toilet','pass_lodge','pass_resthouse'):
            cam.location.z=1.55 if asset=='pass_resthouse' else 2.50
            cam.rotation_euler=(Vector(center)-cam.location).to_track_quat('-Z','Y').to_euler()
        cd.type='ORTHO';cd.ortho_scale=extent*1.40
        if name.startswith('reference'):cd.type='PERSP';cd.lens=52
        prefix=asset+'_'+part_name if part_name else asset
        s.render.filepath=str(CMP/(prefix+'_'+name+'.png'));bpy.ops.render.render(write_still=True)
    return {'asset':asset,'rendered':list(views or allviews)}

def render_parts(asset):
    info=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text())
    for p in info['parts']:render(asset,part_name=p['name'])
    return {'asset':asset,'individual_parts_rendered':len(info['parts'])}

def source(asset):
    import bpy
    path=ROOT/'assets/landmarks'/asset/(asset+'.blend');bpy.ops.wm.open_mainfile(filepath=str(path));bpy.context.view_layer.update()
    info=json.loads((path.parent/'asset.json').read_text());dg=bpy.context.evaluated_depsgraph_get();results=[]
    for p in info['parts']:
        col=bpy.data.collections.get(p['name']);assert col is not None,(asset,'collection',p['name'])
        bounds=[];tri=0
        for o in col.objects:
            if o.type not in ('MESH','CURVE','FONT'):continue
            ev=o.evaluated_get(dg);me=ev.to_mesh();me.calc_loop_triangles();assert len(me.vertices)>0
            bounds.extend([tuple(o.matrix_world@v.co) for v in me.vertices]);tri+=len(me.loop_triangles);ev.to_mesh_clear()
        assert tri==p['triangles'],(asset,p['name'],'source triangles',tri,p['triangles'])
        b=np.array(bounds);assert np.isfinite(b).all() and np.allclose(np.ptp(b,axis=0),p['dimensions_m'],atol=.00011)
        with np.load(ROOT/'data/processed/landmarks'/asset/(p['name']+'.npz')) as data:
            restored=data['v']+np.array(p['assembly_offset_m'])
        assert np.allclose(restored.min(0),b.min(0),atol=2e-5) and np.allclose(restored.max(0),b.max(0),atol=2e-5),(asset,p['name'],'assembly offset')
        results.append({'name':p['name'],'triangles':tri,'objects':len(col.objects),'bounds_m':[b.min(0).tolist(),b.max(0).tolist()]})
    assert all(im.packed_file for im in bpy.data.images if im.source=='FILE')
    result={'asset':asset,'blend_reopened':True,'packed_images':True,'parts':results,'passed':True}
    (CMP/(asset+'_source_validation.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    for asset in args:
        _,r=validate(asset);print(asset,r['triangles'],'PASS')
