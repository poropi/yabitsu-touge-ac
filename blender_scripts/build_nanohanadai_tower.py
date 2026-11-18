"""菜の花台物見塔。写真は比較専用、材質画像は数式で自作する。"""
from pathlib import Path
import sys
import json
import math
import shutil
import random
import numpy as np
import bpy
import bmesh
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'assets/landmarks/nanohanadai_tower'
OUT = ROOT/'data/processed/landmarks/nanohanadai_tower'
CMP = ROOT/'export/compare/nanohanadai'
N = 12
DECK = 9.30
RNG = random.Random(10630)
COL = None
MAT = None
STATS = []
TAU = math.tau


def point(r, a, z):
    return Vector((r*math.cos(a), r*math.sin(a), z))


def atlas():
    """16区画の自作色画像。木目、木口、金属、石を共通画像に収める。"""
    size = 512
    arr = np.ones((size*4, size*4, 4), dtype=np.float32)
    gen = np.random.default_rng(10630)
    yy, xx = np.mgrid[0:size, 0:size]/size
    bases = [(75,63,55),(83,66,54),(70,61,55),(91,76,62),
             (78,68,59),(94,80,68),(105,99,86),(84,70,57),
             (88,110,109),(62,77,75),(156,148,125),(102,107,107),
             (147,149,138),(152,155,141),(116,117,107),(127,155,166)]
    for tile, base in enumerate(bases):
        noise = gen.normal(0, 1, (size, size))
        smooth = (np.sin(xx*TAU*3+yy*7)+np.cos(xx*13+yy*8))*.5
        if tile < 7:
            bend = xx + .006*np.sin(yy*15+tile)+.002*np.sin(yy*46+xx*18)
            grain = .06*np.sin(bend*TAU*83)+.035*np.sin(bend*TAU*213)
            scratches = np.power(np.maximum(0, np.cos(bend*TAU*49)), 45)
            value = 1+grain+.10*smooth+.017*noise-.20*scratches
            rgb = np.array(base)[None,None,:]/255*value[...,None]
            # 木の塗装が部分的に灰色へ退色する。色は撮影光を含まない。
            fade = np.clip(.10+smooth*.10, 0, .22)
            rgb = rgb*(1-fade[...,None])+np.array([.43,.41,.36])*fade[...,None]
        elif tile == 7:
            rr = np.sqrt((xx-.38)**2+(yy-.55)**2)
            value = 1+.13*np.sin(rr*TAU*34)+noise*.025
            rgb = np.array(base)/255*value[...,None]
        else:
            value = 1+smooth*(.045 if tile < 12 else .12)+noise*(.012 if tile < 12 else .035)
            rgb = np.array(base)/255*value[...,None]
            if tile == 10:
                rgb *= (1-.18*(np.mod(xx*12, 1)<.035))[...,None]
        row, column = divmod(tile, 4)
        rgb = np.clip(rgb, 0, 1)
        arr[row*size:(row+1)*size,column*size:(column+1)*size,:3] = rgb
    im = bpy.data.images.new('tower_atlas 自作色', size*4, size*4, alpha=False)
    im.pixels.foreach_set(arr.ravel())
    im.filepath_raw = str(SRC/'tower_atlas.png')
    im.file_format = 'PNG'
    im.save()
    # Image.save()は生成バッファの値を保存する。保存PNGを再読込し、制作元と納品の色を一致させる。
    im = bpy.data.images.load(str(SRC/'tower_atlas.png'),check_existing=False)
    im.pack()
    mat = bpy.data.materials.new('風化した木と屋根と石 自作画像')
    nodes = mat.node_tree.nodes
    bs = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    bs.inputs['Roughness'].default_value = .82
    tx = nodes.new('ShaderNodeTexImage')
    tx.image = im
    mat.node_tree.links.new(tx.outputs['Color'], bs.inputs['Base Color'])
    return mat


def collection(name):
    global COL
    COL = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(COL)
    return COL


def mesh(name, vertices, faces, tile=0, uvs=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(vertices, [], faces)
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    COL.objects.link(ob)
    me.materials.append(MAT)
    layer = me.uv_layers.new(name='UVMap')
    # 各面の平面投影で木目の長さ方向を維持する。島は画像区画内に収める。
    coords = np.array(vertices)
    for poly in me.polygons:
        idx = list(poly.vertices)
        co = coords[idx]
        if uvs is not None:
            uv = np.array([uvs[i] for i in idx])
        else:
            delta = np.ptp(co, axis=0)
            axes = sorted(range(3), key=lambda k:delta[k], reverse=True)[:2]
            # 板はローカルZに木目を通す。
            v_axis = 2 if delta[2] > .001 else axes[0]
            u_axis = next(k for k in axes if k != v_axis) if v_axis in axes else axes[1]
            uv = np.c_[(co[:,u_axis]-co[:,u_axis].min())/max(delta[u_axis],.00001),
                       (co[:,v_axis]-co[:,v_axis].min())/max(delta[v_axis],.00001)]
        t = tile if isinstance(tile,int) else tile[poly.index % len(tile)]
        row, column = divmod(t,4)
        for li, q in zip(poly.loop_indices, uv):
            layer.data[li].uv = ((column+.015+.97*float(q[0]))/4,
                                (row+.015+.97*float(q[1]))/4)
    return ob


def box(name, center, dims, tile=None):
    x,y,z = (d/2 for d in dims)
    vertices = [(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),
                (-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)]
    faces = [(3,2,1,0),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7)]
    tile = RNG.randrange(6) if tile is None else tile
    ob = mesh(name,vertices,faces,tile)
    ob.location = center
    return ob


def beam(name, a, b, width, depth=None, tile=None):
    a,b = Vector(a), Vector(b)
    ob = box(name,(a+b)/2,(width,depth or width,(b-a).length),tile)
    ob.rotation_euler = (b-a).to_track_quat('Z','Y').to_euler()
    return ob


def prism(name, xy, bottom, top, tile):
    count = len(xy)
    vertices = [(x,y,bottom) for x,y in xy]+[(x,y,top) for x,y in xy]
    faces = [tuple(range(count-1,-1,-1)),tuple(range(count,count*2))]
    faces += [(i,(i+1)%count,(i+1)%count+count,i+count) for i in range(count)]
    return mesh(name,vertices,faces,tile)


def wedge(name, a0, a1, ri, ro, z, thickness, tile=None):
    xy = [point(ri,a0,0)[:2],point(ro,a0,0)[:2],point(ro,a1,0)[:2],point(ri,a1,0)[:2]]
    return prism(name,xy,z-thickness,z,RNG.randrange(6) if tile is None else tile)


def bolt(name, p, normal):
    # 閉じた六角ボルト頭。写真の留め金を立体として残す。
    r = .025
    xy = [(r*math.cos(i*TAU/6),r*math.sin(i*TAU/6)) for i in range(6)]
    ob = prism(name,xy,-.007,.007,11)
    ob.location = p
    ob.rotation_euler = Vector(normal).to_track_quat('Z','Y').to_euler()


def frame():
    collection('frame')
    for i in range(N):
        a = i*TAU/N
        beam(f'main_post_{i:02d}',point(3.10,a,.025),point(2.30,a,9.24),.29,.32)
        beam(f'deck_bracket_{i:02d}',point(2.44,a,7.65),point(3.34,a,9.12),.25,.28)
        if i%2 == 0:
            beam(f'inner_post_{i:02d}',point(1.04,a,0),point(1.04,a,9.30),.19,.22)
        for z in (1.65,4.75,7.85):
            r = 3.10-.80*z/9.24
            beam(f'ring_beam_{i:02d}_{z}',point(r,a,z),point(r,a+TAU/N,z),.19,.25)
        # 筋かいは六つの交互の柱間にXを配する。
        if i%2 == 0:
            for lo,hi in [(1.35,2.65),(4.45,5.75),(7.55,8.85)]:
                rlo = 3.10-.80*lo/9.24+.12
                rhi = 3.10-.80*hi/9.24+.12
                for k in range(2):
                    beam(f'X_brace_{i:02d}_{lo}_{k}',
                         point(rlo,a+k*TAU/N,lo),point(rhi,a+(1-k)*TAU/N,hi),.13,.09)
        for z in (.20,1.65,4.75,7.85,9.10):
            r = 3.10-.80*z/9.24+.17
            bolt(f'post_bolt_{i:02d}_{z}',point(r,a,z),point(1,a,0))


def stairs():
    collection('stairs')
    paths = []
    for branch,offset in [('A',0),('B',math.pi)]:
        records=[]
        z = 0
        tread_id = 0
        for seg in range(18):
            a0 = math.radians(-60)-seg*TAU/N+offset
            a1 = a0-TAU/N
            rise = 0 if seg%3 == 2 else DECK/12
            z1 = z+rise
            ro0, ro1 = 2.95-.70*z/DECK,2.95-.70*z1/DECK
            if seg==17:
                ro0=ro1=2.40
            if rise:
                # 扇形の楔段ではなく、十二角形の直線辺を4枚の矩形状踏板に分ける。
                po0,po1 = point(ro0,a0,0),point(ro1,a1,0)
                pi0,pi1 = point(1.13,a0,0),point(1.13,a1,0)
                for j in range(4):
                    t0,t1=j/4,(j+1)/4
                    xy=[(pi0.lerp(pi1,t0))[:2],(po0.lerp(po1,t0))[:2],
                        (po0.lerp(po1,t1))[:2],(pi0.lerp(pi1,t1))[:2]]
                    prism(f'tread_{branch}_{tread_id:02d}',xy,z+rise*t1-.075,z+rise*t1,RNG.randrange(6))
                    tread_id+=1
            else:
                wedge(f'landing_{branch}_{seg//3:02d}',a1,a0,1.13,ro0,z,.11)
            for radius,side in [(ro0,'outer'),(1.13,'inner')]:
                r_end = ro1 if side=='outer' else radius
                beam(f'stringer_{branch}_{seg:02d}_{side}',point(radius,a0,z-.16),point(r_end,a1,z1-.16),.14,.21)
                if seg==17 and side=='outer':
                    continue
                # 5枚の板と上の笠木。写真で見える隙間を実形状で残す。
                for k in range(5):
                    h=.20+k*.155
                    beam(f'stair_rail_{branch}_{seg:02d}_{side}_{k}',point(radius,a0,z+h),point(r_end,a1,z1+h),.105,.045)
                beam(f'stair_handrail_{branch}_{seg:02d}_{side}',point(radius,a0,z+1.05),point(r_end,a1,z1+1.05),.13,.085)
                beam(f'stair_newel_{branch}_{seg:02d}_{side}',point(radius,a0,z-.04),point(radius,a0,z+1.10),.11,.11)
            records.append(dict(segment=seg,start_angle_deg=math.degrees(a0),end_angle_deg=math.degrees(a1),z0=z,z1=z1,landing=rise==0))
            z=z1
        paths.append(dict(name=branch,treads=tread_id,risers=48,riser_m=DECK/48,segments=records))
    (CMP/'nanohanadai_tower_stair_paths.json').write_text(json.dumps(paths,indent=2)+'\n')


def deck():
    collection('deck')
    # 二重らせんの上段全体を吹抜けにし、外周を回れる環状デッキにする。
    openings = {4,10}
    for i in range(N):
        a0,a1=i*TAU/N,(i+1)*TAU/N
        # 床の継ぎ目は帯ごとに実形状。内側の中央吹抜けも残す。
        for j in range(10):
            ri=2.40+j*(3.42-2.40)/10+.003
            ro=2.40+(j+1)*(3.42-2.40)/10-.003
            wedge(f'deck_board_{i:02d}_{j:02d}',a0+.0005,a1-.0005,ri,ro,DECK,.10)
        beam(f'deck_fascia_{i:02d}',point(3.42,a0,9.10),point(3.42,a1,9.10),.31,.16)
        beam(f'deck_joist_{i:02d}',point(2.40,a0,9.11),point(3.42,a0,9.11),.19,.24)
        beam(f'deck_column_{i:02d}',point(3.22,a0,DECK),point(3.22,a0,11.90),.21,.23)
        beam(f'deck_rail_top_{i:02d}',point(3.24,a0,10.40),point(3.24,a1,10.40),.14,.12)
        beam(f'deck_rail_bottom_{i:02d}',point(3.24,a0,9.44),point(3.24,a1,9.44),.12,.10)
        for z in (10.56,10.72):
            beam(f'deck_upper_rail_{i:02d}_{z}',point(3.24,a0,z),point(3.24,a1,z),.055,.045)
        for j in range(1,10):
            p0=point(3.24,a0,9.49).lerp(point(3.24,a1,9.49),j/10)
            beam(f'deck_baluster_{i:02d}_{j:02d}',p0,p0+Vector((0,0,.85)),.062,.065,tile=6 if j%4==0 else None)
        if i not in openings:
            for z in (9.53,9.86,10.20):
                beam(f'well_guard_{i:02d}_{z}',point(2.40,a0,z),point(2.40,a1,z),.10,.065)
            beam(f'well_post_{i:02d}',point(2.40,a0,DECK),point(2.40,a0,10.25),.1)


def roof():
    collection('roof')
    r0,r1,z0,z1=.48,3.90,12.74,11.90
    def height(r):
        return z0+(z1-z0)*(r-r0)/(r1-r0)
    for i in range(N):
        a,b=i*TAU/N,(i+1)*TAU/N
        pts=[point(r0,a,z0),point(r1,a,z1),point(r1,b,z1),point(r0,b,z0)]
        # 金属屋根と板張りの裏を別々の閉じた立体にする。
        for suffix,delta,depth,tile in [('metal',0,.035,8),('soffit',-.050,.07,10)]:
            vv=[tuple(p+Vector((0,0,delta-depth))) for p in pts]+[tuple(p+Vector((0,0,delta))) for p in pts]
            mesh(f'roof_{suffix}_{i:02d}',vv,[(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],tile,
                 [(0,1),(0,0),(1,0),(1,1)]*2)
        beam(f'eave_fascia_{i:02d}',point(r1,a,11.83),point(r1,b,11.83),.17,.105)
        beam(f'roof_hip_cap_{i:02d}',point(r0,a,z0+.015),point(r1,a,z1+.015),.065,.035,9)
        beam(f'roof_rafter_{i:02d}',point(.48,a,z0-.16),point(3.88,a,z1-.16),.12,.16)
        beam(f'roof_ring_{i:02d}',point(3.22,a,11.74),point(3.22,b,11.74),.20,.24)
        for j in range(1,14):
            r=r0+(r1-r0)*j/14
            beam(f'metal_horizontal_seam_{i:02d}_{j:02d}',point(r,a,height(r)+.015),point(r,b,height(r)+.015),.018,.022,9)
    # 写真の頂に見える低い採光部と金物。透過は使わず灰青の色面で近似。
    for i in range(12):
        wedge(f'skylight_flashing_{i:02d}',i*TAU/12,(i+1)*TAU/12,.40,.60,12.75,.05,9)
    for i in range(6):
        a,b=i*TAU/6,(i+1)*TAU/6
        cap=[point(.48,a,12.90),point(.48,b,12.90),Vector((0,0,12.99))]
        vv=[tuple(p-Vector((0,0,.015))) for p in cap]+[tuple(p) for p in cap]
        mesh(f'skylight_cap_{i:02d}',vv,[(2,1,0),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)],15)
        pts=[point(.48,a,12.72),point(.48,b,12.72),point(.48,b,12.90),point(.48,a,12.90)]
        normal=point(.012,(a+b)/2,0)
        vv=[tuple(p-normal) for p in pts]+[tuple(p+normal) for p in pts]
        mesh(f'skylight_pane_{i:02d}',vv,[(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],15)
        beam(f'crown_metal_{i:02d}',point(.48,a,12.88),point(0,0,13.00),.035,.035,11)
        beam(f'crown_sill_{i:02d}',point(.48,a,12.73),point(.48,b,12.73),.035,.035,9)
        beam(f'crown_mullion_{i:02d}',point(.48,a,12.73),point(.48,a,12.90),.03,.03,9)
    # 屋根写真の棟沿いのワイヤーと留め板。目立つ2辺だけを簡略化。
    for a in (0,math.pi):
        beam('roof_wire_'+str(a),point(.52,a,height(.52)+.045),point(3.75,a,height(3.75)+.045),.015,.015,11)
        for r in (1.20,2.90):
            ob=box('roof_anchor_plate',point(r,a,height(r)+.027),(.20,.15,.025),11)
            ob.rotation_euler[1] = math.atan((z0-z1)/(r1-r0))*(1 if a==0 else -1)


def base():
    collection('base')
    xy=[point(4.0,i*TAU/48,0)[:2] for i in range(48)]
    prism('concrete_platform',xy,-.38,0,12)
    # 野面石の低い壁は台の外周の被覆。通路側に開きを残す。
    for row in range(3):
        for j in range(56):
            a0=(j+(row%2)*.5)*TAU/56
            a1=a0+TAU/56*.96
            mid=(a0+a1)/2
            bearing=(90-math.degrees(mid))%360
            if 151<bearing<179:
                continue
            z=-.36+row*.12
            # 不揃いの6角形の石を厚み付きで前面に積む。
            r=4.01
            center=point(r,mid,z+.058)
            tangent=point(1,mid+math.pi/2,0)
            radial=point(1,mid,0)
            w=(a1-a0)*r
            outline=[(-w*.46,-.026),(-w*.32,-.056),(w*.32,-.051),(w*.48,-.015),(w*.35,.049),(-w*.29,.056)]
            vv=[]
            for dep in (-.01,.058+RNG.random()*.018):
                for u,v in outline:
                    vv.append(tuple(center+tangent*u+Vector((0,0,v))+radial*dep))
            faces=[tuple(range(5,-1,-1)),tuple(range(6,12))]
            faces += [(k,(k+1)%6,(k+1)%6+6,k+6) for k in range(6)]
            mesh(f'fieldstone_{row}_{j:02d}',vv,faces,13)
    # 南南東への石段。上端は台上面、下端は1.08m低い園路。
    outward=point(1,math.radians(-75),0)
    tangent=point(1,math.radians(15),0)
    for j in range(6):
        near=3.85+j*.32
        far=near+.325
        top=-j*.18
        xy=[tuple(outward*d+tangent*w)[:2] for d,w in [(near,-.82),(far,-.82),(far,.82),(near,.82)]]
        prism(f'stone_step_{j:02d}',xy,-1.09,top,12)
    for i in range(12):
        a=i*TAU/N
        xy=[tuple(point(3.10,a,0)+Vector((dx,dy,0)))[:2] for dx,dy in [(-.20,-.20),(.20,-.20),(.20,.20),(-.20,.20)]]
        prism(f'post_foot_{i:02d}',xy,0,.045,12)


def export_part(name):
    bpy.context.view_layer.update()
    vertices,faces,uvs=[],[],[]
    for ob in bpy.data.collections[name].objects:
        me=ob.data
        me.calc_loop_triangles()
        layer=me.uv_layers['UVMap']
        lookup={}
        for tri in me.loop_triangles:
            f=[]
            for li in tri.loops:
                vi=me.loops[li].vertex_index
                uv=layer.data[li].uv
                key=(vi,round(uv.x,7),round(uv.y,7),*tuple(round(n,6) for n in tri.normal))
                if key not in lookup:
                    lookup[key]=len(vertices)
                    vertices.append(tuple(ob.matrix_world @ me.vertices[vi].co))
                    uvs.append((uv.x,1-uv.y))
                f.append(lookup[key])
            faces.append(f)
    v=np.array(vertices,np.float64)
    f=np.array(faces,np.int32)
    u=np.array(uvs,np.float64)
    np.savez_compressed(OUT/(name+'.npz'),v=v,f=f,uv=u)
    shutil.copyfile(SRC/'tower_atlas.png',OUT/(name+'.png'))
    STATS.append(dict(name=name,triangles=len(f),vertices=len(v),
                      dimensions_m=np.ptp(v,axis=0).tolist(),bounds_m=[v.min(0).tolist(),v.max(0).tolist()]))


def build():
    global MAT
    STATS.clear()
    RNG.seed(10630)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    SRC.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    s=bpy.context.scene
    s.unit_settings.system='METRIC'
    s.unit_settings.scale_length=1
    MAT=atlas()
    for fn in (frame,stairs,deck,roof,base):
        fn()
        print('BUILT '+fn.__name__,flush=True)
    bpy.context.view_layer.update()
    for name in ('frame','stairs','deck','roof','base'):
        export_part(name)
    total=sum(p['triangles'] for p in STATS)
    assert total<=40000,total
    meta=dict(asset_id='nanohanadai_tower',generator='blender_scripts/build_nanohanadai_tower.py',
              source='nanohanadai_tower.blend',source_texture='tower_atlas.png',
              polygon_sides=12,stair_paths=2,parts=STATS,total_triangles=total,
              coordinates=dict(unit='m',x='east true',y='north true',z='up',origin='塔中心直下の台上面'),
              direction=dict(roof_vertex_azimuth_deg=90,flat_face_normals_azimuth_deg=[15+30*i for i in range(12)],approach_steps_azimuth_deg=165,entrance_first_tread_azimuth_deg=150,exit_first_tread_azimuth_deg=330),
              dimensions_m=dict(roof_diameter=7.8,deck_outer_diameter=6.84,post_base_center_diameter=6.2,post_deck_center_diameter=4.6,deck_floor=9.3,rail_main_top=10.4,rail_thin_upper_top=10.7475,eave=11.9,roof_crown=13.0,base_radius=4.0),
              npz=dict(keys=['v','f','uv'],v='m +Z up',f='triangles',uv='v=0 at image top; Blender reload uses 1-v'),
              textures=dict(size=[2048,2048],alpha=False,source='数式で自作した16区画の色画像、写真画素を流用せず、照明を含まない',parts_share_same_image=True),
              refs=['export/compare/nanohanadai/tower/ref_yakei_nanohanadai4.jpg','export/compare/nanohanadai/tower/tw_0rX1Rg_045.png','export/compare/nanohanadai/tower/tw_0rXDBQ_345.png','export/compare/nanohanadai/tower/tw_4_znQE_015.png','export/compare/nanohanadai/tower/tw_IfLggE_000.png','export/compare/nanohanadai/tower/tw_q6-KBA_090.png','export/compare/nanohanadai/tower/tw_z8nbWA_270.png','export/compare/nanohanadai/tower/aerial_tower_north_up.png'],
              primary_reference='https://www.kubodera-architect.co.jp/works/index/27',
              reference_metadata={'yakei':{'date':'2010-01','url':'https://yakei.jp/photo/1280/nanohanadai4-1280.jpg','view':'南西から見上げ'},
                  '0rX1Rg':{'panoid':'CIHM0ogKEICAgICc0rX1Rg','date':'2019-12','heading':45,'tilt':105},
                  '0rXDBQ':{'panoid':'CIHM0ogKEICAgICc0rXDBQ','date':'2019-12','heading':345,'tilt':95},
                  '4_znQE':{'panoid':'CIHM0ogKEICAgIDqi4_znQE','date':'2014-05','heading':15,'tilt':100},
                  'IfLggE':{'panoid':'CIHM0ogKEICAgICnzIfLggE','date':'2024-09'},
                  'q6-KBA':{'panoid':'CIHM0ogKEICAgICCq6-KBA','date':'2020-09'},
                  'z8nbWA':{'panoid':'CIHM0ogKEICAgID4z8nbWA','date':'2016-07'}},
              dimension_basis='設計者の全高13m。北上航空写真の屋根約190px×約0.041m/px=約7.8m。人物、手すり、柱の写真射影は別の採寸図で照合。各高さ、部材断面、48段/系統は推定。',
              approximations=['階段の段数は全経路を写真で数えられず48段/系統を採用。','アプローチSSE165°は航空写真の南南東の道から推定、初段150°と別系統330°の割り当ては経路の仮定。','柱の継手、基礎の地中部、金属屋根の留め板、採光部の詳細は簡略化。','野面石の個別輪郭と木材の風化の位置は自作模様で近似。','木材の光沢と金属反射はPNGの色以外は配置側の材質設定が必要。'],
              game_status='NPZとPNGまで。コース配置、KN5、ゲーム導入は依頼範囲外。')
    meta['measurement_records']=['export/compare/nanohanadai/nanohanadai_tower_measurements.json',
                                 'export/compare/nanohanadai/nanohanadai_tower_aerial_measurements.json']
    meta['parts_meaning']={'frame':'下へ広がる柱、横梁、X筋かい、腕木、留め金',
        'stairs':'二重らせんの2系統各48段、ささら、各6踊り場、板の手すり',
        'deck':'環状床、縦格子と上の横桟、外周柱、吹抜けの手すり',
        'roof':'十二角の金属屋根、裏板、垂木、軒、継ぎ目、採光部と金物',
        'base':'コンクリート台、野面石の低い壁、南南東の石段、柱脚の台'}
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_location=(0,0,6)
                area.spaces.active.region_3d.view_distance=23
                area.spaces.active.region_3d.view_rotation=Vector((18,24,-12)).to_track_quat('-Z','Y')
                area.spaces.active.shading.type='MATERIAL'
    (SRC/'asset.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(SRC/'nanohanadai_tower.blend'))
    print('TOWER_RESULT '+json.dumps(STATS),flush=True)
    return meta


if __name__=='__main__':
    build()
