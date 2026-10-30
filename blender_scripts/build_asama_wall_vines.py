"""浅間神社の蔓草3区切りを自作RGBA/両面メッシュとして生成する。
実行: .venv/bin/python blender_scripts/build_asama_wall_vines.py --textures
      blender -b --python-exit-code 1 --python blender_scripts/build_asama_wall_vines.py
コースの配置と壁は含めない。テクスチャは参照写真を使わない手続き生成。
"""
from pathlib import Path
import sys, math, json
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'assets/landmarks/asama_wall_vines'; OUT=ROOT/'data/processed/landmarks/asama_wall_vines'; CMP=ROOT/'export/compare/asama'
for p in (SRC,OUT,CMP): p.mkdir(parents=True,exist_ok=True)

def textures():
    import numpy as np
    from PIL import Image
    n=1024; a=np.zeros((n,n,4),np.uint8)
    rng=np.random.default_rng(155)
    # 15枚の葉と1枚の茎用セル。葉身はハート形、先端・葉脈・中肋・斑を自作する。
    yy,xx=np.mgrid[0:256,0:256]; x=(xx-127.5)/116; t=(yy-10)/236
    for j in range(16):
        # つる性の三出葉と単葉を混ぜる。3枚の葉も写真なしの数式形状。
        tile=np.zeros((256,256,4),np.uint8);tile[:,:,:3]=[64,91,26]
        forms=[(0,0,1,1,0)] if j>=8 else [(-.35,-.10,.53,.73,-.58),(.36,-.10,.53,.73,.58),(0,.23,.60,.84,0)]
        for cx,cy,sx,sy,rot in forms:
            X=(xx-127.5)/116-cx;Y=(yy-127.5)/116-cy
            xr=(X*math.cos(rot)-Y*math.sin(rot))/sx
            tr=(X*math.sin(rot)+Y*math.cos(rot))/sy*.5+.5
            width=np.sin(np.clip(tr,0,1)*math.pi)**.80*(.83+.11*np.cos(3*tr*math.pi+j))
            notch=(tr<.14)&(abs(xr)<(.14-tr)*.8)
            mask=(tr>0)&(tr<1)&(abs(xr)<width)&~notch
            if j in [11,15]:
                width*=.86+.14*np.cos(tr*6*math.pi)
                mask=(tr>0)&(tr<1)&(abs(xr)<width)&~notch
            shade=.76+.16*np.sqrt(np.clip(1-(xr/(width+1e-3))**2,0,1))+.07*np.cos(tr*math.pi)
            shade+=rng.normal(0,.032,(256,256))+.025*np.sin(xx*.19+yy*.065)
            vein=np.exp(-abs(xr)*90)*.11
            for k in range(1,7):
                line=k*.12+abs(xr)*.23
                vein+=np.exp(-abs(tr-line)*230)*.045*(abs(xr)<width*.9)
            base=np.array([(107,143,47),(90,129,39),(120,155,55),(82,117,35)][j%4],float)
            rgb=np.clip((shade[...,None]+vein[...,None])*base,0,255).astype(np.uint8)
            tile[mask,:3]=rgb[mask];tile[mask,3]=255
        a[(j//4)*256:(j//4+1)*256,(j%4)*256:(j%4+1)*256]=tile
    # 右下の1セルは茎専用。葉には使わず、葉先に茎色の線が出るのを防ぐ。
    a[768:1024,768:1024,:3]=[65,66,29];a[768:1024,768:1024,3]=255
    im=Image.fromarray(a,'RGBA');im.save(SRC/'vines_atlas_v2.png');im.save(OUT/'vines_atlas_v2.png')
    for c in 'abc': im.save(OUT/f'vines_{c}.png')

def sheets():
    from PIL import Image,ImageDraw,ImageFont
    from pathlib import Path
    p=CMP
    font=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',22)
    small=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',17)
    sheet=Image.new('RGB',(1440,1350),(238,239,233));d=ImageDraw.Draw(sheet)
    d.text((20,14),'浅間神社の蔓草 / 単体の形状・裏面・接続確認',font=font,fill=(20,36,23))
    labels=[('variant_a','A：斜め'),('front','正面 −Y'),('left','左側面 −X'),('right','右側面 +X'),('back','背面 +Y（推定）'),('top','真上（内部・背面は推定）')]
    for i,(name,label) in enumerate(labels):
     x=(i%3)*480;y=60+(i//3)*425
     im=Image.open(p/f'vines2_{name}.png').convert('RGB');im.thumbnail((475,390));sheet.paste(im,(x+(480-im.width)//2,y+28));d.text((x+12,y),label,font=small,fill=(20,36,23))
    im=Image.open(p/'vines2_three_joined.png').convert('RGB');im.thumbnail((1400,400));sheet.paste(im,((1440-im.width)//2,935));d.text((20,913),'B・A・C を2.5m間隔で連結（壁・斜面・配置は含まない）',font=small,fill=(20,36,23))
    sheet.save(p/'vines2_views.jpg',quality=94)
    comp=Image.new('RGB',(1440,2210),(238,239,233));d=ImageDraw.Draw(comp)
    d.text((20,12),'蔓草：写真の見込み方向を近似した単体比較',font=font,fill=(20,36,23));d.text((20,45),'寸法は依頼の壁高3mが基準。単体を周回撮影し見込み角・距離・照明を近似。設置位置・画角は一致していない。写真はGoogle Street View（2025年7月）。',font=small,fill=(20,36,23))
    for i,a in enumerate([0,45,90,315]):
     y=90+i*525
     im=Image.open(p/f'sv_a_{a:03}.png').convert('RGB');im.thumbnail((750,500));comp.paste(im,(0,y+24))
     im=Image.open(p/f'vines2_direction_{a:03}.png').convert('RGB');im.thumbnail((680,500));comp.paste(im,(755,y+24))
     d.text((10,y),f'参照 {a:03}°（原画像全体・出典表示保持）',font=small,fill=(20,36,23));d.text((765,y),f'制作モデル / SV {a:03}を参考にした推定方向 / 区切りA',font=small,fill=(20,36,23))
    comp.save(p/'vines2_reference_comparison.jpg',quality=94)

if '--sheets' in sys.argv:
    sheets();sys.exit(0)

if '--textures' in sys.argv:
    textures();sys.exit(0)

import bpy
import numpy as np
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.name='蔓草A（壁は含まない）';scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
# 明示的な裏面を使い、ゲームでbackface cullingを有効にしても両面を表示。
img=bpy.data.images.load(str(SRC/'vines_atlas_v2.png'));img.pack()
mat=bpy.data.materials.new('蔓草 自作RGBA（両面メッシュ）');mat.use_nodes=True
bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bs.inputs['Roughness'].default_value=.86
tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=img
mat.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color']);mat.node_tree.links.new(tex.outputs['Alpha'],bs.inputs['Alpha'])
mat.use_backface_culling=True
if hasattr(mat,'surface_render_method'): mat.surface_render_method='DITHERED'

class Mesh:
    def __init__(self):self.v=[];self.f=[];self.uv=[]
    def card(self,c,u,v,tile):
        c=np.array(c);u=np.array(u);v=np.array(v);base=len(self.v)
        points=[c-u/2-v/2,c+u/2-v/2,c+u/2+v/2,c-u/2+v/2];self.v.extend(points+points)
        x=tile%4/4;y=tile//4/4
        coords=[(x+.001,y+.001),(x+.249,y+.001),(x+.249,y+.249),(x+.001,y+.249)];self.uv.extend(coords+coords)
        self.f.extend([(base,base+2,base+1),(base,base+3,base+2),(base+4,base+5,base+6),(base+4,base+6,base+7)])
    def stem(self,points,width):
        for p,q in zip(points[:-1],points[1:]):
            p=np.array(p);q=np.array(q);d=q-p;cross=np.cross(d,[0,1,0]);cross=cross/(np.linalg.norm(cross)+1e-9)*width/2
            base=len(self.v);self.v.extend([p-cross,p+cross,q+cross,q-cross]*2);self.uv.extend([(.76,.76),(.98,.76),(.98,.98),(.76,.98)]*2)
            self.f.extend([(base,base+1,base+2),(base,base+2,base+3),(base+4,base+6,base+5),(base+4,base+7,base+6)])
    def object(self,name,col):
        me=bpy.data.meshes.new(name);me.from_pydata(self.v,[],self.f);me.update()
        uv=me.uv_layers.new(name='UVMap')
        for p in me.polygons:
            for li in p.loop_indices:
                a=self.uv[me.loops[li].vertex_index];uv.data[li].uv=(a[0],1-a[1])
        ob=bpy.data.objects.new(name,me);col.objects.link(ob);me.materials.append(mat);return ob

def crown_height(x,k):
    # 2.5 m周期の連続した丸い冠。壁上端から1.8〜2.25m。
    return 2.02+.18*math.cos(2*math.pi*x/2.5)+.08*math.sin(4*math.pi*x/2.5+k*.6)

def bottom(x,k):
    # 端の高さ・位相は3変種共通。連結部に段差を作らない。
    edge=min(1,max(0,(1.1-abs(x))/.35))
    return 1.72+.20*math.cos(x*3.1)+edge*(.18*math.sin(x*8+k*1.7)-.14*math.sin(x*4+k))

stats=[];scenes=[]
for k,c in enumerate('abc'):
    sc=scene if k==0 else bpy.data.scenes.new('蔓草'+c.upper()+'（壁は含まない）');sc.unit_settings.system='METRIC';sc.unit_settings.scale_length=1;sc.use_fake_user=True;scenes.append(sc)
    col=bpy.data.collections.new('01 蔓草'+c.upper());sc.collection.children.link(col)
    m=Mesh();rng=np.random.default_rng(500+k)
    # 全変種で左右端に同じ葉群を置き、繰り返しの継ぎ目を揃える。
    for n in range(2700):
        x=rng.uniform(-1.22,1.22)
        if n<850:
            # 壁の上半分を厚く覆う前面。壁面から0.30〜0.60mの前方。
            z=rng.uniform(bottom(x,k)+.08,3.18)
            y=.33*z-rng.uniform(.36,.49)
            u=np.array([1,rng.uniform(-.25,.25),rng.uniform(-.3,.3)])
            v=np.array([rng.uniform(-.3,.3),rng.uniform(-.3,.3),-1])
        elif n<2100:
            # 薄い幕から、奥行き2m強の丸い冠と前方へ張り出す前面へ。
            theta=rng.uniform(.015,math.pi-.015)
            y=1.55-1.12*math.cos(theta)
            z=3.03+crown_height(x,k)*math.sin(theta)
            ripple=.08*math.sin(x*8+theta*4+k)+rng.uniform(-.07,.07)
            y+=ripple*math.cos(theta);z+=ripple*math.sin(theta)
            u=np.array([1,rng.uniform(-.25,.25),rng.uniform(-.2,.2)])
            v=np.array([rng.uniform(-.25,.25),math.sin(theta),math.cos(theta)])
        else:
            # 単体の側面も透けないよう、内部の葉群を推定して補う。
            x=rng.uniform(-1.22,1.22);theta=rng.uniform(.05,math.pi-.05)
            r=math.sqrt(rng.uniform(.2,1))
            y=1.55-1.12*math.cos(theta)*r
            z=3.03+crown_height(x,k)*math.sin(theta)*r
            u=np.array([rng.uniform(-.2,.2),1,rng.uniform(-.4,.4)])
            v=np.array([rng.uniform(-.2,.2),rng.uniform(-.4,.4),-1])
        size=rng.uniform(.24,.35)
        u=u/np.linalg.norm(u)*size;v=v/np.linalg.norm(v)*size*rng.uniform(1.05,1.25)
        if n>=850:
            rot=rng.uniform(-math.pi,math.pi);u,v=u*math.cos(rot)+v*math.sin(rot),v*math.cos(rot)-u*math.sin(rot)
        x=float(np.clip(x,-1.25+abs(u[0])/2+abs(v[0])/2,1.25-abs(u[0])/2-abs(v[0])/2))
        m.card([x,y,z],u,v,int(rng.integers(0,15)))
    # 左右共通の端葉を25枚配置。端が一直線にならない小さな空隙を残す。
    for s in [-1,1]:
        edge_rng=np.random.default_rng(867)
        for n in range(25):
            z=edge_rng.uniform(1.67,3.38);y=.33*z-.2
            m.card([s*1.205,y,z],[.09,0,0],[0,.025,-.22],n%15)
    # 葉身の先端を含め、壁前方への張り出しを0.30〜0.60mに収める。
    for p in m.v:
        if p[2]<=3.0 and p[1]<.33*p[2]-.05:p[1]=np.clip(p[1],.33*p[2]-.60,.33*p[2]-.30)
        elif p[2]>3.0:p[1]=max(.39,p[1])
    # 自重で曲がるつる。露出した下端は細く、上の葉群へ連続。
    for n in range(42):
        x=rng.uniform(-1.18,1.18);bz=max(1.31,bottom(x,k)-rng.uniform(.05,.18))
        points=[]
        for j in range(6):
            t=j/5;z=3.25+(bz-3.25)*t;xx=x+.045*math.sin(t*7+n)
            points.append([xx,.33*z-.12-.04*math.sin(t*math.pi),z])
        m.stem(points,.006)
    ob=m.object('葉の幕・上の低木_'+c,col)
    v=np.asarray(m.v,np.float64);f=np.asarray(m.f,np.int64);uv=np.asarray(m.uv,np.float64)
    assert len(f)<=12000 and np.isfinite(v).all()
    np.savez_compressed(OUT/f'vines_{c}.npz',v=v,f=f,uv=uv)
    info={'name':f'vines_{c}','triangles':len(f),'vertices':len(v),'bounds_m':[v.min(0).tolist(),v.max(0).tolist()],'bottom_envelope_m':[float(min(bottom(x,k) for x in np.linspace(-1.25,1.25,100))),float(max(bottom(x,k) for x in np.linspace(-1.25,1.25,100)))]};stats.append(info)
    # 撮影用の世界・カメラ・灯りはアセットと別コレクション。
    stage=bpy.data.collections.new('90 撮影用（書出対象外）_'+c);sc.collection.children.link(stage)
    world=bpy.data.worlds.new('曇天_'+c);world.use_nodes=True;next(n for n in world.node_tree.nodes if n.type=='BACKGROUND').inputs[0].default_value=(.21,.24,.28,1);next(n for n in world.node_tree.nodes if n.type=='BACKGROUND').inputs[1].default_value=.8;sc.world=world
    ld=bpy.data.lights.new('大きい面光源_'+c,'AREA');lo=bpy.data.objects.new(ld.name,ld);stage.objects.link(lo);lo.location=(1,-4,7);ld.energy=1100;ld.shape='DISK';ld.size=5;lo.rotation_euler=(Vector((0,1.2,2.8))-lo.location).to_track_quat('-Z','Y').to_euler()
    camd=bpy.data.cameras.new('確認カメラ_'+c);cam=bpy.data.objects.new(camd.name,camd);stage.objects.link(cam);cam.location=(5,-8,5.5);cam.rotation_euler=(Vector((0,1.3,3.3))-cam.location).to_track_quat('-Z','Y').to_euler();camd.type='ORTHO';camd.ortho_scale=5.8;sc.camera=cam
    sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=12;sc.render.threads_mode='FIXED';sc.render.threads=3;sc.render.resolution_x=720;sc.render.resolution_y=600;sc.render.resolution_percentage=100;sc.view_settings.view_transform='Standard';sc.render.image_settings.file_format='PNG';sc.render.film_transparent=False

# 初期画面はA。各sceneが原点0で1変種を持ち、メッシュに配置変換を含めない。
bpy.context.window.scene=scenes[0]
bpy.context.preferences.filepaths.save_version=0
bpy.context.preferences.filepaths.file_preview_type = "NONE"   # .blend の保存でサムネイルを作らない（tests/test_blender_save_no_thumbnail.py）
bpy.ops.wm.save_as_mainfile(filepath=str(SRC/'asama_wall_vines_v2.blend'))
meta={'asset_id':'asama_wall_vines','source_blend':'asama_wall_vines_v2.blend','generator':'blender_scripts/build_asama_wall_vines.py','units':'m','axes':{'up':'+Z','front':'-Y','along_wall':'+X','origin':'壁の根元線の区切り中央'},'reference':{'panoid':'nG1sfblUAy-_p5ZKlAB6pw','date':'2025-07','images':[f'export/compare/asama/sv_a_{a:03}.png' for a in [0,45,90,315]],'urls':[f'https://www.google.com/maps/@35.4072027,139.215505,3a,90y,{a}h,85.87t/data=!3m6!1e1!3m4!1snG1sfblUAy-_p5ZKlAB6pw!2e0!7i16384!8i8192' for a in [0,45,90,315]],'fov_vertical_deg':90,'pitch_down_deg':4.13,'copyright':'Google Street View参照のみ。アセットへの画像転用なし。'},'dimension_basis':'壁高3.0m・勾配1:0.33・区切り長2.5mは依頼値。v2の冠高は壁上から葉先まで2.42〜2.46m、葉群全体の奥行は2.85〜2.87m。現地未実測。葉の大きさと輪郭・密度は写真から推定。壁は含まない。','wall_plane':'y=0.33*z, wall top y=0.99,z=3.0','shrub_form_basis':'壁上端Z=3.0mから約1.8〜2.25m高の楕円断面の冠を形成。前面は壁面から0.30〜0.60m突出。2.5m周期の起伏と立体的葉群。枝・内部・背面は推定。','variants':stats,'materials':{'atlas':'vines_atlas_v2.png','format':'RGBA 1024x1024, 自作の葉脈・葉形','two_sided':'各カードは逆順の三角形を明示的に複製。表裏の頂点も分離し法線相殺を防止。材質backface culling=True。','uv':'npz v=0はPNG画像上。Blenderでは1-vで設定。'},'limitations':['背面と上面の内部構造・枝分かれは写真に見えないため推定。','植物種は断定せず、写真のつる性広葉と低木の混生外観を近似。','30m全長の写真ごとの連続輪郭は配置側で3変種を組み合わせる必要がある。','ゲーム内での材質・アルファ・色調は未確認（単体素材のみの依頼）。'],'status':{'source_saved':True,'npz_exported':True,'rgba_exported':True,'verification_reports':['export/compare/asama/vines2_readback_check.json','export/compare/asama/vines2_npz_check.json'],'preview_images':['export/compare/asama/vines2_views.jpg','export/compare/asama/vines2_reference_comparison.jpg'],'course_placement':False,'game_install':False}}
(SRC/'asset.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
(CMP/'vines2_export_check.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
# 単純な輪郭先行確認。--shape-onlyなら粘土表示の1枚だけを描画。
if '--shape-only' in sys.argv:
    for s in scenes:
        s.render.engine='BLENDER_WORKBENCH';s.display.shading.light='STUDIO';s.display.shading.color_type='SINGLE';s.display.shading.single_color=(.28,.43,.12);s.display.shading.show_shadows=True
    scenes[0].render.filepath=str(CMP/'vines2_shape_front.png');bpy.ops.render.render(write_still=True,scene=scenes[0].name)
elif '--export-only' not in sys.argv:
    for i,s in enumerate(scenes):
        s.render.filepath=str(CMP/f'vines2_variant_{"abc"[i]}.png');bpy.ops.render.render(write_still=True,scene=s.name)
    s=scenes[0];cam=s.camera
    for label,pos in [('front',(0,-7,3.3)),('left',(-7,1.4,3.3)),('right',(7,1.4,3.3)),('back',(0,8,3.3)),('top',(0,1.4,10))]:
        cam.location=pos;cam.rotation_euler=(Vector((0,1.4,3.3))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=5.8;s.render.filepath=str(CMP/f'vines2_{label}.png');bpy.ops.render.render(write_still=True,scene=s.name)
    # 写真の北・北東・東・北西方向。単体を同じ側から眺める方向比較。
    for angle in [0,45,90,315]:
        a=math.radians(angle);cam.location=(6*math.sin(a),1.3-6*math.cos(a),3.25);cam.rotation_euler=(Vector((0,1.3,3.3))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=5.8;s.render.filepath=str(CMP/f'vines2_direction_{angle:03}.png');bpy.ops.render.render(write_still=True,scene=s.name)
    s=scenes[0];cam=s.camera
    for c,x in [('b',-2.5),('c',2.5)]:
        source=bpy.data.objects['葉の幕・上の低木_'+c];copy=source.copy();s.collection.children[0].objects.link(copy);copy.location.x=x
    cam.location=(4,-10,4.7);cam.rotation_euler=(Vector((0,1.2,2.8))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=9.3
    s.render.resolution_x=1300;s.render.resolution_y=650;s.render.filepath=str(CMP/'vines2_three_joined.png');bpy.ops.render.render(write_still=True,scene=s.name)
print('VINES',json.dumps(stats))
