"""浅間神社の標柱・草株。写真は参照のみ、自作の材質を使用する。

材質: .venv/bin/python blender_scripts/build_asama_small_assets.py textures
制作: blender -b --factory-startup --python-exit-code 1 --python blender_scripts/build_asama_small_assets.py -- post
      同じコマンドの末尾をgrassとして草3種を生成。
確認: 保存blendを-bで開き、末尾をrender-post / render-grassとする。
比較表: .venv/bin/python blender_scripts/build_asama_small_assets.py sheets
ゲームNPZは画像上端v=0。撮影用の地面・照明は書き出さない。
"""
from pathlib import Path
import json
import math
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / 'export/compare/asama'
POST = ROOT / 'assets/landmarks/asama_shrine_post'
GRASS = ROOT / 'assets/landmarks/asama_grass_tufts'


def textures():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    rng = np.random.default_rng(301030)
    h, w = 2048, 1024
    yy, xx = np.mgrid[0:h, 0:w]
    # 縦繊維、雨筋、木目のうねり。写真の画素は使用しない。
    grain = np.zeros((h, w))
    for freq, amp in [(110, 5), (63, 6), (28, 5), (9, 3)]:
        grain += amp * np.sin(xx / freq * 10 + .7 * np.sin(yy / 210 + xx / 170))
    streak = rng.normal(0, 7, w)[None, :]
    noise = rng.normal(0, 2.6, (h, w))
    dirt = 25 * np.clip((yy / h - .73) / .27, 0, 1) ** 1.6
    # 幅の広い雨染みと灰色の繊維の帯を追加する。
    coarse = Image.fromarray(np.uint8(rng.uniform(0,255,(5,100))))
    coarse = np.asarray(coarse.resize((w,h),Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(3)),dtype=float)
    lum = grain*.5 + streak*.55 + noise - dirt + (coarse-127)*.37
    a = np.stack([174+lum, 176+lum, 163+lum], axis=-1)
    im = Image.fromarray(np.uint8(np.clip(a, 0, 255)), 'RGB')
    d = ImageDraw.Draw(im)
    # 天端用の木口、下端用の暗色も同じアトラスに格納。
    d.rectangle((0, 1940, 1023, 2047), fill=(137, 144, 127))
    for k in range(10):
        d.ellipse((30+k*8, 1944+k*2, 230-k*8, 2045-k*2), outline=(122+k, 130+k, 113+k))
    # 写真では印字は上端直下ではなく約1.8mから始まる。
    font_path = next(Path('/System/Library/Fonts').glob('*明朝*'))
    font = ImageFont.truetype(str(font_path), 172, index=2)
    ink = Image.new('RGBA', im.size, (0, 0, 0, 0))
    di = ImageDraw.Draw(ink)
    for side in (0, 1):
        for ch, z in zip('浅間神社', (1.76, 1.43, 1.10, .77)):
            cy = 32 + (1-z/2.4)*1880
            bb = di.textbbox((0, 0), ch, font=font)
            di.text((side*240+120-(bb[2]-bb[0])/2, cy-(bb[3]-bb[1])/2-bb[1]), ch, font=font, fill=(21, 26, 22, 245),stroke_width=1,stroke_fill=(21,26,22,240))
    ia = np.array(ink)
    ia[..., 3] = np.uint8(ia[..., 3].astype(float)*rng.uniform(.76, 1, (h, w)))
    im = Image.alpha_composite(im.convert('RGBA'), Image.fromarray(ia))
    # 細い割れ・節を文字と競合しない場所へ。
    d = ImageDraw.Draw(im)
    rr = random.Random(303)
    for side in range(4):
        for j in range(23):
            x = side*240 + rr.randint(9, 230)
            y = rr.randint(35, 1850)
            length = rr.randint(30, 240)
            d.line([(x,y),(x-1,y+length//2),(x+2,min(y+length,1930))], fill=(99,108,92,120), width=1)
    POST.mkdir(parents=True, exist_ok=True)
    im.save(POST/'post_atlas.png')
    # 草は輪郭が透過する4色の葉。中心脈と細い平行脈を自作。
    h, w = 1024, 512
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx % 128)/127
    t = 1-yy/(h-1)
    edge = np.clip((.49-np.abs(u-.5))*50, 0, 1)
    alpha = np.uint8(edge*255)
    palettes = np.array([[121,148,43],[151,170,57],[95,127,36],[167,166,76]])
    col = palettes[xx//128].astype(float)
    variation = (7*np.sin(xx*2)+rng.normal(0,2,(h,w)) + t*12-8*(np.abs(u-.5)*2))[...,None]
    col += variation
    midrib = np.exp(-((u-.5)/.025)**2)*18
    col += midrib[...,None]
    rgba = np.concatenate((np.uint8(np.clip(col,0,255)), alpha[...,None]), axis=2)
    GRASS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba,'RGBA').save(GRASS/'grass_atlas.png')


def init_scene():
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.unit_settings.system = 'METRIC'
    s.unit_settings.scale_length = 1
    s.render.engine = 'CYCLES'
    s.cycles.samples = 24
    s.cycles.use_denoising = True
    s.render.threads_mode = 'FIXED'
    s.render.threads = 3
    s.render.resolution_x = 650
    s.render.resolution_y = 700
    s.render.resolution_percentage = 100
    s.view_settings.view_transform = 'Standard'
    world = bpy.data.worlds.new('柔らかい屋外光')
    world.use_nodes = True
    world.node_tree.nodes.get('Background').inputs[0].default_value=(.36,.40,.46,1)
    world.node_tree.nodes.get('Background').inputs[1].default_value=.7
    s.world = world
    return s


def material(path, name, alpha=False):
    import bpy
    m = bpy.data.materials.new(name); m.use_nodes=True
    bs = next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    bs.inputs['Roughness'].default_value=.88
    tex=m.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image=bpy.data.images.load(str(path));tex.image.pack()
    m.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
    if alpha:
        m.surface_render_method='DITHERED'
        m.use_backface_culling=True
        transluc=m.node_tree.nodes.new('ShaderNodeBsdfTranslucent')
        mix=m.node_tree.nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=.24
        m.node_tree.links.new(tex.outputs['Color'],transluc.inputs['Color'])
        m.node_tree.links.new(bs.outputs[0],mix.inputs[1]);m.node_tree.links.new(transluc.outputs[0],mix.inputs[2])
        output=next(n for n in m.node_tree.nodes if n.type=='OUTPUT_MATERIAL')
        transparent=m.node_tree.nodes.new('ShaderNodeBsdfTransparent')
        cutout=m.node_tree.nodes.new('ShaderNodeMixShader')
        m.node_tree.links.new(tex.outputs['Alpha'],cutout.inputs[0])
        m.node_tree.links.new(transparent.outputs[0],cutout.inputs[1])
        m.node_tree.links.new(mix.outputs[0],cutout.inputs[2])
        m.node_tree.links.new(cutout.outputs[0],output.inputs['Surface'])
    return m


def mesh_object(name, verts, faces, uvs, col, mat):
    import bpy
    me=bpy.data.meshes.new(name);me.from_pydata(verts,[],faces);me.update()
    ob=bpy.data.objects.new(name,me);col.objects.link(ob)
    uv=me.uv_layers.new(name='UVMap')
    for p, puv in zip(me.polygons,uvs):
        for li, co in zip(p.loop_indices,puv):
            uv.data[li].uv=(co[0],1-co[1])
    me.materials.append(mat)
    return ob


def post_geometry(col, mat):
    # 角を3mm面取りした0.20m角材。上端に小さな摩耗を入れる。
    a=.10;b=.003
    ring=[(-a+b,-a),(a-b,-a),(a,-a+b),(a,a-b),(a-b,a),(-a+b,a),(-a,a-b),(-a,-a+b)]
    verts=[]
    zs=[0,.08,.7,1.4,2.395,2.4]
    for ri,z in enumerate(zs):
        for x,y in ring:
            f=.98 if ri==5 else 1
            verts.append((x*f,y*f,z))
    faces=[];uvs=[]
    for j in range(len(zs)-1):
        for k in range(8):
            side=k//2
            u0=(side*240+4)/1024;u1=(side*240+236)/1024
            if k%2:
                u0=(side*240+235)/1024;u1=(side*240+239)/1024
            v0=(32+(1-zs[j]/2.4)*1880)/2048
            v1=(32+(1-zs[j+1]/2.4)*1880)/2048
            faces.append((j*8+k,j*8+(k+1)%8,(j+1)*8+(k+1)%8,(j+1)*8+k))
            uvs.append([(u0,v0),(u1,v0),(u1,v1),(u0,v1)])
    for ri,rev in [(0,True),(len(zs)-1,False)]:
        ids=list(range(ri*8,ri*8+8));ids=ids[::-1] if rev else ids
        faces.append(ids)
        uvs.append([(.03+(verts[i][0]+a)/.2*.18,.95+(verts[i][1]+a)/.2*.045) for i in ids])
    return mesh_object('標柱本体_浅間神社',verts,faces,uvs,col,mat)


def grass_geometry(i,col,mat):
    import numpy as np
    rng=random.Random(1200+i)
    height=[.55,.79,1.04][i]
    count=[64,67,70][i]
    verts=[];faces=[];uvs=[]
    for n in range(count):
        angle=n*2.399963+rng.uniform(-.16,.16)
        # 中央は直立、外側ほど大きく弧を描いて垂れる。
        inner=n%5==0
        peak=height*rng.uniform(.25,.64) if n>=20 else height*rng.uniform(.64,1.0)
        reach=height*rng.uniform(.18,.43) if inner else height*rng.uniform(.43,.83)*[1,.85,.77][i]
        droop=rng.uniform(.10,.24) if inner else rng.uniform(.34,.58)
        origin=np.array([math.cos(angle)*rng.uniform(.005,.06),math.sin(angle)*rng.uniform(.005,.06),rng.uniform(0,.014)])
        axis=np.array([math.cos(angle),math.sin(angle),0.0])
        side=np.array([-math.sin(angle),math.cos(angle),rng.uniform(-.18,.18)])
        base=len(verts)
        seg=8 if n<20 else 4
        blade_width=rng.uniform(.012,.018)
        palette=n%4
        for j in range(seg+1):
            t=math.sin(j/seg*math.pi*.5)
            z=peak*(math.sin(t*math.pi*.5)-droop*t**3)
            # 高さ指定は葉の最高点。葉先も折線でなく連続した弧。
            center=origin+axis*(reach*t**1.6)+np.array([0,0,z])
            width=blade_width*(1-t)**.65*(.45+.55*min(t*5,1))+.0001
            for sign in (-1,1):
                verts.append(tuple(center+side*sign*width/2))
        front=list(verts[base:]);verts.extend(front)
        for j in range(seg):
            ids=[base+j*2,base+j*2+1,base+(j+1)*2+1,base+(j+1)*2]
            us=[((palette*128+3)/512,1-j/seg),((palette*128+125)/512,1-j/seg),((palette*128+125)/512,1-(j+1)/seg),((palette*128+3)/512,1-(j+1)/seg)]
            faces.append(ids);uvs.append(us)
            faces.append([k+2*(seg+1) for k in ids[::-1]]);uvs.append(us[::-1])
    ob=mesh_object(f'草株_{"ABC"[i]}_両面',verts,faces,uvs,col,mat)
    # 最高点の寸法を指定高さに合わせる。原点は地面のまま。
    zmax=max(v.co.z for v in ob.data.vertices)
    for v in ob.data.vertices:v.co.z=max(0,v.co.z*height/zmax)
    return ob


def export_npz(ob,out):
    import bpy
    import numpy as np
    me=ob.data;me.calc_loop_triangles();uv=me.uv_layers.active.data
    vs=[];us=[];fs=[];keys={}
    for tri in me.loop_triangles:
        f=[]
        for li in tri.loops:
            vi=me.loops[li].vertex_index
            u=tuple(float(x) for x in uv[li].uv)
            k=(vi,u)
            if k not in keys:
                keys[k]=len(vs);vs.append(tuple(me.vertices[vi].co));us.append((u[0],1-u[1]))
            f.append(keys[k])
        fs.append(f)
    np.savez_compressed(out,v=np.array(vs,dtype=np.float64),f=np.array(fs,dtype=np.int64),uv=np.array(us,dtype=np.float64))
    return {'name':out.stem,'triangles':len(fs),'vertices':len(vs),'bounds_m':[np.min(vs,axis=0).tolist(),np.max(vs,axis=0).tolist()]}


def setup_stage(s,col):
    import bpy
    from mathutils import Vector
    for name,loc,power,size in [('太陽側',(-3,-4,6),650,5),('空の補助',(4,1,5),350,4)]:
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
        o=bpy.data.objects.new(name,data);col.objects.link(o);o.location=loc
        o.rotation_euler=(Vector((0,0,1))-o.location).to_track_quat('-Z','Y').to_euler()
    data=bpy.data.cameras.new('確認カメラ');cam=bpy.data.objects.new('確認カメラ',data);col.objects.link(cam);s.camera=cam
    # Blenderを開いたときも対象が収まる。
    data.type='ORTHO';data.ortho_scale=2.9;cam.location=(3,-5,2.9)
    cam.rotation_euler=(Vector((0,0,1.2))-cam.location).to_track_quat('-Z','Y').to_euler()
    s.render.film_transparent=False


def build(kind):
    import bpy
    import shutil
    s=init_scene()
    col=bpy.data.collections.new('01 標柱' if kind=='post' else '01 草株');s.collection.children.link(col)
    stage=bpy.data.collections.new('90 撮影用（非出力）');s.collection.children.link(stage)
    asset_id='asama_shrine_post' if kind=='post' else 'asama_grass_tufts'
    folder=POST if kind=='post' else GRASS
    out=ROOT/'data/processed/landmarks'/asset_id;out.mkdir(parents=True,exist_ok=True)
    tex=folder/('post_atlas.png' if kind=='post' else 'grass_atlas.png')
    mat=material(tex,'風化した木と墨文字' if kind=='post' else '草葉_RGBA',kind=='grass')
    if kind=='post':
        obs=[post_geometry(col,mat)]
        cols=[col]
    else:
        s.collection.children.unlink(col);bpy.data.collections.remove(col)
        cols=[];obs=[]
        for i in range(3):
            c=bpy.data.collections.new(f'01 草株 {"ABC"[i]}');s.collection.children.link(c);cols.append(c)
            obs.append(grass_geometry(i,c,mat))
    reports=[]
    for i,ob in enumerate(obs):
        stem='post' if kind=='post' else f'grass_{"abc"[i]}'
        reports.append(export_npz(ob,out/(stem+'.npz')))
        shutil.copyfile(tex,out/(stem+'.png'))
    setup_stage(s,stage)
    # 各変種を独立sceneへ保存。すべて根元原点を保持し、重ねて表示しない。
    if kind=='grass':
        for i in range(1,3):
            ns=s.copy();ns.name=f'grass_{"abc"[i]}';ns.use_fake_user=True
            for c in cols:
                if c!=cols[i]:ns.collection.children.unlink(c)
        for c in cols[1:]:s.collection.children.unlink(c)
        s.name='grass_a';s.use_fake_user=True
    else:s.name='post';s.use_fake_user=True
    bpy.context.preferences.filepaths.save_version=0
    bpy.context.preferences.filepaths.file_preview_type = "NONE"   # .blend の保存でサムネイルを作らない（tests/test_blender_save_no_thumbnail.py）
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/(asset_id+'.blend')))
    info={
        'asset_id':asset_id,'source':asset_id+'.blend','generator':'blender_scripts/build_asama_small_assets.py',
        'coordinates':{'unit':'m','up':'+Z','front':'-Y','origin':'根元中央','east':'+X'},
        'references':{'panoid':'nG1sfblUAy-_p5ZKlAB6pw','date':'2025-07','fov_vertical_deg':90,'pitch_down_deg':4.13,'images':[f'export/compare/asama/sv_a_{h}.png' for h in ['000','045','090','315']], 'urls':[f'https://www.google.com/maps/@35.4072027,139.215505,3a,90y,{h}h,85.87t/data=!3m6!1e1!3m4!1snG1sfblUAy-_p5ZKlAB6pw!2e0!7i16384!8i8192' for h in [0,45,90,315]],'usage':'比較参照のみ。制作元・テクスチャに写真を含めない。'},
        'dimension_basis':'写真の柱の高さ幅比・階段に対する大きさから推定。未実測。0.20m角×2.40m。' if kind=='post' else '写真の長い葉の株を参照。3変種の最高点0.55/0.79/1.04mは指定範囲から設計。未実測。',
        'parts':reports,'texture':tex.name,'npz':{'keys':['v','f','uv'],'v':'m,+Z上,正面-Y','uv':'v=0画像上端','f':'三角形'},
        'double_sided':'閉じた角柱' if kind=='post' else '裏面は頂点を分けた逆巻き面で明示的に二重化。三角形数に裏面を含む。',
        'approximations':['見えない北面・西面・木口・底面は推定。','文字は読めた浅間神社のみ。既存の明朝体を近似に使い、筆跡の完全一致はしない。','木目・割れ・汚れは写真の分布を参考にした自作手続き材質。'] if kind=='post' else ['植物種は写真から未同定。細長い弓状の葉の株として近似。','葉の本数・裏面・個々の曲がり方は推定。穂は写真で明確でないため作らない。'],
        'game_status':'NPZとPNGまで。コースへの配置・KN5・実ゲーム表示は依頼範囲外のため未実施。',
        'checks':['export/compare/asama/post_validation.json'] if kind=='post' else [f'export/compare/asama/grass_{c}_validation.json' for c in 'abc'],
        'delivery_check':'export/compare/asama/post_grass_vines_delivery_check.json',
        'npz_preview':[f'export/compare/asama/{kind}_npz_{i}.png' for i in range(len(obs))],
        'comparison':'export/compare/asama/'+kind+'_reference_comparison.jpg',
        'preview':'export/compare/asama/'+kind+'_views.jpg',
    }
    (folder/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(reports,ensure_ascii=False))
    return reports


def render_views(kind):
    import bpy
    from mathutils import Vector
    views=[('front',(0,-6,0)),('right',(6,0,0)),('left',(-6,0,0)),('back',(0,6,0)),('top',(0,0,6)),('persp',(4,-6,2.4)),('sv000',(3.5,-7,1.45) if kind=='post' else (1,-5,2.8)),('sv315',(3.5,-7,1.45) if kind=='post' else (1,-5,2.8))]
    paths=[]
    scenes=[bpy.data.scenes['post']] if kind=='post' else [bpy.data.scenes[f'grass_{c}'] for c in 'abc']
    for i,s in enumerate(scenes):
        bpy.context.window.scene=s
        cam=s.camera;cam.data.type='ORTHO'
        s.cycles.samples=16
        s.render.resolution_x=350 if kind=='post' else 520
        s.render.resolution_y=700 if kind=='post' else 520
        target=Vector((0,0,1.2 if kind=='post' else [.55,.79,1.04][i]*.45))
        scale=2.7 if kind=='post' else [1.15,1.65,2.1][i]
        for label,direction in views:
            cam.location=target+Vector(direction)
            cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
            cam.data.ortho_scale=scale
            s.render.filepath=str(CMP/f'{kind}_{i}_{label}.png')
            bpy.ops.render.render(write_still=True);paths.append(s.render.filepath)
    return paths


def sheets():
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    font_path=next(Path('/System/Library/Fonts').glob('*明朝*'))
    font=ImageFont.truetype(str(font_path),20)
    small=ImageFont.truetype(str(font_path),16)
    labels={'front':'正面 -Y','right':'東 +X','left':'西 -X（推定）','back':'背面 +Y（推定）','top':'真上（推定）','persp':'斜め'}
    for kind,n in [('post',1),('grass',3)]:
        tw,th=(240,520) if kind=='post' else (265,300)
        out=Image.new('RGB',(tw*6,th*n+50),'#eef0ed');d=ImageDraw.Draw(out)
        d.text((15,10),'浅間神社 標柱 / 六方向確認' if kind=='post' else '浅間神社 草株 A/B/C / 六方向確認',font=font,fill='black')
        for i in range(n):
            for j,(view,label) in enumerate(labels.items()):
                im=Image.open(CMP/f'{kind}_{i}_{view}.png').convert('RGB')
                im=ImageOps.contain(im,(tw,th-40))
                out.paste(im,(j*tw+(tw-im.width)//2,50+i*th))
                d.text((j*tw+8,50+i*th+th-32),('ABC'[i]+' ' if kind=='grass' else '')+label,font=small,fill='black')
        out.save(CMP/f'{kind}_views.jpg',quality=94)
    for kind in ['post','grass']:
        out=Image.new('RGB',(1400,1160),'#eef0ed');d=ImageDraw.Draw(out)
        d.text((20,10),f'浅間神社 {"木製標柱" if kind=="post" else "細長い葉の草株"}：SV 000 / 315 局部比較',font=font,fill='black')
        for row,heading in enumerate(['000','315']):
            ref=Image.open(CMP/f'sv_a_{heading}.png').convert('RGB')
            y=50+row*530
            # 全景も併記して、写真の出典表示を残す。
            full=ref.resize((600,400));out.paste(full,(10,y))
            d.text((12,y+407),f'参照 SV {heading} / Google Maps 2025-07',font=small,fill='black')
            if kind=='post':
                crop=(405,330,480,710) if heading=='000' else (838,345,908,665)
                detail=ImageOps.contain(ref.crop(crop),(190,450))
                model=ImageOps.contain(Image.open(CMP/f'post_0_sv{heading}.png').convert('RGB'),(300,450))
                out.paste(detail,(660,y));out.paste(model,(950,y))
                d.text((660,y+465),'写真の拡大',font=small,fill='black');d.text((950,y+465),'制作モデル（視線推定）',font=small,fill='black')
            else:
                crop=(700,580,900,740) if heading=='000' else (1270,635,1500,835)
                detail=ImageOps.contain(ref.crop(crop),(350,300))
                model=ImageOps.contain(Image.open(CMP/f'grass_{row}_sv{heading}.png').convert('RGB'),(400,400))
                out.paste(detail,(620,y+60));out.paste(model,(990,y+20))
                d.text((620,y+420),'写真の株の拡大',font=small,fill='black');d.text((990,y+450),'制作モデル（視線推定）',font=small,fill='black')
        d.text((20,1120),'局部形状・寸法比の比較。配置は変更せず、カメラ距離・照明は近似。写真を材質に使用していない。',font=small,fill='black')
        out.save(CMP/f'{kind}_reference_comparison.jpg',quality=94)


def npz_preview(kind):
    """制作元ではなく、ゲーム用データから材質とメッシュを復元して確認する。"""
    import bpy
    import numpy as np
    from mathutils import Vector
    s=init_scene()
    stage=bpy.data.collections.new('90 NPZ確認撮影');s.collection.children.link(stage);setup_stage(s,stage)
    asset='asama_shrine_post' if kind=='post' else 'asama_grass_tufts'
    out=ROOT/'data/processed/landmarks'/asset
    stems=['post'] if kind=='post' else ['grass_a','grass_b','grass_c']
    paths=[]
    for i,stem in enumerate(stems):
        col=bpy.data.collections.new('01 '+stem);s.collection.children.link(col)
        with np.load(out/(stem+'.npz')) as data:
            v,f,uv=data['v'],data['f'],data['uv']
            mesh_object(stem,v.tolist(),f.tolist(),[[uv[j].tolist() for j in face] for face in f],col,material(out/(stem+'.png'),stem+'_game',kind=='grass'))
        target=Vector((0,0,1.2 if kind=='post' else [.55,.79,1.04][i]*.45))
        s.camera.data.type='ORTHO';s.camera.data.ortho_scale=2.7 if kind=='post' else [1.15,1.65,2.1][i]
        s.camera.location=target+Vector((3.5,-7,1.45) if kind=='post' else (1,-5,2.8))
        s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler()
        s.render.resolution_x=350 if kind=='post' else 520;s.render.resolution_y=700 if kind=='post' else 520
        s.cycles.samples=16;s.render.filepath=str(CMP/f'{kind}_npz_{i}.png')
        bpy.ops.render.render(write_still=True);paths.append(s.render.filepath)
        for ob in list(col.objects):bpy.data.objects.remove(ob,do_unlink=True)
        bpy.data.collections.remove(col)
    return paths


if __name__=='__main__':
    if 'textures' in sys.argv:textures()
    elif 'sheets' in sys.argv:sheets()
    elif 'render-post' in sys.argv:render_views('post')
    elif 'render-grass' in sys.argv:render_views('grass')
    elif 'post' in sys.argv:build('post')
    elif 'grass' in sys.argv:build('grass')
