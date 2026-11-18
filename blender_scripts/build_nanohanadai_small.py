"""菜の花台のミラー・車止め・看板・歩道小物。写真の画素は使用しない。"""
from pathlib import Path
import math,random
ROOT=Path(__file__).resolve().parents[1]

def textures():
    import numpy as np
    from PIL import Image,ImageDraw,ImageFilter
    root=ROOT/'assets/landmarks'
    for asset in ['nanohanadai_mirror','nanohanadai_fence_signs','nanohanadai_road_planters']:(root/asset).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1030)
    # ミラーは静的な銀灰色の空/地面の明度勾配。参照写真を鏡面に転用しない。
    yy,xx=np.mgrid[0:512,0:512];t=yy/511
    bands=np.where(t<.5, .85-t*.45, .47+(t-.5)*.40)
    a=np.stack([bands*.87,bands*.96,bands],-1)*255
    a+=rng.normal(0,.65,(512,512,1))
    Image.fromarray(np.uint8(np.clip(a,0,255))).save(root/'nanohanadai_mirror/mirror_surface.png')
    im=Image.new('RGB',(768,512),(223,216,169));d=ImageDraw.Draw(im)
    for x in [30,264,498]:d.polygon([(x,20),(x+105,20),(x+229,256),(x+105,492),(x,492),(x+124,256)],fill=(184,40,27))
    for _ in range(900):
        x,y=rng.integers(0,768),rng.integers(0,512);d.ellipse((x,y,x+2,y+1),fill=(208,199,165))
    im.save(root/'nanohanadai_mirror/arrow_art.png')
    # 判読不能の注意文は文字を創作せず、短い線群として近似する。
    for name,size in [('yellow_sign',(450,550)),('red_white_sign',(900,450))]:
        im=Image.new('RGB',size,(224,202,65) if name=='yellow_sign' else (220,218,198));d=ImageDraw.Draw(im);w,h=size
        d.rectangle((3,3,w-4,h-4),outline=(94,87,55),width=7)
        if name=='yellow_sign':
            d.ellipse((100,35,350,285),outline=(55,100,128),width=15)
            d.polygon([(150,220),(205,95),(255,95),(310,220)],fill=(68,109,135));d.line((95,55,355,290),fill=(181,38,30),width=27)
            for y in [330,367,404,456,490]:
                for x in range(35,w-30,39):d.line((x,y,x+21,y),fill=(55,57,44),width=8)
        else:
            d.rectangle((14,18,w-15,65),fill=(165,40,33))
            for y in [92,132]:
                for x in range(38,w-40,55):d.line((x,y,x+34,y),fill=(160,45,39),width=13)
            for y in range(195,407,41):
                for x in range(40,w-50,40):d.line((x,y,x+24,y),fill=(69,66,55),width=7)
        im.save(root/'nanohanadai_fence_signs'/(name+'_art.png'))
    # 道路側に見える赤白市松の円筒。用途は写真だけで断定しない。
    im=Image.new('RGB',(768,512),(223,218,191));d=ImageDraw.Draw(im)
    for j in range(4):
        for i in range(12):
            if (i+j)%2==0:d.rectangle((i*64,j*128,(i+1)*64,(j+1)*128),fill=(173,55,49))
    im.save(root/'nanohanadai_road_planters/checker_art.png')

def build(asset_id):
    import bpy
    import nanohanadai_common as C
    parts={};offsets={};refs=[];approx=[]
    stone=C.proc('灰褐色の風化コンクリート',(159,158,144),scale=13)
    pale=C.proc('側面の白灰色',(185,183,166),scale=8)
    moss=C.proc('根元の苔',(88,99,52),scale=21)
    wood=C.proc('灰褐色に風化した木',(119,112,87),'wood',4)
    silver=C.proc('鈍い亜鉛金属',(143,150,145),scale=8)
    def start(name):return C.collection(name)
    def end(name,col,offset=(0,0,0)):parts[name]=list(col.objects);offsets[name]=offset
    def chip(name,size,mat,seed):
        rr=random.Random(seed);w,d,h=size
        # 8隅の局所摩耗を2層の八角リングで作る。単なる直方体と区別。
        b=.025;ring=[(-w/2+b,-d/2),(w/2-b,-d/2),(w/2,-d/2+b),(w/2,d/2-b),(w/2-b,d/2),(-w/2+b,d/2),(-w/2,d/2-b),(-w/2,-d/2+b)]
        verts=[]
        for z in [0,h]:
            for x,y in ring:verts.append((x+rr.uniform(-.012,.012),y+rr.uniform(-.008,.008),z if z==0 else z+rr.uniform(-.015,.009)))
        faces=[tuple(range(7,-1,-1)),tuple(range(8,16))]+[(k,(k+1)%8,(k+1)%8+8,k+8) for k in range(8)]
        return C.mesh(name,verts,faces,mat)
    if asset_id=='nanohanadai_mirror':
        col=start('01 カーブミラー')
        orange=C.proc('退色した橙の塗装',(193,91,38),scale=11)
        C.cylinder('橙色柱',(0,.095,1.36),.035,2.72,orange,12)
        C.box('根元コンクリート',(0,.095,.055),(.22,.22,.11),stone,.015)
        # 球面状の凸鏡。XYでなくXZの同心円を前面-Yへ張る。
        n=40;r=.335;verts=[(0,-.055,2.60)];faces=[]
        for j in range(1,5):
            rad=r*j/4
            for i in range(n):
                a=2*math.pi*i/n;verts.append((rad*math.cos(a),-.055+.060*(rad/r)**2,2.60+rad*math.sin(a)))
        for i in range(n):faces.append((0,1+i,1+(i+1)%n))
        for j in range(3):
            for i in range(n):a=1+j*n+i;b=1+j*n+(i+1)%n;faces.append((a,b,b+n,a+n))
        mat=C.image_material('自作銀灰色鏡面',ROOT/'assets/landmarks'/asset_id/'mirror_surface.png')
        o=C.mesh('凸面の鏡',verts,faces,mat)
        for p in o.data.polygons:
            for li in p.loop_indices:
                co=o.data.vertices[o.data.loops[li].vertex_index].co;o.data.uv_layers['UVMap'].data[li].uv=(.5+co.x/(2*r),.5+(co.z-2.60)/(2*r))
        back=C.cylinder('灰色背面',(0,.030,2.60),r+.014,.065,silver,40);back.rotation_euler[0]=math.pi/2
        for i in range(12):
            a=2*math.pi*i/12;C.pipe('背面の放射リブ',(0,.068,2.60),(.30*math.cos(a),.068,2.60+.30*math.sin(a)),.008,silver)
        for i in range(40):
            a=2*math.pi*i/40;b=2*math.pi*(i+1)/40
            C.pipe('外周橙枠',(.347*math.cos(a),-.002,2.60+.347*math.sin(a)),(.347*math.cos(b),-.002,2.60+.347*math.sin(b)),.012,orange,6)
        # 雨よけは上半円だけの薄い金属帯。
        for i in range(20):
            a=math.pi*i/20;b=math.pi*(i+1)/20
            C.mesh('上縁のひさし',[(.36*math.cos(a),-.075,2.60+.36*math.sin(a)),(.36*math.cos(b),-.075,2.60+.36*math.sin(b)),(.36*math.cos(b),.045,2.60+.36*math.sin(b)),(.36*math.cos(a),.045,2.60+.36*math.sin(a))],[(0,1,2,3)],orange)
        for ob in col.objects:ob.location.y-=.095
        end('mirror',col)
        col=start('02 ガードレール用矢印板')
        C.box('矢印板の裏',(0,0,.30),(.72,.025,.60),silver,.008)
        mat=C.image_material('自作赤白矢印',ROOT/'assets/landmarks'/asset_id/'arrow_art.png');C.front_panel('矢印印刷',(0,-.014,.30),.70,.58,mat)
        for x in [-.23,.23]:C.box('取付金具',(x,.055,.30),(.035,.10,.32),silver)
        end('arrow_board',col,(1.05,0,0))
        refs=['sv_road_180.png','sv_road_225.png','sv_road_315.png','sv_ne_090.png'];approx=['鏡は凸面形状。PNGは自作の銀灰色の明度勾配で、ゲーム内の動的な反射ではない。','鏡径0.67m、中心高2.60mと柱/板寸法は写真比率による推定。矢印板は鏡と独立配置。']
    elif asset_id=='nanohanadai_curb_block':
        for i,(name,size) in enumerate([('curb_a',(.90,.40,.28)),('curb_b',(.65,.40,.25))]):
            col=start('0%d 車止め %s'%(i+1,name));chip('角欠け本体',size,pale,10+i)
            for j in range(5):
                rr=random.Random(i*20+j);x=rr.uniform(-size[0]*.43,size[0]*.43)
                C.box('根元に残る苔',(x,-.201,.025),(.07+rr.random()*.12,.004,.025+rr.random()*.035),moss)
            for j in range(4):
                x=-size[0]*.35+j*size[0]*.22;C.box('細い雨筋',(x,-.202,.12),(.012,.003,.13),stone)
            end(name,col,(i*1.2,0,0))
        col=start('03 北東の木箱ベンチ')
        for side in [-1,1]:
            cx=side*.31
            C.box('箱の暗い内側',(cx,0,.185),(.52,.46,.32),C.proc('箱内部の暗い木',(64,63,50),'wood',4))
            for j in range(5):
                C.box('座面板',(cx,-.22+j*.11,.375),(.60,.102,.05),wood,.006)
                C.box('前面の縦板',(cx-.24+j*.12,-.252,.19),(.11,.035,.33),wood)
                C.box('背面の縦板',(cx-.24+j*.12,.252,.19),(.11,.035,.33),wood)
            for xx in [-.267,.267]:
                for yy in [-.22,.22]:C.box('箱の隅柱',(cx+xx,yy,.19),(.065,.065,.38),wood)
            for yy in [-.17,-.055,.055,.17]:
                for xx in [-.282,.282]:C.box('側面板',(cx+xx,yy,.18),(.035,.105,.33),wood)
        end('wood_box_bench',col,(3,0,0));refs=['sv_mid_000.png','sv_mid_315.png','sv_ne_000.png','sv_ne_315.png'];approx=['車止めの二種類は写真に見える長さ/摩耗差による変種。規格寸法の異なる製品と確定したものではない。','木箱は写真で横並び2箱と判断、内部構造と背面は推定。']
    elif asset_id=='nanohanadai_fence_signs':
        for i,(name,w,h) in enumerate([('yellow_sign',.45,.55),('red_white_sign',.90,.45)]):
            col=start('0%d フェンスの板'%(i+1));C.box('板の裏',(0,0,h/2),(w,.012,h),silver)
            mat=C.image_material('自作注意板',ROOT/'assets/landmarks'/asset_id/(name+'_art.png'));C.front_panel('印刷面',(0,-.007,h/2),w*.98,h*.98,mat)
            for xx in [-w*.35,w*.35]:
                for zz in [h*.50]:C.box('固定バンド',(xx,.018,zz),(.017,.05,.014),silver)
            end(name,col,(i*1.25,0,0))
        refs=['sv_mid2_090.png','sv_mid2_135.png','sv_mid2_180.png'];approx=['文字は解像度不足で判読できず、文字を創作せず短い線と色面で近似。絵柄は色と斜線を簡略化した自作図。','吊下げ位置は配置担当側で調整するため各板の下辺中央が原点。フェンス自体は含めない。']
    elif asset_id=='nanohanadai_path_items':
        col=start('01 木の車止め杭');o=C.cylinder('風化木杭',(0,0,.52),.066,1.04,wood,12)
        for z in [.66,.80]:C.cylinder('小さい釘痕',(0,-.065,z),.007,.004,silver,6).rotation_euler[0]=math.pi/2
        end('wood_bollard',col)
        col=start('02 低いコンクリートの台');chip('コンクリート台',(1.50,.40,.37),stone,72)
        C.box('明るい座面',(0,0,.383),(1.50,.40,.046),pale,.012)
        for xx in [-.58,-.22,.31,.56]:C.box('雨染み',(xx,-.201,.20),(.025,.003,.22),stone)
        end('concrete_bench',col,(1.2,0,0))
        col=start('03 歩道縁のL字コンクリート')
        # 上から見たL、継ぎ足しがしやすい単体形状。全周の配置は含めない。
        ring=[(-.90,-.36),(.90,-.36),(.90,-.11),(-.65,-.11),(-.65,.36),(-.90,.36)]
        vs=[(x,y,z) for z in [0,.14] for x,y in ring];n=len(ring)
        faces=[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
        C.mesh('L字縁石',vs,faces,stone);end('l_curb',col,(3.2,0,0))
        refs=['sv_sw_090.png','sv_sw_135.png','sv_sw_180.png','sv_sw_225.png','sv_user_090.png','sv_user_180.png','sv_user_270.png'];approx=['杭とベンチの寸法は写真比率からの推定。','縁石は見える局所形状をL字部品とした。連続するジグザグ全体は配置側で組み、見えない区間は作っていない。']
    elif asset_id=='nanohanadai_road_planters':
        for i,name in enumerate(['checker_planter','white_planter']):
            col=start('0%d 道路側の円筒'%(i+1));mat=stone if i else C.image_material('自作赤白市松',ROOT/'assets/landmarks'/asset_id/'checker_art.png')
            ob=C.cylinder('底付き円筒',(0,0,.25),.24,.50,mat,24)
            # 側面を全面0..1へ展開し、市松の4段を保持する。
            for poly in ob.data.polygons:
                if abs(poly.normal.z)<.5:
                    us=[(math.atan2(ob.data.vertices[ob.data.loops[li].vertex_index].co.y,ob.data.vertices[ob.data.loops[li].vertex_index].co.x)/math.tau+1)%1 for li in poly.loop_indices]
                    if max(us)-min(us)>.5:us=[u+1 if u<.5 else u for u in us]
                    for li,u in zip(poly.loop_indices,us):
                        co=ob.data.vertices[ob.data.loops[li].vertex_index].co;ob.data.uv_layers['UVMap'].data[li].uv=(u,co.z/.5+.5)
            C.cylinder('上端の縁',(0,0,.51),.255,.04,pale,24)
            C.cylinder('土',(0,0,.535),.217,.008,C.proc('鉢内の土',(85,77,54),scale=20),24)
            end(name,col,(i*.85,0,0))
        refs=['sv_road_315.png'];approx=['赤白と白の円筒は写真に存在するが用途不明。鉢としての上面/内部は推定。植栽は配置側の植生と分離。']
    else:raise ValueError(asset_id)
    for name,obs in parts.items():
        if len(parts)>1 and name!=next(iter(parts)):
            for ob in obs:ob.hide_set(True)
    return {'source_visibility':'独立部品は重なりを避けるため制作元では最初の部品以外をviewport非表示。各コレクションの目を切替えて確認。NPZは全バリエーションを個別の原点で出力。','dimension_basis':'写真の既知物体との比率から推定。現地実測ではない。','refs':refs,'approximations':approx,'parts':parts,'preview_offsets':offsets,'reference_views':{'reference':(1,-2,.55)}}
