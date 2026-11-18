"""依頼07 公衆トイレ。自作表示は通常Python、形状/ベイクはBlenderで生成。"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1]
ASSET='pass_toilet'

def textures():
    from PIL import Image,ImageDraw,ImageFont
    import random
    root=ROOT/'assets/landmarks'/ASSET;root.mkdir(parents=True,exist_ok=True)
    font='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'
    def f(size):return ImageFont.truetype(font,size)
    def lines(d,box,color=(143,146,133),spacing=23,seed=0):
        r=random.Random(seed);x0,y0,x1,y1=box
        for y in range(y0,y1,spacing):
            x=x0
            while x<x1:
                w=r.randrange(20,85);d.line((x,y,min(x1,x+w),y),fill=color,width=3);x+=w+r.randrange(8,18)
    im=Image.new('RGB',(2048,256),(38,39,34));d=ImageDraw.Draw(im)
    text='ヤビツ峠公衆トイレ';font_obj=f(142);b=d.textbbox((0,0),text,font=font_obj)
    d.text(((2048-b[2])/2,25),text,font=font_obj,fill=(231,231,217));im.save(root/'toilet_title_art.png')
    im=Image.new('RGB',(1536,1152),(217,222,200));d=ImageDraw.Draw(im)
    d.rectangle((20,20,1516,138),fill=(208,198,152));lines(d,(45,40,1460,130),spacing=22,seed=1)
    d.rectangle((44,185,1490,1005),fill=(211,229,223))
    # 白い案内図は実画像を使わず色面と経路の抽象図だけ。
    for pts,col in [([(90,900),(320,720),(450,800),(610,450),(830,560),(1040,290),(1420,340)],(133,171,145)),([(150,460),(430,590),(640,290),(1020,440),(1330,700)],(161,179,145))]:
        d.line(pts,fill=col,width=12)
    for x,y in [(210,450),(520,290),(880,390),(1170,650)]:d.ellipse((x-18,y-18,x+18,y+18),fill=(165,170,139))
    for x,y,w,h,c in [(140,260,260,225,(211,189,93)),(575,250,250,350,(101,141,165)),(875,250,480,310,(172,196,185))]:
        d.rectangle((x,y,x+w,y+h),fill=c);lines(d,(x+15,y+20,x+w-15,y+h-10),spacing=25,seed=x)
    lines(d,(60,1050,1470,1120),spacing=22,seed=2);im.save(root/'toilet_map_art.png')
    im=Image.new('RGB',(1536,1152),(91,61,43));d=ImageDraw.Draw(im)
    rr=random.Random(703)
    for i,(x,y,w,h,c) in enumerate([(70,100,165,260,(223,225,205)),(340,90,310,175,(218,215,157)),(760,90,250,350,(177,204,201)),(1140,120,250,230,(217,219,205)),(150,620,330,240,(209,224,210)),(590,450,240,330,(224,215,186)),(950,680,430,190,(205,210,181))]):
        d.rectangle((x,y,x+w,y+h),fill=c);lines(d,(x+15,y+20,x+w-15,y+h-10),spacing=24,seed=i)
    for x in [805,900]:d.polygon([(x,510),(x+45,560),(x,610),(x-45,560)],fill=(200,204,169))
    im.save(root/'toilet_notices_art.png')
    im=Image.new('RGB',(1536,1152),(36,37,32));d=ImageDraw.Draw(im)
    for title,y in [('階段や木道を歩きましょう',40),('野生動物と共存するために',385)]:
        d.text((45,y),title,font=f(72),fill=(225,225,211));lines(d,(55,y+115,1480,y+285),color=(166,170,154),spacing=27,seed=y)
    # 未判読の残り見出しも文字にせず線の太さだけ近似。
    for y in [755,970]:
        lines(d,(45,y,1420,y+45),color=(208,210,193),spacing=18,seed=y);lines(d,(55,y+65,1460,min(1140,y+180)),color=(161,166,149),spacing=24,seed=y+1)
    im.save(root/'toilet_rules_art.png')
    for sex,c in [('men',(43,83,112)),('women',(151,54,62))]:
        im=Image.new('RGB',(256,384),(225,225,211));d=ImageDraw.Draw(im)
        d.ellipse((98,35,158,95),fill=c)
        if sex=='women':d.polygon([(105,115),(151,115),(185,254),(71,254)],fill=c)
        else:d.rectangle((97,112,159,249),fill=c)
        d.line((103,245,103,345),fill=c,width=21);d.line((153,245,153,345),fill=c,width=21)
        d.line((94,122,65,236),fill=c,width=17);d.line((162,122,191,236),fill=c,width=17)
        im.save(root/('toilet_'+sex+'_art.png'))
    im=Image.new('RGB',(1024,768),(216,215,196));d=ImageDraw.Draw(im)
    for i in range(4):
        x=40+i*245;d.rectangle((x,50,x+205,700),fill=[(213,222,199),(178,192,183),(225,219,184),(203,212,199)][i]);lines(d,(x+15,90,x+190,660),spacing=25,seed=i)
    im.save(root/'west_posters_art.png')

def build(asset=ASSET):
    assert asset==ASSET
    import bpy
    import pass_common as C
    parts={};part_info={}
    plaster=C.proc('少し汚れた白い壁',(210,211,194),scale=17)
    cream=C.proc('西長壁のクリーム波板',(199,199,176),scale=15)
    dark=C.proc('暗い開口の奥',(31,34,30),scale=8)
    wood=C.proc('暗緑黒茶の木架構',(51,57,42),'wood',5)
    boardwood=C.proc('掲示板の茶木',(99,68,47),'wood',5)
    benchwood=C.proc('腰掛けの暗褐木',(78,58,46),'wood',6)
    roof=C.proc('灰紫の金属屋根',(108,105,113),scale=11)
    ridge=C.proc('屋根の黒茶縁',(47,44,39),scale=13)
    base=C.proc('低い灰色基礎',(133,136,120),scale=13)
    metal=C.proc('金属樋',(58,61,53),scale=17)
    glass=C.proc('明るい換気採光面',(167,188,184),scale=6)
    lamp=C.proc('乳白の丸い灯具',(230,229,208),scale=8)
    def start(name,meaning):
        col=C.collection(name);part_info[name]={'meaning':meaning,'front':'-Y'};return col
    def end(name,col):parts[name]=list(col.objects)
    def imat(name):return C.image_material(name,ROOT/'assets/landmarks'/ASSET/(name+'_art.png'))
    def slab(name,corners,thick,mat):
        return C.mesh(name,corners+[(x,y,z-thick) for x,y,z in corners],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],mat)
    def wallsegment(name,a,b,z,h,thick,mat):
        return C.beam(name,(a[0],a[1],z+h/2),(b[0],b[1],z+h/2),h,mat,thick)
    def segment_box(name,a,b,z,width,height,mat):
        # 水平線を長手Xとして厚さY、高さZの角材を生成。
        ax,ay=a;bx,by=b;L=math.hypot(bx-ax,by-ay)
        o=C.box(name,((ax+bx)/2,(ay+by)/2,z),(L,width,height),mat);o.rotation_euler[2]=math.atan2(by-ay,bx-ax);return o
    wall_w=7.6;wall_d=10.7;front=-4.85;back=5.85;wall_h=2.72;eave=2.93;peak=4.85
    col=start('walls','低い基礎、白い前柱、前掲示壁、クリーム色の縦リブ西長壁。左右入口は奥へ開く。')
    C.box('低い本体基礎',(0,.5,.10),(wall_w,wall_d,.20),base)
    # 側壁は前面から1m奥へ後退。入口左右を暗い開口として残す。
    for x in [-3.8,3.8]:
        C.box('長い側壁',(x,.92,1.46),(.13,9.86,2.52),cream if x<0 else plaster)
        for y in [-4.02,5.85]:C.box('側壁端の柱',(x,y,1.46),(.18,.18,2.52),plaster)
    C.box('背面の低い壁',(0,5.79,1.46),(7.6,.13,2.52),plaster)
    # 正面は浅い折れ。両端柱外側の進入口を面で埋めない。
    line=[(-3.46,-4.98),(-2,-4.73),(-.8,-4.63),(.8,-4.63),(2,-4.73),(3.46,-4.98)]
    for a,b in zip(line,line[1:]):segment_box('折れた白い掲示背壁',a,b,1.40,.15,2.40,plaster)
    for x in [-3.65,3.65]:
        C.box('前面の白い柱',(x,-5.28,1.43),(.30,.38,2.86),plaster)
        C.box('入口の奥の暗い面',(x*.945,-3.35,1.25),(.52,.06,2.2),dark)
        C.box('入口上の梁',(x*.975,-4.42,2.55),(.50,1.70,.24),wood)
    # 西側の縦波板は実際の凹凸を簡略立体化。
    for i in range(75):
        y=-3.94+i*.13
        C.box('西壁の縦リブ',(-3.88,y,1.46),(.035,.018,2.47),cream)
    # 西面の貼紙は自作小字線・色面。C.front_panelを回して−X面へ。
    o=C.front_panel('西壁の貼紙',(0,0,0),2.65,.85,imat('west_posters'))
    o.rotation_euler[2]=-math.pi/2;o.location=(-3.902,.9,1.45)
    end('walls',col)

    col=start('metal_roof','9.2m×12.4m外包の左右金属斜面、棟包み、黒茶鼻隠し、樋。')
    for sign in [-1,1]:
        corners=[(0,-6.2,peak),(sign*4.6,-6.2,eave),(sign*4.6,6.2,eave),(0,6.2,peak)]
        slab('左右の金属屋根',corners,.085,roof)
        for y in [-6.2,6.2]:C.beam('妻の黒茶鼻隠し',(0,y,peak),(sign*4.6,y,eave),.14,ridge,.18)
        C.pipe('軒先の雨樋',(sign*4.54,-6.15,eave-.06),(sign*4.54,6.15,eave-.06),.055,metal,8)
        C.pipe('後ろの縦樋',(sign*4.48,5.75,eave-.08),(sign*4.48,5.75,.18),.04,metal,8)
        for j in range(15):
            y=-6.1+j*.87
            C.beam('屋根の細い継ぎ',(0,y,peak+.007),(sign*4.6,y,eave+.007),.015,ridge,.012)
    C.box('棟包み',(0,0,peak+.025),(.22,12.4,.09),ridge)
    end('metal_roof',col)

    col=start('wood_trusses','前後の開いたfan truss。中央下端から放射する斜材、中央束、三角外枠と主要垂木。')
    for y in [-5.88,5.88]:
        bottom=(0,y,2.92)
        C.beam('妻の下弦梁',(-4.25,y,2.92),(4.25,y,2.92),.15,wood,.19)
        C.beam('妻の中央縦束',bottom,(0,y,4.72),.13,wood,.16)
        for sign in [-1,1]:
            C.beam('三角外周の登り梁',(sign*4.30,y,2.98),(0,y,4.76),.14,wood,.19)
            # 中央下の接合から屋根斜面へ放射。間は開放。
            for x in [.75,1.55,2.40,3.25]:
                z=4.76-x*(1.78/4.3)
                C.beam('放射状の斜材',bottom,(sign*x,y,z),.085,wood,.12)
    for y in [-4.85,-3.1,-1.35,.4,2.15,3.9,5.55]:
        for sign in [-1,1]:C.beam('主要垂木',(0,y,4.69),(sign*4.4,y,2.86),.08,wood,.12)
    for x in [-2.5,0,2.5]:
        z=4.7-abs(x)*1.83/4.4
        C.beam('奥へ通る母屋',(x,-5.93,z),(x,5.95,z),.10,wood,.13)
    end('wood_trusses',col)

    col=start('front_boards','妻面全面の案内図/中央茶掲示/右濃茶注意板と、ヤビツ峠公衆トイレの上帯。')
    C.box('題字の上帯',(0,-5.04,2.64),(6.94,.16,.22),wood)
    C.front_panel('判読できた題字',(0,-5.125,2.64),4.9,.19,imat('toilet_title'))
    for name,x,y,w,h,z,angle,mat in [
        ('toilet_map',-2.37,-4.91,2.12,1.70,1.62,-.17,plaster),
        ('toilet_notices',-.12,-4.735,2.48,1.67,1.62,0,boardwood),
        ('toilet_rules',2.30,-4.91,2.01,1.73,1.62,.17,boardwood)]:
        o=C.box('掲示板の額',(x,y+.02,z),(w+.09,.085,h+.08),mat);o.rotation_euler[2]=angle
        o=C.front_panel('自作の掲示表示',(0,0,0),w,h,imat(name));o.rotation_euler[2]=angle;o.location=(x,y-.026,z)
    end('front_boards',col)

    col=start('built_in_bench','白い前壁沿いに5区間で緩く折れる脚付き腰掛け。長手板4枚、背なし。')
    # 壁面と同じ折れを保って独立した5区間にする。
    benchline=[(x,y-.28) for x,y in line]
    for a,b in zip(benchline,benchline[1:]):
        angle=math.atan2(b[1]-a[1],b[0]-a[0]);nx,ny=-math.sin(angle),math.cos(angle)
        for k in range(4):
            off=(k-1.5)*.095
            segment_box('腰掛けの長手板',(a[0]+nx*off,a[1]+ny*off),(b[0]+nx*off,b[1]+ny*off),.43,.088,.045,benchwood)
        for t in [.16,.84]:
            x=a[0]+(b[0]-a[0])*t;y=a[1]+(b[1]-a[1])*t
            C.box('腰掛けの短脚',(x,y,.215),(.105,.30,.43),base)
    end('built_in_bench',col)

    col=start('entrance_marks','左右前柱の男女記号、丸い乳白灯、妻の小スポット灯。')
    for x,sex in [(-3.65,'men'),(3.65,'women')]:
        C.box('男女記号の白札',(x,-5.482,2.20),(.20,.018,.32),plaster)
        C.front_panel('自作男女記号',(x,-5.494,2.20),.18,.29,imat('toilet_'+sex))
        o=C.cylinder('丸い乳白灯',(x,-5.505,2.64),.175,.09,lamp,16);o.rotation_euler[0]=math.pi/2
    for x in [-2.6,2.6]:
        z=4.76-abs(x)*1.78/4.3
        C.pipe('スポット灯の台',(x,-5.98,z-.10),(x,-6.14,z-.10),.07,metal,8)
        o=C.cylinder('小さな灯面',(x,-6.16,z-.10),.065,.02,lamp,12);o.rotation_euler[0]=math.pi/2
    end('entrance_marks',col)

    col=start('west_clerestory','西側屋根上の連続三角換気/採光帯。窓材・背面は推定。')
    # 西斜面上の浅い高窓帯。白い連続三角と黒い対角枠を実形状で分離。
    x=-1.05;zbase=4.35;ztop=4.78;ys=[-4.7+i*1.25 for i in range(9)]
    for a,b in zip(ys,ys[1:]):
        mid=(a+b)/2
        C.mesh('明るい三角換気採光面',[(x,a,zbase),(x,mid,ztop),(x,b,zbase)],[(0,1,2)],glass)
        C.beam('三角左の細枠',(x,a,zbase),(x,mid,ztop),.035,metal,.035)
        C.beam('三角右の細枠',(x,mid,ztop),(x,b,zbase),.035,metal,.035)
        C.mesh('三角間の暗い開口',[(x,a,zbase),(x,a,ztop),(x,mid,ztop)],[(0,1,2)],ridge)
    C.beam('高窓上端',(x,ys[0],ztop),(x,ys[-1],ztop),.055,metal,.06)
    C.beam('高窓下端',(x,ys[0],zbase),(x,ys[-1],zbase),.05,metal,.06)
    slab('高窓の細い上蓋',[(x-.1,ys[0],ztop+.03),(.05,ys[0],4.81),(.05,ys[-1],4.81),(x-.1,ys[-1],ztop+.03)],.035,roof)
    end('west_clerestory',col)

    col=start('rear_shed','正面右脇の小さい茶色物置状設備。組立位置を保持、隠れた背面は推定。')
    shed=C.proc('物置の茶板',(81,58,39),'wood',6)
    C.box('小物置本体',(4.75,-3.65,1.08),(1.25,1.70,2.16),shed)
    C.box('小物置の扉',(4.75,-4.515,1.05),(1.06,.035,1.92),boardwood)
    C.box('扉の細い縁',(4.75,-4.543,1.05),(.035,.025,1.92),wood)
    for z in [.25,.43,.61,.79,.97,1.15,1.33,1.51,1.69]:C.box('茶小屋の横板目地',(4.75,-4.54,z),(1.08,.012,.012),wood)
    C.box('換気口',(4.75,-4.56,1.72),(.29,.02,.23),dark)
    for z in [1.65,1.71,1.77]:C.box('換気ルーバー',(4.75,-4.578,z),(.26,.015,.014),metal)
    slab('茶小屋の低い屋根',[(4.02,-4.60,2.30),(5.48,-4.60,2.30),(5.48,-2.69,2.22),(4.02,-2.69,2.22)],.08,ridge)
    end('rear_shed',col)

    return {'parts':parts,'part_info':part_info,'generator':'blender_scripts/build_pass_toilet.py','resolution':2048,'placement_exclusion_group':'pass_toilet_building','alternative_asset':'pass_lodge','front_azimuth_deg':166,'roof_footprint_m':[9.2,12.4],
      'refs':['codex07/'+x for x in ['gsat20.png','sv_toilet_000.png','sv_toilet_315.png','sv_toilet_270.png','sv_u2_045.png','sv_u2_090.png','sv_u2_135.png','sv_u2_180.png','sv_shop_090.png','sv_mid_045.png']],
      'dimension_basis':'work07/building_measurements.json: GSAT20屋根外包NW(738,258),NE(811,242),SE(836,339),SW(763,355)、10m=81px、幅9.23m/棟方向12.37mから9.2×12.4mを採用（±約1m）。壁幅7.6m・長さ10.7m、ポーチ1.35m、軒高2.93m/棟高4.85mは写真比率推定で実測ではない。',
      'approximations':['北の白い長壁は同じトイレ西長壁の可能性が高いため本体に含めた。pass_lodgeは同じ本体の代替データ。独立2棟として二重配置しない。','背面、内部、右小物置の奥と屋根裏の接合は遮蔽のため推定。男女記号は写真の色と一般的な形から自作。','正面三角妻は板で塞がず、前後の放射状トラスと主要垂木を立体化。斜材本数/断面は写真から推定。','西側の連続三角換気/採光帯は外形と明暗を形状化。窓材と奥行きは未確認。','題字ヤビツ峠公衆トイレと注意板の判読できる大見出し2行のみ文字化。本文、案内図、貼紙は自作の短線と抽象図。2014年Mt.Tounodake壁文字は2025年壁へ入れていない。','腰掛けは写真の緩い折れを5区間で近似、各座面4枚板。階段/路面/縁石/自然石/人物は含めない。'],
      'reference_views':{'reference_toilet000':[-.15,-2,.25],'reference_toilet315':[-1,-2,.35]}}

if __name__=='__main__':
    if 'bpy' not in sys.modules:textures()
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parent));import pass_common as C
        C.init_scene();C.save_asset(ASSET,build())
