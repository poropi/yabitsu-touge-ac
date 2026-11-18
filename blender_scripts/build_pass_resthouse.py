"""依頼07: 市公式設計図と竣工写真によるレストハウス。+Z/-Y、方位は未焼込。"""
from pathlib import Path
import sys, math
ROOT=Path(__file__).resolve().parents[1]
ASSET='pass_resthouse'


def textures():
    """写真を使わず、窓面の反射/室内色だけを自作する。"""
    from PIL import Image, ImageDraw
    out=ROOT/'assets/landmarks'/ASSET;out.mkdir(parents=True,exist_ok=True)
    im=Image.new('RGB',(512,1024),(94,107,99));d=ImageDraw.Draw(im)
    for y in range(1024):
        t=y/1023
        c=tuple(round(a*(1-t)+b*t) for a,b in zip((100,132,139),(111,108,76)))
        d.line((0,y,511,y),fill=c)
    # 森/内装を示す控えめな抽象反射。写真の画素/看板文字は転用しない。
    for x,w,c in [(15,28,(67,82,73)),(101,20,(104,119,102)),(196,32,(72,86,77)),(330,24,(117,125,106)),(433,38,(77,95,85))]:
        d.polygon([(x,0),(x+w,0),(x+w-18,1024),(x-18,1024)],fill=c)
    d.polygon([(0,550),(511,400),(511,445),(0,595)],fill=(132,132,102))
    im.save(out/'glass_reflection_art.png')


def build():
    import bpy
    import pass_common as C
    parts={};part_info={}
    wall=C.proc('茶色横板調サイディング',(155,101,65),'wood',7)
    grain=C.proc('板目地の濃い茶',(113,75,49),'wood',8)
    roof=C.proc('濃灰ガルバリウム屋根',(53,62,64),scale=13)
    roofedge=C.proc('屋根の濃い鼻隠し',(43,50,51),scale=15)
    soffit=C.proc('木目軒裏',(91,76,53),'wood',7)
    frame=C.proc('窓の黒アルミ枠',(35,40,37),scale=13)
    glass=C.proc('三角高窓の淡い室内反射',(124,130,104),scale=3)
    glassblue=C.proc('三角高窓の淡青反射',(130,158,157),scale=3)
    deckwood=C.proc('淡い黄褐木のデッキ',(176,139,84),'wood',5)
    railwood=C.proc('斜め格子の明るい木',(189,151,95),'wood',5)
    concrete=C.proc('灰色基礎と束',(137,143,139),scale=19)
    canopy=C.proc('前面横長庇の茶金属',(137,111,96),scale=13)
    gutter=C.proc('灰褐色雨樋',(159,151,132),scale=16)
    lampglass=C.proc('壁灯の白いガラス',(219,217,178),scale=3)
    dark=C.proc('基礎奥の暗い壁',(66,65,52),scale=9)
    reflection=C.image_material('自作の窓反射',ROOT/'assets/landmarks'/ASSET/'glass_reflection_art.png')

    def start(name,meaning):
        col=C.collection(name);part_info[name]={'meaning':meaning,'front':'-Y'};return col
    def end(name,col):parts[name]=list(col.objects)
    def slab(name,corners,t,mat):
        return C.mesh(name,corners+[(x,y,z-t) for x,y,z in corners],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],mat)
    def sidebox(name,x,y,z,w,h,t,mat,side=1):
        return C.box(name,(x,y,z),(t,w,h),mat)
    def side_panel(name,x,y,z,w,h,mat):
        o=C.front_panel(name,(0,0,0),w,h,mat);o.rotation_euler[2]=math.pi/2;o.location=(x,y,z);return o
    def side_window(name,x,y,z,w,h,door=False):
        sidebox(name+'黒額',x,y,z,w+.13,h+.13,.11,frame)
        side_panel(name+'自作反射',x+.062,y,z,w,h,reflection)
        sidebox(name+'中央桟',x+.075,y,z,.055,h,.065,frame)
        sidebox(name+'上雨戸箱',x+.11,y,z+h/2+.21,w+.25,.28,.20,frame)
        if door:
            C.box(name+'把手',(x+.105,y-.10,z),(.05,.035,.35),frame)
    def frontrect(name,x,z,w,h,mat,y=-3.475):
        C.front_panel(name,(x,y,z),w,h,mat)

    col=start('walls','主屋9.0×6.3m、前面の張出し・茶横板壁と裏妻。FL=1.0m。開口部を壁で塞がない。')
    # 前面の大開口は両端の900mm袖壁だけ残す。
    for x in [-4.05,4.05]:C.box('前面袖壁',(x,-3.15,2.455),(.9,.18,2.91),wall)
    C.box('前面ガラス下の腰台',(0,-3.43,1.07),(7.2,.20,.14),wall)
    # 東側は引戸と窓の実開口を空ける。 y<0に引戸、y>0に窓。
    for ya,yb in [(-3.15,-2.45),(-.55,.2),(2.25,3.15)]:
        C.box('東側の縦壁',(4.5,(ya+yb)/2,2.455),(.18,yb-ya,2.91),wall)
    C.box('東側開口上の壁',(4.5,0,3.66),(.18,6.3,.50),wall)
    C.box('東窓下の腰壁',(4.5,1.225,1.65),(.18,2.05,1.3),wall)
    C.box('西長壁',(-4.5,0,2.455),(.18,6.3,2.91),wall)
    C.box('北の裏壁',(0,3.15,2.455),(9,.18,2.91),wall)
    C.mesh('裏の三角妻',[(-4.5,3.15,3.91),(4.5,3.15,3.91),(0,3.15,6.96)],[(0,2,1)],wall)
    # 横板目地は平面材質のみにせず、見える薄い線として残す。
    for i in range(15):
        z=1.16+i*.195
        C.box('西壁の横板目地',(-4.597,0,z),(.012,6.28,.010),grain)
        C.box('裏壁の横板目地',(0,3.247,z),(8.98,.012,.010),grain)
        for x in [-4.05,4.05]:C.box('前袖壁の横目地',(x,-3.247,z),(.88,.012,.010),grain)
        if z<3.90:
            for ya,yb in [(-3.15,-2.45),(-.55,.2),(2.25,3.15)]:C.box('東壁端の横目地',(4.597,(ya+yb)/2,z),(.012,yb-ya,.010),grain)
        if z<2.29:C.box('東窓下の横目地',(4.597,1.225,z),(.012,2.04,.010),grain)
    end('walls',col)

    col=start('roof','公式軸9×6.3m、出寸各900mmの大切妻。幅10.8×奥行8.1、棟7.1m、壁位置軒高4.05m。')
    eave=7.1-(3.05/4.5)*5.4
    for s in [-1,1]:
        corners=[(0,-4.05,7.1),(s*5.4,-4.05,eave),(s*5.4,4.05,eave),(0,4.05,7.1)]
        slab('大きい屋根の左右斜面',corners,.11,roof)
        # 軒裏は屋根と別色。僅かに下げて干渉を避ける。
        C.mesh('木目の軒裏',[(x,y,z-.115) for x,y,z in corners],[(3,2,1,0) if s>0 else (0,1,2,3)],soffit)
        for y in [-3.97,3.97]:C.beam('切妻鼻隠し',(0,y,7.020),(s*5.32,y,eave-.035),.14,roofedge,.13)
        C.beam('軒先の黒縁',(s*5.32,-4.02,eave-.035),(s*5.32,4.02,eave-.035),.14,roofedge,.13)
        for j in range(12):
            y=-3.96+j*.72
            C.beam('屋根の立平継ぎ',(0,y,7.088),(s*5.35,y,7.1-3.05/4.5*5.35+.006),.012,roofedge,.012)
    # 最高点は7.1mのまま。棟包みを本体内に納める。
    C.box('細い棟包み',(0,0,7.055),(.15,8.1,.09),roofedge)
    end('roof',col)

    col=start('front_glazing','中央引戸と左右の大きいガラス・欄間・三角高窓、黒い方立。窓は自作の不透明反射近似。')
    # ガラスは前面7.2m、FL直上から庇下まで。中央2枚が引戸。
    for a,b in [(-3.6,-1.8),(-1.8,-.9),(-.9,0),(0,.9),(.9,1.8),(1.8,3.6)]:
        frontrect('下部ガラス',(a+b)/2,2.20,b-a-.06,2.10,reflection)
    for x in [-3.64,-1.8,-.9,0,.9,1.8,3.64]:
        C.box('前面の縦アルミ方立',(x,-3.49,2.52),(.075,.12,2.86),frame)
    for z in [1.10,3.29,3.91]:C.box('前面の横アルミ枠',(0,-3.49,z),(7.35,.12,.075),frame)
    for a,b in [(-3.6,-1.8),(-1.8,0),(0,1.8),(1.8,3.6)]:frontrect('横長欄間',(a+b)/2,3.60,b-a-.075,.52,glassblue)
    for x in [-.1,.1]:C.box('中央引戸の把手',(x,-3.58,2.10),(.035,.055,.43),frame)
    # 頂部のガラスは屋根裏から150mm下、左右の肩を持つ五角形。
    verts=[(-3.6,-3.48,4.06),(3.6,-3.48,4.06),(3.6,-3.48,4.46),(0,-3.48,6.90),(-3.6,-3.48,4.46)]
    C.mesh('大切妻の高いガラス',verts,[(0,1,2,3,4)],glass)
    for a,b in zip(verts,verts[1:]+verts[:1]):C.beam('三角窓外枠',a,b,.075,frame,.085)
    for x in [-1.8,0,1.8]:
        top=6.90-abs(x)*2.44/3.6
        C.beam('三角窓の方立',(x,-3.53,4.08),(x,-3.53,top),.065,frame,.09)
    # 三角の最下部に淡青の反射を細く入れてガラス性を示す。
    C.mesh('高窓の淡い空反射',[(-1.74,-3.535,4.10),(-.08,-3.535,4.10),(-.08,-3.535,4.48)],[(0,1,2)],glassblue)
    # 西側の窓と背面の小窓は非参照面の推定。
    for y in [-1.6,1.3]:
        C.box('西小窓の黒額',(-4.61,y,2.55),(.08,1.25,1.15),frame)
        o=side_panel('西小窓の反射',-4.66,y,2.55,1.12,1.02,reflection);o.rotation_euler[2]=-math.pi/2
    C.box('北小窓の黒枠',(-1.0,3.26,2.55),(1.1,.09,1.05),frame)
    C.box('北小窓の反射',(-1.0,3.315,2.55),(.96,.015,.91),glass)
    end('front_glazing',col)

    col=start('deck','平面図の南面デッキと東側L字張出し。板厚と梁を分離。床上端1.0m。')
    # 主屋床と、L字の3矩形。板の端に壁と重複する面を作らない。
    C.box('主屋の床',(0,0,.92),(9,6.3,.16),deckwood)
    deckrects=[(-4.65,6.3,-4.95,-3.15),(4.5,6.3,-3.15,.45),(4.5,8.1,.45,3.15)]
    for x0,x1,y0,y1 in deckrects:
        C.box('デッキ床板の下地',((x0+x1)/2,(y0+y1)/2,.90),(x1-x0,y1-y0,.16),deckwood)
        # 写真の長手床板を170mm幅で分割。薄い板で大きい隙間を出さない。
        count=math.ceil((y1-y0)/.17);dy=(y1-y0)/count
        for i in range(count):C.box('長手のデッキ板',((x0+x1)/2,y0+(i+.5)*dy,.992),(x1-x0,dy-.007,.016),deckwood)
    end('deck',col)

    col=start('foundation','床下の灰色梁/束と奥壁。外周を全面巨大箱にしない開口付き基礎。')
    C.box('主屋床下の奥壁',(0,2.9,.46),(8.8,.18,.92),dark)
    for x in [-4.5,0,4.5]:
        for y in [-3.0,0,3.0]:C.box('主屋を支える基礎束',(x,y,.42),(.28,.32,.84),concrete)
    edges=[((-4.65,-4.95),(6.3,-4.95)),((6.3,-4.95),(6.3,.45)),((6.3,.45),(8.1,.45)),((8.1,.45),(8.1,3.15)),((8.1,3.15),(4.5,3.15))]
    foundation_posts=set()
    for a,b in edges:
        C.beam('床下外周の灰色梁',(a[0],a[1],.80),(b[0],b[1],.80),.24,concrete,.27)
        length=math.dist(a,b);num=math.ceil(length/2.7)
        for i in range(num+1):
            t=i/num;x=a[0]+t*(b[0]-a[0]);y=a[1]+t*(b[1]-a[1])
            key=(round(x,5),round(y,5))
            if key not in foundation_posts:
                C.box('外周梁を支える灰色束',(x,y,.36),(.30,.30,.72),concrete);foundation_posts.add(key)
    end('foundation',col)

    col=start('rails','竣工写真の斜め格子手すり。上桟/下桟/柱と両方向の角材。図面の縦格子から変更。')
    rail_posts=set()
    def lattice_run(a,b,z0=1.02,z1=2.0):
        length=math.dist(a,b);n=math.ceil(length/1.8)
        ux,uy=(b[0]-a[0])/length,(b[1]-a[1])/length
        def point(t,z,off=0):return (a[0]+ux*t-uy*off,a[1]+uy*t+ux*off,z)
        for i in range(n+1):
            t=length*i/n;pos=point(t,z0-.03);key=tuple(round(v,5) for v in pos)
            if key not in rail_posts:
                C.beam('格子手すりの柱',pos,point(t,z1+.055),.085,railwood,.085);rail_posts.add(key)
        C.beam('幅広の笠木',point(0,z1+.045),point(length,z1+.045),.09,railwood,.11)
        C.beam('手すりの下桟',point(0,z0+.06),point(length,z0+.06),.07,railwood,.08)
        h=z1-z0-.14;bottom=z0+.10
        for k in range(n):
            lo=length*k/n+.06;hi=length*(k+1)/n-.06;w=hi-lo
            # 交差間隔約22cm。写真の密度に近い2方向格子を各枠内で切る。
            for slope in [-1,1]:
                intercept=-h
                while intercept<=w+.01:
                    points=[]
                    for t in [0,w]:
                        z=slope*t+intercept
                        if 0<=z<=h:points.append((t,z))
                    for z in [0,h]:
                        t=(z-intercept)/slope
                        if 0<=t<=w:points.append((t,z))
                    uniq=[]
                    for p in points:
                        if p not in uniq:uniq.append(p)
                    if len(uniq)>=2 and math.dist(uniq[0],uniq[1])>.045:
                        p,q=uniq[:2];C.beam('斜め格子の角材',point(lo+p[0],bottom+p[1],.012*slope),point(lo+q[0],bottom+q[1],.012*slope),.033,railwood,.026)
                    intercept+=.22
    for a,b in [((-4.65,-4.95),(-.95,-4.95)),((.95,-4.95),(6.3,-4.95)),((6.3,-4.95),(6.3,.45)),((6.3,.45),(8.1,.45)),((8.1,.45),(8.1,1.30)),((8.1,3.15),(4.5,3.15)),((-4.65,-4.95),(-4.65,-3.15))]:lattice_run(a,b)
    end('rails',col)

    col=start('stairs','南中央と東北側の2階段。踏面、側桁、斜め手すりを分離。')
    def staircase(center,direction,width=1.8,length=1.8):
        dx,dy=direction;nx,ny=-dy,dx;cx,cy=center;steps=6
        # centerは踊場の上端。方向ベクトルの先へ降りる。
        for i in range(steps):
            t=(i+.5)*length/steps;h=1-i/steps
            o=C.box('灰色の階段踏面',(cx+dx*t,cy+dy*t,h-.055),(width,length/steps+.008,.11),concrete)
            o.rotation_euler[2]=math.atan2(-dx,dy)
        for side in [-1,1]:
            a=(cx+nx*side*(width/2+.04),cy+ny*side*(width/2+.04),.90)
            b=(cx+dx*length+nx*side*(width/2+.04),cy+dy*length+ny*side*(width/2+.04),.15)
            C.beam('階段の灰色側桁',a,b,.18,concrete,.23)
            for t in [0,.5,1]:
                x=a[0]+dx*length*t;y=a[1]+dy*length*t;z=1-.88*t
                C.beam('階段の木手すり柱',(x,y,z),(x,y,z+.92),.065,railwood,.065)
            C.beam('階段の斜め笠木',(a[0],a[1],1.97),(b[0],b[1],1.08),.075,railwood,.075)
            C.beam('階段手すりの中桟',(a[0],a[1],1.52),(b[0],b[1],.63),.045,railwood,.045)
    staircase((0,-4.95),(0,-1))
    staircase((8.1,2.25),(1,0))
    end('stairs',col)

    col=start('fittings','東側の引戸/窓/カウンター/雨戸箱、前面の横長庇、雨樋/縦樋と壁灯。')
    side_window('東側引戸',4.61,-1.50,2.12,1.74,2.15,True)
    side_window('東側窓',4.61,1.225,2.82,1.89,1.16)
    C.box('東窓下の木カウンター',(4.85,1.225,2.19),(.65,2.20,.075),deckwood)
    for y in [.45,2.0]:C.beam('カウンターの小持送',(4.66,y,1.91),(5.08,y,2.15),.045,deckwood,.06)
    # 前面横長庇は奥行を持つ箱。ガラスの方立より前へ出す。
    C.box('前面の横長庇',(0,-3.74,4.02),(7.50,.72,.36),canopy)
    for s in [-1,1]:
        x=s*5.38;z=eave-.08
        C.pipe('横の雨樋',(x,-4.03,z),(x,4.03,z),.058,gutter,8)
        for y in [-2.92,2.92]:
            C.pipe('樋の斜め接続',(x,y,z),(s*4.78,y,3.12),.045,gutter,8)
            C.pipe('縦樋',(s*4.78,y,3.12),(s*4.78,y,.94),.045,gutter,8)
            C.pipe('縦樋の足元曲がり',(s*4.78,y,.94),(s*4.94,y,.75),.045,gutter,8)
    for y in [-2.7,-.2,2.65]:
        C.box('東壁灯の台',(4.65,y,3.10),(.075,.14,.25),frame)
        C.box('東壁灯の白ガラス',(4.78,y,3.24),(.12,.12,.27),lampglass)
        C.box('東壁灯の笠',(4.78,y,3.405),(.20,.20,.055),frame)
        for sy in [-1,1]:C.box('壁灯の縦枠',(4.86,y+sy*.065,3.24),(.025,.018,.29),frame)
    for x in [-4.05,4.05]:
        C.box('前壁灯の台',(x,-3.28,3.1),(.14,.075,.25),frame)
        C.box('前壁灯のガラス',(x,-3.40,3.24),(.13,.12,.26),lampglass)
        C.box('前壁灯の笠',(x,-3.40,3.40),(.20,.20,.05),frame)
    end('fittings',col)

    return {'parts':parts,'part_info':part_info,'resolution':2048,'generator':'blender_scripts/build_pass_resthouse.py',
      'refs':['work07/reference_resthouse_elevation.png','work07/reference_resthouse_plan.png','work07/reference_resthouse_210329095329_3.jpg','work07/reference_resthouse_210329091500_2.jpg','codex07/gsat21.png'],
      'front_azimuth_deg':197,'roof_footprint_m':[10.8,8.1],
      'dimension_basis':'秦野市公式 https://www.city.hadano.kanagawa.jp/material/files/group/2/020507_bucyokaigi_jiansyo_all.pdf のPDF13頁資料1-4/A-20立面図、14頁資料1-5/A-18平面図。2020年3月設計寸法: 軸9.000×6.300m、左右軒各0.900m、屋根10.800×8.100m。GL→FL1.000m、FL→軒高3.050m、軒高→棟3.050m、GL→棟7.100m。屋根軒先は壁位置軒高から勾配延長で下がる。正面197度は北矢印から読取、ローカルには焼込まない。',
      'approximations':['公式竣工写真 https://www.city.hadano.kanagawa.jp/material/images/group/51/210329095329_3.jpg と 210329091500_2.jpg を形状参照。手すりは施工前図の縦格子を採らずactualphotoの斜め格子を角材化。格子幅/密度は写真比率の近似。','近接SVはcodex07のshop/mid/toilet/lanes/south全40方向とlotn4方向で高台の植生に隠れている。https://tanzawamon.jp/yabitsutouge-resthouse/ と https://www.city.hadano.kanagawa.jp/kanko/spot-shisetsu/2/1343.html を探し、歩行近接panoidは取得できず公式写真を優先した。','デッキの東L字張出しと二方向階段は平面図を参照、床下外周梁/束と開口は竣工写真を優先。地面の起伏/背面基礎の埋まりは未測定なので水平GLからの束で近似。','三角高窓/正面引戸/東側引戸と窓・カウンターは竣工写真で補完。ガラスはゲーム用Diffuse COLORベイクのため自作色面/抽象反射による不透明近似で透過や実景反射はない。','裏側と西面の小窓、内部、屋根裏の接合、壁灯形状、樋の取回しは推定。人物/自転車ラック/周辺地面は含めない。','外壁は木目調窯業系サイディングを茶色横板目地と手続き材質で再現。写真画素/完成予想CG画素は素材に転用していない。'],
      'reference_views':{'reference_official':[1,-2,.55],'reference_side':[2,-1,.5]}}


if __name__=='__main__':
    if 'bpy' not in sys.modules:textures()
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parent));import pass_common as C
        C.init_scene();C.save_asset(ASSET,build())
