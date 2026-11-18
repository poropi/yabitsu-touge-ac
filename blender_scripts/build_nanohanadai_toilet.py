"""航空写真9.44×8.57mの屋根とSVの白い支持壁/木トラス/玉石壁。
単体アセットのみ。全ての部品は建物平面中心の地面を共通原点とする。
"""
import math,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build(asset_id):
    import bpy
    import nanohanadai_common as C
    assert asset_id=='nanohanadai_toilet'
    parts={}
    def start(name):return C.collection(name)
    def end(name,col):parts[name]=list(col.objects)
    white=C.proc('風化した白いコンクリート',(205,205,190),scale=8)
    edge=C.proc('笠木の灰色',(151,151,136),scale=16)
    floor=C.proc('舗装の灰褐色',(134,129,108),scale=24)
    dark=C.proc('入口奥の影色',(57,62,57),scale=8)
    mortar=C.proc('玉石の目地',(107,108,95),scale=24)
    wood=C.proc('木トラスの茶色',(152,100,58),'wood',5)
    steel=C.proc('接合用亜鉛金属',(111,118,112),scale=20)
    roof=C.proc('灰緑色の金属葺き',(77,89,84),scale=9)
    silver=C.proc('銀灰色のパイプ',(158,165,159),scale=20)
    # 壁体は7.8×6.8、軒は9.44×8.57。中央支持壁と前室の奥行を持つ。
    col=start('01 白壁と入口')
    for x in [-3.78,3.78]:C.box('左右の壁下地',(x,.15,1.72),(.24,6.50,2.46),white)
    C.box('奥の壁',(0,3.28,1.72),(7.8,.24,2.46),white)
    # 正面の広い頭と細い足。単純箱の正面を置かない。
    poly=[(-.55,.50),(.55,.50),(1.22,3.08),(-1.22,3.08)]
    v=[(x,y,z) for y in [-3.46,-3.02] for x,z in poly]
    C.mesh('白い中央の斜め支持壁',v,[(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],white)
    for s in [-1,1]:
        C.box('前室奥の白い間仕切り',(s*2.38,-1.65,1.71),(2.28,.18,2.42),white)
        C.box('入口の暗い奥行',(s*2.35,-1.765,1.48),(.94,.02,1.96),dark)
        C.box('入口の横壁',(s*3.10,-2.52,1.70),(.17,1.55,2.40),white)
        # 外側の壁支持。下から上へ張り出す斜め輪郭。
        for y in [-3.22,2.9]:C.beam('軒下の白い斜め柱',(s*2.72,y,.52),(s*3.55,y,2.95),.25,white,.22)
        C.box('通気用の縦窓',(s*.90,-3.471,2.15),(.12,.017,.58),dark)
    # 中央の掲示板: 内容は読めないため自作の色面/刻線。
    notice=C.proc('掲示板の白',(222,224,209),scale=22)
    C.box('正面の掲示板',(0,-3.488,1.86),(.66,.025,.96),notice)
    blue=C.proc('掲示の見出し青',(113,151,161),scale=25)
    C.box('掲示板の青い見出し',(0,-3.503,2.25),(.55,.006,.055),blue)
    for j in range(7):C.box('判読不能の掲示の短線',(-.02,-3.506,2.12-j*.075),(.45-(j%3)*.06,.004,.014),edge)
    green=C.proc('設備の緑',(65,103,73),scale=8)
    C.cylinder('入口脇の緑の設備',(.96,-3.45,1.18),.10,1.35,green,12)
    end('walls',col)

    col=start('02 灰緑の金属屋根')
    W=4.72;D=4.285;E=3.40;H=5.00;SH=.90
    def roofz(x):return H-(H-E)*max(0,abs(x)-SH)/(W-SH)
    for side in [-1,1]:
        xa,xb=(SH,W) if side>0 else (-W,-SH)
        coords=[(xa,-D,roofz(xa)),(xb,-D,roofz(xb)),(xb,D,roofz(xb)),(xa,D,roofz(xa))]
        vv=coords+[(x,y,z-.10) for x,y,z in coords]
        C.mesh('金属屋根_右' if side>0 else '金属屋根_左',vv,[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],roof)
        for j in range(1,18):
            x=side*(SH+j*(W-SH)/18);C.beam('金属葺きの折り目',(x,-D,roofz(x)+.018),(x,D,roofz(x)+.018),.026,roof,.026)
        for x in [side*(W-.055)]:C.beam('軒先の縁板',(x,-D,E),(x,D,E),.11,roof,.13)
    C.box('中央の平坦な屋根',(0,0,H-.05),(SH*2,8.57,.10),roof)
    end('metal_roof',col)

    col=start('03 茶色の木トラス')
    # 妻側の大三角形、下の逆三角、内部のW形斜材。
    for y in [-4.18,4.10]:
        for a,b in [((-4.63,y,3.35),(-.90,y,4.94)),((-.90,y,4.94),(.90,y,4.94)),((.90,y,4.94),(4.63,y,3.35)),((-4.63,y,3.35),(4.63,y,3.35)),((-3.8,y,2.88),(3.8,y,2.88))]:C.beam('妻面トラスの外周',a,b,.18,wood)
        for side in [-1,1]:
            for a,b in [((side*.90,y,4.94),(side*1.35,y,3.35)),((side*2.05,y,4.45),(side*1.35,y,3.35)),((side*2.05,y,4.45),(side*3.15,y,3.35)),((side*3.15,y,3.35),(side*3.75,y,2.52)),((side*4.63,y,3.35),(side*3.75,y,2.52))]:C.beam('妻面トラスの筋かい',a,b,.135,wood)
            C.box('妻面の接合金具',(side*3.75,y-.105,2.57),(.24,.025,.26),steel)
    for x in [-4.58,4.58]:
        C.beam('側面上弦',(x,-4.12,3.38),(x,4.12,3.38),.18,wood)
        C.beam('側面下弦',(x,-3.82,2.40),(x,3.82,2.40),.16,wood)
        ys=[-4.12,-2.06,0,2.06,4.12]
        for a,b in zip(ys[:-1],ys[1:]):
            mid=(a+b)/2
            C.beam('側面の逆三角',(x,a,3.38),(x,mid,2.40),.145,wood)
            C.beam('側面の逆三角',(x,mid,2.40),(x,b,3.38),.145,wood)
            C.box('側面筋かい金具',(x-.10,mid,2.44),(.025,.22,.21),steel)
    for y in [-3.2,-1.6,0,1.6,3.2]:
        C.beam('屋根の下の横架材',(-3.7,y,3.04),(3.7,y,3.04),.16,wood)
    end('wood_trusses',col)

    col=start('04 屋上中央の設備通路と手すり')
    C.box('中央の細長い設備通路',(0,.20,5.08),(1.38,6.50,.13),silver)
    for x in [-.69,.69]:
        for z in [5.50,6.07]:C.pipe('屋上手すりの横材',(x,-3.05,z),(x,3.45,z),.023,silver)
        for i in range(23):
            y=-3.05+i*6.5/22;C.pipe('屋上の縦格子',(x,y,5.17),(x,y,6.07),.016,silver)
    for y in [-3.05,3.45]:
        for z in [5.50,6.07]:C.pipe('通路端の横材',(-.69,y,z),(.69,y,z),.023,silver)
        for i in range(1,4):
            x=-.69+i*.345;C.pipe('通路端の縦格子',(x,y,5.17),(x,y,6.07),.016,silver)
    C.box('屋上奥の設備箱',(0,2.22,5.59),(1.0,1.35,.85),edge)
    C.pipe('屋上の排気管',(.40,2.80,5.75),(.40,2.80,6.40),.055,silver,10)
    end('roof_rails',col)

    col=start('05 玉石の壁と基壇')
    C.box('建物の基壇',(0,0,.24),(8.0,7.10,.48),edge,.025)
    mats=[C.proc('玉石の色%d'%i,c,scale=18) for i,c in enumerate([(151,148,126),(173,166,143),(130,133,116),(190,185,162),(143,140,121)])]
    rr=random.Random(103006)
    def stonewall(axis,const,start,end,zlo=.48,zhi=2.96):
        if axis=='x':C.box('石壁の目地下地',(const,(start+end)/2,(zlo+zhi)/2),(.19,end-start,zhi-zlo),mortar)
        else:C.box('石壁の目地下地',((start+end)/2,const,(zlo+zhi)/2),(end-start,.19,zhi-zlo),mortar)
        z=zlo;row=0
        while z<zhi-.10:
            h=min(rr.uniform(.22,.34),zhi-z);x=start
            while x<end-.08:
                w=min(rr.uniform(.25,.39),end-x);cx=x+w/2;cz=z+h/2;N=7
                ring=[]
                for i in range(N):
                    a=2*math.pi*i/N;r=rr.uniform(.83,1.0)
                    ring.append((cx+math.cos(a)*w*.49*r,cz+math.sin(a)*h*.48*r))
                vs=[]
                for depth in [0,.11]:
                    for a,b in ring:
                        if axis=='x':vs.append((const+(-1 if const<0 else 1)*(.11+depth),a,b))
                        else:vs.append((a,const+(-1 if const<0 else 1)*(.11+depth),b))
                faces=[tuple(range(N-1,-1,-1)),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)]
                C.mesh('野面石の丸い玉石',vs,faces,mats[rr.randrange(len(mats))]);x+=w
            z+=h;row+=1
    stonewall('x',-3.82,-3.05,3.36);stonewall('x',3.82,-3.05,3.36)
    stonewall('y',3.35,-3.80,3.80)
    # 写真で前面端に回り込む玉石。
    for a,b in [(-3.8,-3.35),(3.35,3.80)]:stonewall('y',-3.35,a,b)
    end('stone_base',col)

    col=start('06 正面のスロープと白い段状擁壁')
    # 左右の端から中央踊り場へ上がる床、原点地面Z=0を維持。
    C.box('入口前の踊り場',(0,-4.16,.43),(3.0,1.58,.14),floor)
    for side in [-1,1]:
        x0=side*1.5;x1=side*8.9
        top=[(x0,-3.38,.50),(x0,-4.96,.50),(x1,-4.96,.08),(x1,-3.38,.08)]
        vs=top+[(x,y,max(0,z-.10)) for x,y,z in top]
        C.mesh('左右から中央へ上がるスロープ',vs,[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],floor)
        for j in range(10):
            t=(j+.5)/10;x=side*(1.5+7.4*t);z=.50-.42*t
            C.box('段状白壁',(x,-5.04,(z+.16)/2),(.75,.25,z+.16),white)
            C.box('灰色笠木',(x,-5.04,z+.175),(.75,.29,.045),edge)
        # 山側は低い縁石。前面白壁には目地を描画可能な薄い刻線を実形状で入れる。
        for j in range(10):
            t=(j+.5)/10;x=side*(1.5+7.4*t);z=.50-.42*t
            C.box('山側の灰色縁石',(x,-3.32,z+.06),(.75,.18,.18),edge)
            if j<8:C.box('白壁の水平目地',(x,-5.168,z*.50),(.73,.004,.009),edge)
    C.box('正面踊り場の白壁',(0,-5.04,.31),(3,.25,.62),white)
    C.box('踊り場笠木',(0,-5.04,.64),(3,.29,.045),edge)
    end('ramp_walls',col)

    col=start('07 丸パイプのスロープ手すり')
    for side in [-1,1]:
        for h in [.50,.93]:
            C.pipe('スロープに沿う二段の丸手すり',(side*1.45,-4.92,.50+h),(side*8.75,-4.92,.08+h),.026,silver,10)
            # 端で滑らかに下がる曲線を3分節で近似。
            for k in range(3):
                a=math.pi*.5*k/3;b=math.pi*.5*(k+1)/3
                C.pipe('端部の曲がったパイプ',(side*(8.75+.14*math.sin(a)),-4.92,.08+h-.14*(1-math.cos(a))),(side*(8.75+.14*math.sin(b)),-4.92,.08+h-.14*(1-math.cos(b))),.026,silver,10)
        for j in range(5):
            t=j/4;x=side*(1.5+7.2*t);z=.50-.42*t
            C.pipe('手すり支柱',(x,-4.92,z),(x,-4.92,z+.93),.027,silver,10)
            C.box('支柱のベース',(x,-4.92,z+.025),(.11,.11,.035),steel)
    end('ramp_rails',col)
    return {'parts':parts,'origin':'屋根平面中心の真下の地面。全7部品が同じ原点を共有。正面−Yは駐車場側(南東/約149度)。+Xは正面から見て右(北東)。','dimension_basis':'gsat21の屋根四隅(373,615),(506,535),(578,656),(445,736)px。zoom21/緯度35.4104428のWebMercator 0.06083777m/pxより9.44×8.57m。5mスケール約82pxと照合。輪郭読み±0.5m程度。壁体7.8×6.8m、高さ等はSV写真から推定。','roof_section_basis':'sw000/mid270の同一点を透視ray交差。SWカメラ高1.8m固定/MID相対高も解いた場合、左右肩4.747/4.761m、軒3.400/3.322m、肩間1.579m、軒間8.153m。両斜面約23度。航空写真外形9.44mに拡大して軒3.4m/肩5.0m/中央平坦幅1.8mで近似。絶対高はカメラ高の仮定に依存。','aerial_measurement':{'image':'gsat21.png','roof_corners_px':[[373,615],[506,535],[578,656],[445,736]],'m_per_pixel':.06083777,'roof_dimensions_m':[9.44,8.57],'front_azimuth_deg':149,'width_azimuth_deg':59},'approximations':['背面は樹木に隠れるため、壁/石/トラスを見えている面から推定して補完。屋根外形は航空採寸、壁体寸法・高さ・棟位置・設備通路断面は写真からの推定。','石の個々の並び、木材の継手、屋上設備箱/パイプの詳細は簡略化。','トイレ内部は再現せず、入口の奥行と暗い開口だけを外観用に作った。','掲示文は判読できないため短い線と色面。','スロープは両端から中央へ上がる17.8mの局所モデル。現地の舗装面の細かな不陸・道路縦断勾配は配置担当側で調整。'],'refs':['gsat21.png','sv_sw_000.png','sv_sw_045.png','sv_sw_315.png','sv_mid_225.png','sv_mid_270.png','sv_mid_315.png','sv_user_000.png','sv_user_090.png'],'reference_views':{'reference_sw000':(-1.65,-2.5,-.20),'reference_mid270':(1.8,-2.6,-.20)}}
