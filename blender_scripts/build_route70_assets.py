"""依頼08: 部品をasset-id順に再生成。通常Pythonで実行後、route70_blender.pyで制作元を保存。"""
from pathlib import Path
import argparse,math,json
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from route70_common import *
ASSETS=['r70_curve_mirror','r70_utility_pole','r70_pylon','r70_lights','r70_km_post','r70_delineator','r70_road_signs','r70_boards','r70_guardrail','r70_fence','r70_retaining_wall','r70_drain','r70_construction','r70_gate','r70_misc']
LIMITS=[1500,2000,3000,1500,300,300,1000,1000,800,600,800,400,800,800,1500]

def mirror_art():
    y,x=np.mgrid[0:768,0:768]/767
    light=145+48*(1-y)+18*np.sin(x*4+y*2);a=np.zeros((768,768,4),np.uint8);a[:,:,:3]=np.clip(light[:,:,None]*[.91,1.,1.03],0,255);a[:,:,3]=255
    return Image.fromarray(a)

def mirror(m,w,h,z,angle=0,x=0):
    start=len(m.v)
    m.disc('鏡の橙の縁と背面',(0,0,z),w/2,h/2,.075,'orange',32)
    # 凸面。鏡は正面-Yに張り出す。画像は自作の銀色勾配で動的反射ではない。
    vs=[(0,-.142,z)];uv=[(.5,.5)];fs=[]
    for rr in [.34,.68,.94]:
        for t in np.arange(32)*2*math.pi/32:
            vs.append((w/2*rr*math.cos(t),-.042-.1*(1-rr*rr),z+h/2*rr*math.sin(t)));uv.append((.5+.5*rr*math.cos(t),.5-.5*rr*math.sin(t)))
    fs=[(0,1+i,1+(i+1)%32) for i in range(32)]
    for j in range(2):
        for i in range(32):a=1+j*32+i;b=1+j*32+(i+1)%32;fs.append((a,a+32,b+32,b))
    m.surface('凸面鏡',vs,fs,uv=uv)
    # 上半分の庇。閉じた上部パネルのみ、反射面を覆わない。
    ts=np.linspace(.10,math.pi-.10,13);vs=[(w*.51*math.cos(t),yy,z+h*.51*math.sin(t)) for yy in [.04,-.125] for t in ts]
    m.surface('上半分の雨よけ',vs,[(i,i+1,i+14,i+13) for i in range(12)],'orange')
    for zz in [z-.16,z+.16]:m.box('裏面金具',(0,.07,zz),(.13,.07,.055),'metal')
    m.transform(start,angle,(x,0,0))

def pole(m,kind='concrete',transformer=False):
    wood=kind=='wood';green=kind=='steel_green';h=9 if wood else 8 if green else 10
    c='wood' if wood else 'green' if green else 'concrete';r=.145 if wood else .10 if green else .20
    m.tube('柱本体',(0,0,0),(0,0,h),r,c,12,r2=r*.62)
    for z in [h-.6,h-1.7]:
        m.box('腕金',(0,0,z),(1.65,.09,.09),'metal')
        for x in [-.68,0,.68]:
            m.tube('碍子の心棒',(x,0,z),(x,0,z+.22),.025,'metal',6)
            for zz in [z+.08,z+.14,z+.20]:m.tube('碍子ひだ',(x,0,zz-.025),(x,0,zz+.025),.065,'white',8)
        m.beam('腕金の斜材',(-.65,0,z),(0,0,z-.5),.035,c='metal')
    for i,z in enumerate(np.arange(1.8,h-1,.45)):
        side=-1 if i%2 else 1;m.tube('足場ボルト',(side*r*.75,0,z),(side*(r+.17),0,z),.012,'metal',5)
    for z in [3.0,6.5,h-1.1]:m.tube('締付バンド',(0,0,z-.035),(0,0,z+.035),r*(1-.38*z/h)+.012,'metal',12)
    if transformer:
        m.box('機器支持棚',(0,-.22,h-2.6),(.85,.7,.10),'metal')
        m.tube('円筒変圧器',(0,-.43,h-2.6),(0,-.43,h-1.5),.26,'metal',16)
        m.tube('上蓋',(0,-.43,h-1.52),(0,-.43,h-1.46),.29,'metal',16)
        for x in [-.13,.13]:
            for i in range(3):m.tube('変圧器ブッシング',(x,-.43,h-1.45+i*.055),(x,-.43,h-1.42+i*.055),.055,'white',8)
    return h

def arm_light(m,arc=False):
    h=7.4 if arc else 7.;r=.065
    m.tube('細い鋼管柱',(0,0,0),(0,0,h-1.0),r,'metal',12,r2=.04)
    # 道路方向-Yへなめらかに曲がる管。
    pts=[(0,-1.15*(1-math.cos(t)),h-1+1.0*math.sin(t)) for t in np.linspace(0,math.pi/2,9)]
    for a,b in zip(pts,pts[1:]):m.tube('円弧のアーム',a,b,.039,'metal',8)
    m.sphere('水銀灯風の灯具',(0,-1.30,h-.05),(.19,.36,.12),'rust',16,5)
    m.sphere('灯具下面のガラス',(0,-1.31,h-.115),(.16,.30,.04),'white',12,4)
    m.tube('柱下端の補強',(0,0,0),(0,0,.25),.08,'metal',12)

def net_art(w,h,c='metal'):
    # 実寸の縦横比を保って菱形を描き、最後にアトラス正方形へ写す。
    iw=1024;ih=max(128,int(1024*h/w));im=Image.new('RGBA',(iw,ih),(0,0,0,0));d=ImageDraw.Draw(im);step=max(12,int(iw*.05/w));col=PALETTE[c]
    for off in range(-ih,iw+ih,step*2):
        d.line((off,0,off+ih,ih),fill=(*col,255),width=2);d.line((off,0,off-ih,ih),fill=(*col,255),width=2)
    return im

def wall_art(kind):
    N=1024;y,x=np.mgrid[0:N,0:N]/(N-1);rng=np.random.default_rng(721)
    noise=3*np.sin(2*math.pi*(x*77+y*63))+2*np.sin(2*math.pi*(x*149-y*121))+6*np.sin(2*math.pi*x*4)*np.sin(2*math.pi*y*2)
    if kind=='stone':
        seeds=[((i+rng.uniform(.15,.85))/10,(j+rng.uniform(.15,.85))/3) for j in range(3) for i in range(10)]
        near=np.full((N,N),1e5);second=near.copy();ix=np.zeros((N,N),int)
        for i,(xx,yy) in enumerate(seeds):
            dx=np.abs(x-xx);dx=np.minimum(dx,1-dx)*4;dy=np.abs(y-yy);dy=np.minimum(dy,1-dy);dist=dx*dx+dy*dy;mask=dist<near;second=np.where(mask,near,np.minimum(second,dist));ix=np.where(mask,i,ix);near=np.minimum(near,dist)
        edge=np.sqrt(second)-np.sqrt(near);base=75+(ix*17%35)+noise;groove=edge<.016;base=np.where(groove,41+noise*.4,base);base+=np.minimum(edge/.035,1)*12
        rgb=np.stack([base*.95,base*1.01,base*.96],axis=2)
    elif kind=='block':
        a=x*10+y*4;b=x*10-y*4;fa=a%1;fb=b%1;edge=np.minimum.reduce([fa,1-fa,fb,1-fb]);base=117+noise;idx=(np.floor(a)*3+np.floor(b)*7)%5;base+=idx*3;base=np.where(edge<.025,63+noise,base);base+=np.clip(edge/.08,0,1)*9;rgb=np.stack([base*.97,base,base*.97],axis=2)
    elif kind=='wood_lagging':
        base=104+noise+10*np.sin(x*2*math.pi*27+y*2*math.pi*3);base=np.where((y*4)%1<.025,38,base);rgb=np.stack([base,base*.79,base*.57],axis=2)
    else:
        base=(143 if kind=='concrete' else 131 if kind=='shotcrete' else 165)+noise
        drip=10*(np.sin(2*math.pi*x*11)+1)*(np.cos(2*math.pi*y)+1);base-=drip
        rgb=np.stack([base*.97,base,base*.92],axis=2)
    moss=np.clip((np.sin(2*math.pi*(x*3+y))+np.cos(2*math.pi*(x*7-y*2))-.65)*.20,0,.35)
    rgb=rgb*(1-moss[:,:,None])+np.array([62,78,41])*moss[:,:,None]
    a=np.zeros((N,N,4),np.uint8);a[:,:,:3]=np.clip(rgb,0,255);a[:,:,3]=255
    # 周期関数に加え最終画素も一致させて隣接タイルのピクセル境界を保証。
    a[:,-1]=a[:,0];a[-1]=a[0];return Image.fromarray(a)

def build(asset):
    A=Asset(asset,LIMITS[ASSETS.index(asset)])
    if asset=='r70_curve_mirror':
        for n,w,h,z,kind in [('round_1',.8,.8,2.85,'single'),('round_2',.6,.6,3.2,'double'),('oval_1',1.,.72,3.05,'single'),('oval_1_twin_post',1.2,.8,3.0,'twin')]:
            m=Mesh();m.tube('橙色支柱',(0,0,0),(0,0,z+.1),.0508,'pole_orange',12)
            if kind=='twin':m.anchor=(.07,0,0)
            if kind=='twin':m.tube('並列の橙色支柱',(.14,0,0),(.14,0,z+.1),.0508,'pole_orange',12)
            mirror(m,w,h,z,angle=-.2 if kind=='double' else 0,x=.06 if kind=='double' else 0)
            if kind=='double':mirror(m,w,h,z-.76,angle=math.radians(-55),x=-.08)
            if n=='round_1':
                m.box('注意札',(0,-.058,1.8),(.16,.015,.54),'white')
                m.panel('注意の縦書き',0,-.066,1.8,.145,.51,rect=(8,784,200,1016))
            art=mirror_art();p=A.add(n,'丸2面・上下と向きを変えた鏡' if kind=='double' else '楕円鏡・二本柱' if kind=='twin' else '丸1面鏡' if n=='round_1' else '横長の楕円鏡',m,art,{n+'_gray.png':(None,{'pole_orange':PALETTE['metal']})},refpart='oval_1' if kind=='twin' else n,notes=['鏡面は自作銀灰色の勾配。実景反射や動的反射ではない。','背面金具・庇の奥行き・鏡の曲率は推定。'])
            if n=='round_1':
                for file in p['textures']:
                    for folder in [A.out,A.src]:
                        im=Image.open(folder/file);im.paste(lettering('注意',192,232,(239,239,221),vertical=True),(8,784));im.save(folder/file)
        A.reclassified=['S5-0995-03は横並び規制標識の背面でありround_2へ割り当てない。','S4-0361-03の2本柱はoval_1_twin_post。丸1面に見える鏡が写真の斜視で楕円になる場合がある。']
    elif asset=='r70_utility_pole':
        for n in ['concrete','concrete_transformer','wood','steel_green']:
            m=Mesh();pole(m,n if n in ['wood','steel_green'] else 'concrete',n=='concrete_transformer')
            A.add(n,{'concrete':'テーパー付きコンクリート電柱','concrete_transformer':'円筒変圧器付きコンクリート電柱','wood':'木製電話柱風の腕金・碍子・足場付き柱','steel_green':'緑塗装鋼管柱'}[n],m,notes=['電線・支線は依頼範囲外。腕金の本数と金具の細部、埋設長を除いた高さは推定。','傾きは設置時に指定するため原型は直立。'])
        A.reclassified=['thin_concreteの参照4例から独立した電話専用コンクリート柱を確認できないため、重複型を作らずconcreteを流用。','wood参照は材質確証に乏しい。木柱の幹色・径は推定。制御箱はmiscの独立部品。']
    elif asset=='r70_pylon':
        m=Mesh();levels=[(0,4.1),(7,3.4),(14,2.7),(21,2.0),(28,1.3),(34,.7),(39,.22)]
        for (z,r),(zz,rr) in zip(levels,levels[1:]):
            for i in [-1,1]:
                for j in [-1,1]:m.beam('四隅の主柱',(i*r,j*r,z),(i*rr,j*rr,zz),.18,c='metal')
            for side in range(4):
                xy=[(-r,-r),(r,-r),(r,r),(-r,r)];top=[(-rr,-rr),(rr,-rr),(rr,rr),(-rr,rr)];i=side;j=(side+1)%4
                m.beam('水平材',(*xy[i],z),(*xy[j],z),.09,c='metal')
                m.beam('交差斜材A',(*xy[i],z),(*top[j],zz),.07,c='metal');m.beam('交差斜材B',(*xy[j],z),(*top[i],zz),.07,c='metal')
        for z,length in [(24,6.8),(30,5.5),(35,4.3)]:
            for side in [-1,1]:
                for yy in [-.5,.5]:
                    m.beam('腕の下弦',(0,yy,z),(side*length,yy,z),.11,c='metal');m.beam('腕の斜弦',(0,yy,z+2),(side*length,yy,z),.09,c='metal')
                m.beam('腕の先端',(side*length,-.5,z),(side*length,.5,z),.1,c='metal')
                for i in range(7):m.tube('吊下げ碍子',(side*length,0,z-.2-i*.14),(side*length,0,z-.12-i*.14),.18,'white',6)
        A.add('tower','遠景用の格子送電鉄塔。格子も立体の鋼材。',m,notes=['高さ39mは依頼目安40mに基づく推定。写真では基礎・全段格子を解像できず、腕の長さ・回線配置は近似。送電線は制作しない。'])
    elif asset=='r70_lights':
        for n in ['arm','arm_arc']:
            m=Mesh();arm_light(m,n=='arm_arc');A.add(n,'円弧アームの独立柱街灯',m,refpart='wall' if n=='arm_arc' else n,notes=['灯具輪郭と腕の曲率は写真からの近似。発光・夜間設定は含まない。'])
        m=Mesh();m.box('壁の取付板',(0,.04,.28),(.12,.03,.32),'metal');m.beam('短い壁付け腕',(0,.02,.28),(0,-.20,.28),.04,c='metal');m.tube('グローブ台',(0,-.20,0),(0,-.20,.08),.10,'metal',12);m.sphere('円筒グローブ',(0,-.20,.20),(.09,.09,.15),'white',12,5)
        A.add('wall','土留め壁の小型グローブ灯',m,notes=['写真の解像度が低いため細部と寸法は推定。原点は灯具下面を含む取付位置基準。'],origin='取付基準中央の最下点。地上に直接立てる部品ではない。')
        A.reclassified=['S5-0981-01は壁付けではなくarm_arc。wallはS5-0784-04のみ。']
    elif asset=='r70_km_post':
        def face(s):return lettering(s,900,500,(31,85,146),(247,247,235),border=22)
        m=Mesh();board(m,.45,.25,1.6,1,'white',.025)
        vals=['20.5','21.5','25.5','26.5','27.0','27.5','28.0']
        variants={f'post_face_{s}.png':(face(s+'km'),None) for s in vals}
        A.add('post','白い細柱と青い距離札。数字は画像差替。',m,face('20.5km'),variants,notes=['26.5と28.0は台帳の読み取りに不確かさがあるため、画像を用意するが設置前に再確認が必要。','板0.45×0.25m、中心高1.6m。遠近写真の寸法目安を採用。'])
    elif asset=='r70_delineator':
        m=Mesh();m.tube('柔軟ポールの広い根元',(0,0,0),(0,0,.06),.115,'orange',12);m.tube('柔軟オレンジポール',(0,0,.06),(0,0,1.),.042,'orange',12,r2=.036)
        for z in [.75,.88]:m.tube('白反射帯',(0,0,z),(0,0,z+.065),.043,'white',12)
        A.add('pole','白反射帯2本の樹脂ポール',m)
        m=Mesh();m.box('黄色い角杭',(0,0,.75),(.15,.15,1.5),'yellow');A.add('pole_yellow_square','黄色角形の標識杭（樹脂ポールとは別形状）',m,refpart='pole')
        m=Mesh();m.tube('反射板の取付脚',(0,.03,0),(0,.03,.15),.014,'metal',8);m.disc('反射板の白縁',(0,0,.17),.063,.063,.025,'white',16);m.disc('橙反射板',(0,-.015,.17),.05,.05,.006,'orange',16)
        A.add('rail_reflector','ガードレールに後付けする丸反射体',m,origin='取付脚の根元中央。支柱上面へ設置。')
        def chev(w,h):
            im=Image.new('RGBA',(600,600),(191,39,37,255));d=ImageDraw.Draw(im);d.rectangle((4,4,595,595),outline=(238,238,217),width=22);d.polygon([(130,65),(290,65),(500,300),(290,535),(130,535),(340,300)],fill=(238,238,217));return im
        m=Mesh();board(m,.6,.6,1.25,1,'metal',.025);A.add('chevron','赤地白矢羽根の角板と細い柱',m,chev(.6,.6),notes=['写真の右向きに合わせた汎用角板。左右反転は画像差替で指定可能。'])
        m=Mesh();m.box('ビーム取付板',(0,0,.08),(.30,.02,.16),'white');m.panel('矢羽根印刷',0,-.0105,.08,.29,.15);A.add('chevron_rail','ガードレールへ直接付ける小さな横長矢羽根',m,chev(.3,.16),refpart='chevron',origin='板の下辺中央。柱なし。')
    else:return build_remaining(A)
    return A.save()

# 後半のカテゴリは同ファイル内のbuild_remainingで順次生成。
def build_remaining(A):
    asset=A.id
    if asset=='r70_road_signs':
        law='https://laws.e-gov.go.jp/law/335M50004002003/ 別表第二（図092:警戒45cm、120:丸60cm、059/060:交差道路標識80cm）。work/reference_audit.md参照。'
        def warning(kind):
            im=Image.new('RGBA',(768,768),(230,186,42,255));d=ImageDraw.Draw(im)
            d.polygon([(384,23),(745,384),(384,745),(23,384)],fill=(229,186,42),outline=(30,31,24),width=14)
            if kind=='deer':
                # 跳躍する鹿の外形。現物図柄を写真から輪郭近似した自作線画。
                d.polygon([(204,415),(255,339),(358,346),(432,300),(466,227),(492,221),(509,258),(489,327),(458,370),(421,407),(350,406),(285,431)],fill=(25,28,22))
                for pts in [[(285,387),(239,464),(160,492)],[(332,401),(299,476),(258,505)],[(418,373),(484,415),(554,400)],[(443,351),(503,380),(562,360)],[(263,350),(222,307),(188,300)],[(488,239),(490,180),(462,150)],[(490,201),(527,170)],[(474,183),(455,179)]]:d.line(pts,fill=(25,28,22),width=20 if len(pts)==3 and pts[0][1]>300 else 9)
            else:
                pts=[(344,600),(344,388),(490,276)] if kind=='right' else [(424,600),(424,388),(278,276)]
                d.line(pts,fill=(23,26,20),width=63,joint='curve')
                x,y=pts[-1];d.polygon([(x+(-74 if kind=='right' else 74),y-33),(x+(45 if kind=='right' else -45),y-52),(x+(23 if kind=='right' else -23),y+70)],fill=(23,26,20))
            return im
        m=Mesh();half=.45/math.sqrt(2);out=[(0,2.2-half),(half,2.2),(0,2.2+half),(-half,2.2)]
        m.anchor=(0,.055,0);m.prism('45cm角の警戒ひし形',out,.024,'yellow',True);m.tube('白い支柱',(0,.055,0),(0,.055,2.4),.03,'white',10)
        A.add('warning_diamond','45cm角の警戒ひし形・白柱',m,warning('deer'),{f'warning_diamond_face_{n}.png':(warning(n),None) for n in ['deer','left','right']},basis=law,notes=['鹿の図柄は写真の輪郭から近似。支柱中心高2.2mは写真推定。'])
        m=Mesh();center=2.65;half=.45/math.sqrt(2);out=[(0,center-half),(half,center),(0,center+half),(-half,center)]
        m.prism('警戒ひし形と補助板の上板',out,.024,'yellow',True)
        m.tube('共用の一本柱',(0,.055,0),(0,.055,2.8),.03,'white',10);m.anchor=(0,.055,0)
        m.box('黄橙色の補助板',(0,0,1.95),(.4,.025,.7),'yellow');m.panel('補助板の右カーブ注意',0,-.0135,1.95,.39,.69,rect=(8,784,312,1012))
        p=A.add('warning_diamond_with_text','右カーブの警戒ひし形＋縦書き補助板・一本柱',m,warning('right'),refpart='warning_diamond',basis=law)
        for folder in [A.src,A.out]:
            im=Image.open(folder/'warning_diamond_with_text.png');im.paste(lettering('右カーブ\n注意',304,532,vertical=True).resize((304,228)),(8,784));im.save(folder/'warning_diamond_with_text.png')
        m=Mesh();board(m,.4,.8,2.0,1,'white',.026)
        texts={'construction_50m':'50m先\n工事中','narrow':'この先\n幅員狭小','right_curve':'右カーブ\n注意','left_curve':'左カーブ\n注意','slow':'速度落せ'}
        A.add('warning_text','黄橙の縦長注意板（ひし形とは別形）',m,lettering('右カーブ\n注意',384,768,vertical=True),{f'warning_text_face_{k}.png':(lettering(t,384,768,vertical=True),None) for k,t in texts.items()},refpart='warning_diamond')
        def regulatory(s):
            im=Image.new('RGBA',(768,768),(243,243,231,255));d=ImageDraw.Draw(im);d.ellipse((12,12,756,756),fill=(200,43,41));d.ellipse((73,73,695,695),fill=(244,244,235))
            if s in ['30','40']:d.text((384,390),s,font=font(340),fill=(23,63,137),anchor='mm')
            else:
                d.arc((240,185,510,500),180,360,fill=(26,65,132),width=48);d.line((240,343,240,540),fill=(26,65,132),width=48);d.line((510,340,510,540),fill=(26,65,132),width=48);d.polygon([(240,592),(167,481),(313,481)],fill=(26,65,132));d.line((152,147,620,620),fill=(202,43,41),width=57)
            return im
        m=Mesh();m.anchor=(0,.055,0);m.disc('直径60cm丸標識',(0,0,2.2),.30,.30,.025,'metal',40,True);m.tube('支柱',(0,.055,0),(0,.055,2.4),.03,'metal',10)
        A.add('regulatory_round','最高速度・転回禁止の丸標識',m,regulatory('30'),{f'regulatory_round_face_{s}.png':(regulatory(s),None) for s in ['30','40','no_uturn']},basis=law)
        m=Mesh();m.anchor=(0,.08,0);m.tube('共通支柱',(0,.08,0),(0,.08,2.6),.036,'metal',10);m.box('横腕',(0,.06,2.4),(1.35,.06,.06),'metal')
        art=Image.new('RGBA',(768,768));art.paste(regulatory('no_uturn').resize((384,768)),(0,0));art.paste(regulatory('40').resize((384,768)),(384,0))
        for i,x in enumerate([-.35,.35]):
            m.disc('丸板'+str(i),(x,0,2.4),.30,.30,.025,'metal',32,True)
            for j in range(len(m.uv)-32,len(m.uv)):
                u,v=m.uv[j];m.uv[j]=((8+(u*1024-8)*.5+i*380)/1024,v)
        m.box('補助標識の厚み',(0,0,1.9),(.55,.025,.15),'white');m.panel('左右矢印',0,-.014,1.9,.53,.13,rect=(8,784,768,1016))
        p=A.add('regulatory_double','転回禁止＋最高速度40の横並び二枚と左右矢印',m,art,refpart='regulatory_round',basis=law)
        for folder in [A.src,A.out]:
            im=Image.open(folder/'regulatory_double.png');im.paste(lettering('↔',760,232,(241,241,230),(196,43,38)),(8,784));im.save(folder/'regulatory_double.png')
        # 県道盾は国道のおむすびにせず左右が尖った六角形。
        def route_face(s):
            im=Image.new('RGBA',(768,768),(26,76,139,255));d=ImageDraw.Draw(im)
            d.line([(18,384),(140,26),(628,26),(750,384),(628,742),(140,742),(18,384)],fill=(240,241,227),width=12)
            d.text((384,140),'県道',font=font(76),fill=(243,243,232),anchor='mm')
            d.text((384,393),s,font=font(320),fill=(243,243,232),anchor='mm')
            d.text((384,634),'神奈川',font=font(74),fill=(243,243,232),anchor='mm')
            return im
        m=Mesh();z=2.3;w=.46;h=.40;out=[(-w/2,z),(-w*.32,z-h/2),(w*.32,z-h/2),(w/2,z),(w*.32,z+h/2),(-w*.32,z+h/2)]
        m.anchor=(0,.05,0);m.prism('県道番号六角板',out,.025,'blue',True);m.tube('細い白い柱',(0,.05,0),(0,.05,z+.12),.028,'white',10)
        A.add('route_hexagon','県道70の六角形路線番号板',m,route_face('70'),{'route_hexagon_face_70.png':(route_face('70'),None)},basis=law+' 通常県道盾は外接幅0.46m・高さ0.40mを写真比で推定。')
        m.box('県道盾下の白い地名板',(0,0,1.99),(.42,.025,.15),'white');m.panel('清川村の印刷面',0,-.0135,1.99,.40,.13,rect=(8,784,768,1016))
        p=A.add('route_hexagon_place','県道70＋白い清川村の地名小板',m,route_face('70'),refpart='route_hexagon',basis=law,notes=['白い小板の下段は判読できないため空白。'])
        for folder in [A.src,A.out]:
            im=Image.open(folder/'route_hexagon_place.png');im.paste(lettering('清川村',760,232,(232,232,218),(42,47,38)),(8,784));im.save(folder/'route_hexagon_place.png')
        m=Mesh();z=2.6;out=[(-.4,z),(-.3,z-.12),(-.155,z-.12),(-.10,z-.135),(.10,z-.135),(.155,z-.12),(.3,z-.12),(.4,z),(.3,z+.12),(.155,z+.12),(.10,z+.135),(-.10,z+.135),(-.155,z+.12),(-.3,z+.12)]
        # 中央の盾が上下へ15mm突出する外形。長い横板と中央盾を立体で重ねる。
        out=[(-.4,z),(-.3,z-.12),(.3,z-.12),(.4,z),(.3,z+.12),(-.3,z+.12)]
        m.prism('80cm幅の交差道路横板',out,.025,'blue',True)
        shield=[(-.155,z),(-.10,z-.135),(.10,z-.135),(.155,z),(.10,z+.135),(-.10,z+.135)]
        m.prism('中央の青い六角盾64',shield,.030,'blue',True,y=-.019)
        # 中央盾はアトラス下側の独立領域を使い、横板の矢印表示と分離。
        for j in range(len(m.uv)-6,len(m.uv)):
            u,v=m.uv[j];m.uv[j]=((8+(u*1024-8)*.40)/1024,(784+(v*1024-8)*.30)/1024)
        m.tube('支柱',(0,.06,0),(0,.06,z+.06),.03,'metal',10);m.anchor=(0,.06,0)
        cross=Image.new('RGBA',(1024,308),(33,100,112,255));dd=ImageDraw.Draw(cross)
        dd.line([(20,154),(150,16),(874,16),(1004,154),(874,292),(150,292),(20,154)],fill=(239,240,226),width=12)
        dd.polygon([(125,154),(230,70),(230,238)],fill=(239,240,226));dd.polygon([(899,154),(794,70),(794,238)],fill=(239,240,226))
        p=A.add('route_crossroad_64','交差道路標識118の2-B/C、中央六角盾64',m,cross,{'route_crossroad_64_face_64.png':(cross,None)},refpart='route_hexagon',basis=law,notes=['中央の青い盾を横板より上下各15mm突出させた。微細な県道欄は省略。'])
        for file in p['textures']:
            for folder in [A.src,A.out]:
                im=Image.open(folder/file);im.paste(lettering('64',304,228,(28,76,138),(242,242,230),border=6),(8,784));im.save(folder/file)
        m=Mesh();board(m,.35,.95,2.05,1,'metal',.025)
        bridge={name:lettering(name,350,950,(233,235,219) if name=='檜沢橋' else (35,84,139),(36,81,137) if name=='檜沢橋' else (239,241,226),True,12) for name in ['檜沢橋','常世橋','地獄沢橋']}
        A.add('bridge_name_plate','青白の縦書き橋名板',m,bridge['檜沢橋'],{f'bridge_name_plate_face_{k}.png':(im,None) for k,im in bridge.items()},notes=['札掛橋・清滝橋・?小屋橋は判読不確かなため文字画像を作らない。'])
        m=Mesh();board(m,.5,1.3,1.85,1,'metal',.028);im=lettering('吉田橋',500,1300,(217,207,166),(35,33,24),True,9);A.add('bridge_name_plate_yoshida','クリーム色の大きい吉田橋名板',m,im,{'bridge_name_plate_yoshida_face_吉田橋.png':(im,None)},refpart='bridge_name_plate')
        A.reclassified=['guide_blueは地名が確定しないため依頼の指示どおり制作しない。','駐車注意の長方形はr70_boards.small_plateに分離。','県道64はroute_crossroad_64。70と同じ盾の画像替えにはしない。','縦長注意文字板をwarning_textとしてひし形から分離。']
    elif asset=='r70_boards':
        def addboard(n,meaning,w,h,z,text,bg=(234,202,53),fg=(27,30,24),legs=1,frame=False,vertical=False,ref='post_board',c='white',stand=False):
            m=Mesh();board(m,w,h,z,legs,c,.025,frame=frame)
            if stand:
                for x in [-w*.4,w*.4]:
                    m.beam('後ろへ開いた自立脚',(x,.45,0),(x,.04,z+h*.32),.04,c='metal');m.beam('転倒防止の足',(x,-.16,.025),(x,.48,.025),.04,c='metal')
            im=lettering(text,int(768*w/h),768,bg,fg,vertical,10)
            if n=='post_board':
                d=ImageDraw.Draw(im);ww,hh=im.size;cx=ww*.79;cy=hh*.82;outer=min(ww,hh)*.14
                pts=[(cx+outer*(1 if i%2==0 else .40)*math.cos(i*math.pi/7),cy+outer*(1 if i%2==0 else .40)*math.sin(i*math.pi/7)) for i in range(14)];d.polygon(pts,fill=(36,34,25))
            elif n=='post_board_forest':
                im=Image.new('RGBA',im.size,(205,191,137,255));d=ImageDraw.Draw(im);ww,hh=im.size
                for i,t in enumerate(['森林整備を','行っています']):d.text((ww/2,hh*(.18+i*.21)),t,font=font(ww*.118),fill=(37,84,139),anchor='mm')
                d.rectangle((ww*.04,hh*.57,ww*.96,hh*.71),fill=(38,91,149))
                # 小文字は判読不能なので帯以下は空白。原図の配色だけを保持。
            elif n=='post_board_bosco':
                im=Image.new('RGBA',im.size,(27,66,49,255));d=ImageDraw.Draw(im);ww,hh=im.size
                # 写真の中央に大きな黄色のS型ロゴ2本。
                for cx in [.46,.63]:
                    pts=[(ww*(cx+.05),hh*.06),(ww*(cx-.035),hh*.13),(ww*(cx-.07),hh*.25),(ww*(cx-.025),hh*.43),(ww*(cx+.025),hh*.53),(ww*(cx+.015),hh*.61),(ww*(cx-.07),hh*.68)]
                    d.line(pts,fill=(225,191,43),width=int(ww*.075),joint='curve')
                d.text((ww*.23,hh*.34),'BO',font=font(ww*.245),fill=(239,233,195),anchor='mm');d.text((ww*.80,hh*.34),'CO',font=font(ww*.245),fill=(239,233,195),anchor='mm')
                d.text((ww*.18,hh*.72),'50M',font=font(ww*.14),fill=(242,239,220),anchor='mm');d.text((ww*.48,hh*.86),'直進',font=font(ww*.15),fill=(242,239,220),anchor='mm')
                d.polygon([(ww*.78,hh*.64),(ww*.87,hh*.51),(ww*.96,hh*.64),(ww*.92,hh*.64),(ww*.92,hh*.81),(ww*.82,hh*.81),(ww*.82,hh*.64)],fill=(240,238,217))
            elif n.startswith('construction_board'):
                ww,hh=im.size;im=Image.new('RGBA',im.size,(235,235,222,255));d=ImageDraw.Draw(im)
                d.rectangle((6,6,ww-7,hh-7),outline=(30,76,141),width=12);d.rectangle((16,16,ww-17,hh*.18),fill=(32,76,140))
                d.text((ww/2,hh*.10),'ご迷惑をおかけします' if n.endswith('bridge') else 'ご協力をお願いします',font=font(ww*.064),fill=(243,242,224),anchor='mm')
                rows=['橋の塗り替えを','しています'] if n.endswith('bridge') else ['土砂崩壊から','道路を守る工事を','しています']
                for i,row in enumerate(rows):d.text((ww/2,hh*(.27+i*.12)),row,font=font(ww*.10),fill=(31,69,127),anchor='mm')
                d.rectangle((20,hh*.66,ww-20,hh*.82),fill=(31,76,140))
                d.text((ww/2,hh*.71),'令和7年8月29日まで',font=font(ww*.065),fill=(241,239,219),anchor='mm');d.text((ww/2,hh*.78),'8:00〜17:00',font=font(ww*.073),fill=(241,239,219),anchor='mm')
                if not n.endswith('bridge'):d.text((ww/2,hh*.90),'道路災害防除工事',font=font(ww*.07),fill=(33,72,130),anchor='mm')
            elif n in ['stand_board','stand_board_narrow']:
                im=Image.new('RGBA',im.size,(235,205,45,255));d=ImageDraw.Draw(im);ww,hh=im.size
                d.polygon([(ww*.5,hh*.05),(ww*.85,hh*.27),(ww*.15,hh*.27)],outline=(32,31,21),width=5)
                d.line([(ww*.64,hh*.10),(ww*.60,hh*.19),(ww*.73,hh*.25)],fill=(32,31,21),width=6)
                for xx,yy,rr in [(.47,.17,.023),(.40,.23,.030),(.31,.25,.020)]:d.ellipse((ww*(xx-rr),hh*(yy-rr),ww*(xx+rr),hh*(yy+rr)),fill=(32,31,21))
                for i,ch in enumerate('落石注意' if n=='stand_board' else '幅員狭少'):d.text((ww*.5,hh*(.38+i*.15)),ch,font=font(ww*.38),fill=(31,31,22),anchor='mm')
            if n=='stand_board_narrow':
                d=ImageDraw.Draw(im);ww,hh=im.size;d.rectangle((ww*.12,hh*.03,ww*.88,hh*.28),fill=(235,205,45))
                d.polygon([(ww*.5,hh*.04),(ww*.85,hh*.27),(ww*.15,hh*.27)],outline=(32,31,21),width=5)
                d.line([(ww*.32,hh*.24),(ww*.43,hh*.15),(ww*.43,hh*.10)],fill=(32,31,21),width=6);d.line([(ww*.68,hh*.24),(ww*.57,hh*.15),(ww*.57,hh*.10)],fill=(32,31,21),width=6)
            A.add(n,meaning,m,im,refpart=ref,notes=['判読できた文字だけ描画。小さな発行者名・本文・電話番号の不明箇所は空白。','裏面の固定具・脚の奥行きは推定。'])
        addboard('post_board','事故多発・通行注意の白い単柱看板',.4,.8,2.2,'事故多発\n通行注意',vertical=True)
        addboard('post_board_narrow','幅員狭小の単柱注意板',.4,.8,2.,'この先\n幅員狭小',vertical=True)
        addboard('post_board_forest','森林整備の木枠・木柱2本の看板',1.5,1.2,.95,'森林整備を\n行っています',(203,189,131),(36,85,144),2,True,c='wood')
        addboard('post_board_wall_cafe','壁に付く横長の店看板（柱なし）',3.,.8,.4,'名水 きまぐれ喫茶',(237,236,219),(45,84,132),0,ref='post_board')
        addboard('post_board_bosco','BOSCOの木脚2本の緑の案内板',1.8,1.1,.85,'BOSCO\n50M 直進 ↑',(29,67,51),(237,230,171),2,c='wood')
        addboard('post_board_no_parking','ここは駐車場ではありませんの看板',.75,1.1,1.0,'ここは駐車場では\nありません\n迷惑駐車は\nお止めください',(232,232,217),(35,38,31),2,ref='small_plate')
        addboard('post_board_step','出入口段差ありの注意板',.45,.8,1.4,'出入口\n段差あり',vertical=True)
        addboard('post_board_camera','監視中の道路施設注意板',.6,.9,1.3,'駐車禁止\n監視中',(231,231,219),(174,40,33))
        addboard('stand_board','落石注意の金属自立看板',.6,1.,.75,'落石注意',vertical=True,legs=2,stand=True,ref='stand_board')
        addboard('stand_board_slow','蛍光黄の徐行してください看板',.4,1.2,.85,'徐行してください',(225,230,65),vertical=True,legs=2,stand=True,ref='stand_board')
        addboard('stand_board_narrow','幅員狭少の自立看板',.4,1.2,.85,'幅員狭少',vertical=True,legs=2,stand=True,ref='stand_board')
        addboard('stand_board_bridge_parking','橋上の駐車禁止看板',.7,1.4,1.05,'交通妨害となるので\n橋の上での駐車禁止',(234,234,220),vertical=True,legs=2,stand=True,ref='stand_board')
        addboard('stand_board_coffee','店頭の低い名水コーヒー看板',1.,.5,.55,'名水コーヒー',(237,235,225),(187,72,106),legs=2,stand=True,ref='stand_board')
        addboard('small_plate','支柱2本の白い駐車注意札',.5,1.1,.8,'ここは駐車場ではありません\n迷惑駐車はお止めください',(235,235,223),vertical=True,legs=2,ref='small_plate')
        addboard('small_plate_attention','ミラーの柱へ付く注意札（柱なし）',.16,.5,.25,'注意',(234,234,219),vertical=True,legs=0,ref='small_plate')
        addboard('small_plate_parking','木に付く黄色いP札（柱なし）',.4,.4,.2,'P',legs=0,ref='small_plate')
        addboard('small_plate_flick','木の右向きFLICK案内札（柱なし）',.5,.2,.1,'FLICK →',(95,72,46),(239,231,205),legs=0,ref='small_plate',c='wood')
        addboard('parking_clearance_notice','ガードレールの木支えに付く長方形の駐車注意板',.60,.90,.78,'この場所に駐車\nしますとすれちがい\nに支障があります\nので、駐車しないで\n下さい。',(222,220,195),(33,35,28),legs=2,ref='small_plate',c='wood')
        # 不明本文は紙の面だけ。架空の短文や疑似文字は描かない。
        m=Mesh();board(m,1.5,1.15,1.0,2,'wood',.045,frame=True);im=Image.new('RGBA',(900,690),(108,112,101,255));d=ImageDraw.Draw(im)
        for y in range(3):
            for x in range(4):d.rectangle((12+x*220,12+y*223,218+x*220,221+y*223),fill=(226,228,221),outline=(172,179,173),width=3)
        A.add('notice_board','木枠と複数の白い掲示紙。本文は未判読で空白。',m,im)
        addboard('notice_board_aoyamaso','壁付けの額入り青山荘案内板',.7,.9,.45,'青山荘',(231,220,190),(87,65,53),legs=0,frame=True,ref='notice_board')
        m=Mesh();m.tube('Γ型の太い柱',(0,0,0),(0,0,5.2),.14,'wood',12);m.tube('道路上への横腕',(0,0,5.15),(1.9,0,5.15),.11,'wood',10);m.box('LED情報板の筐体',(1.25,0,4.85),(1.5,.28,.52),'dark');m.box('表示面',(1.25,-.15,4.85),(1.4,.018,.42),'black');m.tube('赤色警告灯台',(1.25,0,5.12),(1.25,0,5.24),.07,'metal',10);m.sphere('赤色警告灯',(1.25,0,5.30),(.085,.085,.11),'red',10,4)
        A.add('notice_board_led','Γ柱・黒い電光掲示板と赤い警告灯',m,refpart='notice_board',notes=['表示文は判読できないため消灯の黒面。柱径と張出し長は推定。'])
        addboard('nobori','細い木柱2本に固定された縦長の黄色看板',.5,2.2,1.5,'この先 幅員狭小',vertical=True,legs=2,c='wood',ref='nobori')
        addboard('construction_board','青枠の工事看板・緑支柱2本',1.5,2.,1.15,'ご協力をお願いします\n土砂崩壊から道路を守る\n工事をしています\n令和7年8月29日まで\n時間 8:00-17:00\n道路災害防除工事',(232,232,215),(31,76,133),legs=2,c='green',ref='construction_board')
        addboard('construction_board_bridge','仮囲いへ付ける橋塗替え工事看板',1.,1.5,.75,'ご迷惑をおかけします\n橋の塗り替えをしています\n令和7年8月29日まで\n8:00〜17:00',(232,232,218),(34,77,132),legs=0,ref='construction_board')
        A.reclassified=['post_boardの店看板は壁付けで柱なし。森林板は木枠と2本柱。','stand_board/small_plate/notice_boardには形状差があるため名前を分離。','noboriは写真では布旗ではなく2本の竿に留めた縦長の板。','未判読の文章は空白。雨量規制板の区間名が不明なので全文画像は作らない。']

    elif asset=='r70_guardrail':
        def beam_piece(m,L,end=False):
            profile=[(.035,.50),(-.018,.53),(-.065,.60),(-.028,.665),(-.065,.73),(-.018,.80),(.035,.82)]
            stations=[(0,0),(L,0)] if not end else [(0,0),(.22,0),(.42,.045),(.58,.13),(.65,.23)]
            for sign in [0,.004]:
                vs=[(x,y+dy+sign,z) for x,dy in stations for y,z in profile];N=len(profile)
                m.surface('W波形ビームの表裏',vs,[(j*N+i,(j+1)*N+i,(j+1)*N+i+1,j*N+i+1) if not sign else (j*N+i,j*N+i+1,(j+1)*N+i+1,(j+1)*N+i) for j in range(len(stations)-1) for i in range(N-1)],'white')
            # 薄鋼板の上下折返し。
            for y,z in [profile[0],profile[-1]]:m.beam('端の折返し',(0,y,z),(stations[-1][0],y+stations[-1][1],z),.009,.014,'white')
            m.anchor=(0,.10,0)
            m.tube('丸鋼管支柱',(0,.10,0),(0,.10,.79),.057,'white',12)
            m.box('支柱とビームの間隔金具',(0,.025,.66),(.15,.10,.12),'metal')
            for z in [.575,.745]:m.tube('取付ボルト',(0,-.073,z),(0,-.083,z),.013,'metal',6)
            if end:
                m.box('袖端の赤い注意プレート',(.62,.20,.66),(.08,.02,.20),'red');m.sphere('端の白い反射球',(.08,.10,.86),(.035,.035,.035),'white',8,4)
        variants=lambda n:{f'{n}_cream.png':(None,{'white':PALETTE['cream']}),f'{n}_rust.png':(None,{'white':(145,113,79)})}
        for n,L,end in [('w_beam',4.,False),('w_beam_2m',2.,False),('w_beam_end',.65,True)]:
            m=Mesh();beam_piece(m,L,end);A.add(n,'W波形ガードレールの'+('外へ巻いた袖端' if end else f'{L:g}mスパン＋始端支柱1本'),m,variants=variants(n),span=None if end else L,refpart='w_beam',notes=['板幅0.32m、ビーム上端0.82m、柱径0.114mは写真目安。','標準原型4mは要件の1支柱1スパン。写真の約2m間隔用はw_beam_2m。袖端の赤札・球は代表例から近似。'])
        for n,mesh in [('pipe',False),('pipe_mesh',True)]:
            m=Mesh();m.box('角形支柱',(0,0,.52),(.10,.10,1.04),'white')
            for z in [.28,.62,1.]:m.tube('横管',(0,-.035,z),(2.,-.035,z),.035,'white',10)
            art=None
            if mesh:art=net_art(2.,.55,'white');m.panel('上半分の金網',1.,-.04,.73,2.,.52)
            v=variants(n);v[n+'_black.png']=(None,{'white':PALETTE['dark']})
            A.add(n,'3段ガードパイプ'+('＋菱形金網' if mesh else ''),m,art,v,span=2.,refpart='pipe')
        m=Mesh();m.tube('端部支柱',(0,0,0),(0,0,1.),.05,'white',10)
        for z in [.28,.62,1.]:m.tube('丸めた終端の短管',(0,0,z),(.30,.10,z),.035,'white',8)
        A.add('pipe_end','ガードパイプの短い端部',m,variants=variants('pipe_end'),refpart='pipe')
        A.reclassified=['pseudo_woodの提示写真は黄白の金属Wビーム。擬木原型は作らずw_beam_cream.pngへ統合。']
    elif asset=='r70_fence':
        for n,c,h,post in [('wire_mesh','green',2.,'tube'),('wire_mesh_concrete','metal',2.,'concrete'),('wire_mesh_orange','orange',1.2,'tube')]:
            m=Mesh();L=2.5
            if post=='concrete':m.box('コンクリート角柱',(0,0,h/2),(.12,.12,h),'concrete')
            else:m.tube('鋼管支柱',(0,0,0),(0,0,h+.18),.032,c,10)
            for z in [.15,h-.15]:m.tube('上下の支持横管',(0,0,z),(L,0,z),.022,c,8)
            m.panel('菱形の透過金網',L/2,-.006,h/2,L,h-.25)
            for zz in [h+.04,h+.15]:
                m.tube('上端の有刺線',(0,0,zz),(L,0,zz),.005,c,5)
                for xx in [.4,1.1,1.8]:m.beam('短い有刺の形',(xx,0,zz-.025),(xx+.035,0,zz+.025),.006,c=c)
            A.add(n,'菱形金網と始端支柱・上部有刺線',m,net_art(L,h-.25,c),span=L,refpart='wire_mesh',notes=['網目は約50mmの菱形を自作RGBAへ描画。材質はalpha test/両面描画を指定すること。柱間隔2.5m。'])
        m=Mesh();L=3.;h=3.0;slope=.45
        for z in [0,h]:m.tube('上下ワイヤー',(0,slope*z,z),(L,slope*z,z),.012,'metal',6)
        m.tube('法面の支柱',(0,0,0),(0,slope*h,h),.04,'metal',10)
        m.tube('斜め支持ワイヤー',(0,.01,0),(L,slope*h+.01,h),.01,'metal',6)
        vs=[(0,-.015,0),(L,-.015,0),(L,slope*h-.015,h),(0,slope*h-.015,h)];m.surface('斜面に張る透過網',vs,[(0,1,2,3)],uv=[(0,1),(1,1),(1,0),(0,0)])
        A.add('rockfall_net','斜面に張る菱形金網・支持ワイヤー・始端柱',m,net_art(L,h,'metal'),span=L,notes=['原型は高さ3m・後退1.35mの斜面。地形への変形が必要。写真の高さ3〜5mに合わせて配置側で調整。'])
        m=Mesh();L=2.;m.box('始端角柱',(0,0,.55),(.07,.07,1.1),'dark')
        for z in [.16,1.04]:m.box('上下の横桟',(L/2,0,z),(L,.045,.05),'dark')
        for x in np.arange(.15,L,.15):m.box('細い縦格子',(x,0,.60),(.018,.025,.85),'dark')
        A.add('steel_vertical','黒い縦格子の転落防止柵',m,span=L)
        m=Mesh();L=2.;m.tube('丸太支柱',(0,0,0),(0,0,1.05),.09,'wood',10)
        for z in [.40,.85]:m.tube('丸太横木',(0,-.015,z),(L,-.015,z),.065,'wood',10)
        A.add('wood','丸太の柱と横木2段の木柵',m,span=L,notes=['丸太径0.18/0.13m、支柱間隔2mは写真比率の推定。木目の微細凹凸を簡略化。'])
        m=Mesh();m.box('木杭',(0,0,.75),(.13,.13,1.5),'wood');m.panel('木杭間の網',1.,-.015,.75,2.,1.4)
        A.add('wood_mesh','木杭と落石止め金網',m,net_art(2.,1.4,'metal'),span=2.,refpart='wood')
        A.reclassified=['woodの横板土留めはr70_retaining_wall.wood_laggingへ統合。横管柵の例はguardrail.pipe。']
    elif asset=='r70_retaining_wall':
        for n in ['stone','block','shotcrete','concrete','slope_frame']:
            m=Mesh();surface=wall_art(n);slope=.4
            # 穴を含む前面2区画。長方形外周から内側の円孔へ16本の四辺形。
            for x0 in [0,2]:
                cx=x0+1;cz=.45;N=16;outer=[];inner=[]
                outline=[(x0,0),(x0+.5,0),(x0+1,0),(x0+1.5,0),(x0+2,0),(x0+2,.25),(x0+2,.5),(x0+2,.75),(x0+2,1),(x0+1.5,1),(x0+1,1),(x0+.5,1),(x0,1),(x0,.75),(x0,.5),(x0,.25)]
                for xx,zz in outline:
                    t=math.atan2(zz-cz,xx-cx);outer.append((xx,slope*zz,zz));inner.append((cx+.043*math.cos(t),slope*(cz+.043*math.sin(t)),cz+.043*math.sin(t)))
                vv=outer+inner;uv=[(x/4,1-z) for x,y,z in vv];m.surface('水抜き穴を避けた壁の面',vv,[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)],uv=uv)
                m.tube('水抜きの暗い奥',(cx,slope*cz+.035,cz),(cx,slope*cz+.20,cz),.042,'dark',16,caps=False)
                m.disc('水抜き穴の奥底',(cx,slope*cz+.205,cz),.042,.042,.005,'black',16)
            # 天端・裏・両側。壁全体は幅4m・高さ1m。正面に0.4m後退する勾配。
            vs=[(0,0,0),(4,0,0),(4,.4,1),(0,.4,1),(0,.34,0),(4,.34,0),(4,.74,1),(0,.74,1)]
            m.surface('壁の厚みと天端',vs,[(3,2,6,7),(4,5,1,0),(0,4,7,3),(1,5,6,2),(4,7,6,5)],'concrete')
            if n=='slope_frame':
                for x in [0,1,2,3]:m.beam('縦のコンクリート法枠',(x,-.05,0),(x,.35,1),.14,.16,'white')
                for z in [.08,.92]:m.beam('横のコンクリート法枠',(0,.4*z-.06,z),(4,.4*z-.06,z),.14,.16,'white')
                for x in [.05,1,2,3]:m.tube('交点のアンカー頭',(x,.4*.92-.16,.92),(x,.4*.92-.19,.92),.05,'metal',10)
            A.add(n,{'stone':'不揃いな自然石積み','block':'斜め谷積みの間知ブロック','shotcrete':'吹付けコンクリート法面','concrete':'雨だれ付きの平らなコンクリート壁','slope_frame':'突出するコンクリート格子法枠'}[n],m,surface,span=4.,notes=['基本壁は幅4m・高さ1m・法勾配1:0.4。高さの拡大で石/網目も伸びるため、表面UVを実寸へ補正するか1m区画を重ねる。','水抜き2穴と天端は形状。法枠は1m格子、突出0.16mを立体化。','石肌・苔・雨だれは写真を貼らず自作の周期模様。現場の石一個ずつの輪郭とは異なる。'])
            surface.resize((1024,1024)).save(A.out/(n+'_surface.png'));shutil.copy2(A.out/(n+'_surface.png'),A.src/(n+'_surface.png'))
        m=Mesh()
        # H鋼は3枚の鋼板で断面を作る。2m間隔で4mモジュール。
        for x in [0,2]:
            for y in [-.11,.11]:m.box('H鋼フランジ',(x,y,.5),(.28,.025,1.),'rust')
            m.box('H鋼ウェブ',(x,0,.5),(.025,.22,1.),'rust')
        for z in [.12,.37,.62,.87]:m.box('厚い横板',(2,.045,z),(4.,.15,.235),'wood')
        A.add('wood_lagging','H鋼の支柱と厚い横板の土留め壁',m,span=4.,notes=['4m幅・1m高。H鋼支柱は始端と中間の2m間隔。高さ方向へ増す場合は支柱を連続化する。'])
        im=wall_art('wood_lagging');im.save(A.out/'wood_lagging_surface.png');shutil.copy2(A.out/'wood_lagging_surface.png',A.src/'wood_lagging_surface.png')
        A.reclassified=['concrete参照の0979は石張りに見えるためblockも候補。0359〜0362の方形石はstone/block寄りで無地壁へ一律に割り当てない。','wood柵の横板土留めはwood_laggingと共用。']

    elif asset=='r70_drain':
        m=Mesh();m.box('U溝の底',(1.,0,.035),(2.,.40,.07),'concrete')
        for y in [-.165,.165]:m.box('U溝の両側壁',(1.,y,.20),(2.,.07,.26),'concrete')
        A.add('u_gutter','幅0.4mの開渠U字側溝・2m区切り',m,span=2.,notes=['底面がz=0、溝天端z=.33m。路面と天端を合わせる場合zを路面-.33mとする。土砂・落葉は別途配置。'])
        def grate(n,w,h,meaning,ref):
            m=Mesh()
            for yy in [-h/2,h/2]:m.box('長辺の枠',(0,yy,.025),(w+.04,.035,.05),'metal')
            for xx in [-w/2,w/2]:m.box('短辺の枠',(xx,0,.025),(.035,h,.05),'metal')
            count=int(w/.035)
            # 縦桟は厚み付き上面と側面のコの字断面。裏は省略し400tris内。
            for x in np.linspace(-w/2+.03,w/2-.03,count):
                m.surface('縦の格子',[(x-.004,-h/2,.05),(x+.004,-h/2,.05),(x+.004,h/2,.05),(x-.004,h/2,.05),(x-.004,-h/2,.013),(x-.004,h/2,.013)],[(0,1,2,3),(4,0,3,5)],'metal')
            for y in np.linspace(-h/2+.08,h/2-.08,3):m.surface('細い横格子',[(-w/2,y-.004,.052),(w/2,y-.004,.052),(w/2,y+.004,.052),(-w/2,y+.004,.052)],[(0,1,2,3)],'metal')
            A.add(n,meaning,m,refpart=ref,notes=['薄い蓋・格子のすき間は実形状。裏面・桟の片側を省略した低負荷形状。路面に置く際は上面を路面へ合わせる。'])
        grate('grating',1.,.6,'1.0×0.6mの長方形グレーチング','grating');grate('grating_cross',3.,.4,'道路横断用3.0×0.4mのグレーチング','grating');grate('catch_basin',.4,.4,'0.4m角の集水ますの格子蓋','catch_basin')
        m=Mesh();m.tube('鉄蓋の縁',(0,0,0),(0,0,.035),.35,'rust',40)
        vs=[(0,0,.0355)]+[(.335*math.cos(t),.335*math.sin(t),.0355) for t in np.arange(40)*2*math.pi/40];uv=[(.5,.5)]+[(.5+.5*math.cos(t),.5+.5*math.sin(t)) for t in np.arange(40)*2*math.pi/40];m.surface('鉄蓋上面',vs,[(0,i+1,(i+1)%40+1) for i in range(40)],uv=uv)
        im=Image.new('RGBA',(768,768),(104,75,52,255));d=ImageDraw.Draw(im)
        for r in [365,340,320]:d.ellipse((384-r,384-r,384+r,384+r),outline=(143,114,78),width=5)
        for y in range(70,700,26):
            for x in range(70,700,26):
                if (x-384)**2+(y-384)**2<310**2:d.line((x-6,y-6,x+6,y+6),fill=(155,120,81),width=5)
        for x in [210,558]:d.rectangle((x-24,370,x+24,395),fill=(43,35,28))
        A.add('manhole','直径0.7mの錆色鉄蓋',m,im,notes=['自治体の紋章・未判読文字は描かず、滑止めの幾何学模様だけを自作。'])
    elif asset=='r70_construction':
        def checker(w=768,h=768):
            im=Image.new('RGBA',(w,h),(235,232,205,255));d=ImageDraw.Draw(im)
            for y in range(3):
                for x in range(12):
                    if (x+y)%2==0:d.rectangle((x*w/12,y*h/3,(x+1)*w/12,(y+1)*h/3),fill=(194,44,38))
            return im
        def drum(n,h,r,ref):
            m=Mesh();N=24
            levels=[(0,r*.94),(.08,r),(.14,r),(h-.18,r),(h-.12,r*1.02),(h,r*1.02)]
            vs=[(rr*math.cos(t),rr*math.sin(t),z) for z,rr in levels for t in np.linspace(0,2*math.pi,N+1)];uv=[(i/N,1-z/h) for z,rr in levels for i in range(N+1)]
            for j in range(len(levels)-1):
                inds=[(j*(N+1)+i,j*(N+1)+i+1,(j+1)*(N+1)+i+1,(j+1)*(N+1)+i) for i in range(N)]
                if j==2:
                    local_uv=[(i/N,1. if k<=2 else 0.) for k in range(len(levels)) for i in range(N+1)]
                    m.surface('赤白の反射チェッカー胴',vs,inds,uv=local_uv)
                else:m.surface('黄色い胴の縁',vs,inds,'yellow')
            m.tube('黄色い上蓋',(0,0,h-.025),(0,0,h),r*1.025,'yellow',24);m.tube('注水口',(r*.35,0,h),(r*.35,0,h+.018),.055,'yellow',10)
            for x in [-r*.55,r*.55]:m.box('蓋の取っ手',(x,0,h+.018),(.10,.045,.035),'yellow')
            A.add(n,'黄色い蓋・赤白の市松反射を持つ円筒緩衝体',m,checker(),refpart=ref,notes=['一般的な円筒型の写真輪郭を近似。製品固有の刻印、落書き、連結穴の微細形状は省略。'])
        drum('drum',.9,.36,'drum');drum('water_barrier',.8,.34,'water_barrier')
        m=Mesh();m.box('四角いコーンの台',(0,0,.025),(.38,.38,.05),'dark')
        for za,zb,ra,rb,c in [(.05,.25,.16,.115,'red'),(.25,.36,.115,.087,'white'),(.36,.48,.087,.059,'red'),(.48,.57,.059,.035,'white'),(.57,.70,.035,.012,'red')]:m.tube('コーンの帯',(0,0,za),(0,0,zb),ra,c,16,r2=rb)
        A.add('cone','0.7m赤白カラーコーン',m,variants={'cone_green.png':(None,{'red':(39,116,71)}),'cone_orange.png':(None,{'red':PALETTE['orange']})})
        m=Mesh()
        for x in [-.50,.50]:
            for y in [-.32,.32]:m.beam('木のA脚',(x,y,0),(x,0,1.05),.07,c='wood')
        for z in [.58,.90]:m.box('木の横板',(0,-.04,z),(1.25,.06,.16),'wood')
        A.add('wood_A','古い木製A型バリケード',m,refpart='water_barrier')
        m=Mesh();m.sphere('詰まった大型土のう',(0,0,.38),(.52,.44,.38),'white',16,7)
        m.tube('袋の口元',(0,0,.68),(0,0,.84),.12,'white',10,r2=.07)
        for x in [-.28,.28]:
            for y in [-.22,.22]:m.beam('吊りベルト',(x,y,.52),(x*.85,y*.85,.80),.035,c='white')
        A.add('sandbag','吊りベルトと絞った口の大型土のう1袋',m,variants={'sandbag_pink.png':(None,{'white':PALETTE['pink']}),'sandbag_gray.png':(None,{'white':(159,161,149)})},notes=['単体袋。護岸の積み方・かご枠は配置側で再現。解像度が低く袋のしわ・結び方は推定。'])
        m=Mesh()
        for x in [-.8,.8]:
            for y in [-.30,.30]:m.beam('鋼管A脚',(x,y,0),(x,0,1.1),.035,c='metal')
        for z in [.55,.88]:m.box('反射横板',(0,-.02,z),(1.8,.035,.18),'white');m.panel('斜め反射塗り',0,-.039,z,1.78,.17)
        stripe=Image.new('RGBA',(768,180),(234,230,210,255));d=ImageDraw.Draw(stripe)
        for x in range(-180,900,150):d.polygon([(x,180),(x+90,180),(x+270,0),(x+180,0)],fill=(197,45,39))
        A.add('barricade','赤白反射板と金属A脚のバリケード',m,stripe,notes=['0856の水詰め型はwater_barrierを使用。これは0733の分岐入口用の鋼管A脚型。'])
        m=Mesh()
        for x in [-1.5,1.5]:m.tube('赤い支柱',(x,0,0),(x,0,1.),.035,'red',8)
        for i in range(10):m.tube('赤白交互の横管',(-1.5+i*.3,0,.9),(-1.2+i*.3,0,.9),.03,'red' if i%2 else 'white',8)
        A.add('barricade_pipe','脇道の入口を塞ぐ赤白の鋼管3m区間',m,refpart='barricade')
        m=Mesh();m.box('黒黄の低い防護板',(0,0,.3),(1.5,.03,.6),'yellow');m.panel('黒黄斜め縞',0,-.016,.3,1.48,.58)
        blackstripe=stripe.copy();arr=np.asarray(blackstripe).copy();mask=arr[:,:,0]>220;arr[mask,:3]=PALETTE['yellow'];arr[~mask,:3]=PALETTE['black'];A.add('barricade_panel','柵の足元に付く黒黄斜め縞の防護板',m,Image.fromarray(arr),refpart='barricade')
        A.reclassified=['water_barrier/drum/barricadeの樽型は同系列。0856はwater_barrier、1018はdrum。木製A型・鋼管型・板型を分離。']
    elif asset=='r70_gate':
        m=Mesh()
        for x in [-1.5,1.5]:
            m.box('低いコンクリート台',(x,0,.10),(.36,.40,.20),'concrete');m.tube('細い端の縦柱',(x,0,.20),(x,0,.83),.045,'metal',10)
        N=16;vs=[(x,.075*math.cos(t),.8+.075*math.sin(t)) for x in [-1.5,1.5] for t in np.linspace(0,2*math.pi,N+1)];uv=[(j, i/N) for j in [0,1] for i in range(N+1)]
        m.surface('斜めの赤白塗りの横棒',vs,[(i,i+N+1,i+N+2,i+1) for i in range(N)],uv=uv)
        m.surface('横棒の端面',vs,[tuple(range(N-1,-1,-1)),tuple(range(N+1,2*N+1))],'white')
        stripe=Image.new('RGBA',(1200,240),(236,233,215,255));dd=ImageDraw.Draw(stripe)
        for x in range(-100,1300,150):dd.polygon([(x,0),(x+75,0),(x+175,240),(x+100,240)],fill=(186,43,38))
        A.add('bollard','低いコンクリート台と長さ3mの白赤斜め縞車止めバー',m,stripe)
        m=Mesh();m.box('赤白のコンクリートブロック',(0,0,.35),(2.,.42,.70),'white')
        for x in [-.8,-.3,.2,.7]:m.box('赤い塗り分け',(x,-.213,.35),(.22,.006,.70),'red')
        A.add('bollard_block','赤白塗装のコンクリート車止め',m,refpart='bollard')
        m=Mesh()
        pts=[(-1.,0,0),(-1.,0,.65),(-.94,0,.8),(.94,0,.8),(1.,0,.65),(1.,0,0)]
        for i,(a,b) in enumerate(zip(pts,pts[1:])):m.tube('逆U字の鋼管',a,b,.045,'white' if i%2 else 'red',10)
        m.tube('下段横管',(-1.,0,.4),(1.,0,.4),.032,'white',8)
        A.add('bollard_u','白赤の逆U字車止めと下横管',m,variants={'bollard_u_rust.png':(None,{'white':PALETTE['rust'],'red':PALETTE['rust']})},refpart='bollard')
        m=Mesh();L=4.;h=1.5
        for x in [-L/2,L/2]:m.tube('門柱',(x,0,0),(x,0,h+.1),.045,'metal',10)
        for x0,x1 in [(-L/2,0),(0,L/2)]:
            for z in [.15,h]:m.tube('門扉の横枠',(x0,0,z),(x1,0,z),.025,'metal',8)
            for x in [x0,x1]:m.tube('門扉の縦枠',(x,0,.15),(x,0,h),.025,'metal',8)
            m.tube('門扉の斜材',(x0,0,.15),(x1,0,h),.017,'metal',6)
        m.panel('両開き門の金網',0,-.008,.82,L,1.30)
        A.add('gate','灰色の金網両開き門・4m幅',m,net_art(L,1.3,'metal'),notes=['門の奥の低解像度写真からの原型。留め金とヒンジの細部は推定。'])
        m=Mesh();w=1.5;h=2.5
        for x in [-w/2,w/2]:m.tube('緑の門柱',(x,0,0),(x,0,h),.035,'green',10)
        for z in [.12,h]:m.tube('扉枠の上下',(-w/2,0,z),(w/2,0,z),.028,'green',8)
        m.tube('斜めの補強',(-w/2,0,.12),(w/2,0,h),.02,'green',8);m.panel('緑色の透過網',0,-.008,h/2,w,h-.2)
        A.add('gate_green','緑色の片開き金網ゲート・1.5m幅',m,net_art(w,h-.2,'green'),refpart='gate')
        m=Mesh()
        for x in [-1.5,1.5]:
            for yy in [-.35,.35]:m.beam('黄色いX脚',(x,yy,0),(x,-yy,1.),.07,c='yellow')
        for z in [.35,.82]:m.box('黄色い長い横木',(0,0,z),(3.3,.07,.08),'yellow')
        A.add('gate_wood_x','広場入口の黄色いX脚木柵・原型3m',m,refpart='gate',notes=['写真の長さ約6mは原型を2区間並べる。X脚の開きは推定。'])
    elif asset=='r70_misc':
        m=Mesh();m.box('角形の標柱',(0,0,.37),(.11,.11,.74),'orange');m.box('上部の黄色い帽子',(0,0,.77),(.115,.115,.06),'yellow')
        A.add('stake','高さ0.8m角杭・黄帽子',m,variants={f'stake_{c}.png':(None,{'orange':PALETTE[c]}) for c in ['white','yellow','red']})
        m=Mesh();m.tube('細い無地の標柱',(0,0,0),(0,0,1.5),.025,'white',8);A.add('stake_thin','先端に物の付かない細い白標柱',m,variants={'stake_thin_orange.png':(None,{'white':PALETTE['orange']})},refpart='stake')
        def cabinet(n,w,h,d,doors=1):
            m=Mesh();m.box('基礎',(0,0,.06),(w+.08,d+.08,.12),'concrete');m.box('機器筐体',(0,0,h/2+.12),(w,d,h),'metal');m.box('薄い庇',(0,-.015,h+.14),(w+.045,d+.05,.035),'metal')
            for i in range(doors):
                x=-w/2+(i+.5)*w/doors;m.box('扉の面',(x,-d/2-.008,h/2+.12),(w/doors-.015,.015,h-.04),'concrete');m.box('扉の取手',(x+w/doors*.32,-d/2-.024,h*.55),(.018,.02,.095),'dark')
                for z in [.26,.31,.36,.41]:m.box('通気スリット',(x,-d/2-.017,z),(w/doors*.58,.006,.008),'dark')
            A.add(n,'灰色の機器箱'+str(doors)+'枚扉',m,refpart='equipment_box',notes=['扉の分割は写真の見える範囲から、鍵・換気口・奥行きは推定。'])
        cabinet('equipment_box',1.4,1.2,.45,3);cabinet('equipment_box_small',.75,1.1,.45)
        m=Mesh();m.box('柱に付く小型機器箱',(0,0,.15),(.3,.22,.3),'metal');m.box('小型箱の扉',(0,-.115,.15),(.275,.014,.275),'concrete');A.add('equipment_box_pole','電柱へ付く0.3mの小型箱（柱なし）',m,refpart='equipment_box',origin='箱の下辺中央。任意の柱へ取り付ける。')
        m=Mesh();rng=np.random.default_rng(12)
        for row,num in [(0,5),(1,4),(2,3)]:
            for j in range(num):
                y=(j-(num-1)/2)*.29;z=.16+row*.25;L=3.2+rng.uniform(-.3,.3);m.tube('積まれた丸太',(-L/2,y,z),(L/2,y,z),.145,'wood',10);m.tube('明るい切口',(-L/2-.003,y,z),(-L/2-.006,y,z),.137,'cream',10)
        A.add('log_pile','杉丸太12本の小山・長さ約3.5m',m,notes=['写真全体の置場約8mは原型を並べる。木材の本数と各丸太の位置は推定。'])
        m=Mesh();rng=np.random.default_rng(3)
        for i in range(22):
            x,y=rng.uniform(-1,1,2);z=rng.uniform(.04,.6);ang=rng.uniform(-.8,.8);start=len(m.v);m.box('廃材の板',(0,0,z),(rng.uniform(.6,1.5),.11,.055),'wood');m.transform(start,ang,(x,y,0))
        A.add('log_pile_scrap','伐採材・板材の小山',m,refpart='log_pile',notes=['個々の廃材の形・積順は写真から確定できず推定。'])
        m=Mesh();m.tube('切株状の丸太腰掛け',(0,0,0),(0,0,.3),.25,'wood',16);m.tube('切口',(0,0,.30),(0,0,.304),.245,'cream',16);A.add('log_stool','高さ0.3m・直径0.5mの切株状腰掛け1個',m,refpart='log_pile',notes=['参照では石か木か不確か。木製とした推定型。'])
        for n,num,w,c in [('stone_steps',8,1.5,'concrete'),('concrete_steps',5,1.,'concrete')]:
            m=Mesh()
            for i in range(num):m.box('階段の踏面と蹴上げ',(0,(i+.5)*.30,(i+1)*.16/2),(w,.30,(i+1)*.16),c)
            A.add(n,f'幅{w:g}m・{num}段の階段',m,refpart='stone_steps',notes=['正面は最下段へ向く-Y。踏面0.30m/蹴上げ0.16mは写真比率の推定。石段の苔・石目の細部は簡略化。'])
        m=Mesh()
        for y in [0,1.2,2.4]:m.tube('階段脇の支柱',(0,y,y*.533),(0,y,y*.533+.86),.025,'metal',8)
        m.tube('斜めの上手すり',(0,-.1,.81),(0,2.5,2.20),.025,'metal',8);A.add('stairs_rail','階段脇の勾配手すり1区間',m,refpart='stone_steps',notes=['手すり延長約8mは区間を(0,+2.4,+1.28)mずつ連結。端部と勾配の細部は推定。'])
        m=Mesh();m.box('赤い筐体',(0,0,.94),(1.,.72,1.72),'red')
        for x in [-.36,.36]:m.box('黒い脚',(x,0,.07),(.13,.58,.14),'dark')
        m.box('商品窓の奥',( -.12,-.367,1.25),(.69,.025,.92),'dark')
        m.box('決済部',( .36,-.373,.98),(.20,.035,.65),'white');m.box('取出口の凹み',(0,-.376,.40),(.76,.045,.20),'black')
        m.box('取出口のふた',(0,-.400,.43),(.69,.015,.13),'dark')
        for z in [.98,1.24,1.50]:
            m.box('白い商品棚',(-.12,-.39,z-.13),(.67,.035,.035),'white')
            for j in range(6):
                x=-.40+j*.11;m.tube('飲料の見本',(x,-.385,z-.09),(x,-.385,z+.06),.029,['red','white','orange','green','blue','yellow'][j],8);m.sphere('飲料の肩',(x,-.385,z+.06),(.029,.029,.015),'white',8,3)
        for z in [1.07,1.0,.92]:m.box('小さな決済スロット',(.36,-.394,z),(.12,.009,.024),'dark')
        m.panel('下部ブランド表示',-.17,-.369,.67,.56,.18)
        m.surface('側面の縦ブランド面',[(.501,-.28,.65),(.501,.28,.65),(.501,.28,1.65),(.501,-.28,1.65)],[(0,1,2,3)],uv=[(0,1),(1,1),(1,0),(0,0)],rect=(8,784,312,1012))
        im=lettering('Coca-Cola',900,260,(184,31,31),(244,240,222));p=A.add('vending','北原手前の赤い飲料自販機・商品窓と決済部を立体化',m,im,notes=['pass_vendingは変更せず新規制作。商品ラベルとロゴ書体は近似。銘柄の細字・商品価格は未判読のため省略。'])
        for folder in [A.src,A.out]:
            pic=Image.open(folder/'vending.png');pic.paste(im.rotate(90,expand=True).resize((304,228)),(8,784));pic.save(folder/'vending.png')
        # 灯器は横型。左青(緑)・中黄・右赤、庇付き。
        m=Mesh();m.tube('信号の鋼管柱',(0,0,0),(0,0,5.5),.09,'metal',12);m.tube('張出し横腕',(0,0,5.1),(2.0,0,5.1),.055,'metal',10)
        m.box('横型信号筐体',(1.5,-.05,4.85),(1.15,.24,.36),'white')
        for i,c in enumerate(['green','yellow','red']):
            x=1.12+i*.38;m.disc('灯の黒い縁',(x,-.183,4.85),.145,.145,.03,'dark',20);m.disc('3色のレンズ',(x,-.203,4.85),.12,.12,.012,c,20)
            ts=np.linspace(0,math.pi,11);vs=[(x+.15*math.cos(t),yy,4.85+.15*math.sin(t)) for yy in [-.19,-.36] for t in ts];m.surface('上半分の信号庇',vs,[(j,j+1,j+12,j+11) for j in range(10)],'dark')
        # 写真では白地・青字。依頼の「青地」の記述と異なることを記録。
        m.box('交差点名板',(1.5,-.05,5.34),(1.2,.035,.40),'white');m.panel('宮ヶ瀬北原の印刷面',1.5,-.069,5.34,1.17,.37)
        im=Image.new('RGBA',(1200,400),(237,239,225,255));dd=ImageDraw.Draw(im);dd.rectangle((5,5,1194,394),outline=(32,78,137),width=8);dd.text((600,140),'宮ヶ瀬北原',font=font(148),fill=(32,78,137),anchor='mm');dd.text((600,308),'Miyagase-Kitahara',font=font(60),fill=(32,78,137),anchor='mm')
        A.add('signal','横型3灯信号と宮ヶ瀬北原の交差点名板',m,im,notes=['写真では交差点名板は白地に青字。依頼の青地という記述より写真を優先した。','柱高さ・腕長・裏面金具は推定。静的3色レンズ、交通制御・発光・配線は含まない。'])
    else:raise ValueError(A.id)

    return A.save()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--asset',choices=ASSETS);args=ap.parse_args()
    for asset in ([args.asset] if args.asset else ASSETS):
        info=build(asset);print(asset,len(info['parts']),sum(p['triangles'] for p in info['parts']),flush=True)
