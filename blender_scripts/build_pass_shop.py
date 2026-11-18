"""依頼07 売店と自販機。texturesは通常Python、buildはBlenderで実行。"""
from pathlib import Path
import sys, math, random
ROOT=Path(__file__).resolve().parents[1]

def textures():
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    rng=np.random.default_rng(71031)
    font='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'
    def f(n):return ImageFont.truetype(font,n)
    for a in ['pass_shop','pass_vending']:(ROOT/'assets/landmarks'/a).mkdir(parents=True,exist_ok=True)
    w,h=2048,360
    x=np.arange(w)[None,:,None];y=np.arange(h)[:,None,None]
    noise=rng.normal(0,4,(h,w,1))+8*np.sin(x*.105)+4*np.sin(x*.91)+9*np.sin(x*.007+y*.01)
    arr=np.clip(np.array([133,121,105])+noise,0,255).astype('uint8')
    im=Image.fromarray(arr);d=ImageDraw.Draw(im)
    d.text((82,77),'国民宿舎',font=f(54),fill=(236,235,221));d.text((50,158),'丹沢ホーム',font=f(63),fill=(237,236,225))
    d.text((510,53),'ヤビツ峠売店',font=f(166),fill=(49,43,38),stroke_width=2)
    d.text((505,46),'ヤビツ峠売店',font=f(166),fill=(243,242,226))
    im.save(ROOT/'assets/landmarks/pass_shop/shop_sign_art.png')
    im=Image.new('RGB',(768,1024),(205,202,173));d=ImageDraw.Draw(im)
    for j in range(0,1024,12):
        d.line((0,j,768,j),fill=(142,140,124),width=2);d.line((0,j+3,768,j+3),fill=(224,221,193),width=2)
    # 実際の落書きの文字は判読しない。黒い曲線の密度/位置だけを近似。
    rr=random.Random(71)
    for k in range(31):
        cx=rr.randrange(60,720);cy=rr.randrange(330,920);sx=rr.randrange(15,110);sy=rr.randrange(30,155)
        d.arc((cx-sx,cy-sy,cx+sx,cy+sy),rr.randrange(0,70),rr.randrange(200,360),fill=(43,46,40),width=rr.randrange(3,7))
        if k%3==0:d.line((cx,cy-sy,cx-35,cy+sy),fill=(43,46,40),width=5)
    im.save(ROOT/'assets/landmarks/pass_shop/shutter_art.png')
    im=Image.new('RGB',(1024,900),(52,57,51));d=ImageDraw.Draw(im)
    for x in [5,505,1008]:d.rectangle((x,0,x+12,900),fill=(172,173,151))
    for j,(x,y,w,h) in enumerate([(55,75,150,260),(245,45,155,290),(560,110,135,155),(715,40,170,220),(710,400,235,160),(120,620,270,150)]):
        d.rectangle((x,y,x+w,y+h),fill=(210-j*5,209-j*4,187-j*5))
        for yy in range(y+25,y+h-10,22):d.line((x+10,yy,x+w-12,yy),fill=(90,94,84),width=3)
    d.rectangle((0,870,1024,900),fill=(150,139,112));im.save(ROOT/'assets/landmarks/pass_shop/window_art.png')
    for typ in ['a','b']:
        im=Image.new('RGB',(768,1408),(195,27,29));d=ImageDraw.Draw(im)
        d.rounded_rectangle((30,48,620,760),radius=12,fill=(213,223,222),outline=(151,153,147),width=8)
        for row in range(3):
            for col in range(9):
                bx=52+col*61;by=78+row*211
                colors=[(220,56,45),(230,227,217),(47,114,184),(242,173,42),(63,143,99)]
                c=colors[(col+row+(typ=='b'))%len(colors)]
                d.rounded_rectangle((bx+6,by+12,bx+40,by+145),radius=7,fill=c,outline=(130,140,138),width=2)
                d.rectangle((bx+8,by+57,bx+38,by+90),fill=(225,230,216))
                d.rectangle((bx+8,by+3,bx+38,by+15),fill=(157,164,162))
                d.rectangle((bx-2,by+157,bx+49,by+182),fill=(77,103,127))
                d.rounded_rectangle((bx+4,by+188,bx+44,by+202),radius=5,fill=(193,205,166))
        d.rectangle((643,548,735,588),fill=(31,35,33));d.ellipse((662,618,704,660),fill=(142,149,148));d.rectangle((643,695,736,760),fill=(216,213,195))
        d.ellipse((315,811,551,962),fill=(234,229,212));d.polygon([(30,1080),(735,1050),(735,1380),(30,1380)],fill=(220,221,210))
        d.rounded_rectangle((93,1120,688,1255),radius=12,fill=(45,49,46),outline=(154,160,151),width=9)
        d.rectangle((120,1173,650,1204),fill=(104,112,103));d.rectangle((70,1306,212,1330),fill=(51,106,60))
        d.polygon([(0,910),(90,876),(75,986),(0,1075)],fill=(236,233,219))
        im.save(ROOT/'assets/landmarks/pass_vending'/('vending_'+typ+'_art.png'))

def build(asset):
    import bpy
    import pass_common as C
    parts={};pi={}
    wood=C.proc('風雨で褪せた杉板',(111,91,71),'wood',4)
    pale=C.proc('看板支柱の灰白木',(145,136,114),'wood',5)
    dark=C.proc('暗い木の奥壁',(66,53,42),'wood',4)
    roof=C.proc('錆赤褐色の金属屋根',(111,70,55),scale=9)
    trim=C.proc('濃茶色の鼻隠し',(83,53,42),'wood',5)
    cream=C.proc('退色した白塗装',(214,208,182),scale=11)
    stone=C.proc('野面石',(124,123,104),scale=16)
    concrete=C.proc('石段の上のコンクリート',(151,148,125),scale=14)
    def start(name,meaning):
        col=C.collection(name);parts[name]=[];pi[name]={'meaning':meaning,'front':'-Y'};return col
    def end(name,col):parts[name]=list(col.objects)
    if asset=='pass_shop':
        # 航空写真の主棟+右下屋。寸法は計測JSONを最終記録へ反映。
        W,D=5.1,4.7;z=.62;front=-D/2
        col=start('shop_body','主棟の板壁・前柱・梁・引っ込んだ店先。自販機は含まない。')
        C.box('奥の壁',(0,.1,z+1.65),(W,.16,3.3),dark)
        C.box('背面板壁',(0,D/2-.07,z+1.80),(W,.14,3.60),wood)
        for x in [-W/2+.06,W/2-.06]:
            ring=[(-2.35,z),(2.35,z),(2.35,z+3.70),(-2.35,z+4.67)]
            vs=[(x+dx,yy,zz) for dx in [-.06,.06] for yy,zz in ring]
            C.mesh('主棟の勾配に沿う側壁',vs,[(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],wood)
        for x in [-W/2+.10,0,W/2-.10]:C.box('正面の高い木柱',(x,front+.15,z+2.3),(.15,.20,4.6),pale)
        C.box('正面の横梁',(0,front+.17,z+3.46),(W,.21,.20),roof)
        C.box('シャッターの巻上箱',(-1.19,-.12,z+2.36),(1.52,.28,.30),cream)
        C.box('窓上の下がり壁',(.61,-.07,z+2.28),(1.85,.20,.56),wood)
        C.box('窓下の腰壁',(.63,-.07,z+.43),(1.89,.18,.86),wood)
        # 板張りの継ぎ目を立体化。
        for xx in [i*.17-W/2+.05 for i in range(30)]:
            C.box('正面奥の縦板継ぎ',(xx,-.105,z+1.65),(.012,.018,3.22),wood)
        for yy in [i*.18-D/2+.15 for i in range(25)]:
            C.box('側壁縦板継ぎ',(-W/2-.008,yy,z+1.7),(.014,.012,3.37),dark)
        C.front_panel('シャッターの自作塗装',(-1.19,-.225,z+1.10),1.43,2.15,C.image_material('自作シャッター画像',ROOT/'assets/landmarks/pass_shop/shutter_art.png'))
        C.front_panel('窓ガラスと貼紙',(.62,-.215,z+1.27),1.74,1.36,C.image_material('自作窓画像',ROOT/'assets/landmarks/pass_shop/window_art.png'))
        for xx in [-.27,.60,1.49]:C.box('窓枠縦',(xx,-.235,z+1.26),(.045,.07,1.44),cream)
        for zz in [.54,1.99]:C.box('窓枠横',(.61,-.235,z+zz),(1.83,.07,.045),cream)
        C.box('右の入口ドア',(1.99,-.11,z+1.07),(.70,.12,2.14),dark)
        for xx in [1.60,2.38]:C.box('入口の白枠',(xx,-.20,z+1.13),(.085,.17,2.26),cream)
        C.box('入口の庇',(1.99,-.36,z+2.30),(.96,.68,.16),cream)
        C.box('窓台',(.62,-.33,z+.54),(1.92,.36,.075),pale)
        # 高い前面から奥へ落ちる片流れ。
        C.beam('主屋根',(0,front-.04,z+4.67),(0,D/2+.20,z+3.63),W+.30,roof,depth=.105)
        # beamの幅方向を屋根板に使うと長軸Zになる。屋根は明示4隅の厚板。
        bpy.data.objects.remove(bpy.data.objects['主屋根'],do_unlink=True)
        v=[(-2.75,-2.5,z+4.7),(2.75,-2.5,z+4.7),(2.75,2.60,z+3.65),(-2.75,2.60,z+3.65)]
        C.mesh('片流れ金属屋根',v+[(x,y,zz-.10) for x,y,zz in v],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],roof)
        C.box('看板下の深い軒天',(0,-1.48,z+4.43),(W,1.58,.09),dark).rotation_euler[0]=math.radians(-12)
        for xx in [-2.5,-1.25,0,1.25,2.5]:C.beam('軒裏の垂木',(xx,-2.35,z+4.52),(xx,-.5,z+4.15),.065,wood)
        for yy in [-2.48,2.58]:C.box('屋根端の細い鼻隠し',(0,yy,z+(4.71 if yy<0 else 3.66)),(5.56,.075,.13),trim)
        end('shop_body',col)
        col=start('shop_sign','正面上部の白い看板文字。国民宿舎・丹沢ホーム・ヤビツ峠売店を自作画像化。')
        C.box('看板の板',(0,-2.50,z+4.32),(5.28,.095,.75),pale)
        C.front_panel('読めた看板文字',(0,-2.551,z+4.32),5.22,.72,C.image_material('自作売店題字',ROOT/'assets/landmarks/pass_shop/shop_sign_art.png'))
        end('shop_sign',col)
        col=start('shop_annex','主棟右の低い下屋・自販機囲い・屋根。')
        x=3.76
        C.box('下屋背面',(x,2.15,z+1.36),(2.57,.15,2.72),wood)
        C.box('下屋右側壁',(5.02,.15,z+1.36),(.12,4.05,2.72),wood)
        C.box('自販機の左袖壁',(2.64,-1.38,z+1.40),(.13,1.30,2.80),wood)
        for xx in [2.67,4.95]:C.box('自販機まわりの白い枠',(xx,-2.01,z+1.20),(.075,.14,2.40),cream)
        C.box('自販機上の白い庇',(3.81,-2.13,z+2.49),(2.44,.49,.18),cream)
        # 片流れ屋根と自販機の庇の間は、写真で見える茶色い妻板で閉じる。
        # 前後両方を閉じ、側面図で空が抜ける隙間を残さない。
        for yy,low in [(-2.18,2.58),(2.20,2.60)]:
            outline=[(2.50,z+low),(5.25,z+low),(5.25,z+2.67),(2.50,z+3.33)]
            vv=[(xx,yy+dy,zz) for dy in [-.04,.04] for xx,zz in outline]
            C.mesh('下屋の勾配に沿う前後の妻板',vv,[(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],wood)
        # 主棟から右へ落ちる下屋。
        vs=[(2.50,-2.27,z+3.33),(5.25,-2.27,z+2.67),(5.25,2.43,z+2.67),(2.50,2.43,z+3.33)]
        C.mesh('下屋の片流れ屋根',vs+[(a,b,c-.10) for a,b,c in vs],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],roof)
        C.beam('下屋正面の斜め鼻隠し',(2.48,-2.29,z+3.32),(5.25,-2.29,z+2.66),.10,trim)
        for xx in [2.78,3.05,3.32,3.59,3.86,4.13,4.40,4.67,4.94]:C.box('下屋の縦板継ぎ',(xx,2.065,z+1.35),(.014,.018,2.68),dark)
        end('shop_annex',col)
        col=start('shop_stone_base','売店の足元だけの野面石台。周辺の擁壁は含めない。')
        C.box('台の内部',(1.21,-.04,.27),(8.02,5.43,.54),concrete)
        rr=random.Random(77)
        for row in range(3):
            for i in range(27):
                xx=-2.67+i*.294+(.12 if row%2 else 0)
                if -1.45<xx<1.55:continue
                bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1,radius=1,location=(xx,-2.78,.105+row*.16))
                o=C.move(bpy.context.object);o.name='台の不揃いな野面石';o.scale=(rr.uniform(.145,.19),.12,rr.uniform(.10,.13));o.data.materials.append(stone)
        C.box('石台の天端',(1.21,-.04,.585),(8.14,5.48,.07),concrete)
        end('shop_stone_base',col)
        col=start('shop_steps','道路側の石段3段。台とは独立した部品。')
        for i in range(3):
            C.box('石段'+str(i+1),(.0,-3.56+i*.32,(i+1)*.185/2),(2.8,.98-i*.32,(i+1)*.185),stone,.025)
        end('shop_steps',col)
        for obs in parts.values():
            for o in obs:
                o.location.x*=1.0625;o.location.y*=.9
                o.scale.x*=1.0625;o.scale.y*=.9
        return {'parts':parts,'part_info':pi,'generator':'blender_scripts/build_pass_shop.py','resolution':2048,'front_azimuth_deg':59,'roof_footprint_m':[8.5,4.59],'refs':['codex07/sv_shop_225.png','codex07/sv_shop_270.png','codex07/sv_mid_180.png','codex07/sv_lotn_180.png'],'dimension_basis':'航空写真計測と写真のシャッター/自販機の比率。詳細はwork07/building_measurements.jsonと納品報告。','approximations':['背面は植生に隠れるため推定。壁の厚さ・内部・屋根裏は推定。','シャッター落書きは文字未判読のため黒い曲線で形と密度を近似。窓の貼紙は色面と線。','下屋の奥の壁と台奥側の形状は推定。'],'reference_views':{'reference_shop225':(-.8,-2,.45),'reference_shop270':(1,-2,.45)}}
    if asset=='pass_vending':
        red=C.proc('赤い焼付塗装',(190,28,29),scale=18);metal=C.proc('機器側面',(132,137,129),scale=12)
        for k in ['a','b']:
            name='vending_'+k;col=start(name,'赤い自販機'+k+'。商品列・選択ボタン・硬貨部・取出口を自作画像で再現。')
            C.box('自販機筐体',(0,0,.95),(.99,.78,1.80),red,.025)
            C.box('前扉',(0,-.402,.95),(1.01,.055,1.80),red,.016)
            C.front_panel('自作商品パネル',(0,-.434,.95),.97,1.77,C.image_material('自作商品列'+k,ROOT/'assets/landmarks/pass_vending'/(name+'_art.png')))
            for xx in [-.35,.35]:
                for yy in [-.26,.26]:C.box('調整脚',(xx,yy,.035),(.11,.11,.07),metal)
            C.box('背面の排熱部',(0,.398,.73),(.68,.025,.81),metal)
            for zz in [.42,.50,.58,.66,.74,.82,.90, .98,1.06]:C.box('背面通気スリット',(0,.414,zz),(.61,.009,.018),dark)
            end(name,col)
        return {'parts':parts,'part_info':pi,'generator':'blender_scripts/build_pass_shop.py','refs':['codex07/sv_shop_225.png','codex07/sv_shop_270.png','codex07/sv_u1_135.png'],'preview_offsets':{'vending_a':[-.55,0,0],'vending_b':[.55,0,0]},'dimension_basis':'一般的な飲料機の高さ1.85mを仮定した写真比率。現地実測ではない。','approximations':['右の2台目は正面が隠れるため商品列と操作部を左機と同構成に推定。','商品ラベル・商標の細文字は再現せず、自作の色面で近似。'],'reference_views':{'reference_shop270':(1,-2,.45)}}
    raise ValueError(asset)

if __name__=='__main__':
    if 'bpy' not in sys.modules:
        textures()
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        import pass_common as C
        arg=sys.argv[sys.argv.index('--')+1:][0]
        C.init_scene();C.save_asset(arg,build(arg))
