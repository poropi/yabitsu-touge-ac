"""依頼07 小物。textures()は通常Python、build()はBlender内で実行する。"""
from pathlib import Path
import sys, math
ROOT=Path(__file__).resolve().parents[1]
ASSETS=['pass_info_boards','pass_trail_stairs','pass_benches','pass_trail_posts','pass_bike_rack','pass_bus_stop','pass_brown_signs','pass_road_items','pass_utility_pole','pass_barriers','pass_site_equipment']

def textures():
    from PIL import Image,ImageDraw,ImageFont
    import random
    font='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'
    def art(asset,name,bg,words=(),lines=0,vertical=False):
        folder=ROOT/'assets/landmarks'/asset;folder.mkdir(parents=True,exist_ok=True)
        sizes={'trail_direction':(768,160),'south_wide':(768,230),'lotn_small':(768,170),'u1_upper':(1024,280),'u1_lower':(1024,220)}
        iw,ih=sizes.get(name,(768,768))
        im=Image.new('RGB',(iw,ih),bg);d=ImageDraw.Draw(im);rng=random.Random(name)
        # 読めない細字を文字に置換せず、短い水平線と薄い色面で表示。
        for i in range(lines):
            y=int(ih*(.61 if words else .20))+i*max(9,int(ih*(.31 if words else .62)/max(1,lines)));x=55
            while x<iw-68:
                width=rng.randint(22,90);d.line((x,y,min(iw-58,x+width),y),fill=(224,223,206) if sum(bg)<400 else (122,124,112),width=7 if sum(bg)<400 else 3);x+=width+12
        for j,word in enumerate(words):
            if vertical:
                size=min(100,600//max(1,len(word)));f=ImageFont.truetype(font,size)
                x=515-j*240
                for i,ch in enumerate(word):d.text((x,50+i*size),ch,font=f,fill=(97,68,41))
            else:
                size=min(88,int((iw-100)/max(1,len(word))));f=ImageFont.truetype(font,size)
                d.text((45,25+j*int(ih*.42)),word,font=f,fill=(234,232,215) if sum(bg)<400 else (48,52,42))
        if name=='large_white':
            d.rectangle((160,250,580,450),outline=(184,188,178),width=3)
            d.line((180,430,290,310,410,380,550,280),fill=(186,188,178),width=3)
            d.text((470,670),'神奈川県',font=ImageFont.truetype(font,48),fill=(42,58,96))
        if name=='parking_notice':d.rectangle((22,20,746,68),fill=(53,75,110))
        if name=='bus_timetable':
            d.rectangle((55,200,715,680),fill=(240,240,227));d.rectangle((35,705,735,721),fill=(183,95,113))
        if name=='south_tall':
            d.polygon([(145,72),(290,72),(290,110),(217,110),(319,212),(286,245),(184,143),(184,216),(145,216)],fill=(231,231,210))
        if name=='rack_label':
            im=Image.new('RGB',(1024,128),(165,160,139));d=ImageDraw.Draw(im)
            d.text((40,20),'CYCLE RACK',font=ImageFont.truetype(font,65),fill=(66,67,53))
        im.save(folder/(name+'_art.png'))
    art('pass_info_boards','large_white',(220,222,211),lines=18)
    art('pass_info_boards','small_weathered',(153,156,137),lines=5)
    art('pass_info_boards','parking_notice',(227,228,216),lines=18)
    art('pass_trail_posts','trail_direction',(67,53,40),lines=3)
    art('pass_trail_posts','green_notice',(43,103,80),lines=7)
    art('pass_trail_posts','blue_notice',(162,187,179),lines=5)
    art('pass_trail_posts','white_post',(206,200,170))
    art('pass_bike_rack','rack_label',(165,160,139))
    art('pass_bus_stop','bus_timetable',(224,218,181),['ヤビツ峠'],lines=26)
    art('pass_brown_signs','south_tall',(63,46,39),['','ヤビツ峠','レストハウス'],lines=0)
    art('pass_brown_signs','south_wide',(67,48,41),['ハイキングコース'],lines=3)
    art('pass_brown_signs','lotn_small',(130,107,70),lines=2)
    art('pass_brown_signs','u1_upper',(45,35,29),['三ノ塔・塔ノ岳','Mt.Tounodake'])
    art('pass_brown_signs','u1_lower',(45,35,29),lines=2)
    art('pass_road_items','camera_pole',(227,222,199),['監視カメラ','作動中'],vertical=True)
    art('pass_road_items','camera_ground',(227,222,199),['カメラ','作動中'],vertical=True)
    art('pass_site_equipment','island_notice',(78,67,49),lines=4)
    art('pass_site_equipment','yellow_notice',(205,166,57),lines=7)
    art('pass_site_equipment','roof_notice',(210,201,161),lines=9)
    art('pass_barriers','small_tags',(213,211,184),lines=3)
    import math
    im=Image.new('RGB',(512,512),(91,81,67));d=ImageDraw.Draw(im)
    for r in [235,224,210,90,75]:d.ellipse((256-r,256-r,256+r,256+r),outline=(125,112,90),width=3)
    for y in range(35,480,19):
        for x in range(35,480,19):
            if (x-256)**2+(y-256)**2<205**2 and (x-256)**2+(y-256)**2>95**2:
                d.polygon([(x,y-5),(x+5,y),(x,y+5),(x-5,y)],outline=(133,119,97))
    im.save(ROOT/'assets/landmarks/pass_road_items/manhole_top_art.png')

def build(asset):
    import bpy
    import pass_common as C
    parts={};pi={};offset={};refs=[];notes=[]
    wood=C.proc('褪色した木',(114,100,76),'wood',5)
    pale=C.proc('灰白の木',(165,156,129),'wood',4)
    darkwood=C.proc('暗褐色の木',(65,54,43),'wood',5)
    metal=C.proc('銀灰金属',(157,163,153),scale=17)
    black=C.proc('黒灰金属',(42,49,47),scale=12)
    white=C.proc('白い塗装',(218,218,194),scale=11)
    concrete=C.proc('灰色コンクリート',(137,140,123),scale=12)
    rust=C.proc('錆褐色',(119,79,48),scale=15)
    soil=C.proc('踏面の土',(101,91,69),scale=21)
    def start(n,meaning):
        col=C.collection(n);pi[n]={'meaning':meaning,'front':'-Y'};return col
    def end(n,col):
        parts[n]=list(col.objects);i=len(parts)-1;offset[n]=[(i%3)*2.8,(i//3)*2.2,0]
    def tex(n):return C.image_material(n,ROOT/'assets/landmarks'/asset/(n+'_art.png'))
    def panel(n,x,z,w,h,mat,depth=.035,y=0):
        C.box(n,(x,y,z),(w,depth,h),mat)
    def art(n,x,z,w,h,y=-.021):C.front_panel(n+' 自作表示',(x,y,z),w,h,tex(n))
    def prism(n,outline,depth,mat):
        # XZ外周を奥行き方向に押し出す。
        N=len(outline);v=[(x,y,z) for y in [-depth/2,depth/2] for x,z in outline]
        return C.mesh(n,v,[tuple(range(N-1,-1,-1)),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)],mat)
    def post(n,x,y,h,mat=wood,r=.045):return C.cylinder(n,(x,y,h/2),r,h,mat,8)
    def notice(n,meaning,w,h,z,texture,posts=True,mat=white):
        col=start(n,meaning);panel(n,0,z,w,h,mat);art(texture,0,z,w*.97,h*.97)
        if posts:
            for x in [-w*.38,w*.38]:C.box('細い支持柱',(x,.04,(z+h/2)/2),(.055,.065,z+h/2),wood)
        end(n,col)
    if asset=='pass_info_boards':
        notice('large_white','登山口の大きい白い案内板。題字・本文は未判読、神奈川県のみ判読。',1.05,.70,1.18,'large_white')
        notice('small_weathered','大白板左下の小型褪色板。全文未判読。',.50,.27,.82,'small_weathered')
        notice('parking_notice','駐車場前の青見出し付き白い利用案内板。文字未判読。',.65,.5,1,'parking_notice')
        refs=['sv_toilet_000.png','sv_toilet_045.png','sv_u2_225.png'];notes=['題字・本文・図を創作せず薄い線と色面。支持部裏面は推定。']
    elif asset=='pass_trail_stairs':
        col=start('timber_step','土踏面と丸太蹴上げ1段。推奨8〜11段、全上部段数未確認。')
        C.box('土踏面',(0,.13,.07),(1.6,.32,.14),soil)
        C.pipe('横丸太蹴上げ',(-.88,-.03,.08),(.88,-.03,.08),.085,wood,8)
        for x in [-.83,.83]:C.pipe('縁丸太',(x,-.11,.08),(x,.30,.08),.055,wood,8)
        end('timber_step',col)
        for name,x,mat in [('rail_left',-.8,black),('rail_right',.8,black),('wood_rail_left',-.8,wood),('wood_rail_right',.8,wood)]:
            col=start(name,('写真で確認できる左側の金属手すり1区間。' if name=='rail_left' else '組合せ用の推定手すり1区間。右側連続金属手すりと木製は写真未確認。'))
            for y in [-.64,.64]:C.pipe('縦支柱',(x,y,0),(x,y,.84+(y+.64)*.5),.025,mat,8)
            for h in [.43,.84]:C.pipe('勾配に沿う桟',(x,-.7,h),(x,.7,h+.7),.023,mat,8)
            end(name,col)
        refs=['sv_toilet_090.png','sv_toilet_135.png','sv_lanes_045.png','sv_u2_225.png'];notes=['蹴上げ0.16m/踏面0.32mを推定。段1個と手すり1区間を納品し巨大階段一体化を避ける。','写真は左側黒金属手すり。右側/木製手すりはユーザー要件に合わせた推定の別部品。']
    elif asset=='pass_benches':
        col=start('bench_low_plank','低い板ベンチ。長手5枚板と2つの短脚。推奨3基。')
        for y in [-.22,-.11,0,.11,.22]:C.box('長手の座板',(0,y,.30),(1.95,.10,.05),darkwood)
        for x in [-.72,.72]:
            C.box('短い灰色脚',(x,0,.125),(.16,.46,.25),concrete)
            C.box('座面受け',(x,0,.26),(.10,.53,.06),wood)
        end('bench_low_plank',col)
        for name,L,num in [('bench_box_short',.92,6),('bench_box_long',1.2,8)]:
            col=start(name,'背なし箱形腰掛け。'+('短型。' if num==6 else '長型。')+'内部構造は推定。')
            for x in [-L/2+.035,L/2-.035]:
                for y in [-.26,.26]:C.box('黒い角枠',(x,y,.22),(.07,.07,.44),black)
                C.box('端上枠',(x,0,.415),(.07,.58,.05),black)
            for i in range(num):
                x=-L/2+.09+i*(L-.18)/(num-1)
                for y in [-.25,.25]:C.box('側面の縦板',(x,y,.22),((L-.12)/num-.006,.045,.37),pale)
            for i in range(5):C.box('上面板',(0,-.22+i*.11,.423),(L-.09,.10,.035),darkwood)
            for x in [-L/2+.06,L/2-.06]:C.box('端面板',(x,0,.21),(.025,.45,.35),wood)
            end(name,col)
        refs=['sv_toilet_045.png','sv_toilet_090.png','sv_lanes_000.png','sv_lanes_045.png'];notes=['箱形2種は木板側面と太い暗色端枠を分離。背面/内部は推定。']
    elif asset=='pass_trail_posts':
        col=start('trail_white_post','白木標柱の単体推定形。写真に寸法・題字の確実な裏付けなし。')
        prism('上部を切り落とした標柱',[(-.065,0),(.065,0),(.065,1.43),(.02,1.50),(-.065,1.46)],.13,pale)
        art('white_post',0,.9,.11,.75,y=-.066);end('trail_white_post',col)
        col=start('trail_brown_directions','登山口の複数段茶色方向板。行先未判読。')
        C.box('角柱',(0,.05,.70),(.09,.09,1.40),darkwood)
        for i,w in enumerate([.68,.56,.72,.58]):
            z=1.28-i*.19
            outline=[(-w/2,z-.065),(w/2-.07,z-.065),(w/2,z),(w/2-.07,z+.065),(-w/2,z+.065)]
            if i%2:outline=[(-x,zz) for x,zz in outline]
            prism('方向板'+str(i),outline,.04,darkwood)
            art('trail_direction',-.025,z,w-.13,.115,y=-.022)
        end('trail_brown_directions',col)
        col=start('trail_blue_notice','茶方向標の下の小青白注意板。文章未判読。')
        panel('小青板',0,.30,.33,.21,white);art('blue_notice',0,.30,.32,.2);C.box('小杭',(0,.035,.20),(.04,.04,.4),wood)
        end('trail_blue_notice',col)
        notice('trail_green_notice','登山口右の緑地白表示の小板。文字未判読。',.30,.32,1.05,'green_notice',False,black)
        # 緑板の支柱は同一partへ追加。
        C.box('緑板の単柱',(0,.04,.52),(.035,.035,1.04),black);parts['trail_green_notice']=list(C.CURRENT.objects)
        refs=['sv_toilet_045.png','sv_toilet_090.png','sv_toilet_135.png'];notes=['白木標柱の寸法/文字は未確定の推定部品。実際に観察できた茶方向板/緑小板とは別部品。','茶方向板は白い短線で文字密度のみ近似。']
    elif asset=='pass_bike_rack':
        col=start('cycle_rack_A','自転車用木製A脚台。上横材CYCLE RACKは判読できた範囲。')
        for x in [-.8,.8]:
            for y in [-.34,.34]:C.beam('開いたA脚',(x,y,0),(x,0,.88),.075,pale)
            C.box('A脚の貫',(x,0,.34),(.065,.48,.065),wood)
            C.box('上部金属留め',(x,-.045,.84),(.10,.018,.10),black)
        C.box('サドル用上横材',(0,0,.88),(1.85,.09,.09),pale)
        art('rack_label',0,.88,1.55,.075,y=-.046)
        end('cycle_rack_A',col);refs=['sv_toilet_045.png','sv_toilet_090.png','sv_u2_225.png'];notes=['木材の角断面・黒留め板を採用。細ネジは省略。']
    elif asset=='pass_bus_stop':
        for name,mat in [('bus_round_blank',metal),('bus_round_blank_dark',concrete)]:
            col=start(name,'円形道路標識/丸板の推定単体。正面未確認につき無地灰色。バス停と断定しない。')
            post('銀色支柱',0,.04,2.0,metal,.023)
            o=C.cylinder('円形薄板',(0,0,1.85),.28,.025,mat,16);o.rotation_euler[0]=math.pi/2
            for z in [1.75,1.95]:C.box('裏面固定桟',(0,.04,z),(.39,.035,.025),metal)
            end(name,col)
        col=start('bus_tall_timetable','登山口右の実在する縦長丸頭バス時刻表。地名以外は未判読。')
        C.box('台形台座',(0,0,.15),(.34,.34,.30),concrete)
        post('一本支柱',0,.02,.80,metal,.025)
        outline=[(-.16,.72)]+[(.16*math.cos(t),1.94+.16*math.sin(t)) for t in [math.pi-i*math.pi/8 for i in range(9)]]+[(.16,.72),(.10,.60),(0,.55),(-.1,.60)]
        prism('丸頭クリーム板',outline,.035,white)
        art('bus_timetable',0,1.32,.29,1.20,y=-.019)
        end('bus_tall_timetable',col);refs=['sv_toilet_090.png','sv_toilet_135.png','sv_lanes_225.png','sv_lanes_270.png'];notes=['青い買い物袋の丸印はGoogle Maps POIであり実物を制作しない。丸板2種はユーザー指定の無地灰variantとして提供し正面未確認を明記。','実物バス停は縦長丸頭。細かい時刻/路線情報は線のみ。']
    elif asset=='pass_brown_signs':
        notice('south_tall_sign','南側階段脇の茶色大板。ヤビツ峠・レストハウス、左上矢印。',.95,.92,1.0,'south_tall',True,darkwood)
        # 板の左右に白っぽい縦縁。
        for x in [-.46,.46]:C.box('白っぽい板端',(x,-.023,1),(.025,.025,.92),pale)
        parts['south_tall_sign']=list(C.CURRENT.objects)
        notice('south_wide_sign','南側の横長ハイキングコース板。先頭地名・距離未判読。',1.35,.38,1.12,'south_wide',True,darkwood)
        prism('右向き矢尻',[(.675,.93),(.84,1.12),(.675,1.31)],.035,darkwood);parts['south_wide_sign']=list(C.CURRENT.objects)
        col=start('lotn_small_sign','北入口ガードレール奥の2本丸太柱と小横板。文字は未判読。')
        for x in [-.45,.45]:post('丸太支柱',x,.05,.88,wood,.055)
        panel('横木案内',0,.67,.94,.22,wood);art('lotn_small',0,.67,.91,.19)
        end('lotn_small_sign',col)
        col=start('u1_black_direction','2014年u1の黒い左右二段方向標。上段のみ文字判読。')
        post('黒茶柱',0,.04,1.96,darkwood,.034);C.cylinder('白い頭帯',(0,.04,1.95),.039,.035,white,8)
        for n,z,w,left in [('u1_upper',1.70,1.14,True),('u1_lower',1.36,.94,False)]:
            outline=[(-w/2+.12,z-.14),(w/2,z-.14),(w/2,z+.14),(-w/2+.12,z+.14),(-w/2,z)] if left else [(-w/2,z-.11),(w/2-.12,z-.11),(w/2,z),(w/2-.12,z+.11),(-w/2,z+.11)]
            prism('左右矢羽根',outline,.045,darkwood);art(n,.03 if left else -.03,z,w-.18,.24 if left else .18,y=-.024)
        end('u1_black_direction',col);refs=['sv_south_090.png','sv_south_180.png','sv_lotn_000.png','sv_lotn_045.png','sv_u1_000.png','sv_u1_315.png'];notes=['南の下部アイコン/小字、北入口本文、u1下段の行先は未判読。文字は創作せず線。','u1は2014年写真。2025年の残存未確認。']
    elif asset=='pass_road_items':
        orange=C.proc('オレンジ弾性樹脂',(232,70,33),scale=17)
        col=start('orange_pole','銀白反射帯3本の弾性オレンジポール。円錐コーンではない。')
        C.box('角丸風フランジ',(0,0,.015),(.20,.20,.03),orange)
        C.cylinder('円柱ポール',(0,0,.36),.037,.68,orange,8)
        for z in [.47,.56,.65]:C.cylinder('反射帯',(0,0,z),.038,.035,white,8)
        for x in [-.07,.07]:C.cylinder('根元ボルト',(x,0,.034),.01,.008,metal,6)
        end('orange_pole',col)
        for name,label,w,h,z,feet in [('camera_pole_board','camera_pole',.35,1.10,.92,True),('camera_standing','camera_ground',.46,1.02,.53,False)]:
            col=start(name,'縦書きカメラ作動中の白板。'+('mid柱付型。' if feet else 'lotn地面立て型、上端の監視は未読。'))
            panel('錆色の縁板',0,z,w,h,rust,.045);art(label,0,z,w-.035,h-.035,y=-.024)
            if feet:post('灰色細柱',0,.04,2.75,metal,.035)
            else:
                for x in [-w*.35,w*.35]:C.box('短い裏脚',(x,.06,.25),(.035,.05,.50),wood)
            end(name,col)
        col=start('white_cylinder','赤白の区画帯付き白い小円筒。用途未確認。')
        pink=C.proc('褪せた赤桃',(184,115,116),scale=14)
        C.cylinder('円筒本体',(0,0,.27),.21,.54,white,12)
        for z in [.17,.37]:
            for i in range(0,12,2):
                a=i*math.tau/12;b=(i+1)*math.tau/12
                C.mesh('桃赤の区画',[(.211*math.cos(a),.211*math.sin(a),z-.065),(.211*math.cos(b),.211*math.sin(b),z-.065),(.211*math.cos(b),.211*math.sin(b),z+.065),(.211*math.cos(a),.211*math.sin(a),z+.065)],[(0,1,2,3)],pink)
        C.cylinder('黄色味の蓋',(0,0,.552),.224,.035,pale,12);C.box('蓋の持ち手',(0,0,.58),(.10,.035,.025),white)
        end('white_cylinder',col)
        col=start('guardrail','白い鋼製W断面ガードレール1区間。丸い端部付き。')
        # 板厚を持つW断面をX方向へ押し出す。
        cross=[(-.045,.52),(-.09,.57),(-.09,.64),(-.025,.69),(-.09,.74),(-.09,.81),(-.045,.86)]
        ring=cross+[(y+.004,z) for y,z in reversed(cross)];N=len(ring)
        v=[(x,y,z) for x in [-1,1] for y,z in ring]
        C.mesh('厚さ4mmのW形ビーム',v,[tuple(range(N-1,-1,-1)),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)],white)
        for x in [-.85,.85]:post('丸い支持柱',x,.015,.70,white,.045)
        for x in [-.85,.85]:
            for z in [.59,.79]:
                o=C.cylinder('ビーム留めボルト',(x,-.102,z),.012,.014,metal,6);o.rotation_euler[0]=math.pi/2
        # 折り返し末端は角を落とした外周を立体化。
        prism('丸角末端',[(1,.52),(1.20,.52),(1.25,.57),(1.25,.81),(1.20,.86),(1,.86)],.035,white).location.y=-.045
        end('guardrail',col)
        col=start('wood_fence','北入口の丸太2柱と横木の柵1区間。')
        for x in [-.65,.65]:post('丸太柱',x,0,.85,wood,.06)
        for z in [.34,.67]:C.pipe('水平丸太横木',(-.7,0,z),(.7,0,z),.043,wood,8)
        end('wood_fence',col)
        col=start('island_rope_fence','島外周の丸太杭2本と垂れたロープ1区間。')
        for x in [-.7,.7]:post('低い木杭',x,0,.58,wood,.045)
        points=[(-.7,0,.45),(-.35,0,.34),(0,0,.30),(.35,0,.34),(.7,0,.45)]
        for i in range(4):C.pipe('垂れた暗色ロープ',points[i],points[i+1],.009,black,6)
        end('island_rope_fence',col)
        col=start('tree_tripod','若木用三脚支柱。樹木を含まない。')
        for t in [0,math.tau/3,math.tau*2/3]:C.pipe('斜め丸木',(math.cos(t)*.45,math.sin(t)*.45,0),(math.cos(t)*.10,math.sin(t)*.10,1.13),.034,pale,6)
        for y in [-.10,.10]:C.pipe('上部の横丸木',(-.34,y,.96),(.34,y,.96),.035,pale,6)
        for x in [-.11,.11]:C.pipe('結束縄',(x,-.15,.95),(x,.15,.95),.012,black,6)
        end('tree_tripod',col)
        col=start('white_stepping_stone','白踏石1個。トイレ横は5個推奨。')
        ring=[(-.18,-.06),(-.13,-.12),(.09,-.12),(.18,-.04),(.15,.08),(.05,.13),(-.13,.10)]
        N=len(ring);v=[(x,y,z) for z in [0,.055] for x,y in ring]
        C.mesh('真上も不規則な平石',v,[tuple(range(N-1,-1,-1)),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)],white)
        end('white_stepping_stone',col)
        col=start('manhole','円形鉄蓋1個。文字と市章未判読。')
        C.cylinder('外周鉄輪',(0,0,.012),.34,.024,rust,16);C.cylinder('内側の蓋',(0,0,.027),.31,.008,concrete,16)
        o=C.mesh('自作の鉄蓋模様',[(0,0,.0315)]+[(.305*math.cos(i*math.tau/24),.305*math.sin(i*math.tau/24),.0315) for i in range(24)],[(0,i+1,(i+1)%24+1) for i in range(24)],tex('manhole_top'))
        for poly in o.data.polygons:
            for li in poly.loop_indices:
                co=o.data.vertices[o.data.loops[li].vertex_index].co;o.data.uv_layers['UVMap'].data[li].uv=(.5+co.x/.61,.5+co.y/.61)
        for x in [-.18,.18]:C.box('引上げ穴',(x,0,.0307),(.055,.018,.002),black)
        end('manhole',col)
        refs=['sv_mid_045.png','sv_mid_225.png','sv_lotn_000.png','sv_lotn_225.png','sv_lotn_270.png','sv_toilet_045.png','sv_lanes_225.png','sv_lanes_270.png'];notes=['木柵の北入口横木型と島のロープ型を別部品。','マンホールの文字/細意匠、白円筒の用途は未確認。白踏石は単体5個配置推奨。','ガードレールは厚さ4mmの閉じたW断面、末端は角を落とした丸角形。']
    elif asset=='pass_utility_pole':
        for name,h,red in [('pole_shop',10,False),('pole_north',9.2,False)]:
            col=start(name,'電柱1本。'+('北入口の灰白コンクリート柱。' if name=='pole_north' else '売店脇の灰白テーパー柱。')+'短い引込線のみ。')
            mat=rust if red else concrete
            N=12;v=[(r*math.cos(i*math.tau/N),r*math.sin(i*math.tau/N),z) for z,r in [(0,.15),(h,.085)] for i in range(N)]
            C.mesh('テーパー柱',v,[tuple(range(N-1,-1,-1)),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)],mat)
            for z in [1.7,3.0,h-.85,h-1.3]:C.cylinder('締付金属帯',(0,0,z),.145 if z<4 else .105,.035,black,12)
            C.box('上部腕木',(0,0,h-.65),(1.45,.09,.09),metal)
            for x in [-.57,0,.57]:
                C.pipe('碍子支持',(x,0,h-.65),(x,0,h-.42),.014,black,6)
                for z in [h-.56,h-.50,h-.44]:C.cylinder('碍子のひだ',(x,0,z),.055,.035,white,8)
            C.box('変圧器機器箱',(.30,.08,h-2.0),(.38,.36,.62),metal)
            C.box('箱のふた',(.30,.08,h-1.67),(.42,.39,.035),black)
            C.box('低い接続箱',(-.03,-.18,1.65),(.33,.16,.52),white)
            C.pipe('縦の配管',(-.13,-.07,.4),(-.13,-.07,h-1),.015,black,6)
            for y in [-.1,.1]:
                C.pipe('短い引込線',(0,y,h-.40),(.55,y-.25,h-.58),.008,black,6)
                C.pipe('短い引込線垂れ',(.55,y-.25,h-.58),(1.2,y-.55,h-.52),.008,black,6)
            for z in [1,1.5,2,2.5,3,3.5,4]:C.pipe('短い足場金具',(-.13,0,z),(-.3,0,z),.012,metal,6)
            end(name,col)
        col=start('pole_opposite_red','対岸の赤褐色の細い支柱。全機器と型式は未確認、単純形。')
        C.cylinder('赤褐の細い柱',(0,0,4.5),.075,9.0,rust,8)
        C.pipe('短い腕木',(-.5,0,8.6),(.5,0,8.6),.025,metal,6)
        C.pipe('短い架線支持',(0,0,8.6),(.7,0,8.5),.008,black,6)
        end('pole_opposite_red',col)
        offset={'pole_shop':[-1.5,0,0],'pole_north':[0,0,0],'pole_opposite_red':[1.5,0,0]};refs=['sv_lotn_000.png','sv_lotn_225.png','sv_mid_270.png','sv_toilet_225.png'];notes=['腕木/碍子/箱/引込線は写真の可視範囲と一般構造から推定。箱の型式と管理銘板未確認。','北入口variantの全設備構成は写真の遮蔽で未確認のため推定。全コース用架線は含めない。']
    elif asset=='pass_barriers':
        for name,L in [('barrier_A_short',1.5),('barrier_A_long',1.8)]:
            col=start(name,'売店右空地のA脚バリケード。白い小札の文章未判読。')
            for x in [-L/2+.1,L/2-.1]:
                for y in [-.30,.30]:C.beam('A脚',(x,y,0),(x,0,.80),.055,wood)
                C.box('A脚貫',(x,0,.34),(.045,.43,.045),wood)
                C.box('黒い端留め',(x,-.035,.76),(.075,.016,.08),black)
            C.box('赤褐の上横材',(0,0,.80),(L,.06,.07),rust)
            C.front_panel('小札自作線',(0,-.04,.56),.21,.17,tex('small_tags'))
            end(name,col)
        refs=['sv_lotn_225.png','sv_lotn_135.png'];notes=['自転車台に似るが別の赤茶横材・小札付きバリケードとして分離。隠れた脚は推定。']
    elif asset=='pass_site_equipment':
        col=start('electric_cabinet_small','柱脇の小灰白機器箱。型式未確認。')
        C.box('箱',(0,0,.66),(.34,.32,1.1),white)
        C.box('扉の段差',(0,-.17,.68),(.29,.025,1),metal)
        C.box('黒い小窓',(0,-.187,.95),(.12,.008,.16),black)
        C.box('地面据付脚',(0,0,.055),(.25,.26,.11),concrete)
        end('electric_cabinet_small',col)
        col=start('electric_cabinet_large','柱脇の大型白箱。上部/側面は推定。')
        C.box('大箱',(0,0,.66),(.65,.45,1.32),white)
        for z in [.32,.63,.94]:C.box('扉継ぎ目',(0,-.23,z),(.61,.008,.012),black)
        for x in [-.24,.24]:C.box('箱脚',(x,0,.05),(.09,.35,.1),concrete)
        end('electric_cabinet_large',col)
        col=start('equipment_fence','機器箱周りの低い黒金属囲い1区間。')
        for x in [-.60,.60]:C.box('囲い柱',(x,0,.40),(.035,.035,.8),black)
        for z in [.15,.74]:C.box('囲い横枠',(0,0,z),(1.23,.03,.03),black)
        for x in [-.4,-.2,0,.2,.4]:C.box('囲い縦棒',(x,0,.44),(.012,.018,.56),black)
        end('equipment_fence',col)
        col=start('island_sign','島内の太い丸太2柱の暗茶横長板。全文未判読。')
        for x in [-.48,.48]:post('太い柱',x,.04,1.45,darkwood,.07)
        panel('暗茶板',0,1.13,1.0,.56,darkwood,.06);art('island_notice',0,1.13,.95,.51,y=-.032)
        end('island_sign',col)
        col=start('opposite_yellow_notice','道路対岸の黄色縦板と上の白い板。本文未判読。')
        post('細い支持柱',0,.03,2.45,metal,.025)
        panel('黄色板',0,.82,.32,.72,rust);art('yellow_notice',0,.82,.30,.70)
        panel('上の白板',0,1.75,.38,.65,white)
        end('opposite_yellow_notice',col)
        col=start('small_roof_notice_post','トイレ正面の小屋根付き掲示柱。用途・文字未判読。')
        C.box('方形台座',(0,0,.10),(.25,.25,.20),concrete)
        C.box('掲示柱',(0,.02,.73),(.075,.075,1.26),wood)
        panel('札板',0,.90,.28,.52,pale);art('roof_notice',0,.90,.25,.49)
        for side in [-1,1]:
            vs=[(0,-.21,1.39),(side*.24,-.21,1.20),(side*.24,.21,1.20),(0,.21,1.39)]
            C.mesh('柱に接続する小切妻屋根',vs+[(x,y,z-.035) for x,y,z in vs],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],darkwood)
        end('small_roof_notice_post',col)
        col=start('green_column','駐車場前の濃緑円筒設備。用途未確認。')
        green=C.proc('緑塗装',(43,75,57),scale=13)
        C.cylinder('円筒',(0,0,.60),.105,1.2,green,12)
        C.cylinder('丸い蓋',(0,0,1.21),.112,.035,green,12)
        C.pipe('小さい上部金具',(0,0,1.23),(.08,0,1.23),.015,metal,6)
        end('green_column',col)
        refs=['sv_lotn_225.png','sv_lanes_225.png','sv_lanes_270.png','sv_toilet_000.png','sv_toilet_225.png','sv_toilet_315.png','sv_u2_045.png'];notes=['追加人工物を独立部品化。用途や看板文章を推測で記入しない。機器の背面と内部は推定。']
    else:raise ValueError(asset)
    # 独立部品の制作元もXY中央・最下点へ統一する。展示位置はpreview_offsetsのみ。
    from mathutils import Vector
    bpy.context.view_layer.update()
    for obs in parts.values():
        vv=[o.matrix_world @ Vector(v) for o in obs for v in o.bound_box]
        origin=Vector(((min(v.x for v in vv)+max(v.x for v in vv))/2,(min(v.y for v in vv)+max(v.y for v in vv))/2,min(v.z for v in vv)))
        for o in obs:o.location-=origin
    return {'parts':parts,'part_info':pi,'preview_offsets':offset,'generator':'blender_scripts/build_pass_props.py','resolution':1024 if asset in ['pass_trail_posts','pass_brown_signs','pass_bus_stop'] else 512,'refs':['codex07/'+x for x in refs],'dimension_basis':'export/compare/pass/work07/props_inventory.jsonの写真比率推定。現地実測ではない。小物±20〜30%、階段全体/電柱±40%。','approximations':notes,'reference_views':{'reference_props':(1,-2,.8)},'stair_assembly':{'step_advance_y_m': 0.32, 'step_rise_z_m': 0.16, 'clear_width_m': 1.6, 'visible_lower_step_count': [8, 11], 'total_step_count': '上部は未確定', 'rail_post_spacing_m': 1.28, 'rail_segment_advance_y_m': 1.28, 'rail_segment_rise_z_m': 0.64, 'rail_left_x_m': -0.8, 'rail_right_x_m': 0.8, 'uphill_direction': '+Y', 'origin_note': '各部品は根元の外接矩形中央。左/右の横位置は配置時に加える。手すり1区間は段4個分。'} if asset=='pass_trail_stairs' else None,'assembly_notes':'各部品は根元中央。preview_offsetsは確認用の展示位置。配置座標ではない。'}

if __name__=='__main__':
    if 'bpy' not in sys.modules:textures()
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        import pass_common as C
        a=sys.argv[sys.argv.index('--')+1:][0];C.init_scene();C.save_asset(a,build(a))
