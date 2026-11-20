"""依頼60。ローカルの自作の枝葉を集めたアトラスと、KN5用の軽い植生。"""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/landmarks/trees_mountain"
DEST = ROOT / "assets/landmarks/trees_light"
EXPORT = DEST / "export"
CHECK = ROOT / "export/compare/trees_light/work/shape_check.json"
BLENDER = "/opt/homebrew/bin/blender"
SPECIES = ("sugi", "hinoki", "broadleaf")
SIZE = 2048
MIDDLE = "trees_light_middle.png"
DARK = "trees_light_dark.png"
ALPHA = 0.30
RULE = ("舗装端から水平18 mと26 mの2列。地形高に合わせ、道沿いの間隔は12 mに±1 mのゆらぎを付ける。"
        "奥列を6 mずらし、板の幅方向を道路接線から交互に+32°、−32°振る。奥列は手前列と逆符号。")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def leaf_rect(species, variant):
    top = SPECIES.index(species) * 512 + variant * 256
    return (1536, top, 2048, top + 256)


def bark_rect(species):
    left = 1536 + SPECIES.index(species) * 170
    return (left + 3, 1540, left + 167, 2044)


def far_rect(index):
    x, y = 512 * (index % 3), 1024 * (index // 3)
    return x + 3, y + 3, x + 509, y + 1021


def uv(rect, u, v):
    x0, y0, x1, y1 = rect
    return ((x0 + .5 + (x1-x0-1)*u)/SIZE, 1-(y1-.5-(y1-y0-1)*v)/SIZE)


def bough_image(species, variant):
    """一本の枝先の画像を二次分枝の先に載せ、数mの枝群のシルエットを作る。"""
    from PIL import Image, ImageDraw
    rng = random.Random(713 + SPECIES.index(species)*65 + variant)
    source_name = "oak" if species == "broadleaf" else species
    sprig = Image.open(SOURCE / f"leaf_{source_name}.png").convert("RGBA")
    sprig.thumbnail((160,160), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA",(512,512))
    draw = ImageDraw.Draw(canvas)
    target = np.array({"sugi":(33,46,25),"hinoki":(34,47,28),"broadleaf":(41,51,27)}[species])
    raw = np.asarray(sprig)
    source_mean = raw[raw[:,:,3]>100,:3].mean(axis=0)
    stamps = []
    def limb(a,b,width):
        draw.line((tuple(a),tuple(b)), fill=(51,43,34,255), width=width)
    def foliage(x,y,size,angle,light):
        stamps.append((x,y,size,angle,light))
    if species in ("sugi","hinoki"):
        # 枝の取付側は上中央。枝の先は左右・下へ扇状に分かれて垂れる。
        # 一本の垂れた生枝を数本の亜主枝に分け、その両脇から別々の房を分ける。
        # 根元から末端まで大きな刷毛で埋めると、走行視点で一枚の幅広い葉に見える。
        junction=np.array((244.,16.))
        tips=[(-157,240),(-207,184),(-83,349),(12,399),(107,344),(182,243),(212,138)]
        if variant:
            tips=[(-203,168),(-175,267),(-108,350),(-25,406),(69,362),(157,284),(207,204)]
        for k,(offset,length) in enumerate(tips):
            sign=1 if offset>0 else -1
            stem=junction+[rng.uniform(-8,8),rng.uniform(1,24)]
            end=stem+[offset+rng.uniform(-10,10),length+rng.uniform(-14,17)]
            bend=stem*.42+end*.58+[sign*(27 if species=="sugi" else 43),-48]
            def axis(t):
                return (1-t)**2*stem+2*t*(1-t)*bend+t*t*end
            nodes=[axis(i/10) for i in range(11)]
            for a,bp in zip(nodes,nodes[1:]):
                limb(a,bp,2)
            limb(junction,stem,3)
            stem_value=rng.uniform(.79,1.12)
            for pair in range(1,9):
                along=(pair+rng.uniform(-.17,.14))/9
                inner=axis(along)
                foliage(inner[0]+rng.uniform(-5,5),inner[1]+7,
                        rng.uniform(27,43),-sign*rng.uniform(108,170),stem_value*rng.uniform(.73,1.03))
                for flank in (-1,1):
                    base=axis(along+rng.uniform(-.027,.025))
                    spread=(25+20*math.sin(math.pi*along))*(1.1-.46*along)
                    dx=flank*spread*rng.uniform(.63,1.28)+offset*.065
                    dy=rng.uniform(15,34)+(11 if species=="sugi" else -6)
                    tip=base+[dx,dy]
                    turn=base+[dx*.75,dy*.28]
                    limb(base,turn,2); limb(turn,tip,1)
                    branch_value=stem_value*rng.uniform(.83,1.22)
                    for n in range(1,6):
                        t=n/5
                        loc=(1-t)**2*base+2*t*(1-t)*turn+t*t*tip
                        # 針葉の先は枝軸から離れる。房の間の切れ込みは縮小後も残す。
                        angle=-math.degrees(math.atan2(dx,-dy))+(-1)**n*19
                        radius=rng.uniform(26,44)*(1-.36*t)
                        foliage(loc[0]+(-1)**n*rng.uniform(1,5),loc[1]+rng.uniform(-4,4),radius,angle,
                                branch_value*(.90+.21*t)+rng.uniform(-.08,.10))
            foliage(*end,31,-sign*155,stem_value*1.11)
    else:
        # 大枝から分かれる小枝を板の下半分にも描き、葉叢と葉叢の間に枝の線を残す。
        root=np.array((201. if variant else 269.,492.))
        cluster_centers=[(-.86,.03),(-.53,-.49),(-.18,-.78),(.23,-.62),(.64,-.42),
                         (.88,-.02),(.40,.23),(-.28,.31),(.03,-.19)]
        if variant:
            cluster_centers=[(-.81,.18),(-.61,-.34),(-.20,-.47),(.07,-.80),(.49,-.54),
                             (.82,-.32),(.68,.08),(.05,.17),(-.31,-.08)]
        for k,(cx,cy) in enumerate(cluster_centers):
            center=np.array((247+cx*193+rng.uniform(-9,9),226+cy*189+rng.uniform(-13,13)))
            elbow=root*.56+center*.44+[rng.uniform(-18,20),-17]
            stem=elbow*.42+center*.58+[rng.uniform(-17,16),9]
            limb(root,elbow,5); limb(elbow,stem,4); limb(stem,center,2)
            tone=rng.uniform(.69,1.23)
            for j in range(33):
                theta=rng.random()*math.tau; r=math.sqrt(rng.random())
                dx,dy=math.cos(theta)*r*53,math.sin(theta)*r*41
                size=rng.uniform(52,92)
                foliage(center[0]+dx,center[1]+dy,size,rng.uniform(-110,110),
                        tone*(1-.14*dy/41)+rng.uniform(-.15,.18))
    for x,y,size,angle,light in stamps:
        # 光はゲームに委ねる。枝の世代・葉群内の粗密だけを色に残す。
        rgb = np.clip(raw[:,:,:3] * (target/source_mean*light),0,255).astype("uint8")
        stamp = Image.fromarray(np.dstack((rgb,raw[:,:,3])))
        stamp = stamp.resize((int(size),int(size)),Image.Resampling.LANCZOS)
        stamp = stamp.rotate(angle,Image.Resampling.BICUBIC,expand=True)
        canvas.alpha_composite(stamp,(int(x-stamp.width/2),int(y-stamp.height/2)))
    # 外縁を抜いたまま、タイルの中へ収める。
    crop = canvas.crop(canvas.getbbox())
    tile = Image.new("RGBA",(512,256))
    tile.alpha_composite(crop.resize((502,250),Image.Resampling.LANCZOS),(5,3))
    return tile


def bark_image(species, size):
    """縦溝と裂けた樹皮。縦方向は周期を持たせて幹のリングの継ぎ目をつなぐ。"""
    from PIL import Image, ImageDraw
    rng=random.Random(7051+SPECIES.index(species))
    src=Image.open(SOURCE/("bark_"+("oak" if species=="broadleaf" else species)+".png")).convert("RGB")
    width,height=size
    a=np.asarray(src.resize(size,Image.Resampling.LANCZOS)).astype(float)
    base=np.array({"sugi":(54,43,36),"hinoki":(56,47,39),"broadleaf":(54,50,44)}[species])
    a=(a/a.mean(axis=(0,1))-1)*1.46+1
    tile=Image.fromarray(np.clip(base*a,0,255).astype("uint8"))
    draw=ImageDraw.Draw(tile)
    # 薄くめくれる条片、割れ目の陰とわずかに明るい縁を描く。
    for i in range(570):
        x=rng.uniform(0,width); y=rng.uniform(0,height)
        length=rng.uniform(6,48)*(1.48 if species=="sugi" else .88)
        breadth=rng.uniform(.45,2.25)*(1.45 if species=="broadleaf" else 1)
        tilt=rng.uniform(-2.8,3.1); bend=rng.uniform(-1.8,1.8)
        start=np.array((x,y)); end=start+[tilt,length]
        middle=start+[bend+.38*tilt,length*.46]
        gain=rng.uniform(.50,1.46)
        col=tuple(np.clip(base*gain,0,255).astype(int))
        edge=tuple(np.clip(base*(gain+rng.uniform(.12,.32)),0,255).astype(int))
        for sy in ((0,-height) if end[1]>=height else (0,)):
            off=np.array((0,sy))
            points=[tuple(v+off) for v in (start,middle-[breadth,0],end,middle+[breadth*.38,0])]
            draw.polygon(points,fill=col)
            if length>26: draw.line([points[0],points[3],points[2]],fill=edge,width=1)
    # 画像の端を跨ぐ読み出しでも同色になる2画素の余白。
    data=np.array(tile)
    data[-1]=data[0]; data[:,-1]=data[:,0]
    return Image.fromarray(data)


def paint():
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from trees_light_groves import paint_groves
    atlas = Image.new("RGBA",(SIZE,SIZE))
    for sp in SPECIES:
        for variant in range(2):
            atlas.paste(bough_image(sp, variant), leaf_rect(sp,variant)[:2])
        rect=bark_rect(sp)
        atlas.paste(bark_image(sp,(rect[2]-rect[0],rect[3]-rect[1])),rect[:2])
    for i,tile in enumerate(paint_groves()):
        rect = far_rect(i)
        atlas.paste(tile.resize((rect[2]-rect[0],rect[3]-rect[1]),Image.Resampling.LANCZOS),rect[:2])
    data=np.array(atlas)
    # 低アルファの無色縁へRGBを延ばし、縮小時に黒い輪郭が付くのを防ぐ。
    for x in range(0,SIZE,512):
        for y in range(0,SIZE,512):
            block=data[y:y+512,x:x+512]
            empty=block[:,:,3]<32
            idx=distance_transform_edt(empty, return_distances=False,return_indices=True)
            block[:,:,:3][empty]=block[idx[0][empty],idx[1][empty],:3]
    Image.fromarray(data).save(DEST/MIDDLE)
    data[:,:,:3]=np.rint(data[:,:,:3]*.72).astype("uint8")
    Image.fromarray(data).save(DEST/DARK)


class Mesh:
    def __init__(self):
        self.verts, self.uvs, self.faces = [], [], []
        self.parts = []

    def part(self, name):
        self.parts.append(dict(name=name, face_start=len(self.faces)))

    def polygon(self, points, coords, both=False):
        # 裏側の頂点も分ける。KN5の面積加重法線が逆面で打ち消されない。
        for back in range(1+int(both)):
            offset=len(self.verts)
            self.verts.extend(points)
            self.uvs.extend(coords)
            for n in range(1,len(points)-1):
                tri=(0,n+1,n) if back else (0,n,n+1)
                self.faces.append([offset+i for i in tri])

    def tube(self, name, centers, radii, sides, rect):
        self.part(name)
        p=np.array(centers,dtype=float)
        axis=np.array((0.,0.,1.)) if name=="trunk" else p[-1]-p[0]
        axis/=np.linalg.norm(axis)
        u=np.cross((0,1,0),axis); u/=np.linalg.norm(u)
        v=np.cross(axis,u)
        angles=np.arange(sides)*math.tau/sides
        ring=np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
        rings=[c+r*ring for c,r in zip(p,radii)]
        for seg,(lo,hi) in enumerate(zip(rings,rings[1:])):
            bottom_v=1-min(1.,np.linalg.norm(p[1]-p[0])/4.2) if seg==0 and name=="trunk" else 0.
            for j in range(sides):
                k=(j+1)%sides
                self.polygon([lo[j],lo[k],hi[k],hi[j]],
                             [uv(rect,j/sides,bottom_v),uv(rect,(j+1)/sides,bottom_v),
                              uv(rect,(j+1)/sides,1),uv(rect,j/sides,1)])
        for cap in (rings[0][::-1],rings[-1]):
            self.polygon(cap,[uv(rect,.5+.43*math.cos(a),.5+.43*math.sin(a)) for a in angles])

    def card(self, bottom,top,angle,half_bottom,half_top,rect,flip=False):
        across=np.array((-math.sin(angle),math.cos(angle),0.))
        pts=[bottom-half_bottom*across,bottom+half_bottom*across,
             top+half_top*across,top-half_top*across]
        self.polygon(pts,[uv(rect, int(flip)^x,y) for x,y in ((0,0),(1,0),(1,1),(0,1))],True)


def near(species,variant):
    rng=random.Random(170+SPECIES.index(species)*31+variant)
    m=Mesh()
    h={"sugi":(22.,20.4),"hinoki":(16.,17.3),"broadleaf":(19.8,18.5)}[species][variant]
    rect=bark_rect(species)
    if species in ("sugi","hinoki"):
        width={"sugi":(6.4,5.9),"hinoki":(6.6,7.1)}[species][variant]
        lean=np.array((.10,.12))*(1 if variant else -1)
        trunk=[(*tuple(lean*f*f),z) for f,z in ((0,-.3),(.04,.55),(.22,h*.22),(.44,h*.44),
                                                      (.66,h*.66),(.85,h*.85),(1,h-.16))]
        r=(.25 if species=="sugi" else .22)*(1+.09*variant)
        m.tube("trunk",trunk,[r*1.43,r,r*.89,r*.76,r*.54,r*.30,.026],8,rect)
        # 短い枯れ枝を二か所残す。下の生枝は路側、後ろには短い折れ口。
        for k,(level,az,length) in enumerate(((.388,.31-.60*variant,1.82),
                                               (.267,2.07+variant,.81))):
            az+=.37 if species=="hinoki" else 0
            radial=np.array((math.cos(az),math.sin(az),0.))
            foot=np.array((*tuple(lean*level**2),h*level))
            mid=foot+radial*length*.56+[0,0,.22]
            end=foot+radial*length+[0,0,.14 if k==0 else .36]
            m.tube(f"stub_{k+1}",[foot,mid,end],[.085 if k==0 else .064,.041,.012],5,rect)
        m.part("foliage")
        # 上部4段は全周、下部は道路側だけ。樹冠の内側を多数の板で埋めない。
        tiers=[(.58,1.00,4),(.695,.87,4),(.795,.67,4),(.889,.44,3),(.953,.235,3)]
        if species=="hinoki":
            tiers=[(.53,1.,4),(.635,.97,4),(.75,.78,4),(.851,.63,3),(.94,.36,3)]
        if variant:
            # Bは林縁で片側が生長した形。Aと異なる高さに下段を残す。
            tiers=[(level-(.027 if k<2 else .016 if k<4 else .008),
                    ratio*(1.12 if k==1 else .92 if k==3 else 1.),count)
                   for k,(level,ratio,count) in enumerate(tiers)]
        def tier_card(az,level,span,depth,j,top_level=None):
            outward=np.array((math.cos(az),math.sin(az),0.))
            b=outward*(span*.70)+[0,0,h*level-depth*.52]
            t=outward*(span*.09)+[0,0,h*level+depth*.48 if top_level is None else top_level]
            # 林冠の頂に向けて先細りにする。隣の段どうしで枝先が同じ線に並ばない。
            t+=np.array((-math.sin(az),math.cos(az),0))*span*rng.uniform(-.21,.19)
            crown_tip=level>.925
            m.card(b,t,az+rng.uniform(-.19,.19),span*rng.uniform(.67,.86),
                   span*(.12 if crown_tip else rng.uniform(.38,.63)),leaf_rect(species,j%2),bool(j%3))
        for k,(level,ratio,count) in enumerate(tiers):
            spin=k*1.06+variant*.71
            for j in range(count):
                az=spin+(j+rng.uniform(-.11,.11))*math.tau/count
                span=width*.52*ratio*rng.uniform(.90,1.08)
                top=h if k==4 and j==0 else None
                # 一段全部を同じ高さに揃えず、長い一枝が下の段に差しかかる。
                depth=(.185 if k<4 else .124) * (1.17 if (j+k)%3==0 else .97)
                center=level+rng.uniform(-.018,.013)
                if k==4:
                    center=min(center,1-depth*.48-.008*j)
                tier_card(az,center,span,h*depth,j+k,top)
        for j in range(3):
            tier_card((j-1)*.72+.18*variant,.395+.055*j-.047*variant,width*(.45-.023*j),h*.145,j)
    else:
        # 道路を向く +X の太枝と、反対側へ分かれる枝。葉群ごとに前後の厚さを持たせる。
        r=.33 if variant==0 else .30
        side=1 if variant==0 else -1
        fork=np.array((.30,.18*side,h*(.34 if variant else .38)))
        main_top=np.array((-.76,side*.33,h*.82))
        m.tube("trunk",[(0,0,-.3),(.04,.02,.6),(-.14,0,h*.19),fork,main_top],
               [r*1.4,r,r*.88,r*.75,.072],7,rect)
        crowns=[]
        branch_specs=(((.06,5.8,.62), (2.6,3.30,.75),(-1.3,3.4,.81)) if variant==0 else
                      ((-.30,6.15,.59),(1.93,3.70,.70),(-2.32,3.05,.84)))
        for j,(az,reach,level) in enumerate(branch_specs):
            direction=np.array((math.cos(az),math.sin(az),0.))
            # 付け根は主幹の中心線上、枝先ほど寝かせて道路に広げる。
            fraction=(0,.22,.52)[j]
            foot=fork*(1-fraction)+main_top*fraction
            end=direction*reach+[0,0,h*level]
            bend=foot*.58+end*.42-direction*.47+[0,0,.73-.18*j]
            m.tube(f"branch_{j+1}",[foot,bend,end],[r*(.67-.06*j),r*.43,.05],5,rect)
            crowns.append((bend,end,az))
        m.part("foliage")
        for j,(bend,end,az) in enumerate(crowns):
            # 葉の束を5方に並べた輪にせず、枝の途中から外・左右・上へ別々に分ける。
            configs=[(.08,-.50,1.65,2.55,2.05),(.41,1.04,2.65,3.09,2.35),
                     (.57,-1.26,2.36,3.80,2.20),(.72,.15,3.06,2.45,1.98),
                     (.35,2.86,2.42,4.50,2.49)]
            for i,(along,turn,reach,rise,w) in enumerate(configs):
                a=az+turn+rng.uniform(-.21,.21)
                outward=np.array((math.cos(a),math.sin(a),0))
                base=bend*(1-along)+end*along
                # 付け根より下に宙吊りの葉群を作らず、小枝を材のある点から分ける。
                top=base+outward*reach+[0,0,rise+rng.uniform(-.25,.30)]
                m.card(base,top,a,w*.85,w*1.25,leaf_rect(species,(i+j+variant)%2),i%2)
        # 幹際と外枝の間に残る空間へ小枝を分ける。枝の張る方向を変え、
        # 道側の低い枝だけは先の葉叢が水平から少し垂れる形にする。
        for j,(bend,end,az) in enumerate(crowns):
            base=.55*bend+.45*end
            a=az+(-1 if j%2 else 1)*.51
            outward=np.array((math.cos(a),math.sin(a),0))
            m.card(base,base+outward*(2.65 if j==0 else 2.15)+[0,0,.52 if j==0 else 1.25],
                   a,1.4,2.2,leaf_rect(species,variant),j%2)
        for i in range(2):
            a=-.56+i*2.12+.36*variant
            toward=np.array((math.cos(a),math.sin(a),0))
            base=np.array((.48,side*.25,h*(.61+.085*i)))
            m.card(base,base+toward*2.44+[0,0,1.85],a,1.55,2.3,leaf_rect(species,i),i)
        # 上部では角度の浅い2束が真上から見た中心をまたぐ。
        for i,(a,low,high,offset,width) in enumerate((
                (1.23,.68,.94,2.62,2.65),(-1.09,.70,.91,2.91,2.90),
                (2.56,.77,.985,2.28,2.43),(-.48,.79,1.0,1.93,2.36))):
            a=a*side+.23*variant; radial=np.array((math.cos(a),math.sin(a),0))
            start=main_top-1.15*radial; start[2]=h*low
            top=main_top+offset*radial; top[2]=h*high
            m.card(start,top,a,width*.63,width*1.02,leaf_rect(species,i%2),i%2)
    return m


FAR=[("sugi","a",12,22),("hinoki","a",12.5,17),("conifer","mix",14,21),
     ("broadleaf","a",13,19),("broadleaf","b",14,20),("mixed","a",14,22)]


def write_mesh(m,name,species,distance):
    verts=np.asarray(m.verts,dtype=np.float32); faces=np.asarray(m.faces,dtype=np.int32)
    uvs=np.asarray(m.uvs,dtype=np.float32)
    path=EXPORT/(name+".npz")
    # 同じUVの幹面は共有してスムーズに。枝葉は逆面と頂点を共有しない。
    for i,part in enumerate(m.parts):
        part["face_end"]=m.parts[i+1]["face_start"] if i+1<len(m.parts) else len(faces)
    vv,tt,ff=[],[],[]
    for part in m.parts:
        shared={}
        for face in faces[part["face_start"]:part["face_end"]]:
            ids=[]
            for old in face:
                key=tuple(verts[old])+tuple(uvs[old]) if part["name"] not in ("foliage","panel") else int(old)
                if key not in shared:
                    shared[key]=len(vv); vv.append(verts[old]); tt.append(uvs[old])
                ids.append(shared[key])
            ff.append(ids)
    np.savez_compressed(path,verts=np.array(vv),uvs=np.array(tt),faces=np.asarray(ff,dtype=np.int32))
    dims=np.ptp(verts,axis=0)
    near_dist = distance=="near"
    return dict(name=name,species=species,distance=distance,height_m=round(float(verts[:,2].max()),3),
                width_m=round(float(dims[0]),3),depth_m=round(float(dims[1]),3),
                triangles=len(faces),npz=path.name,texture="../"+MIDDLE,texture_dark="../"+DARK,
                parts=m.parts, rules=("舗装端から15 m以内。+Xを路側へ向けて揺らし、0.8〜1.2倍。" if near_dist else RULE))


def geometry():
    models=[]
    for sp in SPECIES:
        for i,variant in enumerate("ab"):
            models.append(write_mesh(near(sp,i),f"near_{sp}_{variant}",sp,"near"))
    for i,(sp,variant,w,h) in enumerate(FAR):
        m=Mesh(); m.part("panel")
        m.card(np.array((0,0,-.3)),np.array((0,0,h)),-math.pi/2,w/2,w/2,far_rect(i))
        entry=write_mesh(m,f"far_{sp}_{variant}",sp,"far")
        entry["species_members"]={"sugi":["sugi"],"hinoki":["hinoki"],
                                  "broadleaf":["broadleaf"],"conifer":["sugi","hinoki"],
                                  "mixed":list(SPECIES)}[sp]
        models.append(entry)
    info=dict(units="m",up="+Z",origin="地表の根元中心。最下端は z=-0.30 m。",
              height_definition="地表から頂点の最大Z。地下0.30 mを含まない。幅はローカルX方向。",uv="Blender bottom-left",
              material=dict(shader="ksPerPixelAT",alpha_ref=ALPHA,texture="../"+MIDDLE,
                            alternate="../"+DARK,add_backfaces=False,backface_culling=True),
              placement=RULE,models=models)
    write_json(EXPORT/"models.json",info)
    return models


def save_blend():
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.unit_settings.system="METRIC"
    mat=bpy.data.materials.new("ksPerPixelAT • 中間"); mat.use_nodes=True
    mat.use_backface_culling=True
    nodes=mat.node_tree.nodes
    shader=next(n for n in nodes if n.type=="BSDF_PRINCIPLED")
    shader.inputs["Roughness"].default_value=.94
    shader.inputs["Specular IOR Level"].default_value=.10
    tex=nodes.new("ShaderNodeTexImage"); tex.image=bpy.data.images.load(str(DEST/MIDDLE))
    tex.image.filepath="//"+MIDDLE
    test=nodes.new("ShaderNodeMath"); test.operation="GREATER_THAN"; test.inputs[1].default_value=ALPHA
    links=mat.node_tree.links
    links.new(tex.outputs["Color"],shader.inputs["Base Color"])
    links.new(tex.outputs["Alpha"],test.inputs[0]); links.new(test.outputs[0],shader.inputs["Alpha"])
    items=json.loads((EXPORT/"models.json").read_text())["models"]
    for i,entry in enumerate(items):
        name=entry["name"]
        collection=bpy.data.collections.new(name); scene.collection.children.link(collection)
        anchor=bpy.data.objects.new(name,None); collection.objects.link(anchor)
        anchor.location=((i%6)*18,(i//6)*28,0)
        with np.load(EXPORT/entry["npz"]) as arrays:
            model={k:arrays[k] for k in ("verts","uvs","faces")}
            for part in entry["parts"]:
                role=part["name"]
                is_wood=role=="trunk" or role.startswith(("branch","stub"))
                fs=model["faces"][part["face_start"]:part["face_end"]]
                unique={}; verts=[]; faces=[]
                for f in fs:
                    tri=[]
                    for v in f:
                        co=model["verts"][v]
                        key=tuple(co) if is_wood else int(v)
                        if key not in unique:
                            unique[key]=len(verts); verts.append(tuple(co))
                        tri.append(unique[key])
                    faces.append(tri)
                mesh=bpy.data.meshes.new(name+"."+role); mesh.from_pydata(verts,[],faces)
                mesh.update()
                obj=bpy.data.objects.new(role+"."+name,mesh); collection.objects.link(obj); obj.parent=anchor
                mesh.materials.append(mat)
                layer=mesh.uv_layers.new(name="UVMap")
                for poly,indices in zip(mesh.polygons,fs):
                    poly.use_smooth=is_wood
                    for loop,v in zip(poly.loop_indices,indices):
                        layer.data[loop].uv=model["uvs"][v]
    bpy.context.preferences.filepaths.save_version=0
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(DEST/"trees_light.blend"),relative_remap=False)


def checks():
    command=[BLENDER,"-b","--factory-startup","--offline-mode",str(DEST/"trees_light.blend"),"--python-exit-code","1","--python",
             str(ROOT/".agents/skills/yabitsu-object-modeling/scripts/check_asset.py"),"--",
             "--closed","trunk.","--closed","branch_","--closed","stub_","--min-volume",".001"]
    for model in json.loads((EXPORT/"models.json").read_text())["models"]:
        for part in model["parts"]:
            command+=["--require",part["name"]+"."+model["name"]]
    CHECK.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(command+["--out",str(CHECK)],check=True,stdout=subprocess.DEVNULL)
    audit=json.loads(CHECK.read_text())
    parts={p["name"]:p for p in audit["objects"]}
    assert audit["ok"]
    for m in json.loads((EXPORT/"models.json").read_text())["models"]:
        result=[parts[p["name"]+"."+m["name"]] for p in m["parts"]]
        assert sum(p["triangles"] for p in result)==m["triangles"]
        assert m["triangles"] <= (250 if m["distance"]=="near" else 4)
        lo=np.min([p["bounds_world"][0] for p in result],axis=0)
        hi=np.max([p["bounds_world"][1] for p in result],axis=0)
        assert np.allclose([hi[2],hi[0]-lo[0],lo[2]],[m["height_m"],m["width_m"],-.3],atol=.001)
        with np.load(EXPORT/m["npz"]) as a:
            f=a["faces"]; v=a["verts"]; t=a["uvs"]
            assert v.shape[1]==3 and t.shape==(len(v),2) and f.shape==(m["triangles"],3)
            assert np.isfinite(v).all() and t.min()>=0 and t.max()<=1 and f.min()>=0 and f.max()<len(v)
            assert np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1).min()>1e-7
            for p in m["parts"]:
                if p["name"] in ("foliage","panel"):
                    for front in range(p["face_start"],p["face_end"],4):
                        for j in range(2):
                            one=f[front+j]; other=f[front+j+2][[0,2,1]]
                            assert set(one).isdisjoint(other) and np.array_equal(v[one],v[other])
                            assert np.array_equal(t[one],t[other])


def main():
    DEST.mkdir(parents=True,exist_ok=True); EXPORT.mkdir(exist_ok=True)
    paint()
    models=geometry()
    write_json(DEST/"asset.json",dict(
        name="trees_light",units="metres",up="+Z",origin="根元・地表、地下に0.30 m",
        source="trees_light.blend",generator="blender_scripts/build_trees_light.py",
        model_index="export/models.json",textures=[MIDDLE,DARK],default=MIDDLE,dimensions=[SIZE,SIZE],
        layout="左1536 pxに林6種、右512 pxに3樹種×2種の枝葉と3種の樹皮。同じUVで2色案を排他使用。",
        attribution="自作。近景は依頼55の自作の枝葉と樹皮を編み直した。遠景は trees_light_groves.py で作画。",
        reference_dimensions="旧CSPのtree.iniと現地写真の樹形を参考にした推定寸法。個体の実測ではない。",
        references=["export/compare/route70_inventory_20261030/images/sv/R70-"+s+".jpg"
                    for s in ("0364_f","0365_f","0366_f","0912_f","0366_b","0551_f")],
        limitations="枝・葉の細部と樹冠の背後は推定。ゲームの光、画質設定、縮小時の見え方とFPSは未確認。",
        game_integration=False, triangle_count_includes_backfaces=True,
        model_count=len(models)))
    subprocess.run([BLENDER,"-b","--factory-startup","--offline-mode","--python-exit-code","1","--python",str(Path(__file__).resolve()),
                    "--","--blend"],check=True)
    checks()


if __name__=="__main__":
    if "--blend" in sys.argv:
        save_blend()
    else:
        main()
