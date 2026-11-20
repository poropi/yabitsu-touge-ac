"""丹沢の山林を想定した原点付き樹木ライブラリーを再生成する。

ルートで .venv/bin/python blender_scripts/build_mountain_trees.py
画像は Pillow による自作の線・面のみ。再生成後、同じ .blend を別プロセスで読み直して検査する。
"""
from pathlib import Path
import json
import math
import random
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/landmarks/trees_mountain"
CHECKS = ROOT / "export/compare/trees_mountain/work/model_checks"
BLEND = OUT / "trees_mountain.blend"
BLENDER = "/opt/homebrew/bin/blender"
TYPES = {
    "sugi_forest": ("スギ・林内", [22.0, 23.1], [.23, .21], "sugi", "sugi"),
    "sugi_edge": ("スギ・道ぎわ", [18.0, 18.8], [.26, .28], "sugi", "sugi"),
    "hinoki": ("ヒノキ", [16.8, 17.6], [.28, .25], "hinoki", "hinoki"),
    "oak": ("コナラ型・高木", [14.2, 15.4], [.38, .43], "oak", "oak"),
    "roadside": ("落葉広葉樹・道ぎわ", [7.3, 8.3], [.19, .21], "oak", "oak"),
    "shrub": ("低木と下草", [1.7, 2.2], [.026, .031], "understory", "oak"),
}


def paint_textures():
    from PIL import Image, ImageDraw, ImageEnhance
    import numpy as np

    OUT.mkdir(parents=True, exist_ok=True)
    for idx, kind in enumerate(("sugi", "hinoki", "oak")):
        rng = np.random.default_rng(901 + idx)
        size = 512
        yy, xx = np.mgrid[:size, :size]
        def noise(w, h):
            # 周縁を折り返して上下・左右の継ぎ目を合わせる。
            small = rng.normal(0, 1, (h, w))
            small = np.pad(small, 1, mode="wrap")
            f = Image.fromarray(small.astype("float32"))
            return np.asarray(f.resize((size, size), Image.Resampling.BICUBIC,
                                       box=(1, 1, w+1, h+1)))
        coarse = noise(24, 5)
        fibres = noise(125, 14)
        warp = noise(8, 6) * .82
        fissure = np.abs(np.sin(2 * math.pi * (xx * (22+idx*4) / size) + warp + .8*coarse))
        grooves = np.exp(-fissure * (5 if kind == "oak" else 8)) * np.clip(.7 + fibres*.35, .1, 1.4)
        base = {"sugi": [96,73,60], "hinoki":[112,86,67], "oak":[103,100,85]}[kind]
        value = (coarse*10 + fibres*6 - grooves*30 + noise(200, 32)*3
                 + rng.normal(0, 2.8, (size,size)))
        rgb = np.clip(np.array(base)[None,None,:] + value[:,:,None], 0, 255).astype("uint8")
        Image.fromarray(rgb).save(OUT / f"bark_{kind}.png")

    size = 1024
    for index, species in enumerate(("sugi", "hinoki", "oak", "understory")):
        rng = random.Random(4302 + index)
        im = Image.new("RGBA", (size,size), (59,79,41,0))
        draw = ImageDraw.Draw(im)
        def tone(base, d=0):
            return tuple(max(0, min(255, round(c+d))) for c in base) + (255,)
        def pos(x,y):
            return (x*size, (1-y)*size)
        def line(a,b,col,w):
            draw.line([pos(*a),pos(*b)], fill=col, width=max(1, round(size*w)))
        def twig(t):
            return .5 + .024*math.sin(t*4.4), .025+t*.93
        def leaf(base, tip, width, color, serrated=False):
            x,y=base; dx=tip[0]-x; dy=tip[1]-y
            length=math.hypot(dx,dy)
            normal=(-dy/length*width,dx/length*width)
            def edge(t,f):
                return pos(x+dx*t+normal[0]*f,y+dy*t+normal[1]*f)
            fractions=[(.12,.40),(.25,.67),(.42,.93),(.63,1.0),(.8,.73),(.91,.37)]
            sides=[]
            for side in (1,-1):
                points=[edge(0,0)]
                for t,f in fractions:
                    if serrated:
                        points.append(edge(t-.03,side*(f-.08)))
                    points.append(edge(t,side*f))
                points += [edge(1,0)]
                sides.append(points)
            draw.polygon(sides[0]+sides[1][::-1],fill=tone(color,-4))
            draw.polygon(sides[0]+[edge(.48,.04)],fill=tone(color,4))
            if serrated:
                line(base,(x+dx*.82,y+dy*.82),tone(color,11),.0015)
        for n in range(24):
            line(twig(n/24),twig((n+1)/24),tone((83,78,48)),.013*(1-n/30))
        if species == "sugi":
            # スギの小枝は平たい一枚の扇にならない。中心軸から何束かに分け、
            # 短枝それぞれに前屈する針を多方向からかぶせる。
            def brush(a,b,young=False):
                dx=b[0]-a[0]; dy=b[1]-a[1]; ln=math.hypot(dx,dy)
                ux=dx/ln; uy=dy/ln; nx=-uy; ny=ux
                bend=rng.uniform(-.12,.12)*ln
                def point(t):
                    bow=math.sin(t*math.pi)*bend
                    return (a[0]+t*dx+bow*nx, a[1]+t*dy+bow*ny)
                for m in range(8):
                    line(point(m/8),point((m+1)/8),tone((72,81,57)),.0055)
                steps=max(13,int(ln*175))
                for n in range(steps):
                    t=(n+rng.uniform(.03,.86))/steps
                    start=point(t); side=(-1 if n%2 else 1)
                    for _ in range(3):
                        length=rng.uniform(.024,.045)*(1-.43*t)
                        width=side*length*rng.uniform(.42,1.10)
                        forward=length*rng.uniform(.46,.98)
                        mid=(start[0]+ux*forward*.19+nx*width*.68,
                             start[1]+uy*forward*.19+ny*width*.68)
                        tip=(start[0]+ux*forward+nx*width, start[1]+uy*forward+ny*width)
                        c=tone((61,85,61),rng.uniform(-15,16)+11*t+(5 if young else 0))
                        line(start,mid,c,.0050)
                        leaf(mid,tip,.0021,c[:3])
            for side in (-1,1):
                first=rng.uniform(.025,.08)
                for j in range(7):
                    t=first+j*.108+rng.uniform(-.028,.033)
                    a=twig(t)
                    length=rng.uniform(.29,.405)*(1-.45*t)
                    b=(a[0]+side*length, a[1]+length*rng.uniform(.48,.78))
                    brush(a,b)
                    for k in range(7):
                        f=.10+k*.126+rng.uniform(-.03,.027)
                        st=(a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f)
                        spread=length*(1-f*.68)*rng.uniform(.34,.82)
                        # 遠近方向の短枝は長さが違って見える。
                        end=(st[0]+side*spread*rng.uniform(.26,.93),
                             st[1]+spread*rng.uniform(-.43,1.16))
                        brush(st,end,k>2)
            brush(twig(.61),twig(.96),True)
            for side in (-1,1):
                brush(twig(.75),(.50+side*.085,.961),True)
            # 奥の小枝も同じ軸からわずかに違う向きへ伸びる。密な枝束はミップで
            # 縮小しても緑の芯が残り、枝束間の大きな空隙は透けたままになる。
            back=im.rotate(-3.5,Image.Resampling.BICUBIC,center=(.5*size,.92*size))
            back=ImageEnhance.Brightness(back).enhance(.83)
            back.alpha_composite(im)
            im=back
        elif species == "hinoki":
            # 間を空けた扁平な扇を重ね、鱗片の先端が粒状に見えるようにする。
            base=(73,97,66)
            pairs=15
            for j in range(pairs):
                t=.055+j*.87/pairs
                for side in (-1,1):
                    start=twig(t+rng.uniform(-.026,.018))
                    wid=(.095+.33*math.sin(math.pi*(t**.72))**.65)*(1-.24*t)
                    wid*=rng.uniform(.74,1.08)
                    rise=min(.95-start[1],.10+.10*(1-t))
                    def lateral(u):
                        return start[0]+side*wid*u, start[1]+rise*(u*.7+.3*u*u)
                    col=tone(base,rng.uniform(-10,12))
                    line(start,lateral(1),col,.011)
                    for k in range(14):
                        u=(k+rng.uniform(.15,.65))/14
                        st=lateral(u)
                        length=(.018+.055*math.sin(u*math.pi)**.55)*(1-.42*t)
                        for sign in (-1,1):
                            end=(st[0]+side*length*(.28+.32*u),st[1]+sign*length*rng.uniform(.83,1.16))
                            c=tone(base,rng.uniform(-12,19)+u*8)
                            leaf(st,end,.010,c[:3])
                            for f in (.3,.53,.75):
                                px=st[0]+(end[0]-st[0])*f; py=st[1]+(end[1]-st[1])*f
                                leaf((px,py),(px+side*.014,py+sign*.015),.0065,c[:3])
        else:
            for j in range(10):
                t=.065+j*.084
                for side in (-1,1):
                    start=twig(t+rng.uniform(-.02,.02))
                    wid=(.12+.25*math.sin(math.pi*t)**.72)*(1-.23*t)*rng.uniform(.85,1.)
                    rise=min(.92-start[1],.10+.10*(1-t))*rng.uniform(.8,1.1)
                    def lateral(u):
                        return (start[0]+side*wid*u, start[1]+rise*(.66*u+.34*u*u))
                    line(start,lateral(1),tone((91,86,53)),.006)
                    for k in range(6):
                        u=(k+1)/6; st=lateral(u); sign=(-1)**(k+j)
                        ll=(.085+.022*math.sin(u*math.pi))*(1-.33*t)*rng.uniform(.83,1.21)
                        tip=(st[0]+side*ll*rng.uniform(.50,.93),st[1]+sign*ll*rng.uniform(.52,.91))
                        shade=rng.uniform(-17,20)+9*u
                        color=tone((79,103,50) if species=="oak" else (84,110,49),shade)
                        leaf(st,tip,ll*rng.uniform(.21,.33),color[:3],True)
            # 下生えの同じアトラスの右半分に、別UVで使うシダ葉と細長い草を描く。
            if species=="understory":
                shrub=im.resize((500,1000),Image.Resampling.LANCZOS)
                im=Image.new("RGBA",(size,size),(61,81,39,0))
                im.paste(shrub,(4,12))
                draw=ImageDraw.Draw(im)
                def axis(t): return (.754+.023*math.sin(t*3),.02+.94*t)
                for j in range(22):
                    line(axis(j/22),axis((j+1)/22),tone((105,113,61)),.003)
                for j in range(13):
                    t=.08+j*.068
                    for side in (-1,1):
                        f=t+rng.uniform(-.014,.014)
                        a=axis(f)
                        width=(.024+.165*math.sin(math.pi*f)**.75)*(1-.32*f)*rng.uniform(.87,1.06)
                        b=(a[0]+side*width,a[1]+(.04+.038*(1-f))*rng.uniform(.82,1.2))
                        line(a,b,tone((85,103,52)),.0025)
                        for k in range(8):
                            u=.12+k*.115
                            x=a[0]+(b[0]-a[0])*u;y=a[1]+(b[1]-a[1])*u
                            for sign in (-1,1):
                                ln=(.018+.012*math.sin(math.pi*u))*(1-.49*u)*(1-.38*f)
                                dx=side*ln*(.28+.35*u)
                                dy=sign*ln*rng.uniform(.72,.99)
                                leaf((x,y),(x+dx,y+dy),ln*.18,
                                     tone((64,90,45),rng.uniform(-9,15))[:3])
        im.save(OUT / f"leaf_{species}.png", optimize=False)


if __name__ == "__main__" and "bpy" not in sys.modules and not any(flag in sys.argv for flag in ("--inside-blender","--check-library")):
    paint_textures()
    subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                    str(Path(__file__).resolve()), "--", "--inside-blender"], cwd=ROOT, check=True)
    subprocess.run([BLENDER, "-b", "--factory-startup", str(BLEND), "--python-exit-code", "1",
                    "--python", str(Path(__file__).resolve()), "--", "--check-library"], cwd=ROOT, check=True)
    sys.exit()

import bpy
from mathutils import Vector

Z = Vector((0,0,1))
TAU = math.tau


class Geometry:
    def __init__(self):
        self.v=[]; self.faces=[]; self.uv=[]

    def face(self, ids, uv):
        self.faces.append(ids)
        self.uv.extend(uv)

    def tube(self, points, radii, sides, circumference_uv=1):
        direction=(points[-1]-points[0]).normalized()
        across=direction.cross(Vector((0,1,0)))
        if across.length < .01:
            across=direction.cross(Vector((1,0,0)))
        across.normalize()
        base=len(self.v); distance=0.
        distances=[0.]
        for i, (p,r) in enumerate(zip(points,radii)):
            tangent=(points[min(i+1,len(points)-1)]-points[max(i-1,0)]).normalized()
            a=(across-tangent*across.dot(tangent)).normalized(); b=tangent.cross(a)
            if i:
                distance+=(p-points[i-1]).length
                distances.append(distance)
            for j in range(sides):
                angle=TAU*j/sides
                self.v.append(p+r*(math.cos(angle)*a+math.sin(angle)*b))
        for i in range(len(points)-1):
            for j in range(sides):
                k=(j+1)%sides; offset=base+i*sides
                u=j/sides*circumference_uv; u1=(j+1)/sides*circumference_uv
                self.face([offset+j,offset+k,offset+k+sides,offset+j+sides],
                          [(u,distances[i]*.55),(u1,distances[i]*.55),
                           (u1,distances[i+1]*.55),(u,distances[i+1]*.55)])
        for i in (0,len(points)-1):
            order=list(range(sides))
            if i==0: order.reverse()
            self.face([base+i*sides+j for j in order],
                      [(.5+.45*math.cos(TAU*j/sides),.5+.45*math.sin(TAU*j/sides)) for j in order])

    def card(self, a, b, halfwidth, tilt, bow, uv_range=(0,1), simple=False):
        tangent=(b-a).normalized()
        side=tangent.cross(Z)
        if side.length<.01: side=Vector((1,0,0))
        side.normalize()
        up=side.cross(tangent).normalized()
        cross=side*math.cos(tilt)+up*math.sin(tilt)
        base=len(self.v); u,v=uv_range
        if simple:
            self.v.extend([a-cross*halfwidth,a+cross*halfwidth,b+cross*halfwidth,b-cross*halfwidth])
            self.face([base+i for i in range(4)],[(u,0),(v,0),(v,1),(u,1)])
        else:
            # 縦の折れと中ほどの弓なりで板端の一直線を崩す。
            for f in (0,.52,1):
                center=a.lerp(b,f)+up*bow*math.sin(math.pi*f)
                width=halfwidth*(.88+.12*math.sin(math.pi*f))
                self.v.extend([center-cross*width,center+up*.13*halfwidth,center+cross*width])
            for j in (0,1):
                for i in (0,1):
                    n=base+j*3+i
                    self.face([n,n+1,n+4,n+3],[(u+(v-u)*i*.5,j*.5),(u+(v-u)*(i+1)*.5,j*.5),
                                                    (u+(v-u)*(i+1)*.5,(j+1)*.5),(u+(v-u)*i*.5,(j+1)*.5)])

    def mesh(self,collection,name,material,smooth):
        mesh=bpy.data.meshes.new(name)
        mesh.from_pydata(self.v,[],self.faces)
        mesh.materials.append(material)
        uv=mesh.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv",[coord for point in self.uv for coord in point])
        for p in mesh.polygons: p.use_smooth=smooth
        mesh.update()
        obj=bpy.data.objects.new(name,mesh)
        collection.objects.link(obj)
        return sum(len(f)-2 for f in self.faces)


def make_material(prefix, name):
    mat=bpy.data.materials.new(f"{prefix}_{name}")
    mat.use_nodes=True
    tree=mat.node_tree
    shader=next(n for n in tree.nodes if n.type=="BSDF_PRINCIPLED")
    shader.inputs["Roughness"].default_value=.9
    shader.inputs["Specular IOR Level"].default_value=.16
    tex=tree.nodes.new("ShaderNodeTexImage"); tex.location=(-530,200)
    image=bpy.data.images.load(str(OUT/f"{prefix}_{name}.png"),check_existing=True)
    tex.image=image
    tree.links.new(tex.outputs["Color"],shader.inputs["Base Color"])
    if prefix=="leaf":
        clip=tree.nodes.new("ShaderNodeMath"); clip.operation="GREATER_THAN"
        clip.inputs[1].default_value=.40; clip.location=(-260,-100)
        tree.links.new(tex.outputs["Alpha"],clip.inputs[0])
        tree.links.new(clip.outputs[0],shader.inputs["Alpha"])
        mat.use_backface_culling=False
    image.pack()
    return mat


def horizontal(angle):
    return Vector((math.cos(angle),math.sin(angle),0))


def conifer(kind,h,diam,rng,low):
    wood=Geometry(); leaf=Geometry(); flat=kind=="hinoki"
    sway=horizontal(rng.uniform(0,TAU))*rng.uniform(.13,.4)
    def axis(z): return Z*z+sway*max(0,z/h)**2
    heights=[-.3,0,.42,1.3,.28*h,.52*h,.74*h,.90*h,h]
    diameters=[diam*1.45,diam*1.56,diam*1.18,diam,diam*.85,diam*.68,diam*.49,diam*.27,.012]
    wood.tube([axis(z) for z in heights],[d*.5 for d in diameters],5 if low else 10)
    start={"sugi_forest":.665,"sugi_edge":.27,"hinoki":.40}[kind]
    levels={"sugi_forest":13,"sugi_edge":19,"hinoki":16}[kind]
    phase=rng.uniform(0,TAU)
    master=rng
    for i in range(levels):
        f=i/levels
        level=start+(1-start)*f
        n=6 if flat else 8
        for j in range(n):
            rng=random.Random(master.getrandbits(48))
            angle=phase+i*2.40+j*TAU/n+rng.uniform(-.19,.19)
            z=h*(level+rng.uniform(-.018,.016))
            base=axis(z); run=horizontal(angle); aside=horizontal(angle+math.pi/2)
            # 道路側 (+X) には低い枝が残る。裏側にも短い枯枝を残す。
            edge=kind=="sugi_edge"
            if edge:
                openness=(1+run.x)*.5
                # 林側に回り込むにつれて生枝の下端を徐々に引き上げる。
                branch_floor=.27+.35*(1-openness)**1.18+rng.uniform(-.07,.055)
                if level < branch_floor:
                    if not low and j%3==0 and i%2==0:
                        p=base+run*rng.uniform(.14,.55)-Z*.16
                        wood.tube([base,p],[.025,.007],3)
                    continue
            length=(h*(.19 if flat else .145))*(1-f)**(.62 if flat else .75)
            length*=rng.uniform(.74,1.25)
            if edge:
                taper=min(1,max(.15,(level-branch_floor+.047)/.155))
                length*=taper*(.85+.68*openness)
            # 下方の古枝は外まで下垂し、上の若枝は少し上を向く。
            sag=length*(.24 if flat else .33)*(1-f)
            lift=length*(.01+f*.23)
            tip=base+run*length+Z*(lift-sag)
            mid=base+run*length*.46-Z*sag
            if low:
                if i%2==0 and j%3==0:
                    wood.tube([base,mid],[max(.014,diam*.14*(1-f)),.008],3)
                if i%2==0 or (i==levels-1):
                    leaf.card(base,tip,length*(.51 if flat else .49),rng.uniform(-.69,.69),0,simple=True)
                continue
            wood.tube([base,mid,tip],[max(.012,diam*.14*(1-f)),.014,.004],4)
            for k in range(3 if flat else 2):
                u=.09+k*(.23 if flat else .33)
                origin=base.lerp(mid,u/.46) if u<.46 else mid.lerp(tip,(u-.46)/.54)
                for sign in (-1,1):
                    reach=length*(.49 if flat else .64)*(1-k*.25)*rng.uniform(.86,1.12)
                    direction=(run*.81+aside*(.58*sign)).normalized()
                    target=origin+direction*reach+Z*(lift*.24-sag*.81+rng.uniform(-.16,.11))
                    leaf.card(origin,target,reach*(.41 if flat else .47),
                              sign*rng.uniform(.18,.84 if flat else 1.04),-reach*.07)
            a=mid.lerp(tip,.16)
            leaf.card(a,tip+run*.08, length*.24,rng.uniform(-.85,.85),-.06*length)
    # 頂芽の先まで葉を連続させる。
    for j in range(6):
        a=axis(h*.958)
        b=axis(h)+horizontal(j*TAU/6)*.08
        leaf.card(a,b,h*.009,j*.8,0,simple=low)
    return wood,leaf


def deciduous(kind,h,diam,rng,low):
    wood=Geometry(); leaves=Geometry(); road=kind=="roadside"
    downhill=Vector((1,0,0)) if road else horizontal(rng.uniform(0,TAU))
    cross=horizontal(math.atan2(downhill.y,downhill.x)+math.pi/2)
    lean=h*(rng.uniform(.15,.23) if road else rng.uniform(.015,.075))
    def axis(z):
        t=max(0,z/h)
        return Z*z+downhill*lean*t+cross*(.018*h*math.sin(5*t)*t)
    top=h*.60
    trunk_z=[-.3,0,.40,1.30,h*.30,h*.45,top]
    if road and trunk_z[3] >= trunk_z[4]:
        trunk_z.pop(4)
    radius=lambda z: max(.02,diam*.5*(1-.78*max(0,z-1.3)/(top-1.3)))
    rr=[radius(z) for z in trunk_z]; rr[:3]=[diam*.80,diam*.89,diam*.62]
    wood.tube([axis(z) for z in trunk_z],rr,6 if low else 11)
    master=rng
    offset=master.uniform(0,TAU)
    leaders=6 if road else 7
    for i in range(leaders):
        r=random.Random(master.getrandbits(48))
        ang=offset+i*2.399+r.uniform(-.33,.33)
        run=horizontal(ang)
        at=r.uniform(.22,.42) if i<leaders-1 else .60
        base=axis(at*h)
        # 広葉樹は主幹を中腹で複数の伸長軸に分ける。
        tallness=1 if i==leaders-1 else r.uniform(.76,.95)
        radius_fraction=(.24 if road else .20)*r.uniform(.86,1.23)
        radial=run*radius_fraction*h
        if road: radial+=downhill*h*(.12+.11*max(0,run.x))
        if i==leaders-1:
            radial*=.44
        crown=base+radial+Z*h*(tallness-at)*.84
        elbow=base+(crown-base)*.47+cross*r.uniform(-.04,.04)*h-run*h*.038
        elbow.z-=.02*h
        wood.tube([base,elbow,crown],[diam*.29*(1.3-at),diam*.185,diam*.042],4 if low else 7)
        for j in range(5):
            t=.21+j*.17
            st=base.lerp(elbow,t/.47) if t<.47 else elbow.lerp(crown,(t-.47)/.53)
            sign=(-1 if j%2 else 1)
            angle=ang+sign*r.uniform(.42,1.30)
            shoot=horizontal(angle)
            if road: shoot=(shoot+downhill*.32).normalized()
            length=h*(.255 if road else .202)*(1-.39*t)*r.uniform(.79,1.18)
            reach=st+shoot*length+Z*length*(.42 if road else .73)*r.uniform(.7,1.26)
            arc=st.lerp(reach,.55)-Z*length*.075+horizontal(angle+math.pi/2)*length*.08*sign
            if low:
                if j%2==0 and (j!=4 or i%2==0):
                    wood.tube([st,reach],[diam*.13,.007],3)
            else:
                wood.tube([st,arc,reach],[diam*.13,diam*.073,.009],5)
            # 枝の途中にも葉の塊を置く。塊同士には空を透かす不揃いのすき間を残す。
            for k in range(3):
                at=.33+.27*k
                root=st.lerp(arc,at/.55) if at<.55 else arc.lerp(reach,(at-.55)/.45)
                if low:
                    root=st.lerp(reach,at)
                side_angle=angle+(-1 if k%2 else 1)*r.uniform(.43,1.21)
                axis3=horizontal(side_angle)
                span=(h*.125 if road else h*.112)*r.uniform(.76,1.19)
                apex=root+axis3*span*.8+Z*span*r.uniform(.19,.65)
                if not low:
                    wood.tube([root,apex],[diam*.038,.0035],4)
                # 枝先ごとに5方向へ枝葉を分ける。
                for p in range(5):
                    theta=side_angle+(p-2)*.65+r.uniform(-.33,.33)
                    unit=horizontal(theta)
                    rise=r.uniform(-.18,.71)
                    end=apex+unit*span*r.uniform(.66,1.20)+Z*span*rise
                    origin=root.lerp(apex,r.uniform(.14,.79))
                    tilt=r.uniform(-.90,.90)
                    width=span*r.uniform(.43,.60)
                    if low:
                        if p in (0,3):
                            leaves.card(root, end, width*1.62, tilt,0,simple=True)
                    else:
                        leaves.card(origin,end,width,tilt,span*.09)
    return wood,leaves


def understory(h,diam,rng,low):
    wood=Geometry(); leaves=Geometry()
    master=rng
    axis=horizontal(master.uniform(0,TAU)); cross=axis.cross(Z)
    for i in range(7):
        r=random.Random(master.getrandbits(48))
        root=axis*(h*r.uniform(-.63,.63))+cross*(h*r.uniform(-.33,.33))
        if i==0: root=Vector((0,0,0))
        stalk=h*(1 if i==0 else r.uniform(.47,.90))
        base=root-Z*.3
        bent=horizontal(r.uniform(0,TAU))*stalk*.25
        st=root+Z*stalk*.39+bent*.23; tip=root+bent+Z*.85*stalk
        wood.tube([base,st,tip],[diam*.6,diam*.42,.005],4 if low else 5)
        for j in range(6):
            az=r.uniform(-.33,.33)+j*2.399
            shoot=horizontal(az)
            size=stalk*r.uniform(.34,.59)
            start=st.lerp(tip,.14+j*.15)
            end=start+shoot*size+Z*size*r.uniform(.24,.81)
            if not low and j%2==0:
                wood.tube([start,end],[diam*.26,.003],3)
            for k in range(3):
                a=start.lerp(end,k*.16)
                b=end+horizontal(az+(k-1)*.9)*size*.44+Z*size*.30
                tilt=r.uniform(-1.06,1.06)
                if not low or k!=1:
                    leaves.card(a,b,size*.36*(1.30 if low else 1),tilt,.07*size,(0,.5),simple=low)
    for i in range(12):
        r=random.Random(master.getrandbits(48)); center=axis*(h*r.uniform(-1.,1.))+cross*(h*r.uniform(-.52,.52))
        for j in range(7):
            direction=horizontal(j*TAU/7+r.uniform(-.45,.45))
            size=h*r.uniform(.23,.48)
            end=center+direction*size+Z*size*r.uniform(.70,1.31)
            if not low or j%2==0:
                leaves.card(center,end,size*.23,r.uniform(-.9,.9),size*.18,(.5,1),simple=(low or j%3==0))
    return wood,leaves


def bounds(vertices):
    return ([min(p[i] for p in vertices) for i in range(3)],
            [max(p[i] for p in vertices) for i in range(3)])


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene
    scene.unit_settings.system="METRIC"
    bpy.context.preferences.filepaths.save_version=0
    bark={key:make_material("bark",key) for key in ("sugi","hinoki","oak")}
    foliage={key:make_material("leaf",key) for key in ("sugi","hinoki","oak","understory")}
    # 検査用シーンには原点を共有する元コレクション。普段開くシーンは並べたインスタンス。
    scene.name="Source • local coordinates"
    display=bpy.data.scenes.new("Mountain tree library")
    display.unit_settings.system="METRIC"
    models=[]
    for index,(kind,(label,heights,diameters,leaf_key,bark_key)) in enumerate(TYPES.items()):
        for variant in range(2):
            identifier=f"{kind}_{'AB'[variant]}"
            seed=55100+index*73+variant*163
            h=heights[variant]; diam=diameters[variant]
            info={"id":identifier,"name":label+" "+"AB"[variant],"seed":seed,
                  "dbh_m":diam,"bark":f"bark_{bark_key}.png","leaf":f"leaf_{leaf_key}.png"}
            for level in (0,1):
                r=random.Random(seed)
                if kind=="shrub":
                    w,l=understory(h,diam,r,level==1)
                elif kind in ("roadside","oak"):
                    w,l=deciduous(kind,h,diam,r,level==1)
                else:
                    w,l=conifer(kind,h,diam,r,level==1)
                # 樹先を目標高へ。地下根の長さと胸高以下の幹の形は一定。
                positive=max(v.z for v in w.v+l.v)
                for v in w.v+l.v:
                    if v.z>1.3:
                        v.z=1.3+(h-1.3)*(v.z-1.3)/(positive-1.3)
                name=f"{identifier}_LOD{level}"
                group=bpy.data.collections.new(name)
                group["seed"]=seed; group["road_side"]="+X" if kind in ("sugi_edge","roadside") else ""
                scene.collection.children.link(group)
                n=w.mesh(group,name+"_wood",bark[bark_key],True)
                m=l.mesh(group,name+"_leaves",foliage[leaf_key],False)
                info[f"triangles_lod{level}"]=n+m
                info[f"leaf_triangles_lod{level}"]=m
                if level==0:
                    lo,hi=bounds(w.v+l.v)
                    info.update(height_m=round(hi[2],3),spread_m=round(max(hi[0]-lo[0], hi[1]-lo[1]),3),
                                spread_xy_m=[round(hi[i]-lo[i],3) for i in (0,1)],
                                bounds_m=[list(lo),list(hi)])
                    instance=bpy.data.objects.new(info["name"],None)
                    instance.instance_type="COLLECTION";instance.instance_collection=group
                    instance.empty_display_size=.01
                    instance.location=(index*14,variant*24,0)
                    display.collection.objects.link(instance)
            models.append(info)
    outfile={
        "asset_id":"trees_mountain","source":"trees_mountain.blend",
        "generator":"blender_scripts/build_mountain_trees.py","units":"m","up":"+Z",
        "origin":"根元中央、地下の根の端は Z=-0.30 m。低木は株群の中心。",
        "collections":"<id>_LOD0 / <id>_LOD1。元の座標は全てローカル原点。別シーン Mountain tree library は LOD0 の展示専用インスタンスのみ。",
        "road_facing":"+X（スギ道ぎわ、落葉広葉樹道ぎわ）",
        "textures":{"leaf_px":[1024,1024],"bark_px":[512,512],"authoring":"Pillowによる枝・葉の描画と数値合成の樹皮。全画像をblendに同梱。","alpha_test":.4,"double_sided":"葉はBlenderで両面。幾何の裏面は二重計上しない。"},
        "dimension_method":"樹高は地上高。枝張りはLOD0のX・Y軸の大きい方（葉の透明余白を含む）。胸高径は生成時の樹幹直径。低木は胸高径を定義しづらく、記載値は主な茎の設定径。",
        "references":"export/compare/route70_inventory_20261030/images/sv/ に保存済みの R70-0365_f,0366_f,0912_f,0364_f,0366_b,0363_f,0551_f,0552_b,0913_b。写真は観察用のみ。",
        "basis":"スギの目標は依頼記載の長野県林業総合センター21〜40年生の樹高18〜27m・胸高径14〜28cm。ほかの寸法、枝張り、株間は写真の見かけから推定。写真で針葉樹の種の同定ができず、第3種は指定どおりヒノキ。",
        "limits":"葉は枝葉の塊の板。個葉の立体、風揺れ、季節差、樹種同定、配置・ゲーム上の見え方は未検証。KN5/CSP変換とゲーム反映なし。",
        "models":models}
    (OUT/"asset.json").write_text(json.dumps(outfile,ensure_ascii=False,indent=2)+"\n")
    bpy.context.window.scene=display
    for scr in bpy.data.screens:
        for area in scr.areas:
            if area.type=="VIEW_3D":
                space=area.spaces.active; space.region_3d.view_location=(31,12,9)
                space.region_3d.view_distance=100
                space.region_3d.view_rotation=Vector((.4,1,-.33)).to_track_quat("-Z","Y")
                space.shading.type="MATERIAL"
    for image in bpy.data.images:
        if image.packed_file:
            image.filepath="//"+Path(image.filepath).name
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND),compress=True)
    print("MODEL COUNTS",[(m['id'],m['triangles_lod0'],m['triangles_lod1']) for m in models])


def check_library():
    bpy.context.window.scene=bpy.data.scenes["Source • local coordinates"]
    models=json.loads((OUT/"asset.json").read_text())["models"]
    checker=ROOT/".agents/skills/yabitsu-object-modeling/scripts/check_asset.py"
    CHECKS.mkdir(parents=True,exist_ok=True)
    for model in models:
        for lod in (0,1):
            name=f"{model['id']}_LOD{lod}"
            path=CHECKS/(name+".json")
            sys.argv=["check","--","--collection",name,"--require",name+"_leaves","--closed",name+"_wood",
                      "--min-height",str(model["height_m"]*.5),"--out",str(path)]
            runpy.run_path(str(checker),run_name="__main__")
            objects=json.loads(path.read_text())["objects"]
            assert sum(o["triangles"] for o in objects)==model[f"triangles_lod{lod}"]
            assert abs(max(o["bounds_world"][1][2] for o in objects)-model["height_m"])<.001
            assert min(o["bounds_world"][0][2] for o in objects)<-.29
            if lod==0:
                span=max(max(o["bounds_world"][1][j] for o in objects)
                         -min(o["bounds_world"][0][j] for o in objects) for j in (0,1))
                assert abs(span-model["spread_m"])<.001


if __name__ == "__main__":
    if "--check-library" in sys.argv:
        check_library()
    else:
        build()
