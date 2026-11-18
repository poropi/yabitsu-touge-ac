"""依頼08専用: 寸法指定メッシュ、自作カラーアトラス、上端v=0のNPZ。"""
from pathlib import Path
import json, math, hashlib, shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
ROOT=Path(__file__).resolve().parents[1]
CMP=ROOT/'export/compare/route70'
FONT='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'
PALETTE={'white':(217,218,206),'metal':(140,149,149),'dark':(43,49,47),'orange':(211,91,27),'yellow':(236,195,43),'red':(185,43,38),'blue':(30,84,143),'green':(60,92,75),'wood':(110,87,61),'rust':(118,72,40),'concrete':(138,141,130),'glass':(192,214,219),'black':(22,25,23),'cream':(198,191,154),'soil':(90,84,65),'pink':(179,116,136),'pole_orange':(211,91,27)}
KEYS=list(PALETTE)

def font(n):return ImageFont.truetype(FONT,max(8,int(n)))

def lettering(words,w=768,h=768,bg=(235,199,48),fg=(24,27,23),vertical=False,border=0):
    im=Image.new('RGBA',(w,h),(*bg,255));d=ImageDraw.Draw(im)
    if border:d.rectangle((border/2,border/2,w-border/2-1,h-border/2-1),outline=fg,width=border)
    margin=min(w,h)*.10
    if vertical:
        cols=words.split('\n'); cw=(w-2*margin)/len(cols)
        size=min(cw*.85,(h-2*margin)/max(map(len,cols)))
        for j,word in enumerate(cols):
            for i,ch in enumerate(word):d.text((w-margin-cw*(j+.5),h/2+(i-(len(word)-1)/2)*size),ch,font=font(size*.91),fill=fg,anchor='mm')
    else:
        rows=words.split('\n');size=min((h-2*margin)/len(rows)*.78,(w-2*margin)/max(max(map(len,rows))*.94,1))
        for j,word in enumerate(rows):
            f=font(size)
            while d.textlength(word,font=f)>w-2*margin:size-=1;f=font(size)
            d.text((w/2,h/2+(j-(len(rows)-1)/2)*size*1.24),word,font=f,fill=fg,anchor='mm')
    return im

class Mesh:
    def __init__(self):self.v=[];self.f=[];self.uv=[];self.components=[]
    def surface(self,name,verts,faces,color='metal',uv=None,rect=None):
        start=len(self.f)
        for face in faces:
            ids=[]
            for i in face:
                ids.append(len(self.v));self.v.append(verts[i])
                if uv is None:
                    k=KEYS.index(color);x=800+(k%2)*112;y=(k//2)*112
                    # 色パッチの中を広く使い、塗装の細かい汚れを残す。
                    q=[(.12,.12),(.88,.12),(.88,.88),(.12,.88)][len(ids)%4]
                    self.uv.append(((x+q[0]*108)/1024,(y+q[1]*108)/1024))
                else:
                    u,v=uv[i];r=rect or (8,8,768,768)
                    self.uv.append(((r[0]+u*(r[2]-r[0]))/1024,(r[1]+v*(r[3]-r[1]))/1024))
            for j in range(1,len(ids)-1):self.f.append((ids[0],ids[j],ids[j+1]))
        self.components.append({'name':name,'triangle_start':start,'triangle_count':len(self.f)-start})
    def box(self,n,loc,size,c='metal'):
        x,y,z=loc;a,b,h=np.asarray(size)/2
        vs=[(x+i*a,y+j*b,z+k*h) for i,j,k in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        self.surface(n,vs,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],c)
    def tube(self,n,a,b,r=.04,c='metal',seg=10,r2=None,caps=True):
        a,b=np.array(a,float),np.array(b,float);axis=b-a;axis/=np.linalg.norm(axis)
        u=np.cross(axis,[0,1,0] if abs(axis[1])<.9 else [1,0,0]);u/=np.linalg.norm(u);v=np.cross(axis,u)
        vs=[tuple(p+(r if end==0 or r2 is None else r2)*(math.cos(t)*u+math.sin(t)*v)) for end,p in enumerate([a,b]) for t in np.arange(seg)*2*math.pi/seg]
        fs=[(i,(i+1)%seg,(i+1)%seg+seg,i+seg) for i in range(seg)]
        if caps:fs += [tuple(range(seg-1,-1,-1)),tuple(range(seg,seg*2))]
        self.surface(n,vs,fs,c)
    def beam(self,n,a,b,w=.06,d=None,c='metal'):
        a,b=np.array(a,float),np.array(b,float);axis=b-a;axis/=np.linalg.norm(axis)
        u=np.cross(axis,[0,1,0] if abs(axis[1])<.9 else [1,0,0]);u/=np.linalg.norm(u);v=np.cross(axis,u)
        vs=[tuple(p+u*i*w/2+v*j*(d or w)/2) for p in [a,b] for i,j in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        self.surface(n,vs,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],c)
    def panel(self,n,x,y,z,w,h,c='white',rect=None,top=False):
        vs=[(x-w/2,y,z-h/2),(x+w/2,y,z-h/2),(x+w/2,y,z+h/2),(x-w/2,y,z+h/2)] if not top else [(x-w/2,y-h/2,z),(x+w/2,y-h/2,z),(x+w/2,y+h/2,z),(x-w/2,y+h/2,z)]
        self.surface(n,vs,[(0,1,2,3)],c,[(0,1),(1,1),(1,0),(0,0)],rect)
    def prism(self,n,outline,depth=.025,c='metal',face=False,y=0):
        N=len(outline);vs=[(x,yy+y,z) for yy in [-depth/2,depth/2] for x,z in outline]
        self.surface(n,vs,[tuple(range(N)),tuple(range(2*N-1,N-1,-1))]+[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)],c)
        if face:
            xx=np.array(outline);lo=xx.min(0);sp=np.ptp(xx,axis=0);verts=[(x,y-depth/2-.0003,z) for x,z in outline]
            self.surface(n+' 印刷面',verts,[tuple(range(N))],c,[((x-lo[0])/sp[0],1-(z-lo[1])/sp[1]) for x,z in outline])
    def disc(self,n,center,rx,rz,depth=.035,c='orange',seg=32,face=False):
        x,y,z=center;out=[(x+rx*math.cos(t),z+rz*math.sin(t)) for t in np.arange(seg)*2*math.pi/seg]
        self.prism(n,out,depth,c,face,y)
    def sphere(self,n,center,scale,c='white',seg=12,rings=5):
        x,y,z=center;sx,sy,sz=scale
        # 端部は三角扇。重複極点による退化三角形を避ける。
        vs=[(x,y,z-sz)]+[(x+sx*math.cos(t)*math.sin(p),y+sy*math.sin(t)*math.sin(p),z-sz*math.cos(p)) for p in np.arange(1,rings)*math.pi/rings for t in np.arange(seg)*2*math.pi/seg]+[(x,y,z+sz)]
        fs=[(0,1+(i+1)%seg,1+i) for i in range(seg)]
        for j in range(rings-2):
            for i in range(seg):a=1+j*seg+i;b=1+j*seg+(i+1)%seg;fs.append((a,b,b+seg,a+seg))
        last=len(vs)-1;start=1+(rings-2)*seg
        fs += [(last,start+i,start+(i+1)%seg) for i in range(seg)]
        self.surface(n,vs,fs,c)
    def transform(self,start,angle=0,offset=(0,0,0)):
        co,si=math.cos(angle),math.sin(angle)
        for i in range(start,len(self.v)):
            x,y,z=self.v[i];self.v[i]=(x*co-y*si+offset[0],x*si+y*co+offset[1],z+offset[2])

def atlas(art=None,changes=None,seed='0'):
    rng=np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8],16));a=np.zeros((1024,1024,4),np.uint8);a[:,:,:]=[110,115,106,255]
    im=Image.fromarray(a)
    if art is not None:im.paste(art.convert('RGBA').resize((760,760)),(8,8))
    ar=np.asarray(im).copy()
    for k,c in enumerate(KEYS):
        x=800+(k%2)*112;y=(k//2)*112;col=(changes or {}).get(c,PALETTE[c]);noise=rng.normal(0,2,(108,108,1));patch=np.clip(np.array(col)+noise,0,255).astype(np.uint8)
        
        if c=='wood':
            yy,xx=np.mgrid[0:108,0:108];grain=(8*np.sin(xx*1.1+2*np.sin(yy*.025))+4*np.sin(xx*2.8+yy*.02))[:,:,None];patch=np.clip(patch.astype(float)+grain,0,255).astype(np.uint8)
        if c=='white' and tuple(col)==(145,113,79):
            yy,xx=np.mgrid[0:108,0:108];mask=np.sin(xx*.15+np.sin(yy*.19))+np.sin(yy*.08)>0.2;patch[~mask]=np.array(PALETTE['cream'])
        ar[y:y+108,x:x+108,:3]=patch;ar[y:y+108,x:x+108,3]=255
    return Image.fromarray(ar)

class Asset:
    def __init__(self,asset_id,limit):
        self.id=asset_id;self.limit=limit;self.parts=[];self.notes=[];self.reclassified=[]
        self.src=ROOT/'assets/landmarks'/asset_id;self.out=ROOT/'data/processed/landmarks'/asset_id
        self.src.mkdir(exist_ok=True,parents=True);self.out.mkdir(exist_ok=True,parents=True)
        self.refs=json.loads((CMP/'work/references.json').read_text())[asset_id]
    def add(self,name,meaning,m,art=None,variants=None,span=None,basis=None,notes=None,refpart=None,origin=None):
        v=np.asarray(m.v,float);f=np.asarray(m.f,np.int32);uv=np.asarray(m.uv,float)
        v-=np.array(getattr(m,'anchor',(0,0,0)))
        v[:,2]-=v[:,2].min()
        if self.id=='r70_retaining_wall':v[:,2]/=v[:,2].max()
        assert len(f)<=self.limit,(name,len(f),self.limit)
        area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
        assert area.min()>1e-11,(name,'退化面',area.min())
        # 基準点は制作時に設定。外接矩形へ勝手に移さない（連結端や片腕柱がずれるため）。
        np.savez_compressed(self.out/(name+'.npz'),v=v,f=f,uv=uv)
        atlas(art,seed=name).save(self.out/(name+'.png'));shutil.copy2(self.out/(name+'.png'),self.src/(name+'.png'))
        if art is not None:art.save(self.src/(name+'_art.png'))
        textures=[name+'.png']
        for fname,extra in (variants or {}).items():
            va,changes=extra
            atlas(va if va is not None else art,changes,seed=name).save(self.out/fname);shutil.copy2(self.out/fname,self.src/fname);textures.append(fname)
        p=dict(name=name,meaning=meaning,triangles=len(f),vertices=len(v),dimensions_m=np.ptp(v,axis=0).round(5).tolist(),bounds_m=[v.min(0).round(6).tolist(),v.max(0).round(6).tolist()],origin=origin or ('始端支柱の根元 (0,0,0)' if span else '支持部の根元中央 (0,0,0)'),front='-Y',textures=textures,components=[dict(c) for c in m.components],dimension_basis=basis or '指定写真の物の比率と依頼文の目安寸法を用いた推定。現地実測ではない。',approximations=notes or [],references=self.refs.get(refpart or name,[]))
        if span:p.update(repeat_length_m=span,connect_direction='+X',connect_points_m=[[0,0,0],[span,0,0]])
        self.parts.append(p);return p
    def save(self):
        if self.id=='r70_road_signs':
            contents={'warning_diamond_with_text':'right_curve','regulatory_double':'no_uturn_40','route_hexagon_place':'70_清川村'}
            for p in self.parts:
                n=p['name']
                if n in contents:
                    file=n+'_face_'+contents[n]+'.png'
                    for folder in [self.src,self.out]:shutil.copy2(folder/(n+'.png'),folder/file)
                    if file not in p['textures']:p['textures'].append(file)
        info=dict(asset_id=self.id,source=self.id+'.blend',generator='blender_scripts/build_route70_assets.py',source_image_usage='参照写真は比較専用。配布画像はすべて自作。',coordinates={'unit':'m','up':'+Z','front':'-Y','origin':'部品別のorigin参照。単体支持部根元/反復物始端支柱根元'},npz={'keys':['v','f','uv'],'uv':'v=0画像上端。Blenderでは1-vで読込。'},texture_method='自作画像と手続き色を1024x1024アトラスへ直接ラスタライズ。照明を焼き込まない。',game_status='単体NPZ/PNGのみ。配置/KN5/インストールは範囲外。',triangle_limit_per_part=self.limit,parts=self.parts,approximations=self.notes,reclassified=self.reclassified,references=self.refs)
        (self.src/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');return info

def board(m,w,h,z,legs=1,c='white',r=.03,depth=.035,frame=False):
    m.box('板の厚み',(0,0,z),(w,depth,h),c);m.panel('差替用の表面',0,-depth/2-.0005,z,w*.98,h*.98)
    if frame:
        for x in [-w/2,w/2]:m.box('縦枠',(x,-.007,z),(.04,.06,h+.06),'wood')
        for zz in [z-h/2,z+h/2]:m.box('横枠',(0,-.007,zz),(w+.08,.06,.04),'wood')
    for x in ([0] if legs==1 else [-w*.4,w*.4] if legs==2 else []):m.tube('支柱',(x,.055,0),(x,.055,z+h*.35),r,c,8)
    if legs:m.anchor=(0,.055,0)
    for zz in [z-h*.3,z+h*.3]:m.box('裏面取付桟',(0,.035,zz),(w*.8,.035,.03),'metal')
