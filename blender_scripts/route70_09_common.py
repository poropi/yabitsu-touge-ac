"""依頼09専用。既存の小物メッシュを再利用し、中心原点と根拠を保存。"""
from pathlib import Path
import json,shutil,math
import numpy as np
from PIL import Image,ImageDraw
from route70_common import Mesh,atlas,lettering,PALETTE,KEYS
R=Path(__file__).resolve().parents[1]; C=R/'export/compare/route70'; W=C/'work09'
class Delivery:
    def __init__(self,a):
        self.id=a;self.src=R/'assets/landmarks'/a;self.out=R/'data/processed/landmarks'/a
        self.src.mkdir(parents=True,exist_ok=True);self.out.mkdir(parents=True,exist_ok=True)
        self.parts=[];self.locations=[];self.refs=json.loads((W/'references.json').read_text())[a]
    def add(self,name,meaning,m,refs=(),limit=15000,span=None,art=None,notes=(),basis=None,extra=None,changes=None):
        v=np.array(m.v,float);f=np.array(m.f,np.int32);uv=np.array(m.uv,float)
        assert len(f)<=limit,(name,len(f),limit)
        area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
        assert area.min()>1e-10,(name,area.min())
        # XYの中央、Zの最下点。連結基準は中心±L/2とする。
        anchor=np.array([(v[:,0].min()+v[:,0].max())/2,(v[:,1].min()+v[:,1].max())/2,v[:,2].min()]);v-=anchor
        np.savez_compressed(self.out/(name+'.npz'),v=v,f=f,uv=uv)
        im=atlas(art,changes=changes,seed=name)
        # 自作の石・板・塗装の模様。写真の画素は使わない。
        ar=np.array(im);rng=np.random.default_rng(809)
        for col in ['concrete','wood','rust','cream']:
            k=KEYS.index(col);x=800+k%2*112;y=k//2*112
            p=ar[y:y+108,x:x+108,:3].astype(float);yy,xx=np.mgrid[:108,:108]
            if col=='concrete':
                mott=3*np.sin(xx*.09+np.sin(yy*.14))+2*np.sin(yy*.21+xx*.06);p+=mott[:,:,None]
                moss=(np.sin(xx*.18+yy*.11)+np.cos(yy*.09)>1.65);p[moss]=p[moss]*.91+np.array([1,5,0])
            if col=='rust':
                joints=(yy%18<2)|((xx+(yy//18%2)*24)%48<2);p[joints]*=.72
            if col=='wood':p[(xx%27)<2]*=.55
            ar[y:y+108,x:x+108,:3]=np.clip(p,0,255).astype('uint8')
        Image.fromarray(ar).save(self.out/(name+'.png'));shutil.copy2(self.out/(name+'.png'),self.src/(name+'.png'))
        if art:art.save(self.src/(name+'_art.png'))
        rp=[r for r in self.refs if r['oid'] in refs]
        p=dict(name=name,meaning=meaning,triangles=len(f),triangle_limit=limit,vertices=len(v),dimensions_m=np.ptp(v,axis=0).round(6).tolist(),bounds_m=[v.min(0).round(6).tolist(),v.max(0).round(6).tolist()],origin='XY外接矩形中央・最下点 (0,0,0)',front='-Y',textures=[name+'.png'],components=m.components,references=rp,dimension_basis=basis or '指定写真の手すり高さ・柱間隔・道路幅と依頼文の目安から推定。現地実測ではない。',approximations=list(notes),source_anchor_m=anchor.tolist())
        if span:p.update(repeat_length_m=span,connect_direction='+X',connect_points_m=[[-span/2,0,0],[span/2,0,0]],connection_note='区切り中央に支柱1本。端面を突合せ、終端柱は別部品。隣接区切りの柱を重ねない。')
        if extra:p.update(extra)
        self.parts.append(p);return p
    def save(self):
        info=dict(asset_id=self.id,source=self.id+'.blend',generator='blender_scripts/build_route70_09_'+self.id.removeprefix('r70_')+'.py',coordinates=dict(unit='m',up='+Z',front='-Y',origin='部品別、XY中央・最下点'),npz=dict(keys=['v','f','uv'],uv='v=0画像上端。Blender読込時に1-v。'),texture_method='自作色・木目・石目・文字を1024角RGBAアトラスへ直接ラスタライズ。参照写真は使用しない。',game_status='NPZ/PNG単体納品。コース配置・KN5変換・ゲーム検証は依頼範囲外。',parts=self.parts,locations=self.locations,references=self.refs)
        (self.src/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n');return info

def merge(dst,src,angle=0,offset=(0,0,0)):
    st=len(dst.v);nf=len(dst.f);dst.v.extend(src.v);dst.uv.extend(src.uv);dst.f.extend(tuple(i+st for i in f) for f in src.f)
    dst.components.extend(dict(c,triangle_start=c['triangle_start']+nf) for c in src.components);dst.transform(st,angle,offset)

def plate(m,w=.35,h=.18,z=.55,text=None,c='black',y=-.21):
    m.box('名板の厚み',(0,y,z),(w,.018,h),c)
    if text:m.panel('判読した橋名',0,y-.010,z,w*.94,h*.88,rect=(8,8,768,768))

def arch_path(width,height,steps=32,spring=None):
    r=width/2;s=height-r if spring is None else spring
    return [(-r,0),(-r,s)]+[(r*math.cos(t),s+r*math.sin(t)) for t in np.linspace(math.pi,0,steps+1)[1:-1]]+[(r,s),(r,0)]
