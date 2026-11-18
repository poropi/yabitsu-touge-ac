"""依頼09の実データ検査。テスト用フォルダ・コースを変更しない。"""
from pathlib import Path
import json,sys,copy
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];C=R/'export/compare/route70'
ASSETS=['r70_bridges','r70_tunnels','r70_buildings']

def y_ray(v,f,x,z,y0,y1):
    tri=v[f];a=tri[:,0];b=tri[:,1];c=tri[:,2]
    den=(b[:,2]-c[:,2])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,2]-c[:,2])
    valid=np.abs(den)>1e-10;den=np.where(valid,den,1)
    u=((b[:,2]-c[:,2])*(x-c[:,0])+(c[:,0]-b[:,0])*(z-c[:,2]))/den
    w=((c[:,2]-a[:,2])*(x-c[:,0])+(a[:,0]-c[:,0])*(z-c[:,2]))/den
    t=1-u-w;yy=u*a[:,1]+w*b[:,1]+t*c[:,1]
    return bool(np.any(valid&(u>=-1e-7)&(w>=-1e-7)&(t>=-1e-7)&(yy>=y0)&(yy<=y1)))

def check_arrays(v,f,u,lim):
    assert v.ndim==2 and v.shape[1]==3 and len(v)>2
    assert f.ndim==2 and f.shape[1]==3 and 0<len(f)<=lim
    assert u.shape==(len(v),2) and np.isfinite(v).all() and np.isfinite(u).all()
    assert np.issubdtype(f.dtype,np.integer) and f.min()>=0 and f.max()<len(v)
    assert u.min()>=0 and u.max()<=1 and len(v)<65536
    assert abs(v[:,2].min())<1e-7
    assert np.allclose(v[:,:2].min(0)+v[:,:2].max(0),0,atol=1e-6)
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
    assert area.min()>1e-10
    return area.min()

def main(assets=ASSETS):
    reports=[]
    for a in assets:
        src=R/'assets/landmarks'/a;out=R/'data/processed/landmarks'/a;j=json.loads((src/'asset.json').read_text());checks=[]
        assert (src/(a+'.blend')).stat().st_size>10000
        expected={p['name'] for p in j['parts']};assert expected=={p.stem for p in out.glob('*.npz')},(a,'余分/不足NPZ')
        for p in j['parts']:
            n=p['name']
            with np.load(out/(n+'.npz')) as d:
                assert set(d.files)=={'v','f','uv'};v,f,u=d['v'],d['f'],d['uv']
            minimum=check_arrays(v,f,u,p['triangle_limit'])
            assert len(f)==p['triangles'] and np.allclose(np.ptp(v,axis=0),p['dimensions_m'],atol=1e-5)
            counts=np.zeros(len(f),int)
            for co in p['components']:
                assert co['triangle_count']>0;counts[co['triangle_start']:co['triangle_start']+co['triangle_count']]+=1
            assert (counts==1).all(),(a,n,'構成部品の割当')
            for t in p['textures']:
                im=Image.open(out/t);assert im.size==(1024,1024) and im.mode=='RGBA'
                assert (src/t).read_bytes()==(out/t).read_bytes()
            if 'repeat_length_m' in p:
                L=p['repeat_length_m'];assert p['connect_direction']=='+X'
                assert np.isclose(v[:,0].min(),-L/2) and np.isclose(v[:,0].max(),L/2),(a,n,'中心連結')
                assert p['connect_points_m']==[[-L/2,0,0],[L/2,0,0]]
            if 'lining' in n:
                L=p['repeat_length_m'];ends=[]
                for xx in [-L/2,L/2]:
                    yz=np.unique(np.round(v[np.isclose(v[:,0],xx),1:],5),axis=0);ends.append(yz)
                assert ends[0].shape==ends[1].shape and np.allclose(ends[0],ends[1]),(n,'断面の接続')
                co=[c for c in p['components'] if c['name']=='反復内壁'];normalchecks=[]
                w=p['cross_section_width_m'];h=p['cross_section_height_m'];spring=h-w/2
                for c in co:
                    ff=f[c['triangle_start']:c['triangle_start']+c['triangle_count']];vv=v[ff]
                    nn=np.cross(vv[:,1]-vv[:,0],vv[:,2]-vv[:,0]);center=vv.mean(1)
                    inward=np.column_stack([np.zeros(len(ff)),-center[:,1],spring-center[:,2]])
                    normalchecks.extend(np.sum(nn*inward,axis=1).tolist())
                assert min(normalchecks)>0,(n,'内壁法線')
            if 'portal' in n:
                plane=p['portal_plane_y_m'];ww=p['opening_width_m'];hh=p['opening_height_m']
                for x,z in [(0,.5),(0,hh*.6),(-ww*.2,hh*.4),(ww*.2,hh*.4)]:
                    assert not y_ray(v,f,x,z,plane-2,plane+2),(n,'坑口を塞ぐ面',x,z)
                assert y_ray(v,f,0,hh+.4,plane-2,plane+2),(n,'坑門上部がない')
                for co in p['components']:
                    if co['name'].startswith('読めた坑名 '):
                        ff=f[co['triangle_start']:co['triangle_start']+co['triangle_count']];ids=np.unique(ff);sp=np.ptp(u[ids],axis=0)
                        assert np.isclose(sp[0],sp[1],rtol=.01),(n,'個別文字の縦横UV比')
            if a=='r70_buildings':
                for co in p['components']:
                    if co['name']=='片流れ屋根下の三角壁':
                        ff=f[co['triangle_start']:co['triangle_start']+co['triangle_count']];vv=v[ff]
                        normal=np.cross(vv[:,1]-vv[:,0],vv[:,2]-vv[:,0]);face_x=vv.mean(1)[:,0]+p['source_anchor_m'][0]
                        assert np.all(normal[:,0]*face_x>0),(n,'片流れ妻壁の法線')
                if n=='s5_11637_mokko_office':
                    def points(co):return v[np.unique(f[co['triangle_start']:co['triangle_start']+co['triangle_count']])]
                    land=points(next(c for c in p['components'] if c['name']=='階段踊り場'))
                    doors=[points(c) for c in p['components'] if c['name']=='開口の奥のガラス']
                    door=next(vv for vv in doors if vv[:,0].mean()+p['source_anchor_m'][0]>p['footprint_dimensions_m'][0]/2-.1)
                    assert land[:,0].min()<door[:,0].mean()<land[:,0].max(),(n,'踊り場と扉のX接続')
                    assert land[:,1].min()<=door[:,1].min() and land[:,1].max()>=door[:,1].max(),(n,'踊り場と扉のY接続')
                    steps=[points(c) for c in p['components'] if c['name']=='外階段の踏板'];last=max(steps,key=lambda vv:vv[:,2].mean())
                    assert last[:,1].max()>=land[:,1].min() and last[:,2].max()>=land[:,2].min(),(n,'階段と踊り場の接続')
            checks.append(dict(name=n,triangles=len(f),dimensions_m=p['dimensions_m'],minimum_area_m2=float(minimum),texture_ok=True,origin_ok=True))
        if a=='r70_bridges':
            assert len(j['locations'])==36
            assert not any(l['chainage_m']==2442 and l['section']=='S5' for l in j['locations'])
            assert all(l['left_rail'] in expected and l['right_rail'] in expected and (not l['end_post'] or l['end_post'] in expected) for l in j['locations'])
        if a=='r70_tunnels':
            assert len(j['locations'])==3 and sum('portal' in p['name'] for p in j['parts'])==4
            assert {p['repeat_length_m'] for p in j['parts'] if p['name'].endswith('_20m')}=={20}
            alpha=np.array(Image.open(out/'small_lining_20m.png'))[:,:,3];assert alpha.min()==0 and alpha.max()==255
        if a=='r70_buildings':
            assert len(j['parts'])==19
            ids={i for p in j['parts'] for i in p['source_object_ids']};assert len(ids)==22,(a,ids)
            assert all(p['triangles']<=15000 for p in j['parts'])
        result=dict(asset=a,passed=True,parts=checks);reports.append(result)
        (C/(a+'_npz_validation_09.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    negatives=[]
    for kind in ['bad_uv','degenerate','negative_ground','shifted_origin','over_budget']:
        vv=v.copy();ff=f.copy();uu=u.copy();lim=p['triangle_limit']
        if kind=='bad_uv':uu[0,0]=2
        elif kind=='degenerate':ff[0]=ff[0,0]
        elif kind=='negative_ground':vv[:,2]-=1
        elif kind=='shifted_origin':vv[:,0]+=2
        else:lim=len(ff)-1
        try:check_arrays(vv,ff,uu,lim)
        except AssertionError:negatives.append(kind)
        else:raise AssertionError(('検査が故障を検知しなかった',kind))
    result=dict(passed=True,parts=sum(len(x['parts']) for x in reports),triangles=sum(p['triangles'] for x in reports for p in x['parts']),negative_controls=negatives,assets=reports)
    (C/'route70_validation_09.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in result.items() if k!='assets'})
if __name__=='__main__':main(sys.argv[1:] or ASSETS)
