"""ゲームの片面描画向けに、露出一枚面の法線を納品NPZで検査する。"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load(asset,name):
    j=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text());p=next(p for p in j['parts'] if p['name']==name)
    with np.load(ROOT/'data/processed/landmarks'/asset/(name+'.npz')) as d:v,f=d['v']+np.array(p['assembly_offset_m']),d['f']
    t=v[f];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=np.linalg.norm(n,axis=1)[:,None];return t,n

def main():
    checks=[]
    t,n=load('pass_road_items','guardrail');q=(np.ptp(t[:,:,0],axis=1)>1.99)&(np.abs(n[:,1])>.5)
    checks.append({'surface':'厚さのあるW形ビームの表−Y/裏+Y','triangles':int(q.sum()),'passed':bool(q.sum()==24 and (n[q,1]<-.5).sum()==12 and (n[q,1]>.5).sum()==12),'normals':n[q].round(6).tolist()})
    t,n=load('pass_resthouse','walls');q=(np.abs(t[:,:,1]-3.15).max(1)<1e-4)&(t[:,:,2].mean(1)>4.8)
    checks.append({'surface':'北の裏三角妻+Y','triangles':int(q.sum()),'passed':bool(q.sum()==1 and (n[q,1]>.99).all()),'normals':n[q].round(6).tolist()})
    t,n=load('pass_resthouse','roof');plane=6.985-np.abs(t[:,:,0])*(3.05/4.5);q=np.abs(t[:,:,2]-plane).max(1)<2e-4
    checks.append({'surface':'左右の木目軒裏の下向き','triangles':int(q.sum()),'passed':bool(q.sum()==4 and (n[q,2]<-.8).all()),'normals':n[q].round(6).tolist()})
    result={'passed':all(x['passed'] for x in checks),'checks':checks}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    (ROOT/'export/compare/pass/pass_surface_orientation_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    assert result['passed'],'露出面の巻き方向が外向きではない'
if __name__=='__main__':main()
