"""依頼08の出力契約・予算・透明度・継ぎ目・原点の検査。既存コースは変更しない。"""
from pathlib import Path
import json,copy
import numpy as np
from PIL import Image
from build_route70_assets import ASSETS,LIMITS
R=Path(__file__).resolve().parents[1];C=R/'export/compare/route70'
def arrays(v,f,u,limit):
    assert v.ndim==2 and v.shape[1]==3 and len(v)>2
    assert f.ndim==2 and f.shape[1]==3 and 0<len(f)<=limit
    assert u.shape==(len(v),2) and np.isfinite(v).all() and np.isfinite(u).all()
    assert np.issubdtype(f.dtype,np.integer) and f.min()>=0 and f.max()<len(v)
    assert u.min()>=0 and u.max()<=1 and len(v)<65536
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)/2
    assert area.min()>1e-11
    assert abs(v[:,2].min())<1e-8
    return float(area.min())
def main():
    reports=[]
    for a,lim in zip(ASSETS,LIMITS):
        src=R/'assets/landmarks'/a;out=R/'data/processed/landmarks'/a;info=json.loads((src/'asset.json').read_text());checks=[]
        assert (src/(a+'.blend')).stat().st_size>10000
        assert {f.stem for f in out.glob('*.npz')}=={p['name'] for p in info['parts']}
        for p in info['parts']:
            n=p['name']
            with np.load(out/(n+'.npz')) as d:
                assert set(d.files)=={'v','f','uv'};v,f,u=d['v'],d['f'],d['uv']
            assert sum(c['triangle_count'] for c in p['components'])==len(f)
            assert all(c['triangle_count']>0 and c['triangle_start']+c['triangle_count']<=len(f) for c in p['components'])
            minimum=arrays(v,f,u,lim);assert len(f)==p['triangles'] and np.allclose(np.ptp(v,axis=0),p['dimensions_m'],atol=1e-5)
            for file in p['textures']:
                im=Image.open(out/file);assert im.mode=='RGBA' and im.size==(1024,1024);assert (src/file).read_bytes()==(out/file).read_bytes()
            if 'repeat_length_m' in p:
                L=p['repeat_length_m'];assert p['connect_direction']=='+X' and p['connect_points_m']==[[0,0,0],[L,0,0]]
                # 連結面の主形状が始端・終端を持つこと。柱や枠の半径は外にはみ出してよい。
                assert np.any(np.isclose(v[:,0],0,atol=.025)) and np.any(np.isclose(v[:,0],L,atol=.025)),(a,n,'連結点')
            if a=='r70_retaining_wall':
                assert np.isclose(np.ptp(v[:,2]),1)
                ar=np.asarray(Image.open(out/(n+'_surface.png')));assert ar.shape[:2]==(1024,1024)
                assert np.array_equal(ar[:,0],ar[:,-1]) and np.array_equal(ar[0],ar[-1])
            if n in ['wire_mesh','wire_mesh_concrete','wire_mesh_orange','wood_mesh','rockfall_net','pipe_mesh','gate','gate_green']:
                alpha=np.asarray(Image.open(out/(n+'.png')))[:,:,3];assert alpha.min()==0 and alpha.max()==255
            checks.append(dict(name=n,triangles=len(f),min_triangle_area_m2=minimum,finite=True,uv=True,origin=True,textures=len(p['textures'])))
        report={'asset':a,'passed':True,'parts':checks};reports.append(report);(C/(a+'_npz_validation.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    # 同じ検査が異常を検知することを故意に壊したメモリ上の配列で確認。実ファイルは変更しない。
    negative=[]
    for kind in ['uv_out_of_range','degenerate_face','negative_ground']:
        vv=v.copy();ff=f.copy();uu=u.copy()
        if kind=='uv_out_of_range':uu[0,0]=1.5
        elif kind=='degenerate_face':ff[0]=ff[0,0]
        else:vv[:,2]-=.5
        try:arrays(vv,ff,uu,lim)
        except AssertionError:negative.append(kind)
        else:raise AssertionError(('検査が異常を見逃した',kind))
    result=dict(passed=True,assets=len(reports),parts=sum(len(x['parts']) for x in reports),triangles=sum(p['triangles'] for x in reports for p in x['parts']),negative_controls=negative,checks=reports)
    (C/'route70_validation_08.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in result.items() if k!='checks'})
if __name__=='__main__':main()
