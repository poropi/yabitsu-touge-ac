"""看板の実形状へのrayと元の自作画像を照合し、NPZ上端v=0を確認。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def cast(path,origin,flip=False):
    with np.load(path) as d:v,f,u=d['v'],d['f'],d['uv'].copy()
    if flip:u[:,1]=1-u[:,1]
    a,b,c=v[f[:,0]],v[f[:,1]],v[f[:,2]];e1=b-a;e2=c-a;direction=np.array([0.,1.,0.])
    h=np.cross(direction,e2);det=(e1*h).sum(1);ok=np.abs(det)>1e-10
    inv=np.divide(1,det,out=np.zeros_like(det),where=ok);s=origin-a;bu=(s*h).sum(1)*inv;q=np.cross(s,e1);bv=q[:,1]*inv;t=(e2*q).sum(1)*inv
    ok&=(bu>=-1e-7)&(bv>=-1e-7)&(bu+bv<=1.0000001)&(t>0)
    ix=np.where(ok)[0];assert len(ix),'ray missed';j=ix[np.argmin(t[ix])]
    uv=u[f[j,0]]*(1-bu[j]-bv[j])+u[f[j,1]]*bu[j]+u[f[j,2]]*bv[j]
    im=np.asarray(Image.open(path.with_suffix('.png')).convert('RGB'));px=np.clip(np.rint(uv*np.array([im.shape[1]-1,im.shape[0]-1])).astype(int),[0,0],[im.shape[1]-1,im.shape[0]-1]);return im[px[1],px[0]].astype(float),uv.tolist()

def check_shop():
    art=np.asarray(Image.open(ROOT/'assets/landmarks/pass_shop/shop_sign_art.png').convert('RGB'))
    # 左上の国民宿舎の白い文字から、上下反転すると暗い地になる画素を選ぶ。
    y,x=np.mgrid[0:art.shape[0],0:art.shape[1]]
    mask=(x>45)&(x<450)&(y>65)&(y<135)&(art[:,:,0]>225)&(art[::-1,:,0]<160)
    points=np.argwhere(mask);assert len(points)>10
    meta=json.loads((ROOT/'assets/landmarks/pass_shop/asset.json').read_text());p=next(p for p in meta['parts'] if p['name']=='shop_sign')
    offset=np.array(p['assembly_offset_m']);results=[]
    for py,px in points[::max(1,len(points)//15)][:15]:
        uu=px/(art.shape[1]-1);vv=py/(art.shape[0]-1)
        world=np.array([(uu-.5)*5.22*1.0625,-20, .62+4.32+(.5-vv)*.72]);origin=world-offset
        good,uv=cast(ROOT/'data/processed/landmarks/pass_shop/shop_sign.npz',origin)
        wrong,_=cast(ROOT/'data/processed/landmarks/pass_shop/shop_sign.npz',origin,True)
        err=np.abs(good-art[py,px]).mean();err_bad=np.abs(wrong-art[py,px]).mean()
        results.append({'art_pixel':[int(px),int(py)],'expected':art[py,px].tolist(),'sample':good.tolist(),'wrong_flip_sample':wrong.tolist(),'error':float(err),'wrong_error':float(err_bad),'baked_uv':uv})
    good=[r for r in results if r['error']<45 and r['wrong_error']>r['error']+40];assert len(good)>=3,results
    out={'passed':True,'asymmetric_label':'国民宿舎の白文字','hits':len(good),'samples':results,'conclusion':'NPZ uvをそのままPNGの上端0で採色すると元画像と一致。反転を追加すると不一致。'}
    (ROOT/'export/compare/pass/pass_shop_uv_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('UV orientation PASS',len(good))

if __name__=='__main__':check_shop()
