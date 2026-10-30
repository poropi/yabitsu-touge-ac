"""書き出した7個のNPZ/PNGを読み戻し、受け渡し仕様を検証する。
実行: .venv/bin/python blender_scripts/verify_asama_assets.py
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
results=[]
for asset,stems,limit in [
    ('asama_wall_vines',['vines_a','vines_b','vines_c'],8000),
    ('asama_shrine_post',['post'],1000),
    ('asama_grass_tufts',['grass_a','grass_b','grass_c'],1500),
]:
    for stem in stems:
        folder=ROOT/'data/processed/landmarks'/asset
        with np.load(folder/(stem+'.npz')) as data:
            assert set(data.files)=={'v','f','uv'},stem
            v,f,uv=[data[key] for key in ['v','f','uv']]
        assert v.ndim==2 and v.shape[1]==3 and uv.shape==(len(v),2),stem
        assert f.ndim==2 and f.shape[1]==3 and len(f)<=limit,stem
        assert f.dtype.kind in 'iu' and f.min()>=0 and f.max()<len(v),stem
        assert np.isfinite(v).all() and np.isfinite(uv).all(),stem
        assert uv.min()>=-1e-6 and uv.max()<=1+1e-6,stem
        tri=v[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(n,axis=1)
        assert (area>1e-12).all(),(stem,'degenerate triangle')
        normals=np.zeros_like(v)
        for i in range(3):np.add.at(normals,f[:,i],n)
        nl=np.linalg.norm(normals,axis=1)
        # 実ゲームのKN5生成器と同じ頂点法線集計。逆面共有の相殺を検出する。
        assert (nl>1e-12).all(),(stem,'zero vertex normals',int((nl<=1e-12).sum()))
        im=Image.open(folder/(stem+'.png'))
        alpha=np.array(im.convert('RGBA'))[:,:,3]
        is_plant=stem!='post'
        if is_plant:
            assert im.mode=='RGBA' and alpha.min()==0 and alpha.max()==255,stem
        if stem.startswith('vines'):
            assert abs(np.ptp(v[:,0])-2.5)<1e-5,stem
            assert 1.25<v[:,2].min()<2.1 and 3.5<v[:,2].max()<4.1,stem
        if stem=='post':
            assert np.allclose(np.ptp(v,axis=0),[.2,.2,2.4],atol=1e-5),stem
        if stem.startswith('grass'):
            assert v[:,2].min()>=-1e-6 and .5<v[:,2].max()<1.1,stem
        results.append({'asset_id':asset,'name':stem,'vertices':len(v),'triangles':len(f),'limit':limit,
                        'bounds_m':[v.min(0).tolist(),v.max(0).tolist()],'uv_bounds':[uv.min(0).tolist(),uv.max(0).tolist()],
                        'png_mode':im.mode,'png_size':list(im.size),'transparent_pixels':int((alpha==0).sum()),
                        'zero_vertex_normals':int((nl<=1e-12).sum()),'degenerate_triangles':int((area<=1e-12).sum()),'ok':True})
out=ROOT/'export/compare/asama/post_grass_vines_delivery_check.json'
out.write_text(json.dumps({'ok':True,'assets':results},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'ok':True,'assets':[{k:r[k] for k in ['name','triangles','vertices','zero_vertex_normals']} for r in results],'report':str(out)},ensure_ascii=False))
