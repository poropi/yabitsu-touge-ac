"""菜の花台単体アセット。コース生成や配置はしない。
.venv/bin/python blender_scripts/build_nanohanadai_assets.py textures
blender -b --factory-startup --python-exit-code 1 --python blender_scripts/build_nanohanadai_assets.py -- <asset-id>
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))

def reference_info(images):
    sites={
     'sw':(35.4102783,139.2092117,'hCGwhUE1XaPosSAeVCOneg','2020-11'),
     'mid':(35.4103796,139.2093967,'MC9FOu1heTQ47rJo-eqsKA','2020-11'),
     'mid2':(35.4104211,139.2095236,'P7jvxAJ_IJlvRcEGVeXJSA','2020-11'),
     'ne':(35.4105395,139.209692,'1UbJBI3qyKnEo7x_lS6j3w','2020-11'),
     'road':(35.4106689,139.2098766,'Py9NMnj-9UKVFZrB9gImAQ','2025-07'),
     'user':(35.4101451,139.2091123,'CIHM0ogKEICAgICEzoqyEg','2017-07')}
    refs=[]
    for name in images:
        if not name.startswith('sv_'):
            refs.append({'image':'export/compare/nanohanadai/'+name,'usage':'航空写真採寸/比較のみ。配布素材への転用なし。'});continue
        _,site,angle=name[:-4].split('_');lat,lon,pano,date=sites[site]
        user=site=='user';kind='!2e10!7i8704!8i4352' if user else '!2e0!7i16384!8i8192'
        refs.append({'image':'export/compare/nanohanadai/'+name,'panoid':pano,'date':date,'fov_vertical_deg':90,'heading_label_deg':int(angle),'pitch_down_deg':0 if user else 4.13,'url':f'https://www.google.com/maps/@{lat},{lon},3a,90y,{int(angle)}h,{90 if user else 85.87}t/data=!3m6!1e1!3m4!1s{pano}'+kind,'note':'利用者360の方向ラベルは真方位として扱わない。' if user else 'Google Street View。比較参照のみ。'})
    return refs

def main(asset):
    import nanohanadai_common as C
    C.init_scene()
    if asset in ('nanohanadai_monument','nanohanadai_park_sign'):
        import build_nanohanadai_entrance as module
    elif asset=='nanohanadai_toilet':
        import build_nanohanadai_toilet as module
    else:
        import build_nanohanadai_small as module
    meta=module.build(asset);meta['references']=reference_info(meta['refs']);meta['generator']='blender_scripts/'+Path(module.__file__).name
    C.save_asset(asset,meta)

if __name__=='__main__':
    arg=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else sys.argv[-1]
    if arg=='references-only':
        import json
        for p in (Path(__file__).resolve().parents[1]/'assets/landmarks').glob('nanohanadai_*/asset.json'):
            d=json.loads(p.read_text());d['references']=reference_info(d['refs']);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    elif arg=='textures':
        import build_nanohanadai_entrance as E
        import build_nanohanadai_small as S
        E.textures();S.textures()
    else:main(arg)
