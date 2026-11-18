"""コース全体を組み直さず、既存地形と新しい南西端だけを描くための入力を作る。"""
from pathlib import Path
import json,sys
import numpy as np
from shapely.geometry import Polygon,box,Point,LineString
from shapely.ops import triangulate
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from yabitsu import start_lot,landmarks
from yabitsu.crosssection import Section
OUT=ROOT/'export/compare/nanohanadai';WORK=OUT/'work/sw_fix';WORK.mkdir(exist_ok=True)
spec=json.loads((ROOT/'data/manual/s1_start_lot.json').read_text());F=start_lot.frame(spec)
lot=start_lot.lot_polygon(spec,Section.load_json(ROOT/'data/processed/track_section.json'))
lab=np.c_[F.ab(np.asarray(lot.exterior.coords))];lot_ab=Polygon(lab)
poly=Polygon()
for entry in spec['paved'][1:]:poly=poly.union(Polygon(entry['points']))
geometry=[]
def surface(p,z):
 vv=[];ff=[]
 for tri in triangulate(p):
  if not p.covers(tri.representative_point()):continue
  q=np.asarray(tri.exterior.coords)[:3]-[-66,0];i=len(vv);vv.extend([[x,y,z] for x,y in q]);ff.append([i,i+1,i+2])
 return vv,ff
geometry.append(['土と草の下地',*surface(poly,.003),'grass'])
for row,b in enumerate(np.arange(-2.5,12.0,.45)):
 for a in np.arange(-85,-61.5,.60):
  p=poly.intersection(box(a+.022,b+.022,a+.578,b+.428))
  if p.area<.008:continue
  for g in ([p] if p.geom_type=='Polygon' else p.geoms):
   if g.area<.008:continue
   geometry.append([f'敷石_{row}_{a:.2f}',*surface(g,.022+(row%3)*.002),'stone'])
for a in np.arange(-84.5,-62,.6):
 for b in np.arange(-2.5,12.0,.45):
  if poly.contains(Point(a,b)):
   x,y=a+66,b;geometry.append(['目地の短い草',[[x-.025,y,.004],[x+.025,y,.004],[x,y,.06]],[[0,1,2]],'grass'])
(WORK/'flag_geometry.json').write_text(json.dumps(geometry,ensure_ascii=False))
pads=json.loads((ROOT/'data/processed/track_pads.json').read_text());pad=next(p for p in pads if p['name']=='nanohanadai_lot');pad={**pad,'polygon':np.asarray(lot.exterior.coords).tolist()}
# 地形・既存部品は読取のみ。近景の木は使わず、配置が見える状態で検証。
new=[]
for k,(name,v,f,uv,mat,show) in enumerate(start_lot.meshes(spec,pad)):
 if not show:continue
 p=WORK/f'mesh_{k:03d}.npz';np.savez_compressed(p,verts=v,faces=f,uvs=uv)
 new.append({'name':name,'file':str(p),'material':mat})
materials={}
for mat,(p,alpha) in landmarks.materials(spec['landmarks']).items():materials[mat]={'texture':str(p),'alpha':alpha}
info={'meshes':new,'materials':materials,'pad':pad,'spec':spec,'lot_ab':lab.tolist(),'flag_outline':[list(p.exterior.coords) for p in ([poly] if poly.geom_type=='Polygon' else poly.geoms)]}
(WORK/'scene.json').write_text(json.dumps(info,ensure_ascii=False,indent=1))
print('prepared',len(new),'meshes; flag surfaces',len(geometry),'area',poly.area,'lot overlap',poly.intersection(lot_ab).area)
# 組立済みコースから、南西端周辺の地形・道路だけを切り出す（元データは変更しない）。
man=json.loads((ROOT/'data/processed/track/manifest.json').read_text());old=[]
newnames={x['name'] for x in new}
for k,(name,e) in enumerate(man['meshes'].items()):
 if not e.get('renderable',True) or name in newnames or 's1_lm_' in name or 's1_paved_' in name or 's1_lot_' in name:continue
 if e['material'].startswith('lm_'):continue
 d=np.load(ROOT/'data/processed/track'/e['file']);v=d['verts'];a,b=F.ab(v[:,:2]);inside=(a>=-150)&(a<=12)&(b>=-75)&(b<=80)
 if not inside.any():continue
 fc=d['faces'];sel=inside[fc].any(axis=1)
 if not sel.any():continue
 path=WORK/f'old_{k:04d}.npz';np.savez_compressed(path,verts=v,faces=fc[sel],uvs=d['uvs']);old.append({'name':name,'file':str(path),'material':e['material']})
oldm={k:{**v,'texture':str(ROOT/'data/processed/track'/v['texture']) if v.get('texture') else None} for k,v in man['materials'].items()}
info['old_meshes']=old;info['old_materials']=oldm
(WORK/'scene.json').write_text(json.dumps(info,ensure_ascii=False,indent=1));print('old local terrain/road',len(old))
