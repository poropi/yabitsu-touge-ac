"""写真比較・真上の重ね合わせと配置の数値検証。画像は比較専用。"""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import Polygon,LineString
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'export/compare/nanohanadai/work'))
import xform
from yabitsu import start_lot,config
from yabitsu.crosssection import Section
OUT=ROOT/'export/compare/nanohanadai';W=OUT/'work/sw_fix';D=json.loads((W/'scene.json').read_text());s=D['spec'];F=start_lot.frame(s)
font=ImageFont.truetype(config.font_file('gothic'),25);small=ImageFont.truetype(config.font_file('gothic'),18)
# 3方向。同じカメラ/同じ画角。写真を縦横比のまま縮小。
sheet=Image.new('RGB',(1500,3*540+90),'white');dr=ImageDraw.Draw(sheet)
for k,h in enumerate([135,180,0]):
 y=35+k*540
 for j,(p,label) in enumerate([(OUT/f'sv_sw_{h:03d}.png',f'sw_{h:03d} Google Street View 2020-11'),(OUT/f'nanohanadai_sw_fix_sw_{h:03d}.png',f'sw_{h:03d} 配置修正後（局所描画）')]):
  im=Image.open(p).convert('RGB').resize((750,500));sheet.paste(im,(750*j,y+30));dr.text((750*j+8,y),label,fill='black',font=font)
dr.text((10,1658),'カメラ高2.5m・縦画角90°・下向き4.13°。地形/植生/トイレ本体は既存形状。車・遠景・仕上げ層は省略。',fill='black',font=small)
sheet.save(OUT/'nanohanadai_sw_fix_comparison.jpg',quality=94)
# 航空写真へ正射影をアフィン変換して重ねる。GPSではなく同じ地物変換を両者に使う。
# src_render = affine(gsat_pixel)。xformのa軸倍率もそのまま反映。
center=F.xy(-65,8);CROP=(430,575,735,835);NX,NY=915,780
P=np.array([[0,0],[1000,0],[0,1000]],float);Q=xform.px_ab(P);XY=F.xy(Q[:,0],Q[:,1]);RP=np.c_[800+(XY[:,0]-center[0])*16,800-(XY[:,1]-center[1])*16]
M=np.linalg.solve(np.c_[P,np.ones(3)],RP)
def gp_to_render(x,y):return np.r_[x,y,1]@M
q0=gp_to_render(CROP[0],CROP[1]);qx=gp_to_render(CROP[0]+(CROP[2]-CROP[0])/NX,CROP[1]);qy=gp_to_render(CROP[0],CROP[1]+(CROP[3]-CROP[1])/NY)
coeff=(qx[0]-q0[0],qy[0]-q0[0],q0[0],qx[1]-q0[1],qy[1]-q0[1],q0[1])
aerial=Image.open(OUT/'gsat20.png').convert('RGB').crop(CROP).resize((NX,NY));render=Image.open(OUT/'nanohanadai_sw_fix_top.png').convert('RGB').transform((NX,NY),Image.Transform.AFFINE,coeff,resample=Image.Resampling.BICUBIC)
# a・bから航空写真の画像位置へ
INV=np.linalg.solve(np.c_[Q,np.ones(3)],P)
def abpix(p):
 q=np.r_[p,1]@INV;return ((q[0]-CROP[0])*NX/(CROP[2]-CROP[0]),(q[1]-CROP[1])*NY/(CROP[3]-CROP[1]))
overlay=Image.blend(aerial,render,.45);od=ImageDraw.Draw(overlay)
od.line([abpix(p) for p in D['lot_ab']],fill=(0,255,120),width=3)
for ring in D['flag_outline']:od.line([abpix(p) for p in ring],fill=(255,0,220),width=3)
p=abpix([-64.8442,3.839]);od.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6),outline='yellow',width=3)
for h in [135-56.31,135,135+56.31]:
 t=np.radians(h-56.1);end=np.array([-64.8442,3.839])+8*np.array([np.cos(t),-np.sin(t)]);od.line([p,abpix(end)],fill='yellow',width=2)
canvas=Image.new('RGB',(NX*3,NY+90),'white');cd=ImageDraw.Draw(canvas)
for j,(im,title) in enumerate([(aerial,'Google航空写真（gsat20）'),(render,'修正後・真上（同じ地物変換）'),(overlay,'航空写真＋修正後45%の重ね合わせ')]):canvas.paste(im,(j*NX,60));cd.text((j*NX+10,10),title,font=font,fill='black')
cd.text((10,NY+67),'緑：駐車場の舗装端　紫：新しい敷石帯　黄：今回のswカメラと135°の視野。航空写真の地上帯とは入口付近で約3〜4mの差が残る。',font=small,fill='black')
canvas.save(OUT/'nanohanadai_sw_fix_top_overlay.jpg',quality=94)
# 数値検査（描画の代用にせず、重なり・離隔・改変範囲を確認）
lot=start_lot.lot_polygon(s,Section.load_json(ROOT/'data/processed/track_section.json'));flag=Polygon() 
for ring in D['flag_outline']:flag=flag.union(Polygon(F.xy(*np.array(ring).T)))
path=LineString(F.xy(*np.array(s['path']['points']).T)).buffer(s['path']['width']/2)
paved=Polygon()
for item in s['paved'][:1]:paved=paved.union(Polygon(F.xy(*np.array(item['points']).T)))
flats=[F.rect(r['a'],r['b']).distance(lot) for r in s['flats']]
auto_flat=LineString(F.xy(*np.array(s['path']['points']).T)).buffer(s['path']['width']/2+1).distance(lot)
before=json.loads((OUT/'work/sw_fix_before.json').read_text());outside=[it for it in s['landmarks'] if it['asset'] not in ['nanohanadai_path_items','nanohanadai_sw_end_v1']];old=[it for it in before['spec']['landmarks'] if it['asset']!='nanohanadai_path_items']
protected_changed=[p for p,stat in before['protected'].items() if [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]!=stat]
asset_changed=[p for p,sha in before['assets'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha]
newparts=[]
for p in (ROOT/'data/processed/landmarks/nanohanadai_sw_end_v1').glob('*.npz'):
 d=np.load(p);v=d['v'];fc=d['f'];uv=d['uv'];area=np.linalg.norm(np.cross(v[fc[:,1]]-v[fc[:,0]],v[fc[:,2]]-v[fc[:,0]]),axis=1)/2
 newparts.append({'name':p.stem,'triangles':len(fc),'vertices':len(v),'dimensions_m':np.ptp(v,axis=0).tolist(),'bounds_m':[v.min(0).tolist(),v.max(0).tolist()],'all_finite':bool(np.isfinite(v).all() and np.isfinite(uv).all()),'min_triangle_area':float(area.min()),'uv_range':[float(uv.min()),float(uv.max())]})
result={'pavement_overlap_m2':{'flag_lot':float(flag.intersection(lot).area),'path_lot':float(path.intersection(lot).area),'paved_lot':float(paved.intersection(lot).area),'flag_existing_path':float(flag.intersection(path).area),'flag_far_existing_paved':float(flag.intersection(paved).area)},'flat_to_lot_distance_m':flats,'automatic_path_flat_distance_m':float(auto_flat),'non_sw_landmarks_unchanged':outside==old,'non_sw_landmarks_count':len(outside),'far_path_items_unchanged':[it for it in s['landmarks'] if it['asset']=='nanohanadai_path_items' and it['a']<=-85]==[it for it in before['spec']['landmarks'] if it['asset']=='nanohanadai_path_items' and it['a']<=-85],'protected_files_changed':protected_changed,'existing_asset_files_changed':asset_changed,'new_parts':newparts,'camera_ab_m':[-64.8442,3.839],'camera_model_yaw_offset_deg':-1.3,'flag_decoration_above_drivable_surface_m':0.003,'checks':'有限値・UV範囲・非縮退・旧アセット不変・舗装重複・平地離隔。写真への一致判定は画像の目視と別途行う。'}
(OUT/'nanohanadai_sw_fix_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
assert not protected_changed and not asset_changed and old==outside
assert all(v<1e-8 for v in result['pavement_overlap_m2'].values())
assert min(flats)>=.5-1e-8 and auto_flat>=.5-1e-8
assert all(p['all_finite'] and p['min_triangle_area']>1e-10 and p['vertices']<65535 for p in newparts)
