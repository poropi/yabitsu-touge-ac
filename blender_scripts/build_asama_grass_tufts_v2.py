"""浅間神社の密生する黄緑の草株 v2。立札・コース配置は変更しない。
.venv/bin/python blender_scripts/build_asama_grass_tufts_v2.py textures/sheets
Blender内: runpy.run_path(...,run_name='__main__') (sys.argv末尾 build/render/npz)
"""
from pathlib import Path
import sys,json,math,random,shutil
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'assets/landmarks/asama_grass_tufts';OUT=ROOT/'data/processed/landmarks/asama_grass_tufts';CMP=ROOT/'export/compare/asama'
HEIGHTS=[.58,.81,1.06];WIDTHS=[.66,.84,.99];COUNTS=[104,106,107]

def textures():
 import numpy as np
 from PIL import Image
 h,w=1024,512;yy,xx=np.mgrid[0:h,0:w];u=(xx%128)/127;t=1-yy/(h-1);rng=np.random.default_rng(301031)
 # 明るい黄緑、若葉・やや枯れた葉を交ぜる。写真は使わない。
 palettes=np.array([[151,190,88],[158,197,91],[137,179,72],[163,190,94]])
 rgb=palettes[xx//128].astype(float)
 lum=4*np.sin(xx*1.6)+rng.normal(0,1.7,(h,w))+t*5-5*abs(u-.5)*2
 rgb+=lum[...,None]+(np.exp(-((u-.5)/.03)**2)*11)[...,None]
 alpha=np.uint8(np.clip((.49-abs(u-.5))*80,0,1)*255)
 a=np.concatenate([np.uint8(np.clip(rgb,0,255)),alpha[...,None]],axis=2)
 SRC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 Image.fromarray(a,'RGBA').save(SRC/'grass_atlas_v2.png')
 for c in 'abc':Image.fromarray(a,'RGBA').save(OUT/f'grass_{c}.png')

def grass_geometry(i,col,mat):
 import numpy as np
 import build_asama_small_assets as old
 rng=random.Random(2200+i);height=HEIGHTS[i];count=COUNTS[i];verts=[];faces=[];uvs=[]
 for n in range(count):
  angle=n*2.399963+rng.uniform(-.2,.2)
  inner=n%8==0;peak=height*rng.uniform(.85,1) if inner else height*rng.uniform(.45,.87)
  reach=rng.uniform(.04,.18) if inner else rng.uniform(.20,WIDTHS[i]*.52)
  bend=rng.uniform(1.65,2.05) if inner else rng.uniform(2.65,3.05)
  origin=np.array([math.cos(angle)*rng.uniform(.006,.075),math.sin(angle)*rng.uniform(.006,.075),rng.uniform(0,.012)])
  axis=np.array([math.cos(angle),math.sin(angle),0.0]);side=np.array([-math.sin(angle),math.cos(angle),rng.uniform(-.17,.17)])
  base=len(verts);seg=7;width=rng.uniform(.014,.022);palette=n%4
  for j in range(seg+1):
   t=j/seg;theta=bend*t;z=peak*math.sin(theta)
   center=origin+axis*(reach*(1-math.cos(theta))/(1-math.cos(bend)))+np.array([0,0,z]);ww=width*(1-t)**.75*(.5+.5*min(t*5,1))+.00008
   for sign in [-1,1]:verts.append(tuple(center+side*sign*ww/2))
  verts.extend(verts[base:])
  for j in range(seg):
   ids=[base+j*2,base+j*2+1,base+(j+1)*2+1,base+(j+1)*2];us=[((palette*128+3)/512,1-j/seg),((palette*128+125)/512,1-j/seg),((palette*128+125)/512,1-(j+1)/seg),((palette*128+3)/512,1-(j+1)/seg)]
   faces.extend([ids,[k+2*(seg+1) for k in ids[::-1]]]);uvs.extend([us,us[::-1]])
 ob=old.mesh_object(f'草株_{"ABC"[i]}_密生_両面',verts,faces,uvs,col,mat)
 zmax=max(v.co.z for v in ob.data.vertices)
 xmin=min(v.co.x for v in ob.data.vertices);xmax=max(v.co.x for v in ob.data.vertices);ymin=min(v.co.y for v in ob.data.vertices);ymax=max(v.co.y for v in ob.data.vertices)
 scale=WIDTHS[i]/max(xmax-xmin,ymax-ymin)
 for v in ob.data.vertices:v.co.x*=scale;v.co.y*=scale;v.co.z=max(0,v.co.z*height/zmax)
 return ob

def build():
 import bpy
 import build_asama_small_assets as old
 s=old.init_scene();stage=bpy.data.collections.new('90 撮影用（非出力）');s.collection.children.link(stage);old.setup_stage(s,stage)
 mat=old.material(SRC/'grass_atlas_v2.png','草葉v2_RGBA_明るい黄緑',True);cols=[];reports=[]
 for i,c in enumerate('abc'):
  col=bpy.data.collections.new(f'01 草株 {c.upper()}');s.collection.children.link(col);cols.append(col);ob=grass_geometry(i,col,mat)
  r=old.export_npz(ob,OUT/f'grass_{c}.npz');r.update({'blades':COUNTS[i],'height_m':HEIGHTS[i],'width_max_m':WIDTHS[i]});assert r['triangles']<=3000;reports.append(r)
 for i in range(1,3):
  ns=s.copy();ns.name=f'grass_{"abc"[i]}';ns.use_fake_user=True
  for col in cols:
   if col!=cols[i]:ns.collection.children.unlink(col)
 for col in cols[1:]:s.collection.children.unlink(col)
 s.name='grass_a';s.use_fake_user=True;bpy.context.preferences.filepaths.save_version=0
 bpy.context.preferences.filepaths.file_preview_type = "NONE"   # .blend の保存でサムネイルを作らない（tests/test_blender_save_no_thumbnail.py）
 bpy.ops.wm.save_as_mainfile(filepath=str(SRC/'asama_grass_tufts_v2.blend'))
 meta=json.loads((SRC/'asset.json').read_text());meta.update({'source':'asama_grass_tufts_v2.blend','previous_source':'asama_grass_tufts.blend','generator':'blender_scripts/build_asama_grass_tufts_v2.py','texture':'grass_atlas_v2.png','parts':reports,'dimension_basis':'依頼の写真に合わせ高さ0.58/0.81/1.06m、株幅0.66/0.84/0.99m。104/106/107枚の葉を幅14〜22mm、7区間の弓状曲線で密生。株全体の寸法を合わせる縮尺調整を含む。現地未実測。','comparison':'export/compare/asama/grass2_reference_comparison.jpg','preview':'export/compare/asama/grass2_views.jpg','checks':['export/compare/asama/grass2_readback_check.json','export/compare/asama/grass2_npz_check.json'],'delivery_check':'export/compare/asama/plants2_delivery_check.json','npz_preview':[f'export/compare/asama/grass2_npz_{i}.png' for i in range(3)]})
 (SRC/'asset.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');(CMP/'grass2_export_check.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
 return reports

def render():
 import bpy
 from mathutils import Vector
 views=[('front',(0,-6,0)),('right',(6,0,0)),('left',(-6,0,0)),('back',(0,6,0)),('top',(0,0,6)),('persp',(3,-6,2.8)),('sv000',(0,-6,1.3)),('sv045',(4,-4,1.3)),('sv315',(-4,-4,1.3))]
 checks=[]
 for i,c in enumerate('abc'):
  s=bpy.data.scenes[f'grass_{c}'];bpy.context.window.scene=s;ob=next(o for o in s.objects if o.type=='MESH');ob.data.calc_loop_triangles();checks.append({'scene':s.name,'triangles':len(ob.data.loop_triangles),'origin':list(ob.location),'dimensions':list(ob.dimensions)})
  cam=s.camera;target=Vector((0,0,HEIGHTS[i]*.44));cam.data.type='ORTHO';cam.data.ortho_scale=max(HEIGHTS[i],WIDTHS[i])*1.3
  s.cycles.samples=12;s.render.resolution_x=500;s.render.resolution_y=500
  for label,dr in views:
   cam.location=target+Vector(dr);cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();s.render.filepath=str(CMP/f'grass2_{i}_{label}.png');bpy.ops.render.render(write_still=True)
 (CMP/'grass2_readback_check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
 return checks

def sheets():
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 font=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',21);small=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',16)
 views=[('front','正面 -Y'),('right','右 +X'),('left','左 -X'),('back','背面（推定）'),('top','真上（推定）'),('persp','斜め')]
 sheet=Image.new('RGB',(1620,960),'#eef0ed');d=ImageDraw.Draw(sheet);d.text((14,12),'草株 v2：3種類 / 六方向・密度確認',font=font,fill='black')
 for i in range(3):
  for j,(v,l) in enumerate(views):
   im=ImageOps.contain(Image.open(CMP/f'grass2_{i}_{v}.png').convert('RGB'),(265,265));sheet.paste(im,(j*270,50+i*300));d.text((j*270+8,318+i*300),f'{"ABC"[i]} {l}',font=small,fill='black')
 sheet.save(CMP/'grass2_views.jpg',quality=94)
 comp=Image.new('RGB',(1530,1630),'#eef0ed');d=ImageDraw.Draw(comp);d.text((18,10),'草株 v2：Google Street View 2025年7月 / 見込み方向を近似した単体比較',font=font,fill='black')
 crops=[('000',(700,570,920,750),'sv000',0),('045',(240,635,470,850),'sv045',1),('315',(1250,610,1500,845),'sv315',2)]
 for row,(h,crop,label,i) in enumerate(crops):
  y=55+row*515;ref=Image.open(CMP/f'sv_a_{h}.png').convert('RGB');comp.paste(ref.resize((675,450)),(0,y+22));detail=ImageOps.contain(ref.crop(crop),(350,340));comp.paste(detail,(680,y+55));model=ImageOps.contain(Image.open(CMP/f'grass2_{i}_{label}.png').convert('RGB'),(490,450));comp.paste(model,(1040,y+22));d.text((8,y),f'参照 {h}° 全景（出典表示保持）',font=small,fill='black');d.text((686,y),'株の拡大',font=small,fill='black');d.text((1047,y),f'制作モデル {"ABC"[i]} SV {h}を参考にした推定方向',font=small,fill='black')
 d.text((15,1600),'距離・照明は近似。植生種・個々の葉の本数・背面は推定。写真を材質に転用していない。',font=small,fill='black');comp.save(CMP/'grass2_reference_comparison.jpg',quality=94)

if __name__=='__main__':
 if 'textures' in sys.argv:textures()
 elif 'sheets' in sys.argv:sheets()
 else:
  sys.path.insert(0,str(ROOT/'blender_scripts'))
  if 'render' in sys.argv:render()
  else:build()
