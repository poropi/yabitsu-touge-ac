import json, os, subprocess, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString
from scipy.spatial import cKDTree
from yabitsu import config, nagoki, forest_ground, track
from yabitsu.crosssection import Section
from scripts import dress_compare as dc, forest_build as fb
out=sys.argv[1]
sec=Section.load_json(config.DATA_PROCESSED/'track_section.json'); sp=nagoki.load_splits(sec)[0]
road=forest_ground.load_ground(config.DATA_PROCESSED/'track',prefixes=('1ROAD_',))
T=cKDTree(np.c_[sec.x,sec.y]); views=[]
for wid,tag in (('254266932','up'),('254266927','down')):
    L=np.asarray(sp.lines[wid].coords); s0=sec.s[T.query(L[0])[1]]; s1=sec.s[T.query(L[-1])[1]]
    if (tag=='up')!=(s1>s0): L=L[::-1]
    ls=LineString(L)
    for f in (0.0,0.3,0.6):
        p=np.array(ls.interpolate(f,normalized=True).coords[0]); q=np.array(ls.interpolate(min(1,f+0.25),normalized=True).coords[0])
        d=(q-p)/np.linalg.norm(q-p); p=p-d*(15 if f==0 else 0); q=p+d*40
        z=float(np.nan_to_num(road.z(p[None])[0],nan=0)); zt=float(np.nan_to_num(road.z(q[None])[0],nan=z))
        views.append({"name":f"br_{tag}_{int(f*100):02d}","type":"persp","eye":[*p,z+1.2],"target":[*q,zt+1.0],"lens":22,"lod":0,"res":[900,500],"window":[p[0]-200,p[1]-200,p[0]+200,p[1]+200]})
cl=np.load(config.DATA_PROCESSED/'track_centerline.npz'); b=track.load_bounds()
c,_=fb._at('s3',3410,cl,b); z=float(road.z(c[None])[0])
views.append({"name":"top_split","type":"top","center":c.tolist(),"size":110.0,"z":z,"window":[c[0]-80,c[1]-80,c[0]+80,c[1]+80],"lod":2})
(dc.FOREST/'views.json').write_text(json.dumps(views),encoding='utf-8')
r=subprocess.run([dc.BLENDER,"-b","--factory-startup","--python","blender_scripts/render_forest.py","--",str(dc.FOREST),"s3"],env={**os.environ,"DRESS_DIR":str(dc.DRESS)},capture_output=True,timeout=2400); print(r.returncode)
font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',22)
SV=dc.SV/'sv'
sh=Image.new('RGB',(2700,500*4),(255,255,255))
for k,v in enumerate(views[:6]):
    im=Image.open(dc.FOREST/'views'/f"{v['name']}.png").convert('RGB'); d=ImageDraw.Draw(im)
    d.rectangle((0,0,220,32),fill=(0,0,0)); d.text((6,2),v['name'],fill=(255,255,0),font=font)
    sh.paste(im,((0 if '_up_' in v['name'] else 900),(k%3)*500))
for k,n in enumerate(['s3_3350_f','s3_3400_f','s3_3450_f','s3_3450_b']):
    sv=Image.open(SV/f'{n}.png').convert('RGB').resize((900,500)); d=ImageDraw.Draw(sv); d.rectangle((0,0,300,32),fill=(0,0,0)); d.text((6,2),'SV '+n[3:],fill=(255,255,0),font=font)
    sh.paste(sv,(1800,k*500))
top=Image.open(dc.FOREST/'views/top_split.png').convert('RGB').resize((500,500)); sh.paste(top,(0,1500)); sh.paste(Image.open(SV/'s3_3400_b.png').convert('RGB').resize((900,500)),(900,1500))
sh.save(out,quality=88); print('ok')
