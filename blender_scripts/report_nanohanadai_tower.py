"""採寸図、方向別シート、写真比較、航空写真の重ねを組む。通常Pythonで実行。"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[1]
CMP=ROOT/'export/compare/nanohanadai'
REF=CMP/'tower'
PREFIX='nanohanadai_tower_'
FONT='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'


def font(size):
    return ImageFont.truetype(FONT,size)


def label(im,xy,text,size=22,fill='#202b30'):
    ImageDraw.Draw(im).text(xy,text,font=font(size),fill=fill)


def opaque(path):
    im=Image.open(path).convert('RGBA')
    bg=Image.new('RGBA',im.size,(237,240,239,255))
    bg.alpha_composite(im)
    return bg.convert('RGB')


def fitted(im,size):
    return ImageOps.contain(im,size,Image.Resampling.LANCZOS)


def source_crop(filename,box):
    return Image.open(REF/filename).convert('RGB').crop(box)


def sheet():
    views=[('south','南'),('southeast','南東'),('east','東'),('northeast_estimated','北東（細部推定）'),
           ('north_estimated','北（細部推定）'),('northwest_estimated','北西（細部推定）'),('west','西'),('southwest','南西'),
           ('top','真上、北が上'),('persp','斜め上'),('deck_detail','デッキ'),('roof_detail','屋根')]
    out=Image.new('RGB',(1600,1600),'#f3f5f4')
    for k,(name,title) in enumerate(views):
        col,row=k%4,k//4
        label(out,(col*400+12,row*520+12),title,21)
        im=fitted(opaque(CMP/(PREFIX+name+'.png')),(390,465))
        out.paste(im,(col*400+(400-im.width)//2,row*520+48+(465-im.height)//2))
    label(out,(16,1566),'納品NPZ + PNGの再読込。周囲の地形、樹木、人物は撮影対象外。',19)
    out.save(CMP/(PREFIX+'views.jpg'),quality=94)
    for name in ['stairs','deck','roof','base']:
        opaque(CMP/(PREFIX+name+'_detail.png')).save(CMP/(PREFIX+name+'_detail_view.jpg'),quality=95)


def compare():
    cases=[('ref_yakei_nanohanadai4.jpg',(60,35,540,750),'reference_yakei','yakei.jp / 2010-01 / 南西'),
           ('tw_0rX1Rg_045.png',(500,190,1040,995),'reference_0rX1Rg_045','Google Maps / 天晴小號 / 2019-12 / 045h'),
           ('tw_0rXDBQ_345.png',(550,200,910,735),'reference_0rXDBQ_345','Google Maps / 天晴小號 / 2019-12 / 345h'),
           ('tw_4_znQE_015.png',(440,0,1030,950),'reference_4_znQE_015','Google Maps / kobayashi m / 2014-05 / 015h')]
    out=Image.new('RGB',(2400,2100),'#f3f5f4')
    for k,(fn,box,name,title) in enumerate(cases):
        x=(k%2)*1200
        y=(k//2)*1030
        label(out,(x+18,y+12),title,22)
        label(out,(x+18,y+50),'参照写真（切り抜き）',21)
        label(out,(x+615,y+50),'納品NPZ（同じ側からの近似視点）',21)
        left=fitted(source_crop(fn,box),(574,875))
        im=Image.open(CMP/(PREFIX+name+'.png')).convert('RGBA')
        bbox=im.getchannel('A').getbbox()
        im=im.crop((max(0,bbox[0]-20),max(0,bbox[1]-20),min(im.width,bbox[2]+20),min(im.height,bbox[3]+20)))
        bg=Image.new('RGBA',im.size,(237,240,239,255));bg.alpha_composite(im)
        right=fitted(bg.convert('RGB'),(574,875))
        out.paste(left,(x+12+(574-left.width)//2,y+95+(875-left.height)//2))
        out.paste(right,(x+612+(574-right.width)//2,y+95+(875-right.height)//2))
        label(out,(x+18,y+985),'左の出典は上記。参照画素はアセット材質に使用していません。',17)
    label(out,(20,2068),'画角と撮影位置は推定。2010年の赤茶色は再現色にせず、2019〜2024年の濃い茶を採用。',20)
    out.save(CMP/(PREFIX+'reference_comparison.jpg'),quality=95)


def measurement():
    im=Image.open(REF/'ref_yakei_nanohanadai4.jpg').convert('RGB')
    draw=ImageDraw.Draw(im)
    cyan='#4efff0'
    person={'head_px':[296,114],'foot_estimated_px':[296,205],'assumed_height_m':[1.6,1.7],
            'foot_visibility':'手すりの奥に隠れるため足位置は推定。全高の直接換算には使わない。',
            'main_rail_px':[[303,146],[303,205]],'rail_estimate_m':[59/91*1.6,59/91*1.7]}
    for start,end in [((296,114),(296,205)),((303,146),(303,205))]:
        draw.line([start,end],fill=cyan,width=3)
        for x,y in (start,end):draw.ellipse((x-4,y-4,x+4,y+4),fill=cyan)
    draw.line([(238,694),(266,207)],fill='#ffca4c',width=3)
    draw.line([(266,207),(276,82)],fill='#ffca4c',width=3)
    for pos,text in [((530,90),'全高13m：設計者の作品紹介'),((530,135),'人物：頭114px、足205px（足は推定）'),
                     ((530,180),'人物1.6〜1.7mを仮定、手すり59px'),((530,225),'手すりの概算：約1.04〜1.10m'),
                     ((530,270),'採用値：床9.30m、主手すり10.40m'),((530,315),'軒11.90m、屋根頂13.00m（推定内訳）'),
                     ((530,360),'黄色：柱とデッキ柱を読む区間'),((530,405),'透視差があり単純なpx比で全高へ換算しない')]:
        draw.rectangle((pos[0]-5,pos[1]-3,1265,pos[1]+33),fill='#f2f2e9')
        label(im,pos,text,23)
    label(im,(530,770),'参照: yakei.jp/photo/1280/nanohanadai4-1280.jpg',19)
    im.save(CMP/(PREFIX+'measure_yakei.jpg'),quality=95)

    source=CMP/(PREFIX+'architect_25.jpg')
    im=Image.open(source).convert('RGB').resize((1200,866))
    draw=ImageDraw.Draw(im)
    # 設計者の下向き写真で、中央の石床を囲う12区画の端を順に追う。
    nodes=np.array([(306,54),(352,70),(389,105),(404,151),(394,198),(355,230),
                    (307,243),(262,230),(229,197),(217,151),(230,108),(263,72)])*2
    for i,p in enumerate(nodes):
        q=nodes[(i+1)%12]
        draw.line([tuple(p),tuple(q)],fill='#ffe45d',width=3)
        draw.ellipse((p[0]-16,p[1]-16,p[0]+16,p[1]+16),fill='#ffe45d')
        label(im,(p[0]-10,p[1]-15),str(i+1),22)
    label(im,(20,780),'12区画を追った読み取り。各印の位置は写真上の手動指定。',24,'#ffffff')
    label(im,(20,825),'出典: 久保寺敏郎 都市・建築設計事務所 / works/index/27',22,'#ffffff')
    im.save(CMP/(PREFIX+'polygon_evidence.jpg'),quality=95)
    data=dict(person=person,heights_m={'deck':9.30,'rail':10.40,'eave':11.90,'crown':13.00},
              heights_status='全高13mのみ設計者公表。他は写真の人物尺度と形の比率を整合させた推定。',
              polygon_readout={'sides':12,'source':'nanohanadai_tower_architect_25.jpg','nodes_px':(nodes/2).tolist()},
              stair_count={'visible_detail':'tw_IfLggE_000.pngの右下などに連続した踏面。全経路と全段数は写真では不可視。',
                           'chosen_risers_per_path':48,'riser_m':9.3/48,'total_run_per_path_degrees':540,
                           'landings_per_path':6,'status':'段数は推定、二重らせんは設計者資料で確認。'})
    (CMP/(PREFIX+'measurements.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def aerial():
    # 元画像の比率は変えず、与えられた縮尺で屋根だけを重ねる。
    ref=Image.open(REF/'aerial_tower_north_up.png').convert('RGBA')
    pts=np.array([(391,295),(440,306),(476,343),(489,390),(475,435),(440,470),
                  (393,482),(345,468),(311,437),(299,388),(311,344),(345,310)],float)
    center=np.mean(pts,axis=0)
    radii=np.linalg.norm(pts-center,axis=1)
    mpp=.041
    roof=Image.open(CMP/(PREFIX+'roof_top.png')).convert('RGBA')
    # 元レンダーは11.2m幅。屋根の中心はカメラ中心。
    span=round(11.2/mpp)
    roof=roof.resize((span,span),Image.Resampling.LANCZOS)
    layer=Image.new('RGBA',ref.size)
    layer.alpha_composite(roof,(round(center[0]-span/2),round(center[1]-span/2)))
    alpha=np.array(layer.getchannel('A'))
    mask=Image.fromarray(np.uint8(alpha*.53))
    layer.putalpha(mask)
    over=Image.alpha_composite(ref,layer).convert('RGB')
    d=ImageDraw.Draw(over)
    d.line([tuple(p) for p in pts]+[tuple(pts[0])],fill='#ffe64c',width=2)
    d.line([(int(center[0]),230),(int(center[0]),180)],fill='red',width=4)
    label(over,(int(center[0])+10,180),'N',24,'red')
    comp=Image.new('RGB',(1440,800),'#f3f5f4')
    comp.paste(ref.convert('RGB'),(0,50))
    comp.paste(over,(720,50))
    label(comp,(14,10),'Google航空写真（北が上）',24)
    label(comp,(735,10),'同縮尺の屋根NPZを重ねる（黄：手動の外周）',24)
    label(comp,(14,775),'約0.041m/px。屋根中心を合わせた形状検査で、基礎との地上位置を検証した図ではありません。',21)
    comp.save(CMP/(PREFIX+'top_overlay.jpg'),quality=96)
    data=dict(source='tower/aerial_tower_north_up.png',m_per_pixel=mpp,scale_status='依頼文に示された概算縮尺を使用。独立した地上測量ではない。',
              manual_roof_vertices_px=pts.tolist(),center_px=center.tolist(),
              measured_circum_diameter_px=float(radii.mean()*2),
              measured_circum_diameter_m=float(radii.mean()*2*mpp),
              measured_axis_width_px=np.ptp(pts,axis=0).tolist(),
              model_nominal_roof_diameter_m=7.8,model_max_width_including_fascia_m=7.964207649230957,
              orientation='ほぼ東西南北に頂点、面の法線は真北から15+30k度。画像のぼけによる数度の不確かさがある。',
              path_readout_px=[[402,485],[458,611]],path_bearing_deg=math.degrees(math.atan2(56,-126))%360,
              limitation='屋根と地面の視差がある。台は屋根に隠れ、台径8.0mと石段の詳細は地上写真からの推定。')
    (CMP/(PREFIX+'aerial_measurements.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    sheet()
    compare()
    measurement()
    aerial()
