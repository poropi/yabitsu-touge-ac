"""ゲーム用NPZの屋根の真上の輪郭を、航空写真の読取点と同じ縮尺/方位へ重ねる。"""
from pathlib import Path
import math,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import MultiPoint
ROOT=Path(__file__).resolve().parents[1];CMP=ROOT/'export/compare/pass'
FONT='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'

def hull(asset,name):
    meta=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text());p=next(p for p in meta['parts'] if p['name']==name)
    with np.load(ROOT/'data/processed/landmarks'/asset/(name+'.npz')) as d:xy=d['v'][:,:2]+np.array(p['assembly_offset_m'][:2])
    return list(MultiPoint(xy).convex_hull.exterior.coords)

def make(asset,image,center,mpp,theta,names,reference_polys,notes):
    im=Image.open(CMP/'codex07'/image).convert('RGBA');overlay=Image.new('RGBA',im.size);d=ImageDraw.Draw(overlay)
    t=math.radians(theta);cs,sn=math.cos(t),math.sin(t)
    def project(xy):
        x,y=xy;return center[0]+(-cs*x-sn*y)/mpp,center[1]+(-sn*x+cs*y)/mpp
    model=[]
    for name in names:
        poly=[project(xy) for xy in hull(asset,name)];model.append(poly);d.polygon(poly,fill=(0,255,245,42));d.line(poly,fill=(0,255,245,255),width=4)
    for poly in reference_polys:
        q=poly+[poly[0]];d.line(q,fill=(255,82,60,255),width=3)
        for x,y in poly:d.ellipse((x-4,y-4,x+4,y+4),fill=(255,82,60,255))
    d.ellipse((center[0]-5,center[1]-5,center[0]+5,center[1]+5),fill=(250,245,50,255))
    im=Image.alpha_composite(im,overlay);d=ImageDraw.Draw(im)
    d.rectangle((75,5,1490,100),fill=(246,246,235,245));f=ImageFont.truetype(FONT,23);sm=ImageFont.truetype(FONT,17)
    d.text((90,12),asset+' 航空写真と納品NPZの屋根を同縮尺/同方位で重ねる',font=f,fill=(25,40,35))
    d.text((90,47),f'赤：写真の読取輪郭　青緑：モデル　{mpp:.6f}m/px　正面 {theta}°　北は上',font=sm,fill=(25,40,35))
    d.text((90,73),notes,font=sm,fill=(25,40,35))
    # 元画像の著作権/縮尺表示は残す。
    im.convert('RGB').save(CMP/(asset+'_aerial_overlay.jpg'),quality=94)
    (CMP/(asset+'_aerial_overlay.json')).write_text(json.dumps({'asset':asset,'image':image,'center_px':center,'m_per_px':mpp,'front_azimuth_deg':theta,'model_hulls_px':model,'reference_polygons_px':reference_polys,'note':notes,'alignment':'屋根の中心を読取輪郭の中心へ置き、モデル実寸の倍率と計測方位を固定。コース座標の配置検証ではない。'},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    j=json.loads((CMP/'work07/building_measurements.json').read_text());b=j['buildings'];sc=j['scales']
    a=b['pass_shop'];main=a['main_roof']['corners_px'];low=a['north_low_roof_annex']['corners_px'];center=np.mean(main,axis=0).tolist()
    make('pass_shop','gsat21.png',center,sc['gsat21.png']['m_per_px'],59,['shop_body','shop_annex'],[main,low],'屋根だけの比較。石台/段は別。背面は樹木とUIで隠れ、赤線は補間を含む。')
    poly=b['pass_toilet']['roof_envelope_corners_px'];center=np.mean(poly,axis=0).tolist()
    make('pass_toilet','gsat20.png',center,sc['gsat20.png']['m_per_px'],166,['metal_roof'],[poly],'屋根外包9.23×12.37mの読取に対し、モデル屋根面9.2×12.4m。隠れた北東隅は推定。')
    make('pass_lodge','gsat20.png',center,sc['gsat20.png']['m_per_px'],256,['lodge_roof'],[poly],'同じトイレの西長壁を正面とする代替データ。独立した別棟の輪郭は確認できない。')
    a=b['pass_resthouse'];poly=a['visible_light_roof_plane']['corners_px'];ridge_mid=(np.array(poly[0])+np.array(poly[3]))/2
    # 明るい東半面の左辺を棟の目安とする。屋根全体の外周と扱わない。
    make('pass_resthouse','gsat21.png',ridge_mid.tolist(),sc['gsat21.png']['m_per_px'],197,['roof'],[poly],'赤は明るい東屋根の片面のみ。モデルは公式寸法10.8×8.1m。撮影投影/角の判読差が残る。')
