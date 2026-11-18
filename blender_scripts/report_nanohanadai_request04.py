"""依頼04の確認用画像と一覧。参照画像をゲーム素材に転用しない。"""
from pathlib import Path
import json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1];CMP=ROOT/'export/compare/nanohanadai'
FONT='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,17)

def fit(im,size):
    im=ImageOps.contain(im.convert('RGB'),size);out=Image.new('RGB',size,(235,236,230));out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out

views=['front','left','right','back_estimated','top','persp']
labels=['正面 −Y','南西 −X','北東 +X','裏面 +Y（写真で隠れる部分は推定）','真上 +Z','斜め（接続部を含む）']
sheet=Image.new('RGB',(1600,1155),'white');d=ImageDraw.Draw(sheet)
for i,(view,label) in enumerate(zip(views,labels)):
    x=(i%2)*800;y=(i//2)*385;d.text((x+12,y+6),label,fill='black',font=font)
    sheet.paste(fit(Image.open(CMP/('nanohanadai_toilet_sw_'+view+'.png')),(800,345)),(x,y+38))
sheet.save(CMP/'nanohanadai_toilet_sw_views.jpg',quality=94)
comp=Image.new('RGB',(1800,1970),'white');d=ImageDraw.Draw(comp)
d.text((12,5),'南西延長部：同じ方位の比較／左：Google ストリートビュー　右：NPZ再読込',font=font,fill='black')
for i,head in enumerate([0,315,270]):
    y=45+i*630
    left=Image.open(CMP/('sv_sw_%03d.png'%head));right=Image.open(CMP/('nanohanadai_toilet_sw_reference_%03d.png'%head))
    comp.paste(left.resize((900,600)),(0,y+30));comp.paste(right.resize((900,600)),(900,y+30))
    d.text((12,y),'sw %03d°／2020年11月'%head,font=font,fill='black');d.text((915,y),'v2壁＋v2手すり＋ramp_sw_approach',font=font,fill='black')
d.text((12,1940),'右は依頼の仮カメラ（−9.4,−9.8,2.5m）／縦画角90°。周囲の草木とコース舗装は省略。',font=small,fill='black')
comp.save(CMP/'nanohanadai_toilet_sw_reference_comparison.jpg',quality=94)
# 航空画像とローカルXY輪郭を、旧制作元の屋根四隅を基準に重ねる。
# 航空画像の樹木・屋根の視差があるので測量精度の平面一致は保証しない。
im=Image.open(CMP/'gsat21.png').convert('RGBA');overlay=Image.new('RGBA',im.size,(0,0,0,0));d=ImageDraw.Draw(overlay)
mpp=.06083777;origin=np.array([475.5,635.5]);ux=np.array([math.sin(math.radians(59)),-math.cos(math.radians(59))]);uy=np.array([math.sin(math.radians(329)),-math.cos(math.radians(329))])
def px(x,y):return tuple(origin+(ux*x+uy*y)/mpp)
roof=[px(x,y) for x,y in [(-4.72,4.285),(4.72,4.285),(4.72,-4.285),(-4.72,-4.285),(-4.72,4.285)]]
d.line(roof,fill=(40,230,250,240),width=3)
d.line([px(-8.9,-5.04),px(8.9,-5.04)],fill=(80,255,90,255),width=3)
meta=json.loads((CMP/'nanohanadai_toilet_sw_inventory.json').read_text());st=meta['front_stations_x_y_floor_z_cap_z']
front=[px(x,y-.145) for x,y,z,h in st];rear=[px(x,y+1.66) for x,y,z,h in st]
d.polygon(front+rear[::-1],fill=(255,170,35,65));d.line(front,fill=(255,160,0,255),width=4);d.line(rear,fill=(255,160,0,255),width=3)
for a,b in zip(front,rear):d.line([a,b],fill=(255,180,40,200),width=1)
d.ellipse((px(0,0)[0]-5,px(0,0)[1]-5,px(0,0)[0]+5,px(0,0)[1]+5),fill=(255,255,255,255))
im=Image.alpha_composite(im,overlay).convert('RGB');out=Image.new('RGB',(1500,1100),'white');out.paste(im,(0,70));d=ImageDraw.Draw(out)
d.text((15,6),'真上の照合：水色＝既存屋根、緑＝v2前面、橙＝追加通路（XY輪郭）',font=font,fill='black')
d.text((15,35),'屋根の旧採寸四隅を基準に配置。樹木下の境界と屋根／地面の視差は推定を含む。',font=small,fill='black')
d.text((15,1075),'航空画像：Google マップ。写真に隠れる南西端と曲がりは推定。現地測量や配置後の一致確認ではない。',font=small,fill='black')
out.save(CMP/'nanohanadai_toilet_sw_top_overlay.jpg',quality=94)
# 判読用の拡大：写真の白壁と手すり終端を切り出す。出典表示は別帯で保持。
detail=Image.new('RGB',(1800,950),'white');d=ImageDraw.Draw(detail)
d.text((12,5),'sw315：壁と通路の拡大／単位m、壁高と地面勾配は写真からの推定',font=font,fill='black')
photo=Image.open(CMP/'sv_sw_315.png');game=Image.open(CMP/'nanohanadai_toilet_sw_reference_315.png')
detail.paste(fit(photo.crop((200,500,1500,945)),(900,820)),(0,45));detail.paste(fit(game.crop((130,490,1450,930)),(900,820)),(900,45))
strip=photo.crop((950,958,1500,1000));detail.paste(fit(strip,(700,55)),(15,875))
d.text((15,925),'写真：Google ストリートビュー／右：v2と延長部。撮影位置は仮値、画素単位の一致ではない。',font=small,fill='black')
detail.save(CMP/'nanohanadai_toilet_sw_detail_comparison.jpg',quality=94)
print('COMPARISON_IMAGES_WRITTEN')
