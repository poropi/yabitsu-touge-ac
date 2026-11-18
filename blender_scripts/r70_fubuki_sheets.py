"""保存済みの写真と単体レンダーを比較用のJPEGに組む（Pillow）。"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'export/compare/route70'
W = OUT/'work/fubuki10'
SV = ROOT/'export/compare/route70_inventory_20261030/images/sv'
PAPER, INK, MUTED, EDGE = '#f6f5f0','#242d2d','#4a5656','#d4d9d3'
MODEL = '#e4e8e2'
ACCENT = '#8e363b'
JP = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
BOLD = '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc'


def text(im, x, y, value, size=22, bold=False, color=INK):
    ImageDraw.Draw(im).text((x,y),value,font=ImageFont.truetype(BOLD if bold else JP,size),fill=color,anchor='lt')


def sheet(title,sub,height):
    im = Image.new('RGB',(1480,height),PAPER)
    text(im,38,28,title,37,True)
    text(im,39,86,sub,20,color=MUTED)
    ImageDraw.Draw(im).line((38,127,1442,127),fill=EDGE,width=2)
    return im


def photo(key,box=None):
    p = SV/f'R70-{key}.jpg'
    if not p.exists(): p = W/p.name
    im = Image.open(p).convert('RGB')
    return im.crop(box) if box else im


def render(key):
    rgba = Image.open(W/(key+'.png')).convert('RGBA')
    return rgba


def place(im,img,box,fit=False,bg=MODEL):
    x,y,w,h=box
    if fit:
        obj = ImageOps.fit(img,(w,h),method=Image.Resampling.LANCZOS)
    else:
        obj = ImageOps.contain(img,(w,h),method=Image.Resampling.LANCZOS)
    frame = Image.new('RGB',(w,h),bg)
    frame.paste(obj,((w-obj.width)//2,(h-obj.height)//2),obj.getchannel('A') if obj.mode=='RGBA' else None)
    im.paste(frame,(x,y))


def save(im,name):
    im.save(OUT/('route70_delivery_10_'+name+'.jpg'),quality=93,subsampling=0,optimize=True)


def comparison_fences():
    im=sheet('草の斜面の上端に立つ柵','左は現地写真の拡大、右は2 mの部品を2本つなぎ、最後に終端支柱を置いた単体モデル。',940)
    for i,(title,key,box,n,detail) in enumerate([
        ('金網','0972_f',(1180,64,1600,425),'slope_fence_wire_2m','柱高 1.50 m ／ 網は上下の張線の間。柱の外径は54 mm。'),
        ('橋側の縦格子','0974_f',(1240,138,1580,305),'slope_fence_bar_2m','柱高 1.50 m ／ 2 mごとに角形支柱1本と、細い縦材14本。')]):
        y=151+i*351
        text(im,38,y,title,27,True)
        text(im,277,y+5,n,21,color=MUTED)
        place(im,photo(key,box),(38,y+44,568,229),bg='#d5d9d3')
        place(im,render(n),(622,y+44,820,229))
        text(im,38,y+284,'R70-'+key+'　© 2026 Google',16,color=MUTED)
        text(im,622,y+283,detail,19)
    text(im,38,873,'斜面上の背の高い細柱・生垣は写真の背景。網の目・線径と格子ピッチは推定。',19,color=MUTED)
    save(im,'fences')


def comparison_bridge():
    im=sheet('釜田川橋を進行・逆方向から見る','左は現地、右は同じ進行方向の単体モデル。道路・歩道・白線は位置関係を見るための仮設面。',1958)
    for i,(key,title) in enumerate([('0975_f','橋に進入。右欄干と白線の間に歩道が続く。'),
                              ('0976_f','アーチの内側。赤い縦材は灰色の欄干の外を通る。'),
                              ('0978_b','橋の終わりから振り返る。欄干がアーチの脚の手前を横切る。'),
                              ('0978_f','同じ地点から坑口へ。残る橋面の両側には灰色の欄干が続く。')]):
        y=146+i*447
        text(im,38,y,'R70-'+key,24,True)
        text(im,278,y+2,title,21)
        place(im,photo(key),(38,y+40,693,390),bg='#dde2dc')
        place(im,render('view_'+key),(748,y+40,694,390))
    text(im,38,1930,'写真  Google ストリートビュー 2025年7月　© 2026 Google　／　撮影姿勢と画角は各地点の見かけに合わせた近似。',15,color=MUTED)
    save(im,'bridge_views')


def bridge_structure():
    im=sheet('釜田川橋の脚・床・欄干','単体モデルの構造確認。路面は Z = 0.70 m、進入端は −Y。外接矩形の中心へ原点を移さない。',1266)
    place(im,render('bridge_axo'),(38,149,1404,371))
    text(im,38,539,'側面　赤い短帯は脚の交点付近の各5.10 m。リブ最下点は路面より2.59 m下。',21,True)
    place(im,render('bridge_side'),(38,578,1404,242))
    text(im,38,840,'真上　進行方向は右。写真の右歩道側だけ欄干を2.90 m外へ移した。',21,True)
    # +Yを紙面右に置く。元画像を90度時計回りにする。
    top=render('bridge_plan').transpose(Image.Transpose.ROTATE_270)
    # 仮路面のない真上図に、保存時のカメラ中心X=1.45から道路中心を投影する。
    cy = round(top.height/2-1.45*top.height/106)
    draw = ImageDraw.Draw(top)
    for x in range(26,725,11):
        draw.line((x,cy,x+5,cy),fill=(120,98,37,255),width=1)
    box=top.getchannel('A').getbbox()
    top=top.crop((0,max(0,box[1]-10),top.width,min(top.height,box[3]+10)))
    place(im,top,(38,877,1404,216))
    text(im,38,1116,'破線は道路中心 X = 0',20,True)
    text(im,398,1116,'左欄干 −4.40 m　｜　右欄干 +7.30 m',20)
    text(im,924,1116,'左右のリブ軸 −5.23 / +8.13 m',19)
    text(im,38,1172,'側桁と地覆は灰色。赤はアーチ・横桁・縦材と脚部の短帯に使用。床下横桁と支承側の形状は推定。',19,color=MUTED)
    save(im,'bridge_structure')


def comparison_wall():
    im=sheet('坑口左の擁壁に載る白い壁','写真で見える白い段だけを2 mの部品にした。下のブロック擁壁と草は設置先のもの。',1042)
    text(im,38,152,'橋から見た壁の始まり',23,True)
    text(im,749,152,'坑口寄りで見える壁面と継目',23,True)
    place(im,photo('0978_f',(370,370,835,590)),(38,190,692,294),bg='#dce1da')
    place(im,photo('0980_f',(0,312,713,640)),(748,190,694,294),bg='#dce1da')
    for x,k in [(38,'0978_f'),(748,'0980_f')]:
        text(im,x,496,'R70-'+k+'　© 2026 Google',16,color=MUTED)
    text(im,38,548,'2 m × 3本の連結例',25,True)
    place(im,render('wall'),(38,593,1005,369))
    text(im,1070,637,'高さ　0.86 m',24,True)
    text(im,1070,692,'底厚　0.22 m',24,True)
    text(im,1070,750,'白壁と下段の高さ比は約1：3。',19)
    text(im,1070,790,'厚みと背面の仕上げは推定。',19)
    text(im,38,990,'原点は始端小口の底面中央。正面 −Y を道路側に向け、擁壁の天端で +X 方向へ連結する。',19,color=MUTED)
    save(im,'wall')


def axes():
    im=sheet('6部品の形状確認','保存した制作元を正投影で確認。各枠は部品に合わせて拡大しており、列・行をまたぐ縮尺は共通でない。',1390)
    labels=['正面 −Y','左側 −X','右側 +X','背面 +Y','真上 +Z','斜め']
    for j,a in enumerate(labels): text(im,235+j*203,151,a,20,True)
    place(im,render('six_views'),(215,194,1227,1056))
    for i,a in enumerate(['金網 2 m','金網の終端','縦格子 2 m','縦格子の終端','釜田川橋','天端の白壁']):
        text(im,37,269+i*176,a,18,True)
    text(im,38,1300,'写真に写らない背面・小口・脚の床下側は推定。橋の横方向の離隔は、側面・真上と書き出した頂点で照合した。',18,color=MUTED)
    save(im,'orthographic')


if __name__=='__main__':
    comparison_fences()
    comparison_bridge()
    bridge_structure()
    comparison_wall()
    axes()
