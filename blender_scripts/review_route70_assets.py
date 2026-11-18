"""依頼08の各部品・6方向を並べる。写真は配布テクスチャに使わない。"""
from pathlib import Path
import json,sys
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];C=R/'export/compare/route70';F='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc';font=ImageFont.truetype(F,16)
V=['front','left','right','back_estimated','top','persp']
for p in sorted((R/'assets/landmarks').glob('r70_*/asset.json')):
    info=json.loads(p.read_text());a=info['asset_id'];parts=info['parts']
    for base in range(0,len(parts),6):
        sub=parts[base:base+6];sheet=Image.new('RGB',(1440,285*len(sub)),(241,241,234));d=ImageDraw.Draw(sheet)
        for j,part in enumerate(sub):
            n=part['name'];d.text((7,j*285+2),f'{a} / {n} / {part["triangles"]} tris / '+str(part['dimensions_m'])+'m',font=font,fill='black')
            for i,v in enumerate(V):
                path=C/(a+'_'+n+'_'+v+'.png')
                if not path.exists():continue
                im=Image.open(path).convert('RGB');im.thumbnail((240,240));sheet.paste(im,(i*240,j*285+23));d.text((i*240+5,j*285+263),v,font=font,fill='black')
        sheet.save(C/(a+f'_review_{base//6}.jpg'),quality=90)
