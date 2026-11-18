"""依頼08の参照情報と、写真を転用しない比較専用の一覧。"""
from pathlib import Path
import re,json
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]; D=R/'export/compare/route70'; W=D/'work'
text=(R/'export/compare/route70_inventory_20261030/work/codex_request_08.md').read_text()
assets={}; asset=part=None
for line in text.splitlines():
    m=re.match(r'### \d+\. \*\*(r70_\w+)\*\*',line)
    if m:asset=m[1];assets[asset]={}
    m=re.match(r'- \*\*(\w+)\*\*',line)
    if m and asset:part=m[1];assets[asset][part]=[]
    m=re.search(r'(S[45]-\d+-\d+).*?画像 `([^`]+)`.*?(https://\S+)',line)
    if m and asset and part:assets[asset][part].append(dict(oid=m[1],image=m[2],url=m[3]))
    if '説明：' in line and asset and part and assets[asset][part]:assets[asset][part][-1]['description']=line.split('説明：')[1]
W.mkdir(exist_ok=True,parents=True); (W/'references.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2))
f=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc',18)
for a,ps in assets.items():
    refs=[(p,x) for p,ls in ps.items() for x in ls[:2]]
    for b in range(0,len(refs),6):
        batch=refs[b:b+6];sheet=Image.new('RGB',(1200,440*((len(batch)+1)//2)),(235,235,228));d=ImageDraw.Draw(sheet)
        for i,(p,x) in enumerate(batch):
            xx=(i%2)*600; yy=(i//2)*440
            im=Image.open(R/x['image']);im.thumbnail((600,375));sheet.paste(im,(xx,yy+48))
            d.text((xx+10,yy+4),a+' / '+p,font=f,fill='black');d.text((xx+10,yy+25),x['oid']+' '+Path(x['image']).name,font=f,fill='black')
        sheet.save(W/(a+f'_reference_{b//6}.jpg'),quality=90)
print({a:len(p) for a,p in assets.items()})
