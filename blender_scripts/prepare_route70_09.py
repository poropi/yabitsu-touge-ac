"""依頼09の参照台帳。元の調査データは読み取り専用。"""
from pathlib import Path
import re,json
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1]; C=R/'export/compare/route70'; W=C/'work09'
def main():
    W.mkdir(parents=True,exist_ok=True)
    text=(R/'export/compare/route70_inventory_20261030/work/codex_request_09.md').read_text()
    groups={};a=None;place=''
    for line in text.splitlines():
        m=re.match(r'### \d+\. \*\*(r70_\w+)',line)
        if m:a=m[1];groups[a]=[]
        if line.startswith('- 場所 '):place=line[5:]
        m=re.search(r'(S[45]-\d+-\d+).*?画像 `([^`]+)`、(https://\S+)',line)
        if m and a:groups[a].append(dict(oid=m[1],image=m[2],url=m[3],place=place))
        if '説明：' in line and a and groups[a]:groups[a][-1]['description']=line.split('説明：',1)[1]
    inventory=json.loads((R/'export/compare/route70_inventory_20261030/objects_s4s5.json').read_text())
    byid={o['oid']:o for o in inventory['objects']}
    for group in groups.values():
        for ref in group:
            obj=byid.get(ref['oid'],{})
            sight=next((s for s in obj.get('sightings',[]) if s.get('image') and ref['image'].endswith(s['image'])),None)
            if sight:ref.update(camera=sight.get('camera'),photo_xy_hint=[sight.get('x'),sight.get('y_top'),sight.get('y_base')])
    (W/'references.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
    ff=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',18)
    for a,refs in groups.items():
        for base in range(0,len(refs),6):
            batch=refs[base:base+6];sheet=Image.new('RGB',(1600,550*((len(batch)+1)//2)),(238,238,232));d=ImageDraw.Draw(sheet)
            for j,x in enumerate(batch):
                xx=j%2*800;yy=j//2*550
                im=Image.open(R/x['image']);im.thumbnail((800,500));sheet.paste(im,(xx,yy+42))
                d.text((xx+8,yy+4),x['oid']+' '+x['place'][:45],font=ff,fill='black')
                d.text((xx+8,yy+24),Path(x['image']).name+' / Google Maps Street View',font=ff,fill='black')
            sheet.save(W/(a+f'_references_{base//6:02}.jpg'),quality=92)
    print({a:len(refs) for a,refs in groups.items()})
if __name__=='__main__':main()
