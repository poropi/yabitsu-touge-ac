"""依頼09の納品表・参照比較・多方向シートを実ファイルから生成。"""
from pathlib import Path
import json,math
from PIL import Image,ImageDraw,ImageFont,ImageOps
R=Path(__file__).resolve().parents[1];C=R/'export/compare/route70';W=C/'work09'
ASSETS=['r70_bridges','r70_tunnels','r70_buildings']
FONT='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
def font(s):return ImageFont.truetype(FONT,s)
def drawpic(sh,path,box):
    x,y,w,h=box;im=ImageOps.contain(Image.open(path).convert('RGB'),(w,h));sh.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
def text(d,xy,s,size=18):d.text(xy,s,font=font(size),fill=(28,40,33))

def views(j):
    a=j['asset_id'];result=[]
    for start in range(0,len(j['parts']),6):
        pp=j['parts'][start:start+6];sh=Image.new('RGB',(1440,260*len(pp)+50),(236,237,232));d=ImageDraw.Draw(sh)
        text(d,(8,5),a+' / 正面・左・右・背面（非可視部は推定）・真上・斜め',19)
        for row,p in enumerate(pp):
            yy=50+row*260
            for col,v in enumerate(['front','left','right','back_estimated','top','persp']):drawpic(sh,C/f'{a}_{p["name"]}_{v}.png',(col*240,yy,240,235))
            text(d,(8,yy+236),p['name']+' / '+str(p['triangles'])+' tri',14)
        path=C/f'{a}_views_09_{start//6+1:02}.jpg';sh.save(path,quality=91);result.append(path.name)
    sh=Image.new('RGB',(1200,350*math.ceil(len(j['parts'])/4)+60),(236,237,232));d=ImageDraw.Draw(sh)
    text(d,(12,10),a+' / ゲーム用NPZをBlenderで読み戻した単体モデル',22)
    for i,p in enumerate(j['parts']):
        x=i%4*300;y=60+i//4*350;drawpic(sh,C/f'{a}_{p["name"]}_persp.png',(x,y,300,300))
        text(d,(x+6,y+302),p['name'],13);text(d,(x+6,y+322),f'{p["triangles"]:,} tri / '+'×'.join(f'{v:.2f}' for v in p['dimensions_m'])+'m',12)
    path=C/f'{a}_overview_09.jpg';sh.save(path,quality=92)
    return dict(overview=path.name,views=result)

def compare(rows,a):
    result=[]
    for start in range(0,len(rows),4):
        batch=rows[start:start+4];sh=Image.new('RGB',(1440,465*len(batch)+58),(237,238,231));d=ImageDraw.Draw(sh)
        text(d,(12,8),a+' / 左：Google Street View　右：同じ側からの単体モデル',23)
        for i,(photo,model,label,roi) in enumerate(batch):
            y=58+i*465
            if roi:
                drawpic(sh,R/photo,(0,y,440,385));im=Image.open(R/photo).crop(roi);im=ImageOps.contain(im.convert('RGB'),(450,365));sh.paste(im,(450+(450-im.width)//2,y+(385-im.height)//2));drawpic(sh,C/model,(905,y,535,385))
            else:
                drawpic(sh,R/photo,(0,y,800,390));drawpic(sh,C/model,(805,y,635,390))
            text(d,(10,y+391),label[:115],16)
            text(d,(10,y+417),'© Google / 参照専用。単体の可視面を比較。撮影位置・透視・周辺地形は厳密一致ではない。',15)
        path=C/f'{a}_comparison_09_{start//4+1:02}.jpg';sh.save(path,quality=93);result.append(path.name)
    return result

def comparisons(infos):
    result={}
    j=infos[0];rows=[]
    for l in j['locations']:
        r=next((r for r in j['references'] if r['oid'] in l['references']),None)
        if not r:continue
        rows.append((r['image'],f'r70_bridges_location_{l["section"]}_{l["chainage_m"]}_reference.png',f'{l["section"]} {l["chainage_m"]}m / {r["oid"]} / {Path(r["image"]).name}',None))
    result[j['asset_id']]=compare(rows,j['asset_id'])
    j=infos[1];rows=[]
    for p in j['parts']:
        if 'portal' not in p['name']:continue
        r=p['references'][0];rows.append((r['image'],f'r70_tunnels_{p["name"]}_reference.png',r['oid']+' / '+p['meaning'],None))
    result[j['asset_id']]=compare(rows,j['asset_id'])
    obs=json.loads((W/'building_observations.json').read_text());rows=[]
    for b in obs['buildings']:
        used=set()
        for ref in b['references']:
            photo=ref['image'];fn=Path(photo).name
            if photo in used:continue
            used.add(photo);roi=b['photo_roi_xyxy'].get(fn)
            rows.append((photo,f'r70_buildings_{b["model_name"]}_{Path(photo).stem}_reference.png',b['model_name']+' / '+fn+' / 中央は参照部拡大',roi))
    result['r70_buildings']=compare(rows,'r70_buildings');return result

def main():
    infos=[json.loads((R/'assets/landmarks'/a/'asset.json').read_text()) for a in ASSETS]
    images={j['asset_id']:views(j) for j in infos};comps=comparisons(infos)
    for a in ASSETS:images[a]['comparisons']=comps[a]
    lines=['# ヤビツ峠→宮ヶ瀬北原 単体モデル納品09','',
      '2026-10-31。橋・トンネル・建物の制作元とNPZ/PNGの納品。コース配置、コース組み立て、KN5変換、ゲームへのインストールは行っていない。','',
      f'3 asset-id、計{sum(len(j["parts"]) for j in infos)}部品、{sum(p["triangles"] for j in infos for p in j["parts"]):,}三角形。建物は22記録を19棟案へ整理した。重複候補の統合は写真からの推定であり、独立棟数の実測ではない。','',
      '## 配置担当へ','',
      '- vはm、+Z上、正面−Y。原点は各部品のXY外接矩形中央・最下点。建物の主屋の中心と原点は、外階段・庇などが非対称な場合に異なる。主屋中心は `-source_anchor_m[0:2]`。',
      '- 各blendは部品別のコレクションを持ち、各部品を同じローカル原点に置いている。最初のコレクションだけ表示し、ほかは表示を切り替えて編集する。全コレクションを一度に表示すると部品が重なる。',
      '- fは三角形、uvは画像上端v=0。Blenderの制作元では1-vで読み込む。全画像は自作1024×1024 RGBA。参照写真の画素を配布素材へ転用していない。',
      '- 欄干は+X反復、区切り中央に支柱1本。端面を突合せ、最終端の柱だけ別部品で足す。継目に支柱を2本重ねない。長さ3mの擬木以外は2m。端数長は最後の区切りを長手方向だけ調整する。',
      '- 青宇治橋・赤アーチ全構造は長手+Y、進入端−Y。路面は含まない。路面の高さは原点からZ=0.7m。床下横桁の上へコース路面を置く。',
      '- 内壁は+Xに20m、原点から両端−10m/+10m。坑門の奥+Yへつなぐ際はZ軸+90°回す。断面幅と高の異なる3系列を混ぜない。端数部品で実長へ合わせる。',
      '- S4 6093/6168、S5 9813/9938、10513/10588、10937〜11162付近には同一橋または隣接旧橋の重複候補がある。台帳の地点差を橋長と見なさず、場所を確定してから置く。','',
      '## 写真・資料と未確認範囲','',
      '指定の保存済みGoogle Street View画像を読み取った。写真の年月・panoid・カメラ位置・向き・URLは各asset.jsonとwork09/references.jsonに記録。Google Maps URLをブラウザで新規に巡回してはいない。表のURLは実際に読んだローカル画像に対応する参照URLである。',
      '大洞トンネルと吹風トンネルの名板は拡大画像で判読した。「吹風」は拾い出しの仮称をそのまま採用したものではない。塩水隧道の同定は県道70号の3本の位置順と小断面からの推定で、名板を作っていない。',
      '写真が地形・樹木・車に隠れる部分は確認できない。建物の背面・屋根奥・奥行きは推定。見えない側の窓を一律に増やさず、基本は無窓の壁として補った。',
      '参照方向画像は見える面・進行方向を合わせた単体比較である。建物の撮影距離・焦点距離・周辺地形の厳密な再構成はしていない。比較の仮路面は撮影専用で、NPZや制作元に含まない。',
      'ゲーム内の片面描画や当たり判定、夜間点灯は配置・KN5変換後の確認事項であり、この納品でゲーム動作確認済みとはしていない。','',
      '寸法の一次資料：',
      '- [神奈川県 橋梁定期点検結果等一覧](https://www.pref.kanagawa.jp/documents/114236/kyouryou_tenkenkekka.pdf) PDF4・9・10頁。橋長と参照写真の対応根拠は `work09/bridge_name_evidence.json`。',
      '- [清川村 統計要覧2021](https://www.town.kiyokawa.kanagawa.jp/material/files/group/5/r3toukeiyouran.pdf) PDF53・54頁、紙面33・34頁。釜田川橋99m・幅員8.8m、青宇治橋20mなど。',
      '- [神奈川県 令和6年度所管施設等の概要](https://www.pref.kanagawa.jp/documents/4656/r6shisetsu_gaiyou.pdf#page=25) PDF25頁、紙面22頁。大洞50.4m、塩水21.4m、吹風244m。車道幅・建築限界と坑口開口寸法は同じ量ではない。別年度資料には塩水25mの値もあり、本納品はR6県概要21.4mを採用。',
      '- [LIXIL 建具規格表](https://www2-t.lixil.co.jp/rp/dfw/lixila1330ex/metalproductinfo/files/technical/document/SDC0003825.pdf) p1。喫茶店の隠れた側面に使った推定窓1.69×1.17mの寸法参考。現地の同製品採用を意味しない。',
      '建物全体と橋の高さ・柱間隔などは、写真の屋根・建具・道路幅との比率と依頼文の目安からの推定である。通常±20〜30%、遠景・遮蔽部分は±50%程度の不確かさを見込む。現地実測・写真測量を行った値ではない。','']
    lines+=['寸法を照合した部分は、欄干では路肩の根元から笠木上端（高さ）と隣り合う支柱の中心間（区切り）、親柱では路肩から天端と名板の幅、坑門では左右の開口内縁と路面からアーチ頂部である。これらの数値はピクセルからの測距を行った値ではなく、依頼文の目安と目視による比率推定である。公式寸法を採用した橋長だけを別欄に分けた。',
      '建物の読み取り範囲は下記の各部品に画像座標で記した。矩形は外形や建具を確認した範囲であり、矩形のピクセル寸法をそのまま実寸へ換算したものではない。','']
    for j in infos:
        a=j['asset_id'];lines += [f'## {a}','',f'制作元：`assets/landmarks/{a}/{a}.blend`、`asset.json`、自作PNG。',f'ゲーム用：`data/processed/landmarks/{a}/`。再生成：`{j["generator"]}`。','',
          f'計{len(j["parts"])}部品・{sum(p["triangles"] for p in j["parts"]):,}三角形。',f'[一覧画像]({images[a]["overview"]})。','',
          '| 名前・意味 | NPZ / PNG | 三角形 | 外接寸法 X×Y×Z (m) | 反復長 |', '|---|---|---:|---|---|']
        for p in j['parts']:
            n=p['name'];dim=' × '.join(f'{v:.3f}' for v in p['dimensions_m']);span=str(p.get('repeat_length_m','—'))
            lines.append(f'| `{n}`：{p["meaning"]} | `{n}.npz` / `{n}.png` | {p["triangles"]:,} | {dim} | {span} |')
        lines+=['','### 部品ごとの根拠・差・推定','']
        for p in j['parts']:
            lines += [f'**{p["name"]}**：{p["dimension_basis"]}']
            if p['references']:lines.append('参照：'+', '.join(f'{r["oid"]}（{Path(r["image"]).name}）' for r in p['references'])+'。')
            lines.append('差・推定：'+'／'.join(p['approximations'])+'。')
            if a=='r70_buildings':lines.append('写真で確認した要素：'+'、'.join(p['observed_features'])+'。')
            if a=='r70_buildings':lines.append('読み取り範囲（画像左上原点の x0,y0,x1,y1 px）：'+'、'.join(f'{fn} {roi}' for fn,roi in p['photo_roi_xyxy'].items())+'。')
            lines.append('')
        lines+=['### 確認画像','']
        lines.extend(f'- [{f}]({f})' for f in images[a]['views']+images[a]['comparisons']);lines.append('')
    bridge=infos[0];lines+=['## 橋の場所別割当','',
      '左右はヤビツ峠→宮ヶ瀬の進行方向。長さは左右各側の欄干部の目安である。公式橋長を確認できた橋は取付ガードレールを除く橋長を採用し、その他は写真推定。親柱の必要本数・最終位置は配置担当が現地形に合わせる。',
      '赤アーチは釜田川橋の有力候補。正式銘板の全文は元54×23pxで細画が潰れ、写真だけでは判読し切れなかったため文字を入れていない。全長99mはアーチ単体の支間ではない。アーチ支間82mは編集資料と写真からの推定、立上り9mも推定。',
      '青宇治橋は写真で見える車道脇構造を幅約4.4mに近似した。管理資料の幅員7.5mの断面内訳を確認できず、同じ寸法としてモデル幅に適用していない。トラスの断面・床下桁は近似で、寸法一致を主張しない。','',
      '| 場所 | 左欄干 | 右欄干 | 親柱 | 採用欄干長/側 (m) | 公式橋長 (m) | 根拠・重複注意 |','|---|---|---|---|---:|---:|---|']
    for l in bridge['locations']:
        right=l['right_rail']+(' ＋外側旧橋 '+l['right_outer_rail'] if l.get('right_outer_rail') else '')
        lines.append(f'| {l["section"]} {l["chainage_m"]}m | `{l["left_rail"]}` | `{right}` | {l["end_post"] or "なし/未確認"} | {l["length_m"]} | {l.get("official_bridge_length_m","未特定")} | {l["notes"]}／{l["length_basis"]} |')
    lines+=['','## トンネルの場所別割当','', '| 場所 | 坑門 | 内壁 | 実長 (m) | 組合せ | 根拠・推定 |','|---|---|---|---:|---|---|']
    for l in infos[1]['locations']:
        portals=l['portal']+(' / '+l['other_portal'] if l.get('other_portal') else '')
        lines.append(f'| S5 {l["chainage_m"]}m | {portals} | {l["lining"]} | {l["length_m"]} | {l["assembly"]} | {l["name"]}。{l["name_basis"]} {l["unseen"]} |')
    lines+=['','## 建物の場所別棟数と寸法','',
      '下表は19個体案。庇・軒・外階段の張り出しを含む外接寸法は部品表、主屋の平面推定は下表を使う。棟数をまとめた記録も各棟を別NPZにした。','',
      '| 場所 | 棟 | 対応する拾い出しID | 棟数 | 主屋平面 (m) | 外接X×Y×Z (m) | 推定・重複 |','|---|---|---|---:|---|---|---|']
    for l in infos[2]['locations']:
        lines.append(f'| {l["section"]} {l["chainage_m"]}m | `{l["building"]}` | {", ".join(l["source_object_ids"])} | 1 | {"×".join(map(str,l["footprint_dimensions_m"]))} | {"×".join(f"{v:.2f}" for v in l["bounds_dimensions_m"])} | {"／".join(l["notes"]) or "非可視面・奥行きは推定"} |')
    lines+=['','### 建物の統合と除外','',
      '- S4 0358/0360は同じ小屋の別面候補として1棟。S4 0380/0381は煉瓦側面と白い妻面が同じ片流れ棟の候補として1棟。白い妻面のピンク円はGoogleの宿泊POIなので作っていない。',
      '- S4 0413の小屋群は写真に分かれて見える3棟をa/b/cとして別々に制作した。4棟目の独立輪郭は確認できず、棟を増やしていない。',
      '- S5 0867/0868は同じプレハブ候補として1棟。S5 1018、1020-04、1021-10の2階部分は同じ木工事務所候補として1棟。1019と1021-10の低層部分は同じ倉庫候補として1棟。',
      '- 遠景S4 0298の建物と既存pass_resthouseの同一性は、樹木と低解像度で輪郭全体を確認できず未確定。既存モデルを重ねて置かない。','',
      '## 保存・検証','',
      '制作元blendを別のBlender背景プロセスで読み直し、部品数・構成部品・三角形・外接寸法・画像依存を確認した。サムネイル停止は保存の直前に実施。写真素材をblendへ埋め込んでいない。',
      '配列検査は全68部品のv/f/uv、有限値、退化面、頂点上限、中心原点、画像形式・制作元との一致、反復端点を確認。内壁の内向き法線・断面一致、坑口を塞ぐ面の有無、片流れ側壁の外向き法線、木工事務所の扉・踊り場・階段の接続、名板1字のUV縦横比も検査した。',
      '追加した検査は修正前の実データで `AssertionError: (s4_1995_low_hut, 片流れ妻壁の法線)` と `AssertionError: (s5_10663_portal, 個別文字の縦横UV比)` を検出し、修正後に通過した。故意に壊したUV・退化面・負の接地・原点ずれ・予算超過の5条件も検出した。',
      '検証ログ：`route70_validation_09.json`、各`r70_*_npz_validation_09.json`、`r70_*_source_validation.json`。既存テスト `tests/test_blender_save_no_thumbnail.py` は1 passed。コースやゲーム用の総合テストは、配置・組込みを変更していないため実施していない。','',
      '## 全参照ファイルとURL','']
    seen=set()
    for j in infos:
        for r in j['references']:
            if (r['oid'],r['image']) in seen:continue
            seen.add((r['oid'],r['image']));cam=r.get('camera') or {}
            lines.append(f'- {r["oid"]}：`{r["image"]}`、撮影 {cam.get("date","依頼文では2025年7月、一部2022年11月")}、panoid `{cam.get("pano","URL参照")}`、方位 {cam.get("heading_deg","URL参照")}°。[対応するStreet View]({r["url"]})。')
    lines += ['','Obsidian更新は今回の書き込み許可範囲外のため未実施。追記案は `work09/obsidian_pending.md` に保存した。','']
    total_parts=sum(len(j['parts']) for j in infos)
    body='\n'.join(lines).replace('全68部品',f'全{total_parts}部品')
    # 表とリストの構造を保ち、本文だけ一文一行にする。
    formatted=[]
    for line in body.splitlines():
        if line.startswith('- ') and formatted and formatted[-1] and not formatted[-1].startswith('- '):formatted.append('')
        if line and not line.startswith(('|','- ','#')):line=line.replace('。','。\n').rstrip()
        formatted.append(line)
    (C/'route70_delivery_09.md').write_text('\n'.join(formatted))
    out=dict(assets=infos,images=images,totals=dict(parts=sum(len(j['parts']) for j in infos),triangles=sum(p['triangles'] for j in infos for p in j['parts'])))
    (C/'route70_delivery_09.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(out['totals']);print({a:len(images[a]['comparisons']) for a in ASSETS})
if __name__=='__main__':main()
