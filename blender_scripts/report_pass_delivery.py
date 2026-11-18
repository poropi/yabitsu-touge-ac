"""依頼07の部品一覧、確認画像、納品報告を実ファイルから生成する。"""
from pathlib import Path
import json,math
from PIL import Image,ImageDraw,ImageFont,ImageOps
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CMP=ROOT/'export/compare/pass'
ASSETS=['pass_shop','pass_vending','pass_toilet','pass_lodge','pass_info_boards','pass_trail_stairs','pass_benches','pass_trail_posts','pass_bike_rack','pass_bus_stop','pass_brown_signs','pass_road_items','pass_utility_pole','pass_resthouse','pass_barriers','pass_site_equipment']
FONT='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
def font(n):return ImageFont.truetype(FONT,n)
def fit(path,size):
    im=Image.open(path).convert('RGB');return ImageOps.contain(im,size)
def place(canvas,path,box):
    x,y,w,h=box;im=fit(path,(w,h));canvas.paste(im,(int(x+(w-im.width)/2),int(y+(h-im.height)/2)))

def sheets(asset):
    views=[('front','正面 −Y'),('left','左側 −X'),('right','右側 +X'),('back_estimated','背面 +Y（隠れた面は推定）'),('top','真上 +Z'),('persp','斜め')]
    out=Image.new('RGB',(1440,970),(238,239,234));d=ImageDraw.Draw(out);d.text((20,10),asset+' ゲーム用NPZ再読込',font=font(26),fill=(28,40,34))
    for i,(v,label) in enumerate(views):
        p=CMP/(asset+'_'+v+'.png');assert p.exists(),p;x=(i%3)*480;y=55+(i//3)*450
        place(out,p,(x+4,y,472,405));d.text((x+14,y+408),label,font=font(19),fill=(30,40,35))
    out.save(CMP/(asset+'_views.jpg'),quality=92)

def part_sheets(asset,info):
    views=[('front','正面 −Y'),('left','左側 −X'),('right','右側 +X'),('back_estimated','背面 +Y（非可視部は推定）'),('top','真上 +Z'),('persp','斜め')]
    for p in info['parts']:
        prefix=asset+'_'+p['name']
        if not (CMP/(prefix+'_front.png')).exists():continue
        out=Image.new('RGB',(1260,810),(238,239,234));d=ImageDraw.Draw(out);d.text((14,8),prefix,font=font(21),fill=(28,40,34))
        for i,(v,label) in enumerate(views):
            path=CMP/(prefix+'_'+v+'.png');assert path.exists(),path;x=(i%3)*420;y=42+(i//3)*380
            place(out,path,(x,y,420,350));d.text((x+10,y+350),label,font=font(16),fill=(30,40,35))
        out.save(CMP/(prefix+'_views.jpg'),quality=92)
    if all((CMP/(asset+'_'+p['name']+'_front.png')).exists() for p in info['parts']):
        out=Image.new('RGB',(1320,215*len(info['parts'])+35),(242,242,236));d=ImageDraw.Draw(out)
        d.text((10,4),asset+' / 各部品の正面、左右、背面、真上、斜め',font=font(15),fill=(20,40,30))
        for row,p in enumerate(info['parts']):
            y=35+row*215;d.text((5,y),p['name'],font=font(15),fill=(20,40,30))
            for col,(v,_) in enumerate(views):place(out,CMP/(asset+'_'+p['name']+'_'+v+'.png'),(col*220,y+24,218,186))
        out.save(CMP/(asset+'_individual_review.jpg'),quality=92)

def overview(infos):
    byid={i['asset_id']:i for i in infos};buildings=['pass_shop','pass_toilet','pass_lodge','pass_resthouse']
    for group,names,cols,size in [('buildings',buildings,2,(700,580)),('small',[a for a in ASSETS if a not in buildings],4,(400,345))]:
        w,h=size;rows=(len(names)+cols-1)//cols;out=Image.new('RGB',(w*cols,rows*h+65),(241,242,235));d=ImageDraw.Draw(out)
        d.text((15,10),'依頼07 単体モデル / '+('建物' if group=='buildings' else '周辺設備と小物'),font=font(28),fill=(25,40,30))
        for i,a in enumerate(names):
            j=byid[a];x=i%cols*w;y=65+i//cols*h;place(out,CMP/(a+'_persp.png'),(x,y,w,h-42))
            tri=sum(p['triangles'] for p in j['parts']);d.text((x+10,y+h-39),f'{a}   {len(j["parts"])}部品 / {tri:,} tri',font=font(16 if group=='small' else 23),fill=(25,40,30))
        out.save(CMP/('pass_delivery_07_'+group+'.jpg'),quality=94)

def comparisons(asset,refs):
    rows=[]
    for photo,model,label in refs:
        rp=CMP/photo;mp=CMP/(asset+'_'+model+'.png')
        if rp.exists() and mp.exists():rows.append((rp,mp,label))
    if not rows:return
    out=Image.new('RGB',(1440,len(rows)*510+70),(240,241,236));d=ImageDraw.Draw(out)
    d.text((20,12),asset+' 参照と同じ側の比較（配置前の単体モデル）',font=font(25),fill=(28,40,34))
    for i,(rp,mp,label) in enumerate(rows):
        y=70+i*510;place(out,rp,(0,y,800,465));place(out,mp,(810,y,625,465))
        d.text((15,y+469),label,font=font(16),fill=(30,40,35));d.text((820,y+469),'自作モデル。焦点距離と撮影地点の厳密な一致は未実施。',font=font(15),fill=(30,40,35))
    out.save(CMP/(asset+'_reference_comparison.jpg'),quality=92)

REFS={
 'pass_shop':[('codex07/sv_shop_225.png','reference_shop225','Google Maps Street View / shop 225° / 2025-07'),('codex07/sv_shop_270.png','reference_shop270','Google Maps Street View / shop 270° / 2025-07')],
 'pass_vending':[('codex07/sv_shop_270.png','reference_shop270','Google Maps Street View / shop 270° / 2025-07')],
 'pass_toilet':[('codex07/sv_toilet_000.png','reference_toilet000','Google Maps Street View / toilet 000° / 2022-11'),('codex07/sv_toilet_315.png','reference_toilet315','Google Maps Street View / toilet 315° / 2022-11')],
 'pass_lodge':[('codex07/sv_shop_090.png','reference_shop090','Google Maps Street View / shop 090° / 2025-07')],
 'pass_info_boards':[('codex07/sv_toilet_045.png','reference_props','Google Maps Street View / toilet 045° / 2022-11')],
 'pass_trail_stairs':[('codex07/sv_toilet_090.png','reference_props','Google Maps Street View / toilet 090° / 2022-11')],
 'pass_benches':[('codex07/sv_lanes_045.png','reference_props','Google Maps Street View / lanes 045° / 2022-11')],
 'pass_trail_posts':[('codex07/sv_toilet_090.png','reference_props','Google Maps Street View / toilet 090° / 2022-11')],
 'pass_bike_rack':[('codex07/sv_u2_225.png','reference_props','Google Maps Street View / u2 225° / 投稿2024-07、画像下端2023-12')],
 'pass_bus_stop':[('codex07/sv_toilet_135.png','reference_props','Google Maps Street View / toilet 135° / 2022-11')],
 'pass_brown_signs':[('codex07/sv_south_180.png','reference_props','Google Maps Street View / south 180° / 2025-07')],
 'pass_road_items':[('codex07/sv_mid_045.png','reference_props','Google Maps Street View / mid 045° / 2025-07')],
 'pass_utility_pole':[('codex07/sv_lotn_000.png','reference_props','Google Maps Street View / lotn 000° / 2025-07')],
 'pass_barriers':[('codex07/sv_lotn_225.png','reference_props','Google Maps Street View / lotn 225° / 2025-07')],
 'pass_site_equipment':[('codex07/sv_toilet_000.png','reference_props','Google Maps Street View / toilet 000° / 2022-11')],
 'pass_resthouse':[('work07/reference_resthouse_210329095329_3.jpg','reference_official','秦野市公式写真 210329095329_3.jpg / 比較参照のみ')],
}

def report(infos):
    total=sum(sum(p['triangles'] for p in i['parts']) for i in infos);n=sum(len(i['parts']) for i in infos)
    lines=['# ヤビツ峠の単体オブジェクト納品07','', '2026-10-31。物の形とゲーム用データの納品。コース配置、全コースの組み立て、KN5変換、ゲームへのインストールは未実施。','',
      f'指定14分類と追加{len(infos)-14}分類、計{n}部品、{total:,}三角形。','',
      '確認一覧：[建物](pass_delivery_07_buildings.jpg)、[周辺設備と小物](pass_delivery_07_small.jpg)。部品ごとの寸法とファイルは各節の表に記載。','',
      '## 配置担当へ','',
      '各NPZはメートル、上方向+Z、正面−Y、原点はXY外接矩形の中央かつ最下点。fは三角形、uvはPNGの上端をv=0とする。',
      '複合建物の復元は部品のvにasset.jsonのassembly_offset_mを加える。独立した小物のpreview_offsetsは確認画像用の展示位置で、現地の配置位置ではない。',
      '正面の真北からの角度は後述の建物計測値を使う。NPZへの方位の焼き込みはしていない。','',
      '白い長い建物（pass_lodge）は、トイレを駐車場側から見た長辺と同じ棟である可能性が高い。根拠は同じ屋根輪郭、白壁、撮影地点からの位置関係と、航空写真に独立した第4の屋根を確認できないこと。両アセットを独立2棟として重ねて配置しない。ID別のデータは指定に合わせて供給する。','',
      '青い丸い買い物袋の印はGoogleマップの画面表示。実物のバス停としてモデル化していない。実在の縦長丸頭の時刻表柱と、島の灰色円板を別部品にした。','',
      '## 保存形式と参照資料','',
      '制作元はassets/landmarks/<asset-id>/<asset-id>.blend、asset.json、自作の画像。ゲーム用データはdata/processed/landmarks/<asset-id>/<部品名>.npzと同名.png。',
      'Google等の写真の画素は配布アセットへ貼り付けていない。木材、金属、石の手続き材質と、判読できた題字、商品列などの自作画像をDiffuse Colorだけで焼き付けた。照明や影は焼き付けていない。',
      '参照はcodex07/sv_<地点>_<向き>.png、sheet_<地点>.jpg、gsat19〜21.png。元写真は変更していない。撮影年の相違は各asset.jsonとwork07/props_inventory.jsonに記録。u2は投稿表示2024年7月、画像下端の撮影表記2023年12月。','',
      '## 寸法の根拠と未確認範囲','',
      '建物の平面は航空写真の縮尺バーと輪郭点で求めた投影寸法。現地の測量値ではない。角度は画面北上を前提に真北0°、時計回り。屋根の投影と地上の壁芯は異なり、樹木と画面UIで隠れた縁は推定としている。',
      '計測点、縮尺、読めた縁と推定区間は[building_measurements.json](work07/building_measurements.json)に保存。小物は[props_inventory.json](work07/props_inventory.json)の写真比率による推定で、原則±20〜30%、階段上部や電柱の全高は約±40%。','']
    lines.extend(['## 建物の平面と方位','',
      '| asset-id | 採用した平面寸法（m） | 正面方位 | 根拠 | 真上での照合 |',
      '|---|---|---:|---|---|',
      '| pass_shop | 主屋根の読取約5.7×4.0、下屋込み間口約8.5。モデル屋根外包8.5×4.59 | 59° | gsat21、5m/81px。背面は遮蔽。 | [重ね合わせ](pass_shop_aerial_overlay.jpg) |',
      '| pass_toilet | 読取9.23×12.37、屋根面9.2×12.4 | 166° | gsat20、10m/81px。北東角は補間。 | [重ね合わせ](pass_toilet_aerial_overlay.jpg) |',
      '| pass_lodge | 同じ外包を長壁正面にして12.4×9.2 | 256° | トイレと共有する輪郭候補。独立棟の寸法は未確定。 | [重ね合わせ](pass_lodge_aerial_overlay.jpg) |',
      '| pass_resthouse | 主屋軸9.0×6.3、屋根10.8×8.1、棟高7.1、FL1.0 | 197° | 航空写真の方位と市公式A-18/A-20の寸法。 | [重ね合わせ](pass_resthouse_aerial_overlay.jpg) |','',
      'レストハウスの赤線は明るい東屋根片面だけの読取で、全体輪郭ではない。写真のその片面は約6.6×8.8mに見え、公式屋根の半幅5.4mとの投影差が残る。斜め撮影と高さによる変形も考えられるが原因は確定できず、寸法付き一次図面を採用した。',''])
    for info in infos:
        aid=info['asset_id'];tri=sum(p['triangles'] for p in info['parts']);lines.extend([f'## {aid}','',f'計 {tri:,} 三角形。制作元：[blend](../../../assets/landmarks/{aid}/{aid}.blend)、[asset.json](../../../assets/landmarks/{aid}/asset.json)。', '',
          '| 部品名とファイル | 意味 | 三角形 | X幅 × Y奥行 × Z高さ（m） | 組立 offset_m |','|---|---|---:|---|---|'])
        for p in info['parts']:
            name=p['name'];dim=' × '.join(f'{x:.3f}' for x in p['dimensions_m']);off=', '.join(f'{x:.4f}' for x in p['assembly_offset_m']);base=f'../../../data/processed/landmarks/{aid}/{name}'
            meaning=p.get('meaning',info.get('part_info',{}).get(name,{}).get('meaning',name))
            detail=f' / [個別6方向]({aid}_{name}_views.jpg)' if (CMP/(aid+'_'+name+'_views.jpg')).exists() else ''
            lines.append(f'| {name}：[npz]({base}.npz) / [png]({base}.png){detail} | {meaning} | {p["triangles"]:,} | {dim} | ({off}) |')
        lines.extend(['','全て原点は根元中央、正面は−Y。'+info.get('assembly_notes','複数部品を組み立てる場合だけ上表のoffsetを加える。'),'', '寸法の根拠：'+str(info.get('dimension_basis','写真の比率推定。'))])
        if 'front_azimuth_deg' in info:lines.append('建物正面：真北から約'+str(info['front_azimuth_deg'])+'°。')
        if 'roof_footprint_m' in info:lines.append('モデルの屋根の平面寸法：'+str(info['roof_footprint_m'])+' m。')
        lines.extend(['','写真との差と推定箇所：',''])
        for a in info.get('approximations',[]):lines.append('- '+a)
        lines.extend(['','確認画像：[6方向]('+aid+'_views.jpg)、[写真との比較]('+aid+'_reference_comparison.jpg)。', '','参照写真：'+', '.join('['+r+']('+r+')' for r in info.get('refs',[]))+ '。',''])
    lines.extend(['## 階段の並べ方','',
      'timber_stepは1段。蹴上げ0.16 m、踏面の送り0.32 mを想定し、段iを(0, i×0.32, i×0.16)へ置く。上る方向は+Y、入口側が−Y。下部の写真で形を確認できるのは約8〜11段で、上部までの総段数は未確定。',
      '手すり1区間は段4個分。柱間隔と次区間への送りは1.28m、次区間との高低差は0.64m。左はX=−0.8m、右はX=+0.8mへ置く。各部品の原点は中央なので左右位置を別途加える。左の黒い金属手すりは写真で確認した形。右の連続手すりと木製版は写真で確かめられず、指定に合わせた組立用の推定部品。斜面や長大な階段全体は作っていない。','',
      '## レストハウスの追加一次資料','',
      '[秦野市公式施設紹介](https://www.city.hadano.kanagawa.jp/kanko/spot-shisetsu/2/1343.html)の写真を参照。木造平屋と延床面積59.29㎡が記載されている。',
      '公式寸法図：[市会議資料p13/A-20立面、p14/A-18平面](https://www.city.hadano.kanagawa.jp/material/files/group/2/020507_bucyokaigi_jiansyo_all.pdf)。ローカルの[立面画像](work07/reference_resthouse_elevation.png)と[平面画像](work07/reference_resthouse_plan.png)も同梱。',
      '正面写真：[210329095329_3.jpg](https://www.city.hadano.kanagawa.jp/material/images/group/51/210329095329_3.jpg)、長側面：[210329091500_2.jpg](https://www.city.hadano.kanagawa.jp/material/images/group/51/210329091500_2.jpg)。比較参照のみで、テクスチャには使っていない。','',
      '指定SVのlanes 000/315とsouth 000/315、その周辺の提供視点および市と運営者の公式サイトを調べた。別の近接panoidは取得できなかったため、指定SVに加えて上記の公式写真と図面を使った。参照した指定SV：[lanes 000](https://www.google.com/maps/@35.4278881,139.2193927,3a,90y,0h,85.87t/data=!3m4!1e1!3m2!1se6F2VbzDhC2GsSwe3hIqNg!2e0)、[south 000](https://www.google.com/maps/@35.4277131,139.2194595,3a,90y,0h,85.87t/data=!3m4!1e1!3m2!1syenVqobThM2mn6cKNK0ymw!2e0)。','',
      '## 追加で見つけた物','',
      'pass_barriers：売店横のA脚バリケード。sv_lotn_225、sv_shop_315で確認。',
      'pass_site_equipment：電柱根元の機器箱と低い囲い、島の横長看板、対岸の黄注意板、トイレ正面の小屋根付き掲示物、歩道脇の緑円筒。参照は各asset.jsonの部品説明に記載。用途や本文が写真で確定できない物は断定していない。','',
      'pass_utility_poleのpole_opposite_red：道路対岸の赤褐色の細柱。sv_toilet_225とsv_lanes_270で確認。売店脇と北入口のコンクリート柱2種とは別。機器の全貌は未確認。','',
      '## 検証と未実施の範囲','',
      '各NPZの三角形数、有限座標、非ゼロ面積、頂点参照、UV範囲、寸法、原点、PNGの存在と色をverify_pass_assets.pyで検査。制作元は別のBlenderプロセスで開き直し、主要コレクション、寸法、三角形数、画像の同梱を検査。',
      '確認画像は元のblendからではなく、納品NPZとPNGを読み戻して描画。6方向と写真に対応する側から確認した。背面や樹木で隠れる箇所は、写真で検証できた形と区別して各節に明記。',
      '看板の国民宿舎の白文字へNPZ上でraycastし、自作元画像と照合。uvの上端0が正しく、追加反転すると不一致になることを確認（[検査結果](pass_shop_uv_validation.json)）。',
      '片面描画の回帰検査check_pass_surface_orientation.pyで、W形ビームの表裏、北の裏三角妻、左右の木目軒裏の法線を検査。最終レビューで見つかった3件は、修正前の不合格から修正後の合格へ変わることを確認（[結果](pass_surface_orientation_validation.json)、[修正前出力](work07/surface_orientation_red.log)、[修正後出力](work07/surface_orientation_green.log)）。',
      '保存直前のサムネイル停止は既存のtests/test_blender_save_no_thumbnail.pyを実行。保護対象はwork07/protected_hashes.jsonの内容ハッシュで照合。',
      '通常CLIはMetal初期化でSIGSEGVになったため、先行作業と同じBlender専用MCPの背景プロセスを使用。制作や描画そのものの代替はしていない。',
      'コース全体の配置、路面との接地、衝突、KN5材質、実ゲームの照明での見え方は、今回は単体モデル納品の依頼のため未検証。','',
      '再生成スクリプト：build_pass_shop.py、build_pass_props.py、建物用スクリプト（各asset.jsonのgenerator）。共通書出しpass_common.py、再読込検査と描画verify_pass_assets.py、UV検査check_pass_uv.py、片面法線検査check_pass_surface_orientation.py、航空写真比較compare_pass_aerial.py、報告と比較図report_pass_delivery.py。全てblender_scripts/に保存。','',
      'Obsidianはプロジェクトノートと関連ルールを読み取った。Vaultは今回の書込許可範囲外のため更新できず、反映用内容を[work07/obsidian_pending.md](work07/obsidian_pending.md)に保存。',''])
    (CMP/'pass_delivery_07.md').write_text('\n'.join(lines)+'\n')
    manifest={'assets':[{'asset':i['asset_id'],'parts':i['parts']} for i in infos],'asset_count':len(infos),'part_count':n,'triangles':total}
    (CMP/'pass_delivery_07.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

def main():
    infos=[]
    for a in ASSETS:
        p=ROOT/'assets/landmarks'/a/'asset.json';assert p.exists(),p
        info=json.loads(p.read_text());infos.append(info);sheets(a);part_sheets(a,info);comparisons(a,REFS[a])
    overview(infos);report(infos)
if __name__=='__main__':main()
