"""依頼08の台帳、写真比較、確認画像一覧、納品報告。"""
from pathlib import Path
import json,math,re
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from build_route70_assets import ASSETS
R=Path(__file__).resolve().parents[1];C=R/'export/compare/route70';W=C/'work';F='/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc';font=ImageFont.truetype(F,17)
REFS={
'r70_curve_mirror':('round_1','R70-0338_b',(245,220,480,800)),
'r70_utility_pole':('concrete_transformer','R70-0300_b',(740,180,880,540)),
'r70_pylon':('tower','R70-0302_f',(650,0,930,310)),
'r70_lights':('arm','R70-0962_f',(400,160,640,600)),
'r70_km_post':('post','R70-0641_f',(1150,390,1390,760)),
'r70_delineator':('pole','R70-0723_f',(130,530,480,920)),
'r70_road_signs':('route_hexagon_place','R70-0557_b',(410,300,550,670)),
'r70_boards':('post_board_bosco','R70-0411_f',(150,500,550,1000)),
'r70_guardrail':('w_beam_2m','R70-0386_f',(0,500,780,1000)),
'r70_fence':('wire_mesh_concrete','R70-0541_f',(950,70,1600,430)),
'r70_retaining_wall':('stone','R70-0385_f',(970,480,1600,1000)),
'r70_drain':('grating','R70-0617_f',(460,700,700,930)),
'r70_construction':('drum','R70-0737_f',(40,490,390,850)),
'r70_gate':('bollard','R70-0902_f',(80,500,670,810)),
'r70_misc':('vending','R70-1020_f',(1160,410,1400,650)),
}
# 読み取り済み写真のうち原型の直接根拠になる画像を部品別に記録。
OID={
'round_1':'S4-0337-01','round_2':'S5-0582-01','oval_1':'S5-0934-01','oval_1_twin_post':'S4-0361-03',
'arm_arc':'S5-0981-01','wall':'S5-0784-04','regulatory_double':'S5-0993-01','route_crossroad_64':'S5-1022-01','route_hexagon_place':'S5-0557-01','warning_diamond_with_text':'S4-0385-01',
'post_board_forest':'S4-0315-01','post_board_wall_cafe':'S4-0382-02','post_board_bosco':'S4-0411-04','stand_board':'S5-0782-01','stand_board_slow':'S5-0786-02','stand_board_coffee':'S4-0382-03','stand_board_bridge_parking':'S5-0948-07','parking_clearance_notice':'S5-0777-02','notice_board':'S4-0315-02','notice_board_aoyamaso':'S4-0380-02','notice_board_led':'S4-0545-01','construction_board':'S5-0851-01','construction_board_bridge':'S5-0737-05','small_plate_flick':'S4-0455-01','small_plate_parking':'S4-0459-05','small_plate_attention':'S4-0482-02','nobori':'S5-0890-02',
'wire_mesh_concrete':'S4-0541-02','wire_mesh_orange':'S5-0590-02','wire_mesh':'S5-0771-01','wood':'S5-0652-02','wood_mesh':'S5-0755-02','pipe_mesh':'S5-0980-02','pipe_end':'S4-0509-01',
'wood_A':'S4-0381-03','barricade_pipe':'S4-0490-02','barricade_panel':'S5-0771-03','gate_green':'S5-0738-01','gate_wood_x':'S4-0459-04','bollard_block':'S5-0749-04','bollard_u':'S5-0878-02','equipment_box_small':'S5-0940-03','equipment_box_pole':'S5-1004-02','log_pile_scrap':'S5-0864-07','log_stool':'S4-0486-03','concrete_steps':'S5-0799-03','stairs_rail':'S5-0809-04',
}
refs=json.loads((W/'references.json').read_text());flat={x['oid']:x for ps in refs.values() for ls in ps.values() for x in ls}
infos=[json.loads((R/'assets/landmarks'/a/'asset.json').read_text()) for a in ASSETS]
manifest=[]
for info in infos:
    a=info['asset_id'];part,imname,rect=REFS[a]
    info['comparison_reference']={'part':part,'image':imname+'.jpg','crop_pixels':rect,'camera_match':'同じ面が見える向きの近似。写真測量による校正はしていない。'}
    # フル写真はGoogle表示を保持。右は比較用切抜き、右端は同じ面を向けた原型。
    sheet=Image.new('RGB',(1280,510),(239,239,232));d=ImageDraw.Draw(sheet)
    photo=Image.open(R/'export/compare/route70_inventory_20261030/images/sv'/(imname+'.jpg')).convert('RGB');full=photo.copy();full.thumbnail((640,400));sheet.paste(full,(0,65));crop=photo.crop(rect);crop.thumbnail((300,390));sheet.paste(crop,(640+(300-crop.width)//2,65+(390-crop.height)//2))
    model=Image.open(C/(a+'_'+part+'_reference.png')).convert('RGB');model.thumbnail((320,390));sheet.paste(model,(950,85))
    d.text((10,8),a+' / '+part,font=font,fill='black');d.text((10,32),'Google Street View: '+imname+' / 確認範囲 '+str(rect),font=font,fill='black')
    d.text((650,475),'参照部分',font=font,fill='black');d.text((950,440),'同じ面を見る向きの近似',font=font,fill='black');d.text((950,465),'単体原型、現地配置ではない',font=font,fill='black')
    sheet.save(C/(a+'_reference_comparison.jpg'),quality=92)
    for p in info['parts']:
        if a=='r70_utility_pole' and p['name']=='wood':p['direct_reference']=flat['S5-0806-02']
        elif a=='r70_gate' and p['name']=='gate':p['direct_reference']=flat['S4-0486-04']
        elif p['name'] in OID and OID[p['name']] in flat:p['direct_reference']=flat[OID[p['name']]]
        else:p['direct_reference']=next(iter(p['references']),None)
        p['preview_views']={v:a+'_'+p['name']+'_'+v+'.png' for v in ['front','left','right','back_estimated','top','persp','reference']}
        for file in [p['name']+'.npz']+p['textures']:
            path=R/'data/processed/landmarks'/a/file;manifest.append(str(path.relative_to(R)))
    (R/'assets/landmarks'/a/'asset.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    manifest += [str(path.relative_to(R)) for folder in [R/'assets/landmarks'/a,R/'data/processed/landmarks'/a] for path in folder.iterdir() if path.is_file() and path.suffix in ['.blend','.json','.png','.npz']]
summary=[dict(asset_id=i['asset_id'],parts=len(i['parts']),triangles=sum(p['triangles'] for p in i['parts']),names=[p['name'] for p in i['parts']]) for i in infos]
(C/'route70_delivery_08.json').write_text(json.dumps({'assets':summary,'parts':sum(x['parts'] for x in summary),'triangles':sum(x['triangles'] for x in summary),'files':sorted(set(manifest))},ensure_ascii=False,indent=2)+'\n')
# 15assetを小さく一望できる完成画像。全文字を見るには個別reviewシートを使う。
for page in range(2):
    batch=infos[page*8:(page+1)*8];sht=Image.new('RGB',(1280,370*math.ceil(len(batch)/2)),(239,239,232));dd=ImageDraw.Draw(sht)
    for j,info in enumerate(batch):
        a=info['asset_id'];n=REFS[a][0];x=j%2*640;y=j//2*370;im=Image.open(C/(a+'_'+n+'_persp.png')).convert('RGB');im=im.resize((320,320));sht.paste(im,(x+160,y+28));dd.text((x+8,y+2),a,font=font,fill='black');dd.text((x+8,y+348),f'{len(info["parts"])}部品 / {sum(p["triangles"] for p in info["parts"])}三角形',font=font,fill='black')
    sht.save(C/(f'route70_delivery_08_overview_{page+1}.jpg'),quality=90)
md=['# 依頼08の納品報告','',f'2026-10-31。15 asset-id、{sum(x["parts"] for x in summary)}部品、{sum(x["triangles"] for x in summary):,}三角形。','',
'単体の制作元とゲーム用NPZ/PNGを指定先へ保存した。','コースへの配置、KN5化、インストールは今回の範囲に含めていない。','',
'## 保存形式と使い方','',
'ゲーム用は `data/processed/landmarks/<asset-id>/<部品名>.npz` と同名の `.png`。','制作元は `assets/landmarks/<asset-id>/<asset-id>.blend`、`asset.json`、自作画像。','各blendは部品ごとにコレクションを分け、最初の部品以外を非表示にしている。','',
'NPZのキーは `v`、`f`、`uv`。単位m、上+Z、正面-Y、三角形。','`uv`のv=0はPNGの上。Blenderへの読込では一度だけ1-vへ変換する。','単体の原点は支持部の足元中央、反復物は始端支柱の足元。柱なしの板と壁灯の原点は下表に個別記載した。','',
'色違いと表示面の画像は、本体と同じ1024×1024のアトラスである。','画像をそのまま差し替えられ、柱や裏面のUVは変わらない。','`*_art.png`は制作元の表示原稿であり、そのままゲーム用PNGと交換する画像ではない。','透過網はRGBA。配置側で両面描画とアルファテストを指定する。','',
'キロポストの `post_face_20.5.png` など7枚は距離数字だけを差し替える。','26.5と28.0の元の読み取りは不確かなため、設置位置の距離を確定した記録として扱わない。','擁壁の `*_surface.png` は両方向につながる1024px画像で、ゲーム用アトラスとは別に付属する。','',
'## 一覧','', '| asset-id | 部品数 | 三角形合計 | 制作元 |', '|---|---:|---:|---|']
for i in summary:md.append(f'| {i["asset_id"]} | {i["parts"]} | {i["triangles"]:,} | `assets/landmarks/{i["asset_id"]}/` |')
md += ['', '## 寸法の根拠', '',
'現地実測や写真測量は行っていない。','標識板を除く寸法は、依頼文にある目安寸法の範囲から値を選び、写真で輪郭と部品比率を目視確認して固定した推定値である。','支柱径、裏面金具、板厚、見えない足元の形は推定。数cm単位の施工精度を示す値ではない。','',
'警戒ひし形は一辺0.45m（外接幅0.6364m）、丸い規制標識は直径0.60m。','交差道路標識64は幅0.80m、横板高さ0.24m、中央六角盾0.31×0.27m。','出典は[道路標識、区画線及び道路標示に関する命令](https://laws.e-gov.go.jp/law/335M50004002003/)の別表第二。','図の識別子は `S35F03102010003-092`、`120`、`059/060`。','県道70の盾の外接0.46×0.40mは写真比率の推定で、原図の数字欄30cmを標識全幅として流用していない。','原図と照合内容は [work/reference_audit.md](work/reference_audit.md)。','',
'写真中の確認範囲を `*_reference_comparison.jpg` の上部に画素座標で表示した。','これは対象を確認した切抜き範囲であり、画素を実寸へ変換する校正済みの測定値ではない。','元の写真、撮影年月、panoid、方角、URLは読み取り専用の台帳と `work/references.json` にある。','今回は提供されたローカル写真を使い、Google MapsのURLをブラウザで追加閲覧していない。','',
'## 部品ごとのファイルと寸法','',
'下表のファイル名は各assetのゲーム用フォルダ内の `.npz` と `.png` に共通する。','寸法はXYZ外接寸法。袖や腕を持つ物では柱の直径とは異なる。','各部品の参考画像は表の「根拠」に、詳しい説明はasset.jsonの `direct_reference` と `references` に記録した。','']
for info in infos:
    a=info['asset_id'];md += [f'### {a}','',f'制作元: `assets/landmarks/{a}/{a}.blend` と `asset.json`。',f'ゲーム用: `data/processed/landmarks/{a}/`。','', '| 名前（同名npz/png） | 意味 | 三角形 | XYZ寸法 m | 原点 | 根拠写真 |', '|---|---|---:|---|---|---|']
    for p in info['parts']:
        ref=p.get('direct_reference');rf=Path(ref['image']).name if ref else '同分類の参照写真と依頼の寸法目安'
        md.append('| '+p['name']+' | '+p['meaning']+' | '+str(p['triangles'])+' | '+' × '.join(f'{v:.3f}' for v in p['dimensions_m'])+' | '+p['origin']+' | '+rf+' |')
    md += ['', '正面は全て-Y。原点の位置を変えずに使用する。','', '差替画像: '+ '、'.join('`'+file+'`' for p in info['parts'] for file in p['textures'][1:])+'。' if any(len(p['textures'])>1 for p in info['parts']) else '差替画像はなし。', '']
    for p in info['parts']:
        if p['approximations']:md.append('- `'+p['name']+'`: '+' '.join(p['approximations']))
    for n in info['reclassified']:md.append('- 型分類: '+n)
    sheets=sorted(C.glob(a+'_review_*.jpg'));md += ['', '確認画像: '+ '、'.join(f'[{x.name}]({x.name})' for x in sheets)+'。',f'写真比較: [{a}_reference_comparison.jpg]({a}_reference_comparison.jpg)。','']
md += ['## 連結部品','', '| asset-id / 名前 | 1区間 m | 次の原点 |', '|---|---:|---|']
for info in infos:
    for p in info['parts']:
        if 'repeat_length_m' in p:md.append(f'| {info["asset_id"]} / {p["name"]} | {p["repeat_length_m"]:g} | (+{p["repeat_length_m"]:g}, 0, 0) |')
md += ['',
'ガードレールは4m版と、写真の支柱間隔に近い2m版を納品した。袖端は `w_beam_end`。','連続する最後には袖端、または次の支柱を追加する。原型の1区間は始端支柱1本であり、終端支柱を重複させていない。','擁壁は幅4m、高さ1m。コンクリート法枠は1m格子の出っ張り、水抜きは1区画2穴、天端に厚みがある。','高さ方向へ伸ばすと石や木目も伸びるので、配置側で表面UVの実寸を補正する。','側溝の原型は底面z=0、天端z=0.33m。天端を路面へ合わせるには路面高さから0.33m下げる。','階段の手すりは+Yへ上がる。区間を続ける移動量は (0, +2.4, +1.28)m。','',
'## 写真との差と分類の判断','',
'使い回す型のため、個々の設置場所の傾き、錆の位置、板の反り、柱の埋まり具合は固定していない。','写真で見えない背面と固定具は推定し、背面画像名にも `back_estimated` を付けた。','鏡は凸面形状だが、PNGは銀灰色の自作勾配で、道路の動的反射は行わない。','文字は読めた部分だけを自作した。細字、料金、電話番号、工事発注者、判読不確かな案内先を補っていない。','商品ラベル、Coca-Colaの書体、鹿の図柄、BOSCOのS形は輪郭を近似した。','石やコンクリートの画像は周期模様なので、現地の石一個ずつ、しみの分布、苔の位置とは一致しない。','',
'「細いコンクリート柱」は提示4例で通常のコンクリート電柱との違いを確認できず、独立モデルを作らず `concrete` へ統合した。','全80件を同一と検証した意味ではない。','擬木とされた0847Bは金属のWビームに見えるため、`w_beam_cream.png`へ統合した。','青い方面案内 `guide_blue` は地名が確定せず、依頼の「読めない物は作らない」に従って制作していない。','',
'丸2面の0995Fは規制標識の裏面と判断し、ミラーの根拠から除外した。','0582Fのミラーは上下の高さと向きを変えて制作した。','0982Bは独立柱の街灯なので `arm_arc`、0784Bだけを小型壁灯 `wall` とした。','県道64は横長の中央盾付き標識、70は単体六角盾。丸規制と長方形の駐車注意板も別型にした。','信号の地点名板は、写真で白地に青字と確認したため、その配色を採用した。','',
'## 検証','',
'NPZは全103部品で、配列のキー、有限座標、三角形数の上限、退化面、UVの範囲、原点、画像の存在と色違いを検査した。','透過網のアルファ0/255と、擁壁の表面画像の上下左右の端画素一致を検査した。','検査が異常を検知することは、メモリ上の配列を故意にUV範囲外、退化面、負の接地へ変更して確かめた。実ファイルは壊していない。','結果: [route70_validation_08.json](route70_validation_08.json)。','',
'Blenderは背景実行で保存し、別プロセスで各blendを開き直して全構成部品、三角形数、寸法、画像の参照欠落を検査した。','検査対象と結果は `<asset-id>_source_validation.json`。','NPZを読み直して正面、左右、背面、真上、斜め、参照方向の計7画像を各部品につき出力した。','描画エンジンはCycles CPU、Standard色管理。照明をゲーム用PNGへ焼き込んでいない。','`tests/test_blender_save_no_thumbnail.py` は1 passed。コースのコードを変更していないため全コースのテストは実施していない。','',
'原型の描画を確認した範囲であり、ゲーム内のアルファテスト、連結配置、接地、夜間発光、走行時の性能は確認していない。','今回コース配置とインストールが対象外のため、ACを起動する検証も行っていない。','提供写真の解像度にない板の本文や背面の実形状は確認できず、上記の推定形状と空白に留めた。','',
'## 確認画像と再生成','',
'[一覧1](route70_delivery_08_overview_1.jpg)、[一覧2](route70_delivery_08_overview_2.jpg)。','個別の全6方向は `<asset-id>_review_<番号>.jpg`。','写真と同じ面を見る方向に寄せた比較は `<asset-id>_reference_comparison.jpg`。','比較は原型の面を確認するためのもので、カメラ位置を校正した画素一致や現地配置の再現ではない。','',
'```sh',
'.venv/bin/python blender_scripts/build_route70_assets.py --asset r70_curve_mirror',
'blender -b --python-exit-code 1 --python blender_scripts/route70_blender.py -- --asset r70_curve_mirror --mode build',
'blender -b assets/landmarks/r70_curve_mirror/r70_curve_mirror.blend --python-exit-code 1 --python blender_scripts/route70_blender.py -- --asset r70_curve_mirror --mode verify',
'blender -b --python-exit-code 1 --python blender_scripts/route70_blender.py -- --asset r70_curve_mirror --mode render',
'.venv/bin/python blender_scripts/verify_route70_assets.py',
'.venv/bin/python blender_scripts/review_route70_assets.py',
'.venv/bin/python blender_scripts/report_route70_delivery.py',
'```','',
'このセッションではBlender専用MCPの `execute_blender_code_for_cli` から `route70_blender` の各関数を呼んだ。','通常のBlender CLIをこのセッションで別途試してはいない。既存の環境ノートにあるMetal初期化の問題を避け、背景実行ツールで制作と描画を完了した。','',
'## Obsidianへの引継ぎ','',
'`Projects/yabitsu-touge-ac.md`、`Knowledge/Tools/development-workflow.md`、`Preferences/documentation.md` と関連知見を読んだ。','Vaultは現在の書込み許可範囲外のため更新していない。','追記内容は [work/obsidian_update_pending.md](work/obsidian_update_pending.md) に保存した。','']
(C/'route70_delivery_08.md').write_text('\n'.join(md))
(W/'obsidian_update_pending.md').write_text('# Projects/yabitsu-touge-ac.md への追記案\n\n## 依頼08の使い回す沿道原型（2026-10-31）\n\n- 15 asset-id、103部品、25,064三角形。制作元 assets/landmarks/r70_*/、ゲーム用 data/processed/landmarks/r70_*/。\n- 報告 export/compare/route70/route70_delivery_08.md、機械用.json、validation_08.json。7方向画像と写真比較あり。\n- thin_concreteは提示4例に専用細柱の根拠なしでconcreteへ統合。pseudo_woodはWビームへ統合。未判読guide_blueは制作しない。\n- 丸2面は0582の上下・異方向。0995は規制標識背面。64は中央六角盾付き横長型。0982の灯はarm_arc。信号地点名板は写真の白地青字。\n- 正面-Y、上+Z、m、v=0画像上。連結物は始端支柱原点・+X。画像差替は完成アトラス。詳細寸法と連結長は報告。\n- 各blend別プロセス読戻しとNPZ検査。ゲーム配置/KN5/インストールは未実施、依頼範囲外。既存assetとコース変更なし。\n- 再開ポイント：Claude Codeが台帳に合わせて配置。26.5/28.0kmの読み取り不確かさと材質の両面アルファを配置前に確認。\n')
print(json.dumps(summary,ensure_ascii=False))
