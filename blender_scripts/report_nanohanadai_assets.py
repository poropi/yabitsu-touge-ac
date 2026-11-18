"""納品一覧・多方向シート・写真比較・航空輪郭重ねを生成。外部.venv Pythonで実行。"""
from pathlib import Path
import json,math,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1];CMP=ROOT/'export/compare/nanohanadai'
fontpath='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
font=ImageFont.truetype(fontpath,20);small=ImageFont.truetype(fontpath,16)
ASSETS=['monument','park_sign','toilet','mirror','curb_block','fence_signs','path_items','road_planters']
MEANINGS={'monument':'自然石・銘板・レリーフ・石台・低い植込み','park_sign':'二本柱の園地名看板','walls':'白壁・斜め支持壁・入口の奥行・掲示','metal_roof':'灰緑の金属屋根（中央平坦）','wood_trusses':'茶色の木トラス・接合金具','roof_rails':'屋上中央通路・銀色手すり・設備','stone_base':'玉石壁・目地・建物基壇','ramp_walls':'左右のスロープ・段状白壁・笠木','ramp_rails':'スロープの丸パイプ手すり','mirror':'橙柱・凸面鏡・雨よけ・背面リブ','arrow_board':'ガードレール取付の三連矢印板','curb_a':'長い欠けたコンクリート車止め','curb_b':'短い欠けたコンクリート車止め','wood_box_bench':'横並び二箱の木製ベンチ','yellow_sign':'黄色の注意板と取付バンド','red_white_sign':'赤白の注意板と取付バンド','wood_bollard':'木の車止め杭','concrete_bench':'低い長いコンクリートの台','l_curb':'歩道縁のL字コンクリート','checker_planter':'赤白市松の道路側円筒','white_planter':'白い道路側円筒'}
COMPS={
'monument':('sv_road_225.png',(1070,265,1355,585),'reference_road225'),
'park_sign':('sv_road_225.png',(942,390,1105,530),'reference_road225'),
 'toilet':('sv_sw_000.png',(600,210,1160,650),'reference_sw000'),
'mirror':('sv_road_180.png',(805,395,918,560),'reference'),
'curb_block':('sv_ne_315.png',(395,550,1170,735),'reference'),
'fence_signs':('sv_mid2_135.png',(710,505,1480,660),'reference'),
'path_items':('sv_sw_135.png',(525,585,1495,900),'reference'),
'road_planters':('sv_road_315.png',(535,465,923,585),'reference')}

def tile(path,w,h):
    im=Image.open(path).convert('RGB');im.thumbnail((w,h));out=Image.new('RGB',(w,h),(236,237,232));out.paste(im,((w-im.width)//2,(h-im.height)//2));return out

def sheets():
    overview=Image.new('RGB',(1600,1050),'white');do=ImageDraw.Draw(overview)
    do.text((20,12),'菜の花台：ゲーム用NPZ再読込モデル／独立部品を並列表示（現地配置前）',font=font,fill='black')
    names=['front','left','right','back_estimated','top','persp']
    labels=['正面 −Y','左側面 −X','右側面 +X','背面 +Y（見えない部分は推定）','真上 +Z','斜め']
    for i,key in enumerate(ASSETS):
        asset='nanohanadai_'+key
        sheet=Image.new('RGB',(1500,1000),'white');d=ImageDraw.Draw(sheet);d.text((15,10),asset+'／NPZ・PNG再読込',font=font,fill='black')
        for j,(name,label) in enumerate(zip(names,labels)):
            x=(j%3)*500;y=45+(j//3)*475
            sheet.paste(tile(CMP/(asset+'_'+name+'.png'),500,435),(x,y+27));d.text((x+10,y),label,font=small,fill='black')
        sheet.save(CMP/(asset+'_views.jpg'),quality=92)
        x=i%4*400;y=50+i//4*500
        overview.paste(tile(CMP/(asset+'_persp.png'),400,440),(x,y+30));do.text((x+10,y),key,font=font,fill='black')
        ref,crop,view=COMPS[key];original=Image.open(CMP/ref).convert('RGB');reference=original.crop(crop)
        out=Image.new('RGB',(1500,850),'white');d=ImageDraw.Draw(out)
        d.text((15,12),asset,font=font,fill='black')
        reference=ImageOps.contain(reference,(730,680));out.paste(reference,((750-reference.width)//2,70+(680-reference.height)//2))
        out.paste(tile(CMP/(asset+'_'+view+'.png'),750,680),(750,70))
        d.text((16,44),'基準：'+ref+'（対象の拡大）',font=small,fill='black')
        d.text((766,44),'NPZ再読込：同じ面を見る近似方向／単体・現地配置前',font=small,fill='black')
        # クロップ比較でも元画像のGoogle帰属表示を残す。
        strip=original.crop((950,955,1500,1000));strip.thumbnail((730,45));out.paste(strip,(10,790))
        d.text((765,780),'背面・寸法・材質・細部の推定はasset.jsonと納品報告参照。',font=small,fill='black')
        d.text((765,810),'撮影位置・画角・周辺植生の一致を示す比較ではありません。',font=small,fill='black')
        out.save(CMP/(asset+'_reference_comparison.jpg'),quality=94)
    overview.save(CMP/'nanohanadai_overview.jpg',quality=94)

def top_overlay():
    # 実際に出力した屋根頂点を、測定した平面の方位/尺度へ投影して重ねる。
    d=np.load(ROOT/'data/processed/landmarks/nanohanadai_toilet/metal_roof.npz');v=d['v']
    from scipy.spatial import ConvexHull
    vv=v[:,:2];hull=vv[ConvexHull(vv).vertices]
    cx,cy=475.5,635.5;u=np.array([133,-80],float);u/=np.linalg.norm(u);w=np.array([-72,-121],float);w/=np.linalg.norm(w)
    xy=np.array([cx,cy])+hull[:,0,None]*u/.06083777+hull[:,1,None]*w/.06083777
    im=Image.open(CMP/'gsat21.png').convert('RGB');draw=ImageDraw.Draw(im)
    measured=[(373,615),(506,535),(578,656),(445,736)]
    draw.line(measured+[measured[0]],fill=(255,205,0),width=6)
    pts=[tuple(x) for x in xy];draw.line(pts+[pts[0]],fill=(0,220,255),width=3)
    draw.rectangle((340,810,1140,888),fill=(20,25,25))
    draw.text((352,820),'黄：写真の測定輪郭　水色：NPZ屋根の真上輪郭',font=font,fill='white')
    draw.text((352,852),'0.06083777 m/px、方位59°/149°、輪郭読取の不確かさ約±0.5m',font=small,fill='white')
    im.save(CMP/'nanohanadai_toilet_top_overlay.jpg',quality=95)
    (CMP/'nanohanadai_toilet_top_overlay.json').write_text(json.dumps({'model_footprint_m':np.ptp(v[:,:2],0).tolist(),'measured_roof_corners_px':measured,'projected_npz_outline_px':xy.tolist(),'note':'現地配置を実施したものではなく、単体屋根平面の寸法/方位照合。航空画像の輪郭読取に±0.5m程度の不確かさ。'},ensure_ascii=False,indent=2)+'\n')

def report():
    lines=['# 菜の花台の単体アセット納品（2026-10-30）','','制作に関する確認待ちはありません。指定7区分＋追加の道路側円筒を、制作元と色焼付済みのNPZ・PNGとして保存しました。現地配置・コース全体生成・KN5・インストールは行っていません。','','全体画像: [nanohanadai_overview.jpg](nanohanadai_overview.jpg)','','単位m、上+Z、正面−Y。NPZのキーはv/f/uv、fは三角形、UVは画像上端v=0。Blenderで読む場合は1-vへ反転します。配置側の既存UV処理と二重反転しないでください。','', 'トイレの7部品は建物中心直下の地面を共通原点とし、全7部品を同じ変換で配置します。正面−Yは駐車場側の南東（約149°）、+Xは北東。ほかの小物は部品ごとの根元中央、フェンスの板は下辺中央です。鏡の正面−Yは鏡面、矢印板の正面−Yは印刷面です。','','トイレの屋根外形は航空写真から約9.44×8.57m。壁体約7.8×6.8m、軒高3.4m、中央平坦部幅1.8m・肩高5.0mは写真比率と視線交差からの推定です。採寸点・尺度・撮影条件・URLはasset.jsonにあります。','','## ファイル・三角形・寸法','','各行の「名前」は、ゲーム用ディレクトリ内の同名 `.npz` と `.png` の対です。寸法は書出した全頂点のX×Y×Z外接寸法で、植込み・取付具も含みます。','','| asset-id / 名前 | 内容 | 三角形 | 寸法 X×Y×Z (m) |','|---|---|---:|---|']
    inventory=[];total=0
    for key in ASSETS:
        asset='nanohanadai_'+key;src=ROOT/'assets/landmarks'/asset;info=json.loads((src/'asset.json').read_text());processed=ROOT/'data/processed/landmarks'/asset
        for p in info['parts']:
            total+=p['triangles'];dims=' × '.join(f'{x:.3f}' for x in p['dimensions_m'])
            lines.append(f"| {asset} / `{p['name']}` | {MEANINGS[p['name']]} | {p['triangles']:,} | {dims} |")
        inventory.append({'asset_id':asset,'source_files':[str(p.relative_to(ROOT)) for p in sorted(src.iterdir()) if p.is_file() and p.suffix!='.blend1'],'processed_files':[str(p.relative_to(ROOT)) for p in sorted(processed.iterdir()) if p.is_file()],'parts':info['parts']})
    toilet=json.loads((ROOT/'assets/landmarks/nanohanadai_toilet/asset.json').read_text());nt=sum(p['triangles'] for p in toilet['parts'])
    lines+=['',f'合計21部品、{total:,}三角形。トイレ合計{nt:,}三角形。各部品とも依頼の面数目安以内です。','','制作元は `assets/landmarks/<asset-id>/<asset-id>.blend` と `asset.json`、自作画像。ゲーム用は `data/processed/landmarks/<asset-id>/`。全ファイルの列挙は [nanohanadai_inventory.json](nanohanadai_inventory.json)。独立部品の制作元は、重なりを避けるため最初の部品以外をビューポート非表示にしています。名前付きコレクションの目を切り替えて確認できます。','','再生成: `blender_scripts/build_nanohanadai_assets.py` と `build_nanohanadai_entrance.py`・`build_nanohanadai_toilet.py`・`build_nanohanadai_small.py`。共通処理は `nanohanadai_common.py`、検査・再読込描画は `verify_nanohanadai_assets.py`。この報告と比較画像の再生成は `report_nanohanadai_assets.py`。','','## 写真との差・推定箇所','']
    for key in ASSETS:
        asset='nanohanadai_'+key;info=json.loads((ROOT/'assets/landmarks'/asset/'asset.json').read_text())
        lines.append(f"- **{asset}**："+' '.join(info['approximations']))
    lines+=['','材質は写真の画素を使わない自作画像と手続き材質です。文字が判読不能な銘板・注意板・掲示は、短い線と色面で近似しています。透過のある画像は使わず、葉は輪郭形状の両面メッシュです。鏡面PNGは静的な銀灰色の勾配で、周囲が動的に映る材質ではありません。','','## 確認画像','']
    for key in ASSETS:
        a='nanohanadai_'+key;lines.append(f'- {a}: [全6方向]({a}_views.jpg) / [写真と同じ面を並べた比較]({a}_reference_comparison.jpg)')
    lines+=['','[トイレの航空写真とNPZ真上輪郭の重ね](nanohanadai_toilet_top_overlay.jpg)。比較画像は対象を拡大し、同じ面を見る近似方向で並べています。撮影位置・画角・周辺の配置を復元した画像ではありません。','','## 追加で見つかったもの','','`sv_road_315.png` の画面x855–905,y485–545付近の赤白市松円筒と、x552–626,y517–577付近の白い円筒を **nanohanadai_road_planters** として追加しました。用途は写真だけでは確定できないため、円筒・縁・推定の土面までです。追加の独立した石碑・記念碑は、今回確認した写真一式では確実に識別できませんでした。`sv_ne_090`で鏡の手前に見える板は矢印板の裏面と位置が重なるため、別の案内板として二重には作っていません。','','## 検証と実行上の制限','','8制作元を別プロセスで読み戻し、21 NPZ/PNGを再読込しました。必須部品名、有限座標、三角形・頂点上限、面積ゼロ、UV範囲、頂点法線の相殺、印刷面の−Y法線、画像非一様性、制作元との面数・寸法を検査しています。石碑本体はスキル同梱チェッカーで閉じた形・体積・高さを確認しました。各 `<asset-id>_validation.json`、`nanohanadai_monument_geometry.json` に結果があります。','','園地看板のUV上下は、NPZへ実座標のrayを当てた採色で確認。白枠RGB 227/227/213、意図的に1-vへ誤反転した場合75/54/43。`nanohanadai_park_sign_uv_orientation.json` を参照。保存前のサムネイル無効化は `tests/test_blender_save_no_thumbnail.py` が1 passed。ゲームでの材質・照明・配置後の見た目は、今回の依頼範囲外のため未確認です。','','途中の失敗と修正: 通常CLIはスクリプト前のMetal判定でSIGSEGV。Blender専用ツールの背景実行を使用。空のblend_file指定は未対応ファイル形式で失敗し、実在するblendを入力する方法へ修正。7区分の一括描画は `Blender CLI timed out after 120s` となったため2区分ずつに分割。UV検査の初回は白い対称部分を選び `AssertionError: (inverted UV was not detected, [227,227,213], [227,227,213])` となり、非対称図柄へ採色点を変更。診断を通常pythonで実行した回は `ModuleNotFoundError: No module named numpy`、以後は既存.venvを使用。目視で見つかった黄色板の90度回転は、開いた面の法線再計算によるループ順変化が原因で、UVを座標から計算するよう修正しました。','','Obsidianの関連ルール・プロジェクト・Blender知見を読みました。Vaultは現在の書込許可範囲外のため更新できず、反映用内容を `work/obsidian_pending.json` に保存しています。']
    (CMP/'nanohanadai_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    (CMP/'nanohanadai_delivery.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    sheets();top_overlay();report()
