---
name: japanese-house-kit
description: Use when このコース（yabitsu-touge-ac）の町並み・集落の家を本物らしく作る・作り直すとき、家が「白い箱」「3 階建てに見える」「ストリートビューと全然違う」と言われたとき、名古木・蓑毛・北原などの家を Blender の部品で建てて試作・比較画像を出すとき。
---

# 日本の家の部品集（japanese-house-kit）

## 要点
家は**作り込んだ部品（半間 0.91 m 刻み）を家の輪郭に並べて**作る。Buildify・Procedural Building Generator など、他の人が建物を量産するやり方と同じ。
輪郭を押し上げた箱に窓の絵を貼るやり方（`yabitsu/buildings.py`・`facades.py`）は、色や絵を替えても本物に見えなかった（2026-10-28 ユーザー「結局今までどおり作っただけ」「酷い」）。そのやり方には戻らない。

## スクリプト（`.claude/skills/japanese-house-kit/scripts/`）
| ファイル | 何をする | 流し方 |
|---|---|---|
| `kit.py` | 部品集：壁（羽目板・吹き付け）、引き違い窓（枠・召し合わせ・ガラス・カーテン・網戸・水切り・シャッターの箱）、小窓（面格子）、玄関（袖ガラス・庇・ポーチ・照明）、ベランダ、切妻・寄棟の屋根（軒・軒天・破風・棟）、雨樋、基礎の換気口、室外機・給湯器、生け垣・低木・庭木。`build_house(name, L, W, rnd, roof, storeys, ground_drop, origin)` | Blender の中で import |
| `one_house.py` | 家 1 軒を道から撮る（部品の出来を見る） | `blender -b --factory-startup --python-exit-code 1 --python scripts/one_house.py -- <絶対パス.png> <seed> hip\|gable` |
| `scene_data.py` | 地点のまわりの家の輪郭（ゲームと同じ track_buildings / s3_buildings）、地面の高さ、道、空中写真、カメラ | プロジェクトの根で `uv run python .claude/skills/japanese-house-kit/scripts/scene_data.py --lat --lon --heading --out <フォルダ>` |
| `ground_mask.py` | 空中写真の色から地面の塗り分け（R＝芝、G＝土、残り舗装） | `uv run python …/ground_mask.py <フォルダ>` |
| `gen_textures.py` | 石垣・アスファルト・芝・庭土・土間コンクリート・砂利・歩道の絵を Codex（image_generation）に作ってもらい、継ぎ目なし＋法線の絵にする。`assets/house_kit/`。src に元画像があれば頼まない（作り直すときは src の画像を消す） | プロジェクトの根で `uv run python …/gen_textures.py [name ...]`（4 本並行、1 枚数分） |
| `street.py` | 輪郭に部品の家を建て、地面を塗り分け、電柱（腕木・変圧器）と電線、生け垣・庭木を置き、ストリートビューと同じ画角（水平 90°）で撮る | `blender -b --factory-startup --python-exit-code 1 --python scripts/street.py -- <フォルダの絶対パス> <絶対パス.png>` |

## 手順
0. **作る物の名前で調べてから作る**（2026-10-28 ユーザー「各建物やオブジェクトの名前で作り方の方法を検索したりしないのですか？」）。
   (a) 寸法・規格（サッシ・電柱・ブロック塀・ガードレール・標識などはメーカーや規格の一次資料）、(b) 作り方を公開している例（ゲーム用のモジュール・寸法の決まり）。
   見つけた値は出典の URL と一緒に `references.md`（このスキルの中）に書き、部品の値はそこから取る。見つからなかったら「見つからない・写真から推定」と書く。
   店の商標（ロゴ）は描かない（7-Eleven 風のモデルは報道・解説用のみの制限が付いている）。
   **有料の素材・モデル・キット・講座は使わない、買う候補として挙げもしない**（2026-10-28 ユーザー「有料は使わない。覚えておいて！」）。見つかった有料品は寸法・形の参考に読むだけ。
1. 照合地点のストリートビューを撮る（playwright。URL `https://www.google.com/maps/@<lat>,<lon>,3a,90y,<heading>h,90t/data=!3m1!1e1?hl=ja`、1500×1000。`require` は使えないので `page.screenshot({path})` だけ）。
2. `scene_data.py` → `ground_mask.py` → `street.py`。Cycles（METAL）で 1 枚 2〜4 分。
3. 写真・今のゲーム・試作を縦に並べた比較画像を作り、**`open` で開いてから**報告する。違いは画面で見えたものだけを書く（件数やデータの出どころで出来を語らない）。
4. 家 1 軒の出来から先に見せ、了承を得てから町全体・ゲームへ広げる。

## はまりどころ
| 症状 | 原因と直し |
|---|---|
| 屋根が淡いピンク・白に写る | 強い法線の絵と照り返しが低い角度で空を映す → 屋根は rough 0.85・normal 0.25・spec 0.05、色は暗め |
| 寄棟の屋根だけ淡い | 面の並びが下向きだった（直し済み）。部品を足したら面の法線を `chk` で確かめる |
| 地面に屋根や道が写り、にじむ | 空中写真をそのまま貼らない。色で芝・土・舗装を分けて素材で塗る（`ground_mask.py`） |
| 道ぞいが土手のように盛り上がる | 地形の格子の角。道から 6〜14 m は道の高さへ寄せる（`street.py` 済み） |
| `ModuleNotFoundError: PIL / shapely` | Blender の Python には無い。絵や形の計算は uv の Python で先に作る |
| `KeyError: A_Color` | Mix ノードは `identifier` で取る（`kit._sock`） |
| 補修のつぎはぎが一定間隔で並ぶ | 1 枚の絵に目立つ形（補修の四角など）を頼むと、敷いたときに繰り返しが見える。道・地面の絵には目立つ形を入れない |
| 全体が白く飛ぶ | EEVEE＋Standard は不可。Cycles・AgX・Nishita の空 0.25・太陽 3.5 |

## ゲームの家（`yabitsu/house_kit.py`）
- Blender の試作は 1 軒 9,800 三角形（窓の枠・格子の箱の積み重ね）で、ゲームには重い。ゲーム用は numpy で作る軽い版：窓の枠・召し合わせ・カーテン・網戸・面格子・ドアは**窓の絵（アトラス 4×2、`hk_win_silver`／`hk_win_bronze`）**に焼き、形は壁のへこみ・窓台・シャッターの箱・ベランダ・軒・破風・雨樋・室外機だけ。道から 35 m 以内 約 1,400、遠い家 約 600 三角形。
- `scripts/track_build_manifest.py` が `house_kit.split` で S3 の名古木〜1.7 km（`S3_RANGE`）・道から 80 m の勾配屋根の家を部品の家にする（619 軒・57 万三角形）。店（看板・ガラス・屋根だけ）と陸屋根・260 m² 超は今までどおり。範囲を広げるときは `S3_RANGE` と三角形の合計を見る。
- ゲームでの見た目を Mac で先に見るとき：`game_preview.py --lat --lon --out <scene_data のフォルダ>` で OBJ を出し、`HK_OBJ=<フォルダ>/hk.obj blender … street.py -- <フォルダ> <png>`。
- 壁の面は表が外。面に沿う u と上の z で 4 点を回ると内向きになる（`FQ` で逆にしている）。テスト `tests/test_house_kit.py`。

## まだやっていないこと
- 石垣・道・地面の Codex の絵（`assets/house_kit`）をゲームの材質に入れる（試作の Blender だけで使っている）。
- 横断歩道・車止め・標識・駐車の車・縁石・家ごとの駐車場、生け垣・庭木（ゲームでは森の処理の庭木 4＝garden が受け持つ）。
