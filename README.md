# ヤビツ峠（Assetto Corsa コース）

神奈川県の県道70号（秦野清川線）、名古木からヤビツ峠を越えて宮ヶ瀬北原までの約 29.9km を、地図・標高データ・ストリートビューをもとに Assetto Corsa のコースとして作るプロジェクトです。

## ダウンロード（Assetto Corsa の MOD）

**[最新版のダウンロード（Releases）](https://github.com/poropi/yabitsu-touge-ac/releases/latest)** から `yabitsu-touge-ac-v<版>.zip` を落とし、Content Manager の画面に zip を落とすか、zip の中の `content` フォルダを Assetto Corsa のフォルダに上書きでコピーします。前の版から更新するときは、`content/tracks/yabitsu` を消してから入れると古いファイルが残りません。

### 推奨環境

| | |
|---|---|
| 必要 | Custom Shaders Patch（CSP）。森・路面や擁壁の模様・夜の明かり・雨・環境音は CSP で表示します |
| 確かめた CSP | 0.3.0-preview619（これより古い版では確かめていません） |
| 天気 | Pure 2.57 で確かめました（なくても走れます） |
| 確かめた PC | Core i7-14700K / GeForce RTX 5070 / メモリ 64GB / Windows 11 |
| 重さ | 5120×1440 全画面で全線を走って、平均 約 64 FPS・最低 約 48 FPS（町の区間がいちばん重い） |
| ディスク | 展開後 約 2.2GB（zip は約 1GB） |

### 確かめたときの AC の設定

- 画面：5120×1440 全画面、垂直同期なし、アンチエイリアス 4x、異方性フィルタ 16x
- 影 2048、ワールドの細かさ 3、煙 3、モーションブラーなし
- CSP：Extra FX・Rain FX・Reflections FX・Particles FX・Windscreen FX を入れた状態
- これより弱い PC では、解像度・影・ワールドの細かさを下げてください。

菜の花台・ヤビツ峠の駐車の車は、作った PC の AC の車のデータを使うため、配布する zip には入っていません。

このリポジトリにあるのは、コースを作るためのプログラム（Python）と、手で確かめて書いたデータです。できあがったコースのファイル（.kn5 など）と、ダウンロードした地図・標高・空中写真は入っていません。

## コース

| 区間 | 範囲 | 長さ |
|---|---|---|
| S1 | 名古木 → 蓑毛 | 約 4.3km |
| S2 | 蓑毛 → 菜の花台 | 約 4.2km |
| S3 | 菜の花台 → ヤビツ峠 | 約 3.3km |
| S4 | ヤビツ峠 → 札掛 | 約 6.4km |
| S5 | 札掛 → 宮ヶ瀬北原 | 約 11.7km |

レイアウトは 12 個：区間ごとの上り・下りと、全線（名古木〜宮ヶ瀬北原）の上り・下り。

プログラム内部の区間名（`s1_widths.csv` などのデータ、`layouts.json` の `section`）は作った順の古い番号のまま。内部の `s1` が菜の花台 → ヤビツ峠（表示は S3）、内部の `s3` が名古木 → 蓑毛（表示は S1）。

各区間の確認の記録：`docs/s4-acceptance.md`、`docs/s5-acceptance.md`。設計と作業の計画：`docs/superpowers/`。

## 作り方

必要なもの：Python 3.12 以上と [uv](https://docs.astral.sh/uv/)、Blender 5.2、Assetto Corsa。

```sh
uv sync
uv run pytest -q                                   # テスト

uv run python -m scripts.track_01_fetch            # 地図（OSM）・標高（地理院）を取る
uv run python -m scripts.track_02_profile          # 道の高さ
uv run python -m scripts.track_04_section          # 道の断面・町・土台
uv run python -m scripts.track_05_textures         # 絵
uv run python -m scripts.track_06_terrain          # 地形
uv run python -m scripts.track_build_manifest      # 部品をまとめる
uv run python -m scripts.track_check               # 高さ・位置・道の上に何もないかの確認
uv run python -m scripts.make_kn5 track            # ゲーム用ファイル（.kn5）
uv run python -m scripts.track_08_acfiles          # レイアウト・コース選択の絵・AI の線
```

`.kn5` は Assetto Corsa の `content/tracks/yabitsu/` に書き出されます。

夜間照明・雨・環境音・紅葉と、既存KN5を保持した仕上げの生成手順は [docs/environment.md](docs/environment.md) にあります。

## データの出典

- **OpenStreetMap**：道・建物・橋・トンネル・川・湖・信号などの位置と形。© OpenStreetMap contributors。データは [ODbL](https://opendatacommons.org/licenses/odbl/) のもとで使えます。このリポジトリの `data/manual/` のうち OSM から作ったもの（建物の外形など）も ODbL に従います。
- **国土地理院**：標高タイル（dem5a・dem5b・dem）、空中写真（全国最新写真 seamlessphoto）、基盤地図情報の建物の外形（ベクトルタイル）。出典：国土地理院（[地理院タイル](https://maps.gsi.go.jp/development/ichiran.html)）を加工して作成。
- **Poly Haven**（[polyhaven.com](https://polyhaven.com)、CC0）：森の木の葉・針葉・シダ・樹皮の写真素材（fir_tree_01・tree_small_02・fern_02）を切り出して組み直しました。森の植える場所と樹種は、上の空中写真から判定しています（`docs/forest.md`）。
- **Overture Maps Foundation**（ODbL）：地理院・OSM に無い建物の輪郭の一部。
- **ambientCG**（[ambientcg.com](https://ambientcg.com)、CC0）：家の壁・屋根の写真素材。
- **Freesound**（CC0）：cartoonrob の日本の蝉、tomasloos の日本の鳥・小川の録音を加工して使用。出典と処理は [docs/environment.md](docs/environment.md) に記載。
- **Microsoft Rocketbox Avatar Library**（[github.com/microsoft/Microsoft-Rocketbox](https://github.com/microsoft/Microsoft-Rocketbox)、MIT、Copyright (c) 2020 Microsoft）：菜の花台の見物客の人物。ライセンス文は [assets/free/rocketbox/LICENSE.md](assets/free/rocketbox/LICENSE.md)。
- **Google ストリートビュー**：建物・標識・壁などの見た目を確かめるために見ただけです。画像はこのリポジトリに入っていません。

このプロジェクトは Google、国土地理院、OpenStreetMap、Kunos Simulazioni（Assetto Corsa）とは関係ありません。

## ライセンス

- **プログラム**（`yabitsu/`・`scripts/`・`blender_scripts/`・`tests/` など）：[MIT License](LICENSE)。
- **コース**（配布する zip の中身：`.kn5`・テクスチャ・レイアウトなど、ここで作ったもの）：[クリエイティブ・コモンズ 表示-非営利 4.0 国際（CC BY-NC 4.0）](https://creativecommons.org/licenses/by-nc/4.0/deed.ja)。作者（poropi）とこのリポジトリを表示すれば、改造・再配布してかまいません。販売や有料での配布はできません。
- **出典のデータ・素材**：上の出典それぞれのライセンスに従います。OpenStreetMap・Overture から作ったデータ（`data/manual/` の建物の外形など）は ODbL です。Rocketbox の人物は MIT です。出典の表示は、改造版・再配布でも残してください。
