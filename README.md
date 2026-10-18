# ヤビツ峠（Assetto Corsa コース）

神奈川県の県道70号（秦野清川線）、名古木からヤビツ峠を越えて宮ヶ瀬北原までの約 29.9km を、地図・標高データ・ストリートビューをもとに Assetto Corsa のコースとして作るプロジェクトです。

このリポジトリにあるのは、コースを作るためのプログラム（Python）と、手で確かめて書いたデータです。できあがったコースのファイル（.kn5 など）と、ダウンロードした地図・標高・空中写真は入っていません。

## コース

| 区間 | 範囲 | 長さ |
|---|---|---|
| S3 | 名古木 → 蓑毛 | 約 4.3km |
| S2 | 蓑毛 → 菜の花台 | 約 4.2km |
| S1 | 菜の花台 → ヤビツ峠 | 約 3.3km |
| S4 | ヤビツ峠 → 札掛 | 約 6.4km |
| S5 | 札掛 → 宮ヶ瀬北原 | 約 11.7km |

レイアウトは 16 個：区間ごとの上り・下りと、全線（名古木〜ヤビツ峠、名古木〜札掛、名古木〜宮ヶ瀬北原）の上り・下り。

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

## データの出典

- **OpenStreetMap**：道・建物・橋・トンネル・川・湖・信号などの位置と形。© OpenStreetMap contributors。データは [ODbL](https://opendatacommons.org/licenses/odbl/) のもとで使えます。このリポジトリの `data/manual/` のうち OSM から作ったもの（建物の外形など）も ODbL に従います。
- **国土地理院**：標高タイル（dem5a・dem5b・dem）、空中写真（全国最新写真 seamlessphoto）、基盤地図情報の建物の外形（ベクトルタイル）。出典：国土地理院（[地理院タイル](https://maps.gsi.go.jp/development/ichiran.html)）を加工して作成。
- **Google ストリートビュー**：建物・標識・壁などの見た目を確かめるために見ただけです。画像はこのリポジトリに入っていません。

このプロジェクトは Google、国土地理院、OpenStreetMap、Kunos Simulazioni（Assetto Corsa）とは関係ありません。

## ライセンス

プログラムのライセンスはまだ決めていません。データの部分は、上の出典のライセンスに従います。
