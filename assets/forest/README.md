# 森の素材と出典

## 写真素材（Poly Haven、CC0）

`scripts.forest_build` は、初回実行時に次の素材を `data/raw/polyhaven/` へダウンロードします（https://polyhaven.com 、ライセンス CC0）。
葉・針葉・シダの房を切り出して組み直し、木のテクスチャを作ります（`yabitsu/foliage_textures.py`）。

| 素材 | 使う部分 |
|---|---|
| fir_tree_01 | 針葉の房（twig）→ 杉・檜の枝、樹皮（bark）→ 全樹種の樹皮の元 |
| tree_small_02 | 葉の房（leaves）→ 広葉樹・低木の茂み |
| fern_02 | シダの葉 |

## 広葉樹の枝分かれ原型

`broadleaf_prototypes_v2.json` は Sapling Tree Gen 0.3.5（Blender Foundation、GPL-2.0-or-later）の日本カエデ設定で作った枝の座標データ（Codex 版 `assets/landscape/` から写したもの）。生成器そのものは含みません。

## 空中写真

森と樹種の判定には、地理院タイル（シームレス空中写真）を使います。出典：国土地理院。
