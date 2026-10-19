# 全域の森（CSP の木）

設計：`docs/superpowers/specs/2026-10-19-yabitsu-forest-design.md`　手順：`docs/superpowers/plans/2026-10-19-yabitsu-forest.md`

## 作り方（コマンド1回）

```
uv run python -m scripts.forest_build --area all            # 全域を作って AC へ入れる
uv run python -m scripts.forest_build --area all --render   # ＋確認画像（export/forest/<日時>/views/）
uv run python -m scripts.forest_build --area all --refresh-map   # 空中写真の判定をやり直す（forest.json の classify を変えた時）
```

既存の kn5 は作り直さない。`content/tracks/yabitsu/extension/`（ext_config.ini と trees/）だけを書く。
古い板の木（kn5 の TREES_*、裏側の *_back も）は ext_config で隠す。CSP が必要（CSP なしでは古い木のまま）。
Codex の S1 景観確認レイアウトでは森を切る（レイアウトごとの ext_config）。

## 流れ

1. 素材：Poly Haven（CC0）の葉・針葉・シダ・樹皮（`yabitsu/polyhaven.py`、出典は `assets/forest/README.md`）
2. 木のモデル 13 種類（杉3・檜2・広葉樹4・低木2・シダ2、LOD 3段階、DDS テクスチャ）：`yabitsu/treemodels.py`
3. 空中写真の地図（地理院シームレス空中写真、4 m マス）：`yabitsu/forestmap.py`
   - 森＝きめが粗い（季節の違う写真のつぎはぎでも効く）。なめらかでも周りの森より暗ければ森
   - 樹種＝500 m 区画ごと（冬の写真は茶色＝落葉広葉樹、青緑＝針葉樹。夏の写真は暗い緑＝針葉樹）
   - 集落（家から 40 m）＝暗い緑のまとまりだけ森。OSM の畑・住宅地・ゴルフ場・水などは森にしない
4. 地面と植えない範囲（AC に入っている kn5 の元データ manifest から）：`yabitsu/forest_ground.py`
5. 配置：`yabitsu/forest_place.py` — コース（ゲームの中の道）からの距離で 4 帯（60/250/800 m）、下草は 30 m まで。
   道路の上に張り出す木（枝の最低の高さ＜路面＋5 m）と、運転席からの見通し（6〜35 m 先）をふさぐ木は植えない
6. 書き出し：`yabitsu/forest_ac.py`（1 km 四方ごとの木の一覧）

## 調整

値はすべて `data/manual/forest.json`：`classify`（森・樹種の判定）、`place`（帯ごとの間隔・大きさ、下草）、
`clearance`（植えない幅）、`sight`（見通し）。直したらコマンドを走らせ直すだけ。地点ごとの手直しはしない。
重いとき：`place.tiers` の遠い帯の `spacing` を広げる。
