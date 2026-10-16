# AC の規約（Task 2 の試験コースで確認）

試験コース：`content/tracks/yabitsu_test`（レイアウト `a`、北向き 200m の直線）。
FBX：`export/test/yabitsu_test.fbx`、`export/test/test_objects.fbx`（テクスチャは `export/test/yabitsu_test.fbm/`）。

| 項目 | 確認結果 | 使う値 |
|---|---|---|
| 座標変換 blender_to_ac | 未確認 | (x, z, -y) |
| ダミーの向き DUMMY_YAW_OFFSET | 未確認 | math.pi / 2 |
| ダミーの高さ DUMMY_Z_OFFSET | 未確認 | 0.3 |
| FBX 書き出し設定 | 未確認 | FBX_SETTINGS のまま |
| 1WALL の当たり判定 | 未確認 | surfaces.ini に WALL を定義 |
| 描画しない壁の作り方 | 未確認 | ksEditor で Renderable を外す |
| ダミーだけの kn5 | 未確認 | |
| ミニマップの式 | 未確認 | (X + X_OFFSET) / SCALE_FACTOR |
| AI ラインの記録方法 | 未確認 | |
| ksEditor の操作手順 | 未確認 | |
