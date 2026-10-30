# コースへの取り込みを依頼されたとき

## 読む順序と既存の経路

まず`docs/ac-conventions.md`、対象の`data/manual/`、その読取側、`yabitsu/manifest.py`、`yabitsu/kn5.py`、必要なら`blender_scripts/build_from_manifest.py`を読む。一般的なACの向きや既定値より、このプロジェクトで実際に変換するコードを優先する。

石碑の例には次の経路がある。ただし編集中の場合があるので、存在と現在の実装を確かめてから使う。

| 段階 | ファイル |
|---|---|
| 制作元 | `assets/landmarks/asama_monument/sengen_monument.blend` |
| 色・AOのベイクとローカルNPZ出力 | `blender_scripts/bake_asama_monument.py` |
| 出力先 | `data/processed/landmarks/asama_monument/` |
| コース内の指定 | `data/manual/s2_asama.json`の`monument` |
| 向き・接地・衝突形状 | `yabitsu/asama_area.py`の`_monument` |
| 材質・テクスチャ登録 | `scripts/track_build_manifest.py` |
| 幾何・配置の回帰テスト | `tests/test_asama_area.py` |

`monument.npz`がないと仮形状を使う経路があるため、読み込みが成功しただけでは詳細モデルへ差し替わった証拠にならない。NPZの実体、材質登録、生成したmesh名を確認する。別の看板・設備に石碑専用のコレクション名やベイク処理をそのまま適用しない。

## 手続き材質を画像へ焼く

ゲームへ出す対象から、撮影用の地面・背景・カメラ・照明を除く。見せる形状とベイク元の細部を分ける。文字、目地、地衣類などの薄い面を細かいUV島のまま大量に残すより、本体への色ベイクで保持できる部分はまとめる。

1. 制作元を保護し、複製したメッシュへモディファイアと変換を適用する。
2. UV展開し、ゲームで必要な解像度と余白を決める。ベイク画像を置くImage Textureノードをアクティブにする。
3. 必要に応じ、選択したベイク元からアクティブな出力メッシュへDiffuseの色だけをベイクする。照明を色へ二重に焼かない。ray距離・ケージはモデル寸法と浮かせた細部の距離から決める。
4. AOは必要な場合に加える。線形値とsRGBの変換は1回にし、出力画像を開いて文字・色・暗さを確認する。既存スクリプトのパラメータを万能値にしない。
5. 書き出す前に三角形へそろえ、UVの継ぎ目は頂点を分ける。`mesh.calc_tangents()`を使う場合もngonを残さない。

石碑専用の実行例。プロジェクトのルートで実行し、元blendを上書きしない。

```sh
blender -b assets/landmarks/asama_monument/sengen_monument.blend --python-exit-code 1 --python blender_scripts/bake_asama_monument.py -- "$PWD/data/processed/landmarks/asama_monument" 2048
```

このコマンドは実装を読むための入口であり、このスキルのインストール時にベイクを再実行する必要はない。

## 座標・UV・マニフェスト

- このプロジェクトの座標変換は`yabitsu/acdata.py`の`blender_to_ac`。BlenderのXYZからACへ`(x, z, -y)`。別のKN5インポーターの逆変換を混ぜない。
- 部品はローカル座標で保持し、配置は既存のFrame・壁方向・路面平面に合わせる。変換後に正面が駐車場や道路を向くかを、非対称な文字で確認する。台座の全隅で浮き・沈みを見る。
- `Manifest.add_mesh`の配列は`verts`・`faces`・`uvs`。石碑専用のNPZは`v`・`f`・`uv`なので読取側の変換を確認する。
- マニフェストの頂点上限は1メッシュ65,535。UV分割・法線分割後に増える頂点も数える。複数meshや部品へ分け、ゲームの描画負荷も見る。
- `yabitsu/kn5.py`の`write_from_manifest`はVを`1-v`に変える。Blenderでの確認側はマニフェストのUVをそのまま使う。**ベイク時・NPZ保存時・描画時のどこで反転するかを数えて、二重反転を防ぐ。**
- 石碑のベイク実装にも`1-u[1]`が入っている場合がある。既存ファイルがあるから正しいと仮定せず、上下に非対称な文字・TOPマーカーで最終KN5まで確認する。確認前に「上下は正しい」と報告しない。

## 当たり判定とコース確認

詳細な見せるメッシュとは別に、台座・本体などの単純な衝突形状を使える。`1WALL_`と`renderable=False`など、既存の規約を確認する。元の仮石碑・新モデル・衝突形状が重複して見えない壁を増やしていないか確認する。

対象部分のテストと、出力ファイルを読み戻す検証を行う。NPZの存在やテスト通過だけで、ゲーム内の見た目を確認したことにしない。KN5は書き出した実物を確認し、道路からの視点・接地・向き・衝突を実ゲームで確認できなかった場合は、その理由と未確認範囲を報告する。

全コースの`track_build_manifest`や`make_kn5 track`は多数の生成物を更新する。単体の変換で検証できる間は対象を絞る。インストールや配布を依頼されていなければ、ゲームのフォルダやレイアウト設定まで更新しない。

## スキルの入口

正本は`.agents/skills/yabitsu-object-modeling/`、Claude Codeの入口は`.claude/skills/yabitsu-object-modeling/`。両方に同一内容の実ファイルを置く。Windowsでシンボリックリンクを展開できない環境でも使える構成とする。更新時は正本からClaude側へ同期し、ファイルの一致を検証する。

配置仕様の一次資料：[Codexのローカルスキル](https://learn.chatgpt.com/docs/build-skills)、[Claude Codeのスキル](https://code.claude.com/docs/en/skills)。本スキルは両方のローカルプロジェクトで使うためのもので、クラウド環境への登録や公開は行わない。
