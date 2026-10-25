# 夜間・雨・環境音・紅葉

追加部品で既存コースを仕上げる。既存の本体KN5は再生成しない。CSP 0.3.0-preview619を入れたWindows版Assetto Corsaで検証している。以前のCSPでの動作は未検証。

## 生成と配置

`ffmpeg` をPATHに用意し、既存のtrack/dress/forest生成データがある環境で実行する。

```sh
uv run python -m scripts.parking_props_build --no-install
uv run python -m scripts.visual_updates_build --no-install
uv run python -m scripts.environment_build --no-install
```

`--no-install` は `data/processed/` へ生成する。Windowsの既存コースへ反映するときは、同じコマンドの `--no-install` を外す。各生成処理は部品別INIを統合し、12レイアウトのmodels INIを更新する。環境音は各レイアウトの `data/audio_sources.ini` に管理ブロックを追記し、既存の音設定を保持する。

森林の更新は `scripts.forest_build --area all --no-install`。今回の下草配置範囲は道路から60mで、上層の密度規則・道路の除外範囲を保持する。現地の植生の本数を測ったものではない。

## 実装と確認範囲

| 項目 | できること | 仕様・実物との差、確認範囲 |
|---|---|---|
| 夜間 | 街灯10、トンネル灯68、信号レンズ40、自販機2、窓36組、遠景灯150を点灯 | 遠景灯は街の印象を作る配置で、個々の建物の実測位置ではない。 |
| 雨 | 舗装の濡れ・水たまり、塗装線・金属・土の材質分類、3トンネル174面の雨粒遮蔽 | CSPの雨表現を利用する。測量した轍の凹凸ではない。検証版CSPでは遮雨面の下にも水深約0.9mmが残った。 |
| 環境音 | 小川と鳥6か所、夏の蝉12か所、昼間のみ距離に応じて再生 | 小川と鳥は同一録音。ヤビツ現地録音ではない。ファイルと実機の再生状態を確認、聴感は未評価。 |
| トンネル反響 | トンネル内17ノードとAC標準のREVERB設定 | ノード位置・設定参照を検査。反響の聞こえ方・音量バランスは未評価。 |
| 紅葉 | 日付に応じて広葉樹4モデルを秋色へ。杉・檜・シダ・低木は除外 | 変色時期と強さは景観用の設計値。冬の落葉は含まない。 |

夜間の明るさは天候拡張にも依存する。検証環境のPureは `nlp.lowest_ambient=7.163` で地面が非常に明るかった。標準値1.0の一時比較で夜間照明を確認し、検証後は元のPure設定へ復元した。コース配布物はPureの共通設定を変更しない。

## 音素材

両方とも配布ページでCC0を確認。公式HQ MP3プレビューを取得し、48kHz/16bit/mono WAVへ変換、DC除去・ループ境界処理・音量調整を行う。初回生成時に取得し、取得済み素材もSHA256で照合する。

- cartoonrob: [Cicadas in Japan.wav](https://freesound.org/people/cartoonrob/sounds/643877/)。広島の夏の蝉。
- tomasloos: [Birds, wind through bamboo and a little waterstream](https://freesound.org/people/tomasloos/sounds/843520/)。鳥取の鳥・竹林の風・小川。

配布物の `extension/sfx/CREDITS.json` に出典・作者・許諾・処理後の長さ・振幅を保存する。

## 実装根拠

CSP公式の [Lights](https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Tracks-%E2%80%93-Lights)、[RainFX](https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Tracks-%E2%80%93-RainFX)、[Trees](https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Tracks-%E2%80%93-Trees)、同梱Lua SDKの `ac.AudioEvent.fromFile` を使用する。反響は導入済みKunos Monzaの `data/audio_sources.ini` のCUSTOM設定を基にしている。

比較画像と実行ログは `export/compare/windows-validation/`、作業記録は `export/compare/TASKS.md` に集約する。検証コースは元コースとは別名の `yabitsu_codex_validation`。
