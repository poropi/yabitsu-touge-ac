# S3 道路まわりの仕上げ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** S3（名古木→蓑毛）の路面・柵・壁・路肩・電柱などを、ストリートビューの写真に合わせた見た目にする。今の kn5 は作り直さない。

**Architecture:** 2つのやり方を組み合わせる。(1) 今ある路面・白線・側溝・歩道・壁は、CSP の `[SHADER_REPLACEMENT_...]`（MESHES 指定、`ksPerPixelNM`、`RESOURCE_FILE_n`）で、種類ごとに新しいテクスチャへ貼り替える。UV はそのまま使う。路面の UV は横 1＝約 4 m、長さ 1＝2 m。(2) 今は無いもの・形が違うもの（ガードパイプ・柵・波形ガードレール、電柱と電線、マンホール・グレーチング・補修跡の貼り物）は、別の kn5 `yabitsu_s3_dress.kn5` に作る。全レイアウトの models ini に足し、古い柵の表示は CSP で隠す（IS_RENDERABLE=0 だけ。当たり判定は残す）。テクスチャは Codex に生成を頼む（`codex exec`、image_generation）。

**Tech Stack:** Python 3.12（uv、numpy、pillow、scipy）、`yabitsu/manifest.py`＋`yabitsu/kn5.py`（kn5 の書き出し）、CSP ext_config、Codex CLI 0.163（image_generation 有効）、確認描画は Blender 5.2。

**Spec:** `docs/superpowers/specs/2026-10-19-yabitsu-s3-dress-design.md`

## Global Constraints

- 既存の `yabitsu_*.kn5`・`*_objects.kn5` は書き換えない。足すのは `yabitsu_s3_dress.kn5`、models ini の1行、`extension/` の設定とテクスチャだけ。
- 当たり判定を変えない：既存メッシュは IS_RENDERABLE=0（表示だけ）で隠す。IS_ACTIVE は触らない。
- コマンド1回：`uv run python -m scripts.dress_build --section s3`（テクスチャ生成は別コマンド `scripts.dress_textures`。一度作れば使い回す）。
- 写真（ストリートビュー）は見るだけ。ゲームの画像には使わない。参照写真：`.superpowers/sdd/2026-10-17-yabitsu-s3/sv/s3_<s>_{f,b}.png`（50 m ごと、167 枚）、地点 `sv_s3_points.json`。
- 種類や範囲の食い違いは表（`data/manual/s3_features.csv`、新しい `data/manual/s3_dress.csv`）を直して再実行する。地点ごとの手作業はしない。
- 森の ext_config を消さない：ext_config.ini は部品（`extension/parts/*.ini`）から組み立てる。
- テキストは `newline="\n"`。ユーザーへの報告は日本語。見比べ画像はビューアーで開く。

## Review Focus

1. **当たり判定が消える**：隠す対象が `1ROAD_*`・`1WALL_*` のとき IS_ACTIVE を書かない。→ Task 3 `test_retex_never_deactivates`
2. **S3 の外を変えてしまう**：区間の判定は中心線の s（0〜S3 終わり）＋道から 60 m。蓑毛から先の部品は対象外。→ Task 3 `test_meshes_limited_to_s3`
3. **森の設定を上書きで消す**：→ Task 3 `test_assemble_keeps_forest_part`
4. **貼り物が路面と同じ高さでちらつく**：路面から 2 cm 上げる。→ Task 4 `test_decals_float_above_road`
5. **電柱が道路・歩道・建物に立つ**：植えない範囲（森の KeepOut）で確かめる。→ Task 4 `test_poles_not_on_road`

---

### Task 1: 写真の見本帳と仕上げの表

**Files:** Create `scripts/dress_sheets.py`、`data/manual/s3_dress.csv`。Modify `data/manual/s3_features.csv`（写真と違う行だけ）。

- [ ] **Step 1:** `scripts/dress_sheets.py`：167 枚を s の順に 12 枚ずつ（3×4、1 枚 400×267）並べ、`export/dress/s3/sheets/sheet_NN.jpg` に書く。各写真の左上に `s=####f/b` を書く。
- [ ] **Step 2:** 見本帳を見て、`s3_dress.csv` を書く。列：`s_from,s_to,side,item,value,note`。item は `poles`（value＝間隔 m、電線の本数は note）、`curb`（none/low/high）、`gutter`（open/grating/lid）、`fence`（white_pipe/white_mesh/wbeam/none）、`surface`（new/worn/patched）。`s3_features.csv` の壁の種類（note）を、仕上げのコード `finish`（kenchi＝間知石、block＝ブロック、concrete＝コンクリート、stone＝自然石）に揃える列を足す。
- [ ] **Step 3:** コミット。

### Task 2: テクスチャ（Codex に依頼）

**Files:** Create `scripts/dress_textures.py`、`yabitsu/dress_tex.py`、`assets/dress/README.md`、`assets/dress/*.png`。Test `tests/test_dress_tex.py`。

**Interfaces:** Produces `TEXTURES: dict[str, TexSpec]`（name → 実寸 w×h m、プロンプト）、`height_to_normal(gray: np.ndarray, strength: float) -> np.ndarray`、`make_tileable(img) -> img`、`load(name) -> (diffuse PIL.Image, normal PIL.Image)`。

テクスチャ（実寸）：`asphalt_town`（4×2 m、路面 UV に合わせる）、`line_white_worn`・`line_orange_worn`、`gutter_concrete`、`sidewalk`、`kenchi`（間知石、2×2 m）、`block_wall`（ブロック 0.4×0.2 m の目地、2×1 m）、`concrete_wall`（型枠の線・汚れ・水抜き穴、2×2 m）、`curb`、`grating`（アルファ付き）、`manhole`（アルファ付き）、`patch`（補修跡、アルファ付き）。

- [ ] **Step 1: テスト** `test_height_to_normal_flat`（平らなら (0,0,1)）、`test_make_tileable_edges`（左端と右端・上端と下端の差が小さい）。
- [ ] **Step 2:** 失敗を確認 → `yabitsu/dress_tex.py` を実装 → 通ることを確認。
- [ ] **Step 3:** `scripts/dress_textures.py`：各テクスチャについて `codex exec --skip-git-repo-check -C <作業フォルダ> "<プロンプト>"` を実行し、生成された PNG を `assets/dress/src/<name>.png` に受け取る（プロンプトに保存先と 2048 px、真上から・影なし・継ぎ目なしを指定）。受け取ったら `make_tileable` → 2 の累乗に縮小 → 法線マップ作成 → `assets/dress/<name>.png`・`<name>_nm.png`。済んでいるものは飛ばす。
- [ ] **Step 4:** 全部を並べた確認画像（2×2 回並べて継ぎ目を見る）を作って自分で見る。合わないものはプロンプトを直して頼み直す。出典（Codex で生成）を README に書く。コミット。

### Task 3: 貼り替えの設定（既存メッシュ）と ext_config の組み立て

**Files:** Create `yabitsu/ext_config.py`、`yabitsu/dress_retex.py`。Modify `yabitsu/forest_ac.py`（部品へ書く）。Test `tests/test_dress_retex.py`、`tests/test_forest_ac.py`。

**Interfaces:**
- `ext_config.write_part(track_dir, name, text)`、`ext_config.assemble(track_dir)`（`extension/parts/*.ini` を名前順に MARK の下へ連結）
- `dress_retex.s3_meshes(manifest_dir, s_range, near_m=60) -> dict[name, kind]`（kind：road、gutter、sidewalk、line_white、line_orange、wall、parapet）
- `dress_retex.wall_finish(manifest_dir, features_csv) -> dict[name, finish]`（壁メッシュの中心の s と左右から表の行を引く）
- `dress_retex.section_text(assign: dict[name, texname]) -> str`（テクスチャごとに `[SHADER_REPLACEMENT_...]` を1つ：`MESHES=`、`SHADER=ksPerPixelNM`、`RESOURCE_0=txDiffuse`・`RESOURCE_FILE_0=dress/<tex>.dds`、`RESOURCE_1=txNormal`・`RESOURCE_FILE_1=dress/<tex>_nm.dds`）

- [ ] **Step 1: テスト** `test_meshes_limited_to_s3`（合成 manifest：S3 の中と外のメッシュ）、`test_wall_finish_from_csv`、`test_retex_never_deactivates`（出力に IS_ACTIVE が無い）、`test_assemble_keeps_forest_part`（forest と dress の部品が両方残る）。
- [ ] **Step 2:** 失敗を確認 → 実装 → 通る。forest_ac.write_ext_config を `write_part("forest")`＋`assemble` に変える（既存のテストも通す）。
- [ ] **Step 3:** コミット。

### Task 4: 足す部品（柵・電柱と電線・貼り物）と S3 仕上げの kn5

**Files:** Create `yabitsu/dress_parts.py`、`scripts/dress_build.py`。Test `tests/test_dress_parts.py`。

**Interfaces:**
- `white_pipe_fence(path_xy, z, height=1.1, rails=3)`、`white_mesh_fence(...)`、`wbeam_guardrail(...)`：中心線の左右の範囲から (verts, faces, uvs)
- `utility_poles(path_xy, z_fn, spacing, side_offset, keep) -> list[(x, y, z)]` と `pole_mesh`・`wire_meshes(poles, n_wires)`（たるみ付き、細い帯）
- `road_decals(sec, rng, kinds) -> list[(verts, faces, uvs, tex)]`：マンホール（約 60 m に1つ、車線の中）、グレーチング（側溝のふたの所）、補修跡（surface=patched の区間）。路面＋0.02 m
- `scripts/dress_build.py --section s3`：表を読む → 部品を `Manifest`（`data/processed/dress_s3/`、グループ `yabitsu_s3_dress`）→ `kn5.write_from_manifest` → AC のコースへ。全レイアウトの `models_*.ini` に `yabitsu_s3_dress.kn5` を足す（`acdata.write_models_ini`、`layouts.kn5_files` にも足して、将来の作り直しでも入るようにする）。テクスチャを `extension/dress/` に DDS で置く。Task 3 の部品を書いて assemble。古い柵（S3 の `GUARDRAIL_*`、`1WALL_track_fence*` の表示）は IS_RENDERABLE=0。

- [ ] **Step 1: テスト** `test_decals_float_above_road`、`test_poles_not_on_road`、`test_fence_follows_side`（左右と範囲どおり）、`test_wires_sag_between_poles`。
- [ ] **Step 2:** 失敗を確認 → 実装 → 通る。
- [ ] **Step 3:** `dress_build --section s3` を走らせて AC へ入れる。コミット。

### Task 5: 写真と見比べ → ユーザーの試走

**Files:** Modify `blender_scripts/render_forest.py`（仕上げの kn5 の元 manifest と貼り替えも描く）。Create `scripts/dress_compare.py`。

- [ ] **Step 1:** 写真のある地点から 6 か所（町の中、間知石の壁、ガードパイプ、田畑の横、坂、蓑毛の手前）を選び、同じ位置・向き（`sv_s3_points.json` の lat/lon・brg、目の高さ 2.5 m）で描く。写真｜描画 の並べ画像を作り、ビューアーで開く。自分で見て、明らかにおかしい所は表かテクスチャを直して再実行する。
- [ ] **Step 2:** ユーザーに S3 を走ってもらう（当たり判定が変わっていないこと、ちらつき、重さ）。
- [ ] **Step 3:** `docs/forest.md` の横に `docs/dress.md`（使い方・表の意味）を書いてコミット。
