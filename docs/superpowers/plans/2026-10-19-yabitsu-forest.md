# ヤビツ峠 全域の森 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** マップ全体（遠景地形の範囲）を、空中写真に合わせた本物らしい杉・檜・広葉樹の森にする。CSP の木の機能を使い、コマンド1回で生成して AC へ入れる。

**Architecture:** 純 Python（uv、numpy）で全部作る。木のモデルを CSP の木パッケージ（zip：`tree.ini`＋`.mesh` LOD＋テクスチャ）として書き出す。地理院の空中写真から森・樹種の地図を作る。植えない範囲と地面の高さは、AC に入っている kn5 の元データ `data/processed/track/manifest.json` のメッシュから取る。配置した木を CSP の木一覧 txt と `extension/ext_config.ini` に書き、AC のトラックフォルダへ入れる。既存の kn5 は作り直さない。古い板の木 `TREES_*` は CSP 設定で隠す。

**Tech Stack:** Python 3.12（uv）、numpy、scipy、shapely、pillow。確認用の描画だけ Blender 5.2（背景モード）を使う。AC＋CSP 0.3.0-preview619。

**Spec:** `docs/superpowers/specs/2026-10-19-yabitsu-forest-design.md`

## Global Constraints

- 既存の `yabitsu_*.kn5`、`*_objects.kn5`、レイアウトの data は書き換えない。森は `content/tracks/yabitsu/extension/` に足すだけにする。
- 全部を1つのコマンドで作る：`uv run python -m scripts.forest_build --area pilot|all`。AI による大量の画像確認はしない。確認画像は1回の実行で数枚だけ作る。
- 見た目を直すときは、地点ごとの手直しではなく、`data/manual/forest.json` の値を変えて再実行する。
- 植えない場所：道路・土台・駐車場・分岐路（バリケードの先も）・建物・壁・ガードレール・橋・トンネル坑口・標識類・水面・川原・町。
- 木の根元は地面メッシュに合わせる。浮きも埋まりも 0.3 m 以内。
- 素材は CC0 か自作だけ。出典は `assets/forest/README.md` に書く。
- テキストを書き出すときは `newline="\n"` を使う（リポジトリは LF。Windows の既定 CRLF は不可）。
- ユーザーへの報告は日本語で、初心者向けに書く。
- Codex の未コミット作業は取り込まない。ただし `assets/landscape/broadleaf_prototypes_v2.json`、`assets/landscape/branchleaves_cc0.png`、`assets/landscape/README.md` は除く（Task 3 で `assets/forest/` へ写す）。

## Review Focus

1. **橋の下の川に生える木。** 橋の下の地面に根元がある木は、橋の床（`1WALL_bridge*`、`bridge*`）の平面範囲に入るので、植えない。→ Task 4 `test_keepout_bridge_footprint`
2. **近景と遠景の地形が重なる場所の高さの取り違え。** 近景（`1GRASS_*`）がある場所では、遠景（`TERRAIN_FAR_*`）の高さを使わない。→ Task 4 `test_ground_prefers_near`
3. **地形の端の外側に出る木。** 地面メッシュがない点は捨てる。→ Task 4 `test_ground_nan_outside`
4. **道路の真上で低く張り出す枝。** 樹冠の平面範囲が車道にかかる木は、車道の上での枝の最低高さが 5.0 m 以上になる場合だけ残す。→ Task 5 `test_no_low_crown_over_road`
5. **壁の上の木の根元が道路側にはみ出す。** 壁メッシュの平面範囲＋0.5 m には植えない。→ Task 4 `test_keepout_wall_buffer`

---

### Task 1: CSP 木パッケージの書き出し

**Files:**
- Create: `yabitsu/csptree.py`
- Test: `tests/test_csptree.py`

**Interfaces:**
- Produces:
  - `Submesh(diffuse: int, normal: int, indices: np.ndarray[uint16], verts: np.ndarray)`。`verts` は構造化配列 `VERT_DTYPE = [('p','<3f4'),('n','<3f4'),('uv','<2f4'),('b0','4u1'),('b1','4u1'),('b2','4u1')]`（1頂点44バイト）。
  - `write_mesh(textures: list[str], subs: list[Submesh]) -> bytes`
  - `read_mesh(data: bytes) -> tuple[list[str], list[Submesh]]`
  - `write_package(path: Path, ini: dict, meshes: dict[str, bytes], textures: dict[str, bytes]) -> None`。zip に `tree.ini`、各 `.mesh`、テクスチャを入れる。
  - `tree_line(pkg: str, x: float, y: float, z: float, angle: float, size: float) -> str`。AC 座標で `tree: {pkg}; pos={x:.2f}, {y:.2f}, {z:.2f}; angle = {angle:.0f}; size = {size:.2f}` の形。
  - `to_ac(xyz_local) -> (x, z, -y)`（`docs/ac-conventions.md` の blender_to_ac と同じ）

ファイルの形式（ターンパイクの `treePine1.zip` から読み解いた。LOD1 は全バイトが一致）：
`u32 1`、`u8 1`、`u32 テクスチャ数`。テクスチャごとに `u32 ハッシュ, u32 長さ, 名前`。その後、ファイルの終わりまで部品を繰り返す：`u32 diffuse番号, u32 normal番号, u32 index数, u16[index数], u32 頂点数, 頂点×44B`。
LOD0 には最後に任意の 12 バイト（`ff ff ff ff, u32 4, f32`）が付く。これは書かない。
ハッシュの作り方は不明なので、まず読み込んだ値を保つ。新しく書くときは、名前の FNV-1a 32bit を入れる。AC で読めなかった場合は、Task 6 で CSP のログを見て直す。
座標：y が上。高さは 0〜1、横は ±1 に正規化する。実寸は `tree.ini` の `[BASIC] HEIGHT/WIDTH` で決まる。
`b0` は 127 を中心にまとめた方向ベクトル（xyz、w=0）。`b1` は全部 0。`b2` は `[AO, 255, 255, 0]`。

- [ ] **Step 1: テストを書く**
  - `test_roundtrip_reference`：AC フォルダの `hakoneTurnpike_wmmt5/extension/trees/treePine1.zip` があれば、`Pine_1_LOD1.mesh` を `read_mesh`→`write_mesh` して、元のバイト列と完全一致することを確かめる。ファイルがなければ skip。
  - `test_write_read_synthetic`：三角形2枚・部品2つ → 書いて読むと、`indices`・`verts` が一致する。
  - `test_rejects_over_65535_verts`：頂点が 65536 個の部品は `ValueError` になる。
  - `test_tree_line_format`：`tree_line("sugi_a.zip", 1, 2, 3, 45, 1.2) == "tree: sugi_a.zip; pos=1.00, 2.00, 3.00; angle = 45; size = 1.20"`
- [ ] **Step 2: 実行して失敗を確認する** `uv run pytest tests/test_csptree.py -v` → FAIL（モジュールがない）
- [ ] **Step 3: `yabitsu/csptree.py` を実装する**（ハッシュは読み込んだ値をそのまま保持するため、`read_mesh` が返すテクスチャ名にはハッシュも付ける：`list[tuple[int,str]]` にしてもよい。その場合は Interfaces も合わせる）
- [ ] **Step 4: テストが通ることを確認する** `uv run pytest tests/test_csptree.py -v` → PASS
- [ ] **Step 5: コミットする** `git add yabitsu/csptree.py tests/test_csptree.py && git commit -m "feat: CSP tree package writer"`

### Task 2: 空中写真から森と樹種の地図を作る

**Files:**
- Create: `yabitsu/forestmap.py`、`data/manual/forest.json`
- Test: `tests/test_forestmap.py`

**Interfaces:**
- Consumes: `yabitsu.photo.render_local_image(cache, unproject, x0, y0, size_m, px_size, layer="seamlessphoto", z=18)`、`yabitsu.geo.from_local`
- Produces:
  - `CLASSES = {0: "open", 1: "conifer", 2: "broadleaf", 3: "water"}`
  - `fetch_mosaic(bounds: tuple[float,float,float,float], px_size: float, z: int, cache: Path) -> np.ndarray[H,W,3]`。地理院 seamlessphoto をタイルごとに取得し、キャッシュを使う。
  - `classify(rgb: np.ndarray, px_size: float, cfg: dict) -> np.ndarray[uint8]`。結果は `cell_m`（既定 4 m）ごとのクラス。
  - `ForestMap(xs, ys, cls)` と `ForestMap.at(x, y) -> np.ndarray[uint8]`
  - `preview(rgb, fm, path)`：空中写真｜色分け地図の横並び PNG。
- 範囲：`track_terrain_far.npz` の xs/ys（x −3460〜3760、y −7480〜12200）。写真は z=17（約1.2 m/画素）で取得し、`px_size=1.0` m に並べ直す。
- 判定の考え方（しきい値は `forest.json` の `classify` に置く）：緑の強さ（ExG＝2G−R−B）と明るさで森／森以外を分ける。森の中は、明るさと細かいきめ（3×3 の分散）で針葉樹（暗い・きめが細かい）と広葉樹（明るい・丸いムラ）を分ける。水は manifest の水面と湖で上書きする（Task 4）。
- 出典表示：「地理院タイル（シームレス空中写真）」を README と `assets/forest/README.md` に書く。

- [ ] **Step 1: テストを書く**
  - `test_classify_synthetic`：暗い緑のきめ細かい区画→conifer、明るい緑の区画→broadleaf、灰色・茶色の区画→open。
  - `test_forestmap_at_lookup`：`ForestMap.at` が格子の中心で正しいクラスを返し、範囲外では 0 を返す。
- [ ] **Step 2: 失敗を確認する** `uv run pytest tests/test_forestmap.py -v`
- [ ] **Step 3: 実装し、`forest.json` に初期値を入れる**（`classify`、`density`、`species`、`tiers`、`clearance` の各ブロック。値は Task 4・5 で使う）
- [ ] **Step 4: テストが通ることを確認する**
- [ ] **Step 5: 実データで一度走らせる。** S1 付近 1 km 四方の `preview` を `export/forest/check/` に出し、自分で1枚見る。道路・畑・町が open、山が森になっていなければ、しきい値を直す。
- [ ] **Step 6: コミットする**

### Task 3: 木のモデル（杉・檜・広葉樹・低木・下草）

**Files:**
- Create: `yabitsu/treemodels.py`、`yabitsu/foliage_textures.py`、`assets/forest/README.md`、`assets/forest/broadleaf_prototypes_v2.json`（Codex 版から写す）、`assets/forest/branchleaves_cc0.png`（同）
- Test: `tests/test_treemodels.py`

**Interfaces:**
- Consumes: `csptree.Submesh`、`csptree.write_mesh`、`csptree.write_package`
- Produces:
  - `SPECIES = {"sugi": [...], "hinoki": [...], "broadleaf": [...], "shrub": [...], "fern": [...]}`。各樹種のパターン名の一覧（例 `sugi_a, sugi_b, sugi_c`）。
  - `build_pattern(name: str, seed: int) -> TreeModel`。`TreeModel(lods: list[list[Submesh]], height: float, width: float, crown_base: float, crown_radius: float, textures: dict[str, bytes])`
  - `write_all(out_dir: Path) -> dict[str, TreeModel]`。パターンごとに `out_dir/<name>.zip` を作る。
- 品質の目標（spec：S1 の木並み、おもちゃは不可）：
  - 杉：まっすぐな赤茶の幹。下の方は枝打ちされて枝がない。樹冠は細長い円錐。針葉の房は、写真のような葉テクスチャを貼った板を多数重ねる。
  - 檜：杉より樹冠が丸く、平たい鱗片葉の房。
  - 広葉樹：`broadleaf_prototypes_v2.json` の枝分かれを使う。葉は `branchleaves_cc0.png`（CC0）を元にした房。
  - 低木・シダ：高さ 0.5〜2.5 m。森の地面に使う。
  - LOD は3段階。LOD0 は近くで幹の質感と枝が分かる。LOD2 は遠目で、樹冠の色とムラが LOD0 とそろう。三角形数の目安：LOD0 4,000〜12,000、LOD1 1,500〜4,000、LOD2 300〜1,000（作る人が調整してよい）。
  - 葉のテクスチャ：アルファ付き PNG、法線マップ付き。杉・檜の針葉テクスチャは CC0 素材（ambientCG・Poly Haven など）か、手続きで描く（1024 px 以上、房の重なりと色ムラがあるもの）。出典とライセンスを README に書く。
- `tree.ini` の値：`[BASIC] HEIGHT/WIDTH` はモデルの実寸。`[SHADING]` はターンパイクの例を出発点にする（SPECULAR 0.02〜0.05、SUBSCATTERING 0.3、REFLECTIVITY 0.02）。`[LOD_n] DISTANCE` は 50/120/250 から始める。

- [ ] **Step 1: テストを書く**
  - `test_pattern_bounds_normalized`：全パターン・全 LOD で、y が 0〜1、x/z が ±1 の範囲にある。
  - `test_lod_triangle_budget`：LOD ごとに三角形数が減っていく。LOD0 は 12,000 以下、LOD2 は 1,000 以下。
  - `test_trunk_at_origin`：幹の根元の中心が (0,0,0) から 0.05 以内にある。
  - `test_sugi_has_clear_lower_trunk`：杉の葉の頂点が y < 0.3 にない（枝打ち）。
  - `test_packages_written`：`write_all` が全パターンの zip を作り、zip の中に `tree.ini` と LOD 0〜2 がある。
- [ ] **Step 2: 失敗を確認する**
- [ ] **Step 3: 実装する**（Codex の Blender 用コード `blender_scripts/landscape/tree_assets.py` の枝の作り方を参考にしてよい。ただし numpy だけで動くように書き直す）
- [ ] **Step 4: テストが通ることを確認する**
- [ ] **Step 5: 確認画像を作る。** `blender_scripts/render_tree_patterns.py` で全パターンを横に並べて1枚描画し（`export/forest/check/patterns.png`）、自分で見る。おもちゃっぽさ（単色の塊、規則的すぎる形）がないかを確かめる。
- [ ] **Step 6: コミットする**

### Task 4: 地面の高さと植えない範囲（manifest から）

**Files:**
- Create: `yabitsu/forest_ground.py`
- Test: `tests/test_forest_ground.py`

**Interfaces:**
- Consumes: `data/processed/track/manifest.json` とメッシュ npz（AC に入っている kn5 と同じ 2026-10-19 00:10 のデータ）
- Produces:
  - `load_ground(manifest_dir: Path) -> Ground`。`Ground.z(xy: np.ndarray[N,2]) -> np.ndarray[N]` は、地面がない点では NaN を返す。
  - `load_keepout(manifest_dir: Path, cfg: dict) -> KeepOut`。`KeepOut.blocked(xy) -> np.ndarray[bool]`。`KeepOut.road_distance(xy) -> np.ndarray[float]` は、車道（`1ROAD_*`、`scenery_road*`、`town_lot`）からの平面距離。
- 地面：材質が `ground` のメッシュ。優先順位は、`1GRASS_*`（近景・中景・峠の島）→ `TERRAIN_FAR_*`。三角形の平面に点を落として z を求める（三角形の中心で KD 木を作り、近いものから当てる）。
- 植えない範囲：材質が `ground` と `tree` 以外の全メッシュの平面範囲を、種類ごとに広げる（`forest.json` の `clearance`）。道路 `1ROAD_*`：2.0 m。壁・レール `1WALL_*`、`GUARDRAIL_*`：0.5 m。建物 `*bld_*`、`s?_houses*` など：2.0 m。橋 `bridge*`、`1WALL_bridge*`：3.0 m。水 `water*`、`lake*`、`riverbed*`：1.0 m。トンネル `tunnel*`、`1WALL_tunnel*`：5.0 m。その他：1.0 m。
  これとは別に、Task 2 の open クラスでは植えない。
- 道：Overpass から地形範囲の OSM 車道（`highway` のうち、`track`・`path`・`footway`・`steps`・`cycleway` 以外）を一度だけ取得し、`data/raw/osm_forest_roads.json` に保存する。道幅は `scenery.py` の表と同じにし、+1.5 m 広げて植えない範囲に足す。`highway=track` は足さない（空中写真の判定に任せる）。Overpass は `overpass-api.de` が混んでいたら、`maps.mail.ru` のミラーを使う。

- [ ] **Step 1: テストを書く**（`tests/` に小さな合成 manifest を作るヘルパーを置く）
  - `test_ground_prefers_near`：同じ場所に近景（z=100）と遠景（z=90）がある → 100 を返す。
  - `test_ground_nan_outside`：メッシュの外は NaN を返す。
  - `test_ground_interpolates_triangle`：傾いた三角形の内部で、平面の式どおりの z を返す。
  - `test_keepout_bridge_footprint`：橋の床の真下（地面はある）は blocked。
  - `test_keepout_wall_buffer`：壁から 0.4 m は blocked、0.8 m は blocked でない。
  - `test_road_distance`：車道の帯から 3 m の点で `road_distance` が 3±0.1。
- [ ] **Step 2: 失敗を確認する**
- [ ] **Step 3: 実装する**（植えない範囲は shapely でまとめてから `prepare` し、`contains_xy` で調べる。1つずつ調べると遅い：`yabitsu/trees.py` に同じ注意書きがある）
- [ ] **Step 4: テストが通ることを確認する**
- [ ] **Step 5: 実データで一度読み込む。** 読み込み時間と、植えない範囲の面積を出力する。
- [ ] **Step 6: コミットする**

### Task 5: 木の配置

**Files:**
- Create: `yabitsu/forest_place.py`
- Test: `tests/test_forest_place.py`

**Interfaces:**
- Consumes: `ForestMap`、`Ground`、`KeepOut`、`treemodels.SPECIES` と各パターンの `crown_base/crown_radius/height`、`forest.json`
- Produces: `place(area: Polygon|None, fm, ground, keep, models, cfg, seed=1) -> np.recarray`。各行は `(x, y, z, pattern: str, angle_deg, size)`。

規則（値は `forest.json` に置く）：
- **距離の帯（tiers）**。道路からの距離で、次の3つに分ける。
  - 「歩ける帯」：0〜200 m。本物の密度にする。目安は針葉樹の植林で 1 本 / 12 m²、広葉樹で 1 本 / 25 m²。低木・シダもここに植える。
  - 「見える帯」：200〜800 m。木を大きめにして、本数を減らす。
  - 「遠い帯」：800 m 以上。さらに大きくまばらにする。空から見て地面がほとんど透けないことは保つ。
  帯の境目で急に変わらないように、密度と大きさは距離に応じてなめらかに変える。
- **樹種の選び方**。conifer のマスでは杉・檜を 6:4 で選ぶ。植林らしさを出すため、杉・檜は 30〜80 m 角のまとまりで同じ樹種にそろえる。broadleaf のマスでは広葉樹のパターンをランダムに選ぶ。
- **位置**。ポアソン円盤サンプリング（最小間隔 ≈ 樹冠半径 × 0.8）。根元の z は `Ground.z`。z が NaN の点と、`KeepOut.blocked` の点は捨てる。
- **道路の上の枝**。`road_distance < 樹冠半径×size` の木は、樹冠の底の高さ `crown_base×height×size` が、その地点の道路面＋5.0 m 以上のときだけ残す。満たさない場合は、下枝が高いパターンや大きい size に替えるか、捨てる。
- **向きと大きさ**。angle は 0〜360°。size は 0.8〜1.2 で、帯の倍率をかける。

- [ ] **Step 1: テストを書く**
  - `test_none_in_keepout`：配置した全点で `blocked == False`、z は有限。
  - `test_z_on_ground`：配置した z と `Ground.z` の差が 0.3 m 以内。
  - `test_no_low_crown_over_road`：樹冠が車道にかかる木では、枝の最低高さ ≥ 道路面＋5.0 m。
  - `test_density_by_tier`：合成した全面森の場所で、歩ける帯の 1 ha あたりの本数が、遠い帯より多い。
  - `test_deterministic`：同じ seed なら同じ結果になる。
  - `test_open_class_empty`：open のマスには木がない。
- [ ] **Step 2: 失敗を確認する**
- [ ] **Step 3: 実装する**
- [ ] **Step 4: テストが通ることを確認する**
- [ ] **Step 5: コミットする**

### Task 6: 一括コマンド・AC への導入・確認画像 → AC 試作（ユーザー確認）

**Files:**
- Create: `scripts/forest_build.py`、`yabitsu/forest_ac.py`、`blender_scripts/render_forest.py`
- Modify: `scripts/package_release.py`（Codex の試作ファイルを除く）
- Test: `tests/test_forest_ac.py`、`tests/test_package_release.py`（1件追加）

**Interfaces:**
- Consumes: Task 1〜5 のすべて
- Produces:
  - `forest_ac.write_lists(placements, out_dir: Path, tile_m=1000) -> list[str]`。タイルごとの `trees/forest_<i>_<j>.txt`。先頭行は `configure: angle variance = 0, 0` と `configure: size variance = 1, 1`（ばらつきは各行の angle/size で指定済み）。
  - `forest_ac.write_ext_config(track_dir: Path, lists: list[str], hidden_meshes: list[str]) -> None`。`[TREES]` に `LIST_n=trees\...` と `ACTIVE=1` を書く。古い板の木は `[MESH_ADJUSTMENT_...]` で `MESHES=` に `TREES_*` を列挙し、`IS_ACTIVE=0` にする。既存の `ext_config.ini` が自分の印（`; yabitsu forest`）の付いていないものなら、上書きせずに止まる。
  - Codex の S1 景観確認レイアウト（`s1_landscape_up/down`）では、レイアウトごとの CSP 設定で `[TREES] ACTIVE=0` にする（置き場所は、`shuto_revival_project_beta` の例で確かめる）。
  - コマンド：`uv run python -m scripts.forest_build --area pilot|all [--no-install] [--render]`。順番に、モデル → 地図 → 地面・植えない範囲 → 配置 → `export/forest/<日時>/` へ書き出し → AC へコピー（`--no-install` のときは除く）→ 確認画像（`--render` のとき）。最後に、本数・樹種ごとの数・所要時間を表示する。
  - pilot 範囲：S1 の道路から 400 m 以内の帯、全長分。
  - 確認画像（`blender_scripts/render_forest.py`）：地面メッシュと LOD1 の木（インスタンスで並べる）を Blender で描く。上空から見た画像を、同じ範囲の空中写真と横に並べて2か所。運転席の高さの画像を2か所。合計4枚。
- `package_release.py`：`yabitsu_s1_landscape_test.kn5`、`models_s1_landscape_*.ini`、`s1_landscape_*/`、`ui/s1_landscape_*/` を zip に入れない。

- [ ] **Step 1: テストを書く**
  - `test_write_lists_tiles`：2 タイルにまたがる点が、正しいファイルに分かれる。各行の座標は `to_ac` の変換後のもの。
  - `test_ext_config_contents`：`[TREES]` と `LIST_0`、`MESH_ADJUSTMENT` に `TREES_` の名前が入る。
  - `test_ext_config_refuses_foreign`：印のない既存ファイルがあれば `RuntimeError` になる。
  - `test_release_excludes_codex_test`：パッケージ対象のファイル一覧に `s1_landscape` が含まれない。
- [ ] **Step 2: 失敗を確認する**
- [ ] **Step 3: 実装する**
- [ ] **Step 4: テストが通ることを確認する** `uv run pytest -q`（全体）
- [ ] **Step 5: pilot を走らせる。** `uv run python -m scripts.forest_build --area pilot --render` を、`timeout` と見張り（記憶 feedback-no-passive-waiting）を付けて実行する。確認画像4枚を自分で見て、明らかな破綻（浮いた木、道路上の木、スライム状の塊）があれば `forest.json` かコードを直して再実行する。
- [ ] **Step 6: コミットする**
- [ ] **Step 7: ユーザーに AC で見てもらう。** 普通の「S1 上り」で、走る視点と、降りて森を歩く視点を見てもらう。スクリーンショットをもらう。木が出ないときは、`Documents/Assetto Corsa/logs/custom_shaders_patch.log` を読んで、形式（Task 1 のハッシュなど）を直す。
- [ ] **Step 8: ユーザーの意見を `forest.json` とモデルに反映する。** 直したら再実行し、「S1 の木並み」と OK が出るまで続ける。直したときは、同じ視点の前後画像を見せる（記憶 feedback-verify-with-screenshots）。

### Task 7: 全域に広げる → 走行確認 → まとめ

**Files:**
- Create: `docs/forest.md`（使い方・設定値の意味・出典）
- Modify: `README.md`（森の出典：地理院タイル、CC0 素材）

- [ ] **Step 1: 全域を作る。** `uv run python -m scripts.forest_build --area all --render` を、timeout と見張りを付けて実行する。本数と所要時間を記録する。
- [ ] **Step 2: 確認画像4枚＋空中写真の比較を自分で見る。** 森の形が写真と大きく食い違う場所があれば、地点ごとではなく `classify` のしきい値を直して再実行する。
- [ ] **Step 3: ユーザーに見てもらう。** 比較画像のパスを渡し、AC で「全線」のレイアウトを走ってもらう。FPS を見てもらう。目安は、森を入れる前から大きく下がらないこと。重い場合は、`forest.json` の `tiers`（遠い帯の本数と大きさ）と LOD 距離を下げる。
- [ ] **Step 4: `docs/forest.md` と README を書く。** テストを全部走らせて、コミットする。
- [ ] **Step 5: ブランチ全体のレビュー**（最も性能の高いモデルで1回）。指摘を直し、ユーザーの OK をもらって main へマージする。
