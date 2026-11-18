"""山道（表示の S2〜S5＝内部の s2・s1・s4・s5）の道路まわりの仕上げを作って AC へ入れる（コマンド1回）。
使い方: uv run python -m scripts.dress_mountain [--no-install]
  1. 今ある路面・側溝・白線・壁・地形（土手と林の地面）：CSP の SHADER_REPLACEMENT で山道のテクスチャへ貼り替え（kn5 は作り直さない）
  2. 足す部品（波形のガードレールと支柱・落ち葉）：yabitsu_mountain_dress.kn5
  3. 古いガードレールの表示を隠す（当たり判定は残す）
表：data/manual/{s2,s1,s4,s5}_features.csv（壁・ガードレールの種類）、data/manual/mountain_dress.csv（落ち葉・地面の範囲）。
テクスチャ：assets/dress/（scripts.dress_textures で Codex に作ってもらったもの）。
"""
import argparse
import csv
import json
import shutil

import numpy as np
from PIL import Image, ImageEnhance

from yabitsu import (acdata, config, dress_lines, dress_mountain as dm, dress_parts, dress_retex, dress_tex, ext_config, features, forest_ground, kn5, layouts,
                     geo, nagoki, side_roads, track, tunnel)
from yabitsu.crosssection import Section
from yabitsu.manifest import Manifest

MANIFEST = config.DATA_PROCESSED / "track"
GROUP = "yabitsu_mountain_dress"
OUT = config.DATA_PROCESSED / "dress_mountain"
S3_DRESS = config.DATA_PROCESSED / "dress_s3"
TABLE = config.DATA_MANUAL / "mountain_dress.csv"

# 貼り替えのテクスチャ：キー → (元のテクスチャ, 元の実寸 m, UV の種類)。画像は dress_<キー>.dds（S3 の仕上げと名前が重ならないよう mt_）
RETEX = {
    "mt_road": ("asphalt_mountain", (4, 2), "road"), "mt_plane": ("asphalt_mountain", (4, 2), "plane"),
    "mt_road_dark": ("asphalt_dark", (4, 2), "road"),
    "mt_gutter": ("gutter", (1, 2), "gutter"),        # gutter_mountain は縦の縞が板のように見えた
    "mt_line_white": ("paint_white", (1, 1), "line"), "mt_line_orange": ("paint_orange", (1, 1), "line"),
    "mt_wall_kenchi": ("kenchi", (2, 2), "wall"), "mt_wall_kenchi_moss": ("kenchi_moss", (2, 2), "wall"),
    "mt_wall_block": ("kenchi_block", (2, 2), "wall"), "mt_wall_concrete": ("concrete_wall", (2, 2), "wall"),
    "mt_wall_kitahara": ("kenchi_block", (2, 2), "wall"),
    "mt_wall_block_aged": ("kenchi_block", (2, 2), "wall"),   # 古びて茶色がかった間知ブロック（峠の駐車場の向かい）
    "mt_wall_rock": ("rock_cut", (2, 2), "wall"), "mt_wall_rock_net": ("rock_net", (2, 2), "wall"),
    "mt_wall_norikae": ("norikae", (8, 8), "wall"), "mt_wall_shotcrete": ("shotcrete", (2, 2), "wall"),
    "mt_wall_nozura": ("nozura", (1.9, 1.9), "wall"),     # 無料素材の野面石積み（ヤビツ峠のわき道）
    "mt_terrain": ("forest_floor", (2, 2), "terrain"),
    # 県道 70 号（S4・S5）の擁壁は Codex の表面の画像（data/processed/landmarks/r70_retaining_wall/<名前>_surface.png、横 4 m×縦 1 m で継ぎ目なし、2026-10-31）
    "mt_wall_r70_stone": ("r70:stone", (4, 1), "wall"), "mt_wall_r70_block": ("r70:block", (4, 1), "wall"),
    "mt_wall_r70_shotcrete": ("r70:shotcrete", (4, 1), "wall"), "mt_wall_r70_concrete": ("r70:concrete", (4, 1), "wall"),
    "mt_wall_r70_slope_frame": ("r70:slope_frame", (4, 1), "wall"),
    "mt_grass": ("forest_floor", (2, 2), "terrain"),   # 草地（林の地面の写真を緑に染める。表の grass の範囲と寺山の広場）
}
# ストリートビューと見比べて決めた 明るさ・鮮やかさ・コントラスト
TONE = {"mt_road": 0.68, "mt_plane": 0.68, "mt_road_dark": 0.9, "mt_gutter": 0.72, "mt_wall_kenchi_moss": 0.72, "mt_wall_nozura": 0.62,
        "mt_wall_block": 0.82, "mt_wall_block_aged": 0.56, "mt_wall_rock_net": 0.8, "mt_terrain": 0.85}
SATURATION = {"mt_wall_kenchi_moss": 0.85, "mt_terrain": 0.8, "mt_gutter": 0.8, "mt_wall_rock_net": 0.6}
CONTRAST = {"mt_road": 0.7, "mt_plane": 0.7}


def table_rows():
    return list(csv.DictReader(open(TABLE, encoding="utf-8"))) if TABLE.exists() else []


def ranges(bounds, rows, item):
    """表の item の行（区間名・区間の起点からの s）→ 通しの s の範囲。"""
    return [(bounds[r["section"]][0] + float(r["s_from"]), bounds[r["section"]][0] + float(r["s_to"]), r) for r in rows
            if r["item"] == item]


def inside(s, rngs, side=None):
    return any(a <= s <= b and (side is None or r["side"] in (side, "B", "C")) for a, b, r in rngs)


def override(s, rngs, side):
    """s・左右に当たる表の行の value（なければ None）。"""
    hit = [r["value"] for a, b, r in rngs if a <= s <= b and r["side"] in (side, "B", "C")]
    return hit[0] if hit else None


def side_of(cl, tree, c):
    """点 c が中心線の左（L）か右（R）か。"""
    _, i = tree.query(c)
    j = min(i + 1, len(cl["x"]) - 1); i0 = max(j - 1, 0)
    h = (cl["x"][j] - cl["x"][i0], cl["y"][j] - cl["y"][i0])
    return "L" if h[0] * (c[1] - cl["y"][i]) - h[1] * (c[0] - cl["x"][i]) > 0 else "R"


def track_features(sec, bounds):
    """組み立て（track_build_manifest）と同じ features：番号が GUARDRAIL_s1_<k>・1WALL_s1_retaining_<k> の k。"""
    pads0 = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    _, feats = nagoki.track_stubs(sec, pads0, features.load_features_rows(track.load_section_rows("features", bounds)))
    return feats


def gaps(sec):
    """落ち葉を置かない所：町・分かれ道（多角形の路面）とトンネル。"""
    g = [k.main_skip for _, k in nagoki.other_towns(sec)] + [z.main_skip for z in nagoki.load_splits(sec)]
    g += [(t.s0 - 5, t.s1 + 5) for t in tunnel.from_osm(side_roads.load_elements(), sec)]
    return g


def cut(a, b, gs):
    pieces = [(a, b)]
    for g0, g1 in gs:
        nxt = []
        for x0, x1 in pieces:
            if x1 <= g0 or x0 >= g1:
                nxt.append((x0, x1))
            else:
                if x0 < g0:
                    nxt.append((x0, g0))
                if x1 > g1:
                    nxt.append((g1, x1))
        pieces = nxt
    return [(x0, x1) for x0, x1 in pieces if x1 - x0 >= 5.0]


R70_WALL = {"kenchi_moss": "r70_stone", "kenchi": "r70_stone", "nozura": "r70_stone", "block": "r70_block", "block_aged": "r70_block"}
# 岩・落石防護網は今のまま。Codex の吹き付け・コンクリート・のり枠（*_surface.png）は筋がにじんだ絵で、今の画像より悪く見えたので使わない（2026-10-31 の照合）


# Codex の坑口の絵に掛ける色（ストリートビュー 2025-07 とゲームの撮影 2026-11-01 で、坑口の壁のアーチの上の平均の色の比。ゲームでは白く光り、
# 写真は苔の緑がかった暗い石（大洞）・黒ずんだ古いコンクリート（S5 4091 m）・汚れた灰色のコンクリート（吹風）だった）
PORTAL_TINT = {"s5_2442_portal": np.array([0.23, 0.23, 0.20]),   # 4 回目の撮影で坑門/路面の明るさが写真 0.50・ゲーム 0.80 → 0.62 倍（2026-11-01）
                 # 坑門の面は絵の灰色の地（128,132,120）の一色。0.45・0.50・0.38 と 0.30・0.33・0.26 はゲームで薄い緑に見えた（2026-11-01）
               "s5_4091_portal": np.array([0.49, 0.43, 0.39]),   # 坑門/路面：写真 0.48・ゲーム 0.57 → 0.84 倍
              
               "s5_10663_portal": np.array([0.60, 0.64, 0.67]), "s5_10937_portal": np.array([0.60, 0.64, 0.67])}


# 坑口の上の金網の柵（tunnel の osm id, 端）→ (r70_fence の部品, 並べる数)。S5 4091 m の小さなトンネルの峠側（ストリートビュー R70-0717_f 2025-07）
PORTAL_TOP_FENCE = {("131561907", "a"): ("wire_mesh", 4)}
# 坑口の脇の柱（tunnel の osm id, 端）→ (r70_utility_pole の部品, 塗る色, 坑門の右の端からの横 m, 手前へ m)。S5 4091 m の峠側の右のオレンジの柱
# （ストリートビュー R70-0717_f 2025-07。拾い出し S5-0717-03）
PORTAL_POLE = {("131561907", "a"): ("steel_green", (205, 120, 40), 0.6, -1.0)}
# 坑口の正面を今の坑口の壁の正面に合わせる（tunnel の osm id, 端）。吹風トンネル出口（写真 R70-0992_b）は 1 枚の平らな壁で、Codex の坑口が 2.7 m 張り出して見えた（2026-11-01）
PORTAL_FLUSH = {("32745017", "b")}
# 名板の札の並べ直しと描き直し：坑口の部品 → (字, 札の大きさ m, 札の中心の間隔 m)。写真 R70-0992_b の札は約 0.55 m 角・0.81 m おき、青みの灰色の地に黒い筆文字
PLATE_REDO = {"s5_10937_portal": ("吹風トンネル", 0.55, 0.81)}
# 隠す部品の裏側の写し（kn5 に書き出すとき、両面の材質の部品には裏向きの写し <名前>_back が自動で作られ、元の名前だけを隠すと裏側が残る）。
# 2026-11-01 に吹風トンネル出口の橋で、隠したはずの元の茶色い手すりが欄干の窓と上に見えた。まずこの橋だけ（隠している両面の部品はほかに 94 個ある）
# 2026-11-02：吹風トンネル入口の赤いアーチ橋（S5 10507〜10607 m）で、古い橋の赤い門の裏側の写しが Codex の橋の先に残っていた（写真 R70-0978_f には無い）
HIDE_BACK_TOO = {"bridge_131561909_rail_L", "bridge_131561909_rail_R", "bridge_131561912_arch"}
# 草地の緑（mt_grass）にする地形の塊（表の grass の範囲で分けない）。吹風トンネル入口の坑口の裏の盛り土：写真 R70-0981_f は坑口の上が緑で、
# 植物の隙間から土色の盛り土が横長の帯に見えた（2026-11-01）
GRASS_MESHES = {"1GRASS_s1_terrain_tunnel_32745017_a"}


def r70_part(asset, name):
    """Codex の部品（npz）。無ければ None（今の作り方のまま）。"""
    from yabitsu import landmarks
    return landmarks.load(asset, name) if landmarks.exists(asset, name) else None


def retex_assign(sec, cl, bounds, feats, rows):
    man = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[cl["x"], cl["y"]])
    walls, surface = ranges(bounds, rows, "wall"), ranges(bounds, rows, "surface")
    taken = set()
    if (S3_DRESS / "retex.json").exists():           # S3 の仕上げが貼り替える・隠すものは触らない
        taken |= set(json.loads((S3_DRESS / "retex.json").read_text(encoding="utf-8")))
        taken |= set(json.loads((S3_DRESS / "hidden.json").read_text(encoding="utf-8")))
    assign = {}
    for sname in dm.SECTIONS:
        for name, k in dress_retex.s3_meshes(MANIFEST, cl, bounds[sname]).items():
            if name in taken:
                continue
            if k == "road":
                v = np.load(MANIFEST / man["meshes"][name]["file"])["verts"]
                s = float(cl["s"][tree.query(v[:, :2].mean(0))[1]])
                assign[name] = "mt_road_dark" if override(s, surface, "C") == "dark" else "mt_road"
            elif k == "road_plane":
                assign[name] = "mt_plane"
            elif k == "gutter":
                assign[name] = "mt_gutter"
            elif k in ("line_white", "line_orange"):
                assign[name] = f"mt_{k}"
            elif k == "parapet":
                assign[name] = "mt_wall_concrete"
            elif k == "wall":
                f = feats[int(name.rsplit("_", 1)[1])]
                fin = override((f.s_from + f.s_to) / 2, walls, f.side) or dm.wall_finish(f.note)
                if sname in dm.R70_SECTIONS and fin in R70_WALL and "古い" not in (f.note or ""):   # 「古い」ブロックは古いブロックの絵のまま（2026-11-01）
                    fin = R70_WALL[fin]
                assign[name] = f"mt_wall_{fin}"
    assign.update(dm.town_walls(man["meshes"], taken))
    if "1WALL_s1_objg_retaining_wall_grey" in assign:   # 峠の駐車場の向かいの擁壁の南の端（s4 の擁壁と同じ古びた間知ブロック）
        assign["1WALL_s1_objg_retaining_wall_grey"] = "mt_wall_block_aged"
    if "1WALL_kitahara_obj_retaining" in assign:
        assign["1WALL_kitahara_obj_retaining"] = "mt_wall_kitahara"
    # 地形（土手・林の地面）：中心がいちばん近い道の s が山道の範囲（表の bare_ground を除く）にあるもの
    grass = ranges(bounds, rows, "grass")   # 草地の仕上げ（mt_grass）にする地形の範囲
    for name in man["meshes"]:
        if name.startswith("1GRASS_terayama_") and name not in taken:
            assign[name] = "mt_grass"
        if name.startswith("1GRASS_goma_") and name not in taken:   # 護摩屋敷の水の島：落ち葉の地面（ストリートビュー 045°・090°）
            assign[name] = "mt_terrain"
    green = ranges(bounds, rows, "green_ground")
    for name, e in man["meshes"].items():
        if not name.startswith(("1GRASS_s1_terrain", "1GRASS_track_mid")) or name in taken:
            continue
        v = np.load(MANIFEST / e["file"])["verts"]
        s = float(cl["s"][tree.query(v[:, :2].mean(0))[1]])
        if any(bounds[x][0] <= s <= bounds[x][1] for x in dm.SECTIONS) and not inside(s, green):
            assign[name] = "mt_terrain"
        if inside(s, grass):
            assign[name] = "mt_grass"
    for name in GRASS_MESHES & set(man["meshes"]):
        assign[name] = "mt_grass"
    return assign


def split_grass_terrain(sec, cl, bounds, rows, assign, m: Manifest, tex):
    """草地（mt_grass）に貼った地形の塊のうち、表の grass・green_ground のどちらの範囲にも入らない面（面の真ん中の s で判定）を
    別の形にして mt_terrain にする。残りの面は mt_grass のまま（今までと同じ見た目）。どちらも仕上げの kn5 に足し、元の塊は隠す（当たり判定は元のまま）。
    地形の塊は 300 m ほどあり、寺山富士見橋の草地の範囲に中心が入った塊は、護摩屋敷の水の斜面まで草地になっていた（2026-10-25）。
    戻り値は隠す元の塊の名前。"""
    from scipy.spatial import cKDTree
    man = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    tree = cKDTree(np.c_[cl["x"], cl["y"]])
    near = ranges(bounds, rows, "grass") + ranges(bounds, rows, "green_ground")
    hide = []
    for name, e in man["meshes"].items():
        if assign.get(name) != "mt_grass" or not name.startswith(("1GRASS_s1_terrain", "1GRASS_track_mid")) or name in GRASS_MESHES:
            continue
        z = np.load(MANIFEST / e["file"]); v, f, uv = z["verts"], z["faces"], z["uvs"]
        s = cl["s"][tree.query(v[f].mean(1)[:, :2])[1]]
        g = np.array([inside(x, near) for x in s])
        if g.all():
            continue
        for tag, keep, key in (("grass", g, "mt_grass"), ("terrain", ~g, "mt_terrain")):
            if keep.any():
                used = np.unique(f[keep]); remap = np.full(len(v), -1); remap[used] = np.arange(len(used))
                nm = f"mtsplit_{name[len('1GRASS_'):]}_{tag}"
                m.add_mesh(GROUP, nm, v[used], remap[f[keep]], uv[used], "mt_split_ground")
                assign[nm] = key
        del assign[name]; hide.append(name)
    if hide:   # 絵は貼り替えで決まる（この材質の絵は使われない）
        m.add_material("mt_split_ground", color=(0.30, 0.28, 0.20))
    return hide


def write_retex_textures(keys, ext_dir):
    for key in keys:
        base, base_m, layout = RETEX[key]
        u_m, v_m, rot = dress_retex.UV_LAYOUT[layout]
        if base.startswith("r70:"):
            img = Image.open(config.DATA_PROCESSED / "landmarks" / "r70_retaining_wall" / f"{base[4:]}_surface.png")
        else:
            img, _ = dress_tex.load(base)
        if base in dress_retex.ENLARGE:
            img = dress_retex.enlarge(img.convert("RGB"), dress_retex.ENLARGE[base])
        comp = dress_retex.compose(img.convert("RGB"), base_m, u_m, v_m, rot)
        if TONE.get(key, 1.0) != 1.0:
            comp = ImageEnhance.Brightness(comp).enhance(TONE[key])
        if key == "mt_line_orange":       # Codex の色は実物（ストリートビュー）より鮮やか（S3 と同じ）
            comp = ImageEnhance.Color(comp).enhance(0.75)
        if key in SATURATION:
            comp = ImageEnhance.Color(comp).enhance(SATURATION[key])
        if key in CONTRAST:
            comp = ImageEnhance.Contrast(comp).enhance(CONTRAST[key])
        if key == "mt_wall_nozura":      # ヤビツ峠の石積みは苔で緑がかる（ストリートビュー 2026-10-23）
            comp = Image.fromarray(np.clip(np.asarray(comp, float) * [0.92, 1.0, 0.86], 0, 255).astype(np.uint8))
        if key == "mt_wall_block_aged":   # Street View 駐車場 250°・320° の壁 (141〜153, 127〜136, 106〜117)。ゲームの日なたで約 1.8 倍に明るく見える
            comp = Image.fromarray(np.clip(np.asarray(comp, float) * [1.0, 0.93, 0.84], 0, 255).astype(np.uint8))
        if key == "mt_wall_kitahara":
            comp = Image.fromarray(np.clip(np.asarray(comp, float) * [0.92, 0.86, 0.73], 0, 255).astype(np.uint8))
        if key == "mt_terrain":
            comp = dm.mossy_ground(comp)
        if key == "mt_grass":   # 夏の草地の濃い緑（寺山富士見橋のストリートビュー 270° の広場の草 (155,157,110) にゲームの撮影で合わせた、2026-10-25）
            g = np.asarray(comp.convert("L"), float)[..., None] / 255.0
            comp = Image.fromarray(np.clip((0.55 + 0.9 * g) * np.array([74, 92, 44], float), 0, 255).astype(np.uint8))
        lum = np.asarray(comp.convert("L"), float) / 255.0
        n = dress_tex.height_to_normal(lum, 1.0 if base.startswith("r70:") else dress_tex.TEXTURES[base].normal)
        nm = Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))
        (ext_dir / f"dress_{key}.dds").write_bytes(kn5.dds_bytes(comp))
        (ext_dir / f"dress_{key}_nm.dds").write_bytes(kn5.dds_bytes(nm))


def rail_pieces(f, rails):
    """ガードレール 1 本を、表の rail 行で種類ごとに切る。[(s_from, s_to, 種類)]（種類の既定は白い波形 wbeam）"""
    cuts = sorted({f.s_from, f.s_to} | {x for a, b, r in rails if r["side"] in (f.side, "B") for x in (a, b) if f.s_from < x < f.s_to})
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        kind = override((a + b) / 2, rails, f.side) or "wbeam"
        if out and out[-1][2] == kind:
            out[-1] = (out[-1][0], b, kind)
        elif b - a >= 3.0:
            out.append((a, b, kind))
    return out


def build_parts(sec, bounds, feats, rows, m: Manifest):
    """ガードレール（波形の板と支柱）と落ち葉。隠す古いガードレールの名前を返す。"""
    hide, nb = [], 0
    rails = ranges(bounds, rows, "rail")
    for k, f in enumerate(feats):
        if f.type != "guardrail":
            continue
        sm = (f.s_from + f.s_to) / 2
        if not any(bounds[x][0] <= sm <= bounds[x][1] for x in dm.SECTIONS):
            continue
        for j, (a, b, kind) in enumerate(rail_pieces(f, rails)):
            if kind == "white_bar":              # 白い縦格子の柵（S3 の仕上げと同じ形・画像）
                v, fc, uv = dress_parts.fence_mesh(sec, a, b, f.side, "white_bar", np.zeros(sec.n))
                uv = np.c_[uv[:, 0] * 2.0 / dress_parts.FENCES["white_bar"][0][2], uv[:, 1]]
                m.add_mesh(GROUP, f"mt_rail_{k:03d}_{j}_bars", v, fc, uv, "mt_bars")
                continue
            piece = features.Feature(a, b, f.side, f.type, f.height_m, f.note, f.width_m)   # width_m：路肩の外の縁へ出す幅（2026-11-01）
            in_r70 = any(bounds[x][0] <= (a + b) / 2 <= bounds[x][1] for x in dm.R70_SECTIONS)
            span, end = r70_part("r70_guardrail", "w_beam"), r70_part("r70_guardrail", "w_beam_end")
            if in_r70 and span is not None and end is not None:   # S4・S5：Codex の W 形ガードレールを基準線に沿って並べる
                tone = "rust" if kind == "wbeam_aged" else "white"
                for q, (v, fc, uv, part) in enumerate(dm.r70_guardrail_meshes(sec, piece, {"span": span, "end": end})):
                    m.add_mesh(GROUP, f"mt_rail_{k:03d}_{j}_r70{part}_{q}", v, fc, uv, f"mt_r70_rail_{part}_{tone}")
                continue
            mat = "mt_guardrail_aged" if kind == "wbeam_aged" else "mt_guardrail"
            for q, (v, fc, uv, part) in enumerate(dm.guardrail_meshes(sec, piece)):
                m.add_mesh(GROUP, f"mt_rail_{k:03d}_{j}_{part}_{q}", v, fc, uv, mat if part == "beam" else "mt_post")
        hide += [f"GUARDRAIL_s1_{k:03d}", f"GUARDRAIL_s1_{k:03d}_back"]
        nb += 1
    # 落ち葉：山道の範囲から、町・分かれ道・トンネルと表の no_leaves を除く
    gs = gaps(sec) + [(a, b) for a, b, _ in ranges(bounds, rows, "no_leaves")]
    lane_off = [(a, b) for a, b, _ in ranges(bounds, rows, "no_lane_leaves")]
    k = 0
    for sname in dm.SECTIONS:
        a0, a1 = bounds[sname]
        for a, b in cut(a0, a1, gs):
            for v, fc, uv, part in dm.leaf_meshes(sec, a, b, lane=not any(x0 <= (a + b) / 2 <= x1 for x0, x1 in lane_off)):
                m.add_mesh(GROUP, f"mt_leaves_{part}_{k:03d}", v, fc, uv, f"mt_leaves_{part}")
                k += 1
    print(f"ガードレール {nb} 本、落ち葉 {k} 枚")
    return hide


def add_poles(sec, bounds, m: Manifest):
    """電柱と電線（data/manual/mountain_poles.csv。s は区間の起点から）。並べ方は S3 の町と同じ（dress_parts.utility_poles）。"""
    keep = forest_ground.load_keepout(MANIFEST, {"road": 0.3, "wall": 0.2, "building": 1.0, "bridge": 1.0,
                                                 "water": 0.5, "tunnel": 2.0, "other": 0.3})
    no_sidewalk = np.zeros(sec.n)
    allp = []
    for k, r in enumerate(csv.DictReader(open(config.DATA_MANUAL / "mountain_poles.csv", encoding="utf-8"))):
        s0 = bounds[r["section"]][0]
        poles = dress_parts.utility_poles(sec, s0 + float(r["s_from"]), s0 + float(r["s_to"]), r["side"], float(r["spacing"]),
                                          keep, no_sidewalk, lat_tries=(0.0, 1.5, 3.0))   # 山道は擁壁・ガードレールの外へも探す
        if not poles:
            continue
        allp += poles
        part = r70_part("r70_utility_pole", "concrete") if r["section"] in dm.R70_SECTIONS else None
        if part is not None:      # S4・S5：Codex のコンクリート電柱（腕金・碍子つき）
            v, f, uv = dm.r70_pole_mesh(poles, part)
            m.add_mesh(GROUP, f"mt_poles_{k:03d}", v, f, uv, "mt_r70_pole")
        else:
            v, f, uv = dress_parts.pole_mesh(poles)
            m.add_mesh(GROUP, f"mt_poles_{k:03d}", v, f, uv, "mt_pole")
        v, f, uv = dress_parts.wire_mesh(poles, n_wires=3)
        if len(v):
            m.add_mesh(GROUP, f"mt_wires_{k:03d}", v, f, uv, "mt_wire")
    return len(allp)


def add_r70_structures(sec, bounds, m: Manifest, tex):
    """S4・S5 の橋の欄干・親柱・橋名板と、青宇治橋・赤いアーチ橋の全体、トンネルの坑口を Codex の部品にする
    （data/manual/s45_route70_bridges.json。export/compare/route70/route70_delivery_09.md の割り当て表）。
    近くの今の欄干（bridge_*_rail_L/R、1WALL_track_fence_*）・親柱・坑口の見た目を隠す名前を返す。当たり判定（1WALL_bridge_*_rail・
    トンネルの内壁）は今のまま残す。今の欄干が見つからない橋は置かない（当たり判定の無い欄干を作らない）。"""
    import re
    from scipy.spatial import cKDTree
    from yabitsu import landmarks
    sp = config.DATA_MANUAL / "s45_route70_bridges.json"
    if not sp.exists():
        return []
    spec = json.loads(sp.read_text(encoding="utf-8"))
    meshes = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))["meshes"]
    tree = cKDTree(np.c_[sec.x, sec.y])
    lo, hi = float(bounds["s4"][0]), float(bounds["s5"][1])

    def geo_of(name):
        v = np.load(MANIFEST / meshes[name]["file"])["verts"]
        i = tree.query(v[:, :2])[1]
        h = sec.heading[i]
        d = -(v[:, 0] - sec.x[i]) * np.sin(h) + (v[:, 1] - sec.y[i]) * np.cos(h)
        ss = sec.s[i]
        return {"s0": float(ss.min()), "s1": float(ss.max()), "s": float(np.median(ss)), "side": 1 if np.median(d) > 0 else -1,
                "lat": float(np.median(np.abs(d))), "z": float(v[:, 2].min()), "xy": v[:, :2].mean(0)}

    vis = [n for n, mm in meshes.items() if mm.get("renderable", True)]
    rails = {n: geo_of(n) for n in vis if re.match(r"^(bridge_\d+_rail_[LR]|1WALL_track_fence_\d+)$", n)}
    rails = {n: g for n, g in rails.items() if lo <= g["s"] <= hi}
    mats = {}

    def mat_for(asset, name, tint=None, tag=""):
        key = f"mt_r70_{asset[4:]}_{name}{tag}"
        if key not in mats:
            src = landmarks.texture(asset, name)
            landmarks.copy_texture(src, tex / f"{key}.png")
            if name in PLATE_REDO:   # 名板の札の地と字を写真に合わせて描き直す（色はこのあとの PORTAL_TINT が坑門と同じに掛かる）
                pv, pf, puv = landmarks.load(asset, name)
                im_ = Image.open(tex / f"{key}.png"); a_ = np.asarray(im_.convert("RGB"))
                H_ = a_.shape[0]   # この部品の uv は v＝0 が絵の下（絵の行＝(1−v)×H）
                ys_, cnt_ = np.unique(np.round(pv[:, 1], 3), return_counts=True)
                on_face = (np.abs(np.round(pv[pf][:, :, 1], 3) - ys_[np.argmax(cnt_)]) < 1e-3).all(1)   # 坑門の正面（いちばん頂点の多い y）の面
                c_ = puv[pf[on_face]].mean(1); px = np.c_[c_[:, 0] * H_, (1 - c_[:, 1]) * H_].astype(int).clip(0, H_ - 1)
                cols, n_ = np.unique(a_[px[:, 1], px[:, 0]] // 8, axis=0, return_counts=True)
                face_rgb = cols[np.argmax(n_)] * 8 + 4.0                                # 正面の面でいちばん多い色（坑門の灰色）
                fi = dm._plate_faces(pf, puv, H_)
                cells = sorted({(int(round(puv[pf[t]][:, 0].min() * H_)), int(round((1 - puv[pf[t]][:, 1].max()) * H_)),
                                 int(round(puv[pf[t]][:, 0].max() * H_)), int(round((1 - puv[pf[t]][:, 1].min()) * H_))) for t in fi})
                cells = [c for c in cells if c[2] - c[0] > 20]
                out_ = dm.r70_plate_texture(im_, PLATE_REDO[name][0], cells, face_rgb)
                if im_.mode == "RGBA":
                    out_ = out_.convert("RGBA"); out_.putalpha(im_.getchannel("A"))
                out_.save(tex / f"{key}.png")
            if tint is not None:   # 橋ごとの色（同じ部品を使うほかの橋は変えない）
                im_ = Image.open(tex / f"{key}.png"); a_ = np.asarray(im_.convert("RGBA"), float)
                a_[..., :3] *= np.asarray(tint["rgb"], float)
                if tint.get("brown_dark"):   # 茶色の手すり（赤が青より 25 以上強い所）をさらに暗く
                    br = a_[..., 0] - a_[..., 2] > 25
                    a_[br, :3] *= tint["brown_dark"]
                Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8), "RGBA").convert(im_.mode).save(tex / f"{key}.png")
            if name in PORTAL_TINT:   # 坑口の色をストリートビューに合わせる
                im_ = Image.open(tex / f"{key}.png"); a_ = np.asarray(im_.convert("RGBA"), float)
                a_[..., :3] *= PORTAL_TINT[name]
                Image.fromarray(np.clip(a_, 0, 255).astype(np.uint8), "RGBA").convert(im_.mode).save(tex / f"{key}.png")
            im = Image.open(src)
            alpha = im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 250
            m.add_material(key, texture=f"textures/{key}.png", alpha=bool(alpha), two_sided=True)
            mats[key] = True
        return key

    hide, used, n_rail, n_whole, n_port = [], set(), 0, 0, 0
    lm_path = config.DATA_MANUAL / "s45_route70.json"   # 写真から位置を出して置いた橋名板（本体の組み立て）。同じ橋に二重に置かない
    photo_plates = {t for it in (json.loads(lm_path.read_text(encoding="utf-8"))["landmarks"] if lm_path.exists() else [])
                    for t in ("hinokisawa", "tokoyo", "jigokuzawa", "kiyotaki", "yoshida", "aouji") if t in (it.get("tex") or it["name"])}
    for k, b in enumerate(spec["bridges"]):
        if b.get("skip"):
            continue
        ps = float(bounds[b["section"]][0]) + b["s"]
        half = b["length_m"] / 2 + 30.0
        if "whole" in b:                                  # 青宇治橋・赤いアーチ橋：今の橋（bridge_<id>_*）の見た目を全部隠して、全体を置く
            gids = {re.match(r"^(?:1WALL_)?bridge_(\d+)_girder$", n).group(1) for n in vis if re.match(r"^bridge_\d+_girder$", n)}
            best = min(gids, key=lambda g: abs(geo_of(f"bridge_{g}_girder")["s"] - ps), default=None)
            if best is None or abs(geo_of(f"bridge_{best}_girder")["s"] - ps) > half:
                print("橋が見つからない", b["place"]); continue
            g = geo_of(f"bridge_{best}_girder")
            i = int(np.argmin(np.abs(sec.s - g["s"])))
            part = landmarks.load("r70_bridges", b["whole"])
            v = dm.r70_whole_bridge_verts(sec, i, part[0])   # 部品の +Y を道の進む向きへ（進入の端が −Y＝峠の側）、路面は部品の Z＝0.7 m、高さは道の勾配に沿う
            m.add_mesh(GROUP, f"mt_r70_bridge_{k:02d}_{b['whole']}", v, part[1], part[2], mat_for("r70_bridges", b["whole"]))
            hide += [n for n in vis if re.match(rf"^(1WALL_)?bridge_{best}_", n)]
            used |= {n for n in rails if n.startswith(f"bridge_{best}_")}
            n_whole += 1
            continue
        for side_name, sd in (("left", 1), ("right", -1)):
            cand = [(abs(g["s"] - ps), n) for n, g in rails.items() if n not in used and g["side"] == sd and g["s1"] >= ps - half and g["s0"] <= ps + half]
            if not cand:
                print("欄干が見つからない", b["place"], side_name); continue
            n = min(cand)[1]
            g = rails[n]
            used.add(n); hide.append(n)
            pname = b[side_name]
            span = 3.0 if pname.startswith("log") else 2.0
            v, f, uv = dm.r70_run_mesh(sec, g["s0"], g["s1"], sd, g["lat"], landmarks.load("r70_bridges", pname), span)
            tint = b.get("tint")   # この橋だけの色（写真の苔の付いた古いコンクリートなど）
            m.add_mesh(GROUP, f"mt_r70_bridge_{k:02d}_{side_name}", v, f, uv,
                       mat_for("r70_bridges", pname, tint, f"_b{k:02d}" if tint else ""))
            n_rail += 1
            behind = b.get(f"{side_name}_top")   # 欄干の上に載せる柵（吹風トンネル出口の橋の焦げ茶の鉄の縦格子、写真 R70-0992_b。2026-11-01）
            if behind:
                fa, fn = behind["part"].split(":")
                fp = landmarks.load(fa, fn); fpv = np.array(fp[0], float)
                fpv[:, 2] = behind["z0_m"] + fpv[:, 2] * behind["height_m"] / fpv[:, 2].max()   # 欄干の天端から height_m の高さに縮める
                fv, ff, fu = dm.r70_run_mesh(sec, g["s0"], g["s1"], sd, g["lat"], (fpv, fp[1], fp[2]), 2.0, centered=False)
                key = f"mt_{fa}_{fn}_paint_b{k:02d}"
                if key not in mats:
                    im_ = Image.open(landmarks.texture(fa, fn)).convert("RGB")
                    g_ = np.asarray(im_.convert("L"), float)[..., None] / 255.0
                    Image.fromarray(np.clip((0.45 + 0.75 * g_) * np.array(behind["rgb"], float), 0, 255).astype(np.uint8)).save(tex / f"{key}.png")
                    m.add_material(key, texture=f"textures/{key}.png", two_sided=True); mats[key] = True
                m.add_mesh(GROUP, f"mt_r70_bridge_{k:02d}_{side_name}_top", fv, ff, fu, key)
            if b.get("post"):                             # 親柱：欄干の両端。S5 の今の親柱・銘板は隠す
                for j, at in enumerate((g["s0"], g["s1"])):
                    v, f, uv = dm.r70_post_mesh(sec, at, sd, g["lat"], landmarks.load("r70_bridges", b["post"]))
                    m.add_mesh(GROUP, f"mt_r70_bridge_{k:02d}_{side_name}_post{j}", v, f, uv, mat_for("r70_bridges", b["post"]))
                mm = re.match(r"^bridge_(\d+)_rail_", n)
                if mm:
                    hide += [x for x in vis if re.match(rf"^bridge_{mm.group(1)}_(post|plate)_", x)]
            if b.get("plate") and sd == 1 and b["plate"].replace("_name", "") not in photo_plates:   # 橋名板：左の入口の親柱の脇（写真から位置を出した札が無い橋だけ）
                v, f, uv = dm.r70_post_mesh(sec, g["s0"] - 0.45, sd, g["lat"], landmarks.load("r70_bridges", b["plate"]))
                m.add_mesh(GROUP, f"mt_r70_bridge_{k:02d}_plate", v, f, uv, mat_for("r70_bridges", b["plate"]))
    # トンネルの坑口
    tids = sorted({re.match(r"^tunnel_(\d+)_portal_a$", n).group(1) for n in vis if re.match(r"^tunnel_\d+_portal_a$", n)})
    for t in spec["tunnels"]:
        ps = float(bounds[t["section"]][0]) + t["s"]
        best, bd = None, 1e9
        for tid in tids:
            ga, gb = geo_of(f"tunnel_{tid}_portal_a"), geo_of(f"tunnel_{tid}_portal_b")
            dd = max(0.0, min(ga["s"], gb["s"]) - ps, ps - max(ga["s"], gb["s"]))
            if dd < bd:
                best, bd = (tid, ga, gb), dd
        if best is None or bd > 60:
            print("トンネルが見つからない", t["place"]); continue
        tid, ga, gb = best
        ends = sorted([("a", ga, gb), ("b", gb, ga)], key=lambda e: e[1]["s"])      # 峠の側（s の小さい方）に portal_a の部品
        for (end, g, other), pname in zip(ends, (t["portal_a"], t["portal_b"])):
            out = g["xy"] - other["xy"]; out = out / np.linalg.norm(out)              # 外へ向く向き＝部品の −Y
            if (tid, end) in PORTAL_FLUSH:   # 壁の面に合わせる向き（両端を結ぶ向きはカーブしたトンネルで壁から 10° ほどずれる）
                out = dm.r70_wall_normal(np.load(MANIFEST / meshes[f"tunnel_{tid}_portal_{end}"]["file"])["verts"][:, :2], out)
            uy = -out; ux = np.array([uy[1], -uy[0]])
            part = landmarks.load("r70_tunnels", pname)
            # 置く所はアーチの縁（rim）の真ん中、高さは縁の底（＝路面）。今の坑口の壁（portal：幅 30 m・高さ 10〜37 m）は山肌の切れ目をふさいでいるので
            # 残し、名板だけ隠す。縁は口の内側の裏打ちとして残し、Codex の坑口は縁が外へはみ出さない大きさにする。2026-10-31 に壁ごと隠して壁の底の高さに置いたら、坑口のまわりに穴が開き（奥が見えて鏡のよう）、
            # 坑口が路面より 1〜12.6 m 低く沈んだ（ゲームの画面で見つかった）
            rim = geo_of(f"tunnel_{tid}_rim_{end}")
            rv = np.load(MANIFEST / meshes[f"tunnel_{tid}_rim_{end}"]["file"])["verts"]
            d = rv[:, :2] - rim["xy"]
            k = dm.r70_cover_scale(part, np.c_[d @ ux, rv[:, 2] - rim["z"]])   # 今の縁（高さ 6.5〜7.6 m）が Codex の坑口の外へはみ出さない大きさに
            q = part[0] * k
            if pname in PLATE_REDO:   # 名板の札を写真の大きさ・間隔に並べ直す（倍率 k を掛けたあとの m）
                _, sz, pt = PLATE_REDO[pname]
                q = dm.r70_respace_plate(q, part[1], part[2], sz, pt)
            if (tid, end) in PORTAL_FLUSH:   # 正面（いちばん広い面の y）を今の坑口の壁の正面の 3 cm 手前に
                wv = np.load(MANIFEST / meshes[f"tunnel_{tid}_portal_{end}"]["file"])["verts"]
                wall_out = float(((wv[:, :2] - rim["xy"]) @ out).max())
                ys, cnt = np.unique(np.round(part[0][:, 1], 3), return_counts=True)
                face_y = float(ys[np.argmax(cnt)]) * k
                q = dm.r70_portal_setback(q, face_y, wall_out)
                hide.append(f"tunnel_{tid}_rim_{end}")   # 今の縁取りは壁より 0.25 m 手前に出ていて、下げた坑口の前で名板を隠した（2026-11-01）。口のまわりは今の坑口の壁がふさぐ
                print(f"トンネルの坑口 {t['place']} {pname}：正面を壁の正面（縁から外へ {wall_out:.2f} m）の 3 cm 手前へ {float(q[0, 1] - part[0][0, 1] * k):+.2f} m")
            v = np.c_[rim["xy"][0] + q[:, 0] * ux[0] + q[:, 1] * uy[0], rim["xy"][1] + q[:, 0] * ux[1] + q[:, 1] * uy[1], rim["z"] + q[:, 2]]
            print(f"トンネルの坑口 {t['place']} {pname}：{k:.2f} 倍（今の縁を覆う）")
            m.add_mesh(GROUP, f"mt_r70_tunnel_{tid}_{end}", v, part[1], part[2], mat_for("r70_tunnels", pname))
            hide += [n for n in vis if re.match(rf"^tunnel_{tid}_plate_{end}$", n)]   # 縁は口の内側の裏打ちとして残す
            n_port += 1
            if (tid, end) in PORTAL_TOP_FENCE:   # 坑口の上の金網の柵（坑門の天端の奥寄り、横に並べる）
                fname, nu = PORTAL_TOP_FENCE[(tid, end)]
                fp = landmarks.load("r70_fence", fname); L_ = float(fp[0][:, 0].max())
                yb = float(q[:, 1].min()) + 0.4   # 天端の手前寄り（奥寄りは盛り土の土手に埋まった。2026-11-01）
                ztop = rim["z"] + float(q[q[:, 1] < q[:, 1].min() + 0.8, 2].max())   # 手前の縁の高さ（部品のいちばん高い所では浮いて見えた）
                for u in range(nu):
                    x0 = -nu * L_ / 2 + u * L_
                    fv = np.c_[rim["xy"][0] + (x0 + fp[0][:, 0]) * ux[0] + (yb + fp[0][:, 1]) * uy[0],
                               rim["xy"][1] + (x0 + fp[0][:, 0]) * ux[1] + (yb + fp[0][:, 1]) * uy[1], ztop + fp[0][:, 2]]
                    m.add_mesh(GROUP, f"mt_r70_tunnel_{tid}_{end}_fence_{u}", fv, fp[1], fp[2], mat_for("r70_fence", fname))
            if (tid, end) in PORTAL_POLE:   # 坑口の脇の柱（坑口から見て右＝部品の +X、手前＝−Y）
                pn, rgb, dx, dy = PORTAL_POLE[(tid, end)]
                pp = landmarks.load("r70_utility_pole", pn); x_ = float(q[:, 0].max()) + dx
                pv = np.c_[rim["xy"][0] + (x_ + pp[0][:, 0]) * ux[0] + (dy + pp[0][:, 1]) * uy[0],
                           rim["xy"][1] + (x_ + pp[0][:, 0]) * ux[1] + (dy + pp[0][:, 1]) * uy[1], rim["z"] + pp[0][:, 2]]
                key = f"mt_r70_pole_{pn}_paint"
                if key not in mats:
                    im_ = Image.open(landmarks.texture("r70_utility_pole", pn)).convert("RGB")
                    g_ = np.asarray(im_.convert("L"), float)[..., None] / 255.0
                    Image.fromarray(np.clip((0.45 + 0.75 * g_) * np.array(rgb, float), 0, 255).astype(np.uint8)).save(tex / f"{key}.png")
                    m.add_material(key, texture=f"textures/{key}.png", two_sided=True); mats[key] = True
                m.add_mesh(GROUP, f"mt_r70_tunnel_{tid}_{end}_pole", pv, pp[1], pp[2], key)
    # 坑口の上の帯（壁の上から山の中が見えた）は、坑口の壁を筒のまわりだけにして、その上を盛り土でふさいだ（yabitsu/tunnel.py の _portal・_backfill）
    hide += [f"{n}_back" for n in hide if n in HIDE_BACK_TOO]   # 裏側の写しも隠す
    print(f"Codex の橋：欄干 {n_rail} 本・全体 {n_whole} 本、トンネルの坑口 {n_port} 個、隠す今の見た目 {len(set(hide))}")
    return sorted(set(hide))


def add_road_marks(sec, bounds, m: Manifest, tex):
    """路面の文字・数字・ひし形（data/manual/road_markings.csv。s は区間の起点から）。"""
    marks = [{**r, "s": float(r["s"]) + bounds[r["section"]][0]}
             for r in csv.DictReader(open(config.DATA_MANUAL / "road_markings.csv", encoding="utf-8"))]
    for name, v, f, uv, mat in dress_lines.road_mark_meshes(sec, marks):
        key = mat.removeprefix("mt_mark_")
        if key.startswith("c"):           # 文字 1 字：c<文字コード>_<色>
            code, color = key[1:].split("_")
            img = dress_lines.mark_texture("text", chr(int(code, 16)), color)
        elif key.startswith("l"):         # 横書きの文字列：l<文字コード-文字コード-…>_<色>
            codes, color = key[1:].split("_")
            img = dress_lines.mark_texture("label", "".join(chr(int(c, 16)) for c in codes.split("-")), color)
        elif key.startswith("n"):         # 数字：n<数字>_<色>
            value, color = key[1:].split("_")
            img = dress_lines.mark_texture("number", value, color)
        else:
            img = dress_lines.mark_texture("diamond", "", "white")
        img.save(tex / f"{mat}.png")
        m.add_material(mat, texture=f"textures/{mat}.png", alpha=True)
        m.add_mesh(GROUP, name, v, f, uv, mat)
    return len(marks)


def add_kitahara_details(m, tex):
    """受け入れ済みの北原KN5を保ち、既存の柵を金網に差し替え、電柱と保護カバーを足す。"""
    base = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    old = "kitahara_obj_railing_bars"
    z = np.load(MANIFEST / base["meshes"][old]["file"])
    panel, posts = dm.chainlink_parts(z["verts"], z["faces"], z["uvs"])
    dm.chainlink_texture().save(tex / "chainlink.png")
    dm.pole_cover_texture().save(tex / "pole_cover.png")
    m.add_material("mt_chainlink", texture="textures/chainlink.png", alpha=True, two_sided=True)
    m.add_material("mt_chainlink_post", color=(0.67, 0.69, 0.64))
    m.add_material("mt_pole_cover", texture="textures/pole_cover.png")
    m.add_mesh(GROUP, "mt_kitahara_chainlink", *panel, "mt_chainlink")
    m.add_mesh(GROUP, "mt_kitahara_fence_posts", *posts, "mt_chainlink_post")
    surf = forest_ground.load_ground(MANIFEST, prefixes=("1ROAD_kitahara_sidewalk_",))
    poles = []
    for row in json.loads((config.DATA_MANUAL / "kitahara_poles.json").read_text())["poles"]:
        x, y = map(float, geo.to_local(*row["at"]))
        z0 = surf.z(np.array([[x, y]]))[0]
        if not np.isfinite(z0):
            raise ValueError(f"北原の電柱が歩道の外にあります: {row['at']}")
        poles.append((x, y, float(z0)))
    m.add_mesh(GROUP, "mt_kitahara_poles", *dress_parts.pole_mesh(poles), "mt_pole")
    m.add_mesh(GROUP, "mt_kitahara_wires", *dress_parts.wire_mesh(poles, n_wires=4), "mt_wire")
    for k, (x, y, z0) in enumerate(poles):
        m.add_mesh(GROUP, f"mt_kitahara_cover_{k}", *dm.cylinder((x, y, z0 + 0.5), 0.19, 1.6), "mt_pole_cover")
    return [old, old + "_back"], len(poles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-install", action="store_true")
    a = ap.parse_args()
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    cl = dict(np.load(config.DATA_PROCESSED / "track_centerline.npz"))
    bounds = track.load_bounds()
    feats = track_features(sec, bounds)
    man = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    missing = [f"GUARDRAIL_s1_{k:03d}" for k, f in enumerate(feats) if f.type == "guardrail" and f"GUARDRAIL_s1_{k:03d}" not in man["meshes"]]
    if missing:
        raise SystemExit(f"組み立て後に features が変わっています（番号が合いません）: {missing[:5]}")
    rows = table_rows()

    if OUT.exists():
        shutil.rmtree(OUT)
    m = Manifest(OUT)
    tex = OUT / "textures"; tex.mkdir(parents=True)
    dm.guardrail_texture().save(tex / "guardrail.png")
    dm.guardrail_texture(seed=12, aged=True).save(tex / "guardrail_aged.png")
    from scripts.dress_build import fence_textures
    fence_textures(tex)
    leaves, _ = dress_tex.load("leaves_road")
    edge, lane = dm.leaf_textures(leaves)
    edge.save(tex / "leaves_edge.png"); lane.save(tex / "leaves_lane.png")
    m.add_material("mt_guardrail", texture="textures/guardrail.png", two_sided=True)
    m.add_material("mt_guardrail_aged", texture="textures/guardrail_aged.png", two_sided=True)
    m.add_material("mt_bars", texture="textures/fence_bars.png", alpha=True, two_sided=True)
    m.add_material("mt_post", color=(0.80, 0.81, 0.78))
    m.add_material("mt_leaves_edge", texture="textures/leaves_edge.png", alpha=True)
    m.add_material("mt_leaves_lane", texture="textures/leaves_lane.png", alpha=True)
    hide = build_parts(sec, bounds, feats, rows, m)
    nmarks = add_road_marks(sec, bounds, m, tex)
    pole_img, _ = dress_tex.load("pole")
    # 山道の電柱は雨のしみのある古いコンクリートでくすんだ灰茶色（ストリートビュー 青山荘 B090 の日なたの柱 (78,70,55)）。
    # 元の画像 (176,174,171) はゲームの日なたでほぼ白に見えた（縦の面はゲームで約 1.8 倍に明るく見える）ので、暗く・少し黄土色に（2026-10-26）
    # ゲームの撮影（門戸口橋 v1、2026-10-26）で柱と路面の明るさの比が 0.55（写真 0.42）、赤／青が 0.87（写真 1.42。空の青い光で青く寄る）
    # だったので、さらに 0.76 倍に暗くし、黄土色の側へ寄せた
    Image.fromarray(np.clip(np.asarray(pole_img.convert("RGB"), float) * [0.36, 0.32, 0.236], 0, 255).astype(np.uint8)).save(tex / "pole_mt.png")
    # 名前を S3 の町の仕上げ（yabitsu_s3_dress.kn5 の pole.dds、明るい元の画像）と分ける。同じ名前だとゲームは S3 の画像を山道の電柱にも使い、
    # 暗くした画像が効かなかった（2026-10-26 の撮影 v1・v2 で柱の色が同じ (146,152,164)）
    m.add_material("mt_pole", texture="textures/pole_mt.png")
    m.add_material("mt_wire", color=(0.12, 0.12, 0.12), two_sided=True)
    from yabitsu import landmarks   # S4・S5 の Codex の部品の画像（1 辺 1024 まで）
    LMD = config.DATA_PROCESSED / "landmarks"
    for mat, src in (("mt_r70_pole", LMD / "r70_utility_pole/concrete.png"),
                     ("mt_r70_rail_span_white", LMD / "r70_guardrail/w_beam.png"), ("mt_r70_rail_span_rust", LMD / "r70_guardrail/w_beam_rust.png"),
                     ("mt_r70_rail_end_white", LMD / "r70_guardrail/w_beam_end.png"), ("mt_r70_rail_end_rust", LMD / "r70_guardrail/w_beam_end_rust.png")):
        if src.exists():
            landmarks.copy_texture(src, tex / f"{mat}.png")
            m.add_material(mat, texture=f"textures/{mat}.png", two_sided=mat.startswith("mt_r70_rail"))
    npoles = add_poles(sec, bounds, m)
    hide += add_r70_structures(sec, bounds, m, tex)
    kita_hidden, kita_poles = add_kitahara_details(m, tex)
    hide += kita_hidden
    npoles += kita_poles
    assign = retex_assign(sec, cl, bounds, feats, rows)
    split_hidden = split_grass_terrain(sec, cl, bounds, rows, assign, m, tex)   # 草地の範囲の境をまたぐ地形の塊を分ける
    hide += split_hidden
    m.save()
    kn5_path = OUT / f"{GROUP}.kn5"
    kn5.write_from_manifest(OUT / "manifest.json", GROUP, kn5_path)
    print(f"部品：{len(m.meshes)} メッシュ（路面の標示 {nmarks}、電柱 {npoles} 本）→ {kn5_path} {kn5_path.stat().st_size / 1e6:.1f} MB")

    ext = OUT / "extension"; ext.mkdir(parents=True, exist_ok=True)
    write_retex_textures(sorted(set(assign.values())), ext)
    (OUT / "retex.json").write_text(json.dumps(assign, ensure_ascii=False), encoding="utf-8")   # 確認描画用
    (OUT / "hidden.json").write_text(json.dumps(hide), encoding="utf-8")
    part = dress_retex.section_text(assign) + "\n[MESH_ADJUSTMENT_...]\nACTIVE=1\nMESHES=" + ",".join(hide) + "\nIS_RENDERABLE=0\n"
    (ext / "dress_mountain.ini").write_text(part, encoding="utf-8", newline="\n")
    kinds = {}
    for v in assign.values():
        kinds[v] = kinds.get(v, 0) + 1
    print("貼り替え：", kinds, f"、隠す古いガードレール {(len(hide) - len(split_hidden)) // 2}、分けた地形の塊 {split_hidden}")

    if a.no_install:
        return
    T = config.TRACK_DIR
    shutil.copy2(kn5_path, T / kn5_path.name)
    for f in ext.glob("dress_*.dds"):
        shutil.copy2(f, T / "extension" / f.name)
    ext_config.write_part(T, "dress_mountain", part)
    ext_config.assemble(T)
    for lay in layouts.load_layouts():
        acdata.write_models_ini(T / f"models_{lay.name}.ini", layouts.kn5_files(lay.name))
    print("AC へ入れました：", T)


if __name__ == "__main__":
    main()
