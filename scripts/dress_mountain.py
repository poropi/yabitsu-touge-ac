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

from yabitsu import (acdata, config, dress_mountain as dm, dress_parts, dress_retex, dress_tex, ext_config, features, kn5, layouts,
                     nagoki, side_roads, track, tunnel)
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
    "mt_wall_rock": ("rock_cut", (2, 2), "wall"), "mt_wall_rock_net": ("rock_net", (2, 2), "wall"),
    "mt_wall_norikae": ("norikae", (8, 8), "wall"), "mt_wall_shotcrete": ("shotcrete", (2, 2), "wall"),
    "mt_terrain": ("forest_floor", (2, 2), "terrain"),
}
# ストリートビューと見比べて決めた 明るさ・鮮やかさ・コントラスト
TONE = {"mt_road": 0.68, "mt_plane": 0.68, "mt_road_dark": 0.9, "mt_gutter": 0.72, "mt_wall_kenchi_moss": 0.72,
        "mt_wall_block": 0.82, "mt_wall_rock_net": 0.8, "mt_terrain": 0.85}
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
                assign[name] = f"mt_wall_{fin}"
    assign.update(dm.town_walls(man["meshes"], taken))
    # 地形（土手・林の地面）：中心がいちばん近い道の s が山道の範囲（表の bare_ground を除く）にあるもの
    green = ranges(bounds, rows, "green_ground")
    for name, e in man["meshes"].items():
        if not name.startswith(("1GRASS_s1_terrain", "1GRASS_track_mid")) or name in taken:
            continue
        v = np.load(MANIFEST / e["file"])["verts"]
        s = float(cl["s"][tree.query(v[:, :2].mean(0))[1]])
        if any(bounds[x][0] <= s <= bounds[x][1] for x in dm.SECTIONS) and not inside(s, green):
            assign[name] = "mt_terrain"
    return assign


def write_retex_textures(keys, ext_dir):
    for key in keys:
        base, base_m, layout = RETEX[key]
        u_m, v_m, rot = dress_retex.UV_LAYOUT[layout]
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
        if key == "mt_terrain":
            comp = dm.mossy_ground(comp)
        lum = np.asarray(comp.convert("L"), float) / 255.0
        n = dress_tex.height_to_normal(lum, dress_tex.TEXTURES[base].normal)
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
            piece = features.Feature(a, b, f.side, f.type, f.height_m, f.note)
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
    m.save()
    kn5_path = OUT / f"{GROUP}.kn5"
    kn5.write_from_manifest(OUT / "manifest.json", GROUP, kn5_path)
    print(f"部品：{len(m.meshes)} メッシュ → {kn5_path} {kn5_path.stat().st_size / 1e6:.1f} MB")

    assign = retex_assign(sec, cl, bounds, feats, rows)
    ext = OUT / "extension"; ext.mkdir(parents=True, exist_ok=True)
    write_retex_textures(sorted(set(assign.values())), ext)
    (OUT / "retex.json").write_text(json.dumps(assign, ensure_ascii=False), encoding="utf-8")   # 確認描画用
    (OUT / "hidden.json").write_text(json.dumps(hide), encoding="utf-8")
    part = dress_retex.section_text(assign) + "\n[MESH_ADJUSTMENT_...]\nACTIVE=1\nMESHES=" + ",".join(hide) + "\nIS_RENDERABLE=0\n"
    (ext / "dress_mountain.ini").write_text(part, encoding="utf-8", newline="\n")
    kinds = {}
    for v in assign.values():
        kinds[v] = kinds.get(v, 0) + 1
    print("貼り替え：", kinds, f"、隠す古いガードレール {len(hide) // 2}")

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
