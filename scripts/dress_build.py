"""S3（名古木→蓑毛）の道路まわりの仕上げを作って AC へ入れる（コマンド1回）。
使い方: uv run python -m scripts.dress_build --section s3 [--no-install]
  1. 今ある路面・白線・側溝・歩道・壁：CSP の SHADER_REPLACEMENT で、種類ごとのテクスチャへ貼り替え（kn5 は作り直さない）
  2. 足す部品（柵・電柱と電線・マンホール・グレーチング・補修跡）：yabitsu_s3_dress.kn5
  3. 古い柵の表示を隠す（IS_RENDERABLE=0。当たり判定は残す）
表：data/manual/s3_features.csv（壁・柵の種類）、data/manual/s3_dress.csv（電柱・側溝・路面・柵）。
テクスチャ：assets/dress/（scripts.dress_textures で Codex に作ってもらったもの）。
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from yabitsu import (acdata, config, dress_lines, dress_parts, dress_retex, dress_tex, ext_config, features, forest_ground, kn5,
                     layouts, track)
from yabitsu.crosssection import Section
from yabitsu.manifest import Manifest

MANIFEST = config.DATA_PROCESSED / "track"
GROUP = "yabitsu_s3_dress"
OUT = config.DATA_PROCESSED / "dress_s3"

# 貼り替えのテクスチャ：キー → (元のテクスチャ, 元の実寸 m, UV の種類)
RETEX = {
    "road_town": ("asphalt_town", (4, 2), "road"), "road_dark": ("asphalt_dark", (4, 2), "road"),
    "plane_town": ("asphalt_town", (4, 2), "plane"), "gutter": ("gutter", (1, 2), "gutter"),
    "sidewalk": ("sidewalk", (2, 2), "sidewalk"), "curb": ("curb", (1, 1), "sidewalk"),
    "line_white": ("paint_white", (1, 1), "line"), "line_orange": ("paint_orange", (1, 1), "line"),
    "wall_kenchi": ("kenchi", (2, 2), "wall"), "wall_block": ("block_wall", (2, 1), "wall"),
    "wall_concrete": ("concrete_wall", (2, 2), "wall"),
}


def dress_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def _in(rows, item, s, side=None):
    for r in rows:
        if r["item"] == item and float(r["s_from"]) <= s <= float(r["s_to"]) and (side is None or r["side"] in (side, "C")):
            return r
    return None


# ---------- 貼り替え ----------

def retex_assign(cl, rows, s_range):
    kinds = dress_retex.s3_meshes(MANIFEST, cl, s_range)
    walls = [n for n, k in kinds.items() if k == "wall"]
    finish = dress_retex.wall_finish(MANIFEST, cl, config.DATA_MANUAL / "s3_features.csv", walls)
    m = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[cl["x"], cl["y"]])
    assign = {}
    for name, k in kinds.items():
        if k == "road":
            v = np.load(MANIFEST / m["meshes"][name]["file"])["verts"]
            s = cl["s"][tree.query(v[:, :2].mean(0))[1]]
            r = _in(rows, "surface", s)
            assign[name] = "road_dark" if r is not None and r["value"] == "dark" else "road_town"
        elif k == "road_plane":
            assign[name] = "plane_town"
        elif k == "wall":
            assign[name] = f"wall_{finish.get(name, 'kenchi')}"
        elif k in ("line_white", "line_orange"):
            continue                      # 線は引き直す（dress_lines）ので貼り替えない
        elif k == "parapet":
            assign[name] = "wall_concrete"
        else:
            assign[name] = k
    return assign


def write_retex_textures(keys, ext_dir: Path):
    for key in keys:
        base, base_m, layout = RETEX[key]
        u_m, v_m, rot = dress_retex.UV_LAYOUT[layout]
        img, _ = dress_tex.load(base)
        comp = dress_retex.compose(img.convert("RGB"), base_m, u_m, v_m, rot)
        if key in ("road_town", "plane_town"):   # 町の区間の路面は写真より明るく見えた（2026-10-19 全地点の見直し）
            from PIL import ImageEnhance
            comp = ImageEnhance.Brightness(comp).enhance(0.82)
        if key == "line_orange":       # Codex の色は実物（ストリートビュー）より鮮やか
            from PIL import ImageEnhance
            comp = ImageEnhance.Color(comp).enhance(0.75)
        lum = np.asarray(comp.convert("L"), float) / 255.0
        n = dress_tex.height_to_normal(lum, dress_tex.TEXTURES[base].normal)
        nm = Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8))
        (ext_dir / f"dress_{key}.dds").write_bytes(kn5.dds_bytes(comp))
        (ext_dir / f"dress_{key}_nm.dds").write_bytes(kn5.dds_bytes(nm))


# ---------- 足す部品 ----------

def fence_textures(tex_dir: Path):
    """柵の模様（スクリプトで描く）：横＝高さ（0〜1）、縦＝長さ 4 m。"""
    tex_dir.mkdir(parents=True, exist_ok=True)
    W, H = 128, 512                       # 高さ方向 128 px、長さ 4 m＝512 px（1 px＝7.8 mm）
    bars = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(bars)
    white = (236, 238, 236, 255)
    d.rectangle((0, 0, 8, H), fill=white); d.rectangle((W - 10, 0, W, H), fill=white)        # 下と上の横桟
    for y in range(0, H, 16):                                                                 # 縦格子（12.5 cm おき）
        d.rectangle((0, y, W, y + 3), fill=white)
    bars.save(tex_dir / "fence_bars.png")
    mesh = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(mesh)
    green = (70, 120, 80, 255)
    for k in range(-H, H + W, 12):                                                            # ひし形の金網
        d.line([(0, k), (W, k + W)], fill=green, width=2); d.line([(0, k + W), (W, k)], fill=green, width=2)
    d.rectangle((0, 0, 5, H), fill=green); d.rectangle((W - 5, 0, W, H), fill=green)
    mesh.save(tex_dir / "fence_mesh.png")
    rw = Image.new("RGB", (8, H), (240, 240, 238)); d = ImageDraw.Draw(rw)
    for y in range(0, H, 128):                                                                # 紅白（1 m おき）
        d.rectangle((0, y, 8, y + 63), fill=(200, 40, 40))
    rw.save(tex_dir / "fence_redwhite.png")
    pole, _ = dress_tex.load("pole"); pole.convert("RGB").save(tex_dir / "pole.png")
    for k in ("manhole", "grating"):
        img, _ = dress_tex.load(k); img.save(tex_dir / f"{k}.png")
    # 補修跡：黒いアスファルトをでこぼこの形に切り抜く
    a, _ = dress_tex.load("asphalt_dark")
    a = a.convert("RGB").resize((256, 384))
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:384, 0:256]
    r = 1 - (((xx - 128) / 120) ** 2 + ((yy - 192) / 180) ** 2)
    from scipy.ndimage import gaussian_filter
    r = r + gaussian_filter(rng.normal(0, 1, r.shape), 12) * 3
    patch = a.convert("RGBA"); patch.putalpha(Image.fromarray(((r > 0) * 255).astype(np.uint8)))
    patch.save(tex_dir / "patch.png")


def marking_textures(tex_dir: Path):
    """路面の標示の画像（白・オレンジの塗料、アルファ付き）。v＝進む向き。"""
    from PIL import ImageFont
    W, H = 128, 427                                         # 1.5 m × 5 m
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    white = (236, 236, 232, 255)
    pts = [(W / 2, 4), (W - 4, H / 2), (W / 2, H - 4), (4, H / 2)]
    d.polygon(pts, outline=white, width=14)
    img.save(tex_dir / "diamond.png")
    for t in ("40",):
        img = Image.new("RGBA", (180, 334), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 170)
        tmp = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((90, 90), t, fill=(222, 170, 70, 255), font=font, anchor="mm")
        img = tmp.resize((180, 334)).transpose(Image.FLIP_TOP_BOTTOM)       # 縦に引きのばす（路面の文字）・v が上向き
        img.save(tex_dir / f"text_{t}.png")


FENCE_MAT = {"wbeam": "dress_white", "wbeam_redwhite": "dress_redwhite", "white_bar": "dress_bars",
             "white_pipe": "dress_white", "green_mesh": "dress_mesh"}


def cut_gaps(runs, gaps):
    """(s_from, s_to, ...) の並びから gaps の範囲を切り取る。"""
    out = []
    for a, b, *rest in runs:
        pieces = [(a, b)]
        for g0, g1 in gaps:
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
        out += [(x0, x1, *rest) for x0, x1 in pieces if x1 - x0 >= 3.0]
    return out


def split_gaps():
    """島で道が分かれる所（s3_lines.csv の gap）。ここは本線の左右を基準にした部品を置かない。"""
    return [(float(r["s_from"]), float(r["s_to"])) for r in csv.DictReader(open(config.DATA_MANUAL / "s3_lines.csv", encoding="utf-8"))
            if r["item"] == "gap"]


def fence_runs(feats_csv, rows):
    """柵の範囲：s3_features（ガードパイプ・ガードレール・柵）＋ s3_dress の fence。
    s3_dress（ストリートビュー 166 枚で確かめた表）が重なる所は s3_dress を優先し、features の行はその外だけ残す。"""
    dress = [(float(r["s_from"]), float(r["s_to"]), r["side"], r["value"]) for r in rows if r["item"] == "fence"]
    runs = []
    for r in csv.DictReader(open(feats_csv, encoding="utf-8")):
        if r["type"] == "guardrail":
            kind = "white_pipe" if "パイプ" in r["note"] else "wbeam"
        elif r["type"] == "fence":
            kind = "green_mesh" if "金網" in r["note"] else "white_bar"
        else:
            continue
        pieces = [(float(r["s_from"]), float(r["s_to"]))]
        for d0, d1, side, k in dress:
            if side != r["side"] or k == "green_mesh":      # 金網はガードレールの後ろに足すもの（重ねてよい）
                continue
            nxt = []
            for x0, x1 in pieces:
                if x1 <= d0 or x0 >= d1:
                    nxt.append((x0, x1))
                else:
                    if x0 < d0:
                        nxt.append((x0, d0))
                    if x1 > d1:
                        nxt.append((d1, x1))
            pieces = nxt
        runs += [(x0, x1, r["side"], kind) for x0, x1 in pieces if x1 - x0 >= 3.0]
    return runs + dress


def build_parts(sec, rows, m: Manifest, s_range):
    feats = features.load_features_csv(config.DATA_MANUAL / "s3_features.csv")
    swL, swR = features.sidewalk_offsets(sec, feats)
    sidewalk = {"L": swL, "R": swR}
    # 柵
    gaps = split_gaps()
    for k, (a, b, side, kind) in enumerate(cut_gaps(fence_runs(config.DATA_MANUAL / "s3_features.csv", rows), gaps)):
        if kind == "white_pipe":
            v, f, uv = dress_parts.fence_mesh(sec, a, b, side, "wbeam", sidewalk[side])     # 形は波形の代わりに細い3段
        else:
            v, f, uv = dress_parts.fence_mesh(sec, a, b, side, kind, sidewalk[side])
        if kind in ("white_bar", "green_mesh"):
            hgt = dress_parts.FENCES[kind][0][2]
            uv = np.c_[uv[:, 0] * 2.0 / hgt, uv[:, 1]]          # 板の横（高さ）を 0〜1 に
        m.add_mesh(GROUP, f"dress_fence_{k:03d}", v, f, uv, FENCE_MAT[kind])
    # 電柱と電線（電柱は道路・建物を避ける）
    keep = forest_ground.load_keepout(MANIFEST, {"road": 0.3, "wall": 0.2, "building": 1.0, "bridge": 1.0,
                                                 "water": 0.5, "tunnel": 2.0, "other": 0.3})
    allp = []
    for k, r in enumerate(x for x in rows if x["item"] == "poles"):
        poles = [p for a, b in [(x0, x1) for x0, x1, *_ in cut_gaps([(float(r["s_from"]), float(r["s_to"]))], gaps)]
                 for p in dress_parts.utility_poles(sec, a, b, r["side"], float(r["value"]), keep, sidewalk[r["side"]])]
        if len(poles) < 1:
            continue
        allp += poles
        v, f, uv = dress_parts.pole_mesh(poles)
        m.add_mesh(GROUP, f"dress_poles_{k:02d}", v, f, uv, "dress_pole")
        nw = int("".join(c for c in r["note"].split("電線")[-1][:2] if c.isdigit()) or 4) if "電線" in r["note"] else 4
        v, f, uv = dress_parts.wire_mesh(poles, n_wires=nw)
        if len(v):
            m.add_mesh(GROUP, f"dress_wires_{k:02d}", v, f, uv, "dress_wire")
    # 路面の貼り物
    rules = []
    for r in rows:
        if r["item"] == "manhole":
            rules.append((float(r["s_from"]), float(r["s_to"]), "manhole", float(r["value"])))
        elif r["item"] == "gutter" and r["value"] == "grating" and r["side"] == "L":
            rules.append((float(r["s_from"]), float(r["s_to"]), "grating", 25.0))
    rules = [(a, b, k, sp) for a, b, k, sp in cut_gaps([(a, b, k, sp) for a, b, k, sp in rules], gaps)]
    decals = dress_parts.road_decals(sec, rules, seed=7)
    for kind in ("manhole", "grating"):      # 補修跡は写真に目立つものが無いので置かない（2026-10-19 全地点の見直し）
        parts = [(v, f, uv) for v, f, uv, kk in decals if kk == kind]
        if not parts:
            continue
        off = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        m.add_mesh(GROUP, f"dress_{kind}", np.concatenate([p[0] for p in parts]),
                   np.concatenate([p[1] + o for p, o in zip(parts, off)]), np.concatenate([p[2] for p in parts]),
                   f"dress_{kind}")
    return len(allp)


def old_fence_meshes(cl, s_range):
    m = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[cl["x"], cl["y"]])
    out = []
    for name, e in m["meshes"].items():
        if name.startswith(("GUARDRAIL_", "1WALL_track_fence")):
            v = np.load(MANIFEST / e["file"])["verts"]
            d, i = tree.query(v[:, :2].mean(0))
            if d < 60 and s_range[0] <= cl["s"][i] <= s_range[1]:
                out += [name, f"{name}_back"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="s3", choices=["s3"])
    ap.add_argument("--no-install", action="store_true")
    a = ap.parse_args()
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    cl = dict(np.load(config.DATA_PROCESSED / "track_centerline.npz"))
    s_range = track.load_bounds()["s3"]
    rows = dress_rows(config.DATA_MANUAL / "s3_dress.csv")

    # 足す部品 → kn5
    if OUT.exists():
        shutil.rmtree(OUT)
    m = Manifest(OUT)
    tex = OUT / "textures"
    fence_textures(tex)
    m.add_material("dress_white", color=(0.925, 0.93, 0.92), two_sided=True)
    m.add_material("dress_redwhite", texture="textures/fence_redwhite.png", two_sided=True)
    m.add_material("dress_bars", texture="textures/fence_bars.png", alpha=True, two_sided=True)
    m.add_material("dress_mesh", texture="textures/fence_mesh.png", alpha=True, two_sided=True)
    m.add_material("dress_pole", texture="textures/pole.png")
    m.add_material("dress_wire", color=(0.12, 0.12, 0.12), two_sided=True)
    for k in ("manhole", "grating", "patch"):
        m.add_material(f"dress_{k}", texture=f"textures/{k}.png", alpha=k != "grating")
    for k in ("white", "orange"):
        img, _ = dress_tex.load(f"paint_{k}")
        img = img.convert("RGB").resize((256, 256))
        if k == "orange":       # Codex の色は実物（ストリートビュー）より鮮やか
            from PIL import ImageEnhance
            img = ImageEnhance.Color(img).enhance(0.75)
        img.save(tex / f"line_{k}.png")
        m.add_material(f"dress_line_{k}", texture=f"textures/line_{k}.png")
    m.add_material("dress_green", color=(0.17, 0.27, 0.20))   # 写真の緑の路側帯（くすんだ濃い緑）
    marking_textures(tex)
    m.add_material("dress_diamond", texture="textures/diamond.png", alpha=True)
    for t in {r["value"] for r in csv.DictReader(open(config.DATA_MANUAL / "s3_lines.csv", encoding="utf-8")) if r["item"] == "text"}:
        m.add_material(f"dress_text_{t}", texture=f"textures/text_{t}.png", alpha=True)
    npoles = build_parts(sec, rows, m, s_range)
    # 路面の線：今の線を隠し、表（s3_lines.csv）どおりに引き直す。隠す線が S3 の外（S2 の頭）にはみ出す分は、もとと同じ線を引く
    hide_lines, (lo, hi) = dress_lines.hidden_line_meshes(MANIFEST, cl, s_range)
    line_rows = list(csv.DictReader(open(config.DATA_MANUAL / "s3_lines.csv", encoding="utf-8")))
    if hi > s_range[1]:
        line_rows += [{"s_from": s_range[1], "s_to": hi, "item": "centre", "side": "C", "value": "orange_solid"},
                      {"s_from": s_range[1], "s_to": hi, "item": "edge", "side": "B", "value": "solid"}]
    for name, v, f, uv, mat in dress_lines.line_meshes(sec, line_rows):
        m.add_mesh(GROUP, name, v, f, uv, mat)
    # 島で道が分かれる所：今の白線（ガタガタ・切れて見える）を隠し、なめらかな線を引き直す
    from yabitsu import nagoki
    for spl in nagoki.load_splits(sec):
        for name, v, f, uv, mat in dress_lines.split_lines(spl, spl.ground.z):     # 路面（1ROAD_split）と同じ高さ
            m.add_mesh(GROUP, name, v, f, uv, mat)
    hide_lines += [n for k in range(20) for n in (f"LINES_split_white_{k:02d}", f"LINES_split_white_{k:02d}_back")]
    pave = list(csv.DictReader(open(config.DATA_MANUAL / "s3_pavements.csv", encoding="utf-8")))
    diamonds = [d for d in dress_lines.diamond_spots(sec, pave) if s_range[0] + 200 <= d[0] <= s_range[1]
                and not any(g0 - 3 <= d[0] <= g1 + 3 for g0, g1 in split_gaps())]
    for name, v, f, uv, mat in dress_lines.marking_meshes(sec, diamonds, line_rows):
        m.add_mesh(GROUP, name, v, f, uv, mat)
    m.save()
    kn5_path = OUT / f"{GROUP}.kn5"
    kn5.write_from_manifest(OUT / "manifest.json", GROUP, kn5_path)
    print(f"部品：{len(m.meshes)} メッシュ、電柱 {npoles} 本 → {kn5_path} {kn5_path.stat().st_size / 1e6:.1f} MB")

    # 貼り替え
    assign = retex_assign(cl, rows, s_range)
    ext = OUT / "extension"
    ext.mkdir(parents=True, exist_ok=True)
    write_retex_textures(sorted(set(assign.values())), ext)
    (OUT / "retex.json").write_text(json.dumps(assign, ensure_ascii=False), encoding="utf-8")   # 確認描画用
    hide = old_fence_meshes(cl, s_range) + hide_lines
    (OUT / "hidden.json").write_text(json.dumps(hide), encoding="utf-8")
    part = dress_retex.section_text(assign) + "\n[MESH_ADJUSTMENT_...]\nACTIVE=1\nMESHES=" + ",".join(hide) + \
        "\nIS_RENDERABLE=0\n"
    (ext / "dress_s3.ini").write_text(part, encoding="utf-8", newline="\n")
    print(f"貼り替え：{len(assign)} 部品・{len(set(assign.values()))} 種類、隠す古い柵と線 {len(hide) // 2}")

    if a.no_install:
        return
    T = config.TRACK_DIR
    shutil.copy2(kn5_path, T / kn5_path.name)
    for f in ext.glob("dress_*.dds"):
        shutil.copy2(f, T / "extension" / f.name)
    ext_config.write_part(T, "dress_s3", part)
    ext_config.assemble(T)
    for lay in layouts.load_layouts():
        acdata.write_models_ini(T / f"models_{lay.name}.ini", layouts.kn5_files(lay.name))
    print("AC へ入れました：", T)


if __name__ == "__main__":
    main()
