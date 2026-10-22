"""今ある S3 の路面・白線・歩道・側溝・壁に、CSP（SHADER_REPLACEMENT）で新しいテクスチャを貼る設定を作る。
kn5 は作り直さない。UV はそのまま使うので、部品の種類ごとに UV の寸法（1 が何 m か）へ合わせた画像を作る。"""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

# 種類 → (UV の横 1 が何 m, 縦 1 が何 m, 元の画像を 90 度回すか)。2026-10-19 に manifest の UV を測った値
UV_LAYOUT = {
    "road": (4.0, 4.0, False),        # 1ROAD_s1_*：横＝道の幅方向、縦＝長さ（2026-10-20 に測り直した。縦は 4 m）
    "plane": (4.0, 4.0, False),       # 町の車道・分かれ道：真上から x/4, y/4
    "gutter": (1.0, 4.0, False),      # 側溝：横＝断面に沿って 1 m（垂らした縁を含む）、縦＝長さ 4 m（2026-10-20 に測り直した）
    "line": (1.0, 4.0, False),        # 白線・中央線：横 m、縦 1 = 4 m
    "sidewalk": (1.0, 4.0, False),
    "wall": (2.0, 4.0, True),         # 壁・橋の手すり壁：横＝高さ（1 = 2 m）、縦＝長さ（1 = 4 m）
    "terrain": (8.0, 8.0, False),     # 地形（1GRASS_*）：真上から x/8, y/8
}


def _meshes(manifest_dir):
    manifest_dir = Path(manifest_dir)
    m = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, e in m["meshes"].items():
        yield name, e["material"], np.load(manifest_dir / e["file"])["verts"]


def kind_of(name: str, material: str):
    n = name
    if material in ("ground", "tree"):
        return None
    if n.startswith("1ROAD_s1_gutter"):
        return "gutter"
    if n.startswith("1ROAD_track_sidewalk"):
        return "sidewalk"
    if n.startswith("track_sidewalk") and n.endswith("_curb"):
        return "curb"
    if n.startswith("1ROAD_split_red"):      # 下り側の道の赤い舗装は、ストリートビュー 2025（s=3400・3450 下り）では灰色。
        return "road_plane"                  # 赤いのは南の端だけ（s3_lines.csv の red 行で描く）。空中写真は古い
    if n.startswith(("1ROAD_town_carriageway", "1ROAD_side_stub", "1ROAD_split")) and material == "asphalt":
        return "road_plane"
    if n.startswith("1ROAD_s1_") and material == "asphalt":
        return "road"
    if n.startswith("LINES_") and material in ("line_white", "line_orange"):
        return material
    if n.startswith("1WALL_s1_retaining"):
        return "wall"
    if n.startswith("1WALL_track_parapet"):
        return "parapet"
    return None


def s3_meshes(manifest_dir, cl, s_range, near_m=60.0) -> dict:
    """S3 の範囲（中心線の s）にある、貼り替える部品の {名前: 種類}。"""
    P = np.c_[cl["x"], cl["y"]]
    tree = cKDTree(P)
    out = {}
    for name, mat, v in _meshes(manifest_dir):
        k = kind_of(name, mat)
        if k is None:
            continue
        d, i = tree.query(v[:, :2].mean(0))
        if d <= near_m and s_range[0] <= cl["s"][i] <= s_range[1]:
            out[name] = k
    return out


def _finish(note: str) -> str:
    if "ブロック" in note:
        return "block"
    if "コンクリ" in note:
        return "concrete"
    return "kenchi"        # 間知石・石積み・石垣（S3 の壁はほとんどこれ）


def wall_finish(manifest_dir, cl, features_csv, names, s_offset=0.0) -> dict:
    """壁の部品の中心の s と左右から、表（*_features.csv）の行を引いて仕上げを決める。"""
    rows = [r for r in csv.DictReader(open(features_csv, encoding="utf-8"))
            if r["type"] in ("retaining_wall", "parapet")]
    P = np.c_[cl["x"], cl["y"]]
    tree = cKDTree(P)
    names = set(names)
    out = {}
    for name, mat, v in _meshes(manifest_dir):
        if name not in names:
            continue
        c = v[:, :2].mean(0)
        _, i = tree.query(c)
        j = min(i + 1, len(P) - 1); i0 = max(j - 1, 0)
        h = P[j] - P[i0]
        side = "L" if h[0] * (c[1] - P[i][1]) - h[1] * (c[0] - P[i][0]) > 0 else "R"
        s = cl["s"][i] - s_offset            # 表の s は区間の起点から
        hit = [r for r in rows if r["side"] == side and float(r["s_from"]) - 10 <= s <= float(r["s_to"]) + 10]
        out[name] = _finish(hit[0]["note"]) if hit else "kenchi"
    return out


def enlarge(img: Image.Image, k: float, band=0.125) -> Image.Image:
    """継ぎ目のない画像の模様を k 倍に（大きさは同じ。目地の数が 1/k）。k 倍に広げて左上の元の大きさを切り出し、
    端の帯（幅 band）を、広げた画像で切り出しの手前にある部分と混ぜて、反対の端とつながるようにする。
    目地がそろうのは、元の目地の数 c について c/k と (k-1)·c/k が整数のとき（間知ブロック 4→2 は k=2、間知石 5→2 は k=2.5）。"""
    n_w, n_h = img.size
    W, H = int(round(n_w * k)), int(round(n_h * k))
    p = 8   # 拡大の前に周りへ折り返して貼る（端どうしもなめらかにつながる拡大にする）
    pad = np.pad(np.asarray(img), ((p, p), (p, p)) + ((0, 0),) * (np.asarray(img).ndim - 2), mode="wrap")
    q = int(round(p * k))
    big = np.asarray(Image.fromarray(pad).resize((W + 2 * q, H + 2 * q), Image.LANCZOS), float)[q:q + H, q:q + W]
    bw, bh = int(n_w * band), int(n_h * band)
    tx = np.linspace(0, 1, bw)[None, :, None]
    ty = np.linspace(0, 1, bh)[:, None, None]

    def right(rows):     # 右端の帯を、広げた画像の右端（＝左端のすぐ手前）へ寄せる
        a = big[rows, :n_w].copy()
        a[:, n_w - bw:] = a[:, n_w - bw:] * (1 - tx) + big[rows, W - bw:] * tx
        return a

    out = right(slice(0, n_h))
    out[n_h - bh:] = out[n_h - bh:] * (1 - ty) + right(slice(H - bh, H)) * ty     # 下端の帯も同じく（上端のすぐ手前へ）
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), img.mode)


def compose(base: Image.Image, base_m, u_m, v_m, rotate=False, max_px=2048) -> Image.Image:
    """元の画像（実寸 base_m＝横×縦 m）を、UV の 1 が u_m×v_m m の範囲になる画像にする（並べる・回す）。"""
    img, bw, bh = base, base_m[0], base_m[1]
    if rotate:     # 元の横（長さ方向）を縦へ
        img, bw, bh = img.rotate(90, expand=True), bh, bw
    ppm = img.width / bw
    W, H = int(round(u_m * ppm)), int(round(v_m * ppm))
    canvas = Image.new(img.mode, (W, H))
    for y in range(0, H, img.height):
        for x in range(0, W, img.width):
            canvas.paste(img, (x, y))
    s = min(1.0, max_px / max(W, H))
    p2 = lambda n: 1 << max(6, int(np.floor(np.log2(max(1, n * s)))))
    return canvas.resize((p2(W), p2(H)), Image.LANCZOS)


def section_text(assign: dict) -> str:
    """{部品名: テクスチャのキー} → テクスチャごとの [SHADER_REPLACEMENT_...]。表示だけを変える（当たり判定は触らない）。"""
    by = {}
    for name, key in sorted(assign.items()):
        by.setdefault(key, []).append(name)
    out = []
    for key, names in sorted(by.items()):
        meshes = ",".join(x for n in names for x in (n, f"{n}_back"))
        out += ["[SHADER_REPLACEMENT_...]", "ACTIVE=1", f"MESHES={meshes}", "SHADER=ksPerPixelNM", "FILL_MISSING_TEXTURES=1",
                "RESOURCE_0=txDiffuse", f"RESOURCE_FILE_0=dress_{key}.dds",
                "RESOURCE_1=txNormal", f"RESOURCE_FILE_1=dress_{key}_nm.dds", ""]
    return "\n".join(out)
