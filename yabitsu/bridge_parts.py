"""名前のある橋（S5）の作り込み：両端の親柱と銘板（橋の名前・ひらがな）、路面の下の桁、谷の地面まで立つ橋脚。
欄干・手すりは features の fence（橋の範囲）で作る。橋の下の谷は地形を盛らない（no_fill_mask → terrain.carve）。

data/manual/s5_bridges.json：[{"osm_id", "name", "kana", "rail", "rail_color", "deck_depth_m", "pier_every_m"}]"""
import json
import math

import numpy as np

from .crosssection import GUTTER_WIDTH

POST_W, POST_H = 0.5, 1.3     # 親柱（四角い柱）
RAIL_H = 1.1
ARCH_CLEAR, ARCH_EVERY = 6.5, 12.0   # 赤い鉄のアーチ（釜田川橋）：路面からの高さ、門形の間隔
PIER_W = 1.6


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def ranges(sec, elements, ids):
    """osm_id → 本線の (s0, s1)。"""
    from . import geo
    out = {}
    for w in elements:
        if w.get("type") != "way" or int(w.get("id", 0)) not in ids or len(w.get("geometry", [])) < 2:
            continue
        c = np.array([geo.to_local(g["lat"], g["lon"]) for g in w["geometry"]], float)
        ss = [float(sec.s[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))]) for p in (c[0], c[-1])]
        out[int(w["id"])] = (min(ss), max(ss))
    return out


def no_fill_mask(sec, s_from):
    """s_from より先の橋の行（前後 3m）：terrain.carve の no_fill_rows（谷を盛らない）。"""
    b = np.asarray(sec.bridge, bool) & (sec.s >= s_from)
    if b.any():
        from scipy.ndimage import binary_dilation
        b = binary_dilation(b, iterations=3)
    return b


def _box(c, d, n, w, dep, z0, z1):
    p = [c - n * w / 2 - d * dep / 2, c + n * w / 2 - d * dep / 2, c + n * w / 2 + d * dep / 2, c - n * w / 2 + d * dep / 2]
    v = np.array([[*q, z0] for q in p] + [[*q, z1] for q in p], float)
    f = np.array([[4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], np.int32)
    return v, f, v[:, :2]


def _row(sec, s):
    i = int(np.argmin(np.abs(sec.s - s))); h = sec.heading[i]
    return i, np.array([sec.x[i], sec.y[i]]), np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])


def meshes(sec, bridges, rng, ground_z):
    """(名前, 頂点, 面, UV, 材質, 描画するか)。rng = {osm_id: (s0, s1)}。"""
    out = []
    for b in bridges:
        bid = int(b["osm_id"])
        if bid not in rng:
            continue
        s0, s1 = rng[bid]
        # 親柱と銘板（両端・両側）
        for end, s in (("a", s0), ("b", s1)):
            i, c, d, n = _row(sec, s)
            for sd in (1, -1):
                hw = (sec.hw_left[i] if sd > 0 else sec.hw_right[i]) + GUTTER_WIDTH + POST_W / 2
                p = c + n * sd * hw; z = float(sec.edge_z(sd)[i])
                out.append((f"bridge_{bid}_post_{end}{'L' if sd > 0 else 'R'}", *_box(p, d, n, POST_W, POST_W, z - 0.2, z + POST_H), "stone", True))
                q = p - n * sd * (POST_W / 2 + 0.01)   # 道の側の面に銘板
                pv = np.array([[*(q - d * 0.2), z + 0.5], [*(q + d * 0.2), z + 0.5], [*(q + d * 0.2), z + 1.1], [*(q - d * 0.2), z + 1.1]])
                pf = np.array([[0, 1, 2, 3]] if sd < 0 else [[3, 2, 1, 0]], np.int32)
                out.append((f"bridge_{bid}_plate_{end}{'L' if sd > 0 else 'R'}", pv, np.r_[pf, pf[:, ::-1]],
                            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), f"bridge_plate_{bid}_{'name' if end == 'a' else 'kana'}", True))
        # 欄干：concrete＝コンクリートの壁、steel・wood＝色つきの格子の柵＋見えない壁
        rail = b.get("rail", "steel")
        mat = "concrete" if rail == "concrete" else f"bridge_rail_{str(b.get('rail_color', '#8A8A86')).lstrip('#').upper()}"
        rows_r = [int(np.argmin(np.abs(sec.s - s))) for s in np.r_[np.arange(s0, s1, 2.0), s1]]
        for sd, tag in ((1, "L"), (-1, "R")):
            V, F = [], []
            for k, i in enumerate(rows_r):
                n = np.array([-math.sin(sec.heading[i]), math.cos(sec.heading[i])]); c = np.array([sec.x[i], sec.y[i]])
                hw = (sec.hw_left[i] if sd > 0 else sec.hw_right[i]) + GUTTER_WIDTH; z = float(sec.edge_z(sd)[i])
                p = c + n * sd * hw
                V += [[*p, z - 0.1], [*p, z + RAIL_H]]
                if k:
                    a = 2 * (k - 1); F.append([a, a + 2, a + 3, a + 1])
            v = np.array(V, float); f = np.array(F, np.int32)
            uv = np.c_[np.repeat(np.arange(len(rows_r)) * 2.0 / 2.0, 2), np.tile([0.0, 1.0], len(rows_r))]
            if rail == "concrete":
                out.append((f"bridge_{bid}_rail_{tag}", v, np.r_[f, f[:, ::-1]], uv, "concrete", True))
                out.append((f"1WALL_bridge_{bid}_rail_{tag}", v, np.r_[f, f[:, ::-1]], uv, "concrete", False))
            else:
                out.append((f"bridge_{bid}_rail_{tag}", v, np.r_[f, f[:, ::-1]], uv, mat, True))
                out.append((f"1WALL_bridge_{bid}_rail_{tag}", v, np.r_[f, f[:, ::-1]], uv, "concrete", False))
        if b.get("arch"):   # 道をまたぐ赤い鉄の門形（両側の柱と上の梁）と、上の弦
            parts = []
            for s in np.unique(np.r_[np.arange(s0 + 3.0, s1 - 2.0, ARCH_EVERY), (s0 + s1) / 2, s1 - 3.0]):   # まん中（一番高い所）を含める
                i, c, d, n = _row(sec, s); z = float(sec.z[i])
                l = sec.hw_left[i] + GUTTER_WIDTH + 0.4; r = sec.hw_right[i] + GUTTER_WIDTH + 0.4
                rise = ARCH_CLEAR * (1 - (2 * (s - s0) / max(s1 - s0, 1e-9) - 1) ** 2) * 0.35 + ARCH_CLEAR * 0.65
                for off in (l, -r):
                    parts.append(_box(c + n * off, d, n, 0.5, 0.5, z, z + rise))
                parts.append(_box(c + n * (l - r) / 2, d, n, l + r + 0.5, 0.5, z + rise - 0.6, z + rise))
            from .start_lot import _merge
            v, f, uv = _merge(parts)
            amat = f"bridge_rail_{str(b.get('arch_color', b.get('rail_color', '#C0302A'))).lstrip('#').upper()}"
            out.append((f"bridge_{bid}_arch", v, f, uv, amat, True))
        # 桁（路面の下、deck_depth_m）
        dep = float(b.get("deck_depth_m", 1.2))
        rows = [int(np.argmin(np.abs(sec.s - s))) for s in np.r_[np.arange(s0, s1, 2.0), s1]]
        V, F = [], []
        for k, i in enumerate(rows):
            n = np.array([-math.sin(sec.heading[i]), math.cos(sec.heading[i])]); c = np.array([sec.x[i], sec.y[i]])
            l = sec.hw_left[i] + GUTTER_WIDTH; r = sec.hw_right[i] + GUTTER_WIDTH; z = float(sec.z[i])
            V += [[*(c - n * r), z - 0.1], [*(c - n * r), z - dep], [*(c + n * l), z - dep], [*(c + n * l), z - 0.1]]
            if k:
                a = 4 * (k - 1)
                F += [[a, a + 4, a + 5, a + 1], [a + 1, a + 5, a + 6, a + 2], [a + 2, a + 6, a + 7, a + 3]]
        v = np.array(V, float)
        out.append((f"bridge_{bid}_girder", v, np.array(F, np.int32), v[:, :2] / 3.0, "concrete", True))
        # 橋脚（pier_every_m おき、両端の台は除く）
        every = float(b.get("pier_every_m", 0) or 0)
        if every > 0:
            for k, s in enumerate(np.arange(s0 + every, s1 - every / 3, every)):
                i, c, d, n = _row(sec, s)
                zg = float(np.atleast_1d(ground_z(np.array([c[0]]), np.array([c[1]])))[0])
                w = sec.hw_left[i] + sec.hw_right[i]
                out.append((f"1WALL_bridge_{bid}_pier_{k:02d}", *_box(c, d, n, w * 0.6, PIER_W, zg - 1.0, float(sec.z[i]) - dep), "concrete", True))
    return out


def write_plate_textures(bridges, out_dir):
    """銘板の絵：橋の名前（漢字）と、ひらがな。"""
    from .textures import sign_texture
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for b in bridges:
        for tag, txt in (("name", b.get("name", "")), ("kana", b.get("kana", "") or b.get("name", ""))):
            if not txt:
                continue
            name = f"bridge_plate_{int(b['osm_id'])}_{tag}"
            sign_texture("\n".join(txt) if tag == "name" and len(txt) <= 6 else txt, "#3B3A36", "#E8E2D0", 96, 160, border="#2A2926").save(out_dir / f"{name}.png")
            paths[name] = f"textures/{name}.png"
    return paths


def rail_textures(bridges, out_dir):
    """色つきの格子の柵（bridge_rail_<色>）の絵。"""
    from PIL import Image
    from .textures import railing_bars
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for b in bridges:
        if b.get("rail", "steel") == "concrete" and not b.get("arch"):
            continue
        cols = [(str(b.get("rail_color", "#8A8A86")), False)] + ([(str(b.get("arch_color", "#C0302A")), True)] if b.get("arch") else [])
        for col, solid in cols:
            hexc = col.lstrip("#").upper(); name = f"bridge_rail_{hexc}"
            if name in paths:
                continue
            rgb = tuple(int(hexc[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
            a = railing_bars(rgb=rgb, bars=6, bar_px=10); a[:22, :, 3] = 255; a[-22:, :, 3] = 255   # 上と下の横の棒
            if solid:   # アーチの鉄骨は塗りつぶし
                a[:, :, 3] = 255
            Image.fromarray(a).save(out_dir / f"{name}.png"); paths[name] = f"textures/{name}.png"
    return paths
