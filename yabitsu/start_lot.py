"""菜の花台の駐車場（S1 のスタート側）。手で測った寸法（data/manual/s1_start_lot.json）から形を作る。

寸法は駐車場に沿った座標で書く: a=長手方向（ヘアピン側が+）、b=横方向（山側が+）。
"""
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon

from .crosssection import GUTTER_WIDTH
from .meshgen import pad_mesh

LINE_W, LINE_LIFT = 0.12, 0.01
RAIL_T = 0.05


class Frame:
    def __init__(self, origin, axis_deg):
        self.o = np.asarray(origin, float)
        self.heading = math.radians(axis_deg)
        self.u = np.array([math.cos(self.heading), math.sin(self.heading)])
        self.n = np.array([-math.sin(self.heading), math.cos(self.heading)])

    def xy(self, a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return self.o + a[..., None] * self.u + b[..., None] * self.n

    def ab(self, xy):
        d = np.asarray(xy, float) - self.o
        return d @ self.u, d @ self.n

    def rect(self, a, b):
        return Polygon(self.xy([a[0], a[1], a[1], a[0]], [b[0], b[0], b[1], b[1]]))


def load_spec(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def frame(spec):
    return Frame(spec["origin"], spec["axis_deg"])


def road_formation(sec, s_max=80.0):
    """道の始まり付近の、路面＋側溝が占める範囲。"""
    m = sec.s <= s_max
    hl, hr = sec.hw_left[m] + GUTTER_WIDTH, sec.hw_right[m] + GUTTER_WIDTH
    nx, ny = -np.sin(sec.heading[m]), np.cos(sec.heading[m])
    left = np.c_[sec.x[m] + hl * nx, sec.y[m] + hl * ny]
    right = np.c_[sec.x[m] - hr * nx, sec.y[m] - hr * ny]
    return Polygon(np.r_[left, right[::-1]]).buffer(0)


def lot_polygon(spec, sec):
    poly = frame(spec).rect(spec["lot"]["a"], spec["lot"]["b"]).difference(road_formation(sec))
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def apply(pads, spec, sec):
    """スタート側の駐車場（OSM の形）を、測った形に差し替える。高さは元の駐車場と同じ。"""
    f = frame(spec); out = []
    for p in pads:
        if p["role"] == "start":
            pit = spec["pits"]
            p = {**p, "kind": "lot", "polygon": np.asarray(lot_polygon(spec, sec).exterior.coords).tolist(),
                 "heading": f.heading, "pit_center": f.xy(pit["a"], pit["b"]).tolist(), "pit_spacing": pit["spacing"]}
        out.append(p)
    return out


def flat_pads(spec, pad):
    """建物と石碑のまわりの平らな地面。地形を削るときと、木をどけるときに使う（路面は張らない）。
    地形は「駐車場の高さ−0.3m」まで下げられるので、そのぶん持ち上げて地面が舗装とほぼ同じ高さになるようにする。"""
    f = frame(spec)
    polys = [f.rect(r["a"], r["b"]) for r in spec.get("flats", [])]
    if "tower" in spec:
        t = spec["tower"]
        polys.append(Polygon(_ring(f.xy(t["a"], t["b"]), t["plaza_r"], 16)))
    if "path" in spec:
        pa = spec["path"]
        polys.append(LineString(f.xy(*np.asarray(pa["points"], float).T)).buffer(pa["width"] / 2 + 1.0))
    return [{"name": "flat", "role": "flat", "polygon": np.asarray(p.exterior.coords).tolist(), "z": pad["z"] + 0.25}
            for p in polys]


def flats_for(spec_path, pads):
    """手で測った駐車場があれば、その平らな地面（flat_pads）。なければ空。"""
    spec = load_spec(spec_path)
    lot = [p for p in pads if p.get("kind") == "lot"]
    return flat_pads(spec, lot[0]) if spec and lot else []


def _merge(parts):
    vs, fs, us, base = [], [], [], 0
    for v, f, uv in parts:
        vs.append(v); fs.append(np.asarray(f) + base); us.append(uv); base += len(v)
    return np.concatenate(vs), np.concatenate(fs), np.concatenate(us)


def box(f, a, b, z0, z1, uv_scale=2.0):
    """底のない箱（側面4枚＋天面）。面は外向き。"""
    a0, a1 = a; b0, b1 = b
    faces = [
        ([a0, a1, a1, a0], [b0, b0, b0, b0], [z0, z0, z1, z1]),  # 谷側
        ([a1, a0, a0, a1], [b1, b1, b1, b1], [z0, z0, z1, z1]),  # 山側
        ([a1, a1, a1, a1], [b0, b1, b1, b0], [z0, z0, z1, z1]),  # +a 側
        ([a0, a0, a0, a0], [b1, b0, b0, b1], [z0, z0, z1, z1]),  # −a 側
        ([a0, a1, a1, a0], [b0, b0, b1, b1], [z1, z1, z1, z1]),  # 天面
    ]
    return _merge([_quad(f, *q, uv_scale) for q in faces])


def _quad(f, a, b, z, uv_scale):
    """4隅（a, b, z の各4要素）の四角形。表は頂点が反時計回りに見える側。"""
    v = np.c_[f.xy(a, b), z]
    w = np.linalg.norm(v[1] - v[0]); h = np.linalg.norm(v[3] - v[0])
    uv = np.array([[0, 0], [w, 0], [w, h], [0, h]]) / uv_scale
    return v, np.array([[0, 1, 2, 3]]), uv


def _strip(f, a0, a1, b, w, z):
    """長手方向に伸びる細い帯（白線）。"""
    return _quad(f, [a0, a1, a1, a0], [b - w / 2, b - w / 2, b + w / 2, b + w / 2], [z] * 4, 1.0)


def _crossline(f, a, b0, b1, w, z):
    return _quad(f, [a - w / 2, a + w / 2, a + w / 2, a - w / 2], [b0, b0, b1, b1], [z] * 4, 1.0)


def _gable_roof(f, a, b, z_eave, rise, eaves):
    a0, a1 = a[0] - eaves, a[1] + eaves
    b0, b1 = b[0] - eaves, b[1] + eaves; bm = (b[0] + b[1]) / 2
    zr = z_eave + rise
    s = z_eave - rise * eaves / ((b[1] - b[0]) / 2)  # 軒の出のぶん、少し下がる
    return _merge([_quad(f, [a0, a1, a1, a0], [b0, b0, bm, bm], [s, s, zr, zr], 2.0),
                   _quad(f, [a1, a0, a0, a1], [b1, b1, bm, bm], [s, s, zr, zr], 2.0)])


def _gable_ends(f, a, b, z_eave, rise):
    bm = (b[0] + b[1]) / 2; zr = z_eave + rise
    parts = []
    for aa, order in ((a[1], (b[0], b[1])), (a[0], (b[1], b[0]))):
        xy = f.xy([aa, aa, aa], [order[0], order[1], bm])
        v = np.c_[xy, [z_eave, z_eave, zr]]
        uv = np.array([[0, 0], [b[1] - b[0], 0], [(b[1] - b[0]) / 2, rise]]) / 2.0
        parts.append((v, np.array([[0, 1, 2]]), uv))
    return _merge(parts)


def _ring(c, r, n, phase=0.0):
    ang = phase + np.arange(n) * 2 * np.pi / n
    return np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]


def _wall_quad(p0, p1, z0, z1, uv_scale=1.0):
    """p0→p1 の上に立つ縦の四角形（p0→p1 の右側が表）。"""
    v = np.array([[*p0, z0], [*p1, z0], [*p1, z1], [*p0, z1]], float)
    w = np.linalg.norm(np.subtract(p1, p0))
    return v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [w, 0], [w, z1 - z0], [0, z1 - z0]]) / uv_scale


def _beam(p0, p1, half_w, normal_xy):
    """3D の2点を結ぶ平らな帯（縦の面の中にある斜めの筋かい）。両面材質で使う。"""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0; nrm = np.r_[normal_xy, 0.0]
    side = np.cross(nrm, d); side = side / np.linalg.norm(side) * half_w
    v = np.array([p0 - side, p1 - side, p1 + side, p0 + side])
    L = np.linalg.norm(d)
    return v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [L, 0], [L, 0.25], [0, 0.25]])


def _octagon_prism(c, r, z0, z1, uv_scale=1.0, phase=np.pi / 8):
    """外向きの八角柱の側面。"""
    ring = _ring(c, r, 8, phase)  # 反時計回りなので、進む向きの右側が外
    return _merge([_wall_quad(ring[k], ring[(k + 1) % 8], z0, z1, uv_scale) for k in range(8)])


def _fan(c, ring, z_c, z_ring, uv_scale=2.0):
    """中心と周りの点を結ぶ三角形の扇（上向き）。屋根なら z_c を高くする。"""
    n = len(ring)
    v = np.r_[[[*c, z_c]], np.c_[ring, np.full(n, z_ring)]]
    f = np.array([[0, 1 + k, 1 + (k + 1) % n] for k in range(n)])
    return v, f, v[:, :2] / uv_scale


def tower_meshes(spec, z):
    """菜の花台展望台：八角形の木の塔。柱8本、下の段はX字の筋かい、高さ deck_h に床と手すり、八角すいの屋根。"""
    f = frame(spec); t = spec["tower"]
    c = f.xy(t["a"], t["b"]); R = t["radius"]; hp = t["post"] / 2
    zd, zt = z + t["deck_h"], z + t["post_top"]
    ring = _ring(c, R, 8, np.pi / 8)
    frame_parts = []
    for k in range(8):
        p = ring[k]; ang = math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
        frame_parts.append(box(Frame(p, ang), (-hp, hp), (-hp, hp), z - 0.3, zt, 1.0))
        q = ring[(k + 1) % 8]
        out_n = (p + q) / 2 - c; out_n = out_n / np.linalg.norm(out_n)
        for za, zb in ((z + 0.3, zd), (zd, z + 0.3)):  # X字の筋かい
            frame_parts.append(_beam([*p, za], [*q, zb], 0.08, out_n))
        for zz in (z + 0.3, zd - 0.15, zt - 0.2):  # 横の梁
            frame_parts.append(_beam([*p, zz], [*q, zz], 0.1, out_n))
    frame_parts.append(_octagon_prism(c, 0.6, z - 0.3, zd, 1.0))  # らせん階段の代わりの柱
    out = [("s1_tower_frame", *_merge(frame_parts), "wood", True)]

    deck_ring = _ring(c, R + 0.3, 8, np.pi / 8)
    out.append(("s1_tower_deck", *_fan(c, deck_ring, zd, zd), "wood", True))
    rail = _octagon_prism(c, R + 0.3, zd, zd + t["rail_h"], 0.5)
    rv, rf, ruv = rail; ruv = ruv.copy(); ruv[:, 1] = np.where(rv[:, 2] > zd + 0.01, 1.0, 0.0)
    out.append(("s1_tower_rail", rv, rf, ruv, "wood_bars", True))
    out.append(("s1_tower_roof", *_fan(c, _ring(c, R + t["eaves"], 8, np.pi / 8), zt + t["roof_h"], zt), "roof_metal", True))
    out.append(("1WALL_s1_tower", *_octagon_prism(c, R + hp, z - 0.3, z + 3.0), "wood", False))
    return out


def path_mesh(spec, z):
    """トイレの横から展望台までの歩道（タイル）。折れ線の両側に幅の半分ずつ広げる。"""
    f = frame(spec); pa = spec["path"]
    line = LineString(f.xy(*np.asarray(pa["points"], float).T))
    poly = line.buffer(pa["width"] / 2, cap_style="flat", join_style="mitre")
    v, faces, _ = pad_mesh(np.asarray(poly.exterior.coords), z)
    a, b = f.ab(v[:, :2])
    return v, faces, np.c_[a, b] / 2.0


def meshes(spec, pad):
    """(名前, 頂点, 面, UV, 材質, 描画するか) の並び。"""
    f = frame(spec); z = pad["z"]
    out = []

    v, faces, _ = pad_mesh(pad["polygon"], z)
    a, b = f.ab(v[:, :2])
    out.append(("1ROAD_s1_lot", v, faces, np.c_[a, b] / 4.0, "pavers", True))

    st = spec["stalls"]
    zl = z + LINE_LIFT
    lines = [_crossline(f, a_, st["b"][0], st["b"][1], LINE_W, zl)
             for a_ in np.arange(st["a"][0], st["a"][1] + 1e-9, st["pitch"])]
    lines.append(_strip(f, st["a"][0], st["a"][1], st["b"][1], LINE_W, zl))
    out.append(("LINES_s1_lot_stalls", *_merge(lines), "line_white", True))

    fe = spec["fence"]; fb, fh = fe["b"], fe["height"]
    posts = [box(f, (a_ - 0.04, a_ + 0.04), (fb - 0.04, fb + 0.04), z - 0.3, z + fh)
             for a_ in np.arange(fe["a"][0], fe["a"][1] + 1e-9, fe["post_every"])]
    rails = [box(f, fe["a"], (fb - RAIL_T, fb + RAIL_T), z + zz, z + zz + 0.06) for zz in (0.08, fh - 0.06)]
    out.append(("s1_lot_fence", *_merge(posts + rails), "railing", True))
    v, faces, uv = _quad(f, [fe["a"][0], fe["a"][1], fe["a"][1], fe["a"][0]], [fb] * 4,
                         [z + 0.14] * 2 + [z + fh - 0.06] * 2, 0.5)
    uv[:, 1] = [0, 0, 1, 1]  # 縦は1枚、横は 0.5m ごとにくり返す
    out.append(("s1_lot_fence_bars", v, faces, uv, "railing_bars", True))
    out.append(("1WALL_s1_lot_fence", *_quad(f, [fe["a"][0], fe["a"][1], fe["a"][1], fe["a"][0]], [fb] * 4,
                                              [z - 0.3] * 2 + [z + 1.2] * 2, 2.0), "railing", False))

    bl = spec["blocks"]; L, W, H = bl["size"]
    blocks = [box(f, (a_ - L / 2, a_ + L / 2), (bl["b"] - W / 2, bl["b"] + W / 2), z - 0.2, z + H, 1.0)
              for a_ in np.arange(bl["a"][0], bl["a"][1] + 1e-9, bl["every"])]
    out.append(("1WALL_s1_lot_blocks", *_merge(blocks), "concrete", True))

    mo = spec["monument"]; L, W, H = mo["size"]
    out.append(("1WALL_s1_monument", *box(f, (mo["a"] - L / 2, mo["a"] + L / 2), (mo["b"] - W / 2, mo["b"] + W / 2),
                                          z - 0.3, z + H, 1.0), "stone", True))

    bu = spec["building"]
    walls = box(f, bu["a"], bu["b"], z - 0.3, z + bu["wall_h"], 2.0)
    out.append(("1WALL_s1_building", *walls, "wall_white", True))
    out.append(("s1_building_gables", *_gable_ends(f, bu["a"], bu["b"], z + bu["wall_h"], bu["roof_h"]), "wall_white", True))
    out.append(("s1_building_roof", *_gable_roof(f, bu["a"], bu["b"], z + bu["wall_h"], bu["roof_h"], bu["eaves"]),
                "roof", True))

    if "tower" in spec:
        out += tower_meshes(spec, z)
    if "path" in spec:
        out.append(("1ROAD_s1_path", *path_mesh(spec, z), "tiles", True))
    if "bollards" in spec:
        bo = spec["bollards"]
        posts = [box(Frame(f.xy(bo["a"], b_), spec["axis_deg"]), (-0.075, 0.075), (-0.075, 0.075), z - 0.3, z + bo["height"], 0.5)
                 for b_ in bo["b"]]
        out.append(("1WALL_s1_bollards", *_merge(posts), "wood", True))
    return out
