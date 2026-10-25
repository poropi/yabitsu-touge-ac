"""森の木のモデル（杉・檜・広葉樹・低木・シダ）を作り、CSP の木パッケージとして書き出す。

葉は foliage_textures が写真素材から合成した「枝1本分」「葉の茂み1つ分」の画像を板に貼る。
杉・檜：まっすぐな幹＋段になった枝（下の方は枝打ちで枝がない）。広葉樹：Sapling の枝分かれ原型
（assets/forest/broadleaf_prototypes_v2.json）に葉の茂みを付ける。
LOD は3段階（近い：板を重ねて立体的に／遠い：板を減らして大きく）。座標は CSP の決まり（y が上、
高さ 0〜1・横 ±1）に正規化し、実寸は tree.ini の HEIGHT/WIDTH に書く。
"""
import io
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import csptree, kn5
from . import foliage_textures as ft
from .config import PROJECT_ROOT

PROTOTYPES = PROJECT_ROOT / "assets" / "forest" / "broadleaf_prototypes_v2.json"
UP = np.array([0.0, 1.0, 0.0])

SPECIES = {
    "sugi": ["sugi_a", "sugi_b", "sugi_c"],
    "hinoki": ["hinoki_a", "hinoki_b"],
    "broadleaf": ["broadleaf_a", "broadleaf_b", "broadleaf_c", "broadleaf_d"],
    "sapling": ["sapling_a", "sapling_b"],
    "shrub": ["shrub_a", "shrub_b"],
    "fern": ["fern_a", "fern_b"],
}

# height：高さ m、crown_base：葉が始まる高さ（全高に対する割合）、radius：樹冠の根元の半径 m、
# rise：枝の角度（下段→上段、度）
CONIFERS = {
    "sugi_a": dict(height=22.0, crown_base=0.50, radius=2.4, rise=(-8, 42), seed=11),
    "sugi_b": dict(height=17.0, crown_base=0.45, radius=2.1, rise=(-12, 38), seed=12),
    "sugi_c": dict(height=26.0, crown_base=0.56, radius=2.7, rise=(-5, 45), seed=13),
    "hinoki_a": dict(height=16.0, crown_base=0.38, radius=2.6, rise=(-22, 25), seed=21),
    "hinoki_b": dict(height=20.0, crown_base=0.42, radius=2.9, rise=(-18, 28), seed=22),
}
BROADLEAVES = {"broadleaf_a": (0, 14.0, 31), "broadleaf_b": (1, 12.0, 32), "broadleaf_c": (0, 15.5, 33), "broadleaf_d": (3, 11.0, 34),
               "sapling_a": (0, 5.5, 61), "sapling_b": (3, 4.8, 62)}   # sapling：植えたばかりの若木（細く、葉がまばら）
SAPLING = dict(spread=0.45, leaves=0.13, card=0.45, reach=0.12, low=0.42)   # 広葉樹に比べた横の広がり・葉の数・葉の板の大きさ・樹冠の厚み、葉の付き始め（全高の割合）
SHRUBS = {"shrub_a": (1.4, 41), "shrub_b": (2.3, 42)}
FERNS = {"fern_a": (0.75, 51), "fern_b": (1.0, 52)}

TINTS = {"sugi": [(0.50, 0.62, 0.47), (0.56, 0.66, 0.50)], "hinoki": [(0.70, 0.84, 0.58), (0.64, 0.80, 0.55)],
         "broadleaf": [(0.88, 1.0, 0.80), (0.78, 0.94, 0.70)], "shrub": [(0.86, 0.98, 0.76), (0.80, 0.92, 0.70)],
         "sapling": [(1.02, 1.10, 0.66), (0.95, 1.06, 0.62)]}   # 日の当たる若葉の明るい黄緑
LOD_DISTANCE = (40, 110, 230)
DRAW_DISTANCE = {"sugi": 4000, "hinoki": 4000, "broadleaf": 4000, "sapling": 1500, "shrub": 300, "fern": 120}


def sources_available() -> bool:
    return ft.sources_available() and PROTOTYPES.exists()


@dataclass
class TreeModel:
    lods: list
    height: float
    width: float
    crown_base: float      # 葉が始まる高さ（全高に対する割合）
    crown_radius: float    # m
    textures: dict = field(default_factory=dict)
    ini: dict = field(default_factory=dict)


class Part:
    """1つの材質（葉か樹皮）の頂点と三角形。座標は実寸 m。"""

    def __init__(self):
        self.p, self.n, self.uv, self.t, self.ao, self.f = [], [], [], [], [], []

    def add(self, p, n, uv, t, ao, tris):
        base = sum(len(a) for a in self.p)
        self.p.append(np.asarray(p, float)); self.n.append(np.asarray(n, float)); self.uv.append(np.asarray(uv, float))
        self.t.append(np.asarray(t, float)); self.ao.append(np.asarray(ao, float))
        self.f.append(np.asarray(tris, np.int64) + base)

    def arrays(self):
        if not self.p:
            return None
        return [np.concatenate(a) for a in (self.p, self.n, self.uv, self.t, self.ao, self.f)]


def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _rot(v, axis, ang):
    """ロドリゲスの回転。"""
    axis = _unit(axis)
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * np.dot(axis, v) * (1 - np.cos(ang))


def card(part, root, u, v, length, width, rect, bend=0.0, segs=2, along="x", center=None, ao=(0.5, 1.0)):
    """root から u の向きへ伸びる板（segs 段に折れ、先へ行くほど下へ bend ラジアン垂れる）。
    rect=(u0, v0, u1, v1) は atlas 上の範囲。along="x" なら枝の向きが画像の横、"y" なら画像の下→上。"""
    u, v = _unit(u), _unit(v)
    pts = [np.asarray(root, float)]
    d = u
    for i in range(segs):
        a, b = _unit(_rot(d, v, bend / segs)), _unit(_rot(d, v, -bend / segs))
        d = a if a[1] < b[1] else b          # 下へ垂れる向きに曲げる
        pts.append(pts[-1] + d * length / segs)
    P, UV, T = [], [], []
    u0, v0, u1, v1 = rect
    for i, c in enumerate(pts):
        s = i / segs
        for side in (-0.5, 0.5):
            P.append(c + v * width * side)
            if along == "x":
                UV.append((u0 + (u1 - u0) * s, v0 + (v1 - v0) * (side + 0.5)))
            else:
                UV.append((u0 + (u1 - u0) * (side + 0.5), v1 - (v1 - v0) * s))
            T.append(u)
    P = np.array(P)
    nrm = _unit(np.cross(u, v))
    cen = np.array(center if center is not None else root, float)
    sph = _unit(P - cen)
    if np.dot(nrm, sph.mean(0)) < 0:
        nrm = -nrm
    N = _unit(0.35 * nrm + 0.65 * sph)
    A = np.full(len(P), ao[0]) + (ao[1] - ao[0]) * np.repeat(np.linspace(0, 1, segs + 1), 2)
    tris = []
    for i in range(segs):
        a = 2 * i
        tris += [(a, a + 2, a + 1), (a + 1, a + 2, a + 3)]
    part.add(P, N, UV, T, A, tris)


def tube(part, pts, radii, sides, uv_v=1.0, ao=0.7):
    """点列に沿った樹皮の筒。UV の u は周、v は長さ（m）×uv_v。"""
    pts = np.asarray(pts, float)
    P, N, UV, T = [], [], [], []
    length = 0.0
    prev = None
    for i, (c, r) in enumerate(zip(pts, radii)):
        d = _unit(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        if prev is None:
            ref = UP if abs(d[1]) < 0.9 else np.array([1.0, 0, 0])
            a = _unit(np.cross(d, ref))
        else:
            a = _unit(prev - d * np.dot(prev, d))
        b = np.cross(d, a)
        prev = a
        if i:
            length += np.linalg.norm(c - pts[i - 1])
        for j in range(sides + 1):
            ang = 2 * np.pi * j / sides
            n = a * np.cos(ang) + b * np.sin(ang)
            P.append(c + r * n); N.append(n); UV.append((j / sides, length * uv_v)); T.append(d)
    tris = []
    w = sides + 1
    for i in range(len(pts) - 1):
        for j in range(sides):
            a0, a1, b0, b1 = i * w + j, i * w + j + 1, (i + 1) * w + j, (i + 1) * w + j + 1
            tris += [(a0, b0, a1), (a1, b0, b1)]
    part.add(P, N, UV, T, np.full(len(P), ao), tris)


# ---------- 杉・檜 ----------

def _conifer(name, lod, species):
    c = CONIFERS[name]
    rng = np.random.default_rng(c["seed"] * 10 + lod)
    H, cb, R = c["height"], c["crown_base"] * c["height"], c["radius"]
    spacing, per, cross, segs, wmul, nfill = [(0.30, 6, True, 3, 1.0, 170), (0.62, 5, False, 2, 1.3, 40),
                                              (0.8, 5, False, 1, 2.0, 30)][lod]
    leaf, wood = Part(), Part()
    sink = 0.015 * H
    ys = np.linspace(-sink, H * 0.97, [26, 10, 4][lod])
    r0 = H * 0.009
    tube(wood, [(0, y, 0) for y in ys], [r0 * (1 - max(y, 0) / H) ** 0.8 + 0.02 for y in ys], [10, 6, 4][lod],
         uv_v=0.35, ao=0.75)
    rects = [(0, 0, 1, 0.5), (0, 0.5, 1, 1)]
    center = np.array([0.0, (cb + H) / 2, 0.0])
    y, k = cb, 0
    while y < H * 0.95:
        h = (y - cb) / (H - cb)
        az0 = rng.uniform(0, 2 * np.pi)
        for j in range(per):
            az = az0 + 2 * np.pi * j / per + rng.normal(0, 0.25)
            el = np.radians(np.interp(h, (0, 1), c["rise"]) + rng.normal(0, 7))
            u = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
            L = R * (1 - h) ** 0.9 * rng.uniform(0.8, 1.15) + 0.35
            v = _rot(_unit(np.cross(UP, u)), u, rng.uniform(-0.6, 0.6))
            root = np.array([0.0, y, 0.0]) + u * r0 * (1 - y / H)
            rect = rects[rng.integers(2)]
            aos = (0.45 + 0.2 * h, 1.0)
            card(leaf, root, u, v, L, L * 0.5 * wmul, rect, bend=np.radians(rng.uniform(8, 22)), segs=segs,
                 center=center, ao=aos)
            if cross:
                card(leaf, root, u, _rot(v, u, np.pi / 2), L * 0.9, L * 0.4, rect, bend=np.radians(12), segs=segs,
                     center=center, ao=aos)
            if lod == 0:
                tube(wood, [root, root + u * L * 0.55], [0.025, 0.006], 3, uv_v=1.0, ao=0.35)
            k += 1
        y += spacing * rng.uniform(0.8, 1.2)
    for _ in range(nfill):   # 枝の間を埋める小枝（内側ほど暗い）
        yy = rng.uniform(cb + 0.5, H * 0.92)
        h = (yy - cb) / (H - cb)
        rr = R * (1 - h) * rng.uniform(0.2, 0.7)
        az = rng.uniform(0, 2 * np.pi)
        root = np.array([np.cos(az) * rr, yy, np.sin(az) * rr])
        u = _unit(np.array([np.cos(az), rng.uniform(-0.3, 0.6), np.sin(az)]))
        card(leaf, root, u, _unit(np.cross(UP, u)), rng.uniform(0.8, 1.4), 0.6, rects[rng.integers(2)],
             bend=0.2, segs=2, center=center, ao=(0.4, 0.75))
    top = np.array([0.0, H * 0.9, 0.0])     # 尖った梢
    for j in range(3 if lod < 2 else 2):
        az = j * 2 * np.pi / 3
        u = _unit(np.array([0.15 * np.cos(az), 1.0, 0.15 * np.sin(az)]))
        card(leaf, top, u, np.array([np.cos(az + 1.57), 0, np.sin(az + 1.57)]), H * 0.1, 0.9, rects[0],
             segs=1, center=center, ao=(0.9, 1.0))
    return leaf, wood, cb / H


# ---------- 広葉樹 ----------

@lru_cache(maxsize=None)
def _prototypes():
    return json.loads(PROTOTYPES.read_text(encoding="utf-8"))["trees"]


def _bezier(a, b, t):
    a_co, a_r, b_l, b_co = (np.array(a["co"]), np.array(a["right"]), np.array(b["left"]), np.array(b["co"]))
    return a_co * (1 - t) ** 3 + a_r * 3 * (1 - t) ** 2 * t + b_l * 3 * (1 - t) * t * t + b_co * t ** 3


def _branch_points(br, steps):
    pts, rad = [], []
    for a, b in zip(br[:-1], br[1:]):
        for t in np.linspace(0, 1, steps, endpoint=False):
            pts.append(_bezier(a, b, t)); rad.append(a["radius"] * (1 - t) + b["radius"] * t)
    pts.append(np.array(br[-1]["co"])); rad.append(br[-1]["radius"])
    P = np.array(pts)
    return np.c_[P[:, 0], P[:, 2], P[:, 1]], np.array(rad)   # z 上 → y 上


def _broadleaf(name, lod):
    idx, H, seed = BROADLEAVES[name]
    young = name.startswith("sapling")
    k = SAPLING if young else dict(spread=1.0, leaves=1.0, card=1.0, reach=0.16, low=0.3)
    proto = _prototypes()[idx]
    rng = np.random.default_rng(seed * 10 + lod)
    s = H / proto["height_m"]
    leaf, wood = Part(), Part()
    min_r, sides, steps = [(0.012, 6, 2), (0.05, 4, 1), (0.12, 4, 1)][lod]
    skel = []
    for bi, br in enumerate(proto["branches"]):
        P, r = _branch_points(br, steps)
        sp = 0.72 * k["spread"]
        P, r = P * s * np.array([sp * (-1 if name == "broadleaf_c" else 1), 1.0, sp]), r * s * (0.6 if young else 1.0)   # c は a の骨格を左右反転
        if bi == 0:
            P = np.r_[[P[0] - np.array([0, 0.015 * H, 0])], P]; r = np.r_[r[0] * 1.1, r]
        keep = r >= min_r
        if keep.sum() >= 2:
            tube(wood, P[keep], r[keep], sides if r.max() > 0.08 else max(3, sides - 2), uv_v=0.5, ao=0.65)
        skel.append(P[int(len(P) * 0.3):])
    skel = np.concatenate(skel)
    skel = skel[skel[:, 1] > 0.35 * H]
    # 樹冠：枝の点のまわり（半径 reach）に葉の茂みを置く。外側ほど多く、内側は暗く
    reach = k["reach"] * H
    center = skel.mean(0)
    count, size, crossed = [(420, (1.5, 2.2), True), (170, (2.2, 3.1), False), (60, (3.3, 4.5), False)][lod]
    count = int(count * k["leaves"]); size = tuple(x * k["card"] for x in size)
    base = skel[rng.integers(len(skel), size=count * 3)]
    off = _unit(rng.normal(0, 1, (count * 3, 3)) + _unit(base - center) * 1.2) * reach * rng.uniform(0.3, 1.0, (count * 3, 1))
    cand = base + off
    dist = np.linalg.norm(cand - center, axis=1)
    w = (dist / dist.max()) ** 2 + 0.1
    pick = cand[rng.choice(len(cand), count, replace=False, p=w / w.sum())]
    pick = pick[pick[:, 1] > k["low"] * H]
    rects = [(0, 0, 0.5, 1), (0.5, 0, 1, 1)]
    rmax = np.linalg.norm((pick - center), axis=1).max() + 1e-6
    for p in pick:
        out = p - center
        u = _unit(out * 0.7 + UP * 0.6 + rng.normal(0, 0.35, 3))
        v = _rot(_unit(np.cross(UP, u) + 1e-6), u, rng.uniform(-0.8, 0.8))
        sz = rng.uniform(*size)
        root = p - u * sz * 0.35
        rel = np.linalg.norm(out) / rmax
        ao = (0.35 + 0.45 * rel, 0.7 + 0.3 * rel)
        rect = rects[rng.integers(2)]
        card(leaf, root, u, v, sz, sz, rect, bend=0.15, segs=1, along="y", center=center, ao=ao)
        if crossed:
            card(leaf, root, u, _rot(v, u, np.pi / 2), sz * 0.9, sz * 0.9, rect, bend=0.15, segs=1, along="y",
                 center=center, ao=ao)
    return leaf, wood, max(0.0, (pick[:, 1].min() - 1.0) / H)


# ---------- 低木・シダ ----------

def _shrub(name, lod):
    H, seed = SHRUBS[name]
    rng = np.random.default_rng(seed)       # 枝の形は全 LOD で同じ
    leaf, wood = Part(), Part()
    stems = []
    for _ in range(6):
        az = rng.uniform(0, 2 * np.pi); lean = rng.uniform(0.15, 0.5)
        top = np.array([np.cos(az) * lean * H, H * rng.uniform(0.7, 1.0), np.sin(az) * lean * H])
        stems.append((np.array([0.0, -0.015 * H, 0.0]), top))
        if lod < 2:
            tube(wood, [stems[-1][0], top * 0.95], [0.025, 0.008], 3, ao=0.5)
    center = np.array([0.0, H * 0.6, 0.0])
    rng2 = np.random.default_rng(seed * 10 + lod)
    count, size = [(26, (0.55, 0.85)), (12, (0.8, 1.1)), (5, (1.1, 1.5))][lod]
    for i in range(count):
        a, b = stems[i % len(stems)]
        p = a + (b - a) * rng2.uniform(0.35, 1.0)
        u = _unit(p - center * np.array([0, 0.3, 0]) + rng2.normal(0, 0.3, 3) + UP * 0.5)
        v = _unit(np.cross(UP, u) + 1e-6)
        sz = min(rng2.uniform(*size), H * 0.8)
        root = p - u * sz * 0.3
        root[1] = max(root[1], 0.0)
        card(leaf, root, u, v, sz, sz, ((0, 0, 0.5, 1), (0.5, 0, 1, 1))[i % 2], segs=1, along="y", center=center,
             ao=(0.5, 0.95))
    return leaf, wood, 0.0


def _fern(name, lod):
    H, seed = FERNS[name]
    rng = np.random.default_rng(seed)
    leaf = Part()
    n, segs = [(9, 3), (9, 2), (5, 1)][lod]
    for i in range(n):
        az = 2 * np.pi * i / n + rng.normal(0, 0.2)
        el = np.radians(rng.uniform(45, 70))
        u = np.array([np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)])
        v = _unit(np.cross(UP, u))
        L = H * rng.uniform(1.0, 1.35)
        k = i % 4
        card(leaf, np.zeros(3), u, v, L, L * 0.26, (k / 4, 0, (k + 1) / 4, 1), bend=el * 0.95, segs=segs,
             along="y", center=np.array([0, -0.3, 0]), ao=(0.55, 1.0))
    return leaf, None, 0.0


# ---------- テクスチャ ----------

def _species_of(name):
    return next(k for k, v in SPECIES.items() if name in v)


def _dds(png: bytes) -> bytes:
    """CSP は PNG だとミップマップが無いと警告し、木の準備（遠景画像の作成）が進まなかった（2026-10-19 のログ）。
    他のコースと同じく、ミップマップ付きの DDS にする。"""
    from PIL import Image
    return kn5.dds_bytes(Image.open(io.BytesIO(png)))


@lru_cache(maxsize=None)
def species_textures(species):
    return {k.replace(".png", ".dds"): _dds(v) for k, v in _species_png(species).items()}


def _species_png(species):
    if species == "fern":
        c, n, _ = ft.fern_fronds()
        return {"fern_leaf.png": ft.png_rgba(c), "fern_leaf_nm.png": ft.png_normal(n)}
    tints = TINTS[species]
    if species in ("sugi", "hinoki"):
        dens = 1.25 if species == "sugi" else 1.0
        parts = [ft.conifer_branch(100 + i, tints[i], density=dens) for i in range(2)]
        canvas = np.concatenate([p[0] for p in parts]); nrm = np.concatenate([p[1] for p in parts])
    else:
        n = {"broadleaf": 40, "sapling": 26}.get(species, 30)
        parts = [ft.leaf_cluster(200 + i, tints[i], n=n) for i in range(2)]
        canvas = np.concatenate([p[0] for p in parts], 1); nrm = np.concatenate([p[1] for p in parts], 1)
    bark, bnrm = ft.bark(species if species in ("sugi", "hinoki") else "broadleaf")
    return {f"{species}_leaf.png": ft.png_rgba(canvas), f"{species}_leaf_nm.png": ft.png_normal(nrm),
            f"{species}_bark.png": ft.png_rgb(bark), f"{species}_bark_nm.png": ft.png_normal(bnrm)}


# ---------- まとめ ----------

def _pack(parts, H, half_w, tex_names):
    subs = []
    for k, part in enumerate(parts):
        arr = part.arrays() if part is not None else None
        if arr is None:
            continue
        p, n, uv, t, ao, f = arr
        v = np.zeros(len(p), csptree.VERT_DTYPE)
        v["p"] = np.c_[p[:, 0] / half_w, p[:, 1] / H, p[:, 2] / half_w]
        v["n"] = _unit(n)
        v["uv"] = uv
        v["b0"] = np.c_[np.clip(_unit(t) * 127.5 + 127.5, 0, 255), np.zeros(len(p))].astype(np.uint8)
        v["b2"] = np.c_[np.clip(ao * 255, 0, 255), np.full(len(p), 255), np.full(len(p), 255), np.zeros(len(p))]
        subs.append(csptree.Submesh(2 * k, 2 * k + 1, f.reshape(-1).astype(np.uint16), v))
    return subs


def build_pattern(name: str, seed: int = 1) -> TreeModel:
    species = _species_of(name)
    builders = {"sugi": lambda l: _conifer(name, l, species), "hinoki": lambda l: _conifer(name, l, species),
                "broadleaf": lambda l: _broadleaf(name, l), "sapling": lambda l: _broadleaf(name, l), "shrub": lambda l: _shrub(name, l),
                "fern": lambda l: _fern(name, l)}
    raw = [builders[species](lod) for lod in range(3)]
    allp = np.concatenate([part.arrays()[0] for leaf, wood, _ in raw for part in (leaf, wood)
                           if part is not None and part.arrays() is not None])
    H = float(allp[:, 1].max())
    half_w = float(np.abs(allp[:, [0, 2]]).max())
    textures = species_textures(species)
    names = list(textures)
    lods = [_pack([leaf, wood], H, half_w, names) for leaf, wood, _ in raw]
    # 視界・張り出しの判定用：遠くで使う形（板が大きい）も含めて、いちばん広く・いちばん低い樹冠
    leaves = np.concatenate([leaf.arrays()[0] for leaf, _, _ in raw])
    crown_r = float(np.linalg.norm(leaves[:, [0, 2]], axis=1).max())
    crown_base = max(0.0, float(leaves[:, 1].min()) / H)
    ini = {"BASIC": {"HEIGHT": round(H, 3), "WIDTH": round(2 * half_w, 3), "DISTANCE": DRAW_DISTANCE[species]},
           "SHADING": {"SPECULAR": 0.03, "SUBSCATTERING": 0.3, "REFLECTIVITY": 0.02, "BRIGHTNESS": 1.0},
           "NORMALS": {"BOOST": 1, "SPHERE": 0.3}}
    for i, d in enumerate(LOD_DISTANCE):
        ini[f"LOD_{i}"] = {"MODEL": f"LOD{i}.mesh", "DISTANCE": d}
    return TreeModel(lods, H, 2 * half_w, crown_base, crown_r, textures, ini)


def write_all(out_dir: Path, names=None) -> dict:
    out = {}
    for name in names or [n for v in SPECIES.values() for n in v]:
        m = build_pattern(name)
        tex = list(m.textures)
        meshes = {f"LOD{i}.mesh": csptree.write_mesh(tex, lod) for i, lod in enumerate(m.lods)}
        csptree.write_package(Path(out_dir) / f"{name}.zip", m.ini, meshes, m.textures)
        out[name] = m
    return out
