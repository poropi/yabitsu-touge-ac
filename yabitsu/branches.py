"""県道70号が2本に分かれて走る所（中に島がある所）：本線（中心線の道）と並ぶ、もう1本の県道70号を走れる路面として作り、
間の島（縁石・砂利）と、島の上の物（鳥居・灯籠・石碑・植え込み・木）を置く。

data/manual/s3_branches.json：{"<OSM way id>": {"width_m", "surface": asphalt | paint_red, "island": [[lat, lon], ...],
  "objects": [{"kind": torii | lantern | stele | shrub | tree, "at": [lat, lon], "facing_deg", ...}]}}"""
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from . import geo, osm, site
from .crosssection import GUTTER_WIDTH
from .start_lot import road_formation

MIN_OFFSET_M = 3.0     # 本線の中心線から、これより離れる県道70号の way を「並ぶもう1本」とする
ISLAND_H = 0.15
END_NEAR_M = 6.0
PAD_STEP_M = 8.0
CLOSE_M = 6.0         # 本線と東側の道の間の、これより狭いすき間は路面で埋める（島は上に乗る）      # もう1本の両はしは、本線の中心線からこの距離の中


def _ll(x, y):
    lat, lon = geo.from_local(float(x), float(y))
    return float(lat), float(lon)


def _xy(ll):
    a = np.asarray(ll, float).reshape(-1, 2)
    x, y = geo.to_local(a[:, 0], a[:, 1])
    return np.c_[x, y]


def _merge(parts):
    vs, fs, us, base = [], [], [], 0
    for v, f, uv in parts:
        f = np.asarray(f)
        if f.shape[1] == 3:
            f = np.c_[f, f[:, 2]]
        vs.append(np.asarray(v, float)); fs.append(f + base); us.append(np.asarray(uv, float)); base += len(v)
    return np.concatenate(vs), np.concatenate(fs).astype(np.int32), np.concatenate(us)


class Branch:
    def __init__(self, osm_id, line, sec, spec):
        self.osm_id, self.line, self.sec, self.spec = osm_id, line, sec, spec
        self.width = float(spec.get("width_m", 4.0))
        form = road_formation(sec, 0.0, float(sec.s[-1]))
        # 分かれる所から合わさる所までの本線（両はしの前後 CLOSE_M）と東側の道を合わせ、すき間（島のまわりの地面）を埋める
        main = LineString(np.c_[sec.x, sec.y])
        s_a, s_b = sorted(main.project(Point(q)) for q in (line.coords[0], line.coords[-1]))
        seg = road_formation(sec, s_a - CLOSE_M, s_b + CLOSE_M)
        g = line.buffer(self.width / 2, cap_style="flat").union(seg).buffer(CLOSE_M).buffer(-CLOSE_M)
        g = g.difference(form)
        self.surface = site.largest(g) if not g.is_empty else Polygon()
        self.island = orient(Polygon(_xy(spec["island"])).buffer(0), 1.0) if spec.get("island") else None
        if self.island is not None:
            self.surface = site.largest(self.surface.difference(self.island))

    def z(self, x, y):
        """路面の高さ：一番近い本線の点の、こちら側の縁の高さ（横へ水平に延ばす）。"""
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        sec = self.sec
        i = np.argmin((sec.x[None, :] - x[:, None]) ** 2 + (sec.y[None, :] - y[:, None]) ** 2, axis=1)
        h = sec.heading[i]; lat = -(x - sec.x[i]) * np.sin(h) + (y - sec.y[i]) * np.cos(h)
        return np.where(lat >= 0, sec.edge_z(1)[i], sec.edge_z(-1)[i])

    def meshes(self):
        from .town import grid_drape
        out = []
        mat = self.spec.get("surface", "asphalt")
        for k, (v, f, uv) in enumerate(grid_drape(self.surface, self.z, 0.0, cell=1.0)):
            out.append((f"1ROAD_branch_{self.osm_id}_{k:02d}", v, f, uv, mat, True))
        if self.island is not None:
            for k, (v, f, uv) in enumerate(grid_drape(self.island, self.z, ISLAND_H, cell=1.0)):
                out.append((f"branch_{self.osm_id}_island_{k:02d}", v, f, uv, "gravel", True))
            p = np.asarray(self.island.exterior.segmentize(1.0).coords)
            zg = self.z(p[:, 0], p[:, 1])
            vs, fs = [], []
            for i in range(len(p) - 1):   # 縁石（外向き＝反時計回りの辺の右）
                b = len(vs)
                vs += [[*p[i], zg[i] - 0.1], [*p[i + 1], zg[i + 1] - 0.1], [*p[i + 1], zg[i + 1] + ISLAND_H], [*p[i], zg[i] + ISLAND_H]]
                fs.append([b + 1, b, b + 3, b + 2])
            v = np.array(vs, float)
            out.append((f"branch_{self.osm_id}_curb", v, np.array(fs, np.int32), v[:, :2], "concrete", True))
            out.append((f"1WALL_branch_{self.osm_id}_island_block", *_island_wall(self.island, self.z), "concrete", False))
        out += island_objects(self.spec.get("objects", []), lambda x, y: self.z(x, y) + (ISLAND_H if self.island is not None else 0.0),
                              f"branch_{self.osm_id}")
        return out

    def pads(self, step=PAD_STEP_M):
        """地形合わせ：本線の向きに step ごとに切った、傾いた平面（坂でも地形が路面の上に出ない）。"""
        poly = self.surface.union(self.island) if self.island is not None else self.surface
        main = LineString(np.c_[self.sec.x, self.sec.y])
        ring = np.asarray(poly.exterior.coords)
        ts = [main.project(Point(q)) for q in ring]
        out = []
        for k, a in enumerate(np.arange(min(ts), max(ts), step)):
            pa = np.array(main.interpolate(a).coords[0]); pb = np.array(main.interpolate(a + step).coords[0])
            d = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9); n = np.array([-d[1], d[0]])
            cut = Polygon([pa - n * 40, pb - n * 40, pb + n * 40, pa + n * 40])
            piece = poly.intersection(cut)
            for g in ([piece] if piece.geom_type == "Polygon" else [q for q in getattr(piece, "geoms", []) if q.geom_type == "Polygon"]):
                if g.area < 0.5:
                    continue
                c = np.asarray(g.exterior.coords)
                zc = self.z(c[:, 0], c[:, 1]) + (ISLAND_H if self.island is not None and self.island.intersects(g) else 0.0)
                cx, cy = c[:, 0].mean(), c[:, 1].mean()
                A = np.c_[np.ones(len(c)), c[:, 0] - cx, c[:, 1] - cy]
                z0, gx, gy = np.linalg.lstsq(A, zc, rcond=None)[0]
                out.append({"name": f"branch_{self.osm_id}_{k}_{len(out)}", "role": "flat", "kind": "plaza",
                            "polygon": c.tolist(), "z": float(z0), "plane": [float(cx), float(cy), float(z0), float(gx), float(gy)]})
        return out


def _island_wall(island, zf):
    """島に車が乗り上げないよう、縁石の上に見えない壁（高さ 1m、島の 0.3m 内側）。"""
    ring = island.buffer(-0.3)
    ring = site.largest(ring) if not ring.is_empty else island
    p = np.asarray(ring.exterior.coords); zg = zf(p[:, 0], p[:, 1])
    vs, fs = [], []
    for i in range(len(p) - 1):
        b = len(vs)
        vs += [[*p[i], zg[i]], [*p[i + 1], zg[i + 1]], [*p[i + 1], zg[i + 1] + 1.2], [*p[i], zg[i] + 1.2]]
        fs += [[b, b + 1, b + 2, b + 3], [b + 3, b + 2, b + 1, b]]
    v = np.array(vs, float)
    return v, np.array(fs, np.int32), v[:, :2]


def parallel_branches(elements, sec, specs):
    """本線に並ぶ県道70号の way（本線の中心線から MIN_OFFSET_M 以上離れる所がある way）。"""
    main = LineString(np.c_[sec.x, sec.y])
    out = []
    for w in osm.route70_ways(elements):
        p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
        xy = np.c_[geo.to_local(p[:, 0], p[:, 1])]
        ln = LineString(xy)
        if max(Point(q).distance(main) for q in np.asarray(ln.segmentize(2.0).coords)) < MIN_OFFSET_M:
            continue
        # 両はしが本線につながっていること（道の終わりの先へ続く県道70号は「もう1本」ではない）
        if max(Point(xy[0]).distance(main), Point(xy[-1]).distance(main)) > END_NEAR_M:
            continue
        out.append(Branch(int(w["id"]), ln, sec, specs.get(str(w["id"]), {})))
    return out


# --- 島の上の物 ---

def _cyl(c, r, z0, z1, n=12, r1=None):
    r1 = r if r1 is None else r1
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.array([[c[0] + r * math.cos(t), c[1] + r * math.sin(t), z0] for t in a] +
                 [[c[0] + r1 * math.cos(t), c[1] + r1 * math.sin(t), z1] for t in a] + [[c[0], c[1], z1]], float)
    f = [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)] + [[n + k, n + (k + 1) % n, 2 * n, 2 * n] for k in range(n)]
    uv = np.c_[np.r_[np.arange(n) / n, np.arange(n) / n, 0.5], np.r_[np.zeros(n), np.ones(n), 1.0]]
    return v, np.array(f), uv


def _bar(c, u, n, half_len, half_w, half_h, z, lift_ends=0.0, segs=8):
    """向き u に長い角材（中心 c、高さの中心 z）。lift_ends で両端を持ち上げる（笠木の反り）。"""
    t = np.linspace(-1, 1, segs + 1)
    vs = []
    for s in t:
        zc = z + lift_ends * s * s
        p = c + u * half_len * s
        for dn, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vs.append([*(p + n * half_w * dn), zc + half_h * dz])
    v = np.array(vs, float); m = len(t)
    f = []
    for i in range(m - 1):
        a, b = 4 * i, 4 * (i + 1)
        f += [[a, a + 1, b + 1, b], [a + 1, a + 2, b + 2, b + 1], [a + 2, a + 3, b + 3, b + 2], [a + 3, a, b, b + 3]]
    f += [[3, 2, 1, 0], [4 * (m - 1), 4 * (m - 1) + 1, 4 * (m - 1) + 2, 4 * (m - 1) + 3]]
    return v, np.array(f), v[:, [0, 2]] / 2.0


def torii_meshes(c, z, facing_deg, span=3.0, height=4.2, prefix="torii"):
    """石の明神鳥居。facing_deg はくぐる向き（方位）。柱（少し内へ傾ける）・台石・貫（柱を貫いて出る）・額束と額・島木・反りのある笠木。"""
    d = np.array([math.sin(math.radians(facing_deg)), math.cos(math.radians(facing_deg))])   # くぐる向き
    u = np.array([d[1], -d[0]])                                                            # 横（鳥居の幅の向き）
    r_col = 0.22; parts = []
    for s in (-1, 1):
        base = c + u * s * span / 2
        parts.append(_cyl(base, 0.38, z - 0.2, z + 0.35, 16))                              # 台石
        top = c + u * s * (span / 2 - 0.12)                                                # 内へ転ばす
        v, f, uv = _cyl(base, r_col, z + 0.35, z + height - 0.55, 14)
        n = len(v) // 2
        v[n:-1, :2] += (top - base)[None, :]; v[-1, :2] += (top - base)
        parts.append((v, f, uv))
    parts.append(_bar(c, u, d, span / 2 + 0.45, 0.13, 0.13, z + height - 1.35))                 # 貫
    parts.append(_bar(c, u, d, 0.12, 0.12, 0.30, z + height - 0.92))                           # 額束
    plaque = c - d * 0.14
    pv = np.array([[*(plaque - u * 0.28), z + height - 1.15], [*(plaque + u * 0.28), z + height - 1.15],
                   [*(plaque + u * 0.28), z + height - 0.65], [*(plaque - u * 0.28), z + height - 0.65]])
    parts.append((pv, np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), np.array([[0, 0], [1, 0], [1, 1], [0, 1]])))   # 額
    parts.append(_bar(c, u, d, span / 2 + 0.6, 0.2, 0.12, z + height - 0.5))                   # 島木
    parts.append(_bar(c, u, d, span / 2 + 0.95, 0.26, 0.15, z + height - 0.22, lift_ends=0.22))   # 笠木（反り）
    return [(f"{prefix}", *_merge(parts), "stone", True)]


def lantern_meshes(c, z, facing_deg, height=2.6, prefix="lantern"):
    """石灯籠：基礎（2段）・竿・中台・火袋（窓あき）・笠（角が反る）・宝珠。"""
    d = np.array([math.sin(math.radians(facing_deg)), math.cos(math.radians(facing_deg))]); u = np.array([d[1], -d[0]])
    k = height / 2.6
    parts = [_bar(c, u, d, 0.42 * k, 0.42 * k, 0.12 * k, z + 0.12 * k), _bar(c, u, d, 0.32 * k, 0.32 * k, 0.1 * k, z + 0.34 * k),
             _cyl(c, 0.12 * k, z + 0.44 * k, z + 1.3 * k, 10), _bar(c, u, d, 0.3 * k, 0.3 * k, 0.09 * k, z + 1.39 * k)]
    for s in (-1, 1):   # 火袋：前後に窓のあいた柱の組
        for q in (-1, 1):
            parts.append(_bar(c + u * s * 0.17 * k + d * q * 0.17 * k, u, d, 0.05 * k, 0.05 * k, 0.17 * k, z + 1.65 * k))
    parts.append(_bar(c, u, d, 0.2 * k, 0.2 * k, 0.03 * k, z + 1.5 * k))
    parts.append(_bar(c, u, d, 0.42 * k, 0.42 * k, 0.08 * k, z + 1.92 * k, lift_ends=0.06 * k))   # 笠
    parts.append(_bar(c, d, u, 0.42 * k, 0.42 * k, 0.08 * k, z + 1.92 * k, lift_ends=0.06 * k))
    parts.append(_cyl(c, 0.2 * k, z + 2.0 * k, z + 2.25 * k, 10, r1=0.05 * k))
    parts.append(_cyl(c, 0.09 * k, z + 2.25 * k, z + 2.55 * k, 10, r1=0.02 * k))                   # 宝珠
    return [(f"{prefix}", *_merge(parts), "stone", True)]


def _blob(c, r, z, h, n=10, seed=0):
    """植え込み・木の葉：つぶれた半球。"""
    rng = np.random.default_rng(seed)
    rings = []
    for j, t in enumerate(np.linspace(0, 1, 5)):
        rr = r * math.sqrt(max(1 - t * t, 0.02)) * (0.9 + 0.2 * rng.random())
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rings.append([[c[0] + rr * math.cos(q), c[1] + rr * math.sin(q), z + h * t] for q in a])
    v = np.array([p for ring in rings for p in ring] + [[c[0], c[1], z + h]], float)
    f = []
    for j in range(len(rings) - 1):
        for k in range(n):
            a, b = j * n + k, j * n + (k + 1) % n
            f.append([a, b, b + n, a + n])
    top = len(v) - 1
    f += [[(len(rings) - 1) * n + k, (len(rings) - 1) * n + (k + 1) % n, top, top] for k in range(n)]
    return v, np.array(f), v[:, :2] / 2.0


def island_objects(objects, zf, prefix):
    out, leaves, trunks, stones = [], [], [], []
    for k, ob in enumerate(objects):
        p = _xy([ob["at"]])[0]; z = float(np.atleast_1d(zf(np.atleast_1d(p[0]), np.atleast_1d(p[1])))[0])
        kind = ob["kind"]
        if kind == "torii":
            out += torii_meshes(p, z, float(ob.get("facing_deg", 0.0)), float(ob.get("span_m", 3.0)), float(ob.get("height_m", 4.2)),
                                prefix=f"1WALL_{prefix}_torii_{k}")
        elif kind == "lantern":
            out += lantern_meshes(p, z, float(ob.get("facing_deg", 0.0)), float(ob.get("height_m", 2.6)), prefix=f"{prefix}_lantern_{k}")
        elif kind == "stele":
            w, dpt, h = ob.get("size", [0.7, 0.3, 1.8])
            d = np.array([math.sin(math.radians(ob.get("facing_deg", 0.0))), math.cos(math.radians(ob.get("facing_deg", 0.0)))])
            stones.append(_bar(p, np.array([d[1], -d[0]]), d, w / 2, dpt / 2, h / 2, z + h / 2))
            stones.append(_bar(p, np.array([d[1], -d[0]]), d, w / 2 + 0.2, dpt / 2 + 0.2, 0.15, z + 0.15))
        elif kind == "shrub":
            leaves.append(_blob(p, float(ob.get("r_m", 0.7)), z, float(ob.get("h_m", 0.9)), seed=k))
        elif kind == "tree":
            h = float(ob.get("h_m", 7.0)); r = float(ob.get("r_m", 2.5))
            trunks.append(_cyl(p, 0.2, z - 0.2, z + h * 0.7, 8, r1=0.12))
            leaves.append(_blob(p, r, z + h * 0.58, h * 0.42, seed=k + 50))   # 葉は幹の上の方だけ（下を見通せる）
    if stones:
        out.append((f"{prefix}_stones", *_merge(stones), "stone", True))
    if leaves:
        out.append((f"{prefix}_leaves", *_merge(leaves), "hedge", True))
    if trunks:
        out.append((f"{prefix}_trunks", *_merge(trunks), "wood", True))
    return out
