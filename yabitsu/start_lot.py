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


def road_formation(sec, s_min=0.0, s_max=80.0):
    """s_min〜s_max の、路面＋側溝が占める範囲。"""
    m = (sec.s >= s_min) & (sec.s <= s_max)
    hl, hr = sec.hw_left[m] + GUTTER_WIDTH, sec.hw_right[m] + GUTTER_WIDTH
    nx, ny = -np.sin(sec.heading[m]), np.cos(sec.heading[m])
    left = np.c_[sec.x[m] + hl * nx, sec.y[m] + hl * ny]
    right = np.c_[sec.x[m] - hr * nx, sec.y[m] - hr * ny]
    return Polygon(np.r_[left, right[::-1]]).buffer(0)


LOT_ROAD_RANGE_M = 150.0  # 駐車場の原点に一番近い道の点から、前後この距離の道と重ならないように削る


def lot_polygon(spec, sec):
    o = np.asarray(spec["origin"], float)
    s_c = float(sec.s[int(np.argmin(np.hypot(sec.x - o[0], sec.y - o[1])))])
    road = road_formation(sec, s_min=s_c - LOT_ROAD_RANGE_M, s_max=s_c + LOT_ROAD_RANGE_M)
    f = frame(spec)
    poly = f.rect(spec["lot"]["a"], spec["lot"]["b"])
    for pts in spec.get("lot_extra", []):   # 航空写真の舗装の広がり（北東の扇形・南西の端）：a・b の多角形
        pts = np.asarray(pts, float)
        poly = poly.union(Polygon(f.xy(pts[:, 0], pts[:, 1])).buffer(0))
    poly = poly.difference(road)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


LOT_NAME = "nanohanadai_lot"
LOT_JOIN_M = 4.0    # 入口：舗装を道路（側溝の外の縁）の高さから、この幅で駐車場の高さへ寄せる
LOT_GAP_M = 0.8     # 入口：舗装と側溝の間のすき間（この幅まで）を舗装で埋める
LOT_JOIN_CELL = 0.5


def mouths(pads, sec, reach=0.8):
    """駐車場が本線の側溝に接している区間 [(side, s0, s1)]。この側は側溝の V 字のくぼみを作らず、路面を縁の高さのまま
    延ばす（assembly.add_road の mouths）。2026-10-30 舗装を寄せたあとも、入口に側溝のくぼみ（0.12 m）が段差として残っていた。"""
    import shapely
    out = []
    for p in pads:
        if p.get("kind") != "lot":
            continue
        edge = Polygon(p["polygon"]).buffer(reach)
        for side in (1, -1):
            # 側溝の帯（区間ごとの四角）が駐車場にかかる区間。中心の点だけで見ると、ヘアピンの内側で隣の区間の帯が
            # 斜めに入口へ入り込むのを拾えない（2026-10-30、S1 の 4〜4.5 m に V 字が残った）
            hw = sec.hw_left if side > 0 else sec.hw_right
            nn = np.c_[-np.sin(sec.heading), np.cos(sec.heading)] * side
            a = np.c_[sec.x, sec.y] + nn * hw[:, None]; b = a + nn * GUTTER_WIDTH
            quads = shapely.polygons(np.stack([a[:-1], a[1:], b[1:], b[:-1]], 1))
            hit = np.flatnonzero(shapely.intersects(quads, edge))
            if len(hit):
                out.append((side, float(sec.s[hit.min()]), float(sec.s[hit.max() + 1])))
    return out


def lot_surface(spec, pad, sec=None):
    """駐車場の舗装の面（頂点・三角形・UV）。sec があれば、道路と接する所を段差なくつなぐ（2026-10-30 ゲームで道路から
    駐車場へ入ると段差で衝撃が出た：舗装は平らな 562.27、道路の縁は 562.31〜562.80、間に 0.2〜0.4 m のすき間）。
    すき間を埋め、道路から LOT_JOIN_M の間は、道路の縁の高さ（一番近い所）から駐車場の高さへ一次で寄せる。"""
    import shapely
    from scipy.spatial import cKDTree
    from shapely.geometry import box as _box
    from .crosssection import surface_height
    z = pad["z"]; poly = Polygon(pad["polygon"]).buffer(0)
    if sec is None:
        return pad_mesh(pad["polygon"], z)
    o = np.asarray(spec["origin"], float)
    s_c = float(sec.s[int(np.argmin(np.hypot(sec.x - o[0], sec.y - o[1])))])
    road = road_formation(sec, s_min=s_c - LOT_ROAD_RANGE_M, s_max=s_c + LOT_ROAD_RANGE_M)
    gap = poly.buffer(LOT_GAP_M).intersection(road.buffer(LOT_GAP_M)).difference(road)
    full = poly.union(gap).buffer(0)
    if full.geom_type != "Polygon":
        full = max(full.geoms, key=lambda g: g.area)
    join = full.intersection(road.buffer(LOT_JOIN_M))
    m = (sec.s >= s_c - LOT_ROAD_RANGE_M) & (sec.s <= s_c + LOT_ROAD_RANGE_M)
    edges = []                                   # 道路の両側の、側溝の外の縁（高さは路面の縁）
    for side, hw in ((1, sec.hw_left[m]), (-1, sec.hw_right[m])):
        nx, ny = -np.sin(sec.heading[m]), np.cos(sec.heading[m])
        w = side * (hw + GUTTER_WIDTH)
        edges.append(np.c_[sec.x[m] + w * nx, sec.y[m] + w * ny,
                           surface_height(side * hw, sec.z[m], sec.slope[m], sec.crown[m])])
    edges = np.concatenate(edges)
    tree = cKDTree(edges[:, :2])

    def zf(xy):
        d = shapely.distance(road, shapely.points(xy))
        _, j = tree.query(xy)
        w = np.clip(1.0 - d / LOT_JOIN_M, 0.0, 1.0)
        return z + (edges[j, 2] - z) * w

    def tris(g):
        out = []
        for part in getattr(g, "geoms", [g]):
            if part.geom_type == "Polygon" and part.area > 1e-6:
                out += [np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(part).geoms]
        return out

    flat = tris(full.difference(join.buffer(1e-9)))
    slope = []
    if not join.is_empty:
        x0, y0, x1, y1 = join.bounds; c = LOT_JOIN_CELL
        for gx in np.arange(x0, x1, c):
            for gy in np.arange(y0, y1, c):
                cell = _box(gx, gy, gx + c, gy + c)
                if cell.intersects(join):
                    slope += tris(cell.intersection(join))
    verts, faces = [], []
    for group, is_slope in ((flat, False), (slope, True)):
        if not group:
            continue
        t = np.asarray(group, float)
        ccw = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0]) >= 0
        t[~ccw] = t[~ccw][:, ::-1]
        p = t.reshape(-1, 2)
        zz = zf(p) if is_slope else np.full(len(p), z)
        b = sum(len(v_) for v_ in verts)
        verts.append(np.c_[p, zz]); faces.append(np.arange(len(p)).reshape(-1, 3) + b)
    v = np.concatenate(verts)
    return v, np.concatenate(faces).astype(np.int32), v[:, :2] / 4.0


def apply(pads, spec, sec):
    """OSM の駐車場（名前が spec["osm_name"]）を、測った形に差し替える。高さは元の駐車場と同じ。"""
    f = frame(spec); out = []
    for p in pads:
        if p["name"] == spec["osm_name"]:
            pit = spec["pits"]
            p = {**p, "name": LOT_NAME, "kind": "lot", "polygon": np.asarray(lot_polygon(spec, sec).exterior.coords).tolist(),
                 "heading": f.heading, "pit_center": f.xy(pit["a"], pit["b"]).tolist(), "pit_spacing": pit["spacing"]}
        out.append(p)
    return out


def flat_pads(spec, pad):
    """建物と石碑のまわりの平らな地面。地形を削るときと、木をどけるときに使う（路面は張らない）。
    地形は「駐車場の高さ−0.3m」まで下げられるので、そのぶん持ち上げて地面が舗装とほぼ同じ高さになるようにする。"""
    f = frame(spec)
    # flats の 1 つ：{"a", "b"}（長方形）か {"points"}（a・b の多角形）。"dz"：駐車場からの高さ、"ramp"：{"a": [a0, a1], "dz": [d0, d1]}
    # （a に沿って高さが変わる斜面）、"sink"：上に舗装を張る所（格子の角を面より下げる＝kind "pass"）
    polys = [(_poly(f, r), r) for r in spec.get("flats", [])]
    if "tower" in spec and "tower" not in set(spec.get("replaced", [])):
        t = spec["tower"]
        polys.append((Polygon(_ring(f.xy(t["a"], t["b"]), t["plaza_r"], 16)), {}))
    if "path" in spec:
        pa = spec["path"]
        polys.append((LineString(f.xy(*np.asarray(pa["points"], float).T)).buffer(pa["width"] / 2 + 1.0), {}))
    # 駐車場の縁から 0.5 m は空ける（平らな地面が駐車場に接すると、舗装の下の地形の三角形が舗装より上へ出る）
    if "polygon" in pad and len(pad["polygon"]) >= 4:
        keep = Polygon(pad["polygon"]).buffer(0.5)
        polys = [(g, r) for p, r in polys for g in getattr(p.difference(keep), "geoms", [p.difference(keep)]) if g.area > 0.5]
    out = []
    for p, r in polys:
        e = {"name": "flat", "role": "flat", "polygon": np.asarray(p.exterior.coords).tolist(), "z": pad["z"] + 0.25 + r.get("dz", 0.0)}
        if "ramp" in r:
            (a0, a1), (d0, d1) = r["ramp"]["a"], r["ramp"]["dz"]; gr = (d1 - d0) / (a1 - a0)
            c = f.xy(a0, 0.0); e["z"] = pad["z"] + 0.25 + max(d0, d1)
            e["plane"] = [float(c[0]), float(c[1]), float(pad["z"] + 0.25 + d0), float(gr * f.u[0]), float(gr * f.u[1])]
        if r.get("sink"):
            e["kind"] = "pass"
        out.append(e)
    from . import site
    if "shelter" in spec and "hill" in spec["shelter"]:   # 東屋の島の丘：上の平らな面（地形＝駐車場＋dz）。まわりは道の縁へ地形の決まりの傾きで下る
        from .terrain import UNDER_DEPTH       # （2026-10-30 写真で東屋の床は約 565.6 m。今までの小さな盛り土では島が北の道より低かった）
        hl = spec["shelter"]["hill"]
        out.append({"name": "shelter_hill", "role": "flat", "polygon": np.asarray(_poly(f, hl).exterior.coords).tolist(),
                    "z": pad["z"] + hl["dz"] + UNDER_DEPTH})
    elif "shelter" in spec:             # 東屋の小高い所
        sh = spec["shelter"]; c = site.ll_to_xy([sh["center_ll"]])[0]
        out.append({"name": "shelter_mound", "role": "flat", "polygon": np.asarray(Polygon(_ring(c, max(sh["size"]) * 0.75, 16)).exterior.coords).tolist(),
                    "z": pad["z"] + sh["mound"]})
    if "hill_stairs" in spec:           # 階段の下の地形を段に沿った斜面にする
        hs = spec["hill_stairs"]; p0, p1 = site.ll_to_xy([hs["start_ll"], hs["end_ll"]]); q0, q1 = site.ll_to_xy(hs["side_ll"])
        d = p1 - p0; L = float(np.hypot(*d)); g = (hs["z1"] - hs["z0"]) / L * d / L
        poly = Polygon([p0, p1, q1 + (q1 - p1) * 0.2, q0 + (q0 - p0) * 0.2]).buffer(0.3)
        out.append({"name": "hill_stairs", "role": "flat", "polygon": np.asarray(poly.exterior.coords).tolist(), "z": hs["z0"],
                    "plane": [float(p0[0]), float(p0[1]), float(hs["z0"] - 0.02), float(g[0]), float(g[1])]})
    return out


def _poly(f, r):
    """flats・paved の 1 つの形：points（a・b の多角形）か、a・b の長方形。"""
    return Polygon(f.xy(*np.asarray(r["points"], float).T)).buffer(0) if "points" in r else f.rect(r["a"], r["b"])


def _ramp_dz(r, a):
    """paved・flats の高さ（駐車場から）：dz か、ramp（a0→a1 で d0→d1、外は端の値）。"""
    if "ramp" in r:
        (a0, a1), (d0, d1) = r["ramp"]["a"], r["ramp"]["dz"]
        lo, hi = (a0, d0), (a1, d1)
        if a0 > a1:
            lo, hi = hi, lo
        return np.interp(a, [lo[0], hi[0]], [lo[1], hi[1]])
    return np.full(np.shape(a), r.get("dz", 0.0))


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


def _wood_beam(g, u0, w0, u1, w1, t=0.12):
    """Frame g の a 軸（妻の幅の向き）と高さの面に、(u0, w0)→(u1, w1) の角材（厚み t）。"""
    p0, p1 = g.xy(u0, 0.0), g.xy(u1, 0.0)
    d = np.array([u1 - u0, w1 - w0]); L = float(np.hypot(*d)); d = d / max(L, 1e-9); nrm = np.array([-d[1], d[0]]) * t / 2
    pts = [(u0 + nrm[0], w0 + nrm[1]), (u1 + nrm[0], w1 + nrm[1]), (u1 - nrm[0], w1 - nrm[1]), (u0 - nrm[0], w0 - nrm[1])]
    v = []
    for off in (-t / 2, t / 2):
        for u, w in pts:
            xy = g.xy(u, off); v.append((xy[0], xy[1], w))
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return np.array(v), np.array(faces), np.c_[np.arange(8) % 2, np.arange(8) // 4 * L]


def shelter_meshes(spec, z_lot):
    """ヘアピンの内側の、小高い所のコンクリートの東屋：薄い平らな屋根と 4 本の柱。"""
    sh = spec["shelter"]
    from . import site
    c = site.ll_to_xy([sh["center_ll"]])[0]; g = Frame(c, sh["axis_deg"])
    L, W = sh["size"]; zb = z_lot + sh["mound"]; zt = zb + sh["height"]; t = sh["post"]
    posts = [box(g, (a_ - t / 2, a_ + t / 2), (b_ - t / 2, b_ + t / 2), zb - 0.5, zt, 1.0)
             for a_ in (-L / 2 + 0.5, L / 2 - 0.5) for b_ in (-W / 2 + 0.5, W / 2 - 0.5)]
    roof = box(g, (-L / 2, L / 2), (-W / 2, W / 2), zt, zt + sh["slab"], 2.0)
    under = _quad(g, [-L / 2, -L / 2, L / 2, L / 2], [-W / 2, W / 2, W / 2, -W / 2], [zt] * 4, 2.0)   # 下から見た天井
    return [("1WALL_s1_shelter_posts", *_merge(posts), "concrete", True),
            ("s1_shelter_roof", *_merge([roof, under]), "concrete", True)]


def hill_stairs_meshes(spec):
    """ヘアピンの外側の石積みの切れ目を上る階段（段ごとの箱）と、両側の手すり。"""
    hs = spec["hill_stairs"]
    from . import site
    p0, p1 = site.ll_to_xy([hs["start_ll"], hs["end_ll"]])
    q0, _ = site.ll_to_xy(hs["side_ll"])
    run_dir = p1 - p0; length = float(np.hypot(*run_dir))
    g = Frame(p0, float(np.degrees(np.arctan2(run_dir[1], run_dir[0]))))
    side = 1.0 if g.ab(q0)[1] > 0 else -1.0        # 壁の奥（道から離れる向き）
    w = hs["width"]; b = (0.0, side * w) if side > 0 else (side * w, 0.0)
    n = max(1, int(round((hs["z1"] - hs["z0"]) / hs["rise"]))); run = length / n; rise = (hs["z1"] - hs["z0"]) / n
    steps = [box(g, (k * run, length + 0.5), b, hs["z0"] - 0.3, hs["z0"] + (k + 1) * rise, 1.0) for k in range(n)]
    rails = []
    for b_ in (b[0] + 0.08, b[1] - 0.08):
        for a_ in np.arange(0.2, length + 0.01, 1.4):
            zb = hs["z0"] + a_ / run * rise
            rails.append(box(g, (a_ - 0.025, a_ + 0.025), (b_ - 0.025, b_ + 0.025), zb, zb + 0.9, 0.5))
        for a0, a1 in zip(np.arange(0.2, length, 1.4)[:-1], np.arange(0.2, length, 1.4)[1:]):
            zb = hs["z0"] + a0 / run * rise
            rails.append(box(g, (a0, a1), (b_ - 0.025, b_ + 0.025), zb + 0.85, zb + 0.9, 0.5))
    return [("1ROAD_s1_hill_stairs", *_merge(steps), "concrete", True),
            ("s1_hill_handrails", *_merge(rails), "railing", True)]


def meshes(spec, pad, sec=None):
    """(名前, 頂点, 面, UV, 材質, 描画するか) の並び。sec があれば舗装を道路と段差なくつなぐ（lot_surface）。"""
    f = frame(spec); z = pad["z"]
    out = []

    v, faces, _ = lot_surface(spec, pad, sec)
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
    if "return_b" in fe:                # 南西の端で山側へ直角に折り返す柵（a = fe["a"][0] の線、b は fb → return_b）
        a0, rb = fe["a"][0], fe["return_b"]
        posts = [box(f, (a0 - 0.04, a0 + 0.04), (b_ - 0.04, b_ + 0.04), z - 0.3, z + fh)
                 for b_ in np.arange(fb + fe["post_every"], rb + 1e-9, fe["post_every"])] + \
                [box(f, (a0 - 0.04, a0 + 0.04), (rb - 0.08, rb), z - 0.3, z + fh)]
        rails = [box(f, (a0 - RAIL_T, a0 + RAIL_T), (fb, rb), z + zz, z + zz + 0.06) for zz in (0.08, fh - 0.06)]
        out.append(("s1_lot_fence_return", *_merge(posts + rails), "railing", True))
        v, faces, uv = _quad(f, [a0] * 4, [fb, rb, rb, fb], [z + 0.14] * 2 + [z + fh - 0.06] * 2, 0.5)
        uv[:, 1] = [0, 0, 1, 1]
        out.append(("s1_lot_fence_return_bars", v, faces, uv, "railing_bars", True))
        out.append(("1WALL_s1_lot_fence_return", *_quad(f, [a0] * 4, [fb, rb, rb, fb], [z - 0.3] * 2 + [z + 1.2] * 2, 2.0),
                    "railing", False))

    bl = spec["blocks"]; L, W, H = bl["size"]
    blocks = [box(f, (a_ - L / 2, a_ + L / 2), (bl["b"] - W / 2, bl["b"] + W / 2), z - 0.2, z + H, 1.0)
              for a_ in np.arange(bl["a"][0], bl["a"][1] + 1e-9, bl["every"])]
    out.append(("1WALL_s1_lot_blocks", *_merge(blocks), "concrete", "blocks" not in set(spec.get("replaced", []))))

    rep = set(spec.get("replaced", []))   # Codex の部品（landmarks）に替えた物は、手で測った簡単な形を作らない
    mo = spec["monument"]; L, W, H = mo["size"]
    if "monument" in rep:
        pass
    elif mo.get("shape") == "rough":     # 自然石の石碑（上へすぼまる）と、正面の黒い銘板
        from .asama_area import _slab
        out.append(("1WALL_s1_monument", *_slab(f, mo["a"], mo["b"], L, W, z - 0.3, H + 0.3), "stone", True))
        pw, ph = mo["plaque"]
        out.append(("s1_monument_plaque", *_quad(f, [mo["a"] - pw / 2, mo["a"] + pw / 2, mo["a"] + pw / 2, mo["a"] - pw / 2],
                                                 [mo["b"] - W / 2 - 0.03] * 4, [z + 0.5] * 2 + [z + 0.5 + ph] * 2, 1.0),
                    "tunnel_dark", True))
    else:
        out.append(("1WALL_s1_monument", *box(f, (mo["a"] - L / 2, mo["a"] + L / 2), (mo["b"] - W / 2, mo["b"] + W / 2),
                                              z - 0.3, z + H, 1.0), "stone", True))

    bu = spec["building"]
    if "building" in rep:
        bu = dict(bu, truss=False)
    walls = box(f, bu["a"], bu["b"], z - 0.3, z + bu["wall_h"], 2.0)
    out.append(("1WALL_s1_building", *walls, "wall_white", "building" not in rep))   # 替えたときも当たり判定は残す
    if "building" not in rep:
        out.append(("s1_building_gables", *_gable_ends(f, bu["a"], bu["b"], z + bu["wall_h"], bu["roof_h"]), "wall_white", True))
        out.append(("s1_building_roof", *_gable_roof(f, bu["a"], bu["b"], z + bu["wall_h"], bu["roof_h"], bu["eaves"]),
                    "roof", True))

    if bu.get("truss"):                 # 妻の木の筋かい（X 字）と軒の下の横木
        a0, a1 = bu["a"]; b0, b1 = bu["b"]; zt = z + bu["wall_h"]
        beams = []
        for a_ in (a0 - 0.05, a1 + 0.05):
            g = Frame(f.xy(a_, b0), spec["axis_deg"] + 90.0)
            span = b1 - b0
            for (u0, w0), (u1, w1) in (((0.0, zt), (span, zt)), ((0.0, zt), (span / 2, zt + bu["roof_h"])),
                                        ((span, zt), (span / 2, zt + bu["roof_h"])), ((span * 0.25, zt), (span * 0.5, zt + bu["roof_h"] * 0.5)),
                                        ((span * 0.75, zt), (span * 0.5, zt + bu["roof_h"] * 0.5)), ((span / 2, zt), (span / 2, zt + bu["roof_h"]))):
                beams.append(_wood_beam(g, u0, w0, u1, w1))
        out.append(("s1_building_truss", *_merge(beams), "wood", True))

    if "shelter" in spec:
        out += shelter_meshes(spec, z)
    if "hill_stairs" in spec:
        out += hill_stairs_meshes(spec)
    if "tower" in spec and "tower" not in rep:   # Codex の展望台（landmarks の nanohanadai_tower）に替えたときは作らない
        out += tower_meshes(spec, z)
    if "path" in spec:
        out.append(("1ROAD_s1_path", *path_mesh(spec, z), spec["path"].get("material", "tiles"), True))
    for k, r in enumerate(spec.get("paved", [])):   # 敷石の道・広場（展望台の南の遊歩道など）：長方形か points の多角形、高さは dz・ramp
        g = _poly(f, r)
        if "ramp" in r:
            g = g.segmentize(1.0)
        v, faces, _ = pad_mesh(np.asarray(g.exterior.coords), z)
        a_, b_ = f.ab(v[:, :2])
        v = v.copy(); v[:, 2] = z + _ramp_dz(r, a_)
        out.append((f"1ROAD_s1_paved_{k}", v, faces, np.c_[a_, b_] / 2.0, r.get("material", "tiles"), True))
    for k, w in enumerate(spec.get("white_walls", [])):   # 白い低い壁：points の折れ線、高さは h[0] → h[1] へ 0.8 m ごとに段で下がる
        pts = np.asarray(w["points"], float); t = w.get("t", 0.25); h0, h1 = w["h"]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); total = seg.sum(); parts = []; s0 = 0.0
        for (p, q), L in zip(zip(pts[:-1], pts[1:]), seg):
            g = Frame(f.xy(*p), spec["axis_deg"] + math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])))
            for u0 in np.arange(0.0, L, 0.8):
                u1 = min(u0 + 0.8, L); h = h0 + (h1 - h0) * (s0 + u1) / total
                parts.append(box(g, (u0, u1), (-t / 2, t / 2), z - 0.2, z + h, 1.0))
            s0 += L
        out.append((f"1WALL_s1_white_wall_{k}", *_merge(parts), "wall_white", True))
    if "bollards" in spec and "bollards" not in rep:
        bo = spec["bollards"]
        posts = [box(Frame(f.xy(bo["a"], b_), spec["axis_deg"]), (-0.075, 0.075), (-0.075, 0.075), z - 0.3, z + bo["height"], 0.5)
                 for b_ in bo["b"]]
        out.append(("1WALL_s1_bollards", *_merge(posts), "wood", True))
    if spec.get("landmarks"):   # Codex の部品（石碑・トイレ・看板・ミラー・車止め・歩道の物）
        from . import landmarks
        out += landmarks.place(spec["landmarks"], f, z, "s1_lm")
    return out
