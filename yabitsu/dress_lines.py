"""S3 の路面の線（中央線・外側線・緑の路側帯）を、表（data/manual/s3_lines.csv）どおりに引き直す。
今の kn5 の線は CSP で隠し（表示だけ）、この線を仕上げの kn5 に入れる。2026-10-19 にストリートビュー 166 枚で全区間を確かめた。"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import meshgen as mg
from .crosssection import Section

STEP = 0.25            # 線を引く細かさ m（区切れた線のため、1 m より細かくする）
EDGE_INSET = 0.25      # 外側線の中心：道の端から内側へ
WIDTH = 0.15
LIFT = 0.012           # 今の線（0.005）より少し上
FLANK_OFFSET, FLANK_DASH = 0.9, (2.0, 2.0)
# 中央線の種類 → (材質, 区切り (線, すき間) m)
CENTRE = {"orange_solid": ("dress_line_orange", None), "orange_dotted": ("dress_line_orange", (0.75, 0.25)),
          "orange_flanked": ("dress_line_orange", None), "white_dashed": ("dress_line_white", (5.0, 5.0))}
EDGE_DASH = {"solid": None, "dashed": (2.0, 1.5)}


def resample(sec: Section, s_from: float, s_to: float, step=STEP) -> Section:
    """s_from〜s_to を step m ごとに並べ直した Section（線形補間、向きは unwrap して補間）。"""
    s = np.arange(max(s_from, sec.s[0]), min(s_to, sec.s[-1]) + 1e-9, step)
    f = lambda a: np.interp(s, sec.s, np.asarray(a, float))
    return replace(sec, s=s, x=f(sec.x), y=f(sec.y), z=f(sec.z), heading=np.interp(s, sec.s, np.unwrap(sec.heading)),
                   curvature=f(sec.curvature), lanes=np.round(f(sec.lanes)).astype(np.int32),
                   bridge=np.interp(s, sec.s, sec.bridge.astype(float)) > 0.5, hw_left=f(sec.hw_left),
                   hw_right=f(sec.hw_right), slope=f(sec.slope), crown=f(sec.crown))


def _green(rows, a, b):
    """a〜b に重なる緑の路側帯の幅（無ければ 0）。"""
    for r in rows:
        if r["item"] == "green" and float(r["s_from"]) < b and float(r["s_to"]) > a:
            return float(r["value"])
    return 0.0


def _gaps(rows):
    return [(float(r["s_from"]), float(r["s_to"])) for r in rows if r["item"] == "gap"]


def _pieces(rows, a, b):
    """a〜b を、緑の路側帯の境と gap（線を引かない所）で分ける。[(a', b', 緑の幅)]"""
    cuts = sorted({a, b} | {float(r[k]) for r in rows if r["item"] in ("green", "gap") for k in ("s_from", "s_to")
                            if a < float(r[k]) < b})
    gaps = _gaps(rows)
    return [(x, y, _green(rows, x, y)) for x, y in zip(cuts[:-1], cuts[1:])
            if not any(g0 <= (x + y) / 2 <= g1 for g0, g1 in gaps)]


def line_meshes(sec: Section, rows) -> list:
    """表の行 → [(名前, verts, faces, uvs, 材質)]。"""
    out = []

    def add(name, sub, lat, dash, mat, width=WIDTH):
        v, f, uv = mg.compact(*mg.line(sub, slice(None), lat, width=width, lift=LIFT, dash=dash))
        if len(f):
            out.append((name, v, f, uv, mat))

    for k, r in enumerate(rows):
        a, b = float(r["s_from"]), float(r["s_to"])
        if r["item"] == "centre":
            sub = resample(sec, a, b)
            if sub.n < 2:
                continue
            mat, dash = CENTRE[r["value"]]
            add(f"dress_centre_{k:03d}", sub, np.zeros(sub.n), dash, mat)
            if r["value"] == "orange_flanked":
                for sgn, tag in ((1, "L"), (-1, "R")):
                    add(f"dress_flank{tag}_{k:03d}", sub, np.full(sub.n, sgn * FLANK_OFFSET), FLANK_DASH, "dress_line_white")
        elif r["item"] == "edge":
            for j, (x, y, g) in enumerate(_pieces(rows, a, b)):
                sub = resample(sec, x, y)
                if sub.n < 2:
                    continue
                for sgn, tag in ((1, "L"), (-1, "R")):
                    hw = sub.hw_left if sgn > 0 else sub.hw_right
                    add(f"dress_edge{tag}_{k:03d}_{j}", sub, sgn * (hw - EDGE_INSET - g), EDGE_DASH[r["value"]], "dress_line_white")
        elif r["item"] == "red":            # 片側の車線の赤い舗装（滑り止め）。線より下に。分かれる所の中は split_red で
            gaps = _gaps(rows)
            a = max([a] + [g1 for g0, g1 in gaps if g0 <= a < g1])
            sub = resample(sec, a, b)
            if sub.n < 2:
                continue
            sgn = 1 if r["side"] == "L" else -1
            hw = sub.hw_left if sgn > 0 else sub.hw_right
            v, f, uv = mg.compact(*mg.line(sub, slice(None), sgn * hw / 2, width=float(np.mean(hw)) - 0.1, lift=LIFT - 0.004))
            if len(f):
                out.append((f"dress_red_{k:03d}", v, f, uv, "dress_red"))
        elif r["item"] == "green":
          for j, (x, y, _) in enumerate(_pieces(rows, a, b)):
            sub = resample(sec, x, y)
            if sub.n < 2:
                continue
            g = float(r["value"])
            for sgn, tag in ((1, "L"), (-1, "R")):
                hw = sub.hw_left if sgn > 0 else sub.hw_right
                inner = hw - EDGE_INSET - g + WIDTH / 2          # 外側線の外から道の端まで
                add(f"dress_green{tag}_{k:03d}_{j}", sub, sgn * (inner + hw) / 2, None, "dress_green", width=float(np.mean(hw - inner)))
    return out


def hidden_line_meshes(manifest_dir, cl, s_range, near_m=40.0):
    """隠す今の線（外側線・中央線）の名前と、隠す範囲の s（S3 より外にはみ出した分も引き直すため）。"""
    manifest_dir = Path(manifest_dir)
    m = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    tree = cKDTree(np.c_[cl["x"], cl["y"]])
    names, lo, hi = [], s_range[1], s_range[0]
    for name, e in m["meshes"].items():
        if not name.startswith(("LINES_s1_edge", "LINES_track_centre")):
            continue
        v = np.load(manifest_dir / e["file"])["verts"]
        d, i = tree.query(v[:, :2])
        s = cl["s"][i][d < near_m]
        if len(s) and s.min() <= s_range[1] and s.max() >= s_range[0]:
            names += [name, f"{name}_back"]
            lo, hi = min(lo, s.min()), max(hi, s.max())
    return names, (lo, hi)


# ---------- 路面の標示（ひし形・数字） ----------

DIAMOND_BEFORE = (30.0, 50.0)       # 横断歩道の手前 m（信号のない横断歩道。2026-10-19 ストリートビューで全部この位置）
DIAMOND_SIZE = (1.5, 5.0)           # 幅・長さ m
TEXT_SIZE = (1.4, 2.6)


def diamond_spots(sec, pavements):
    """[(s, 車線 "L"=上り|"R"=下り)]。信号（停止線）のある横断歩道には描かない。"""
    stops = [float(r["s_from"]) for r in pavements if r["kind"].startswith("stop_line")]
    out = []
    for r in pavements:
        if r["kind"] != "crosswalk":
            continue
        a, b = float(r["s_from"]), float(r["s_to"])
        if any(a - 6 <= x <= b + 6 for x in stops):
            continue
        for d in DIAMOND_BEFORE:
            if a - d >= sec.s[0]:
                out.append((a - d, "L"))
            if b + d <= sec.s[-1]:
                out.append((b + d, "R"))
    return out


def _lane_quad(sec, s, side, size, flip):
    """車線の中央に、進む向きにそろえた四角（uv の v が進む向き）。"""
    from .crosssection import surface_height
    i = int(np.clip(np.searchsorted(sec.s, s), 0, sec.n - 1))
    sgn = 1 if side == "L" else -1
    hw = sec.hw_left[i] if sgn > 0 else sec.hw_right[i]
    lat = sgn * hw / 2
    h = sec.heading[i]
    t = np.array([np.cos(h), np.sin(h)]) * (1 if side == "L" else -1)   # 下りは向きが逆
    n = np.array([-t[1], t[0]])
    c = np.array([sec.x[i], sec.y[i]]) + np.array([-np.sin(h), np.cos(h)]) * lat
    w, l = size
    pts = [c - t * l / 2 - n * w / 2, c - t * l / 2 + n * w / 2, c + t * l / 2 + n * w / 2, c + t * l / 2 - n * w / 2]
    # 四隅それぞれの路面の高さ（坂・横断勾配に沿わせる）
    tree = cKDTree(np.c_[sec.x, sec.y])
    nr = np.array([-np.sin(sec.heading), np.cos(sec.heading)]).T
    v = []
    for p in pts:
        j = tree.query(p)[1]
        lj = float(np.dot(p - np.array([sec.x[j], sec.y[j]]), nr[j]))
        sj = sec.s[j] + float(np.dot(p - np.array([sec.x[j], sec.y[j]]), [np.cos(sec.heading[j]), np.sin(sec.heading[j])]))
        zj = np.interp(sj, sec.s, sec.z)
        v.append((p[0], p[1], float(surface_height(lj, zj, sec.slope[j], sec.crown[j])) + LIFT))
    v = np.array(v)
    return v, np.array([(0, 1, 2, 3)]), np.array([(0, 0), (1, 0), (1, 1), (0, 1)], float)


def marking_meshes(sec, diamonds, rows):
    """ひし形と、表の text 行（数字）→ [(名前, verts, faces, uvs, 材質)]。"""
    out = []
    parts = [_lane_quad(sec, s, side, DIAMOND_SIZE, False) for s, side in diamonds]
    if parts:
        off = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        out.append(("dress_diamonds", np.concatenate([p[0] for p in parts]),
                    np.concatenate([p[1] + o for p, o in zip(parts, off)]), np.concatenate([p[2] for p in parts]),
                    "dress_diamond"))
    for k, r in enumerate(x for x in rows if x["item"] == "text"):
        v, f, uv = _lane_quad(sec, float(r["s_from"]), r["side"], TEXT_SIZE, False)
        out.append((f"dress_text_{k:02d}", v, f, uv, f"dress_text_{r['value']}"))
    return out


# ---------- 島で道が分かれる所（split）の線を引き直す ----------

SPLIT_OUTER = 0.55      # 外側線の中心：路面の外周から内側へ（その外に緑の路側帯）
SPLIT_GREEN = (0.05, 0.47)
SPLIT_ISLAND = 0.6      # 島のまわりの破線：島から
ZEBRA_EVERY, ZEBRA_W = 1.5, 0.45


def _dash(line, on, off):
    """線を (on, off) m の破線に切る。"""
    from shapely.ops import substring
    out, t, L = [], 0.0, line.length
    while t < L:
        out.append(substring(line, t, min(L, t + on)))
        t += on + off
    return out


def band_polygon(sec, s_from, s_to, lat_a, lat_b):
    """中心線の s_from〜s_to、横 lat_a〜lat_b（左が正）の帯の多角形。"""
    from shapely.geometry import Polygon
    sub = resample(sec, s_from, s_to, 0.5)
    nx, ny = -np.sin(sub.heading), np.cos(sub.heading)
    a = np.c_[sub.x + nx * lat_a, sub.y + ny * lat_a]
    b = np.c_[sub.x + nx * lat_b, sub.y + ny * lat_b]
    return Polygon(np.r_[a, b[::-1]]).buffer(0)


def split_lines(sp, z_fn, keep_out=None) -> list:
    """split.Split（島・路面・導流帯）から、なめらかな外側線（実線）＋緑の路側帯、島のまわりの破線、導流帯のしまを作る。
    今の線（LINES_split_white）は、路面の多角形の縁をそのままたどるのでガタガタで、ところどころ切れて見えた。"""
    import math

    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    from .town import _polys, grid_drape

    cw = sp.carriageway
    ends = sp._ends()
    smooth = cw.buffer(1.5, join_style="round").buffer(-1.5, join_style="round").intersection(cw.buffer(0.01))
    white, green = [], []

    def lines_of(g):
        return [g] if g.geom_type == "LineString" else [x for x in getattr(g, "geoms", []) if x.geom_type == "LineString"]

    # 外側線と緑：外周（島の穴は除く）
    for g in _polys(smooth):
        ring = Polygon(g.exterior)
        edge = LineString(ring.buffer(-SPLIT_OUTER).exterior.coords) if not ring.buffer(-SPLIT_OUTER).is_empty else None
        if edge is not None:
            for piece in lines_of(edge.difference(ends.buffer(0.5)).difference(sp.zebra.buffer(0.3))):
                white.append(piece.buffer(0.075, cap_style="flat"))
        green.append(ring.buffer(-SPLIT_GREEN[0]).difference(ring.buffer(-SPLIT_GREEN[1])).difference(ends.buffer(0.5)))
    # 島のまわり：白の実線だけ（ストリートビュー s=3400 上り・下り。緑は外側だけ）
    isl = sp.island
    ring = LineString(isl.buffer(SPLIT_ISLAND).exterior.coords)
    for piece in lines_of(ring.difference(sp.zebra.buffer(0.3))):
        white.append(piece.buffer(0.075, cap_style="flat"))
    # 導流帯：外枠と斜めのしま
    for g in _polys(sp.zebra):
        if g.area < 2.0:
            continue
        white.append(LineString(g.exterior.coords).buffer(0.075))
        c = g.centroid
        R = math.hypot(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1])
        best = min(sp.lines.values(), key=lambda ln: ln.distance(c))
        a = best.project(c); p0, p1 = best.interpolate(max(0, a - 1)), best.interpolate(a + 1)
        h = math.atan2(p1.y - p0.y, p1.x - p0.x) + math.radians(45)
        d = np.array([math.cos(h), math.sin(h)]); n = np.array([-d[1], d[0]])
        for t in np.arange(-R, R, ZEBRA_EVERY):
            q = np.array([c.x, c.y]) + n * t
            white.append(LineString([q - d * R, q + d * R]).buffer(ZEBRA_W / 2, cap_style="flat").intersection(g))
    out = []
    W = unary_union(white).intersection(cw).difference(isl)
    if keep_out is not None:            # 線を引かない所（ユーザーの指示で消した所）
        W = W.difference(keep_out)
    G = unary_union(green).intersection(cw).difference(isl.buffer(SPLIT_ISLAND + 0.3)).difference(W)
    for k, (v, f, uv) in enumerate(grid_drape(W, z_fn, LIFT)):
        out.append((f"dress_split_white_{k:02d}", v, f, uv, "dress_line_white"))
    for k, (v, f, uv) in enumerate(grid_drape(G, z_fn, LIFT - 0.002)):
        out.append((f"dress_split_green_{k:02d}", v, f, uv, "dress_green"))
    return out


def split_red(sp, sec, rows, z_fn) -> list:
    """分かれる所（split の窓）の中の、表の red 行の赤い舗装（路面の多角形に合わせ、split の赤とは重ねない）。"""
    from shapely.geometry import LineString
    from .town import grid_drape
    out = []
    for k, r in enumerate(x for x in rows if x["item"] == "red"):
        a, b = max(float(r["s_from"]), sp.main_skip[0]), min(float(r["s_to"]), sp.main_skip[1])
        if b - a < 1:
            continue
        sub = resample(sec, a, b)
        sgn = 1 if r["side"] == "L" else -1
        hw = sub.hw_left if sgn > 0 else sub.hw_right
        nx, ny = -np.sin(sub.heading), np.cos(sub.heading)
        mid = np.c_[sub.x, sub.y]
        from shapely.geometry import Polygon
        if sp.red is not None and not sp.red.is_empty:
            # 分かれた片側の道（split の red＝下り側の道）のうち、この s の範囲：中心線から片側 15 m の帯で切る
            band = Polygon(np.r_[mid, np.c_[sub.x + nx * sgn * 15, sub.y + ny * sgn * 15][::-1]]).buffer(0)
            poly = band.intersection(sp.red).intersection(sp.carriageway)
        else:
            edge = np.c_[sub.x + nx * sgn * hw, sub.y + ny * sgn * hw]
            poly = Polygon(np.r_[mid, edge[::-1]]).buffer(0).intersection(sp.carriageway)
        for j, (v, f, uv) in enumerate(grid_drape(poly, z_fn, LIFT - 0.004)):
            out.append((f"dress_split_red_{k}_{j}", v, f, uv, "dress_red"))
    return out


def zebra_parts(sp, sec, s_max, lat_max):
    """split の導流帯のうち、重心が s_max より手前で、中心線から lat_max より右（負）にあるもの。"""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from .town import _polys
    tree = cKDTree(np.c_[sec.x, sec.y])
    out = []
    for g in _polys(sp.zebra):
        c = g.centroid
        i = tree.query([c.x, c.y])[1]
        h = sec.heading[i]
        lat = -(c.x - sec.x[i]) * np.sin(h) + (c.y - sec.y[i]) * np.cos(h)
        if sec.s[i] < s_max and lat < lat_max:
            out.append(g)
    return unary_union(out) if out else Polygon()


def split_centre(sp, sec, z_fn) -> list:
    """分かれる所の前後のオレンジの中央線：地図（OSM）の2車線の道の線に沿って、窓の端から分かれる点まで引き、
    そこから導流帯の先端まで伸ばす（空中写真：線は道のカーブに沿って導流帯の先につながる）。"""
    from shapely.geometry import LineString
    from .town import _polys, grid_drape
    tree = cKDTree(np.c_[sec.x, sec.y])
    big = max(_polys(sp.zebra), key=lambda g: g.area)
    zv = np.asarray(big.exterior.coords)
    zs = sec.s[tree.query(zv)[1]]
    tips = {"south": zv[np.argmin(zs)], "north": zv[np.argmax(zs)]}
    out = []
    for k, ln in sp.lines.items():
        if k in sp.singles:
            continue
        seg = ln.intersection(sp.window)
        if seg.is_empty or seg.geom_type != "LineString":
            continue
        c = np.asarray(seg.coords)[:, :2]
        cs = sec.s[tree.query(c)[1]]
        if cs[0] > cs[-1]:
            c, cs = c[::-1], cs[::-1]
        south = cs.mean() < np.mean([zs.min(), zs.max()])
        tip = tips["south" if south else "north"]
        pts = np.r_[c, [tip]] if south else np.r_[[tip], c]
        poly = LineString(pts).buffer(0.075, cap_style="flat").intersection(sp.carriageway)
        for j, (v, f, uv) in enumerate(grid_drape(poly, z_fn, LIFT)):
            out.append((f"dress_split_orange_{k}_{j}", v, f, uv, "dress_line_orange"))
    return out
