"""本線の分かれ道。入口から STUB_LENGTH_M までを OSM の道の線どおりに作り、入口を見えない壁でふさぐ
（工事中の柵の見た目は town_objects.barricade_meshes を mouth の線に置く）。

道の高さは、入口で本線の縁の高さ、そこから先は地理院の標高へなめらかに移す。地形を合わせるため、
SEG_M ごとに区切った傾いた平面（terrain.carve の pad の "plane"）にする。区切りの線の上では前後の平面の高さがそろう。"""
import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union

from . import geo, osm, site
from .crosssection import GUTTER_WIDTH
from .start_lot import road_formation

CAR_TYPES = {"primary", "secondary", "tertiary", "unclassified", "residential", "service", "living_street", "track",
             "primary_link", "secondary_link", "tertiary_link", "trunk", "trunk_link"}
DEFAULT_WIDTH = {"trunk": 7.0, "primary": 6.5, "secondary": 6.5, "tertiary": 5.5, "unclassified": 4.5,
                 "residential": 4.0, "living_street": 4.0, "service": 3.0, "track": 3.0}
STUB_LENGTH_M = 40.0
SEG_M = 10.0
MOUTH_SETBACK_M = 0.3   # 入口の壁は、本線の側溝の外側の縁からこれだけ奥
MOUTH_OVERHANG_M = 2.0  # 入口の壁は、道の両端からこれだけはみ出す（斜めに抜けられないように）
BLEND_M = 15.0          # 入口から、この長さで本線の高さから地面の高さへ移す
CW_LEN, CW_BAR, CW_EVERY = 4.0, 0.45, 0.9   # 分かれ道の横断歩道：長さ（道の向き）、帯の幅、帯の間隔


def _width(tags):
    try:
        return float(str(tags.get("width", "")).split()[0])
    except (ValueError, IndexError):
        hw = tags.get("highway", "")
        return DEFAULT_WIDTH.get(hw.replace("_link", ""), 4.0)


def _line(w):
    p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
    x, y = geo.to_local(p[:, 0], p[:, 1])
    return LineString(np.c_[x, y])


def town_exit_s(sec, center_xy, radius):
    """本線が、center_xy を中心とする半径 radius の円を（始まりから）出る所の s。"""
    d = np.hypot(sec.x - center_xy[0], sec.y - center_xy[1])
    out = np.nonzero(d > radius)[0]
    return float(sec.s[out[0]]) if len(out) else float(sec.s[-1])


@dataclass
class Stub:
    osm_id: int
    axis: LineString        # 入口（側溝の外側の縁）から先の道の中心線
    width: float
    surface: Polygon
    mouth: LineString
    z_knots: np.ndarray     # axis の上の SEG_M ごとの高さ
    t_knots: np.ndarray

    def _segments(self):
        """(多角形, 平面) の並び。平面は区切りの両端の高さを通り、道を横切る向きには傾かない。"""
        out = []
        L = self.axis.length
        for a, b, za, zb in zip(self.t_knots[:-1], self.t_knots[1:], self.z_knots[:-1], self.z_knots[1:]):
            pa = np.array(self.axis.interpolate(a).coords[0]); pb = np.array(self.axis.interpolate(b).coords[0])
            d = pb - pa; n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            u = d / n
            g = (zb - za) / n
            # 区切りの線は、区切りの点で道を横切る向き（隣どうしで同じ線を共有する）
            cut_a = self._cut(a); cut_b = self._cut(b)
            box = Polygon([*cut_a.coords, *cut_b.coords[::-1]]).buffer(0)
            piece = self.surface.intersection(box)
            if piece.is_empty or piece.area < 0.01:
                continue
            plane = [float(pa[0]), float(pa[1]), float(za), float(g * u[0]), float(g * u[1])]
            out.append((site.largest(piece) if piece.geom_type != "Polygon" else piece, plane))
            if b >= L:
                break
        return out

    def _cut(self, t):
        """axis の t の所で、道を横切る線（道の幅より十分長い）。向きは前後の区切りで同じになるよう、その点の接線で決める。"""
        L = self.axis.length
        p = np.array(self.axis.interpolate(min(t, L)).coords[0])
        q = np.array(self.axis.interpolate(min(max(t + 0.5, 0.5), L)).coords[0])
        r = np.array(self.axis.interpolate(max(min(t - 0.5, L - 0.5), 0.0)).coords[0])
        d = q - r; d = d / max(np.linalg.norm(d), 1e-9)
        nrm = np.array([-d[1], d[0]]) * (self.width + 10.0)
        if t <= 0:  # 始まりは道の外側へ少し出しておく（側溝の縁の上を切る）
            p = p - d * 0.5
        if t >= L:
            p = p + d * 0.5
        return LineString([p - nrm, p + nrm])

    def mesh(self):
        parts = [site.draped(poly, lambda x, y, pl=pl: site.plane_z(pl, x, y)) for poly, pl in self._segments()]
        vs, fs, us, base = [], [], [], 0
        for v, f, uv in parts:
            vs.append(v); fs.append(f + base); us.append(uv); base += len(v)
        return np.concatenate(vs), np.concatenate(fs), np.concatenate(us)

    def pads(self):
        return [{"name": f"side_stub_{self.osm_id}_{k}", "role": "flat", "kind": "stub",
                 "polygon": np.asarray(poly.exterior.coords).tolist(), "z": float(site.plane_z(pl, *poly.centroid.coords[0])),
                 "plane": pl} for k, (poly, pl) in enumerate(self._segments())]

    def z_at(self, x, y):
        """路面の高さ：axis に下ろした位置の、区切りの高さを直線で補う。"""
        t = self.axis.project(shapely.Point(x, y))
        return float(np.interp(t, self.t_knots, self.z_knots))

    def ground_z(self, x, y):
        """z_at を配列で（柵などを入口の路面に置くため）。"""
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        return np.array([self.z_at(a, b) for a, b in zip(x.ravel(), y.ravel())]).reshape(x.shape)

    def crosswalk_area(self, p):
        """分かれ道の入口から横断歩道（点 p を中心に crosswalk_mesh で描く所）の奥の端までの範囲。柵はこの外へ下げる。"""
        t = self.axis.project(shapely.Point(*p))
        seg = substring(self.axis, 0.0, min(t + CW_LEN / 2, self.axis.length))
        return seg.buffer(self.width / 2, cap_style="flat")

    def crosswalk_mesh(self, p):
        """横断歩道：点 p（道の上）を中心に、道の向きに長さ CW_LEN の白い帯を、道幅いっぱいに CW_EVERY おきに並べる。"""
        t = self.axis.project(shapely.Point(*p))
        c = np.array(self.axis.interpolate(t).coords[0])
        a = np.array(self.axis.interpolate(max(t - 1.0, 0)).coords[0]); b = np.array(self.axis.interpolate(min(t + 1.0, self.axis.length)).coords[0])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-d[1], d[0]])
        vs, fs = [], []
        for off in np.arange(-self.width / 2 + 0.4 + CW_BAR / 2, self.width / 2 - 0.4 - CW_BAR / 2 + 1e-9, CW_EVERY):
            q = c + n * off
            corners = [q - d * CW_LEN / 2 - n * CW_BAR / 2, q - d * CW_LEN / 2 + n * CW_BAR / 2,
                       q + d * CW_LEN / 2 + n * CW_BAR / 2, q + d * CW_LEN / 2 - n * CW_BAR / 2]
            k = len(vs)
            vs += [[*e, self.z_at(*e) + 0.01] for e in corners]
            fs.append([k, k + 1, k + 2, k + 3])
        v = np.array(vs, float)
        f = np.array(fs, np.int32)
        if (v[f[0, 1]] - v[f[0, 0]])[0] * (v[f[0, 2]] - v[f[0, 0]])[1] - (v[f[0, 1]] - v[f[0, 0]])[1] * (v[f[0, 2]] - v[f[0, 0]])[0] < 0:
            f = f[:, ::-1]   # 上向きに
        return v, f, v[:, :2]

    def end_wall(self, height=3.0):
        """分かれ道の奥の終わりを横切る見えない壁（両面）。柵を外して奥まで走っても、その先へは出られない。"""
        e = np.array(self.axis.coords[-1]); t = self.axis.length
        p = np.array(self.axis.interpolate(max(t - 1.0, 0.0)).coords[0])
        d = (e - p) / max(np.linalg.norm(e - p), 1e-9); n = np.array([-d[1], d[0]]); half = self.width / 2 + MOUTH_OVERHANG_M
        a, b = e - n * half, e + n * half; z = float(self.z_knots[-1])
        v = np.array([[*a, z - 1.0], [*b, z - 1.0], [*b, z + height], [*a, z + height]], float)
        return v, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), v[:, :2] / 2.0

    def wall(self, height=3.0):
        """入口をふさぐ見えない壁（両面）。"""
        (x0, y0), (x1, y1) = self.mouth.coords
        za, zb = self.z_at(x0, y0), self.z_at(x1, y1)   # 入口（奥へ下げた所）の路面の高さ
        v = np.array([[x0, y0, za - 0.5], [x1, y1, zb - 0.5], [x1, y1, zb + height], [x0, y0, za + height]], float)
        f = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32)
        return v, f, v[:, :2] / 2.0


def stubs(elements, sec, sampler, keep_out, s_min=0.0, length_m=STUB_LENGTH_M, unproject=geo.from_local):
    """本線に s_min より先でつながる分かれ道（OSM の車道。県道70号・歩道を除く）を、入口から length_m まで切り出す。
    keep_out（ピット・広場の多角形）にかかる道は作らない。"""
    form = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
    r70 = {id(w) for w in osm.route70_ways(elements)}
    out, ends = [], None
    near_form = form.buffer(0.5); shapely.prepare(near_form)   # 1回だけ作る（道の数だけ作ると遅い）
    for w in elements:
        t = w.get("tags", {})
        if w.get("type") != "way" or id(w) in r70 or t.get("highway") not in CAR_TYPES or len(w.get("geometry", [])) < 2:
            continue
        if t.get("bridge", "no") != "no" or t.get("tunnel", "no") != "no":
            continue
        line = _line(w)
        if not near_form.intersects(line):
            continue
        rest = line.difference(form)
        pieces = list(rest.geoms) if rest.geom_type.startswith("Multi") else [rest]
        for pc in pieces:
            if pc.is_empty or pc.length < 1.0:
                continue
            a, b = np.array(pc.coords[0]), np.array(pc.coords[-1])
            from shapely.geometry import Point
            da, db = form.distance(Point(a)), form.distance(Point(b))
            if min(da, db) > 0.3:
                continue
            if db < da:
                pc = LineString(list(pc.coords)[::-1])
            start = np.array(pc.coords[0])
            i = int(np.argmin(np.hypot(sec.x - start[0], sec.y - start[1])))
            if sec.s[i] < s_min:
                continue
            axis = substring(pc, 0.0, min(length_m, pc.length))
            if length_m > STUB_LENGTH_M:   # 長い分かれ道（S4）：道の続きをたどり、本線へ戻ってくる手前で終える（S1〜S3 の 40m は今のまま）
                if pc.length < length_m:
                    ends = ends if ends is not None else _ends_index(elements, r70)
                    pc = _follow(pc, w, ends, length_m)
                axis = _stop_before(substring(pc, 0.0, min(length_m, pc.length)), form, _width(t))
            width = _width(t)
            if any(axis.buffer(width / 2).intersects(Polygon(k) if not isinstance(k, Polygon) else k) for k in keep_out):
                continue
            # 入口の手前へ少し延ばしてから本線の範囲を引く（斜めに交わるとき、本線との間にすき間を残さない）
            p1 = np.array(axis.interpolate(min(1.0, axis.length)).coords[0])
            d = (p1 - start) / max(np.linalg.norm(p1 - start), 1e-9)
            ext = LineString([start - d * (width + 2.0), *axis.coords])
            surf = ext.buffer(width / 2, cap_style="flat").difference(form)
            surf = site.largest(surf) if surf.geom_type != "Polygon" else surf
            if surf.is_empty or surf.area < 1.0:
                continue
            n = np.array([-d[1], d[0]])
            m0 = start + d * MOUTH_SETBACK_M
            half = width / 2 + MOUTH_OVERHANG_M
            mouth = LineString([m0 - n * half, m0 + n * half])
            # 高さ：入口は本線の縁の高さ、BLEND_M で地面へ
            hw_side = 1 if (-(start[0] - sec.x[i]) * math.sin(sec.heading[i]) + (start[1] - sec.y[i]) * math.cos(sec.heading[i])) > 0 else -1
            z_edge = float(sec.edge_z(hw_side)[i])
            tk = np.r_[np.arange(0.0, axis.length, SEG_M), axis.length]
            pts = np.array([axis.interpolate(v).coords[0] for v in tk])
            zg = np.asarray(sampler.sample(*unproject(pts[:, 0], pts[:, 1])), float)
            zg = np.where(np.isnan(zg), z_edge, zg)
            wgt = np.clip(tk / BLEND_M, 0.0, 1.0); wgt = wgt * wgt * (3 - 2 * wgt)
            zk = (1 - wgt) * z_edge + wgt * zg
            out.append(Stub(int(w.get("id", 0)), axis, width, surf, mouth, zk, tk))
    return out


def _ends_index(elements, r70):
    """車道の way の両端 → [(way, 端から始まる座標)]（_follow で道の続きを速く探すため）。"""
    idx = {}
    for w in elements:
        tg = w.get("tags", {})
        if w.get("type") != "way" or id(w) in r70 or tg.get("highway") not in CAR_TYPES or len(w.get("geometry", [])) < 2:
            continue
        c = np.asarray(_line(w).coords)
        for cc in (c, c[::-1]):
            idx.setdefault((round(cc[0][0]), round(cc[0][1])), []).append((w, cc))
    return idx


def _follow(pc, w0, ends, length_m):
    """pc の奥の端につながる車道の way を、長さ length_m になるまで（分かれ道が3本以上の所は、向きが一番まっすぐな方へ）たどる。"""
    used = {id(w0)}; coords = list(pc.coords)
    while LineString(coords).length < length_m:
        end = np.array(coords[-1]); prev = np.array(coords[-2]); d0 = (end - prev) / max(np.linalg.norm(end - prev), 1e-9)
        best = None
        cands = [c for dx in (-1, 0, 1) for dy in (-1, 0, 1) for c in ends.get((round(end[0]) + dx, round(end[1]) + dy), [])]
        for w, cc in cands:
            if id(w) in used:
                continue
            if True:
                if np.hypot(*(cc[0] - end)) < 0.5:
                    d1 = (cc[1] - cc[0]) / max(np.linalg.norm(cc[1] - cc[0]), 1e-9)
                    if best is None or d0 @ d1 > best[0]:
                        best = (float(d0 @ d1), w, cc)
        if best is None:
            break
        used.add(id(best[1])); coords += [tuple(q) for q in best[2][1:]]
    return LineString(coords)


def _stop_before(axis, form, width):
    """入口から出たあと本線の路面（＋側溝）へまた近づく道は、近づく手前（道幅の半分＋CLEAR_M＋1m）で終える。"""
    near = width / 2 + CLEAR_M + 1.0
    for t in np.arange(10.0, axis.length, 1.0):
        if form.distance(Point(axis.interpolate(t))) < near:
            return substring(axis, 0.0, max(t - 1.0, 1.0))
    return axis


ROADSIDE_FILES = ("osm_nagoki.json", "osm_s3_roadside.json", "osm_track_roads.json", "osm_shintomei.json", "osm_s4_roads.json")


def load_elements(raw_dir=None):
    """道ぞいの OSM（町・S3・S2/S1）をまとめて読む。同じ way は1つにする。"""
    import json
    from . import config
    seen, out = set(), []
    for f in ROADSIDE_FILES:
        p = (raw_dir or config.DATA_RAW) / f
        if not p.exists():
            raise FileNotFoundError(f"{p} がありません（scripts.track_01_fetch で取る）")
        for e in json.loads(p.read_text(encoding="utf-8"))["elements"]:
            k = (e.get("type"), e.get("id"))
            if k not in seen:
                seen.add(k); out.append(e)
    return out


def stub_meshes(stub_list):
    """(名前, 頂点, 面, UV, 材質, 描画するか)。路面は走れる面（入口の壁の手前で車が落ちないように）。"""
    out = []
    for k, st in enumerate(stub_list):
        out.append((f"1ROAD_side_stub_{k:03d}", *st.mesh(), "asphalt", True))
        out.append((f"1WALL_side_block_{k:03d}", *st.wall(), "concrete", False))
        if st.axis.length > STUB_LENGTH_M + 1.0:   # 長い分かれ道（S4）：奥の終わりにも見えない壁（入口の柵・壁とは別の部品）
            out.append((f"1WALL_side_end_{k:03d}", *st.end_wall(), "concrete", False))
    return out


KEEP_OPEN_KINDS = ("lot", "pass", "site", "plaza", "branch")
NOT_OPENINGS = ("_house", "_shrine", "side_stub")   # 家・お堂の土台は「ふさがない場所」ではない（家の前の道もふさぐ）


BARRICADE_TYPES = {"primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "track",
                   "primary_link", "secondary_link", "tertiary_link", "trunk", "trunk_link"}
BARRICADE_MIN_M = 30.0   # これより短い道（家の前で行き止まり）には柵を置かない


def needs_barricade(st, ways):
    """工事中の柵を置くのは、車で入っていける分かれ道（公道で、長さ BARRICADE_MIN_M 以上）の入口だけ。
    路地・私道・駐車場の出入口（service）や短い行き止まりは、見えない壁だけにする。"""
    w = ways.get(st.osm_id)
    if w is None:
        return False
    if w.get("tags", {}).get("highway") not in BARRICADE_TYPES:
        return False
    return max(_line(w).length, st.axis.length) >= BARRICADE_MIN_M   # 続きの道までたどって作った長さも見る


CLEAR_M = 1.0   # 柵と見えない壁は、本線の路面（側溝まで）と歩道の外側から、これだけ離す


def set_back_mouth(st, sec, walks=()):
    """入口（柵・見えない壁）を、本線の路面＋側溝と歩道（walks）から CLEAR_M 離れた所まで、分かれ道の奥へ下げる。
    路面は切らない（本線から入口までつながる）。柵の幅は、本線・歩道にかからない長さに切る。"""
    area = unary_union([road_formation(sec, 0.0, float(sec.s[-1]))] + list(walks)).buffer(CLEAR_M)
    rest = st.axis.difference(area)
    pieces = [rest] if rest.geom_type == "LineString" else [g for g in getattr(rest, "geoms", []) if g.geom_type == "LineString"]
    if not pieces:
        return st
    piece = min(pieces, key=lambda g: st.axis.project(Point(g.coords[0])))
    t0 = st.axis.project(Point(piece.coords[0]))
    a = np.array(st.axis.interpolate(t0).coords[0]); b = np.array(st.axis.interpolate(min(t0 + 1.0, st.axis.length)).coords[0])
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-d[1], d[0]])
    half = st.width / 2 + MOUTH_OVERHANG_M
    ln = LineString([a - n * half, a + n * half]).difference(area.buffer(-0.5))
    ln = max(([ln] if ln.geom_type == "LineString" else [g for g in getattr(ln, "geoms", []) if g.geom_type == "LineString"]),
             key=lambda g: g.length, default=None)
    if ln is not None and ln.length > st.width * 0.6:
        st.mouth = LineString([ln.coords[0], ln.coords[-1]])
    return st


def branch_gaps(stubs_, sec, margin=1.0):
    """分かれ道の入口で、本線の道ぞいの壁・ガードレールなどを切る s の範囲（左右ごと）：{"L": [(s0, s1)], "R": [...]}
    入口（柵の線、set_back_mouth で道ぞいの壁の線あたりに下げたもの）の中心の s から、
    分かれ道の幅を本線と交わる角度で割った長さ＋margin だけ前後。入口の先で本線ぞいに曲がる道でも、開いている所だけを切る。"""
    out = {"L": [], "R": []}
    for st in stubs_:
        c = st.mouth.interpolate(0.5, normalized=True)
        # s は本線の行の s で測る（地図上の長さ main.project とは、坂のぶんずれる。S1 の奥で 38m）
        i = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y))); h = sec.heading[i]
        lat = -(c.x - sec.x[i]) * np.sin(h) + (c.y - sec.y[i]) * np.cos(h)
        s = float(sec.s[i]) + (c.x - sec.x[i]) * np.cos(h) + (c.y - sec.y[i]) * np.sin(h)
        t = st.axis.project(c)
        a = np.array(st.axis.interpolate(max(t - 1.0, 0.0)).coords[0]); b = np.array(st.axis.interpolate(min(t + 1.0, st.axis.length)).coords[0])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        sin = max(abs(d[0] * np.sin(h) - d[1] * np.cos(h)), 0.3)   # 本線の向きとの角度の sin（浅すぎる時は 0.3 で止める）
        half = st.width / 2 / sin + margin
        out["L" if lat > 0 else "R"].append((s - half, s + half))
    return out


def move_mouth_out(st, area):
    """分かれ道の入口（壁・柵）が area（ほかの道の路面）の中にあるとき、道の線が area を出る所へ移し、路面も area の外だけにする。"""
    if not st.mouth.intersects(area):
        g = st.surface.difference(area)
        if not g.is_empty:
            st.surface = site.largest(g) if g.geom_type != "Polygon" else g
        return st
    rest = st.axis.difference(area.buffer(0.3))
    pieces = [rest] if rest.geom_type == "LineString" else [g for g in getattr(rest, "geoms", []) if g.geom_type == "LineString"]
    if not pieces:
        st.surface = Polygon()
        return st
    piece = min(pieces, key=lambda g: st.axis.project(Point(g.coords[0])))
    t0 = st.axis.project(Point(piece.coords[0]))
    a = np.array(st.axis.interpolate(t0).coords[0]); b = np.array(st.axis.interpolate(min(t0 + 1.0, st.axis.length)).coords[0])
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-d[1], d[0]])
    half = st.width / 2 + MOUTH_OVERHANG_M
    st.mouth = LineString([a - n * half, a + n * half])
    g = st.surface.difference(area)
    st.surface = (site.largest(g) if g.geom_type != "Polygon" else g) if not g.is_empty else Polygon()
    return st


S4_BRANCH_M = 200.0   # S4 の分かれ道は、柵の先もこの長さまで道として作る（柵を外せば走れるように）


def s4_start(sec):
    """S4 の始まり（峠の先 config.S1_END）の本線の s。"""
    from . import config
    x, y = geo.to_local(*config.S1_END)
    return float(sec.s[int(np.argmin(np.hypot(sec.x - float(x), sec.y - float(y))))])


def _mouth_s(sec, st):
    p = np.array(st.axis.coords[0])
    return float(sec.s[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))])


def for_track(sec, pads, sampler=None, exclude=None):
    """本線全体の分かれ道（町の範囲の中は除く）。ピット・広場・駐車場の入口はふさがない。
    exclude（町の範囲など）と重なる所は路面から削る（町の道と二重にならないように）。"""
    from . import config, gsi
    cx, cy = geo.to_local(*config.NAGOKI)
    s_min = town_exit_s(sec, (float(cx), float(cy)), config.TOWN_RADIUS_M)
    keep = [Polygon(p["polygon"]) for p in pads
            if (p.get("kind") in KEEP_OPEN_KINDS or p.get("role") in ("start", "mid", "end"))
            and not any(k in p.get("name", "") for k in NOT_OPENINGS)]
    el = load_elements(); smp = sampler or gsi.default_sampler()
    s4 = s4_start(sec)   # S4（峠の先）の分かれ道は S4_BRANCH_M、それより手前は今まで通り STUB_LENGTH_M
    out = [st for st in stubs(el, sec, smp, keep, s_min=s_min) if _mouth_s(sec, st) < s4]
    out += [st for st in stubs(el, sec, smp, keep, s_min=s4, length_m=S4_BRANCH_M)]
    if exclude is not None:
        for st in out:
            move_mouth_out(st, exclude)
    return [st for st in out if not st.surface.is_empty and st.surface.area >= 1.0]
