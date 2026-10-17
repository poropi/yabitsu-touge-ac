"""町の範囲（名古木交差点から半径 200m）の道。OSM の道の線から車道・歩道・縁石の多角形を作り、
町の地面の高さ（town_ground）にかぶせる。本線（県道70号）は帯の作り方と同じ形（路面＋側溝の幅）を多角形にしたもので、
町の範囲の円を本線が出る所（main_skip の終わり）から先は、今までどおり帯で作る。

道幅・歩道の幅・路面の表示・封鎖の位置は data/manual/s3_nagoki.json に緯度経度で書く：
  roads: {"<OSM way id>" | "main": {"width_m", "sidewalk_l", "sidewalk_r"}}（書いていない道は ROAD_DEFAULTS）
  markings: [{"kind": stop_line | crosswalk | lane_line | centre_line | arrow | text, "points": [[lat, lon], ...], ...}]
"""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import substring, unary_union

from . import geo, osm
from .crosssection import GUTTER_WIDTH
from .side_roads import town_exit_s

ROAD_DEFAULTS = {"trunk": (7.0, 2.5), "primary": (6.5, 2.0), "secondary": (6.5, 2.0), "secondary_link": (4.5, 1.5),
                 "tertiary": (5.5, 1.5), "unclassified": (4.5, 0.0), "residential": (4.0, 0.0), "service": (3.5, 0.0),
                 "trunk_link": (4.5, 1.5), "primary_link": (4.5, 1.5), "tertiary_link": (4.0, 1.0), "living_street": (4.0, 0.0)}
CURB_H = 0.15
MARK_LIFT = 0.005
CELL = 1.0
MAX_VERTS = 60000
CROSSWALK_W, BAR_W, BAR_EVERY = 4.0, 0.45, 0.9
STOP_W = 0.45
ARROW_LEN, SHAFT_W, HEAD_W, HEAD_LEN = 5.0, 0.3, 1.0, 1.5
TEXT_LEN, TEXT_W = 5.0, 1.6
COLORS = {"white": "line_white", "yellow": "line_orange", "orange": "line_orange"}
# 自動で置く表示と封鎖（OSM の横断歩道・信号から）
R_BOX = 30.0           # 交差点の中心からこの距離に入る道を、交差点の「腕」とする
R_ARM = 15.0           # 横断歩道のない腕は、中心からこの距離で封鎖する
CLOSURE_GAP_M = 3.0    # 横断歩道の外側の縁から封鎖の線まで
STOP_GAP_M = 2.0       # 横断歩道の外側の縁から停止線まで
SIGNAL_NEAR_M = 10.0   # 横断歩道からこの距離の中に信号があれば、停止線を引く
CLOSURE_OVERHANG = 0.5
HOLE_FILL_M2 = 400.0   # 走れる範囲の中の、これより小さい穴（中央の島など）は埋める
LINED = {"trunk", "primary", "secondary"}  # 線を引く道（連絡路には引かない）
EDGE_INSET = 0.5       # 外側線は車道の端からこれだけ内側


def _xy(ll):
    a = np.asarray(ll, float).reshape(-1, 2)
    x, y = geo.to_local(a[:, 0], a[:, 1])
    return np.c_[x, y]


def _way_line(w):
    p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
    return LineString(_xy(p))


def _ll(p):
    lat, lon = geo.from_local(float(p[0]), float(p[1]))
    return [round(float(lat), 8), round(float(lon), 8)]


def _lines(g):
    return [g] if g.geom_type == "LineString" else [q for q in getattr(g, "geoms", []) if q.geom_type == "LineString"]


def _points(g):
    return [g] if g.geom_type == "Point" else [q for q in getattr(g, "geoms", []) if q.geom_type == "Point"]


def _tangent(ln, pt):
    t0 = ln.project(pt)
    q0 = np.array(ln.interpolate(max(t0 - 1.0, 0)).coords[0]); q1 = np.array(ln.interpolate(min(t0 + 1.0, ln.length)).coords[0])
    return (q1 - q0) / max(np.linalg.norm(q1 - q0), 1e-9)


def _closing(g, r):
    return g.buffer(r, join_style="round").buffer(-r, join_style="round") if r > 0 else g


def _polys(g):
    if g.is_empty:
        return []
    return [g] if g.geom_type == "Polygon" else [q for q in getattr(g, "geoms", []) if q.geom_type == "Polygon"]


def side_band(sec, s0, s1, side, inner, outer):
    """本線の s0〜s1 の、横の位置 inner〜outer（道の中心から、側 side）の帯。"""
    m = (sec.s >= s0) & (sec.s <= s1)
    nx, ny = -np.sin(sec.heading[m]), np.cos(sec.heading[m])
    a = np.c_[sec.x[m] + side * inner * nx, sec.y[m] + side * inner * ny]
    b = np.c_[sec.x[m] + side * outer * nx, sec.y[m] + side * outer * ny]
    return Polygon(np.r_[a, b[::-1]]).buffer(0)


def main_formation(sec, s0, s1):
    """本線の路面＋側溝（帯の作り方と同じ形）。"""
    hw_l = sec.hw_left + GUTTER_WIDTH; hw_r = sec.hw_right + GUTTER_WIDTH
    m = (sec.s >= s0) & (sec.s <= s1)
    nx, ny = -np.sin(sec.heading[m]), np.cos(sec.heading[m])
    left = np.c_[sec.x[m] + hw_l[m] * nx, sec.y[m] + hw_l[m] * ny]
    right = np.c_[sec.x[m] - hw_r[m] * nx, sec.y[m] - hw_r[m] * ny]
    return Polygon(np.r_[left, right[::-1]]).buffer(0)


def grid_drape(poly, z_fn, dz=0.0, cell=CELL, uv_scale=4.0):
    """多角形を 1m 格子で切ってから三角形にし、高さ z_fn(x, y) + dz に置く（面は上向き）。頂点は MAX_VERTS ごとに分ける。"""
    out, vs, fs, base = [], [], [], 0
    for pg in _polys(poly):
        x0, y0, x1, y1 = pg.bounds
        gx = np.arange(np.floor(x0 / cell) * cell, x1, cell); gy = np.arange(np.floor(y0 / cell) * cell, y1, cell)
        X, Y = np.meshgrid(gx, gy)
        cells = shapely.box(X.ravel(), Y.ravel(), X.ravel() + cell, Y.ravel() + cell)
        hit = shapely.intersects(cells, pg)
        full = shapely.contains_properly(pg, cells) & hit
        for c in cells[full]:
            bx0, by0, bx1, by1 = c.bounds
            tri = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])
            vs.append(tri); fs.append(np.array([[0, 1, 2], [0, 2, 3]]) + base); base += 4
            if base > MAX_VERTS:
                out.append(_finish(vs, fs, z_fn, dz, uv_scale)); vs, fs, base = [], [], 0
        for piece in shapely.intersection(cells[hit & ~full], pg):
            for q in _polys(piece):
                if q.area < 1e-6:
                    continue
                for t in shapely.constrained_delaunay_triangles(q).geoms:
                    p = np.asarray(t.exterior.coords)[:3]
                    if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
                        p = p[::-1]
                    vs.append(p); fs.append(np.array([[0, 1, 2]]) + base); base += 3
            if base > MAX_VERTS:
                out.append(_finish(vs, fs, z_fn, dz, uv_scale)); vs, fs, base = [], [], 0
    if vs:
        out.append(_finish(vs, fs, z_fn, dz, uv_scale))
    return out


def _finish(vs, fs, z_fn, dz, uv_scale):
    v = np.concatenate(vs); f = np.concatenate(fs).astype(np.int32)
    return np.c_[v, z_fn(v[:, 0], v[:, 1]) + dz], f, v / uv_scale


def _rect(c, d, along, across):
    n = np.array([-d[1], d[0]])
    a, b = d * along / 2, n * across / 2
    return Polygon([c - a - b, c + a - b, c + a + b, c - a + b])


def _arrow(tail, head, kind):
    d = head - tail; L = np.linalg.norm(d); d = d / max(L, 1e-9); n = np.array([-d[1], d[0]])
    parts = []

    def shaft_head(p0, u, length):
        tip = p0 + u * length
        neck = tip - u * HEAD_LEN
        w = np.array([-u[1], u[0]])
        parts.append(LineString([p0, neck]).buffer(SHAFT_W / 2, cap_style="flat"))
        parts.append(Polygon([neck - w * HEAD_W / 2, tip, neck + w * HEAD_W / 2]))

    if "straight" in kind:
        shaft_head(tail, d, L)
    if "left" in kind or "right" in kind:
        sgn = 1 if "left" in kind else -1
        mid = tail + d * L * 0.55
        parts.append(LineString([tail, mid]).buffer(SHAFT_W / 2, cap_style="flat"))
        u = (d + sgn * n) / math.sqrt(2)
        shaft_head(mid, u, L * 0.45)
    return unary_union(parts)


def marking_polygons(spec):
    """路面の表示を (多角形, 材質) の並びに（ローカル座標）。"""
    out = []
    for mk in spec.get("markings", []):
        kind = mk["kind"]; pts = _xy(mk["points"]); mat = COLORS.get(mk.get("color", "white"), "line_white")
        if kind == "crosswalk":
            a, b = pts[0], pts[-1]; L = float(np.linalg.norm(b - a)); d = (b - a) / max(L, 1e-9)
            w = float(mk.get("width_m", CROSSWALK_W))
            for t in np.arange(BAR_EVERY / 2, L - BAR_W / 2 + 1e-9, BAR_EVERY):
                out.append((_rect(a + d * t, d, BAR_W, w), "line_white"))
        elif kind == "stop_line":
            out.append((LineString(pts).buffer(float(mk.get("width_m", STOP_W)) / 2, cap_style="flat"), "line_white"))
        elif kind in ("lane_line", "centre_line"):
            line = LineString(pts); w = float(mk.get("width_m", 0.15)); dash = mk.get("dash")
            if dash:
                on, off = dash
                for t in np.arange(0.0, line.length, on + off):
                    out.append((substring(line, t, min(t + on, line.length)).buffer(w / 2, cap_style="flat"), mat))
            else:
                out.append((line.buffer(w / 2, cap_style="flat"), mat))
        elif kind == "arrow":
            out.append((_arrow(pts[0], pts[-1], mk.get("arrow", "straight")), "line_white"))
        elif kind == "text":
            continue  # 文字は texture を貼るので text_meshes で作る
        else:
            raise ValueError(f"路面の表示の kind が分かりません: {mk}")
    return out


class Town:
    def __init__(self, spec, osm_elements, sec, ground, center_xy, radius):
        self.spec, self.sec, self.ground = spec, sec, ground
        self.center = np.asarray(center_xy, float); self.radius = radius
        s_exit = town_exit_s(sec, center_xy, radius)
        self.main_skip = (0.0, s_exit)
        circle = Point(*self.center).buffer(radius, 128)
        main = main_formation(sec, 0.0, s_exit)
        beyond = main_formation(sec, s_exit, min(s_exit + 400.0, float(sec.s[-1]))).buffer(0.01)
        self.region = circle.union(main).difference(beyond)
        roads = spec.get("roads", {})
        r70 = {id(w) for w in osm.route70_ways(osm_elements)}
        cw, walks = [main], []
        self._roads = []  # (way, 中心線, 車道の幅, 左の歩道, 右の歩道)
        mw = roads.get("main", {})
        for side, key in ((1, "sidewalk_l"), (-1, "sidewalk_r")):
            if mw.get(key):
                hw = float(max(sec.hw_left.max(), sec.hw_right.max())) + GUTTER_WIDTH
                walks.append(side_band(sec, 0.0, s_exit, side, hw - 0.01, hw + float(mw[key])))
        for w in osm_elements:
            t = w.get("tags", {})
            hwy = t.get("highway")
            if w.get("type") != "way" or id(w) in r70 or hwy not in ROAD_DEFAULTS or len(w.get("geometry", [])) < 2:
                continue
            line = _way_line(w)
            if not line.intersects(circle):
                continue
            d_w, d_s = ROAD_DEFAULTS[hwy]
            r = roads.get(str(w.get("id")), {})
            width = float(r.get("width_m", d_w))
            cw.append(line.buffer(width / 2, cap_style="flat"))
            self._roads.append((w, line, width, float(r.get("sidewalk_l", d_s)), float(r.get("sidewalk_r", d_s))))
            for sgn, key in ((1, "sidewalk_l"), (-1, "sidewalk_r")):
                sw = float(r.get(key, d_s))
                if sw > 0:
                    outer = line.buffer(sgn * (width / 2 + sw), single_sided=True, cap_style="flat")
                    inner = line.buffer(sgn * width / 2, single_sided=True, cap_style="flat")
                    walks.append(outer.difference(inner))
        rad = float(spec.get("corner_radius_m", 6.0))
        carriage = _closing(unary_union(cw), rad).intersection(self.region).difference(beyond)
        self.carriageway = carriage.union(main.intersection(self.region)).buffer(0)
        both = _closing(unary_union([self.carriageway, *walks]), rad)
        self.sidewalk = both.intersection(self.region).difference(self.carriageway).difference(beyond).buffer(0)
        self.main_walk = max(float(mw.get("sidewalk_l", 0.0)), float(mw.get("sidewalk_r", 0.0)))
        self.markings = list(spec.get("markings", []))
        self.closures = list(spec.get("closures", []))
        self._signalised = []  # 信号のある横断歩道：(中心, 交差点から外への向き, 端 a, 端 b, 道幅)
        self._bus_stops = [e for e in osm_elements if e.get("type") == "node" and e.get("tags", {}).get("highway") == "bus_stop"
                           and np.hypot(*(_xy([[e["lat"], e["lon"]]])[0] - self.center)) <= radius]
        if spec.get("auto"):
            auto_m, auto_c = self._auto(osm_elements, s_exit)
            self.markings += auto_m
            self.closures += auto_c

    def _auto(self, elements, s_exit):
        """OSM の横断歩道（zebra）の位置に横断歩道、信号のある横断歩道の手前に停止線、
        ヤビツ方面以外の道の入口に封鎖の線を置く（緯度経度の dict で返す）。"""
        sec, c = self.sec, self.center
        main_line = LineString(np.c_[sec.x, sec.y][sec.s <= s_exit + 1.0])
        nodes = [e for e in elements if e.get("type") == "node"]
        sig = np.array([_xy([[e["lat"], e["lon"]]])[0] for e in nodes
                        if e.get("tags", {}).get("highway") == "traffic_signals"]).reshape(-1, 2)
        hw_main = float(np.median(sec.hw_left[sec.s <= s_exit])) + GUTTER_WIDTH
        lines = [(w, ln, wd) for (w, ln, wd, sl, sr) in self._roads] + [(None, main_line, 2 * hw_main)]
        marks, crossings = [], []
        for e in nodes:
            tg = e.get("tags", {})
            if tg.get("highway") != "crossing" or tg.get("crossing:markings", "no") == "no" or tg.get("crossing") == "unmarked":
                continue
            p = _xy([[e["lat"], e["lon"]]])[0]
            if np.hypot(*(p - c)) > self.radius:
                continue
            way, ln, wd = min(lines, key=lambda q: q[1].distance(Point(p)))
            if ln.distance(Point(p)) > 1.5:
                continue
            d = _tangent(ln, Point(p)); n = np.array([-d[1], d[0]])
            on = np.array(ln.interpolate(ln.project(Point(p))).coords[0])   # 道の中心線の上へ寄せ、道幅いっぱいに
            cross = LineString([on - n * wd / 2, on + n * wd / 2]).intersection(self.carriageway)
            parts = [g for g in _lines(cross) if g.distance(Point(p)) < 0.5]
            if not parts:
                continue
            a, b = np.asarray(parts[0].coords)[[0, -1]]
            marks.append({"kind": "crosswalk", "points": [_ll(a), _ll(b)], "auto": True})
            away = d if np.dot(d, p - c) > 0 else -d
            crossings.append((way, p, away))
            if len(sig) and np.hypot(*(sig - p).T).min() <= SIGNAL_NEAR_M:
                self._signalised.append((p, away, a, b, wd))
                q = p + away * (CROSSWALK_W / 2 + STOP_GAP_M)
                left = np.array([away[1], -away[0]])   # 交差点へ向かう向き（-away）の左
                reach = max(float(np.dot(e - p, left)) for e in (a, b))
                marks.append({"kind": "stop_line", "points": [_ll(q), _ll(q + left * reach)], "auto": True})
        # 中央線（オレンジ）と外側線（白）：国道・県道だけ。交差点の中（中心から R_BOX）には引かない
        box = Point(*c).buffer(R_BOX, 128)
        major = [(str(w.get("id")), ln, wd, w["tags"].get("oneway") == "yes") for (w, ln, wd, sl, sr) in self._roads
                 if w["tags"].get("highway") in LINED]
        hw_main = float(np.median(sec.hw_left[sec.s <= s_exit]))
        major.append(("main", main_line, 2 * hw_main, False))
        # ほかの道の車道の中には外側線を引かない（同じ道が何本もの線で描かれている所、道が合わさる所）
        bodies = {str(w.get("id")): ln.buffer(wd / 2 - 0.4, cap_style="flat") for (w, ln, wd, sl, sr) in self._roads}
        bodies["main"] = main_line.buffer(hw_main - 0.4, cap_style="flat")
        for rid, ln, wd, oneway in major:
            others = unary_union([g for k, g in bodies.items() if k != rid])
            for piece in _lines(ln.difference(box)):
                if piece.length < 3.0:
                    continue
                if not oneway:
                    marks.append({"kind": "centre_line", "points": [_ll(q) for q in piece.coords], "color": "orange",
                                  "road": rid, "auto": True})
                for sgn in (1, -1):
                    off = piece.offset_curve(sgn * (wd / 2 - EDGE_INSET))
                    for o in _lines(off.intersection(self.carriageway).difference(others)):
                        if o.length >= 3.0:
                            marks.append({"kind": "lane_line", "points": [_ll(q) for q in o.coords], "color": "white",
                                          "road": rid, "auto": True})
        closures = []
        main_zone = main_formation(sec, 0.0, s_exit).buffer(self.main_walk)
        for (w, ln, wd, sl, sr) in self._roads:
            half = wd / 2 + max(sl, sr) + CLOSURE_OVERHANG
            if ln.distance(Point(*c)) <= R_BOX:   # 交差点の腕：横断歩道の外側、なければ R_ARM で
                dirs = [(away, float(np.dot(p - c, away)) + CROSSWALK_W / 2 + CLOSURE_GAP_M)
                        for way, p, away in crossings if way is w]
                for pt, u in self._arm_hits(ln, dirs):
                    n = np.array([-u[1], u[0]])
                    closures.append({"name": f"arm_{w.get('id')}", "a": _ll(pt - n * half), "b": _ll(pt + n * half),
                                     "barricade": True, "auto": True})
                continue
            for pc in _lines(ln.difference(main_zone)):   # 本線ぞいの横道：本線（＋歩道）の範囲を出た所
                if pc.length < 0.5:
                    continue
                for k, e in enumerate((np.array(pc.coords[0]), np.array(pc.coords[-1]))):
                    if main_zone.distance(Point(e)) > 0.05:
                        continue
                    other = np.array(pc.interpolate(min(1.0, pc.length) if k == 0 else max(pc.length - 1.0, 0)).coords[0])
                    u = (other - e) / max(np.linalg.norm(other - e), 1e-9); n = np.array([-u[1], u[0]])
                    m0 = e + u * 0.3
                    closures.append({"name": f"side_{w.get('id')}", "a": _ll(m0 - n * half), "b": _ll(m0 + n * half),
                                     "barricade": True, "auto": True})
        return marks, closures

    def drivable(self):
        """車で走れる範囲：(車道 ∪ 歩道 ∪ drivable_extra) を封鎖の線で切り、交差点の中心を含むかたまり。
        中の小さな穴（中央の島など、HOLE_FILL_M2 まで）は埋める（見えない壁を交差点の真ん中に立てない）。"""
        extra = [Polygon(_xy(e["points"])) for e in self.spec.get("drivable_extra", [])]
        area = unary_union([self.carriageway, self.sidewalk, *extra])
        cuts = unary_union([LineString(_xy([c["a"], c["b"]])).buffer(0.05) for c in self.closures]) if self.closures else None
        pieces = _polys(area.difference(cuts) if cuts is not None else area)
        c = Point(*self.center)
        best = min(pieces, key=lambda q: q.distance(c))
        holes = [h for h in best.interiors if Polygon(h).area > HOLE_FILL_M2]
        return Polygon(best.exterior, holes)

    def boundary_walls(self):
        """走れる範囲の外周のうち、封鎖の線と、本線の帯へ出る口を除いた線（見えない壁を立てる所）。"""
        d = self.drivable()
        sec, s1 = self.sec, self.main_skip[1]
        i = int(np.argmin(np.abs(sec.s - s1)))
        n = np.array([-np.sin(sec.heading[i]), np.cos(sec.heading[i])]); p = np.array([sec.x[i], sec.y[i]])
        reach = max(sec.hw_left[i], sec.hw_right[i]) + GUTTER_WIDTH + self.main_walk + 0.5
        mouth = LineString([p - n * reach, p + n * reach]).buffer(0.3)
        gaps = [mouth] + [LineString(_xy([c["a"], c["b"]])).buffer(0.06) for c in self.closures]  # 封鎖の線そのものにも壁を立てる
        rings = [LineString(d.exterior.coords)] + [LineString(h.coords) for h in d.interiors]
        rest = unary_union(rings).difference(unary_union(gaps))
        return [g for g in _lines(rest) if g.length > 0.05] if rest.geom_type != "LineString" else [rest]

    def signals(self):
        """信号機：spec の signals に、信号のある横断歩道ごとの自動の信号を足す。
        車用：交差点の向こう側の左の角に立て、腕を道の上へ伸ばし、交差点へ向かう車に向ける。歩行者用：横断歩道の両端に、向こう岸へ向けて。"""
        out = list(self.spec.get("signals", []))
        c = self.center
        bearing = lambda v: (math.degrees(math.atan2(v[0], v[1])) + 360.0) % 360.0
        for p, away, a, b, wd in self._signalised:
            v = -away                                   # 交差点へ向かう向き
            left = np.array([-v[1], v[0]])
            far = c + v * float(np.dot(p - c, away))     # 交差点の向こう側（同じ距離）
            pole = far + left * (wd / 2 + 1.5)
            out.append({"at": _ll(pole), "facing_deg": bearing(-v), "kind": "vehicle", "arm_m": wd / 2 + 0.5,
                        "arm_deg": bearing(-left), "height_m": 5.5, "auto": True})
            for e, o in ((a, b), (b, a)):
                u = (e - o) / max(np.linalg.norm(e - o), 1e-9)
                out.append({"at": _ll(e + u * 0.8), "facing_deg": bearing(-u), "kind": "pedestrian", "height_m": 2.6, "auto": True})
        return out

    def pads(self):
        """町のピット：spec の pits（[[lat, lon, 方位] ×4]）。走れる範囲の中にあること。"""
        pits = self.spec.get("pits", [])
        if not pits:
            return []
        pts = []
        for lat, lon, deg in pits:
            x, y = _xy([[lat, lon]])[0]
            if not self.drivable().buffer(0.01).contains(Point(x, y)):
                raise ValueError(f"町のピット {lat}, {lon} が走れる範囲の外にあります")
            z = float(self.ground.z(np.atleast_1d(x), np.atleast_1d(y))[0])
            pts.append([float(x), float(y), z, math.radians(90.0 - float(deg))])
        poly = unary_union([Point(p[0], p[1]).buffer(3.0) for p in pts]).convex_hull
        return [{"name": "nagoki_pits", "role": "flat", "kind": "site", "polygon": np.asarray(poly.exterior.coords).tolist(),
                 "z": float(np.mean([p[2] for p in pts])), "pit_points": pts}]

    def object_meshes(self):
        """封鎖の柵・見えない壁・信号機・小物（spec の objects）。(名前, 頂点, 面, UV, 材質, 描画するか)"""
        from . import town_objects as to
        d = self.drivable()
        out = []
        for k, c in enumerate(self.closures):
            a, b = _xy([c["a"], c["b"]])
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-u[1], u[0]])
            mid = (a + b) / 2
            front = n if d.contains(Point(*(mid + n * 1.0))) else -n   # 柵の手前＝走れる側
            if c.get("barricade", True):
                out += to.barricade_meshes(a, b, self.ground.z, front, k)
        lines = self.boundary_walls() + [LineString(_xy([c["a"], c["b"]])) for c in self.closures]
        out += to.wall_meshes(lines, self.ground.z)
        out += to.signal_meshes(self.signals(), self.ground.z)
        objects = list(self.spec.get("objects", []))
        for e in self._bus_stops:   # OSM のバス停：道の方を向ける
            p = _xy([[e["lat"], e["lon"]]])[0]
            q = np.array(self.carriageway.exterior.interpolate(self.carriageway.exterior.project(Point(*p))).coords[0])                 if self.carriageway.geom_type == "Polygon" else p
            v = q - p
            objects.append({"kind": "bus_stop", "points": [[e["lat"], e["lon"]]],
                            "facing_deg": (math.degrees(math.atan2(v[0], v[1])) + 360.0) % 360.0})
        if objects:
            out += to.object_meshes(objects, self.ground.z)
        return out

    def _arm_hits(self, ln, dirs):
        """腕の道の線と、中心からの距離 d の円の交わる所（腕の向きごとに1つ）。d は横断歩道があればその外、なければ R_ARM。"""
        c = self.center
        hits = []
        base = max(R_ARM, ln.distance(Point(*c)) + 3.0)   # 中心のそばを通るだけの道（連絡路など）は、その距離＋3m で
        for dist in sorted({base, *[d for _, d in dirs]}):
            for q in _points(ln.intersection(Point(*c).buffer(dist, 256).exterior)):
                pt = np.array(q.coords[0])
                v = (pt - c) / max(np.linalg.norm(pt - c), 1e-9)
                want = next((d for a, d in dirs if float(np.dot(a, v)) > 0.7), base)
                if abs(dist - want) < 1e-6:
                    hits.append((pt, _tangent(ln, q)))
        return hits

    def _curbs(self):
        """車道と歩道の境の縦の面（車道の側を表にする）。"""
        edge = orient(self.carriageway, 1.0) if self.carriageway.geom_type == "Polygon" else self.carriageway
        rings = []
        for pg in _polys(edge):
            pg = orient(pg, 1.0)
            rings += [LineString(pg.exterior.coords)] + [LineString(h.coords) for h in pg.interiors]
        touch = self.sidewalk.buffer(0.05)
        vs, fs = [], []
        for ring in rings:
            seg = ring.segmentize(CELL).intersection(touch)
            for ln in ([seg] if seg.geom_type == "LineString" else list(getattr(seg, "geoms", []))):
                if ln.geom_type != "LineString" or ln.length < 0.05:
                    continue
                p = np.asarray(ln.coords)
                zg = self.ground.z(p[:, 0], p[:, 1])
                for k in range(len(p) - 1):
                    b = len(vs)
                    # 外周は反時計回り（中が左）：左を向く面にするため、q → p の順
                    vs += [[*p[k + 1], zg[k + 1]], [*p[k], zg[k]], [*p[k], zg[k] + CURB_H], [*p[k + 1], zg[k + 1] + CURB_H]]
                    fs.append([b, b + 1, b + 2, b + 3])
        if not vs:
            return None
        v = np.asarray(vs, float)
        return v, np.asarray(fs, np.int32), np.c_[np.hypot(v[:, 0], v[:, 1]), v[:, 2]] / 1.0

    def meshes(self):
        """(名前, 頂点, 面, UV, 材質, 描画するか)"""
        out = []
        for k, (v, f, uv) in enumerate(grid_drape(self.carriageway, self.ground.z)):
            out.append((f"1ROAD_town_carriageway_{k:02d}", v, f, uv, "asphalt", True))
        for k, (v, f, uv) in enumerate(grid_drape(self.sidewalk, self.ground.z, CURB_H)):
            out.append((f"1ROAD_town_sidewalk_{k:02d}", v, f, uv, "concrete", True))
        c = self._curbs()
        if c is not None:
            out.append(("town_curb_00", *c, "concrete", True))
        by_mat = {}
        for poly, mat in marking_polygons({"markings": self.markings}):
            by_mat.setdefault(mat, []).append(poly)
        k = 0
        for mat, polys in sorted(by_mat.items()):
            for v, f, uv in grid_drape(unary_union(polys), self.ground.z, MARK_LIFT):
                out.append((f"LINES_town_{k:03d}", v, f, uv, mat, True)); k += 1
        for mk in self.markings:
            if mk["kind"] == "text":
                out.append((f"LINES_town_{k:03d}", *self._text(mk), "text_tomare", True)); k += 1
        return out

    def _text(self, mk):
        """「止まれ」：a（手前）→ b（奥）の向きに、TEXT_LEN×TEXT_W の四角に文字の絵を貼る（手前の字から読む）。"""
        a, b = _xy(mk["points"])[:2]
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9); n = np.array([-d[1], d[0]])
        c = [a + n * TEXT_W / 2, a - n * TEXT_W / 2, a - n * TEXT_W / 2 + d * TEXT_LEN, a + n * TEXT_W / 2 + d * TEXT_LEN]
        c = np.asarray(c)
        z = self.ground.z(c[:, 0], c[:, 1]) + MARK_LIFT * 2
        v = np.c_[c, z]
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        return v, np.array([[0, 1, 2, 3]], np.int32), uv
