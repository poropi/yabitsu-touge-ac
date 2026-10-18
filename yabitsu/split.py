"""県道70号が島をはさんで2本に分かれる所（s3 の 3414m、神社の鳥居の島）を、地図（OSM）の道の線どおりに作る。

- 路面：その範囲の県道70号の way を、それぞれの幅（2車線は約7m、1車線は約3.6m）で線の位置どおりに広げて1枚にする。
  2本の車線が離れていく所（島の前後）のすき間は導流帯（白いしま模様）として舗装する。島は地図の形のまま。
- 範囲：島の前後の、本線の s の窓（s0〜s1）。窓の中は本線の帯を作らず（assembly.add_road の skip）、
  窓の端で帯と段差なくつなぐ（路面の高さは、一番近い本線の点の路面の高さを横へ延ばしたもの）。
- 線：オレンジの中央線は2車線の所だけ、白い外側線は路面の外周と島のまわり、赤い舗装は spec の red の way の1車線の所。

data/manual/s3_split.json：{"<名前>": {"widths": {"<way id>": 幅}, "island": [[lat, lon], ...], "red": ["<way id>"],
  "objects": [島の上の物（branches.island_objects の形）]}}"""
import math

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from . import geo, osm, site
from .branches import ISLAND_H, _island_wall, island_objects
from .crosssection import GUTTER_WIDTH, surface_height
from .town import grid_drape, _lines, _polys

FILL_R = 20.0          # 分かれる所・合わさる所の、路面の縁をまっすぐつなぐ範囲
MARGIN_M = 25.0        # 窓は、分かれる所・合わさる所からこの距離だけ外まで
WINDOW_HALF = 25.0     # 窓の横の広さ（本線の中心から片側）
EDGE_INSET, LINE_W, LIFT = 0.3, 0.15, 0.005
ZEBRA_EVERY, ZEBRA_W = 1.5, 0.45
SHOULDER = GUTTER_WIDTH   # 路面の外周のコンクリートの帯（帯の側溝と同じ幅）


def _xy(ll):
    a = np.asarray(ll, float).reshape(-1, 2)
    x, y = geo.to_local(a[:, 0], a[:, 1])
    return np.c_[x, y]


def _way_line(w):
    p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
    return LineString(_xy(p))


class SplitGround:
    """路面の高さ：一番近い本線の点の、路面の横断面の高さ（道幅の外は縁の高さのまま横へ）。"""
    def __init__(self, sec):
        self.sec = sec
        self.tree = cKDTree(np.c_[sec.x, sec.y])

    def z(self, x, y):
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        sec = self.sec
        _, i = self.tree.query(np.c_[x.ravel(), y.ravel()])
        h = sec.heading[i]
        lat = -(x.ravel() - sec.x[i]) * np.sin(h) + (y.ravel() - sec.y[i]) * np.cos(h)
        lat = np.clip(lat, -sec.hw_right[i], sec.hw_left[i])
        return surface_height(lat, sec.z[i], sec.slope[i], sec.crown[i]).reshape(np.shape(x))


class Split:
    def __init__(self, spec, elements, sec, sampler=None):
        self.spec, self.sec = spec, sec
        self.ground = SplitGround(sec)
        widths = {str(k): float(v) for k, v in spec["widths"].items()}
        ways = [w for w in osm.route70_ways(elements) if str(w.get("id")) in widths]
        self.lines = {str(w["id"]): _way_line(w) for w in ways}
        self.island = orient(Polygon(_xy(spec["island"])).buffer(0), 1.0)
        main = LineString(np.c_[sec.x, sec.y])
        # 1車線の way（分かれている所）の両はし＝分かれる所・合わさる所
        singles = [k for k in self.lines if widths[k] < 5.0]
        ends = [main.project(Point(q)) for k in singles for q in (self.lines[k].coords[0], self.lines[k].coords[-1])]
        self.main_skip = (min(ends) - MARGIN_M, max(ends) + MARGIN_M)
        self.window = self._window(*self.main_skip)
        lanes = {k: self.lines[k].buffer(widths[k] / 2, cap_style="flat").intersection(self.window) for k in self.lines}
        self.lanes = lanes
        one = unary_union([lanes[k] for k in singles])
        # 2本の車線の間のすき間（島を除く）＝導流帯
        hull = one.convex_hull
        self.zebra = hull.difference(one).difference(self.island).intersection(self.window)
        # 分かれる所・合わさる所のまわり（FILL_R）は、2車線の幅から1車線の2本へ、路面の縁をまっすぐつなぐ
        allr = unary_union(list(lanes.values()))
        q_ends = [q for k in singles for q in (self.lines[k].coords[0], self.lines[k].coords[-1])]
        fills = [allr.intersection(Point(q).buffer(FILL_R)).convex_hull for q in q_ends]
        self.carriageway = unary_union(list(lanes.values()) + [self.zebra] + fills).intersection(self.window).difference(self.island).buffer(0)
        red_ids = [str(k) for k in spec.get("red", [])]
        self.red = unary_union([lanes[k] for k in red_ids]).difference(
            unary_union([lanes[k] for k in lanes if k not in red_ids])).difference(self.island)   # 赤い車線だけ（重なる所は除く）
        self.singles = singles

    def _window(self, s0, s1):
        """本線の s0〜s1 の、本線の向きに直角に切った帯（横 ±WINDOW_HALF）。帯の端と同じ所で切れる。"""
        sec = self.sec   # 本線の帯が終わる行（s0 の手前）から、また始まる行（s1 の先）まで。帯との間にすき間を作らない
        i0 = max(int(np.searchsorted(sec.s, s0, side="right")) - 1, 0); i1 = min(int(np.searchsorted(sec.s, s1, side="left")), sec.n - 1)
        m = slice(i0, i1 + 1); self.end_rows = (i0, i1)
        nx, ny = -np.sin(sec.heading[m]), np.cos(sec.heading[m])
        L = np.c_[sec.x[m] + WINDOW_HALF * nx, sec.y[m] + WINDOW_HALF * ny]
        R = np.c_[sec.x[m] - WINDOW_HALF * nx, sec.y[m] - WINDOW_HALF * ny]
        return Polygon(np.r_[L, R[::-1]]).buffer(0)

    def area(self):
        """ほかの部品（分かれ道・景色の道・地形の合わせ）と重ねない範囲。"""
        return self.carriageway.union(self.island).buffer(SHOULDER)

    def _ends(self):
        """窓の両はしの切り口（本線の帯とつながる所）。外側線と路肩はここで切る。"""
        out = []
        for i in self.end_rows:
            h = self.sec.heading[i]
            n = np.array([-math.sin(h), math.cos(h)]); p = np.array([self.sec.x[i], self.sec.y[i]])
            out.append(LineString([p - n * WINDOW_HALF, p + n * WINDOW_HALF]).buffer(0.4))
        return unary_union(out)

    def meshes(self):
        """(名前, 頂点, 面, UV, 材質, 描画するか)"""
        z = self.ground.z
        out = []
        grey = self.carriageway.difference(self.red)
        for k, (v, f, uv) in enumerate(grid_drape(grey, z, 0.0)):
            out.append((f"1ROAD_split_{k:02d}", v, f, uv, "asphalt", True))
        for k, (v, f, uv) in enumerate(grid_drape(self.red.intersection(self.carriageway), z, 0.0)):
            out.append((f"1ROAD_split_red_{k:02d}", v, f, uv, "paint_red", True))
        # 路肩（帯の側溝と同じ幅のコンクリート）：路面の外周の外、窓の両はしは除く
        ends = self._ends()
        shoulder = self.carriageway.buffer(SHOULDER).difference(self.carriageway).difference(self.island.buffer(0.05)).difference(ends)
        shoulder = shoulder.intersection(self.window)
        for k, (v, f, uv) in enumerate(grid_drape(shoulder, lambda x, y: z(x, y) - 0.02, 0.0)):
            out.append((f"1ROAD_split_shoulder_{k:02d}", v, f, uv, "concrete", True))
        out += self._lines(ends)
        out += self._island()
        return out

    def _lines(self, ends):
        z = self.ground.z
        polys = []
        # 白い外側線：路面の外周（窓の両はしを除く）と、島のまわり
        inner = self.carriageway.buffer(-EDGE_INSET)
        for g in _polys(inner):
            for ring in [g.exterior] + list(g.interiors):
                ln = LineString(ring.coords).difference(ends)
                for piece in _lines(ln):
                    polys.append(piece.buffer(LINE_W / 2, cap_style="flat"))
        # 導流帯：外枠と、斜めのしま
        hatch = []
        for g in _polys(self.zebra):
            if g.area < 2.0:
                continue
            hatch.append(LineString(g.exterior.coords).buffer(LINE_W / 2))
            c = g.centroid; R = math.hypot(*(np.array(g.bounds[2:]) - g.bounds[:2]))
            h = self.sec.heading[int(np.argmin(np.hypot(self.sec.x - c.x, self.sec.y - c.y)))] + math.radians(45)
            d = np.array([math.cos(h), math.sin(h)]); n = np.array([-d[1], d[0]])
            for t in np.arange(-R, R, ZEBRA_EVERY):
                p = np.array([c.x, c.y]) + n * t
                hatch.append(LineString([p - d * R, p + d * R]).buffer(ZEBRA_W / 2, cap_style="flat").intersection(g))
        white = unary_union(polys + hatch).intersection(self.carriageway)
        out = [(f"LINES_split_white_{k:02d}", v, f, uv, "line_white", True) for k, (v, f, uv) in enumerate(grid_drape(white, z, LIFT))]
        # オレンジの中央線：2車線の way の線の上、導流帯と島から離れた所
        orange = []
        for k, ln in self.lines.items():
            if k in self.singles:
                continue
            seg = ln.intersection(self.window).difference(self.zebra.buffer(1.0)).difference(self.island.buffer(1.0))
            for piece in _lines(seg):
                if piece.length > 1.0:
                    orange.append(piece.buffer(LINE_W / 2, cap_style="flat"))
        if orange:
            g = unary_union(orange).intersection(self.carriageway)
            out += [(f"LINES_split_orange_{k:02d}", v, f, uv, "line_orange", True) for k, (v, f, uv) in enumerate(grid_drape(g, z, LIFT))]
        return out

    def _island(self):
        z = self.ground.z
        out = [(f"split_island_{k:02d}", v, f, uv, "gravel", True) for k, (v, f, uv) in enumerate(grid_drape(self.island, z, ISLAND_H, cell=1.0))]
        p = np.asarray(self.island.exterior.segmentize(1.0).coords); zg = z(p[:, 0], p[:, 1])
        vs, fs = [], []
        for i in range(len(p) - 1):   # 縁石（外向き）
            b = len(vs)
            vs += [[*p[i], zg[i] - 0.1], [*p[i + 1], zg[i + 1] - 0.1], [*p[i + 1], zg[i + 1] + ISLAND_H], [*p[i], zg[i] + ISLAND_H]]
            fs.append([b + 1, b, b + 3, b + 2])
        v = np.array(vs, float)
        out.append(("split_island_curb", v, np.array(fs, np.int32), v[:, :2], "concrete", True))
        out.append(("1WALL_split_island_block", *_island_wall(self.island, z), "concrete", False))
        out += island_objects(self.spec.get("objects", []), lambda x, y: z(x, y) + ISLAND_H, "split")
        return out

    def town_override(self):
        """terrain.build_near の town に渡す (地面の高さ, 範囲, 舗装)：路面の下と路肩の外 3m まで地形を路面に合わせる。"""
        from .start_lot import road_formation
        region = self.area().buffer(3.0)
        s0, s1 = self.main_skip   # 窓の先の本線の帯（路面＋側溝）の下も、地形を路面の下へ下げる
        road = road_formation(self.sec, max(s0 - 15.0, 0.0), min(s1 + 15.0, float(self.sec.s[-1])))
        return (self.ground, region, self.area().union(road.intersection(region)))
