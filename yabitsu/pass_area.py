"""ヤビツ峠（下りのスタート）まわり：わき道（ピット）、植え込みの島、ブロック舗装の広場、休憩所。

位置は緯度経度で書く（data/manual/s1_pass.json）。区間の距離 s には頼らないので、道が峠の先（宮ヶ瀬側）へ
延びても、そのまま使える。高さは、まわりの路面に合わせた1枚の傾いた平面にそろえる。
"""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from . import site
from .acobjects import Dummy
from .crosssection import GUTTER_WIDTH, surface_height
from .start_lot import Frame, _gable_ends, _gable_roof, _merge, box, road_formation

_xy = site.ll_to_xy
_largest = site.largest
EDGE_BLEND_M = 4.0  # 道の縁から、この距離で路面の高さを平面へ戻す


class Site:
    def __init__(self, spec, sec):
        self.spec = spec; self.sec = sec
        lane = _xy(spec["lane"]["points"]); self.w = w = spec["lane"]["width"]
        ends = [int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1]))) for p in (lane[0], lane[-1])]
        if ends[0] > ends[1]:  # 出口（道の下り側＝s が小さい方の合流口）から並べる
            lane, ends = lane[::-1], ends[::-1]
        i0, i1 = ends
        self.line = LineString(lane)

        self.plane = site.fit_plane(sec, lane.mean(0))

        road = road_formation(sec, s_min=sec.s[max(i0 - 40, 0)], s_max=sec.s[min(i1 + 40, sec.n - 1)])
        lane_band = self.line.buffer(w / 2, cap_style="flat", join_style="mitre")
        enclosed = Polygon(np.r_[lane, np.c_[sec.x[i0:i1 + 1], sec.y[i0:i1 + 1]][::-1]]).buffer(0)
        island = _largest(enclosed.difference(road).difference(self.line.buffer(w / 2 + 0.05)))
        lane_poly = _largest(lane_band.difference(road))
        if "lane_shape" in spec:     # わき道の外形を直接決める（地図の道の形）。島と本線を除いた所がわき道
            outline = Polygon(_xy(spec["lane_shape"]["points"])).buffer(0)
            island = _largest(Polygon(_xy(spec["island_shape"]["points"])).buffer(0).intersection(outline))
            lane_poly = _largest(outline.difference(road).difference(island))
        elif "island_shape" in spec:   # 島の外形を直接決める（ストリートビューの縁石）。島の外のすき間はわき道の舗装にする
            # 本線の縁では切らない（島の縁石は本線のアスファルトに直に接する。縁より内側に来る分は本線の上に載せる）
            island = _largest(Polygon(_xy(spec["island_shape"]["points"])).buffer(0).intersection(enclosed)
                              .difference(self.line.buffer(w / 2 + 0.05)))
            lane_poly = _largest(lane_band.union(enclosed).difference(road).difference(island))
        paved = lane_poly if "lane_shape" in spec else lane_band
        self.polys = {
            "lane": lane_poly,
            "island": island,
            "plaza": _largest(Polygon(_xy(spec["plaza"]["points"])).difference(paved).difference(road)),
        }
        self.flats = []   # 道の脇の平らな台（売店の前など）：(名前, 多角形, 高さ, 側面の材質)。高さは道の縁＋dz
        for fl in spec.get("flats", []):
            g = _largest(Polygon(_xy(fl["points"])).buffer(0).difference(road))
            c = g.centroid; i = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y)))   # 台の真ん中の横の道の縁の高さ
            lat = -(c.x - sec.x[i]) * np.sin(sec.heading[i]) + (c.y - sec.y[i]) * np.cos(sec.heading[i])
            self.flats.append((fl["name"], g, float(sec.edge_z(1 if lat > 0 else -1)[i]) + float(fl.get("dz", 0.25)),
                               fl.get("edge", "concrete")))
        if "sidewalk" in spec:   # わき道の外の縁の歩道（ベンチ・バス停）。地形もこの高さまで削る
            self.polys["sidewalk"] = _largest(Polygon(_xy(spec["sidewalk"]["points"])).buffer(0)
                                              .difference(paved).difference(road)
                                              .difference(self.polys["plaza"]))

        self.beds = []      # 駐車場・トイレ前の縁石つきの島：(名前, 多角形, 縁石の高さ, 材質, 高さの関数)
        self.occupied = {}  # 列の名前 → 車が止まっている枠の番号
        self.stalls = []    # 駐車枠の列：(名前, 原点, 列の向き u, 奥の向き n, 枠の幅, 奥行き, 台数, 車の向き)
        if pk := spec.get("parking"):   # 峠の駐車場（航空写真の形。本線から入れる走れる路面）
            lot = Polygon(_xy(pk["lot"])).buffer(0)
            c = lot.centroid; j = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y)))
            self.park_plane = self._edge_plane(lot)
            road_p = road_formation(sec, s_min=sec.s[max(j - 60, 0)], s_max=sec.s[min(j + 60, sec.n - 1)])
            near = road.union(road_p)
            taken = paved.union(island).union(self.polys["plaza"])
            for b in pk.get("beds", []):
                g = _largest(Polygon(_xy(b["points"])).buffer(0).difference(near).difference(taken))
                if g.is_empty or g.area < 0.05:   # 道に呑まれて残らない（形の指定の誤り）
                    raise ValueError(f"駐車場の植え込み {b['name']} が本線・わき道と重なって残りません")
                zf = self.surface_z if b.get("plane") == "pass" else self.park_z
                self.beds.append((b["name"], g, float(b.get("curb", 0.15)), b.get("surface", "ground"), zf))
                lot = lot.difference(g)
            beds = {n: g for n, g, *_ in self.beds}
            for k, b in enumerate(pk.get("beds", [])):   # cut：中に別の植え込みがある（れんがの角地の中の植え込み）
                if b.get("cut"):
                    n, g, *rest = self.beds[k]
                    for c_ in b["cut"]:
                        g = g.difference(beds[c_])
                    self.beds[k] = (n, _largest(g), *rest)
            self.polys["parking"] = _largest(lot.difference(near).difference(taken))
            for r in pk.get("stalls", []):
                o, t = _xy([r["origin"], r["toward"]])
                u = (t - o) / np.linalg.norm(t - o); n = np.array([-u[1], u[0]]) * (1 if r.get("side", "left") == "left" else -1)
                self.stalls.append((r["name"], o, u, n, float(r.get("width", 2.5)), float(r.get("depth", 5.0)), int(r["n"]),
                                    r.get("face", "out")))
                self.occupied[r["name"]] = set(r.get("occupied", range(int(r["n"]))))

    def _edge_plane(self, poly, step=1.0):
        """poly の中の各点に、いちばん近い本線の縁の高さを取り、それに合わせた平面 [cx, cy, z0, gx, gy]
        （本線の点だけで合わせると、道がほぼまっすぐなので横の傾きが決まらない。これなら道に沿った勾配だけで、横は水平に近い）。"""
        sec = self.sec; x0, y0, x1, y1 = poly.bounds
        X, Y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step)); X, Y = X.ravel(), Y.ravel()
        k = shapely.contains_xy(poly, X, Y); X, Y = X[k], Y[k]
        i = np.argmin(np.hypot(sec.x[:, None] - X, sec.y[:, None] - Y), axis=0); h = sec.heading[i]
        lat = -(X - sec.x[i]) * np.sin(h) + (Y - sec.y[i]) * np.cos(h)
        Z = surface_height(np.clip(lat, -sec.hw_right[i], sec.hw_left[i]), sec.z[i], sec.slope[i], sec.crown[i])
        c = np.array([X.mean(), Y.mean()])
        z0, gx, gy = np.linalg.lstsq(np.c_[np.ones(len(X)), X - c[0], Y - c[1]], Z, rcond=None)[0]
        return [float(c[0]), float(c[1]), float(z0), float(gx), float(gy)]

    def z(self, x, y):
        return site.plane_z(self.plane, x, y)

    def park_z(self, x, y):
        """駐車場の路面の高さ：本線の縁の高さから EDGE_BLEND_M で駐車場の平面（本線の駐車場の横の高さに合わせた面）へ。"""
        return self.surface_z(x, y, plane=self.park_plane)

    def top_z(self, x, y, reach=1.0):
        """峠の範囲の面（わき道・島・広場・歩道・台・植え込み・駐車場）の上面の高さ。どの面にも reach m 以内で届かなければ NaN。
        面からはみ出した点（面の縁に立てた物）は、いちばん近い面の高さ。小物・看板を面の上に立てるのに使う。"""
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        sp = self.spec; curb = sp.get("island_curb", 0.15)
        faces = [(self.polys["lane"], self.surface_z, 0.0), (self.polys["island"], self.surface_z, curb),
                 (self.polys["plaza"], self.surface_z, 0.0)]
        if "sidewalk" in self.polys:
            faces.append((self.polys["sidewalk"], self.surface_z, sp["sidewalk"].get("curb", 0.12)))
        faces += [(g, (lambda a, b, zf=zf: np.full(np.shape(a), zf)), 0.0) for _, g, zf, _ in self.flats]
        faces += [(g, zf, h) for _, g, h, _, zf in self.beds]          # 植え込みは先に（駐車場より上に載る）
        if "parking" in self.polys:
            faces.append((self.polys["parking"], self.park_z, 0.0))
        out = np.full(len(x), np.nan); best = np.full(len(x), np.inf)
        for g, zf, h in faces:
            d = shapely.distance(g, shapely.points(x, y))
            take = (d <= reach) & (d < best - 1e-6)
            if take.any():
                out[take] = np.asarray(zf(x[take], y[take]), float) + h; best[take] = d[take]
        return out

    def parked_cars(self):
        """止まっている車の位置（occupied の枠）：[(列の名前, 番号, x, y, z, 向き（東=0 の反時計回りの rad）)]。
        face=out は origin→toward の線の側（通路）を向き、in はその反対を向く。"""
        out = []
        for name, o, u, n, w, d, k, face in self.stalls:
            for i in sorted(self.occupied.get(name, ())):
                p = o + u * w * (i + 0.5) + n * d * 0.5
                fwd = -n if face == "out" else n
                out.append((name, i, float(p[0]), float(p[1]), float(self.park_z(*p)), math.atan2(fwd[1], fwd[0])))
        return out

    def surface_z(self, x, y, plane=None):
        """路面（わき道・広場・歩道・島）の高さ。道の縁で道の縁の高さに合わせ、EDGE_BLEND_M 離れると平面に戻る
        （1枚の平面のままだと、本線から入る所に段差が出る）。本線の上では本線の路面の高さ。"""
        shape = np.shape(x); x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)); sec = self.sec
        i = np.argmin(np.hypot(sec.x[:, None] - x, sec.y[:, None] - y), axis=0); h = sec.heading[i]
        lat = -(x - sec.x[i]) * np.sin(h) + (y - sec.y[i]) * np.cos(h)
        left = lat > 0
        d = np.abs(lat) - np.where(left, sec.hw_left[i], sec.hw_right[i]) - GUTTER_WIDTH
        edge = surface_height(np.clip(lat, -sec.hw_right[i], sec.hw_left[i]), sec.z[i], sec.slope[i], sec.crown[i])
        p = self.z(x, y) if plane is None else site.plane_z(plane, x, y)
        return (p + (edge - p) * np.clip(1.0 - d / EDGE_BLEND_M, 0.0, 1.0)).reshape(shape)

    def mouths(self, reach=0.5):
        """わき道と島が本線に接している区間 [(side, s0, s1)]。本線のこの区間は側溝をくぼませず、路肩の線を破線にする。"""
        sec = self.sec; out = []
        ring = np.vstack([np.asarray(self.polys[k].segmentize(0.5).exterior.coords) for k in ("lane", "island")])
        i = np.argmin(np.hypot(sec.x[:, None] - ring[:, 0], sec.y[:, None] - ring[:, 1]), axis=0); h = sec.heading[i]
        lat = -(ring[:, 0] - sec.x[i]) * np.sin(h) + (ring[:, 1] - sec.y[i]) * np.cos(h)
        d = np.abs(lat) - np.where(lat > 0, sec.hw_left[i], sec.hw_right[i]) - GUTTER_WIDTH
        for side in (1, -1):
            k = (d < reach) & (np.sign(lat) == side)
            if k.any():
                out.append((side, float(sec.s[i[k]].min()), float(sec.s[i[k]].max())))
        return out

    def _draped(self, poly, dz=0.0, uv_scale=4.0, z_fn=None):
        return site.draped(poly.segmentize(1.0) if z_fn else poly, z_fn or self.z, dz, uv_scale)

    def _curb(self, poly, h, z_fn=None, uv_m=None):
        """多角形のまわりの縁（下は 0.1m 埋める）。uv_m を渡すと模様をその大きさ（m）で縦横同じに貼る（石積みなど）。"""
        ring = np.asarray(orient(poly, 1.0).exterior.coords)  # 反時計回り：進む向きの右側が外
        parts = []
        for p0, p1 in zip(ring[:-1], ring[1:]):
            za, zb = (float((z_fn or self.z)(*q)) for q in (p0, p1)); L = float(np.hypot(*(p1 - p0)))
            v = np.array([[*p0, za - 0.1], [*p1, zb - 0.1], [*p1, zb + h], [*p0, za + h]])
            u1, v1 = (L, 0.25) if uv_m is None else (L / uv_m, (h + 0.1) / uv_m)
            parts.append((v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [u1, 0], [u1, v1], [0, v1]])))
        return _merge(parts)

    def pits(self):
        """2列×2台。先頭の列は指定の場所、2列目はそのうしろ。どれも出口（下り側の合流口）を向く。"""
        pi = self.spec["pits"]
        d0 = self.line.project(Point(_xy(pi["front"])[0]))
        out = []
        for row in range(2):
            d = d0 + row * pi["spacing"]
            p = np.array(self.line.interpolate(d).coords[0])
            t = np.array(self.line.interpolate(max(d - 0.5, 0)).coords[0]) - np.array(self.line.interpolate(d + 0.5).coords[0])
            t = t / np.linalg.norm(t); n = np.array([-t[1], t[0]]); h = math.atan2(t[1], t[0])
            for side in (1, -1):
                q = p + side * pi["lateral"] * n
                out.append((float(q[0]), float(q[1]), float(self.z(*q)), h))
        return [Dummy(f"AC_PIT_{i}", (x, y, z + 0.3), h) for i, (x, y, z, h) in enumerate(out)]

    def pads(self):
        """地形を削る平らな場所（傾いた平面）。わき道にはピットの場所も持たせる（role=end）。"""
        def pad(name, role, poly, **kw):
            c = poly.centroid
            return {"name": name, "role": role, "kind": "pass", "polygon": np.asarray(poly.exterior.coords).tolist(),
                    "z": float(self.z(c.x, c.y)), "plane": self.plane, **kw}
        pits = [(d.location[0], d.location[1], d.location[2] - 0.3, d.heading) for d in self.pits()]
        flats = [{"name": f"pass_{n}", "role": "flat", "kind": "pass", "polygon": np.asarray(g.exterior.coords).tolist(), "z": z,
                  "buildings_ok": True} for n, g, z, _ in self.flats]   # 台の上には建物が建つ（売店）
        return flats + [pad("pass_lane", "end", self.polys["lane"], pit_points=pits),
                pad("pass_island", "flat", self.polys["island"]),
                pad("pass_plaza", "flat", self.polys["plaza"])] + (
                [pad("pass_sidewalk", "flat", self.polys["sidewalk"])] if "sidewalk" in self.polys else []) + self._park_pads()

    def _park_pads(self):
        """駐車場と植え込みの台（駐車場の平面）。トイレ前の角地（plane=pass）は峠の平面。"""
        if "parking" not in self.polys:
            return []
        def pad(name, poly, plane):
            c = poly.centroid
            return {"name": name, "role": "flat", "kind": "pass", "polygon": np.asarray(poly.exterior.coords).tolist(),
                    "z": float(site.plane_z(plane, c.x, c.y)), "plane": plane}
        out = [pad("pass_parking", self.polys["parking"], self.park_plane)]
        for k, sh in enumerate(self.spec.get("buildings", [])):   # 建物の下も削る（まわりの舗装から地形の草が突き出ないように）
            f = Frame(_xy(sh["center"])[0], sh["axis_deg"]); L, W = sh["size"]
            out.append(pad(f"pass_building_{k}", f.rect((-L / 2, L / 2), (-W / 2, W / 2)), self.plane))
        for n, g, _, _, zf in self.beds:
            out.append(pad(f"pass_bed_{n}", g, self.plane if zf == self.surface_z else self.park_plane))
        return out

    def meshes(self):
        sp = self.spec; curb = sp.get("island_curb", 0.15)
        out = [("1ROAD_s1_pass_lane", *self._draped(self.polys["lane"], z_fn=self.surface_z), "asphalt", True),
               ("1GRASS_s1_pass_island", *self._draped(self.polys["island"], curb, z_fn=self.surface_z), "ground", True),
               ("s1_pass_curb", *self._curb(self.polys["island"], curb, self.surface_z), "concrete", True),
               ("1ROAD_s1_pass_plaza", *self._draped(self.polys["plaza"], z_fn=self.surface_z), "pavers", True)]
        if "sidewalk" in self.polys:
            h = sp["sidewalk"].get("curb", 0.12)
            out.append(("1ROAD_s1_pass_sidewalk", *self._draped(self.polys["sidewalk"], h, z_fn=self.surface_z), "pavers", True))
            if h > 0:   # 0 なら段差なし（わき道と同じ高さでつながる）
                out.append(("s1_pass_sidewalk_curb", *self._curb(self.polys["sidewalk"], h), "concrete", True))
        if fc := sp.get("island_fence"):   # 島の縁の内側に木の杭を並べ、杭の頭の少し下にロープを渡す
            ring = self.polys["island"].buffer(-fc["inset"]).exterior
            n = max(int(ring.length // fc["spacing"]), 3)
            pts = [np.array(ring.interpolate(k / n, normalized=True).coords[0]) for k in range(n)]
            parts = []
            for k, p in enumerate(pts):
                z = float(self.surface_z(*p)) + curb
                parts.append(box(Frame(p, 0.0), (-0.03, 0.03), (-0.03, 0.03), z - 0.1, z + fc["height"], 1.0))
                q = pts[(k + 1) % n]; zq = float(self.surface_z(*q)) + curb; zr = fc["height"] - 0.12
                rope = np.array([[*p, z + zr - 0.015], [*q, zq + zr - 0.015], [*q, zq + zr + 0.015], [*p, z + zr + 0.015]])
                parts.append((rope, np.array([[0, 1, 2, 3], [3, 2, 1, 0]]), np.array([[0, 0], [1, 0], [1, 0.05], [0, 0.05]])))
            out.append(("s1_pass_island_fence", *_merge(parts), "wood", True))
        for n, g, zf, edge in self.flats:   # 台：コンクリートの上面と、まわりの側面（edge の材質：石積みなど）
            flat = lambda x, y, zf=zf: np.full(np.shape(x), zf)
            out += [(f"1ROAD_s1_pass_{n}", *site.draped(g, flat, 0.0, 2.0), "concrete", True),
                    (f"s1_pass_{n}_edge", *self._curb(g, 0.9, lambda x, y, zf=zf: np.full(np.shape(x), zf - 0.9), uv_m=1.5), edge, True)]
        for k, sh in enumerate(sp.get("buildings") or [sp["shelter"]]):   # トイレの棟（切妻。長い辺 size[0] の向きが axis_deg）
            tag = "shelter" if k == 0 else f"shelter{k}"
            c = _xy(sh["center"])[0]; f = Frame(c, sh["axis_deg"])
            L, W = sh["size"]; a, b = (-L / 2, L / 2), (-W / 2, W / 2)
            z = float(min(self.surface_z(*f.xy(aa, bb)) for aa in a for bb in b))   # 四隅のいちばん低い地面
            wall = sh.get("wall", "wall_white")
            out.append((f"1WALL_s1_pass_{tag}", *box(f, a, b, z - 0.3, z + sh["wall_h"], sh.get("uv_m", 2.0)), wall, True))
            if base := sh.get("base_h"):  # 足元の白い腰壁（壁から少し外へ出す）
                e = 0.03
                out.append((f"s1_pass_{tag}_base", *box(f, (a[0] - e, a[1] + e), (b[0] - e, b[1] + e), z - 0.3, z + base, 2.0),
                            "wall_white", True))
            out.append((f"s1_pass_{tag}_gables", *_gable_ends(f, a, b, z + sh["wall_h"], sh["roof_h"]), wall, True))
            out.append((f"s1_pass_{tag}_roof", *_gable_roof(f, a, b, z + sh["wall_h"], sh["roof_h"], sh["eaves"]),
                        sh.get("roof", "roof"), True))
        return out + self._park_meshes()

    def _park_meshes(self):
        """駐車場の路面・植え込み（縁石つき）・駐車枠の白線・入口の白線・止まっている車の当たり判定（見えない箱）。"""
        if "parking" not in self.polys:
            return []
        pk = self.spec["parking"]
        out = [("1ROAD_s1_pass_parking", *self._draped(self.polys["parking"], z_fn=self.park_z), "asphalt", True)]
        for n, g, h, mat, zf in self.beds:
            pre = "1GRASS" if mat == "ground" else "1ROAD"
            out.append((f"{pre}_s1_pass_bed_{n}", *self._draped(g, h, z_fn=zf), mat, True))
            if h > 0:
                out.append((f"s1_pass_bed_{n}_curb", *self._curb(g, h, zf), "concrete", True))
        lines = []
        lift = 0.02
        for _, o, u, n, w, d, k, _ in self.stalls:   # 枠の仕切り線（通路の側から奥へ）
            for i in range(k + 1):
                p0 = o + u * w * i; p1 = p0 + n * d
                lines.append(self._flat_strip(p0, p1, 0.15, lift))
        for ln in pk.get("lines", []):               # 入口の縁の白線など（緯度経度の折れ線）
            q = _xy(ln)
            for p0, p1 in zip(q[:-1], q[1:]):
                lines.append(self._flat_strip(p0, p1, 0.15, lift))
        if lines:
            out.append(("LINES_s1_pass_parking", *_merge(lines), "line_white", True))
        boxes = []   # 止まっている車（見た目は Lua で置く AC の車）の当たり判定：長さ 4.4・幅 1.75・高さ 1.5 m の箱
        for _, _, x, y, z, h in self.parked_cars():
            f = Frame(np.array([x, y]), math.degrees(h))
            boxes.append(box(f, (-2.2, 2.2), (-0.875, 0.875), z - 0.2, z + 1.5, 2.0))
        if boxes:
            out.append(("1WALL_s1_pass_parked_cars", *_merge(boxes), "concrete", False))
        return out

    def _flat_strip(self, p0, p1, w, lift):
        """地面に沿った細い帯（白線）。"""
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); t = p1 - p0; L = float(np.linalg.norm(t))
        nrm = np.array([-t[1], t[0]]) / L * w / 2
        q = [p0 - nrm, p1 - nrm, p1 + nrm, p0 + nrm]
        zf = lambda p: float(self.park_z(*p)) if self.polys["parking"].buffer(0.5).contains(Point(p)) else float(self.surface_z(*p))
        v = np.array([[*p, zf(p) + lift] for p in q])
        return v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [L, 0], [L, w], [0, w]])


def build(spec, sec):
    return Site(spec, sec)


def load(path):
    from .start_lot import load_spec
    return load_spec(path)

