"""寺山富士見橋（S4、峠から約 1.5 km）：道の西の草の広場（コンクリートの踏み石）、西へ上るコンクリートのわき道、
公衆トイレ（板の型枠の跡がある打ち放しの壁、天窓のある苔の付いた茶色い寄棟の屋根）。

位置は緯度経度（data/manual/s4_terayama.json）。広場とわき道の高さは、道の縁の高さから決めた向きへ上る平面。
参考：ストリートビュー panoid fUHhEWDk_c6ht5fUxur0jA（2025-07）、利用者の写真（2025-06〜12）、Google の航空写真。
"""
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from . import site
from .crosssection import GUTTER_WIDTH
from .start_lot import Frame, _merge, box, road_formation


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class Site:
    def __init__(self, spec, sec):
        self.spec, self.sec = spec, sec
        self.road = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
        self.lane = site.largest(Polygon(site.ll_to_xy(spec["lane"]["points"])).buffer(0).difference(self.road))
        self.plaza = site.largest(Polygon(site.ll_to_xy(spec["plaza"]["points"])).buffer(0).difference(self.road).difference(self.lane))
        self.lane_plane = self._plane(spec["lane"])
        self.plaza_plane = self._plane(spec["plaza"])
        taken = self.lane.union(self.plaza)
        self.flats = []   # 道の縁の高さに合わせた草地（東のススキの下・わき道の北の看板のまわり）：(名前, 多角形, 平面)
        for g in spec.get("grass_flats", []):
            poly = site.largest(Polygon(site.ll_to_xy(g["points"])).buffer(0).difference(self.road).difference(taken))
            self.flats.append((g["name"], poly, self._plane(g)))
        t = spec["toilet"]
        fp = site.ll_to_xy(t["footprint"])
        self.tf = Frame(fp.mean(0), t["axis_deg"])          # a：長い辺の向き、b：正面（南）が −b
        L, W = t["size"]
        self.ta, self.tb = (-L / 2, L / 2), (-W / 2, W / 2)
        c = fp.mean(0)
        self.tz = float(site.plane_z(self.plaza_plane, c[0], c[1])) + float(t.get("base_dz", 0.0))

    def _plane(self, sp):
        """道の縁（anchor_ll にいちばん近い点の側溝の外）の高さを通り、道に沿う向きは道と同じ勾配、道から離れる向きへ slope で上る平面
        （道の縁にそって段ができない）。"""
        a = site.ll_to_xy([sp["anchor_ll"]])[0]; sec = self.sec
        i = int(np.argmin(np.hypot(sec.x - a[0], sec.y - a[1]))); h = sec.heading[i]
        lat = -(a[0] - sec.x[i]) * math.sin(h) + (a[1] - sec.y[i]) * math.cos(h)
        side = 1 if lat > 0 else -1
        ez = sec.edge_z(side); j0, j1 = max(i - 5, 0), min(i + 5, sec.n - 1)
        grade = float((ez[j1] - ez[j0]) / (sec.s[j1] - sec.s[j0]))
        t = np.array([math.cos(h), math.sin(h)]); n = np.array([-math.sin(h), math.cos(h)]) * side   # 道に沿う向き・道から離れる向き
        g = grade * t + float(sp["slope"]) * n
        return [float(a[0]), float(a[1]), float(ez[i]) + float(sp.get("dz", 0.0)), float(g[0]), float(g[1])]

    def area(self):
        """広場・わき道・トイレの台・草地を合わせた範囲（OSM の小道・建物を重ねない）。"""
        from shapely.ops import unary_union
        return unary_union([self.lane, self.plaza] + [g for _, g, _ in self.flats]
                           + [self.tf.rect((self.ta[0] - 1.0, self.ta[1] + 1.0), (self.tb[0] - 1.0, self.tb[1] + 1.0))])

    def mouths(self, reach=0.6):
        """わき道が本線に接している区間 [(side, s0, s1)]（この側は側溝を作らず路面を続ける）。"""
        sec = self.sec
        edge = self.lane.buffer(reach)
        out = []
        for side in (1, -1):
            hw = sec.hw_left if side > 0 else sec.hw_right
            n = np.c_[-np.sin(sec.heading), np.cos(sec.heading)] * side
            p = np.c_[sec.x, sec.y] + n * (hw + GUTTER_WIDTH * 0.5)[:, None]
            import shapely
            hit = shapely.contains_xy(edge, p[:, 0], p[:, 1])
            if hit.any():
                s = sec.s[hit]; out.append((side, float(s.min()), float(s.max())))
        return out

    def no_lines(self):
        """路肩の白線を引かない区間 [(side, s0, s1)]（ストリートビューで線が無い）。"""
        sec = self.sec; out = []
        for r in self.spec.get("no_edge_lines", []):
            ss = [float(sec.s[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))]) for p in site.ll_to_xy([r["from_ll"], r["to_ll"]])]
            for side in {"L": (1,), "R": (-1,), "B": (1, -1)}[r.get("side", "B")]:
                out.append((side, min(ss), max(ss)))
        return out

    def pads(self):
        c = self.tf.o
        toilet = self.tf.rect((self.ta[0] - 1.0, self.ta[1] + 1.0), (self.tb[0] - 1.0, self.tb[1] + 1.0))
        return [{"name": "terayama_plaza", "role": "flat", "kind": "site", "polygon": np.asarray(self.plaza.exterior.coords).tolist(),
                 "z": float(site.plane_z(self.plaza_plane, *self.plaza.centroid.coords[0])), "plane": self.plaza_plane},
                {"name": "terayama_lane", "role": "flat", "kind": "site", "polygon": np.asarray(self.lane.exterior.coords).tolist(),
                 "z": float(site.plane_z(self.lane_plane, *self.lane.centroid.coords[0])), "plane": self.lane_plane},
                {"name": "terayama_toilet", "role": "flat", "kind": "site", "polygon": np.asarray(toilet.exterior.coords).tolist(),
                 "z": self.tz - 0.05, "buildings_ok": True}] + [
                {"name": f"terayama_{n}", "role": "flat", "kind": "site", "polygon": np.asarray(g.exterior.coords).tolist(),
                 "z": float(site.plane_z(pl, *g.centroid.coords[0])), "plane": pl} for n, g, pl in self.flats]

    def meshes(self):
        sp, out = self.spec, []
        zl = lambda x, y: site.plane_z(self.lane_plane, x, y)
        zp = lambda x, y: site.plane_z(self.plaza_plane, x, y)
        out.append(("1ROAD_terayama_lane", *site.draped(self.lane, zl, 0.02, 2.0), "concrete", True))
        out.append(("1GRASS_terayama_plaza", *site.draped(self.plaza, zp, 0.0, 4.0), "ground", True))
        for n, g, pl in self.flats:
            out.append((f"1GRASS_terayama_{n}", *site.draped(g, lambda x, y, pl=pl: site.plane_z(pl, x, y), 0.0, 4.0), "ground", True))
        slabs = []
        for s in sp.get("slabs", []):
            c = site.ll_to_xy([s["at"]])[0]; f = Frame(c, s.get("axis_deg", 0.0)); L, W = s["size"]; z = float(zp(*c))
            slabs.append(box(f, (-L / 2, L / 2), (-W / 2, W / 2), z - 0.1, z + 0.025, 1.0))   # 草とほぼ同じ高さに埋まった踏み石
        if sp.get("walkway"):
            w = Polygon(site.ll_to_xy(sp["walkway"]["points"])).buffer(0)
            out.append(("1ROAD_terayama_walkway", *site.draped(w, lambda x, y: np.maximum(zp(x, y), self.tz - 0.25), 0.04, 2.0), "concrete", True))
        if slabs:
            out.append(("terayama_slabs", *_merge(slabs), "concrete", True))
        out += self._toilet()
        return out

    def _toilet(self):
        """打ち放しの壁の箱と、天窓の帯がある寄棟の屋根（軒の出 eaves）。正面（−b）は絵を別に貼る（s4_objects の pole_signs）。"""
        t, f = self.spec["toilet"], self.tf
        (a0, a1), (b0, b1) = self.ta, self.tb
        z0, z1 = self.tz, self.tz + t["wall_h"]
        out = [("1WALL_terayama_toilet", *box(f, (a0, a1), (b0, b1), z0 - 0.4, z1, 2.0), "concrete_board", True)]
        e, rise = t["eaves"], t["roof_rise"]
        A0, A1, B0, B1 = a0 - e, a1 + e, b0 - e, b1 + e
        W = B1 - B0; L = A1 - A0; zr = z1 + rise; ze = z1 - rise * e / (W / 2)
        r0, r1 = A0 + W / 2, A1 - W / 2
        q = lambda aa, bb, zz: np.c_[f.xy(np.asarray(aa, float), np.asarray(bb, float)), np.asarray(zz, float)]
        bm = (B0 + B1) / 2
        roof = [
            (q([A0, A1, r1, r0], [B0, B0, bm, bm], [ze, ze, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [L / 2, 0], [L / 2, 1], [0, 1]]),
            (q([A1, A0, r0, r1], [B1, B1, bm, bm], [ze, ze, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [L / 2, 0], [L / 2, 1], [0, 1]]),
            (q([A1, A1, r1, r1], [B0, B1, bm, bm], [ze, ze, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [W / 2, 0], [W / 2, 1], [0, 1]]),
            (q([A0, A0, r0, r0], [B1, B0, bm, bm], [ze, ze, zr, zr]), [[0, 1, 2, 3]], [[0, 0], [W / 2, 0], [W / 2, 1], [0, 1]]),
        ]
        out.append(("terayama_toilet_roof", *_merge([(v, np.asarray(fc), np.asarray(uv, float)) for v, fc, uv in roof]), "roof_shingle", True))
        out.append(("terayama_toilet_soffit", *_merge([(v[:, :][[3, 2, 1, 0]], np.asarray(fc), np.asarray(uv, float)) for v, fc, uv in roof]),
                    "concrete_board", True))
        # 天窓：棟の両側に、棟から k の割合だけ下までのガラスの帯（屋根の面から 4 cm 浮かす）
        k = float(t.get("skylight", 0.35)); lift = 0.04
        bk0, bk1 = bm - (bm - B0) * k, bm + (B1 - bm) * k; zk = zr - rise * k * (1 + e / (W / 2)) + lift
        sky = [(q([r0 + 0.6, r1 - 0.6, r1 - 0.6, r0 + 0.6], [bk0, bk0, bm, bm], [zk, zk, zr + lift, zr + lift]), [[0, 1, 2, 3]], [[0, 0], [1, 0], [1, 1], [0, 1]]),
               (q([r1 - 0.6, r0 + 0.6, r0 + 0.6, r1 - 0.6], [bk1, bk1, bm, bm], [zk, zk, zr + lift, zr + lift]), [[0, 1, 2, 3]], [[0, 0], [1, 0], [1, 1], [0, 1]])]
        out.append(("terayama_toilet_skylight", *_merge([(v, np.asarray(fc), np.asarray(uv, float)) for v, fc, uv in sky]), "glass_grey", True))
        # 正面（−b、南）の絵：壁から 2 cm 外
        fv = q([a0, a1, a1, a0], [b0 - 0.02] * 4, [z0, z0, z1 - 0.05, z1 - 0.05])
        out.append(("terayama_toilet_front", fv, np.array([[0, 1, 2, 3]]), np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float),
                    "terayama_toilet_front", True))
        # 正面の左の「山のトイレチップ塔」（緑の円柱、直径 0.3 m・高さ 1.6 m）
        from .town_objects import _prism
        c = f.xy(a0 + (a1 - a0) * float(t.get("chip_at", 0.3)), b0 - 0.3)
        out.append(("terayama_chip_tower", *_prism(np.asarray(c, float), 0.15, z0 - 0.1, z0 + 1.6, 12), "pole_green", True))
        return out


def build(spec, sec):
    return Site(spec, sec)
