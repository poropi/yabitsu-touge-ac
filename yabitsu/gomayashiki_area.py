"""護摩屋敷の水（S4、寺山富士見橋の約 90 m 先）：県道の西の分かれ目の舗装の広場（白線の駐車区画 4 台）、
湧き水の前を通って沢を渡り、北東へ回って県道に戻るわき道の輪、道より 0.6 m 低い湧き水の場所（長いコンクリートの段、
石畳の円い水受け、苔むした溶岩石の柱と水の落ちる管）。

位置は緯度経度（data/manual/s4_gomayashiki.json）。広場の高さは道の縁の高さから決めた向きへ上る平面（寺山と同じ）、
わき道の輪は広場の高さから県道の戻り口の縁の高さまで、道の長さに比例して下る（SEG_M ごとの傾いた平面）。
参考：ストリートビュー panoid WAIVHeeCIv5nbVBOQoD-wQ（2024-04）、湧き水の脇の 360° 写真（2020-09）、利用者の写真（2018〜2025）、
OSM（わき道 way 587598745・587598744・587598741、湧き水 node 1950963807）。
"""
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring, unary_union

from . import site
from .crosssection import GUTTER_WIDTH
from .start_lot import _merge, road_formation

SEG_M = 8.0          # わき道の輪を区切る長さ（区切りの線の上で前後の平面の高さがそろう）
STEP_H = 0.2         # 湧き水の段の一段の高さ


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _quad(a, b, c, d, face=(0, 0, 1)):
    """四角形。表（反時計回りに見える側）が face の向きになるように並べる。"""
    v = np.array([a, b, c, d], float)
    if np.dot(np.cross(v[1] - v[0], v[2] - v[0]), face) < 0:
        v = v[::-1]
    return v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)


def _band(p, q, w, zf, dz=0.035):   # 広場の舗装は +0.02 m に置くので、その上
    """p→q の幅 w の帯（路面の白線）。"""
    p, q = np.asarray(p, float), np.asarray(q, float); u = (q - p) / np.linalg.norm(q - p); n = np.array([-u[1], u[0]]) * w / 2
    pts = [p - n, q - n, q + n, p + n]
    v = np.array([[x, y, float(zf(x, y)) + dz] for x, y in pts])
    L = float(np.linalg.norm(q - p))
    return v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [L, 0], [L, 1], [0, 1]], float)


class Site:
    def __init__(self, spec, sec):
        self.spec, self.sec = spec, sec
        self.road = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
        self.apron = site.largest(Polygon(site.ll_to_xy(spec["apron"]["points"])).buffer(0).difference(self.road))
        self.plane = self._plane(spec["apron"])
        isl = spec.get("island")
        self.island = site.largest(Polygon(site.ll_to_xy(isl["points"])).buffer(0).difference(self.road).difference(self.apron)) if isl else None
        sp = spec["spring"]
        self.pad = site.largest(Polygon(site.ll_to_xy(sp["pad"])).buffer(0).difference(self.road).difference(self.apron))
        self.pad_z = float(site.plane_z(self.plane, *self.pad.centroid.coords[0])) - STEP_H * int(sp.get("steps", 3))
        self.spout = site.ll_to_xy([sp["spout"]])[0]
        self._loop()

    # ---- 高さ ----
    def _plane(self, sp):
        """道の縁（anchor_ll にいちばん近い点の側溝の外）の高さを通り、道に沿う向きは道と同じ勾配、道から離れる向きへ slope で上る平面。"""
        a = site.ll_to_xy([sp["anchor_ll"]])[0]; sec = self.sec
        i = int(np.argmin(np.hypot(sec.x - a[0], sec.y - a[1]))); h = sec.heading[i]
        lat = -(a[0] - sec.x[i]) * math.sin(h) + (a[1] - sec.y[i]) * math.cos(h)
        side = 1 if lat > 0 else -1
        ez = sec.edge_z(side); j0, j1 = max(i - 5, 0), min(i + 5, sec.n - 1)
        grade = float((ez[j1] - ez[j0]) / (sec.s[j1] - sec.s[j0]))
        t = np.array([math.cos(h), math.sin(h)]); n = np.array([-math.sin(h), math.cos(h)]) * side
        g = grade * t + float(sp["slope"]) * n
        return [float(a[0]), float(a[1]), float(ez[i]) + float(sp.get("dz", 0.0)), float(g[0]), float(g[1])]

    def _edge_z_at(self, p):
        sec = self.sec; i = int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1]))); h = sec.heading[i]
        lat = -(p[0] - sec.x[i]) * math.sin(h) + (p[1] - sec.y[i]) * math.cos(h)
        return float(sec.edge_z(1 if lat > 0 else -1)[i])

    def _loop(self):
        """わき道の輪：中心線を SEG_M ごとに区切り、区切りごとに傾いた平面。始まりは広場の平面の高さ、終わりは県道の縁の高さ。"""
        lp = self.spec["loop"]
        self.loop_line = LineString(site.ll_to_xy(lp["points"]))
        L = self.loop_line.length; w = float(lp["width_m"])
        p0, p1 = np.asarray(self.loop_line.coords[0]), np.asarray(self.loop_line.coords[-1])
        self.loop_z0 = float(site.plane_z(self.plane, *p0)); self.loop_z1 = self._edge_z_at(p1) + float(lp.get("end_dz", 0.0))
        n = max(int(round(L / SEG_M)), 1)
        self.loop_segs = []
        cover = self.road.union(self.apron).union(self.pad)
        for k in range(n):
            a, b = L * k / n, L * (k + 1) / n
            seg = substring(self.loop_line, a, b)
            poly = seg.buffer(w / 2, cap_style=2, join_style=2).difference(cover)
            for prev, _ in self.loop_segs:
                poly = poly.difference(prev)
            if poly.is_empty:
                continue
            poly = site.largest(poly)
            pa, pb = np.asarray(seg.coords[0]), np.asarray(seg.coords[-1])
            za, zb = self.z_loop(a), self.z_loop(b)
            u = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-6); g = (zb - za) / max(np.linalg.norm(pb - pa), 1e-6)
            self.loop_segs.append((poly, [float(pa[0]), float(pa[1]), float(za), float(g * u[0]), float(g * u[1])]))

    def z_loop(self, t):
        return self.loop_z0 + (self.loop_z1 - self.loop_z0) * t / self.loop_line.length

    def surface_z(self, x, y):
        """広場・わき道の路面の高さ（白線・小物の置き場）。わき道の上なら区切りの平面、それ以外は広場の平面。"""
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        z = site.plane_z(self.plane, x, y)
        for poly, pl in self.loop_segs:
            import shapely
            m = shapely.contains_xy(poly.buffer(0.05), x, y)
            z = np.where(m, site.plane_z(pl, x, y), z)
        return z

    # ---- 範囲 ----
    def area(self):
        parts = [self.apron, self.pad] + [p for p, _ in self.loop_segs] + ([self.island] if self.island else [])
        return unary_union(parts)

    def mouths(self, reach=0.6):
        """広場・わき道が本線に接している区間 [(side, s0, s1)]（この側は側溝を作らず路面を続ける）。"""
        import shapely
        sec = self.sec
        edge = unary_union([self.apron] + [p for p, _ in self.loop_segs]).buffer(reach)
        out = []
        for side in (1, -1):
            hw = sec.hw_left if side > 0 else sec.hw_right
            n = np.c_[-np.sin(sec.heading), np.cos(sec.heading)] * side
            p = np.c_[sec.x, sec.y] + n * (hw + GUTTER_WIDTH * 0.5)[:, None]
            hit = shapely.contains_xy(edge, p[:, 0], p[:, 1])
            if hit.any():
                idx = np.nonzero(hit)[0]
                runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)   # 広場の口とわき道の戻り口は別の区間
                out += [(side, float(sec.s[r[0]]), float(sec.s[r[-1]])) for r in runs]
        return out

    def no_lines(self):
        sec = self.sec; out = []
        for r in self.spec.get("no_edge_lines", []):
            ss = [float(sec.s[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))]) for p in site.ll_to_xy([r["from_ll"], r["to_ll"]])]
            for side in {"L": (1,), "R": (-1,), "B": (1, -1)}[r.get("side", "B")]:
                out.append((side, min(ss), max(ss)))
        return out

    def pads(self):
        # 舗装する面は kind "pass"：面にかかる地形の格子の角を全部 面より下げ、斜面の三角形が舗装の上に出ないようにする（terrain._sink_pass_cells）。
        # 2026-10-25 のゲームの撮影で、広場の西の縁に地形の三角形がはみ出してギザギザに見えた（縁石の線の上で最大 0.81 m 高かった）
        c = self.apron.centroid
        out = [{"name": "goma_apron", "role": "flat", "kind": "pass", "polygon": np.asarray(self.apron.exterior.coords).tolist(),
                "z": float(site.plane_z(self.plane, c.x, c.y)), "plane": self.plane},
               {"name": "goma_spring", "role": "flat", "kind": "pass", "polygon": np.asarray(self.pad.exterior.coords).tolist(),
                "z": self.pad_z}]
        if self.island:
            pl = list(self.plane); pl[2] += float(self.spec["island"].get("dz", 0.12))
            out.append({"name": "goma_island", "role": "flat", "kind": "site", "polygon": np.asarray(self.island.exterior.coords).tolist(),
                        "z": float(site.plane_z(pl, *self.island.centroid.coords[0])), "plane": pl})
        out += [{"name": f"goma_loop_{k:02d}", "role": "flat", "kind": "pass", "polygon": np.asarray(p.exterior.coords).tolist(),
                 "z": float(site.plane_z(pl, *p.centroid.coords[0])), "plane": pl} for k, (p, pl) in enumerate(self.loop_segs)]
        return out

    # ---- 形 ----
    def meshes(self):
        sp, out = self.spec, []
        zp = lambda x, y: site.plane_z(self.plane, x, y)
        out.append(("1ROAD_goma_apron", *site.draped(self.apron, zp, 0.02, 4.0), "asphalt", True))
        for k, (poly, pl) in enumerate(self.loop_segs):
            out.append((f"1ROAD_goma_loop_{k:02d}", *site.draped(poly, lambda x, y, pl=pl: site.plane_z(pl, x, y), 0.02, 4.0), "asphalt", True))
        if self.island:
            dz = float(sp["island"].get("dz", 0.12))
            out.append(("1GRASS_goma_island", *site.draped(self.island, zp, dz, 4.0), "ground", True))
        out += self._spring()
        out += self._paint()
        out += self._curb()
        return out

    def _curb(self):
        """広場の西の縁の低いコンクリートの縁石（高さ 0.15 m・幅 0.25 m）。"""
        c = self.spec.get("west_curb")
        if not c:
            return []
        q = np.asarray(LineString(site.ll_to_xy(c["points"])).segmentize(1.0).coords); parts = []
        for i in range(len(q) - 1):
            a, b = q[i], q[i + 1]; u = (b - a) / np.linalg.norm(b - a); n = np.array([-u[1], u[0]]) * 0.125
            za, zb = float(self.surface_z(*a)[0]), float(self.surface_z(*b)[0])
            top = [[*(a - n), za + 0.15], [*(b - n), zb + 0.15], [*(b + n), zb + 0.15], [*(a + n), za + 0.15]]
            parts.append(_quad(*top))
            for s in (1, -1):
                e = [[*(a + s * n), za - 0.1], [*(b + s * n), zb - 0.1], [*(b + s * n), zb + 0.15], [*(a + s * n), za + 0.15]]
                parts.append(_quad(*e, face=(*(s * n), 0)))
        return [("1WALL_goma_curb", *_merge(parts), "concrete", True)]

    def _paint(self):
        """駐車区画の白線、広場の西の縁の白線と斜線（導流帯）。"""
        sp = self.spec.get("paint", {}); parts = []
        zf = lambda x, y: float(self.surface_z(x, y)[0])
        for a, b in sp.get("bays", []):
            p, q = site.ll_to_xy([a, b]); parts.append(_band(p, q, 0.15, zf))
        if sp.get("edge"):
            q = site.ll_to_xy(sp["edge"])
            for i in range(len(q) - 1):
                parts.append(_band(q[i], q[i + 1], 0.15, zf))
        for a, b in sp.get("hatch", []):
            p, q = site.ll_to_xy([a, b]); parts.append(_band(p, q, 0.30, zf))
        return [("goma_paint", *_merge(parts), "line_white", True)] if parts else []

    def _spring(self):
        """湧き水：低い面（砂利）、広場から下る長いコンクリートの段、石畳の円い水受け、苔むした溶岩石の柱、水の出る管と落ちる水。"""
        sp, out = self.spec["spring"], []
        z = self.pad_z
        out.append(("1ROAD_goma_spring_pad", *site.draped(self.pad, lambda x, y: np.full_like(x, z), 0.0, 3.0), "gravel_dark", True))
        # 段：上の縁（広場の西の縁）の線から、低い面の側へ 0.35 m ずつ下りる
        top = np.asarray(site.ll_to_xy(sp["steps_top"])); nst = int(sp.get("steps", 3))
        u = top[-1] - top[0]; u = u / np.linalg.norm(u); n = np.array([-u[1], u[0]])
        if Point(*(top.mean(0) + n)).distance(self.pad) > Point(*(top.mean(0) - n)).distance(self.pad):
            n = -n                                                                    # n：低い面の側
        steps = []
        for k in range(nst):
            a, b = top[0] + n * 0.35 * k, top[-1] + n * 0.35 * k
            zt = float(site.plane_z(self.plane, *top.mean(0))) - STEP_H * k
            a2, b2 = a + n * 0.35, b + n * 0.35
            steps.append(_quad([*a, zt - STEP_H], [*b, zt - STEP_H], [*b2, zt - STEP_H], [*a2, zt - STEP_H]))   # 踏み面（一段下）
            steps.append(_quad([*a, zt], [*b, zt], [*b, zt - STEP_H], [*a, zt - STEP_H], face=(*n, 0)))        # 蹴上げ（低い面の側を向く）
        out.append(("1ROAD_goma_steps", *_merge(steps), "concrete", True))
        # 水受け：石畳の輪（半径 r、縁の高さ 0.12 m）と、中の水面
        c = self.spout; r = float(sp.get("basin_r", 1.3)); K = 28
        ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
        ring_o = c + np.c_[np.cos(ang), np.sin(ang)] * r; ring_i = c + np.c_[np.cos(ang), np.sin(ang)] * (r - 0.35)
        rim = []
        for i in range(K):
            j = (i + 1) % K
            out_n = (*(ring_o[i] + ring_o[j] - 2 * c), 0)
            rim.append(_quad([*ring_i[i], z + 0.12], [*ring_i[j], z + 0.12], [*ring_o[j], z + 0.12], [*ring_o[i], z + 0.12]))
            rim.append(_quad([*ring_o[i], z + 0.12], [*ring_o[j], z + 0.12], [*ring_o[j], z - 0.05], [*ring_o[i], z - 0.05], face=out_n))
            rim.append(_quad([*ring_i[i], z + 0.12], [*ring_i[j], z + 0.12], [*ring_i[j], z + 0.0], [*ring_i[i], z + 0.0], face=tuple(-np.array(out_n))))
        out.append(("1WALL_goma_basin", *_merge(rim), "flagstone", True))
        wv = np.array([[*c, z + 0.04]] + [[*p, z + 0.04] for p in ring_i]); wf = np.array([[0, i + 1, (i + 1) % K + 1] for i in range(K)])
        out.append(("goma_basin_floor", wv, wf, wv[:, :2] / 2.0, "flagstone", True))            # 水受けの中：濡れた石畳（写真では水は薄い）
        for deg in sp.get("pipes_deg", []):                                                     # 管の水が落ちる所の小さな水たまり
            d = np.array([math.sin(math.radians(deg)), math.cos(math.radians(deg))]); pc = c + d * 0.84
            ring = pc + np.c_[np.cos(ang), np.sin(ang)] * 0.16
            pv = np.array([[*pc, z + 0.05]] + [[*p, z + 0.05] for p in ring])
            out.append((f"goma_basin_pool_{int(deg)}", pv, wf, pv[:, :2], "water", True))
        # 溶岩石の柱：石を積んだごつごつの筒（14 角・5 段、半径と高さを石ごとにばらつかせる）、上は浅い皿
        h = float(sp.get("spout_h", 0.9)); M = 14; rng = np.random.default_rng(7)
        a2 = np.linspace(0, 2 * np.pi, M, endpoint=False)
        rings = [(0.0, 0.42), (0.22, 0.44), (0.45, 0.39), (0.68, 0.37), (h, 0.31)]
        pts = []
        for k, (zz, rr) in enumerate(rings):
            jr = 1.0 + rng.uniform(-0.22, 0.22, M); jz = rng.uniform(-0.05, 0.05, M) if 0 < k < len(rings) - 1 else np.zeros(M)
            pts.append(np.c_[c + np.c_[np.cos(a2), np.sin(a2)] * (rr * jr)[:, None], z + zz + jz])
        col = []
        for k in range(len(pts) - 1):
            for i in range(M):
                j = (i + 1) % M
                col.append((np.array([pts[k][i], pts[k][j], pts[k + 1][j], pts[k + 1][i]]), np.array([[0, 1, 2, 3]]),
                            np.array([[i / M * 2.6, rings[k][0]], [(i + 1) / M * 2.6, rings[k][0]], [(i + 1) / M * 2.6, rings[k + 1][0]], [i / M * 2.6, rings[k + 1][0]]])))
        out.append(("1WALL_goma_spout", *_merge(col), "lava_moss", True))
        topc = np.array([*c, z + h - 0.08])                                   # 上の皿のくぼみ（三角形の扇）
        cap = [(np.array([pts[-1][i], pts[-1][(i + 1) % M], topc]), np.array([[0, 1, 2]]), np.array([[0, 0], [1, 0], [0.5, 0.5]])) for i in range(M)]
        out.append(("goma_spout_top", *_merge(cap), "lava_moss", True))
        # 管と落ちる水：柱の上から外へ出る細い管（長さ 0.35 m）、先から水受けへ弧を描いて落ちる水の帯
        pipes, falls = [], []
        for deg in sp.get("pipes_deg", [60, 200, 320]):
            d = np.array([math.sin(math.radians(deg)), math.cos(math.radians(deg))])   # 方位（北から時計回り）
            p0 = c + d * 0.30; p1 = c + d * 0.62; zt = z + h - 0.18
            nn = np.array([-d[1], d[0]]) * 0.015
            pipes.append(_quad([*(p0 - nn), zt], [*(p1 - nn), zt - 0.03], [*(p1 + nn), zt - 0.03], [*(p0 + nn), zt], face=(0, 0, -1)))
            pipes.append(_quad([*(p0 - nn), zt + 0.03], [*(p1 - nn), zt], [*(p1 + nn), zt], [*(p0 + nn), zt + 0.03]))
            arc = [np.r_[p1 + d * 0.22 * t, zt - 0.03 - (zt - z - 0.06) * t ** 1.6] for t in np.linspace(0, 1, 6)]
            ww = np.array([-d[1], d[0]]) * 0.02
            for i in range(len(arc) - 1):
                a, b = arc[i], arc[i + 1]
                falls.append((np.array([[*(a[:2] - ww), a[2]], [*(a[:2] + ww), a[2]], [*(b[:2] + ww), b[2]], [*(b[:2] - ww), b[2]]]),
                              np.array([[0, 1, 2, 3]]), np.array([[0, i / 5], [1, i / 5], [1, (i + 1) / 5], [0, (i + 1) / 5]], float)))
        out.append(("goma_spout_pipes", *_merge(pipes), "pole_grey", True))
        out.append(("goma_spout_water", *_merge(falls), "water_fall", True))
        return out


def build(spec, sec):
    return Site(spec, sec)
