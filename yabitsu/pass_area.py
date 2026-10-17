"""ヤビツ峠（下りのスタート）まわり：わき道（ピット）、植え込みの島、ブロック舗装の広場、休憩所。

位置は緯度経度で書く（data/manual/s1_pass.json）。区間の距離 s には頼らないので、道が峠の先（宮ヶ瀬側）へ
延びても、そのまま使える。高さは、まわりの路面に合わせた1枚の傾いた平面にそろえる。
"""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from . import geo
from .acobjects import Dummy
from .start_lot import Frame, _gable_ends, _gable_roof, _merge, box, road_formation

PLANE_RADIUS = 40.0  # この半径の中の路面の高さに、平面を合わせる


def _xy(ll):
    a = np.asarray(ll, float).reshape(-1, 2)
    x, y = geo.to_local(a[:, 0], a[:, 1])
    return np.c_[x, y]


def _largest(g):
    return g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)


class Site:
    def __init__(self, spec, sec):
        self.spec = spec
        lane = _xy(spec["lane"]["points"]); self.w = w = spec["lane"]["width"]
        ends = [int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1]))) for p in (lane[0], lane[-1])]
        if ends[0] > ends[1]:  # 出口（道の下り側＝s が小さい方の合流口）から並べる
            lane, ends = lane[::-1], ends[::-1]
        i0, i1 = ends
        self.line = LineString(lane)

        c = lane.mean(0)
        m = np.hypot(sec.x - c[0], sec.y - c[1]) <= PLANE_RADIUS
        A = np.c_[np.ones(m.sum()), sec.x[m] - c[0], sec.y[m] - c[1]]
        ridge = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 道がまっすぐで横の傾きが決まらないときは 0 にする
        z0, gx, gy = np.linalg.lstsq(np.r_[A, ridge], np.r_[sec.z[m], 0.0, 0.0], rcond=None)[0]
        self.plane = [float(c[0]), float(c[1]), float(z0), float(gx), float(gy)]

        road = road_formation(sec, s_min=sec.s[max(i0 - 40, 0)], s_max=sec.s[min(i1 + 40, sec.n - 1)])
        lane_band = self.line.buffer(w / 2, cap_style="flat", join_style="mitre")
        enclosed = Polygon(np.r_[lane, np.c_[sec.x[i0:i1 + 1], sec.y[i0:i1 + 1]][::-1]]).buffer(0)
        self.polys = {
            "lane": _largest(lane_band.difference(road)),
            "island": _largest(enclosed.difference(road).difference(self.line.buffer(w / 2 + 0.05))),
            "plaza": _largest(Polygon(_xy(spec["plaza"]["points"])).difference(lane_band).difference(road)),
        }

    def z(self, x, y):
        cx, cy, z0, gx, gy = self.plane
        return z0 + gx * (np.asarray(x, float) - cx) + gy * (np.asarray(y, float) - cy)

    def _draped(self, poly, dz=0.0, uv_scale=4.0):
        verts, faces = [], []
        for t in shapely.constrained_delaunay_triangles(poly).geoms:
            p = np.asarray(t.exterior.coords)[:3]
            if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) < 0:
                p = p[::-1]
            b = len(verts); verts += p.tolist(); faces.append((b, b + 1, b + 2))
        v = np.array(verts, float)
        return np.c_[v, self.z(v[:, 0], v[:, 1]) + dz], np.array(faces, np.int32), v / uv_scale

    def _curb(self, poly, h):
        ring = np.asarray(orient(poly, 1.0).exterior.coords)  # 反時計回り：進む向きの右側が外
        parts = []
        for p0, p1 in zip(ring[:-1], ring[1:]):
            za, zb = float(self.z(*p0)), float(self.z(*p1)); L = float(np.hypot(*(p1 - p0)))
            v = np.array([[*p0, za - 0.1], [*p1, zb - 0.1], [*p1, zb + h], [*p0, za + h]])
            parts.append((v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [L, 0], [L, 0.25], [0, 0.25]])))
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
        return [pad("pass_lane", "end", self.polys["lane"], pit_points=pits),
                pad("pass_island", "flat", self.polys["island"]),
                pad("pass_plaza", "flat", self.polys["plaza"])]

    def meshes(self):
        sp = self.spec; curb = sp.get("island_curb", 0.15)
        out = [("1ROAD_s1_pass_lane", *self._draped(self.polys["lane"]), "asphalt", True),
               ("1GRASS_s1_pass_island", *self._draped(self.polys["island"], curb), "ground", True),
               ("s1_pass_curb", *self._curb(self.polys["island"], curb), "concrete", True),
               ("1ROAD_s1_pass_plaza", *self._draped(self.polys["plaza"]), "pavers", True)]
        sh = sp["shelter"]; c = _xy(sh["center"])[0]; f = Frame(c, sh["axis_deg"])
        L, W = sh["size"]; a, b = (-L / 2, L / 2), (-W / 2, W / 2); z = float(self.z(*c))
        out.append(("1WALL_s1_pass_shelter", *box(f, a, b, z - 0.3, z + sh["wall_h"], 2.0), "wall_white", True))
        out.append(("s1_pass_shelter_gables", *_gable_ends(f, a, b, z + sh["wall_h"], sh["roof_h"]), "wall_white", True))
        out.append(("s1_pass_shelter_roof", *_gable_roof(f, a, b, z + sh["wall_h"], sh["roof_h"], sh["eaves"]), "roof", True))
        return out


def build(spec, sec):
    return Site(spec, sec)


def load(path):
    from .start_lot import load_spec
    return load_spec(path)


def s_of(sec, ll):
    """緯度経度の地点に、いちばん近い道の s。"""
    p = _xy(ll)[0]
    return float(sec.s[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))])


def layout_marks(spec, sec, pass_ll):
    """上りのゴール・下りの計測開始の基準（峠）と、下りのスタート列の先頭の s。"""
    return {"top_s": s_of(sec, pass_ll), "down_grid_s": s_of(sec, spec["down_start"])}
