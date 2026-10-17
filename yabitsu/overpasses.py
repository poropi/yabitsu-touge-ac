"""本線の上を渡る橋（新東名の高架橋など）。OSM の bridge=yes・layer>=1 の道のうち、本線と交わるものを、
OSM の線に沿った床（床板＋桁＋両側の壁高欄＋柵）と柱で作る。

床の下面（桁の下）の高さは、交わる所の本線の路面＋clearance（標準 CLEARANCE_M。Street View で測った値を
data/manual/{区間}_overpasses.csv の osm_id,clearance_m,width_m,note で上書きする）。床は水平。"""
import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

from . import geo
from .crosssection import GUTTER_WIDTH
from .meshgen import compact, ribbon

CLEARANCE_M = 6.5
SLAB, GIRDER_DEPTH, GIRDER_INSET = 0.35, 2.2, 1.5   # 床板の厚さ、桁の深さ（床板の下）、桁の幅の内側への入り
PARAPET_H, RAIL_H = 1.0, 0.6                        # 壁高欄の高さ、その上の柵の高さ
PIER_EVERY, PIER_LEN, PIER_CLEAR = 30.0, 2.5, 1.0   # 柱の間隔、柱の長さ（橋の向き）、本線の側溝の外から柱までの最小距離
REACH_M = 120.0
LANE_W = {"motorway": 3.6, "trunk": 3.5, "primary": 3.25}
SHOULDERS = {"motorway": 4.0, "trunk": 2.0}
BRIDGE_TYPES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "secondary", "tertiary",
                "unclassified", "residential", "service"}


def _layer(t):
    try:
        return int(str(t.get("layer", "0")).split(";")[0])
    except ValueError:
        return 0


def _width(t):
    hw = t.get("highway", "").replace("_link", "")
    try:
        lanes = int(str(t.get("lanes", "")).split(";")[0])
    except ValueError:
        lanes = 2 if hw in ("motorway", "trunk") else 1 if t.get("oneway") == "yes" else 2
    return lanes * LANE_W.get(hw, 3.0) + SHOULDERS.get(hw, 1.0)


@dataclass
class Overpass:
    osm_id: int
    axis: LineString      # 床の中心線（交わる所の前後 REACH_M まで、橋の範囲で切ったもの）
    width: float
    z_under: float        # 桁の下面の高さ
    clearance: float
    s_cross: float
    t_cross: float        # axis の上の、本線と交わる位置
    keep_clear: float     # axis の上で、交わる所からこの距離の中には柱を立てない
    ground: object

    def _frame(self, step=2.0):
        L = self.axis.length
        t = np.r_[np.arange(0.0, L, step), L]
        p = np.array([self.axis.interpolate(v).coords[0] for v in t])
        d = np.gradient(p, axis=0); h = np.arctan2(d[:, 1], d[:, 0])
        return t, p, h

    def meshes(self):
        """(名前, 頂点, 面, UV, 材質, 描画するか)"""
        t, p, h = self._frame()
        n = len(t); w = self.width / 2; gi = w - GIRDER_INSET
        zb = self.z_under; zs = zb + GIRDER_DEPTH; zt = zs + SLAB
        # 断面を一周：桁の下 → 桁の横 → 床板の下 → 床板の端 → 壁高欄の外 → 上 → 内 → 路面 → 反対側
        lat = [-gi, -gi, -w, -w, -w, -w + 0.3, -w + 0.3, w - 0.3, w - 0.3, w, w, w, gi, gi, -gi]
        zz = [zb, zs, zs, zt, zt + PARAPET_H, zt + PARAPET_H, zt, zt, zt + PARAPET_H, zt + PARAPET_H, zt, zs, zs, zb, zb]
        deck = ribbon(p[:, 0], p[:, 1], h, np.tile(lat, (n, 1)), np.tile(zz, (n, 1)), t, u_scale=2.0)
        rails = []
        for side in (-1, 1):
            la = side * (w - 0.15)
            rails.append(ribbon(p[:, 0], p[:, 1], h, np.tile([la, la], (n, 1)),
                                np.tile([zt + PARAPET_H, zt + PARAPET_H + RAIL_H], (n, 1)), t, u_scale=2.0))
        rv = np.concatenate([r[0] for r in rails]); rf = np.concatenate([rails[0][1], rails[1][1] + len(rails[0][0])])
        ru = np.concatenate([r[2] for r in rails])
        out = [(f"overpass_{self.osm_id}_deck", *deck, "concrete", True),
               (f"overpass_{self.osm_id}_rail", rv, rf, ru, "railing_bars", True)]
        piers = self._piers()
        if piers is not None:
            out.append((f"1WALL_overpass_{self.osm_id}_piers", *piers, "concrete", True))
        return out

    def _piers(self):
        L = self.axis.length
        first = self.keep_clear + PIER_LEN / 2
        ts = sorted({*np.arange(self.t_cross + first, L - 3.0, PIER_EVERY), *np.arange(self.t_cross - first, 3.0, -PIER_EVERY)})
        parts = []
        for tc in ts:
            a = np.array(self.axis.interpolate(max(tc - PIER_LEN / 2, 0)).coords[0])
            b = np.array(self.axis.interpolate(min(tc + PIER_LEN / 2, L)).coords[0])
            d = (b - a) / max(np.linalg.norm(b - a), 1e-9); nrm = np.array([-d[1], d[0]])
            half = self.width / 2 - GIRDER_INSET - 0.5
            c = [a - nrm * half, b - nrm * half, b + nrm * half, a + nrm * half]
            zg = min(float(self.ground(*q)) for q in c) - 0.5
            v = np.array([[*q, zg] for q in c] + [[*q, self.z_under] for q in c], float)
            f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], np.int32)
            parts.append((v, f))
        if not parts:
            return None
        vs = np.concatenate([q[0] for q in parts])
        fs = np.concatenate([q[1] + 8 * k for k, q in enumerate(parts)])
        return vs, fs, vs[:, :2] / 2.0


def from_osm(elements, sec, sampler, overrides=None, reach_m=REACH_M, unproject=geo.from_local, skip_ids=()):
    """本線の上を渡る OSM の橋を Overpass にする。overrides = {osm_id: {"clearance_m":…, "width_m":…}}。"""
    overrides = overrides or {}
    main = LineString(np.c_[sec.x, sec.y])
    ground = lambda x, y: float(np.nan_to_num(sampler.sample(*unproject(np.atleast_1d(x), np.atleast_1d(y)))[0], nan=0.0))
    out = []
    for w in elements:
        t = w.get("tags", {})
        if w.get("type") != "way" or t.get("bridge", "no") == "no" or _layer(t) < 1 or t.get("highway") not in BRIDGE_TYPES:
            continue
        if w.get("id") in skip_ids:   # トンネルまで伸ばして motorway.py で作る橋
            continue
        pts = np.array([[g["lat"], g["lon"]] for g in w.get("geometry", [])])
        if len(pts) < 2:
            continue
        x, y = geo.to_local(pts[:, 0], pts[:, 1])
        line = LineString(np.c_[x, y])
        hit = line.intersection(main)
        if hit.is_empty:
            continue
        c = hit if hit.geom_type == "Point" else list(hit.geoms)[0]
        tc = line.project(c)
        a, b = max(0.0, tc - reach_m), min(line.length, tc + reach_m)
        axis = substring(line, a, b)
        i = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y)))
        ov = overrides.get(int(w.get("id", 0)), {})
        clearance = float(ov.get("clearance_m") or CLEARANCE_M)
        width = float(ov.get("width_m") or _width(t))
        z_road = float(max(sec.edge_z(1)[i], sec.edge_z(-1)[i]))
        # 柱を立てない範囲：本線の側溝の外 PIER_CLEAR までを、橋の向きに測った長さ
        ang = abs(math.sin(math.atan2(*np.diff(np.array(axis.coords)[[0, -1]], axis=0)[0][::-1]) - sec.heading[i]))
        half = (max(sec.hw_left[i], sec.hw_right[i]) + GUTTER_WIDTH + PIER_CLEAR) / max(ang, 0.2)
        out.append(Overpass(int(w.get("id", 0)), axis, width, z_road + clearance, clearance, float(sec.s[i]),
                            tc - a, half + width / 2 * abs(math.cos(math.asin(min(ang, 1.0)))) / max(ang, 0.2), ground))
    return out


def meshes(overpass_list):
    return [m for o in overpass_list for m in o.meshes()]


def load_overrides(manual_dir=None):
    """data/manual/{区間}_overpasses.csv（osm_id,clearance_m,width_m,note）をまとめて読む。"""
    import csv
    from . import config
    out = {}
    for f in sorted((manual_dir or config.DATA_MANUAL).glob("*_overpasses.csv")):
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("osm_id"):
                    out[int(r["osm_id"])] = {k: float(r[k]) for k in ("clearance_m", "width_m") if r.get(k) not in (None, "")}
    return out
