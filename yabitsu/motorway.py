"""本線の上を渡る高速道路（新東名）を、両側のトンネルの入口まで作る。

OSM で本線を渡る橋（bridge=yes, layer>=1）から、つながった高速道路の way を前後にたどり、トンネル（tunnel=yes）に
入る所で止める。路面の高さは「入口の地面の高さ → 本線と交わる所（桁の下＝本線の路面＋クリアランス）→ 反対の入口」を
結ぶ折れ線。地面から VIADUCT_MIN_H 以上高い所は高架（床・桁・壁高欄・柵・柱）、それより低い所は盛り土・切り土の上の道
（地形を pad で路面の高さに合わせる）。トンネルの入口には坑口（コンクリートの面と黒い口）を置く。"""
import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from . import geo
from .crosssection import GUTTER_WIDTH
from .meshgen import compact, ribbon
from .overpasses import (BRIDGE_TYPES, CLEARANCE_M, GIRDER_DEPTH, GIRDER_INSET, PARAPET_H, PIER_CLEAR, PIER_EVERY, PIER_LEN,
                         RAIL_H, SLAB, _layer, _width)

VIADUCT_MIN_H = 2.5    # 地面からこれより高い所は高架
STEP = 2.0
PORTAL_H, PORTAL_EXTRA = 8.5, 3.0   # 坑口の面の高さ（路面から）、道幅より左右に広げる長さ
OPEN_H = 6.5                        # 坑口の口の高さ
PAD_SEG = 10.0


def _line(w):
    p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
    x, y = geo.to_local(p[:, 0], p[:, 1])
    return np.c_[x, y]


def _key(p):
    return (round(float(p[0]), 2), round(float(p[1]), 2))


def _is_mw(w):
    return w.get("type") == "way" and w.get("tags", {}).get("highway", "").startswith(("motorway", "trunk"))


def _chain(ways, start):
    """start の way から、終わり → 始まりのつながりで前後にたどる。トンネルに入ったら止める。
    (点の並び, 使った way の id, (始まりがトンネルの入口か, 終わりがトンネルの入口か))"""
    by_first, by_last = {}, {}
    for w in ways:
        if _is_mw(w) and len(w.get("geometry", [])) >= 2:
            xy = _line(w); by_first.setdefault(_key(xy[0]), []).append((w, xy)); by_last.setdefault(_key(xy[-1]), []).append((w, xy))
    pts = list(_line(start)); ids = {start["id"]}
    portals = [False, False]
    while True:
        nxt = [q for q in by_first.get(_key(pts[-1]), []) if q[0]["id"] not in ids]
        if not nxt:
            break
        w, xy = nxt[0]
        if w["tags"].get("tunnel", "no") != "no":
            portals[1] = True; break
        pts += list(xy[1:]); ids.add(w["id"])
    while True:
        prv = [q for q in by_last.get(_key(pts[0]), []) if q[0]["id"] not in ids]
        if not prv:
            break
        w, xy = prv[0]
        if w["tags"].get("tunnel", "no") != "no":
            portals[0] = True; break
        pts = list(xy[:-1]) + pts; ids.add(w["id"])
    return np.asarray(pts), ids, tuple(portals)


@dataclass
class Motorway:
    osm_id: int
    axis: LineString
    way_ids: set
    portals: tuple
    width: float
    t_cross: float
    knots_t: np.ndarray
    knots_z: np.ndarray      # 路面（床板の上）の高さの折れ線
    ground: object           # ground(x, y) -> 地面の高さ
    keep_clear: float        # 本線との交点から、この距離の中には柱を立てない

    def z_top(self, t):
        return float(np.interp(t, self.knots_t, self.knots_z))

    def z_under(self, t):
        return self.z_top(t) - SLAB - GIRDER_DEPTH

    def _frame(self, step=STEP):
        L = self.axis.length
        t = np.r_[np.arange(0.0, L, step), L]
        p = np.array([self.axis.interpolate(v).coords[0] for v in t])
        d = np.gradient(p, axis=0); h = np.arctan2(d[:, 1], d[:, 0])
        zt = np.interp(t, self.knots_t, self.knots_z)
        zg = np.array([self.ground(*q) for q in p])
        return t, p, h, zt, zg

    def meshes(self):
        """(名前, 頂点, 面, UV, 材質, 描画するか)"""
        t, p, h, zt, zg = self._frame()
        n = len(t); w = self.width / 2; gi = w - GIRDER_INSET
        zs = zt - SLAB; zb = zs - GIRDER_DEPTH
        lat = np.array([-gi, -gi, -w, -w, -w, -w + 0.3, -w + 0.3, w - 0.3, w - 0.3, w, w, w, gi, gi, -gi])
        rel = np.array([-SLAB - GIRDER_DEPTH, -SLAB, -SLAB, 0, PARAPET_H, PARAPET_H, 0, 0, PARAPET_H, PARAPET_H, 0, -SLAB, -SLAB,
                        -SLAB - GIRDER_DEPTH, -SLAB - GIRDER_DEPTH])
        deck = ribbon(p[:, 0], p[:, 1], h, np.tile(lat, (n, 1)), zt[:, None] + rel[None, :], t, u_scale=2.0)
        top_lat = np.array([-w + 0.3, w - 0.3])
        road = ribbon(p[:, 0], p[:, 1], h, np.tile(top_lat, (n, 1)), np.tile(zt[:, None] + 0.01, (1, 2)), t, u_scale=4.0, v_scale=4.0)
        rails = []
        for side in (-1, 1):
            la = side * (w - 0.15)
            rails.append(ribbon(p[:, 0], p[:, 1], h, np.tile([la, la], (n, 1)),
                                np.c_[zt + PARAPET_H, zt + PARAPET_H + RAIL_H], t, u_scale=2.0))
        out = [(f"mw_{self.osm_id}_deck", *deck, "concrete", True),
               (f"mw_{self.osm_id}_road", *road, "asphalt", True),
               (f"mw_{self.osm_id}_rail", *_cat(rails), "railing_bars", True)]
        piers = self._piers(t, p, h, zt, zg)
        if piers is not None:
            out.append((f"1WALL_mw_{self.osm_id}_piers", *piers, "concrete", True))
        for end, flag in zip((0, -1), self.portals):
            if flag:
                out += self._portal(p, h, zt, end)
        return out

    def _piers(self, t, p, h, zt, zg):
        ts = sorted({*np.arange(self.t_cross + self.keep_clear + PIER_LEN / 2, self.axis.length - 5.0, PIER_EVERY),
                     *np.arange(self.t_cross - self.keep_clear - PIER_LEN / 2, 5.0, -PIER_EVERY)})
        parts = []
        for tc in ts:
            if self.z_top(tc) - self.ground(*self.axis.interpolate(tc).coords[0]) < VIADUCT_MIN_H:
                continue
            a = np.array(self.axis.interpolate(max(tc - PIER_LEN / 2, 0)).coords[0])
            b = np.array(self.axis.interpolate(min(tc + PIER_LEN / 2, self.axis.length)).coords[0])
            d = (b - a) / max(np.linalg.norm(b - a), 1e-9); nrm = np.array([-d[1], d[0]])
            half = self.width / 2 - GIRDER_INSET - 0.5
            c = [a - nrm * half, b - nrm * half, b + nrm * half, a + nrm * half]
            z0 = min(self.ground(*q) for q in c) - 0.5; z1 = self.z_under(tc)
            v = np.array([[*q, z0] for q in c] + [[*q, z1] for q in c], float)
            parts.append((v, np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])))
        if not parts:
            return None
        vs = np.concatenate([q[0] for q in parts]); fs = np.concatenate([q[1] + 8 * k for k, q in enumerate(parts)])
        return vs, fs, vs[:, :2] / 2.0

    def _portal(self, p, h, zt, end):
        """坑口：入口の線に立つ面（外向き）と、黒い口（少し奥）。"""
        q = p[end]; hd = h[end] + (math.pi if end == 0 else 0.0)   # 外（トンネルと反対）の向き
        d = np.array([math.cos(hd), math.sin(hd)]); n = np.array([-d[1], d[0]])
        W = self.width / 2 + PORTAL_EXTRA; z = zt[end]
        face = np.array([[*(q - n * W), z - 1.0], [*(q + n * W), z - 1.0], [*(q + n * W), z + PORTAL_H], [*(q - n * W), z + PORTAL_H]])
        o = q - d * 0.3; ow = self.width / 2 - 0.2
        hole = np.array([[*(o - n * ow), z], [*(o + n * ow), z], [*(o + n * ow), z + OPEN_H], [*(o - n * ow), z + OPEN_H]])
        f = np.array([[0, 1, 2, 3]])
        if np.dot(np.cross(np.r_[face[1, :2] - face[0, :2], 0], [0, 0, 1])[:2], d) < 0:
            f = f[:, ::-1]
        k = "a" if end == 0 else "b"
        return [(f"1WALL_mw_{self.osm_id}_portal_{k}", face, f, face[:, [0, 2]] / 2.0, "concrete", True),
                (f"mw_{self.osm_id}_portal_{k}_hole", hole, f, hole[:, [0, 2]] / 2.0, "tunnel_dark", True)]

    def pads(self):
        """地面から低い所（盛り土・切り土）：道の幅＋2m の帯を PAD_SEG ごとの傾いた平面にして地形を合わせる。"""
        out = []
        L = self.axis.length
        for a in np.arange(0.0, L, PAD_SEG):
            b = min(a + PAD_SEG, L)
            m = (a + b) / 2
            pm = self.axis.interpolate(m).coords[0]
            if self.z_top(m) - self.ground(*pm) > VIADUCT_MIN_H:
                continue
            pa = np.array(self.axis.interpolate(a).coords[0]); pb = np.array(self.axis.interpolate(b).coords[0])
            d = pb - pa; Ln = np.linalg.norm(d)
            if Ln < 1e-6:
                continue
            u = d / Ln; nrm = np.array([-u[1], u[0]]); half = self.width / 2 + 1.0
            poly = Polygon([pa - nrm * half, pb - nrm * half, pb + nrm * half, pa + nrm * half])
            g = (self.z_top(b) - self.z_top(a)) / Ln
            out.append({"name": f"mw_{self.osm_id}_{int(a)}", "role": "flat", "kind": "motorway",
                        "polygon": np.asarray(poly.exterior.coords).tolist(), "z": self.z_top(m),
                        "plane": [float(pa[0]), float(pa[1]), self.z_top(a), float(g * u[0]), float(g * u[1])]})
        return out

    def footprint(self):
        return self.axis.buffer(self.width / 2 + 2.0, cap_style="flat")


def _cat(parts):
    v = np.concatenate([q[0] for q in parts]); f = np.concatenate([parts[0][1]] + [q[1] + sum(len(r[0]) for r in parts[:k]) for k, q in enumerate(parts) if k])
    return compact(v, f, np.concatenate([q[2] for q in parts]))


def from_osm(elements, sec, sampler, overrides=None, unproject=geo.from_local):
    """本線を渡る高速道路の橋から、トンネルの入口までの道。overrides = {橋の osm_id: {"clearance_m", "width_m"}}。"""
    overrides = overrides or {}
    main = LineString(np.c_[sec.x, sec.y])
    ground = lambda x, y: float(np.nan_to_num(sampler.sample(*unproject(np.atleast_1d(x), np.atleast_1d(y)))[0], nan=0.0))
    out, used = [], set()
    for w in elements:
        t = w.get("tags", {})
        if (not _is_mw(w) or w.get("id") in used or t.get("bridge", "no") == "no" or _layer(t) < 1
                or t.get("highway") not in BRIDGE_TYPES):
            continue
        xy = _line(w)
        if not LineString(xy).intersects(main):
            continue
        pts, ids, portals = _chain(elements, w)
        used |= ids
        axis = LineString(pts)
        c = axis.intersection(main)
        c = c if c.geom_type == "Point" else list(c.geoms)[0]
        tc = axis.project(c)
        i = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y)))
        ov = overrides.get(int(w["id"]), {})
        clearance = float(ov.get("clearance_m") or CLEARANCE_M); width = float(ov.get("width_m") or _width(t))
        z_road = float(max(sec.edge_z(1)[i], sec.edge_z(-1)[i]))
        zc = z_road + clearance + GIRDER_DEPTH + SLAB
        L = axis.length
        za = ground(*pts[0]) if portals[0] else zc
        zb = ground(*pts[-1]) if portals[1] else zc
        ang = abs(math.sin(math.atan2(*(np.array(axis.interpolate(min(tc + 1, L)).coords[0]) -
                                          np.array(axis.interpolate(max(tc - 1, 0)).coords[0]))[::-1]) - sec.heading[i]))
        keep = (max(sec.hw_left[i], sec.hw_right[i]) + GUTTER_WIDTH + PIER_CLEAR) / max(ang, 0.2) + width / 2 * math.sqrt(max(0.0, 1 - ang ** 2)) / max(ang, 0.2)
        out.append(Motorway(int(w["id"]), axis, ids, portals, width, tc, np.array([0.0, tc, L]), np.array([za, zc, zb]), ground, keep))
    return out


def meshes(mws):
    return [m for x in mws for m in x.meshes()]
