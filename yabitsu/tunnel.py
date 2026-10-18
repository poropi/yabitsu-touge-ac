"""県道70号のトンネル（OSM の tunnel=yes の way）を、本線の s の範囲にする。高さの合わせ（標高データ＝山の表面を使わない）、
地形（道の上の山を削らない）、トンネルの筒・照明・坑口・表札のメッシュに使う。"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

from . import geo, osm


@dataclass
class Tunnel:
    osm_id: int
    name: str
    s0: float
    s1: float


def from_osm(elements, sec, near_m=3.0):
    """sec（s・x・y を持つもの）の上に乗っている、県道70号のトンネルの way の範囲。"""
    main = LineString(np.c_[sec.x, sec.y])
    out = []
    for w in osm.route70_ways(elements):
        t = w.get("tags", {})
        if w.get("type") != "way" or t.get("tunnel", "no") in ("no", "") or len(w.get("geometry", [])) < 2:
            continue
        c = np.array([geo.to_local(g["lat"], g["lon"]) for g in w["geometry"]], float)
        if max(main.distance(Point(p)) for p in c) > near_m:
            continue
        ss = [float(sec.s[int(np.argmin(np.hypot(sec.x - p[0], sec.y - p[1])))]) for p in (c[0], c[-1])]
        out.append(Tunnel(int(w.get("id", 0)), t.get("name", "") if "トンネル" in t.get("name", "") else "", min(ss), max(ss)))
    return out


def row_mask(s, tunnels, margin_m=5.0):
    """s のうち、どれかのトンネルの範囲（前後 margin_m を含む）に入る所。"""
    s = np.asarray(s, float); m = np.zeros(len(s), bool)
    for t in tunnels:
        m |= (s >= t.s0 - margin_m) & (s <= t.s1 + margin_m)
    return m


WALL_H = 3.0          # 縦の壁の高さ（その上が半円のアーチ）
PORTAL_WING = 12.0    # 坑口の面を、道の筒の外へ左右にこれだけ広げる（地形の段差を隠す）
LAMP_EVERY = 10.0
STEP = 2.0
N_ARC = 12


def _frame(sec, i):
    from .crosssection import GUTTER_WIDTH
    h = sec.heading[i]; n = np.array([-np.sin(h), np.cos(h)]); c = np.array([sec.x[i], sec.y[i]])
    l = sec.hw_left[i] + GUTTER_WIDTH; r = sec.hw_right[i] + GUTTER_WIDTH
    return c, n, l, r


def bore_height(sec, i):
    """路面の中心から筒の天井までの高さ：縦の壁 WALL_H ＋ 半円の半径（筒の幅の半分）。"""
    _, _, l, r = _frame(sec, i)
    return WALL_H + (l + r) / 2.0


def _profile(sec, i):
    """行 i の筒の断面（右の下 → 右の壁 → アーチ → 左の壁 → 左の下）。[(lat, dz)]。dz は路面の中心からの高さ。"""
    _, _, l, r = _frame(sec, i)
    rad = (l + r) / 2.0; cl = (l - r) / 2.0
    pts = [(-r, -0.3), (-r, WALL_H)]
    for a in np.linspace(np.pi, 0, N_ARC + 1)[1:-1]:
        pts.append((cl + rad * np.cos(a), WALL_H + rad * np.sin(a)))
    return pts + [(l, WALL_H), (l, -0.3)]


def _rows(sec, t):
    return [int(np.argmin(np.abs(sec.s - s))) for s in np.r_[np.arange(t.s0, t.s1, STEP), t.s1]]


def meshes(sec, tunnels, spec, ground_z):
    """トンネルの筒（当たり判定つき、内向き）・照明・両側の坑口（アーチの穴のあいた面）・表札。
    spec = {"<osm_id>": {"plate": 表札の文字}}。ground_z はできあがった地形の高さ（坑口の面の上の端に使う）。"""
    out = []
    for t in tunnels:
        rows = _rows(sec, t)
        V, F = [], []
        for k, i in enumerate(rows):
            c, n, _, _ = _frame(sec, i)
            for lat, dz in _profile(sec, i):
                p = c + n * lat; V.append([p[0], p[1], sec.z[i] + dz])
        m = len(_profile(sec, rows[0]))
        for k in range(len(rows) - 1):
            for j in range(m - 1):
                a, b = k * m + j, k * m + j + 1
                F.append([a, b, b + m, a + m])
        v = np.array(V, float); f = np.array(F, np.int32)
        from .meshgen import face_normals
        cen = v[f].mean(1); nr = face_normals(v, f)
        ii = np.array([rows[min(q // m, len(rows) - 1)] for q in f[:, 0]])
        axis = np.c_[sec.x[ii], sec.y[ii], sec.z[ii] + WALL_H]
        if np.mean(np.sum((axis - cen) * nr, axis=1) > 0) < 0.5:
            f = f[:, ::-1]
        out.append((f"1WALL_tunnel_{t.osm_id}_bore", v, f, np.c_[np.arange(len(v)) % m, np.repeat(np.arange(len(rows)), m) * STEP / 4.0], "tunnel_wall", True))
        lamps = []
        for s in np.arange(t.s0 + 3.0, t.s1 - 2.0, LAMP_EVERY):
            i = int(np.argmin(np.abs(sec.s - s))); c, n, l, r = _frame(sec, i)
            d = np.array([-n[1], n[0]]); zt = sec.z[i] + bore_height(sec, i) - 0.6
            for lat in (-(l + r) * 0.22, (l + r) * 0.22):
                p = c + n * lat
                q = [p - d * 0.6 - n * 0.12, p + d * 0.6 - n * 0.12, p + d * 0.6 + n * 0.12, p - d * 0.6 + n * 0.12]
                lamps.append([[*e, zt] for e in q])
        if lamps:
            lv = np.array(lamps, float).reshape(-1, 3); lf = np.arange(len(lv), dtype=np.int32).reshape(-1, 4)
            out.append((f"tunnel_{t.osm_id}_lamps", lv, np.r_[lf, lf[:, ::-1]], lv[:, :2], "tunnel_lamp", True))
        sp = (spec or {}).get(str(t.osm_id), {})
        for tag, s, sgn in (("a", t.s0, -1.0), ("b", t.s1, 1.0)):
            out += _portal(sec, t, tag, s, sgn, ground_z, sp.get("plate", t.name), sp.get("portal", "concrete"))
    return out


def _portal(sec, t, tag, s, sgn, ground_z, plate, mat="concrete"):
    """坑口：s の所で道を横切る縦の面。筒の断面の所は穴、その外は下（路面−1m か外の地形の下）から上（内外の地形の高い方＋0.5m）まで。"""
    i = int(np.argmin(np.abs(sec.s - s))); c, n, l, r = _frame(sec, i); zc = sec.z[i]
    d = np.array([-n[1], n[0]])
    prof = _profile(sec, i)
    arch = lambda lat: max([dz for (la, dz) in prof if abs(la - lat) < 1e-6] or [np.interp(lat, [p[0] for p in prof[1:-1]], [p[1] for p in prof[1:-1]])])
    lats = np.r_[np.arange(-r - PORTAL_WING, -r, 1.0), [p[0] for p in prof[1:-1]], np.arange(l + 1.0, l + PORTAL_WING + 1e-6, 1.0)]
    lats = np.unique(np.round(lats, 6))
    V, F = [], []
    for k, lat in enumerate(lats):
        p = c + n * lat
        k_in = PORTAL_INSET + 1.5   # 山を残す所（坑口より PORTAL_INSET 奥から）の高さまで面を立てる
        g_in = float(np.atleast_1d(ground_z(np.array([p[0] - sgn * d[0] * k_in]), np.array([p[1] - sgn * d[1] * k_in])))[0])
        g_out = float(np.atleast_1d(ground_z(np.array([p[0] + sgn * d[0] * 1.0]), np.array([p[1] + sgn * d[1] * 1.0])))[0])
        top = max(g_in, g_out, zc + bore_height(sec, i)) + 0.5
        bot = zc + arch(lat) if -r - 1e-6 <= lat <= l + 1e-6 else min(zc - 1.0, g_out - 0.5, g_in - 0.5)
        V += [[*p, bot], [*p, top]]
        if k:
            a = 2 * (k - 1); F.append([a, a + 2, a + 3, a + 1])
    v = np.array(V, float); f = np.array(F, np.int32)
    out = [(f"tunnel_{t.osm_id}_portal_{tag}", v, np.r_[f, f[:, ::-1]], np.c_[np.repeat(lats, 2) / 3.0, v[:, 2] / 3.0], mat, True)]
    if plate:
        zt = zc + bore_height(sec, i) + 0.3; q = c + d * sgn * 0.05
        pv = np.array([[*(q - n * 1.6), zt], [*(q + n * 1.6), zt], [*(q + n * 1.6), zt + 0.7], [*(q - n * 1.6), zt + 0.7]])
        pf = np.array([[0, 1, 2, 3]] if sgn < 0 else [[3, 2, 1, 0]], np.int32)
        out.append((f"tunnel_{t.osm_id}_plate_{tag}", pv, np.r_[pf, pf[:, ::-1]], np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), f"tunnel_plate_{t.osm_id}", True))
    return out


PORTAL_INSET = 2.5   # 山を残す所（terrain.carve の keep_rows）は、坑口からこれだけ奥から（2m の地形の格子が坑口の前へはみ出さない）


def carve_mask(s, tunnels):
    """terrain.carve の keep_rows：トンネルの範囲を両端 PORTAL_INSET ずつ縮めたもの。"""
    return row_mask(s, tunnels, margin_m=-PORTAL_INSET)
