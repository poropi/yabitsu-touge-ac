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
