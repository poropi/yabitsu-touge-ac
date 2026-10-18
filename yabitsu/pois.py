"""道ぞいの小さな目印を OSM から置く：自動販売機、郵便ポスト、公衆電話、案内板、バス停の待合所、石碑。
店（shop）は、一番近い建物の、道に向いた壁に店名の看板を付ける。すでに手で置いた物の近く（skip）には置かない。"""
import math

import numpy as np
from shapely.geometry import LineString, Point

from . import geo, town_objects
from .town_objects import _box, _merge, _prism, _quad

NEAR_M = 50.0
SKIP_M = 3.0
KINDS = {("amenity", "vending_machine"): "vending_machine", ("amenity", "post_box"): "post_box",
         ("amenity", "telephone"): "telephone", ("tourism", "information"): "info_board", ("amenity", "shelter"): "shelter",
         ("historic", "memorial"): "stele"}


def _kind(tags):
    for (k, v), name in KINDS.items():
        if tags.get(k) == v:
            return name
    return None


def place(elements, sec, skip):
    """OSM の点を、道から NEAR_M 以内の物だけ選び、道の方を向けて並べる。skip（ローカル座標の点）の近くは除く。"""
    main = LineString(np.c_[sec.x, sec.y])
    out = []
    for e in elements:
        if e.get("type") != "node":
            continue
        kind = _kind(e.get("tags", {}))
        if kind is None:
            continue
        x, y = [float(v) for v in geo.to_local(e["lat"], e["lon"])]
        if main.distance(Point(x, y)) > NEAR_M or any(math.hypot(x - a, y - b) < SKIP_M for a, b in skip):
            continue
        x, y = (float(v) for v in town_objects.off_road_xy(sec, (x, y)))   # 道の上の点は、道の外へずらす
        q = main.interpolate(main.project(Point(x, y)))
        face = (math.degrees(math.atan2(q.x - x, q.y - y)) + 360.0) % 360.0
        out.append({"kind": kind, "xy": (x, y), "facing_deg": face, "name": e["tags"].get("name", ""), "osm_id": e.get("id")})
    return out


def meshes(items, ground_z, prefix="poi"):
    groups = {}
    add = lambda mat, part: groups.setdefault(mat, []).append(part)
    for it in items:
        p = np.asarray(it["xy"], float); z = float(np.atleast_1d(ground_z(np.atleast_1d(p[0]), np.atleast_1d(p[1])))[0])
        d = np.array([math.sin(math.radians(it["facing_deg"])), math.cos(math.radians(it["facing_deg"]))]); n = np.array([d[1], -d[0]])
        k = it["kind"]
        if k == "vending_machine":   # 自動販売機：白い箱＋正面の見本の窓
            add("vending", _box(p, d, 1.0, 0.75, z, z + 1.83))
            fc = p + d * 0.38
            add("vending_front", _quad(fc - n * 0.45, fc + n * 0.45, z + 0.9, z + 1.75))
        elif k == "post_box":        # 赤い郵便ポスト（円柱）
            add("post_red", _prism(p, 0.22, z, z + 1.1, 14))
            add("post_red", _box(p, d, 0.5, 0.5, z + 1.1, z + 1.2))
        elif k == "telephone":       # 公衆電話のボックス
            add("glass_grey", _box(p, d, 0.9, 0.9, z, z + 2.1))
            add("pole_grey", _box(p, d, 1.0, 1.0, z + 2.1, z + 2.25))
        elif k == "info_board":      # 案内板：脚2本と板
            for s in (-1, 1):
                add("wood", _box(p + n * s * 0.65, d, 0.1, 0.1, z, z + 1.9))
            add("info_board", _quad(p + d * 0.06 - n * 0.7, p + d * 0.06 + n * 0.7, z + 0.9, z + 1.85))
            add("wood", _quad(p - d * 0.01 + n * 0.7, p - d * 0.01 - n * 0.7, z + 0.9, z + 1.85))
        elif k == "shelter":         # バス停の待合所：柱4本・後ろの壁・屋根
            for s in (-1, 1):
                for t in (-1, 1):
                    add("pole_grey", _prism(p + n * s * 1.4 + d * t * 0.6, 0.05, z, z + 2.3, 6))
            add("glass_grey", _quad(p - d * 0.6 - n * 1.4, p - d * 0.6 + n * 1.4, z + 0.1, z + 2.2, both=True))
            add("roof_metal", _box(p, d, 3.2, 1.6, z + 2.3, z + 2.4))
        elif k == "stele":           # 石碑：台石と碑
            add("stone", _box(p, d, 1.0, 0.6, z - 0.1, z + 0.25))
            add("stone", _box(p, d, 0.7, 0.25, z + 0.25, z + 1.8))
    return [(f"{prefix}_{mat}", *_merge(parts), mat, True) for mat, parts in groups.items()]


def add_shop_signs(elements, buildings, sec, near_m=12.0):
    """OSM の店（shop）の名前を、一番近い建物（near_m 以内）の、道に向いた辺の看板にする。付けた数を返す。"""
    from . import buildings as bd
    main = LineString(np.c_[sec.x, sec.y])
    n = 0
    for e in elements:
        t = e.get("tags", {})
        if e.get("type") != "node" or "shop" not in t or not t.get("name"):
            continue
        p = Point(*[float(v) for v in geo.to_local(e["lat"], e["lon"])])
        if main.distance(p) > NEAR_M:
            continue
        b = min(buildings, key=lambda q: q.footprint.distance(p), default=None)
        if b is None or b.footprint.distance(p) > near_m or b.signs:
            continue
        q = main.interpolate(main.project(b.footprint.centroid))
        face = (math.degrees(math.atan2(q.x - b.footprint.centroid.x, q.y - b.footprint.centroid.y)) + 360.0) % 360.0
        b.signs.append({"text": t["name"], "bg": "#FFFFFF", "fg": "#1A1A1A", "edge": bd.edge_facing(b, face),
                        "w_m": min(5.0, 0.55 * len(t["name"]) + 0.6), "h_m": 0.8, "z_m": 2.6})
        n += 1
    return n
