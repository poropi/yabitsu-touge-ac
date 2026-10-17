"""コースから半径 200m の景色：建物（地理院の外周線＋OSM、形はそのまま）と、走れない道（OSM の車道・歩道を地形の上に）。
町の範囲（名古木交差点）・本線・分かれ道・新東名・蓑毛の家とは重ねない。"""
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

from . import geo, osm

BAND_M = 200.0
WIDTH = {"trunk": 7.0, "primary": 6.5, "secondary": 6.5, "tertiary": 5.5, "unclassified": 4.5, "residential": 4.0,
         "living_street": 4.0, "service": 3.0, "track": 2.5, "footway": 1.5, "path": 1.2, "steps": 1.5, "cycleway": 2.0,
         "pedestrian": 3.0, "trunk_link": 4.5, "primary_link": 4.5, "secondary_link": 4.5, "tertiary_link": 4.0}
LIFT = 0.08
ROOF_PALETTE = ["#4A4A4A", "#6E6E6E", "#9A9A9A", "#C8C8C8", "#5A4034", "#7A5A44", "#3A5A8C", "#5E8C6A", "#A8423A", "#8C8478"]
WALL_PALETTE = ["#EDEBE6", "#E9E2CF", "#D8CCB4", "#C9C9C6", "#A88E74", "#E6E1D8"]


def road_polygons(elements, keep_out):
    """県道70号・高速道路を除く OSM の道の線を、種類ごとの幅の多角形にして合わせ、keep_out の所を除く。"""
    r70 = {id(w) for w in osm.route70_ways(elements)}
    parts = []
    for w in elements:
        t = w.get("tags", {}); hw = t.get("highway", "")
        if w.get("type") != "way" or id(w) in r70 or hw not in WIDTH or len(w.get("geometry", [])) < 2:
            continue
        if t.get("tunnel", "no") != "no" or t.get("bridge", "no") != "no":
            continue
        p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
        x, y = geo.to_local(p[:, 0], p[:, 1])
        parts.append(LineString(np.c_[x, y]).buffer(WIDTH[hw] / 2, cap_style="flat"))
    return unary_union(parts).difference(keep_out) if parts else unary_union([])


def _pick(palette, rgb):
    return min(palette, key=lambda h: float(np.sum((rgb - np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])) ** 2)))


def default_spec(bid, footprint_ll, area, rgb):
    """Street View で確かめていない建物の値：大きさから階数・屋根の形、屋根の色は空中写真、壁の色は決まった色から1つ。"""
    if area < 40:
        storeys, roof = 1, "gable"
    elif area <= 250:
        storeys, roof = 2, "gable"
    else:
        storeys, roof = (3 if area > 600 else 2), "flat"
    h = sum((i + 1) * ord(ch) for i, ch in enumerate(bid))
    return {"id": bid, "name": "", "footprint": footprint_ll, "storeys": storeys,
            "height_m": round(3.0 * storeys + (0.0 if roof == "flat" else 0.3), 1),
            "roof": {"type": roof, "rise_m": 1.4 if area < 40 else 1.8, "color": _pick(ROOF_PALETTE, rgb)},
            "wall": WALL_PALETTE[h % len(WALL_PALETTE)], "windows": area >= 30, "shopfront": [], "signs": [],
            "canopy": None, "lot": None, "checked": False}
