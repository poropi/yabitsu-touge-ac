"""S3 の建物のデータを作る。
- 町の範囲（名古木交差点から 200m）の建物 → data/manual/s3_buildings.json（1棟ずつ。check=true の建物は Street View で
  確かめる：見るための地点と向きを data/processed/review/s3_buildings/jobs.json に書く）
- それより先の、S3 の道から 40m 以内の建物 → data/manual/s3_houses.json（蓑毛と同じ形）
屋根の色は空中写真の外周線の中の画素の中央値。すでにある s3_buildings.json の、Claude が書いた値（checked=true）は残す。"""
import json
import math

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point

from yabitsu import config, footprints, geo, gsi, photo, side_roads, site, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED
NEAR_M, CHECK_M = 40.0, 45.0
BBOX = (35.3775, 139.2250, 35.4165, 139.2415)
ROOF = {"brown": (0.45, 0.30, 0.22), "grey": (0.42, 0.42, 0.42), "blue": (0.25, 0.35, 0.50), "red": (0.55, 0.22, 0.18)}
# Street View で確かめていない建物の壁の色（日本の家に多い色。建物ごとに決まった色を選ぶ）
WALL_PALETTE = ["#EDEBE6", "#E9E2CF", "#D8CCB4", "#C9C9C6", "#A88E74", "#E6E1D8"]


def hash_id(s):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s))


# Street View で確かめていない町の建物の屋根の色は、この中で空中写真の色に一番近いものにする（絵の枚数を減らす）
ROOF_PALETTE = ["#4A4A4A", "#6E6E6E", "#9A9A9A", "#C8C8C8", "#5A4034", "#7A5A44", "#3A5A8C", "#5E8C6A", "#A8423A", "#8C8478"]


def _ll(poly):
    c = np.asarray(poly.exterior.coords)
    lat, lon = geo.from_local(c[:, 0], c[:, 1])
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]


def _roof_rgb(cache, poly):
    x0, y0, x1, y1 = poly.bounds
    size = max(x1 - x0, y1 - y0) + 2.0
    img = photo.render_local_image(cache, geo.from_local, x0 - 1.0, y0 - 1.0, size, 0.5)
    n = img.shape[0]
    m = Image.new("L", (n, n), 0)
    inner = poly.buffer(-0.5)
    ring = site.largest(inner).exterior if not inner.is_empty else poly.exterior
    pts = [((x - (x0 - 1.0)) / 0.5, (y0 - 1.0 + size - y) / 0.5) for x, y in ring.coords]
    ImageDraw.Draw(m).polygon(pts, fill=255)
    px = img[np.array(m) > 0]
    return np.median(px, axis=0) / 255.0 if len(px) else np.array([0.45, 0.45, 0.45])


def main():
    sec = Section.load_json(P / "track_section.json")
    s3 = track.load_bounds()["s3"]
    cx, cy = [float(v) for v in geo.to_local(*config.NAGOKI)]
    center = Point(cx, cy)
    polys = footprints.merged(footprints.gsi_polygons(BBOX), footprints.osm_polygons(side_roads.load_elements()))
    m = sec.s <= s3[1]
    road = LineString(np.c_[sec.x[m], sec.y[m]])
    drive = LineString(np.c_[sec.x[sec.s <= 260.0], sec.y[sec.s <= 260.0]]).buffer(25.0)   # 交差点と県道70号（町の範囲）
    from shapely.ops import unary_union
    from yabitsu import town
    streets = unary_union([road] + [town._way_line(w) for w in side_roads.load_elements()
                                    if w.get("type") == "way" and w.get("tags", {}).get("highway") in town.ROAD_DEFAULTS
                                    and town._way_line(w).distance(center) < config.TOWN_RADIUS_M + 30])
    cache = gsi.TileCache()
    path = config.DATA_MANUAL / "s3_buildings.json"
    old = {b["id"]: b for b in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    town, houses, jobs = [], [], []
    for poly, src, tags in polys:
        if poly.area < 12.0:
            continue
        c = poly.centroid
        rgb = _roof_rgb(cache, poly)
        if c.distance(center) <= config.TOWN_RADIUS_M:
            bid = f"osm{tags['_id']}" if src == "osm" else f"gsi{int(round(c.x))}_{int(round(c.y))}".replace("-", "m")
            if bid in old and old[bid].get("checked"):
                town.append(old[bid]); continue
            levels = int(float(tags.get("building:levels", 0) or 0)) or (1 if poly.area < 40 else 2)
            check = poly.distance(drive) < CHECK_M - 25.0 or bool(tags.get("name"))   # 交差点・県道70号から見える建物と、名前のある店など
            town.append({"id": bid, "name": tags.get("name", ""), "source": src, "footprint": _ll(poly),
                         "storeys": levels, "height_m": round(3.0 * levels + (0.0 if poly.area > 250 else 0.5), 1),
                         "roof": {"type": "flat" if poly.area > 250 else "gable", "rise_m": 1.6,
                                  "color": min(ROOF_PALETTE, key=lambda h: float(np.sum((rgb - np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])) ** 2)))},
                         "wall": WALL_PALETTE[hash_id(bid) % len(WALL_PALETTE)], "windows": True, "shopfront": [], "signs": [], "canopy": None, "lot": None,
                         "check": bool(check), "checked": False, "sv": "", "sv_date": "", "note": ""})
            if check:
                # 見る地点：建物に一番近い、車道（OSM の道の線）の上の点
                p = streets.interpolate(streets.project(c)) if streets.geom_type == "LineString" else \
                    min((g.interpolate(g.project(c)) for g in streets.geoms), key=lambda q: q.distance(c))
                yaw = (math.degrees(math.atan2(c.x - p.x, c.y - p.y)) + 360.0) % 360.0
                la, lo = geo.from_local(p.x, p.y)
                jobs.append([bid, round(float(la), 7), round(float(lo), 7), round(yaw, 1)])
        elif poly.distance(road) <= NEAR_M:
            (pxy, axis_deg, (L, W)) = site.footprint_box(poly)
            if c.distance(center) <= config.TOWN_RADIUS_M + 10:
                continue
            lat, lon = geo.from_local(*pxy)
            colour = min(ROOF, key=lambda k: float(np.sum((rgb - np.array(ROOF[k])) ** 2)))
            houses.append({"center": [round(float(lat), 7), round(float(lon), 7)], "axis_deg": round(axis_deg, 1),
                           "size": [round(L, 1), round(W, 1)], "roof_h": 1.8 if L > 9 else 1.4, "eaves": 0.5,
                           "roof": colour, "wall": ("white", "cream", "beige", "grey", "brown")[hash_id(f"{c.x:.0f}_{c.y:.0f}") % 5],
                           "source": src})
    path.write_text(json.dumps(town, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    (config.DATA_MANUAL / "s3_houses.json").write_text(json.dumps({"buildings": houses}, ensure_ascii=False, indent=1) + "\n",
                                                       encoding="utf-8", newline="\n")
    jp = P / "review" / "s3_buildings"; jp.mkdir(parents=True, exist_ok=True)
    (jp / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    print(f"town buildings {len(town)} (to check {len(jobs)}), roadside houses {len(houses)}")


if __name__ == "__main__":
    main()
