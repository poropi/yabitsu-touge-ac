"""宮ヶ瀬北原交差点（S5 の終わり、町の範囲 半径 200m）の下書きを作る：
- OSM（交差点から 260m の道・建物・信号・横断歩道・バス停）→ data/raw/osm_kitahara.json
- 建物 → data/manual/s5_kitahara_buildings.json（名古木の s3_buildings.py と同じ形。checked=true の建物は残す）
  ストリートビューで見る地点：data/processed/review/s5_kitahara_buildings/jobs.json
- 町の spec がまだなければ data/manual/s5_kitahara.json（auto=true：横断歩道・停止線・封鎖は OSM から自動）"""
import json
import math

import numpy as np
from shapely.geometry import LineString, Point

from yabitsu import config, footprints, geo, gsi, osm, town
from scripts.s3_buildings import ROOF_PALETTE, WALL_PALETTE, _ll, _roof_rgb, hash_id

P = config.DATA_PROCESSED
CHECK_M = 60.0


def main():
    lat, lon = config.KITAHARA
    el = osm.fetch_overpass(osm.TOWN_QUERY.format(r=260, lat=lat, lon=lon), config.DATA_RAW / "osm_kitahara.json")["elements"]
    cx, cy = (float(v) for v in geo.to_local(lat, lon)); center = Point(cx, cy)
    r = config.TOWN_RADIUS_M + 40
    la0, lo0 = geo.from_local(cx - r, cy - r); la1, lo1 = geo.from_local(cx + r, cy + r)
    polys = footprints.merged(footprints.gsi_polygons((float(la0), float(lo0), float(la1), float(lo1))), footprints.osm_polygons(el))
    streets = [town._way_line(w) for w in el if w.get("type") == "way" and w.get("tags", {}).get("highway") in town.ROAD_DEFAULTS]
    cache = gsi.TileCache()
    path = config.DATA_MANUAL / "s5_kitahara_buildings.json"
    old = {b["id"]: b for b in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    out, jobs = [], []
    for poly, src, tags in polys:
        c = poly.centroid
        if poly.area < 12.0 or c.distance(center) > config.TOWN_RADIUS_M:
            continue
        bid = f"osm{tags['_id']}" if src == "osm" else f"gsi{int(round(c.x))}_{int(round(c.y))}".replace("-", "m")
        if bid in old and old[bid].get("checked"):
            out.append(old[bid]); continue
        rgb = _roof_rgb(cache, poly)
        levels = int(float(tags.get("building:levels", 0) or 0)) or (1 if poly.area < 40 else 2)
        near = min((s.distance(c) for s in streets), default=1e9)
        out.append({"id": bid, "name": tags.get("name", ""), "source": src, "footprint": _ll(poly),
                    "storeys": levels, "height_m": round(3.0 * levels + (0.0 if poly.area > 250 else 0.5), 1),
                    "roof": {"type": "flat" if poly.area > 250 else "gable", "rise_m": 1.6,
                             "color": min(ROOF_PALETTE, key=lambda h: float(np.sum((rgb - np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])) ** 2)))},
                    "wall": WALL_PALETTE[hash_id(bid) % len(WALL_PALETTE)], "windows": True, "shopfront": [], "signs": [], "canopy": None,
                    "lot": None, "check": near < CHECK_M, "checked": False, "sv": "", "sv_date": "", "note": ""})
        if near < CHECK_M:
            s = min(streets, key=lambda q: q.distance(c)); p = s.interpolate(s.project(c))
            la, lo = geo.from_local(p.x, p.y)
            jobs.append([bid, round(float(la), 7), round(float(lo), 7), round((math.degrees(math.atan2(c.x - p.x, c.y - p.y)) + 360) % 360, 1)])
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    jp = P / "review" / "s5_kitahara_buildings"; jp.mkdir(parents=True, exist_ok=True)
    (jp / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    sp = config.DATA_MANUAL / "s5_kitahara.json"
    if not sp.exists():
        spec = {"note": "宮ヶ瀬北原交差点（町の範囲 半径200m、S5 の終わり）。横断歩道・停止線・封鎖は auto=true で OSM から。"
                        "県道64号の鳥居原方面は、封鎖の柵の先も町の範囲いっぱいまで道（将来の延長）",
                "auto": True, "corner_radius_m": 8.0, "roads": {"main": {"sidewalk_l": 1.5, "sidewalk_r": 1.5}},
                "closures": [], "markings": [], "signals": [], "objects": [], "drivable_extra": [], "pits": [], "pole_signs": []}
        sp.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"OSM {len(el)} elements, buildings {len(out)} (to check {len(jobs)})")


if __name__ == "__main__":
    main()
