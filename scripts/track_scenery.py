"""コース全体（S1〜S3）から半径 200m の建物を data/manual/track_buildings.json に書く（buildings.py の形）。
地理院の外周線と OSM の建物を合わせ、町の範囲（名古木交差点。1棟ずつ確かめた s3_buildings.json がある）、蓑毛の家
（s2_minoge.json）、駐車場・峠・蓑毛の広場とは重ねない。屋根の色は空中写真。Street View で確かめた建物（checked=true）は残す。"""
import json

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from yabitsu import config, footprints, geo, gsi, nagoki, osm, scenery, side_roads, site
from yabitsu.crosssection import Section
from yabitsu.houses import footprint as house_fp
from yabitsu.start_lot import Frame

P = config.DATA_PROCESSED
BBOX = (35.3740, 139.2030, 35.4330, 139.2440)


def _ll(poly):
    c = np.asarray(poly.exterior.coords)
    lat, lon = geo.from_local(c[:, 0], c[:, 1])
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]


def main():
    sec = Section.load_json(P / "track_section.json")
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    band = LineString(np.c_[sec.x, sec.y]).buffer(scenery.BAND_M)
    el = side_roads.load_elements() + osm.fetch_overpass(osm.TRACK_BUILDINGS_QUERY, config.DATA_RAW / "osm_track_buildings.json")["elements"]
    polys = footprints.merged(footprints.gsi_polygons(BBOX, area=band), footprints.osm_polygons(el))
    minoge = json.loads((config.DATA_MANUAL / "s2_minoge.json").read_text(encoding="utf-8"))
    taken = [house_fp(Frame(site.ll_to_xy([b["center"]])[0], b["axis_deg"]), b["size"]).buffer(1.0) for b in minoge.get("buildings", [])]
    taken += [Polygon(p["polygon"]) for p in pads if p.get("kind") in ("lot", "pass", "site", "plaza", "branch") and "house" not in p["name"]]
    taken = unary_union(taken + [nagoki.circle()])
    path = config.DATA_MANUAL / "track_buildings.json"
    old = {b["id"]: b for b in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    cache = gsi.TileCache()
    out, seen = [], set()
    for poly, src, tags in polys:
        if poly.area < 8.0 or not band.contains(poly.centroid) or poly.intersects(taken):
            continue
        c = poly.centroid
        bid = f"osm{tags['_id']}" if src == "osm" else f"g{int(round(c.x))}_{int(round(c.y))}".replace("-", "m")
        if bid in seen:
            continue
        seen.add(bid)
        if bid in old and old[bid].get("checked"):
            out.append(old[bid]); continue
        d = scenery.default_spec(bid, _ll(poly), poly.area, footprints.roof_rgb(cache, poly))
        d["name"] = tags.get("name", "")
        out.append(d)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(out)} buildings within {scenery.BAND_M:.0f} m -> {path}")


if __name__ == "__main__":
    main()
