"""蓑毛のバス停まわりの家（地理院のベクトルタイルの建物の外周線。OSM はこのあたりの家が少ししか入っていない）を data/manual/s2_minoge.json の buildings に書く（ほかのキーは残す）。
S2 の始点から 500m までの道から 40m 以内の建物だけ。"""
import json
import math

import numpy as np
import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from yabitsu import config, geo, site, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED
NEAR_M, RANGE_M = 40.0, 500.0
BBOX = (35.4130, 139.2240, 35.4180, 139.2310)



def _gsi_polygons(z=18):
    """地理院ベクトルタイル（experimental_fgd）の建物の外周線（BldL）をつないで多角形にする。"""
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    cache = config.DATA_RAW / "gsi_fgd"; cache.mkdir(parents=True, exist_ok=True)
    lines = []
    for x in range(tx(BBOX[1]), tx(BBOX[3]) + 1):
        for y in range(ty(BBOX[2]), ty(BBOX[0]) + 1):
            f = cache / f"{z}_{x}_{y}.geojson"
            if not f.exists():
                r = requests.get(f"https://cyberjapandata.gsi.go.jp/xyz/experimental_fgd/{z}/{x}/{y}.geojson", timeout=30)
                f.write_bytes(r.content if r.ok else b'{"features": []}')
            for feat in json.loads(f.read_text(encoding="utf-8"))["features"]:
                if feat["properties"].get("class") != "BldL":
                    continue
                gm = feat["geometry"]
                for c in (gm["coordinates"] if gm["type"] == "MultiLineString" else [gm["coordinates"]]):
                    c = np.asarray(c); lx, ly = geo.to_local(c[:, 1], c[:, 0])
                    lines.append(LineString(np.c_[lx, ly]))
    return [(p, "gsi") for p in polygonize(unary_union(lines))]


def main():
    sec = Section.load_json(P / "track_section.json")
    s0 = track.load_bounds()["s2"][0]
    m = (sec.s >= s0) & (sec.s <= s0 + RANGE_M)
    road = LineString(np.c_[sec.x[m], sec.y[m]])
    polys = _gsi_polygons()
    out = []
    for poly, src in polys:
        if poly.distance(road) > NEAR_M or poly.area < 12:
            continue
        (cx, cy), axis_deg, (L, W) = site.footprint_box(poly)
        lat, lon = geo.from_local(cx, cy)
        out.append({"center": [round(float(lat), 7), round(float(lon), 7)], "axis_deg": round(axis_deg, 1),
                    "size": [round(L, 1), round(W, 1)], "roof_h": 1.8 if L > 9 else 1.4,
                    "eaves": 0.5, "source": src})
    path = config.DATA_MANUAL / "s2_minoge.json"
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    spec["buildings"] = out
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"{len(out)} buildings -> {path}")


if __name__ == "__main__":
    main()
