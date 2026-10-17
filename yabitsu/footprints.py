"""建物の外周線：地理院のベクトルタイル（experimental_fgd の BldL）と OSM の建物。重なるものは OSM の形と名前を使う。"""
import json
import math

import numpy as np
import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from . import config, geo


def gsi_polygons(bbox, z=18):
    """bbox = (lat0, lon0, lat1, lon1) の地理院の建物の外周線を、ローカル座標の多角形にする（タイルはキャッシュ）。"""
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    cache = config.DATA_RAW / "gsi_fgd"; cache.mkdir(parents=True, exist_ok=True)
    lines = []
    for x in range(tx(bbox[1]), tx(bbox[3]) + 1):
        for y in range(ty(bbox[2]), ty(bbox[0]) + 1):
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
    return list(polygonize(unary_union(lines))) if lines else []


def osm_polygons(elements):
    """OSM の建物（way）：[(多角形, tags)]。"""
    out = []
    for w in elements:
        t = w.get("tags", {})
        if w.get("type") != "way" or "building" not in t or len(w.get("geometry", [])) < 4:
            continue
        p = np.array([[g["lat"], g["lon"]] for g in w["geometry"]])
        x, y = geo.to_local(p[:, 0], p[:, 1])
        poly = Polygon(np.c_[x, y]).buffer(0)
        if poly.area > 4.0:
            out.append((poly, {**t, "_id": w.get("id")}))
    return out


def merged(gsi, osm):
    """[(多角形, 出どころ "gsi"/"osm", tags)]。地理院の外周線と半分以上重なる OSM の建物は、OSM の形と名前で置き換える。"""
    out, used = [], set()
    for g in gsi:
        best = None
        for k, (o, t) in enumerate(osm):
            if k in used or not g.intersects(o):
                continue
            a = g.intersection(o).area / max(min(g.area, o.area), 1e-9)
            if a > 0.5 and (best is None or a > best[0]):
                best = (a, k)
        if best:
            used.add(best[1]); o, t = osm[best[1]]
            out.append((o, "osm", t))
        else:
            out.append((g, "gsi", {}))
    out += [(o, "osm", t) for k, (o, t) in enumerate(osm) if k not in used]
    return out
