"""建物の外周線：地理院のベクトルタイル（experimental_fgd の BldL）と OSM の建物。重なるものは OSM の形と名前を使う。"""
import json
import math

import numpy as np
import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from . import config, geo


def gsi_polygons(bbox, z=18, area=None):
    """bbox = (lat0, lon0, lat1, lon1) の地理院の建物の外周線を、ローカル座標の多角形にする（タイルはキャッシュ）。
    area（ローカル座標の多角形）を渡すと、それにかかるタイルだけ取る。"""
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    lon_of = lambda x: x / n * 360 - 180
    lat_of = lambda y: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    cache = config.DATA_RAW / "gsi_fgd"; cache.mkdir(parents=True, exist_ok=True)
    lines = []
    for x in range(tx(bbox[1]), tx(bbox[3]) + 1):
        for y in range(ty(bbox[2]), ty(bbox[0]) + 1):
            if area is not None:
                from shapely.geometry import box as _box
                x0, y0 = geo.to_local(lat_of(y + 1), lon_of(x)); x1, y1 = geo.to_local(lat_of(y), lon_of(x + 1))
                if not _box(float(x0), float(y0), float(x1), float(y1)).intersects(area):
                    continue
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


def overture_polygons(path):
    """Overture Maps の建物（data/raw/overture_buildings.json。日本は Microsoft の機械学習の輪郭が多い）：[(多角形, tags)]。"""
    import json
    out = []
    for f in json.loads(path.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        for rings in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            c = np.asarray(rings[0], float)
            x, y = geo.to_local(c[:, 1], c[:, 0])
            poly = Polygon(np.c_[x, y]).buffer(0)
            if poly.area > 8.0:
                out.append((poly, {"_id": f["id"], "_src": f.get("src")}))
    return out


def fill_gaps(base, extra, overlap=0.2):
    """base（merged の結果）に、どれとも重ならない（重なりが小さい方の面積の overlap 未満の）extra の建物を "ov" として足す。
    地理院の外周線（基盤地図情報・最適化ベクトルタイルとも）に無い家が名古木の道ぞいで何軒も抜け、芝生に見えた（2026-10-28）。"""
    from shapely.strtree import STRtree
    polys = [p for p, _, _ in base]
    tree = STRtree(polys) if polys else None
    out = list(base)
    for e, t in extra:
        hit = False
        for k in (tree.query(e) if tree is not None else []):
            b = polys[int(k)]
            if e.intersection(b).area > overlap * min(e.area, b.area):
                hit = True; break
        if not hit:
            out.append((e, "ov", t))
    return out


def roof_rgb(cache, poly, px=0.5):
    """空中写真の、外周線の内側（0.5m 内側）の画素の色の中央値（0〜1）。"""
    import numpy as np
    from PIL import Image, ImageDraw
    from . import photo, site
    x0, y0, x1, y1 = poly.bounds
    size = max(x1 - x0, y1 - y0) + 2.0
    img = photo.render_local_image(cache, geo.from_local, x0 - 1.0, y0 - 1.0, size, px)
    n = img.shape[0]
    m = Image.new("L", (n, n), 0)
    inner = poly.buffer(-0.5)
    ring = site.largest(inner).exterior if not inner.is_empty else poly.exterior
    ImageDraw.Draw(m).polygon([((x - (x0 - 1.0)) / px, (y0 - 1.0 + size - y) / px) for x, y in ring.coords], fill=255)
    q = img[np.array(m) > 0]
    return np.median(q, axis=0) / 255.0 if len(q) else np.array([0.45, 0.45, 0.45])
