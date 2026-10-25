import json

import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union

from yabitsu import asama_area, config, crosssection as cs, gsi, minoge_area, nagoki, pass_area, side_roads, start_lot, town_ground, track
from yabitsu.centerline import Centerline

P = config.DATA_PROCESSED


def main():
    cl = Centerline.load(P / "track_centerline.npz")
    prof = np.load(P / "track_profile.npz", allow_pickle=True)
    widths = cs.parse_width_rows(track.load_section_rows("widths", track.load_bounds()))
    sec = cs.build_section(cl, prof["z"], widths)
    bays = cs.parse_bay_rows(track.load_section_rows("busbays", track.load_bounds()))   # バス停の切り込み（片側だけ広い）
    sec.hw_left, sec.hw_right = cs.apply_bays(sec.s, sec.hw_left, sec.hw_right, bays)
    sec.save_json(P / "track_section.json")
    pads = cs.pads_from_parking(json.loads((P / "track_parking.json").read_text(encoding="utf-8")), sec)
    lot = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    if lot:
        pads = start_lot.apply(pads, lot, sec)
    # 道ぞいの広場（data/manual/s3_plazas.json）：道の縁の高さの平らな舗装。道と重なる所は下で削る
    from shapely.geometry import Polygon as _Poly
    from yabitsu import site
    plazas = [pl for f in sorted(config.DATA_MANUAL.glob("s*_plazas.json"))   # 区間ごとの広場（s3・s4 …）
              for pl in json.loads(f.read_text(encoding="utf-8"))]
    # OSM の駐車場で、道から 50m 以内のもの（菜の花台・峠・町の範囲の駐車場は別に作っているので除く）
    from yabitsu import osm, nagoki as _ng
    from shapely.geometry import LineString as _LS
    lots = osm.fetch_overpass('[out:json][timeout:90];way["amenity"="parking"](35.374,139.203,35.433,139.244);out tags geom;',
                              config.DATA_RAW / "osm_track_parking.json")["elements"]
    main_line = _LS(np.c_[sec.x, sec.y]); made = [_Poly(p["polygon"]) for p in pads]
    for w in lots:
        pts = [[g["lat"], g["lon"]] for g in w["geometry"]]
        poly = _Poly(site.ll_to_xy(pts)).buffer(0)
        if poly.distance(main_line) > 50 or poly.intersects(_ng.circle()) or any(poly.intersects(q) for q in made):
            continue
        plazas.append({"name": f"osm_parking_{w['id']}", "points": pts})
    for pl in plazas:
        poly = _Poly(site.ll_to_xy(pl["points"])).buffer(0)
        c = poly.centroid; i = int(np.argmin(np.hypot(sec.x - c.x, sec.y - c.y)))
        pads.append({"name": pl["name"], "role": "flat", "kind": "plaza", "polygon": np.asarray(poly.exterior.coords).tolist(),
                     "z": float(min(sec.edge_z(1)[i], sec.edge_z(-1)[i])), **({"material": pl["material"]} if pl.get("material") else {})})   # material：砂利の広場など（既定はアスファルト）
    # OSM の駐車場が道にかぶるところは削る（峠の先へ道を延ばしたので、峠の駐車場が道に接する）
    road = start_lot.road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
    for p in pads:
        if p.get("kind") != "lot":
            g = Polygon(p["polygon"]).difference(road)
            g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)
            p["polygon"] = [list(c) for c in g.exterior.coords]
    site_spec = pass_area.load(config.DATA_MANUAL / "s1_pass.json")
    if site_spec:
        site_pads = pass_area.build(site_spec, sec).pads()
        taken = unary_union([Polygon(p["polygon"]) for p in site_pads]).buffer(0.5)
        park = [Polygon(p["polygon"]) for p in site_pads if p["name"] == "pass_parking"]
        if park:   # 峠の駐車場を航空写真の形で作ったので、重なる OSM の駐車場は丸ごと外す（削り残しを残さない）
            pads = [p for p in pads if p.get("kind") == "lot" or Polygon(p["polygon"]).buffer(0).intersection(park[0]).area
                    < 0.2 * Polygon(p["polygon"]).buffer(0).area]
        for p in pads:  # 峠の駐車場（OSM）は、わき道・島・広場と重ならないように削る
            if p.get("kind") != "lot":
                g = Polygon(p["polygon"]).difference(taken)
                g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)
                p["polygon"] = [list(c) for c in g.exterior.coords]
        pads = pads + site_pads
    minoge = minoge_area.load(config.DATA_MANUAL / "s2_minoge.json")
    if minoge:
        pads = pads + minoge_area.build(minoge, sec, minoge_area.dem_ground()).pads()
    asama = asama_area.load(config.DATA_MANUAL / "s2_asama.json")   # 浅間神社の入口の広場と石段
    if asama:
        pads = pads + asama_area.build(asama, sec).pads()
    from yabitsu import terayama_area
    tera = terayama_area.load(config.DATA_MANUAL / "s4_terayama.json")   # 寺山富士見橋：広場・わき道・トイレの台
    if tera:
        pads = pads + terayama_area.build(tera, sec).pads()
    from yabitsu import gomayashiki_area
    goma = gomayashiki_area.load(config.DATA_MANUAL / "s4_gomayashiki.json")   # 護摩屋敷の水：分かれ目の広場・湧き水・わき道の輪
    if goma:
        pads = pads + gomayashiki_area.build(goma, sec).pads()
    # 町の範囲（名古木交差点）：地面の高さを作って保存し、町のピットを足す
    ground = town_ground.build(sec, gsi.default_sampler(), nagoki.center())
    ground.save(P / "track_town_ground.npz")
    tw = nagoki.load_town(sec, ground)
    pads = pads + tw.pads()
    others = []   # ほかの交差点の町（宮ヶ瀬北原）：地面を作って保存し、ピットを足す
    for j in nagoki.available():
        if j is nagoki.NAGOKI:
            continue
        g = town_ground.build(sec, gsi.default_sampler(), nagoki.center(j)); g.save(P / j.ground)
        kt = nagoki.load_town(sec, g, j); others.append(kt); pads = pads + kt.pads()
    from shapely.ops import unary_union as _uu
    sps = nagoki.load_splits(sec)   # 島をはさんで2本に分かれる所（地図の線どおり。地形は track_06 で合わせる）
    excl = _uu([tw.region] + [k.region for k in others] + [z.area() for z in sps])
    pads = pads + [q for st in side_roads.for_track(sec, pads, exclude=excl) for q in st.pads()]  # 分かれ道の地形合わせ
    # 新東名など（トンネルの入口まで）：低い所は地形を路面に合わせる
    from yabitsu import motorway, overpasses
    for x in motorway.from_osm(side_roads.load_elements(), sec, gsi.default_sampler(), overpasses.load_overrides()):
        pads = pads + x.pads()
    # 札掛のピット（data/manual/s4_fudakake.json）：札掛橋の先の県道70号の上、峠の方を向けて4台
    fk = json.loads((config.DATA_MANUAL / "s4_fudakake.json").read_text(encoding="utf-8"))
    for name, key in (("fudakake_pits", "pits"), ("s4_pass_pits", "pass_pits")):   # 札掛（上り用）と、峠の先（S4 下り用）
        pts = []
        for lat, lon, deg in fk[key]:
            x, y = (float(v) for v in site.ll_to_xy([[lat, lon]])[0])
            i = int(np.argmin(np.hypot(sec.x - x, sec.y - y)))
            pts.append([x, y, float(sec.z[i]), float(np.radians(90.0 - deg))])
        hull = _uu([_Poly([(p[0] - 1, p[1] - 1), (p[0] + 1, p[1] - 1), (p[0] + 1, p[1] + 1), (p[0] - 1, p[1] + 1)]) for p in pts]).convex_hull
        pads.append({"name": name, "role": "flat", "kind": "site", "polygon": np.asarray(hull.exterior.coords).tolist(),
                     "z": float(np.mean([p[2] for p in pts])), "pit_points": pts})
    # S5 下り（札掛 → 宮ヶ瀬北原）のピット（data/manual/s5_fudakake_pits.json）：札掛橋の先、スタート列の手前に北原の方を向けて4台
    sp = json.loads((config.DATA_MANUAL / "s5_fudakake_pits.json").read_text(encoding="utf-8"))["pits"]
    pts = []
    for lat, lon, deg in sp:
        x, y = (float(v) for v in site.ll_to_xy([[lat, lon]])[0])
        i = int(np.argmin(np.hypot(sec.x - x, sec.y - y)))
        pts.append([x, y, float(sec.z[i]), float(np.radians(90.0 - deg))])
    hull = _uu([_Poly([(p[0] - 1, p[1] - 1), (p[0] + 1, p[1] - 1), (p[0] + 1, p[1] + 1), (p[0] - 1, p[1] + 1)]) for p in pts]).convex_hull
    pads.append({"name": "s5_fudakake_pits", "role": "flat", "kind": "site", "polygon": np.asarray(hull.exterior.coords).tolist(),
                 "z": float(np.mean([p[2] for p in pts])), "pit_points": pts})
    (P / "track_pads.json").write_text(json.dumps(pads, ensure_ascii=False), encoding="utf-8")
    print(f"section rows={sec.n} width {2 * sec.hw_left.min():.1f}-{2 * sec.hw_left.max():.1f} m, "
          f"max slope {np.abs(sec.slope).max() * 100:.1f}%, pads={[(p['name'], p['role']) for p in pads]}")


if __name__ == "__main__":
    main()
