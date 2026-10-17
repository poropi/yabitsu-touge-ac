import json

import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union

from yabitsu import config, crosssection as cs, gsi, minoge_area, nagoki, pass_area, side_roads, start_lot, town_ground, track
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
    plazas = json.loads((config.DATA_MANUAL / "s3_plazas.json").read_text(encoding="utf-8"))
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
                     "z": float(min(sec.edge_z(1)[i], sec.edge_z(-1)[i]))})
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
        for p in pads:  # 峠の駐車場（OSM）は、わき道・島・広場と重ならないように削る
            if p.get("kind") != "lot":
                g = Polygon(p["polygon"]).difference(taken)
                g = g if g.geom_type == "Polygon" else max(g.geoms, key=lambda q: q.area)
                p["polygon"] = [list(c) for c in g.exterior.coords]
        pads = pads + site_pads
    minoge = minoge_area.load(config.DATA_MANUAL / "s2_minoge.json")
    if minoge:
        pads = pads + minoge_area.build(minoge, sec, minoge_area.dem_ground()).pads()
    # 町の範囲（名古木交差点）：地面の高さを作って保存し、町のピットを足す
    ground = town_ground.build(sec, gsi.default_sampler(), nagoki.center())
    ground.save(P / "track_town_ground.npz")
    tw = nagoki.load_town(sec, ground)
    pads = pads + tw.pads()
    pads = pads + [q for b in nagoki.load_branches(sec) for q in b.pads()]   # 2本に分かれる所の、もう1本と島
    from shapely.ops import unary_union as _uu
    brs = nagoki.load_branches(sec)
    excl = _uu([tw.region] + [b.surface.buffer(0.3) for b in brs] + [b.island for b in brs if b.island is not None])
    pads = pads + [q for st in side_roads.for_track(sec, pads, exclude=excl) for q in st.pads()]  # 分かれ道の地形合わせ
    # 新東名など（トンネルの入口まで）：低い所は地形を路面に合わせる
    from yabitsu import motorway, overpasses
    for x in motorway.from_osm(side_roads.load_elements(), sec, gsi.default_sampler(), overpasses.load_overrides()):
        pads = pads + x.pads()
    (P / "track_pads.json").write_text(json.dumps(pads, ensure_ascii=False), encoding="utf-8")
    print(f"section rows={sec.n} width {2 * sec.hw_left.min():.1f}-{2 * sec.hw_left.max():.1f} m, "
          f"max slope {np.abs(sec.slope).max() * 100:.1f}%, pads={[(p['name'], p['role']) for p in pads]}")


if __name__ == "__main__":
    main()
