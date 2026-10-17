import json

import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union

from yabitsu import config, crosssection as cs, geo, gsi, minoge_area, pass_area, side_roads, start_lot, town, town_ground, track
from yabitsu.centerline import Centerline

P = config.DATA_PROCESSED


def main():
    cl = Centerline.load(P / "track_centerline.npz")
    prof = np.load(P / "track_profile.npz", allow_pickle=True)
    widths = cs.parse_width_rows(track.load_section_rows("widths", track.load_bounds()))
    sec = cs.build_section(cl, prof["z"], widths)
    sec.save_json(P / "track_section.json")
    pads = cs.pads_from_parking(json.loads((P / "track_parking.json").read_text(encoding="utf-8")), sec)
    lot = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    if lot:
        pads = start_lot.apply(pads, lot, sec)
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
    pads = pads + [q for st in side_roads.for_track(sec, pads) for q in st.pads()]  # 分かれ道の地形合わせ
    # 町の範囲（名古木交差点）：地面の高さを作って保存し、町のピットを足す
    c = tuple(float(v) for v in geo.to_local(*config.NAGOKI))
    ground = town_ground.build(sec, gsi.default_sampler(), c)
    ground.save(P / "track_town_ground.npz")
    spec = json.loads((config.DATA_MANUAL / "s3_nagoki.json").read_text(encoding="utf-8"))
    pads = pads + town.Town(spec, side_roads.load_elements(), sec, ground, c, config.TOWN_RADIUS_M).pads()
    (P / "track_pads.json").write_text(json.dumps(pads, ensure_ascii=False), encoding="utf-8")
    print(f"section rows={sec.n} width {2 * sec.hw_left.min():.1f}-{2 * sec.hw_left.max():.1f} m, "
          f"max slope {np.abs(sec.slope).max() * 100:.1f}%, pads={[(p['name'], p['role']) for p in pads]}")


if __name__ == "__main__":
    main()
