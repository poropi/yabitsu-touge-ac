import json

import numpy as np

from shapely.geometry import Polygon
from shapely.ops import unary_union

from yabitsu import config, crosssection as cs, pass_area, start_lot
from yabitsu.centerline import Centerline

P = config.DATA_PROCESSED


def main():
    cl = Centerline.load(P / "s1_centerline.npz")
    prof = np.load(P / "s1_profile.npz")
    sec = cs.build_section(cl, prof["z"], cs.load_widths_csv(config.DATA_MANUAL / "s1_widths.csv"))
    sec.save_json(P / "s1_section.json")
    pads = cs.pads_from_parking(json.loads((P / "s1_parking.json").read_text(encoding="utf-8")), sec)
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
    (P / "s1_pads.json").write_text(json.dumps(pads, ensure_ascii=False), encoding="utf-8")
    print(f"section rows={sec.n} width {2 * sec.hw_left.min():.1f}-{2 * sec.hw_left.max():.1f} m, "
          f"max slope {np.abs(sec.slope).max() * 100:.1f}%, pads={[(p['name'], p['role']) for p in pads]}")


if __name__ == "__main__":
    main()
