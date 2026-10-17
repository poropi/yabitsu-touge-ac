"""町の範囲（名古木交差点）の確認画像：空中写真に、車道（黄）・歩道（水色）・路面の表示（白）・封鎖の線（赤）・
OSM の道の id を重ねる。review/s3_town.png（全体）と review/s3_town_center.png（交差点の中心 80m 四方を拡大）。"""
import json

import numpy as np
from PIL import Image, ImageDraw

from yabitsu import config, geo, gsi, photo, side_roads, town, town_ground
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def load_town(sec=None, ground=None):
    sec = sec or Section.load_json(P / "track_section.json")
    spec = json.loads((config.DATA_MANUAL / "s3_nagoki.json").read_text(encoding="utf-8"))
    cx, cy = geo.to_local(*config.NAGOKI)
    c = (float(cx), float(cy))
    gpath = P / "track_town_ground.npz"
    if ground is None:
        ground = town_ground.TownGround.load(gpath) if gpath.exists() else town_ground.build(sec, gsi.default_sampler(), c)
    return town.Town(spec, side_roads.load_elements(), sec, ground, c, config.TOWN_RADIUS_M), spec


def _draw(t, spec, x0, y0, size, px, ids=True):
    img = photo.render_local_image(gsi.TileCache(), geo.from_local, x0, y0, size, px)
    im = Image.fromarray(img).convert("RGB"); d = ImageDraw.Draw(im, "RGBA")
    to_px = lambda c: [((x - x0) / px, (y0 + size - y) / px) for x, y in c]

    def poly(g, col, fill=None):
        for pg in town._polys(g):
            d.polygon(to_px(pg.exterior.coords), outline=col, fill=fill)
            for h in pg.interiors:
                d.polygon(to_px(h.coords), outline=col)
    poly(t.carriageway, (255, 220, 0, 255), (255, 220, 0, 40))
    poly(t.sidewalk, (0, 220, 255, 255), (0, 220, 255, 40))
    for g, mat in town.marking_polygons(spec):
        poly(g, (255, 255, 255, 255), (255, 255, 255, 160))
    for c in spec.get("closures", []):
        d.line(to_px(town._xy([c["a"], c["b"]])), fill=(255, 0, 0, 255), width=3)
    if ids:
        for w in side_roads.load_elements():
            if w.get("type") == "way" and w.get("tags", {}).get("highway") in town.ROAD_DEFAULTS:
                ln = town._way_line(w)
                m = ln.interpolate(0.5, normalized=True)
                (u, v), = to_px([m.coords[0]])
                if 0 <= u < im.width and 0 <= v < im.height:
                    d.text((u, v), str(w["id"]), fill=(255, 80, 80, 255))
    return im


def main():
    t, spec = load_town()
    cx, cy = t.center
    R = config.TOWN_RADIUS_M + 20
    _draw(t, spec, cx - R, cy - R, 2 * R, 0.4).save(P / "review" / "s3_town.png")
    _draw(t, spec, cx - 40, cy - 40, 80, 0.1, ids=False).save(P / "review" / "s3_town_center.png")
    print("written", P / "review" / "s3_town.png", "main_skip", t.main_skip,
          f"carriageway {t.carriageway.area:.0f} m², sidewalk {t.sidewalk.area:.0f} m²")


if __name__ == "__main__":
    main()
