"""S1 の今の状態（道の中心・スタートなどの位置・菜の花台と峠まわり）を data/manual/s1_reference.json に書く。
区間をつなぐ作り替えの前に1回だけ動かし、作り替えのあとで S1 が変わっていないかを確かめるのに使う。"""
import json

import numpy as np

from yabitsu import config, geo, pass_area, s1_layouts, start_lot
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def _ll(x, y):
    lat, lon = geo.from_local(x, y)
    return [round(float(lat), 8), round(float(lon), 8)]


def main():
    sec = Section.load_json(P / "s1_section.json")
    pads = json.loads((P / "s1_pads.json").read_text(encoding="utf-8"))
    marks = pass_area.layout_marks(pass_area.load(config.DATA_MANUAL / "s1_pass.json"), sec, config.S1_PASS)
    cl = [[round(float(s), 3), *_ll(x, y), round(float(z), 3)]
          for s, x, y, z in zip(sec.s[::10], sec.x[::10], sec.y[::10], sec.z[::10])]
    layouts = {}
    for name, direction in s1_layouts.LAYOUTS.items():
        layouts[name] = {d.name: [*_ll(d.location[0], d.location[1]), round(float(d.location[2]), 3), round(float(d.heading), 5)]
                         for d in s1_layouts.layout_objects(sec, pads, direction, **marks)}
    lot = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    f = start_lot.frame(lot)
    lane = next(p for p in pads if p["name"] == "pass_lane")
    lot_pad = next(p for p in pads if p.get("kind") == "lot")
    sites = {"lot_origin": _ll(*lot["origin"]), "lot_z": round(float(lot_pad["z"]), 3),
             "tower": _ll(*f.xy(lot["tower"]["a"], lot["tower"]["b"])),
             "pass_lane_pits": [[*_ll(x, y), round(float(z), 3), round(float(h), 5)] for x, y, z, h in lane["pit_points"]]}
    out = {"note": "S1 の作り替え前の状態（2026-10-17、ブランチ feat/s1-pilot の最後）", "centerline": cl,
           "layouts": layouts, "sites": sites}
    path = config.DATA_MANUAL / "s1_reference.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(path, len(cl), "points", {k: len(v) for k, v in layouts.items()})


if __name__ == "__main__":
    main()
