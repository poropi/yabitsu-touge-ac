"""区間の今の状態（中心線・レイアウトのダミー・まわりの場所）を data/manual/{区間}_reference.json に書く。
区間を足す作り替えの前に1回だけ動かし、回帰テスト（tests/test_{区間}_regression.py）の参照にする。"""
import argparse
import json

import numpy as np

from yabitsu import config, geo, layouts, track
from yabitsu.acobjects import frame_at
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def _ll(x, y):
    lat, lon = geo.from_local(x, y)
    return [round(float(lat), 8), round(float(lon), 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("section", choices=["s2"])
    name = ap.parse_args().section
    sec = Section.load_json(P / "track_section.json")
    pads = json.loads((P / "track_pads.json").read_text(encoding="utf-8"))
    bounds = track.load_bounds()
    s0, s1 = bounds[name]
    path = sec.path()
    cl = []
    for s in np.arange(s0, s1, 10.0):
        x, y, z, _ = frame_at(path, float(s))
        cl.append([round(float(s - s0), 3), *_ll(x, y), round(z, 3)])
    lays = {}
    for l in layouts.load_layouts():
        if l.section == name:
            lays[l.name] = {d.name: [*_ll(*d.location[:2]), round(float(d.location[2]), 3), round(float(d.heading), 5)]
                            for d in layouts.layout_objects(sec, pads, l, bounds)}
    turn = next(p for p in pads if p["name"] == "minoge_turnaround")
    houses = []
    for p in pads:
        if p["name"].startswith("minoge_house_"):
            c = np.asarray(p["polygon"])[:-1].mean(0)
            houses.append([*_ll(*c), round(float(p["z"]), 3)])
    ref = {"note": f"{name} の作り替え前の状態（scripts.snapshot_section で作成）", "centerline": cl, "layouts": lays,
           "sites": {"turnaround_pits": [[*_ll(x, y), round(z, 3), round(h, 5)] for x, y, z, h in turn["pit_points"]],
                     "houses": houses}}
    out = config.DATA_MANUAL / f"{name}_reference.json"
    out.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(out, len(cl), "points", {k: len(v) for k, v in lays.items()}, len(houses), "houses")


if __name__ == "__main__":
    main()
