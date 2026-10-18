"""S5（札掛 → 宮ヶ瀬北原）の道ばたの物を data/manual/s5_objects.json に書く（s4_objects.json と同じ形）。
ストリートビューのメモ（S5 の s、L/R＝札掛から北原へ走って左/右）から、道の縁＋側溝＋0.6m の所に置く。"""
import json
import math

import numpy as np

from yabitsu import config, geo, track
from yabitsu.crosssection import GUTTER_WIDTH, Section

OUT_M = 0.6
# (s, 左右, 種類, その他)。mirror は道の方へ向ける。標識は、左は北原へ向かう車に、右は札掛へ向かう車に向ける（facing で変える）
ITEMS = [
    (700, "L", "mirror", {}), (706, "L", "mirror", {}), (712, "L", "road_sign", {"sign": "warning"}),
    (1100, "R", "mirror", {}), (1210, "R", "mirror", {}), (1500, "R", "mirror", {}), (1506, "R", "road_sign", {"sign": "warning"}),
    (1860, "L", "mirror", {}), (200, "R", "road_sign", {"sign": "warning"}),
    (3100, "R", "mirror", {}), (3300, "R", "mirror", {}), (3700, "R", "mirror", {}), (3950, "L", "mirror", {}),
    (4108, "R", "road_sign", {"sign": "speed30", "facing": "fudakake"}),
    (4750, "R", "road_sign", {"sign": "warning"}), (5050, "R", "mirror", {}), (5150, "R", "mirror", {}),
    (5500, "R", "road_sign", {"sign": "warning"}), (5750, "R", "mirror", {}), (5850, "R", "mirror", {}),
    (6800, "R", "mirror", {}), (7200, "R", "mirror", {}), (8650, "R", "mirror", {}),
    (9050, "L", "road_sign", {"sign": "warning"}), (9350, "R", "road_sign", {"sign": "warning"}),
    (11400, "R", "road_sign", {"sign": "speed40"}),
]
POLE_SIGNS = [   # キャンプ場の入口の青い看板（名前は OSM）と塩水橋の名前の柱
    (7690, "R", "国際一ノ瀬\nキャンプ場", 1.4, 0.8), (4558, "R", "塩水橋", 0.5, 1.0),
]
HOARDING = (4540, 4560, "R", 2.0)   # 塩水橋の分かれ道の工事の囲い（橋の上には置かない）


def main():
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json"); s0 = float(track.load_bounds()["s5"][0])
    bearing = lambda v: round((math.degrees(math.atan2(v[0], v[1])) + 360.0) % 360.0, 1)

    def frame(s, side, out=OUT_M):
        i = int(np.argmin(np.abs(sec.s - (s0 + s)))); h = sec.heading[i]
        n = np.array([-np.sin(h), np.cos(h)]); t = np.array([np.cos(h), np.sin(h)])
        sg = 1.0 if side == "L" else -1.0
        hw = sec.hw_left[i] if side == "L" else sec.hw_right[i]
        p = np.array([sec.x[i], sec.y[i]]) + n * sg * (hw + GUTTER_WIDTH + out)
        la, lo = geo.from_local(p[0], p[1])
        return [round(float(la), 8), round(float(lo), 8)], n * sg, t

    objs = []
    for s, side, kind, opt in ITEMS:
        ll, out, t = frame(s, side)
        if kind == "mirror":
            face = -out
        elif opt.get("facing") == "fudakake":   # 表が札掛の方を向く
            face = -t
        else:
            face = -t if side == "L" else t
        d = {"kind": kind, "points": [ll], "facing_deg": bearing(face), "note": f"ストリートビュー S5 {s}m {side}"}
        if kind == "road_sign":
            d["sign"] = opt["sign"]
        objs.append(d)
    a, b, side, h = HOARDING
    pts = [frame(s, side, 0.4)[0] for s in np.arange(a, b + 0.1, 5.0)]
    objs.append({"kind": "hoarding", "points": pts, "height_m": h, "note": "塩水橋の分かれ道の工事の囲い（ストリートビュー S5 4540〜4570m R、本物は青い川の絵の板）"})
    signs = []
    for s, side, text, w, hh in POLE_SIGNS:
        ll, out, t = frame(s, side, 1.0)
        signs.append({"text": text, "bg": "#1E50A0", "fg": "#FFFFFF", "at": ll, "facing_deg": bearing(-t), "w_m": w, "h_m": hh, "z_m": 1.6,
                      "note": f"ストリートビュー S5 {s}m {side}"})
    out = {"note": "S5 の道ばたの物（scripts/s5_roadside_objects.py で書く）", "objects": objs, "pole_signs": signs}
    (config.DATA_MANUAL / "s5_objects.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(len(objs), "objects,", len(signs), "pole signs")


if __name__ == "__main__":
    main()
