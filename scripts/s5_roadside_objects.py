"""S5（札掛 → 宮ヶ瀬北原）の道ばたの物を data/manual/s5_objects.json に書く（s4_objects.json と同じ形）。
ストリートビューのメモ（S5 の s、L/R＝札掛から北原へ走って左/右）から、道の縁＋側溝＋0.6m の所に置く。"""
import json
import math

import numpy as np

from yabitsu import config, geo, track
from yabitsu.crosssection import GUTTER_WIDTH, Section

OUT_M = 0.6
# (s, 左右, 種類, その他)。mirror は道の方へ向ける。標識は、左は北原へ向かう車に、右は札掛へ向かう車に向ける（facing で変える）
# どれも S5 のストリートビュー（sv/s5_*_f.png＝北原へ向かう向き）で、写っている側を1つずつ確かめた（2026-10-19）
ITEMS = [   # s は、写っている写真の位置（sv_pos の行）＋写真の中の何 m 先か
    (709, "R", "mirror", {}), (711, "R", "mirror", {}), (710, "R", "road_sign", {"sign": "warning"}),   # ヘアピンの内側の土手（右）：写真 699m の 10m 先
    (1510, "R", "mirror", {}), (1508, "R", "road_sign", {"sign": "warning"}),                           # 写真 1496m の 14m 先
    (1875, "L", "mirror", {}),                                                                          # 写真 1850m の 25m 先
    (3117, "R", "mirror", {}), (3712, "R", "mirror", {}), (3974, "R", "mirror", {}),                    # 写真 3105・3702・3949m の 12・10・25m 先
    (4108, "R", "road_sign", {"sign": "speed30", "facing": "fudakake"}),                                # トンネルの出口の先（札掛向きの写真）
    (5166, "R", "mirror", {}), (5765, "R", "mirror", {}),                                               # 写真 5151・5745m の 15・20m 先
    (6832, "R", "mirror", {}), (7208, "R", "mirror", {}),                                               # 写真 6802・7202m の 30・6m 先
    (9377, "L", "road_sign", {"sign": "warning", "out": -0.25}),                                        # 左の擁壁の前（側溝の上）：写真 9347m の 30m 先
]
POLE_SIGNS = [   # (s, 左右, 文字, 幅, 高さ, 地色, 文字色, 板の下の高さ)
    (7708, "L", "\n".join("国際一ノ瀬キャンプ場"), 0.45, 2.2, "#F2C200", "#1A1A1A", 0.4),   # 左の黄色の縦長の板（文字は OSM の名前）：写真 7698m の 10m 先
    (4562, "R", "塩水橋", 0.5, 1.0, "#1E50A0", "#FFFFFF", 1.6),                               # 橋の名前の柱（橋の手前）
]
HOARDINGS = [(4555, 4563, "L", 2.0)]   # 塩水橋の分かれ道の工事の囲い（左。写真 4553m の 2〜10m 先。その先は橋なので置かない）


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
        ll, out, t = frame(s, side, opt.get("out", OUT_M))   # out：道の縁＋側溝からの距離（擁壁の前に立てるときは負）
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
    for a, b, side, h in HOARDINGS:
        pts = [frame(s, side, 0.4)[0] for s in np.arange(a, b + 0.1, 2.0)]
        objs.append({"kind": "hoarding", "points": pts, "height_m": h, "note": f"塩水橋の分かれ道の工事の囲い（S5 {a}〜{b}m {side}、本物は青い川の絵の板）"})
    signs = []
    for s, side, text, w, hh, bg, fg, z in POLE_SIGNS:
        ll, out, t = frame(s, side, 1.0)
        signs.append({"text": text, "bg": bg, "fg": fg, "at": ll, "facing_deg": bearing(-t), "w_m": w, "h_m": hh, "z_m": z,
                      "note": f"ストリートビュー S5 {s}m {side}"})
    out = {"note": "S5 の道ばたの物（scripts/s5_roadside_objects.py で書く）", "objects": objs, "pole_signs": signs}
    (config.DATA_MANUAL / "s5_objects.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(len(objs), "objects,", len(signs), "pole signs")


if __name__ == "__main__":
    main()
