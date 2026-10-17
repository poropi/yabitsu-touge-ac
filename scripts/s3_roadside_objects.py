"""町の範囲の外の S3 の道ぞいの物を data/manual/s3_objects.json に書く：
- 信号機：s3_pavements.csv の停止線（stop_line / stop_line_down）ごとに、その先の横断歩道の向こう側・進む向きの左に車用、
  信号のある横断歩道の両端に歩行者用
- バス停：OSM のバス停（道から 15m 以内）
- 鳥居と石灯籠（s=3400 の右、ストリートビュー）"""
import json
import math

import numpy as np

from yabitsu import config, geo, side_roads, track
from yabitsu.crosssection import GUTTER_WIDTH, Section

P = config.DATA_PROCESSED
SIDEWALK_M = 1.5


def main():
    sec = Section.load_json(P / "track_section.json")
    s0 = track.load_bounds()["s3"][0]
    rows = track.load_section_rows("pavements", track.load_bounds())
    cx, cy = geo.to_local(*config.NAGOKI)
    s_town = side_roads.town_exit_s(sec, (float(cx), float(cy)), config.TOWN_RADIUS_M)
    s_end = track.load_bounds()["s3"][1]

    def at(s, lat):
        i = int(np.argmin(np.abs(sec.s - s))); h = sec.heading[i]
        x = sec.x[i] - lat * math.sin(h); y = sec.y[i] + lat * math.cos(h)
        la, lo = geo.from_local(x, y)
        return [round(float(la), 8), round(float(lo), 8)], i

    bearing = lambda h: (90.0 - math.degrees(h)) % 360.0    # 進行方向（+x から反時計回り）→ 方位
    cws = [(float(r["s_from"]), float(r["s_to"])) for r in rows if r["kind"] == "crosswalk"]
    signals, lit = [], set()
    for r in rows:
        if r["kind"] not in ("stop_line", "stop_line_down"):
            continue
        s_stop = float(r["s_from"]); up = r["kind"] == "stop_line"
        ahead = [c for c in cws if (c[0] > s_stop if up else c[1] < s_stop)]
        if not ahead:
            continue
        far = min(ahead, key=lambda c: abs(c[0] - s_stop))
        lit.add(far)
        s_far = far[1] + 1.5 if up else far[0] - 1.5
        i = int(np.argmin(np.abs(sec.s - s_far)))
        hw = (sec.hw_left if up else sec.hw_right)[i] + GUTTER_WIDTH + SIDEWALK_M / 2
        pos, i = at(s_far, hw if up else -hw)
        h = sec.heading[i] if up else sec.heading[i] + math.pi
        signals.append({"at": pos, "facing_deg": round(bearing(h + math.pi), 1), "kind": "vehicle",
                        "arm_m": round(float(sec.hw_left[i]) + 0.5, 2), "arm_deg": round(bearing(h - math.pi / 2), 1),
                        "height_m": 5.5, "note": f"s={s_stop - s0:.0f} の停止線（{'上り' if up else '下り'}）"})
    for a, b in sorted(lit):
        sm = (a + b) / 2; i = int(np.argmin(np.abs(sec.s - sm)))
        for side in (1, -1):
            hw = (sec.hw_left if side > 0 else sec.hw_right)[i] + GUTTER_WIDTH + 0.8
            pos, _ = at(sm, side * hw)
            signals.append({"at": pos, "facing_deg": round(bearing(sec.heading[i] + (-math.pi / 2 if side > 0 else math.pi / 2)), 1),
                            "kind": "pedestrian", "height_m": 2.6, "note": f"s={sm - s0:.0f} の横断歩道"})
    objects = []
    stops = json.loads((config.DATA_RAW / "osm_s3_busstops.json").read_text(encoding="utf-8"))["elements"]
    for e in stops:
        x, y = geo.to_local(e["lat"], e["lon"]); d = np.hypot(sec.x - x, sec.y - y); i = int(np.argmin(d))
        if d[i] > 15.0 or not (s_town < sec.s[i] < s_end):
            continue
        to_road = math.degrees(math.atan2(float(sec.x[i] - x), float(sec.y[i] - y))) % 360.0
        objects.append({"kind": "bus_stop", "points": [[e["lat"], e["lon"]]], "facing_deg": round(to_road, 1),
                        "note": f"バス停「{e.get('tags', {}).get('name', '')}」（OSM）"})
    for s, lat, kind in ((3405.0, None, "torii"), (3401.0, None, "lantern")):
        i = int(np.argmin(np.abs(sec.s - (s0 + s))))
        off = sec.hw_right[i] + GUTTER_WIDTH + (3.5 if kind == "torii" else 1.8)
        pos, i = at(s0 + s, -off)
        objects.append({"kind": kind, "points": [pos], "facing_deg": round(bearing(sec.heading[i] + math.pi / 2), 1),
                        "note": f"s={s:.0f} の右：神社の入口（ストリートビュー s=3400、バス停「鳥居前」）"})
    out = config.DATA_MANUAL / "s3_objects.json"
    out.write_text(json.dumps({"signals": signals, "objects": objects}, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    print(out, len(signals), "signals", len(objects), "objects")


if __name__ == "__main__":
    main()
