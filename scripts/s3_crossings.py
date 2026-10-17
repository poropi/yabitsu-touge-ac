"""S3 の横断歩道・停止線を OSM の横断歩道（highway=crossing、標示あり）と信号に合わせて書き直す。
- 本線の上の横断歩道（OSM の点が本線の中心から 2.5m 以内）：s3_pavements.csv の crosswalk の行を OSM の位置にそろえる。
  表にあって OSM にない横断歩道（ストリートビューで見たもの）はそのまま残す
- 信号のある横断歩道（crossing=traffic_signals か、10m 以内に信号）：近くの（30m 以内の）ものを1つの交差点とみなし、
  上りの停止線は一番下の横断歩道の手前、下りの停止線は一番上の横断歩道の先に置く
- 横道の上の横断歩道 → data/manual/s3_side_crosswalks.json（[[lat, lon], ...]、分かれ道の路面に描く）
町の範囲（交差点から 200m）は town.py が自動で置くので除く。"""
import csv
import json

import numpy as np

from yabitsu import config, geo, osm, side_roads, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED
QUERY = '[out:json][timeout:90];node["highway"~"crossing|traffic_signals|stop"](35.378,139.225,35.418,139.241);out body;'
ON_MAIN_M, NEAR_M, SIGNAL_M, CLUSTER_M, MATCH_M = 2.5, 25.0, 10.0, 30.0, 15.0
HALF = 2.0   # 横断歩道（長さ 4m）の半分
STOP_GAP = 2.0
STOP_W = 0.45   # 停止線の太さ（pavements の stop_line は s_from〜s_to の幅で引く）


def clusters(s_list, gap=CLUSTER_M):
    out = []
    for s in sorted(s_list):
        if out and s - out[-1][-1] <= gap:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def stop_rows(signalised):
    """信号のある横断歩道の s（区間の距離）→ 停止線の行。"""
    rows = []
    for c in clusters(signalised):
        lo, hi = min(c), max(c)
        a = lo - HALF - STOP_GAP
        rows.append({"s_from": round(a - STOP_W, 2), "s_to": round(a, 2), "kind": "stop_line",
                     "note": "信号の手前（上り、OSM の横断歩道から）"})
        b = hi + HALF + STOP_GAP
        rows.append({"s_from": round(b, 2), "s_to": round(b + STOP_W, 2), "kind": "stop_line_down",
                     "note": "信号の手前（下り、OSM の横断歩道から）"})
    return rows


def main():
    nodes = osm.fetch_overpass(QUERY, config.DATA_RAW / "osm_s3_nodes.json")["elements"]
    sec = Section.load_json(P / "track_section.json")
    bounds = track.load_bounds(); s0, s1 = bounds["s3"]
    cx, cy = geo.to_local(*config.NAGOKI)
    s_town = side_roads.town_exit_s(sec, (float(cx), float(cy)), config.TOWN_RADIUS_M)
    sig = np.array([geo.to_local(e["lat"], e["lon"]) for e in nodes if e["tags"].get("highway") == "traffic_signals"], float).reshape(-1, 2)
    main_cw, signalised, side = [], [], []
    for e in nodes:
        t = e["tags"]
        if t.get("highway") != "crossing" or t.get("crossing") == "unmarked" or t.get("crossing:markings") == "no":
            continue
        x, y = geo.to_local(e["lat"], e["lon"]); d = np.hypot(sec.x - x, sec.y - y); i = int(np.argmin(d))
        s = float(sec.s[i])
        if d[i] > NEAR_M or not (s_town + 5 < s < s1):
            continue
        lit = t.get("crossing") == "traffic_signals" or (len(sig) and np.hypot(*(sig - [x, y]).T).min() <= SIGNAL_M)
        if d[i] <= ON_MAIN_M:
            main_cw.append(s - s0)
            if lit:
                signalised.append(s - s0)
        else:
            side.append([round(e["lat"], 8), round(e["lon"], 8)])
    path = config.DATA_MANUAL / "s3_pavements.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if r["kind"] not in ("crosswalk", "stop_line", "stop_line_down")]
    old_cw = [(float(r["s_from"]) + float(r["s_to"])) / 2 for r in rows if r["kind"] == "crosswalk"]
    old_sig = [float(r["s_from"]) for r in rows if r["kind"] in ("stop_line", "stop_line_down")]
    new = [{"s_from": round(s - HALF, 2), "s_to": round(s + HALF, 2), "kind": "crosswalk", "note": "OSM の横断歩道"} for s in main_cw]
    for s in old_cw:   # ストリートビューで見て OSM にない横断歩道は残す
        if min((abs(s - m) for m in main_cw), default=99) > MATCH_M:
            new.append({"s_from": round(s - HALF, 2), "s_to": round(s + HALF, 2), "kind": "crosswalk", "note": "ストリートビュー（OSM になし）"})
            if old_sig and min(abs(s - q) for q in old_sig) < 12:   # 表で信号つきにしていたもの
                signalised.append(s)
    out = keep + new + stop_rows(signalised)
    out.sort(key=lambda r: float(r["s_from"]))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, ["s_from", "s_to", "kind", "note"], lineterminator="\n"); w.writeheader(); w.writerows(out)
    (config.DATA_MANUAL / "s3_side_crosswalks.json").write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"main crosswalks {len(new)} (OSM {len(main_cw)}), signalised {len(clusters(signalised))} junctions, side crosswalks {len(side)}")


if __name__ == "__main__":
    main()
