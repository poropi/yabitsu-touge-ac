import json

import numpy as np

from yabitsu import config, geo, osm, track

P = config.DATA_PROCESSED


def main():
    route70 = osm.fetch_overpass(osm.ROUTE70_QUERY, config.DATA_RAW / "osm_route70.json")
    town = osm.fetch_overpass(osm.TOWN_QUERY.format(r=260, lat=config.NAGOKI[0], lon=config.NAGOKI[1]),
                              config.DATA_RAW / "osm_nagoki.json")
    try:  # 道ぞいの横道・建物（Task 3・7 で使う）。サーバーが混んでいても道づくりは止めない
        osm.fetch_overpass(osm.ROADSIDE_QUERY, config.DATA_RAW / "osm_s3_roadside.json")
    except RuntimeError as e:
        print("warning: 道ぞいの OSM が取れませんでした（あとで取り直す）:", e)
    # 交差点のすぐ北の県道70号は ref が「県道70号線」で、ROUTE70_QUERY（ref=70）に入らない
    ways = osm.route70_ways(route70["elements"] + town["elements"])
    routes = [osm.shortest_route(ways, d.start, d.end) for d in config.SECTIONS]
    r, joints = track.join_routes(routes)
    d = {"lat": r.lat, "lon": r.lon, "seg_lanes": r.seg_lanes, "seg_bridge": r.seg_bridge,
         "joints": [list(j) for j in joints], "sections": [s.name for s in config.SECTIONS]}
    (P / "track_route.json").write_text(json.dumps(d), encoding="utf-8")
    x, y = geo.to_local(np.array(r.lat), np.array(r.lon))
    print(f"track route: {len(r.lat)} points, {np.hypot(np.diff(x), np.diff(y)).sum():.0f} m, sections {d['sections']}")
    parks = []
    for tag, (lat, lon), radius in (("start", config.S1_START, 150), ("end", config.S1_PASS, 200)):
        q = osm.fetch_overpass(osm.PARKING_QUERY.format(r=radius, lat=lat, lon=lon), config.DATA_RAW / f"osm_parking_{tag}.json")
        parks += osm.parking_polygons(q["elements"])
    (P / "track_parking.json").write_text(json.dumps(parks, ensure_ascii=False), encoding="utf-8")
    print(f"parking polygons: {len(parks)}")


if __name__ == "__main__":
    main()
