import json

import numpy as np

from yabitsu import config, geo, osm, track

P = config.DATA_PROCESSED


def main():
    route70 = osm.fetch_overpass(osm.ROUTE70_QUERY, config.DATA_RAW / "osm_route70.json")
    routes = [osm.shortest_route(route70["elements"], d.start, d.end) for d in config.SECTIONS]
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
