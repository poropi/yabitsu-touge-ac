import json

import numpy as np

from yabitsu import config, geo, osm


def main():
    route70 = osm.fetch_overpass(osm.ROUTE70_QUERY, config.DATA_RAW / "osm_route70.json")
    r = osm.shortest_route(route70["elements"], config.S1_START, config.S1_END)
    r.to_json(config.DATA_PROCESSED / "s1_route.json")
    x, y = geo.to_local(np.array(r.lat), np.array(r.lon))
    print(f"S1 route: {len(r.lat)} points, {np.hypot(np.diff(x), np.diff(y)).sum():.0f} m")
    parks = []
    for tag, (lat, lon), radius in (("start", config.S1_START, 150), ("end", config.S1_END, 200)):
        d = osm.fetch_overpass(osm.PARKING_QUERY.format(r=radius, lat=lat, lon=lon), config.DATA_RAW / f"osm_parking_{tag}.json")
        parks += osm.parking_polygons(d["elements"])
    (config.DATA_PROCESSED / "s1_parking.json").write_text(json.dumps(parks, ensure_ascii=False), encoding="utf-8")
    print(f"parking polygons: {len(parks)} {[p['name'] for p in parks]}")


if __name__ == "__main__":
    main()
