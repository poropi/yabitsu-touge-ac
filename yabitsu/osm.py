import heapq
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
ROUTE70_QUERY = '[out:json][timeout:90];way["ref"="70"]["highway"](35.36,139.15,35.52,139.32);out tags geom;'
PARKING_QUERY = '[out:json][timeout:60];way["amenity"="parking"](around:{r},{lat},{lon});out tags geom;'
# 名古木交差点のまわり（町の範囲＋余裕）の道・建物・信号など
TOWN_QUERY = ('[out:json][timeout:90];(way["highway"](around:{r},{lat},{lon});way["building"](around:{r},{lat},{lon});'
              'node(around:{r},{lat},{lon})["highway"~"traffic_signals|bus_stop|crossing"];);out tags geom;')
# S3 の道ぞいの横道と建物
ROADSIDE_QUERY = ('[out:json][timeout:90];(way["highway"](35.378,139.225,35.418,139.241);'
                  'way["building"](35.378,139.225,35.418,139.241););out tags geom;')
# S2・S1 の道ぞいの車道（分かれ道と、道の上を渡る橋のため）
TRACK_ROADS_QUERY = '[out:json][timeout:90];way["highway"](35.400,139.205,35.431,139.232);out tags geom;'
ROUTE70_NAMES = ("秦野清川線", "県道70号")


def _post(url, query):
    r = requests.post(url, data={"data": query}, timeout=120, headers={"User-Agent": "yabitsu-track-builder"})
    return r.status_code, r.text


def fetch_overpass(query, cache_path, post=None, endpoints=ENDPOINTS, retries_per_endpoint=2, wait=5.0):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    post = post or _post
    errors = []
    for url in endpoints:
        for attempt in range(retries_per_endpoint):
            try:
                status, text = post(url, query)
            except requests.RequestException as e:
                errors.append(f"{url}: {e}"); continue
            if status == 200:
                data = json.loads(text)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")
                return data
            errors.append(f"{url}: {status}")
            if status not in (429, 502, 503, 504):
                break
            if wait:
                time.sleep(wait * (attempt + 1))
    raise RuntimeError("Overpass からの取得に失敗しました: " + "; ".join(errors))


def route70_ways(elements):
    """県道70号の way だけ（ref が 70、または ref・name に路線名を含む）。経路探しが住宅地の近道を通らないように絞る。"""
    out = []
    for w in elements:
        if w.get("type") != "way" or "highway" not in w.get("tags", {}):
            continue
        t = w["tags"]
        if t.get("ref") == "70" or any(n in t.get(k, "") for n in ROUTE70_NAMES for k in ("ref", "name")):
            out.append(w)
    return out


def parse_lanes(v) -> int:
    m = re.match(r"\s*(\d+)", str(v)) if v is not None else None
    return int(m.group(1)) if m else 0


@dataclass
class Route:
    lat: list
    lon: list
    seg_lanes: list
    seg_bridge: list

    def to_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _key(p):
    return (round(p["lat"], 7), round(p["lon"], 7))


def _dist(a, b):
    lat = math.radians((a[0] + b[0]) / 2)
    return math.hypot((b[1] - a[1]) * 111320 * math.cos(lat), (b[0] - a[0]) * 110574)


def shortest_route(elements, start, end) -> Route:
    adj = defaultdict(dict)
    for w in elements:
        if w.get("type") != "way" or "geometry" not in w:
            continue
        ks = [_key(p) for p in w["geometry"]]
        for a, b in zip(ks, ks[1:]):
            adj[a][b] = w.get("tags", {}); adj[b][a] = w.get("tags", {})
    a = min(adj, key=lambda k: _dist(k, start)); b = min(adj, key=lambda k: _dist(k, end))
    dist, prev, h = {a: 0.0}, {}, [(0.0, a)]
    while h:
        d, u = heapq.heappop(h)
        if u == b:
            break
        if d > dist[u]:
            continue
        for v in adj[u]:
            nd = d + _dist(u, v)
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(h, (nd, v))
    if b not in dist:
        raise ValueError("始点と終点がつながっていません")
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    path.reverse()
    tags = [adj[u][v] for u, v in zip(path, path[1:])]
    return Route([p[0] for p in path], [p[1] for p in path], [parse_lanes(t.get("lanes")) for t in tags],
                 [t.get("bridge", "no") != "no" for t in tags])


def parking_polygons(elements):
    return [{"name": w.get("tags", {}).get("name", ""), "lat": [p["lat"] for p in w["geometry"]],
             "lon": [p["lon"] for p in w["geometry"]]}
            for w in elements if w.get("type") == "way" and w.get("tags", {}).get("amenity") == "parking"]
