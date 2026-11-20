"""正式な道ぎわの植栽に使う地物・地面・樹木モデルの読み込み。"""
from collections import defaultdict
from configparser import ConfigParser
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import shapely

from scripts import add_extra_trees
from . import config, csptree, forest_ground, forestmap, geo, scenery, side_roads, treemodels, tunnel


def local_polygons(cache, *, lake=False):
    data = json.loads((config.DATA_RAW / cache).read_text(encoding="utf-8"))
    for entry in data["elements"]:
        tags = entry.get("tags", {})
        is_lake = tags.get("natural") == "water" and tags.get("water") in ("lake", "reservoir")
        if is_lake != lake:
            continue
        rings = [entry["geometry"]] if "geometry" in entry else [
            m["geometry"] for m in entry.get("members", []) if m.get("role") == "outer" and "geometry" in m
        ]
        for ring in rings:
            if len(ring) >= 3:
                xy = np.c_[geo.to_local([p["lat"] for p in ring], [p["lon"] for p in ring])]
                yield shapely.make_valid(shapely.Polygon(xy)), tags


def barriers(road, spec, manifest, *, cover=None):
    """札掛・市街の地物を含む、保存済みの幾何と明示された開放箇所。"""
    cfg = spec["roadside_cover"] if cover is None else cover
    reach = cfg["max_edge_m"] + 2
    def near(polygon, margin=0.):
        return road.pavement.query(polygon, predicate="dwithin", distance=reach + margin).size > 0

    shapes = []
    buffers = defaultdict(list)
    rails = []
    classify = {**forestmap.DEFAULT_CLASSIFY, **spec.get("classify", {})}
    for polygon, tags in local_polygons("osm_forest_open.json"):
        if "building" in tags:
            polygon = polygon.buffer(classify["settle_m"])
        if near(polygon):
            shapes.append(polygon)
    # 路面メッシュになっていない分岐も、保存済みの車道の幅で避ける。
    ways = json.loads((config.DATA_RAW / "osm_forest_roads.json").read_text(encoding="utf-8"))["elements"]
    for way in ways:
        tags = way.get("tags", {})
        route_ref = tags.get("ref", "")
        points = way.get("geometry", [])
        if route_ref in ("70", "県道70号線") or len(points) < 2:
            continue
        x, y = geo.to_local([p["lat"] for p in points], [p["lon"] for p in points])
        line = shapely.LineString(np.c_[x, y])
        width = scenery.WIDTH.get(tags.get("highway"), 4.0)
        margin = width / 2 + spec["clearance"]["osm_road"]
        if near(line, margin):
            buffers[margin].append(line)
    for zone in (spec.get("keep_map", []) + spec.get("view_clear", []) + spec.get("open", [])
                 + [opening for fill in spec.get("fill_bare", []) for opening in fill.get("open", [])]
                 + [opening for patch in spec.get("species_zones", []) for opening in patch.get("open", [])]):
        shapes.append(shapely.Point(*geo.to_local(*zone["center_ll"])).buffer(zone["radius"]))
    bore = tunnel.from_osm(side_roads.load_elements(), SimpleNamespace(s=road.s, x=road.xy[:,0], y=road.xy[:,1]))
    for t in bore:
        spans = (road.s >= t.s0 - 5) & (road.s <= t.s1 + 5)
        if spans.sum() > 1:
            shapes.append(shapely.LineString(road.xy[spans]).buffer(reach + 4))
    pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    shapes.extend(shapely.make_valid(shapely.Polygon(p["polygon"])) for p in pads if "polygon" in p)

    for root in (manifest, *(manifest.parent / name for name in ("dress_mountain", "dress_s3", "visual_updates"))):
        if not (root / "manifest.json").exists():
            continue
        items = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["meshes"]
        for name, entry in items.items():
            material = entry["material"]
            kind = add_extra_trees.mesh_kind(name, material)
            cls = forest_ground.mesh_class(name, material)
            if kind in (None, "card") and cls not in ("building", "road", "wall", "bridge", "tunnel", "water"):
                continue
            if kind == "card" or (kind is None and (name.lower().startswith("lines_") or material in
                                                   ("ground", "ground_far", "valley_forest_floor", "tree"))):
                continue
            margin = (cfg["rail_clear_m"] if kind in ("ガードレール", "柵") else
                      2.0 if cls == "building" else 0.)
            with np.load(root / entry["file"]) as mesh:
                verts = mesh["verts"][:, :2]
                if not len(verts):
                    continue
                lo, hi = verts.min(axis=0)-margin, verts.max(axis=0)+margin
                if not near(shapely.box(*lo, *hi)):
                    continue
                f = mesh["faces"]
                tri = np.vstack((f[:, :3], f[:, [0, 2, 3]])) if f.shape[1] == 4 else f
            # 投影が線になる壁・柵も、凸包を取って線を残す。
            parts = shapely.convex_hull(shapely.polygons(verts[tri]))
            pick = road.pavement.query(parts, predicate="dwithin", distance=reach + margin)[0]
            parts = parts[np.unique(pick)]
            (buffers[margin] if margin else shapes).extend(parts)
            if kind == "ガードレール":
                rails.extend(parts)
    return [(shapely.STRtree(shapes), 0.)] + [(shapely.STRtree(p), r) for r,p in buffers.items()], shapely.STRtree(rails)


def ground_from_manifest(path, base=None):
    if base is None:
        base = forest_ground.load_ground(path)
    patches = forest_ground.load_ground(path, prefixes=("1GRASS_valley_",))
    return forest_ground.Ground(base.parts + patches.parts)


def manual_stems(section, bounds):
    buildings = (config.DATA_MANUAL / "track_buildings.json").read_text(encoding="utf-8")
    records = {r["id"]: r for r in json.loads(buildings)}
    rows, errors, _ = add_extra_trees.read_candidates(config.DATA_MANUAL / "extra_trees.csv",
                                                      section, bounds, records)
    if errors:
        raise ValueError(errors)
    return np.array([(x, y) for _, _, x, y, _ in rows]).reshape(-1, 2)


def package_models(folder):
    """配置判定も ZIP に実際に入っている樹高・枝の高さ・張り出しを使う。"""
    models = {}
    for p in sorted(folder.glob("*.zip")):
        ini = ConfigParser()
        with ZipFile(p) as z:
            ini.read_string(z.read("tree.ini").decode())
            crowns = []
            for key in ini.sections():
                if not key.startswith("LOD_"):
                    continue
                tex, parts = csptree.read_mesh(z.read(ini[key]["MODEL"]))
                leaves = [part for part in parts if "bark" not in tex[part.diffuse][1].lower()]
                crowns.extend(part.verts["p"] for part in leaves or parts)
        height = ini["BASIC"].getfloat("HEIGHT")
        width = ini["BASIC"].getfloat("WIDTH")
        points = np.concatenate(crowns)
        radius = np.linalg.norm(points[:, [0,2]], axis=1).max() * width / 2
        models[p.stem] = treemodels.TreeModel([], height, width, float(points[:,1].min()), radius, {})
    return models


def near_road(rows, road, width):
    keep = []
    for a in range(0, len(rows), 50000):
        block = rows[a:a+50000]
        p = shapely.points(block.x, block.y)
        keep.extend((a+np.unique(road.pavement.query(p, predicate="dwithin", distance=width)[0])).tolist())
    return np.asarray(keep, int)
