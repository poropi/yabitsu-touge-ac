"""合格済みの道ぎわの草と植栽を、保存地形と森から組み立てる。"""
from configparser import ConfigParser
import json
import re
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree
import shapely

from . import kn5, roadside_cover, treemodels


LEGACY_KINDS = {
    "fern": ("fern",),
    "sugi": ("sugi", "hinoki", "broadleaf"),
    "sapling": ("sapling", "sugi", "hinoki", "broadleaf"),
    "shrub": ("shrub",),
}


def grass_meshes(meshes, mountain, hidden):
    """共有材質に頼らず、山の貼り替え面と谷の林床を名前で選ぶ。"""
    return {name for name, mesh in meshes.items() if mesh["renderable"] and name not in hidden and (
        (name in mountain and re.fullmatch(r"(?:1GRASS|mtsplit)_s1_terrain_.+", name)
         and mesh["material"] in ("ground", "mt_split_ground"))
        or (name.startswith("1GRASS_valley_") and mesh["material"] == "valley_forest_floor")
    )}


def overlaps_open_land(triangles, zones):
    polygons = shapely.polygons(triangles)
    face, zone = zones.query(polygons, predicate="intersects")
    return bool(np.any(shapely.area(shapely.intersection(polygons[face], zones.geometries[zone])) > 1.e-6))


def exclude_grass_pad(pad, pavement):
    # この輪の整地範囲はすべて同名の舗装で覆われる。舗装は OCCLUDING_* で指定する。
    name = pad["name"]
    return not (name.startswith("goma_loop_") and f"1ROAD_{name}" in pavement)


def grass_selection(processed, finish, open_land):
    replacement = {}
    for block in finish.split("\n\n"):
        texture = re.search(r"^RESOURCE_FILE_0=dress_(mt_terrain|mt_grass)\.dds$", block, re.M)
        if texture:
            meshes = re.search(r"^MESHES=(.+)$", block, re.M)[1].split(",")
            replacement.update({n: texture[1] for n in meshes if not n.endswith("_back")})
    landuses = {"residential", "farmland", "farmyard", "orchard", "allotments", "greenhouse_horticulture",
                "commercial", "retail", "industrial"}
    polygons = [poly for poly, tags in open_land
                if tags.get("landuse") in landuses or tags.get("amenity") == "parking"]
    pads = json.loads((processed / "track_pads.json").read_text(encoding="utf-8"))
    track_meshes = json.loads((processed / "track/manifest.json").read_text(encoding="utf-8"))["meshes"]
    pavement = {name for name, spec in track_meshes.items() if spec["renderable"] and spec["material"] == "asphalt"}
    polygons.extend(shapely.make_valid(shapely.Polygon(p["polygon"])) for p in pads
                    if "polygon" in p and exclude_grass_pad(p, pavement))
    zones = shapely.STRtree(polygons)
    hidden = set(json.loads((processed / "dress_mountain/hidden.json").read_text(encoding="utf-8")))
    s3_hidden = processed / "dress_s3/hidden.json"
    if s3_hidden.exists():
        hidden.update(json.loads(s3_hidden.read_text(encoding="utf-8")))
    sources, selected, excluded = {}, {}, []
    for folder in ("track", "dress_mountain"):
        root = processed / folder
        meshes = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["meshes"]
        sources.update(meshes)
        for name in sorted(grass_meshes(meshes, replacement, hidden)):
            mesh = meshes[name]
            with np.load(root / mesh["file"]) as source:
                triangles = source["verts"][kn5._triangles(source["faces"]), :2]
            if overlaps_open_land(triangles, zones):
                excluded.append(name)
            else:
                selected[name] = {"material": mesh["material"], "texture": replacement.get(name), "source": folder}
    return selected, excluded, replacement, sources


def grass_ini(selected, sources, settings):
    # 非地形の固い面を手前に置く。対象への誤混入は GRASS_MESHES 側でも防ぐ。
    materials = sorted({m["material"] for n, m in sources.items() if
                        n.startswith(("1ROAD", "1WALL", "lines_", "smallspan_", "water_", "riverbed_"))
                        and m["material"] not in {"ground", "mt_split_ground", "valley_forest_floor", "ground_far"}})
    return "\n".join((
        "; 山の地形と谷の林床。部品名で限定し、共有材質 ground の一括指定は空にする。",
        "[GRASS_FX]",
        "GRASS_MATERIALS=",
        "GRASS_MESHES=" + ",".join(sorted(selected)),
        "OCCLUDING_MATERIALS=" + ",".join(materials),
        f"OCCLUDING_MESHES={settings['OCCLUDING_MESHES']}",
        "OCCLUDING_MATERIALS_ALPHA=",
        "OCCLUDING_MESHES_ALPHA=",
        "ORIGINAL_GRASS_MATERIALS=",
        f"MASK_MAIN_THRESHOLD={settings['MASK_MAIN_THRESHOLD']}",
        f"MASK_RED_THRESHOLD={settings['MASK_RED_THRESHOLD']}",
        f"MASK_MIN_LUMINANCE={settings['MASK_MIN_LUMINANCE']}",
        f"MASK_MAX_LUMINANCE={settings['MASK_MAX_LUMINANCE']}",
        "; SHAPE_SIZE の値を 2.0（小・密）/ 2.5（中）/ 3.0（大・疎）へ切り替える。",
        f"SHAPE_SIZE={settings['SHAPE_SIZE']}",
        f"SHAPE_TIDY={settings['SHAPE_TIDY']}",
        f"SHAPE_CUT={settings['SHAPE_CUT']}",
        f"SHAPE_WIDTH={settings['SHAPE_WIDTH']}",
        "",
    ))


def fern_packages(source, dest, height_factor):
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in "ab":
        with ZipFile(source / f"fern_{suffix}.zip") as original:
            old_ini = original.read("tree.ini").decode()
            ini = ConfigParser()
            ini.read_string(old_ini)
            height = ini["BASIC"].getfloat("HEIGHT") * height_factor
            new_ini, changes = re.subn(r"(?im)^(HEIGHT\s*=\s*)[\d.]+", rf"\g<1>{height:.5f}", old_ini)
            if changes != 1:
                raise ValueError(f"fern_{suffix}: tree.ini の高さを一意に特定できません")
            with ZipFile(dest / f"fern_tall_{suffix}.zip", "w") as out:
                for item in original.infolist():
                    out.writestr(item, new_ini.encode() if item.filename == "tree.ini" else original.read(item))


def spacing(candidates, existing, meters):
    if not len(candidates):
        return np.zeros(0, dtype=bool)
    tree = cKDTree(candidates)
    neighbors = tree.query_ball_point(candidates, np.max(meters))
    free = cKDTree(existing).query(candidates)[0] >= meters - 1.e-8
    accepted = np.zeros(len(candidates), bool)
    for i in np.flatnonzero(free):
        if free[i]:
            accepted[i] = True
            near = np.asarray(neighbors[i])
            dist = np.linalg.norm(candidates[near] - candidates[i], axis=1)
            free[near[dist < np.maximum(meters[i], meters[near]) - 1.e-8]] = False
    return accepted


def scale_limits(model, height, edge, span=0., *, max_scale):
    minimum = np.ceil(100 * max(height[0] / model.height, span / (2 * model.crown_radius))) / 100
    maximum = np.minimum.reduce((np.full(len(edge), max_scale), np.full(len(edge), height[1] / model.height),
                                 (edge - .015) / model.crown_radius))
    return np.full(len(edge), minimum), np.floor(100 * maximum) / 100


def conifer_neighborhood(xy, old, kinds, road, cfg):
    tall = np.isin(kinds, ("sugi", "hinoki", "broadleaf"))
    positions = np.c_[old.x[tall], old.y[tall]]
    far = road.edge_distance(positions) > cfg["max_edge_m"]
    if not np.any(far):
        return np.zeros(len(xy), bool)
    distance, index = cKDTree(positions[far]).query(xy)
    return (distance <= cfg["forest_neighbor_m"]) & np.isin(kinds[tall][far][index], ("sugi", "hinoki"))


def place(grid, old, road, fm, ground, exclusion, models, manual, spec, *, bridge_station, variants):
    cfg = spec["roadside_vegetation"]
    rng = np.random.default_rng(cfg["seed"])
    jitter = cfg["jitter_m"]
    xy = np.round(grid + rng.uniform(-jitter, jitter, grid.shape), 2)
    good, edge, z = roadside_cover.eligible(xy, road, fm, ground, exclusion, cfg)
    # 0366_f の橋直後は、地図上で開地・庭に見える帯にもスギ林と低い植生を確認した。
    station = road.locate(xy).s - bridge_station
    photo_start, photo_end = cfg["bridge_photo_span_m"]
    photo_stand = (station >= photo_start) & (station <= photo_end)
    good |= photo_stand & np.isfinite(z)
    good &= cKDTree(manual).query(xy)[0] >= cfg["manual_clear_m"]
    good &= roadside_cover.bare(xy, old, cfg["existing_clear_m"])
    order = rng.permutation(np.flatnonzero(good))
    xy, edge, z = xy[order], edge[order], np.round(z[order], 2)
    photo_stand = photo_stand[order]
    old_kind = roadside_cover.genus(old.pattern)
    layers = []
    occupied = np.zeros(len(xy), bool)
    for band in cfg["plant_bands"]:
        label = band["kind"]
        min_edge, max_edge = band["edge_m"]
        gaps = band["spacing_m"]
        heights = band.get("height_m")
        ids = np.flatnonzero(~occupied & (edge >= min_edge) & (edge <= max_edge))
        if not len(ids):
            continue
        if label == "sugi":
            ids = ids[photo_stand[ids] | conifer_neighborhood(xy[ids], old, old_kind, road, cfg)]
        if not len(ids):
            continue
        species = variants.get(label, treemodels.SPECIES.get(label))
        sizes = cfg["sugi_scale"] if label == "sugi" else cfg["plant_scale"]
        rows = roadside_cover._rows(xy[ids], z[ids], [label] * len(ids), {label: species}, rng, sizes)
        candidate = np.ones(len(rows), bool)
        if heights:
            for pattern in species:
                is_pattern = rows.pattern == pattern
                lo, hi = scale_limits(models[pattern], heights, edge[ids[is_pattern]], band.get("span_m", 0.),
                                      max_scale=cfg["max_scale"])
                candidate[is_pattern] = hi >= lo
                weights = rng.random(len(lo))
                weights = np.where(photo_stand[ids[is_pattern]] & (label in ("fern", "shrub")), np.sqrt(weights), weights)
                scale = lo + weights * (np.maximum(lo, hi) - lo)
                rows.scale[is_pattern] = np.round(scale, 2)
        if label == "sugi":
            towards = road.locate(xy[ids]).towards
            rows.angle_deg = np.degrees(np.arctan2(towards[:, 1], towards[:, 0])) % 360
        candidate &= roadside_cover.safe_trees(rows, road, models, cfg, spec["sight"])
        ids, rows = ids[candidate], rows[candidate]
        existing = np.c_[old.x, old.y][np.isin(old_kind, LEGACY_KINDS[label])]
        previous = [np.c_[r.x, r.y] for kind, r in layers if kind == label or
                    (label == "sapling" and kind == "sugi")]
        if previous:
            existing = np.concatenate([existing, *previous])
        chosen = spacing(xy[ids], existing, rng.uniform(*gaps, len(ids)))
        layers.append((label, rows[chosen]))
        occupied[ids[chosen]] = True
    return np.concatenate([r for _, r in layers]).view(np.recarray) if layers else old[:0].copy()
