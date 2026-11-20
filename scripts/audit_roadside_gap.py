"""旧森林の S4 1600–1850 m を、保存済み台帳と局所候補で監査する。"""
import argparse
import configparser
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from yabitsu import config, csptree, forest_ground as fg, forest_place as fp, forestmap as fm, geo, track
from yabitsu.crosssection import Section
from yabitsu.treemodels import SPECIES

BASE = config.PROJECT_ROOT / "export/compare/route70/work"
BUILT = config.PROJECT_ROOT / "export/forest/20261031_093533"
VALLEY = BASE / "supervisor_41.fmEVx6"
MANIFEST = BASE / "before_34/track"
RADIUS = 1.5


def xy(trees):
    return np.c_[trees["x"], trees["y"]]


def band(sec):
    start = track.load_bounds()["s4"][0]
    select = (sec.s >= start + 1550) & (sec.s <= start + 1900)
    center = np.c_[sec.x[select], sec.y[select]]
    ids = np.flatnonzero(select)
    tree = cKDTree(center)

    def locate(points):
        idx = ids[tree.query(points)[1]]
        dx, dy = (points - np.c_[sec.x[idx], sec.y[idx]]).T
        cos, sin = np.cos(sec.heading[idx]), np.sin(sec.heading[idx])
        along = sec.s[idx] + dx * cos + dy * sin - start
        lateral = -dx * sin + dy * cos
        edge = np.abs(lateral) - np.where(lateral >= 0, sec.hw_left[idx], sec.hw_right[idx])
        return along, lateral, edge

    return locate, np.r_[np.floor(center.min(0) - 32), np.ceil(center.max(0) + 32)]


def grid_xy(bounds):
    x, y = np.meshgrid(np.arange(bounds[0] + 0.5, bounds[2]), np.arange(bounds[1] + 0.5, bounds[3]))
    return np.c_[x.ravel(), y.ravel()]


def contained(points, bounds):
    return ((points >= bounds[:2]) & (points <= bounds[2:])).all(axis=1)


def scopes(points, locate):
    s, lateral, edge = locate(points)
    return {
        name: (s >= a) & (s < b) & (edge >= low) & (edge < high) & ((lateral >= 0) == left)
        for prefix, a, b, low, high in (
            ("S4_1600_1850_0_20", 1600, 1850, 0, 20),
            ("S4_1600_1850_0_8", 1600, 1850, 0, 8),
            ("S4_1680_1722_0_8", 1680, 1722, 0, 8),
            ("S4_1680_1722_8_25", 1680, 1722, 8, 25),
            ("S4_1680_1710_0_8", 1680, 1710, 0, 8),
            ("S4_1680_1710_8_25", 1680, 1710, 8, 25),
            ("S4_1697_1722_0_8", 1697, 1722, 0, 8),
        )
        for side, left in (("L", True), ("R", False))
        for name in (f"{prefix}_{side}",)
    }


def overlaps(points, bounds):
    return (points.max(axis=0) >= bounds[:2]).all() and (points.min(axis=0) <= bounds[2:]).all()


def mesh_inputs(bounds):
    """各メッシュの端で格子の全体原点を確定し、帯に届く面だけを持つ。"""
    entries = json.loads((MANIFEST / "manifest.json").read_text())["meshes"]
    extent = {key: np.array([np.inf, np.inf, -np.inf, -np.inf]) for key in ("ground", "items")}
    pieces = []
    for name, entry in entries.items():
        material = entry["material"]
        kind = fg.mesh_class(name, material)
        land = material in fg.GROUND_MATERIALS
        if kind is None and not land:
            continue
        with np.load(MANIFEST / entry["file"]) as mesh:
            verts = mesh["verts"]
            lo, hi = verts[:, :2].min(axis=0), verts[:, :2].max(axis=0)
            for key in (["ground"] if land else []) + (["items"] if kind is not None else []):
                extent[key][:2] = np.minimum(extent[key][:2], lo)
                extent[key][2:] = np.maximum(extent[key][2:], hi)
            if (hi < bounds[:2]).any() or (lo > bounds[2:]).any():
                continue
            faces = mesh["faces"]
            triangles = np.r_[faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]] if faces.shape[1] == 4 else faces
        triangle_xy = verts[triangles, :2]
        mask = ((triangle_xy.max(axis=1) >= bounds[:2]) & (triangle_xy.min(axis=1) <= bounds[2:])).all(axis=1)
        if mask.any():
            indices, new_faces = np.unique(triangles[mask], return_inverse=True)
            pieces.append((name, material, kind, verts[indices], new_faces.reshape(-1, 3)))
    return pieces, extent


def terrain_meshes(pieces, *, road=False):
    far, near = [], []
    for name, material, _, v, f in pieces:
        if (not name.startswith("1ROAD_")) if road else material not in fg.GROUND_MATERIALS:
            continue
        try:
            interp = fg.LinearTriInterpolator(fg.Triangulation(v[:, 0], v[:, 1], f), v[:, 2])
        except (RuntimeError, ValueError):
            interp = fg.TriSoup(v, f)
        box = (*v[:, :2].min(axis=0), *v[:, :2].max(axis=0))
        (far if name.startswith("TERRAIN_FAR") else near).append((box, interp))
    return fg.Ground(far + near)


def keepout(pieces, extent, bounds, options):
    """旧クラス別ラスタを保存 manifest の格子原点のまま帯に切り詰める。"""
    cfg = {**fg.DEFAULT_CLEARANCE, **options}
    margin = max(cfg[name] for name in fg.CLASSES) + 10
    fine_anchor = extent["items"][[0, 3]] + [-margin, margin]
    coarse_anchor = extent["ground"][[0, 3]]

    def grid(anchor, cell):
        origin = anchor + np.floor((bounds[[0, 3]] - anchor) / cell) * cell
        origin[1] += cell
        return fg.Grid(origin[0], bounds[1], bounds[2], origin[1], cell)

    fine = grid(fine_anchor, cfg["cell"])
    coarse = grid(coarse_anchor, cfg["coarse_cell"])
    label, road, coarse_road = fine.canvas(), fine.canvas(), coarse.canvas()
    for kind in sorted(fg.CLASSES, key=lambda name: cfg[name]):
        for _, _, item_kind, v, f in pieces:
            if item_kind != kind:
                continue
            fine.draw_tris(label, v, f, fg.CLASSES.index(kind) + 1)
            if kind == "road":
                fine.draw_tris(road, v, f, 255)
                coarse.draw_tris(coarse_road, v, f, 255)
    lab = np.asarray(label)
    radius = np.array([0.0] + [cfg[kind] for kind in fg.CLASSES])
    distance, index = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    nearest_kind = lab[index[0], index[1]]
    blocked = (nearest_kind > 0) & (distance * fine.cell <= radius[nearest_kind] + 1e-6)
    road_d = (ndimage.distance_transform_edt(np.asarray(road) == 0) * fine.cell).astype(np.float32)
    coarse_road = np.asarray(coarse_road) > 0
    keep = fg.KeepOut(fine, blocked, road_d, coarse, coarse_road, np.zeros(coarse.shape, bool), cfg)
    cache = config.DATA_RAW / "osm_forest_roads.json"
    lines, widths = fg.osm_road_lines(bounds, cache=cache.resolve(strict=True))
    pairs = [(p, w) for p, w in zip(lines, widths) if overlaps(p, bounds)]
    keep.add_lines(*zip(*pairs))
    return keep, nearest_kind


def crop_map(whole, bounds):
    c0 = int(np.floor((bounds[0] - whole.x0) / whole.cell))
    c1 = int(np.ceil((bounds[2] - whole.x0) / whole.cell))
    r0 = int(np.floor((whole.y1 - bounds[3]) / whole.cell))
    r1 = int(np.ceil((whole.y1 - bounds[1]) / whole.cell))
    return r0, r1, c0, c1


def raw_photo(rgb, whole, cell_box, cfg, k):
    """関連する 500 m 区画を読み、全域のブロック座標で空中写真の判定を補間。"""
    r0, r1, c0, c1 = cell_box
    block = max(1, round(cfg["block_m"] / whole.cell))
    total = np.array(whole.cls.shape)
    block_count = np.ceil(total / block).astype(int)
    margin = int(cfg["smooth_cells"])
    rr = np.arange(r0 - margin, r1 + margin)
    cc = np.arange(c0 - margin, c1 + margin)
    block_idx = [(a + 0.5) * n / length - 0.5 for a, n, length in zip((rr, cc), block_count, total)]
    top, left = [int(np.floor(x.min())) for x in block_idx]
    bottom, right = [int(np.ceil(x.max())) + 1 for x in block_idx]
    ir0, ir1 = top * block, min(bottom * block, total[0])
    ic0, ic1 = left * block, min(right * block, total[1])
    win = max(3, int(round(cfg["tex_window_m"] / (whole.cell / k))) | 1)
    halo = win // 2
    tile = rgb[ir0*k-halo:ir1*k+halo, ic0*k-halo:ic1*k+halo].astype(np.float32) / 255
    lum = tile.mean(axis=-1)
    mean = ndimage.uniform_filter(lum, win)
    sd = np.sqrt(np.maximum(ndimage.uniform_filter(lum * lum, win) - mean * mean, 0))
    crop = np.s_[halo:halo+(ir1-ir0)*k, halo:halo+(ic1-ic0)*k]
    cv = fm._cells((sd / np.maximum(mean, 1e-3))[crop], k).mean((1, 3))
    color = fm._cells(tile[crop], k).mean((1, 3))
    light = color.mean(axis=-1)
    rough = (cv >= cfg["cv_min"]) | (light < cfg["shadow_lum"])
    brown = color[..., 0] / np.maximum(color[..., 1], 1e-3) >= cfg["brown_rg"]
    share, green_med, rough_med = (np.full((bottom-top, right-left), np.nan, np.float32) for _ in range(3))
    for i in range(bottom-top):
        for j in range(right-left):
            part = np.s_[i*block:(i+1)*block, j*block:(j+1)*block]
            forest, brown_part, lum_part = rough[part], brown[part], light[part]
            share[i, j] = brown_part[forest].mean() if forest.any() else 0
            rough_med[i, j] = np.median(lum_part[forest]) if forest.any() else np.nan
            green = forest & ~brown_part
            green_med[i, j] = np.median(lum_part[green]) if green.any() else np.nan
    if not np.isfinite([green_med, rough_med]).all():
        raise RuntimeError("対象区画に中央値がないため、全域の中央値を参照する必要があります")
    bc, br = np.meshgrid(block_idx[1] - left, block_idx[0] - top)

    def interpolate(values):
        return ndimage.map_coordinates(values, [br, bc], order=1, mode="nearest")

    subset = np.ix_(rr - ir0, cc - ic0)
    winter = interpolate(share) >= cfg["winter_brown_share"]
    forest = (rough[subset] | (light[subset] < interpolate(rough_med) * cfg["open_rel_lum"])) & ~(brown[subset] & ~winter)
    conifer = ~brown[subset] & (winter | (light[subset] < interpolate(green_med) * cfg["conifer_rel_lum"]))
    classes = np.where(forest, np.where(conifer, 1, 2), 0).astype(np.uint8)
    votes = np.stack([ndimage.uniform_filter((classes == i).astype(np.float32), 2 * margin + 1, mode="nearest")
                      for i in range(3)])
    return votes.argmax(0).astype(np.uint8)[margin:margin + (r1-r0), margin:margin + (c1-c0)]


def circle_mask(points, zones):
    out = np.zeros(len(points), bool)
    for z in zones:
        center = np.array(geo.to_local(*z["center_ll"]))
        out |= np.linalg.norm(points - center, axis=1) <= z["radius"]
    return out


def local_species_zones(zones, bounds):
    result = []
    for zone in zones:
        center = np.asarray(geo.to_local(*zone["center_ll"]), float)
        radius = zone["radius"]
        if (center + radius < bounds[:2]).any() or (center - radius > bounds[2:]).any():
            continue
        polygon = None
        if "inside_ll" in zone:
            polygon = Polygon(np.c_[geo.to_local(*np.asarray(zone["inside_ll"], float).T)]).buffer(0)
        result.append((center, radius, polygon))
    return result


def species_masks(points, zones):
    """旧コードが既存候補を外す円と、植え直す多角形の差を記録する。"""
    removed = np.zeros(len(points), dtype=bool)
    replanted = np.zeros(len(points), dtype=bool)
    for center, radius, polygon in zones:
        circle = np.linalg.norm(points - center, axis=1) < radius
        removed |= circle
        replanted |= circle if polygon is None else circle & shapely.contains_xy(polygon, *points.T)
    return removed, replanted


def maps(bounds, pieces, ground, cfg):
    whole = fm.ForestMap.load(config.DATA_PROCESSED / "forest_map.npz")
    r0, r1, c0, c1 = crop_map(whole, bounds)
    raster = fm.ForestMap(whole.x0 + c0 * whole.cell, whole.y1 - r0 * whole.cell,
                          whole.cell, whole.cls[r0:r1, c0:c1].copy())
    saved = fm.ForestMap(raster.x0, raster.y1, raster.cell, raster.cls.copy())
    px = cfg["map"]["px_size"]
    k = round(whole.cell / px)
    rgb = np.load(config.DATA_PROCESSED / "forest_photo.npy", mmap_mode="r")
    classify_cfg = {**fm.DEFAULT_CLASSIFY, **cfg["classify"]}
    raw = raw_photo(rgb, whole, (r0, r1, c0, c1), classify_cfg, k)
    tile = rgb[r0*k:r1*k, c0*k:c1*k]
    cache = (config.DATA_RAW / "osm_forest_open.json").resolve(strict=True)
    polys = fm.osm_open_areas(bounds, cache=cache)
    lakes = fm.osm_lake_polygons(bounds, cache=cache)
    buildings = [(v, t) for _, _, kind, v, t in pieces if kind == "building"]
    osm_buildings = [p for p in fm.osm_buildings(bounds, cache=cache) if overlaps(p, bounds)]
    open_map = fm.ForestMap(raster.x0, raster.y1, raster.cell, np.ones(raster.cls.shape, np.uint8))
    fm.clear_polygons(open_map, [p for p in polys if overlaps(p, bounds)])
    osm_open = open_map.cls == 0
    open_map.cls[:] = 1
    fm.clear_polygons(open_map, [p for p in lakes if overlaps(p, bounds)])
    house_mask = fm.settlement_mask(raster, osm_buildings, buildings, 25)
    rebuilt = fm.ForestMap(raster.x0, raster.y1, raster.cell, raw.copy())
    fm.clear_polygons(rebuilt, [p for p in polys if overlaps(p, bounds)])
    settlement = fm.settlement_mask(rebuilt, osm_buildings, buildings, classify_cfg["settle_m"])
    fm.settlement_filter(rebuilt, tile, px, settlement, classify_cfg)
    after_settlement = rebuilt.cls.copy()
    fm.farm_filter(rebuilt, tile, px, osm_buildings, buildings, classify_cfg)
    rr, cc = np.indices(raster.cls.shape)
    points = np.c_[raster.x0 + (cc.ravel() + 0.5) * raster.cell, raster.y1 - (rr.ravel() + 0.5) * raster.cell]
    manual = circle_mask(points, cfg["species_zones"] + cfg["fill_bare"] + cfg["keep_map"]).reshape(raster.cls.shape)
    keep_open = (osm_open & (open_map.cls != 0)) | house_mask | manual
    slope = fm.slope_map(raster, ground.z)
    level = json.loads((config.DATA_MANUAL / "s5_water.json").read_text())["lake_level_m"]
    fm.restore_above_water(raster, raw, [p for p in lakes if overlaps(p, bounds)], ground.z, level)
    fm.mountain_forest(raster, raw, tile, px, keep_open, slope)
    diagnostics = {
        "photo_open": raw == 0,
        "cached_map_open": saved.cls == 0,
        "cached_map_garden": saved.cls == 4,
        "map_open": raster.cls == 0,
        "map_garden": raster.cls == 4,
        "map_other_than_forest": ~np.isin(raster.cls, [1, 2]),
        "reconstructed_map_differs_from_cache": rebuilt.cls != saved.cls,
        "reconstructed_forest_class_lost_to_settlement": (raw > 0) & ~osm_open & (after_settlement == 0),
        "reconstructed_garden_class_from_settlement": (raw > 0) & ~osm_open & (after_settlement == 4),
        "reconstructed_forest_or_garden_class_lost_to_farm": (after_settlement > 0) & (rebuilt.cls == 0),
        "map_open_slope_ge_15": (raster.cls == 0) & (slope >= 15),
        "map_open_slope_ge_15_keep_map": (raster.cls == 0) & (slope >= 15) & manual,
        "map_open_slope_ge_15_protected": (raster.cls == 0) & (slope >= 15) & keep_open,
        "map_open_slope_ge_15_exempt_from_planting_by_photo": (raster.cls == 0) & (slope >= 15) & ~keep_open,
        "map_restored_by_slope": (saved.cls == 0) & (raster.cls > 0),
        "keep_map_or_species_or_fill": manual,
        "protected_landuse": osm_open,
        "protected_near_building": house_mask,
    }
    return raster, {k: fm.ForestMap(raster.x0, raster.y1, raster.cell, v.astype(np.uint8))
                    for k, v in diagnostics.items()}, whole


def crown_models():
    models = {}
    for name in SPECIES["sugi"] + SPECIES["hinoki"] + SPECIES["broadleaf"] + SPECIES["sapling"]:
        with ZipFile(BUILT / "extension" / "trees" / f"{name}.zip") as file:
            ini = configparser.ConfigParser()
            ini.read_string(file.read("tree.ini").decode("utf-8-sig"))
            height = ini.getfloat("BASIC", "HEIGHT")
            half = ini.getfloat("BASIC", "WIDTH") / 2
            leaves = np.concatenate([csptree.read_mesh(file.read(f"LOD{lod}.mesh"))[1][0].verts["p"]
                                     for lod in range(3)])
        models[name] = SimpleNamespace(height=height, crown_base=max(0, leaves[:, 1].min()),
                                       crown_radius=np.linalg.norm(leaves[:, [0, 2]], axis=1).max() * half)
    return models


def coverage(points, survey):
    if not len(points):
        return np.zeros(len(survey), dtype=bool)
    return cKDTree(points).query(survey)[0] <= RADIUS


def archives(bounds):
    def load(path):
        a = np.load(path, mmap_mode="r")
        mask = ((a["x"] >= bounds[0]) & (a["x"] <= bounds[2])
                & (a["y"] >= bounds[1]) & (a["y"] <= bounds[3]))
        return a[mask]

    historical = load(BUILT / "trees_before_r70.npy")
    sight = load(BUILT / "removed_sight.npy")
    pre_valley = load(VALLEY / "forest_work/trees_before_valleys.npy")
    post_valley = load(VALLEY / "deliver/forest/trees.npy")
    current = load(BASE / "trees_54/forest/trees.npy")
    key = lambda row: (row["x"], row["y"], row["pattern"])
    before = {key(r): r for r in pre_valley}
    after = {key(r): r for r in post_valley}
    missing = np.array([r for r in pre_valley if key(r) not in after], dtype=pre_valley.dtype)
    moved_z = np.array([r for r in post_valley if key(r) in before and r["z"] != before[key(r)]["z"]],
                       dtype=pre_valley.dtype)
    if len(missing) != len(pre_valley) - len(post_valley) or len(np.unique(xy(missing), axis=0)) != len(missing):
        raise ValueError("谷の前後の対応を一意に取れません")
    return {
        "forest_20261031_after_sight": historical,
        "removed_sight": sight,
        "pre_valley": pre_valley,
        "removed_valley": missing,
        "adjusted_height_valley": moved_z,
        "post_valley": post_valley,
        "trees_54": current,
    }


def archive_totals(data, survey, locate, models):
    result = {}
    for name, trees in data.items():
        points = xy(trees)
        low = fp._under(trees["pattern"])
        heights = np.array([models[p].height if p in models else 0. for p in trees["pattern"]]) * trees["scale"]
        covered = coverage(points, survey)
        scope = scopes(points, locate)
        result[name] = {
            side: {"trees": int((inside & ~low).sum()), "understory": int((inside & low).sum()),
                   "trees_10m_or_taller": int((inside & (heights >= 10)).sum())}
            for side, inside in scope.items()
        }
        if name in ("removed_valley", "removed_sight"):
            left = "post_valley" if name == "removed_valley" else "forest_20261031_after_sight"
            survivor = coverage(xy(data[left]), survey)
            inside_footprint = {key: coverage(points[inside], survey) for key, inside in scope.items()}
            for side, survey_mask in scopes(survey, locate).items():
                footprint = inside_footprint[side]
                result[name][side]["removed_footprint_m2"] = int((footprint & survey_mask).sum())
                result[name][side]["newly_empty_m2"] = int((covered & ~survivor & survey_mask).sum())
        if name in ("pre_valley", "post_valley", "trees_54"):
            tree_covered = coverage(points[~low], survey)
            for side, survey_mask in scopes(survey, locate).items():
                result[name][side]["empty_m2"] = int((~covered & survey_mask).sum())
                result[name][side]["empty_of_trees_m2"] = int((~tree_covered & survey_mask).sum())
    return result


def shifted_grid(origin, bounds, spacing, jitter, rng):
    offset = np.floor((bounds[:2] - origin) / spacing) - 1
    limits = np.r_[origin + offset * spacing, bounds[2:] + spacing]
    points = fp._grid(limits, spacing, jitter, rng)
    return points[contained(points, bounds)]


def replay(seed, bounds, whole, raster, ground, surface, keep, settings, models, zones):
    """旧間隔と確率の局所試行。全域生成の乱数列や手置きの木は再現しない。"""
    rng = np.random.default_rng(seed)
    origin = np.array([whole.x0, whole.y1 - whole.cell * whole.cls.shape[0]])
    tier = settings["tiers"][0]
    u = settings["understory"]
    young = settings["young_broadleaf"]
    garden = settings["garden"]
    for name, spacing, is_class, p, distance, size_range in (
        ("conifer", tier["spacing"]["conifer"], [1], 1., 45., tier["size"]),
        ("broadleaf", tier["spacing"]["broadleaf"], [2], 1., 45., tier["size"]),
        ("fern", u["fern_spacing"], [1, 2], u["fern_p"], u["max_d"], [.8, 1.25]),
        ("shrub", u["shrub_spacing"], [1, 2], u["shrub_p"], u["max_d"], [.8, 1.25]),
        ("young_broadleaf", young["spacing"], [1], young["p"], young["max_d"], young["size"]),
        ("garden_tree", garden["spacing"], [4], garden["p"], np.inf, garden["size"]),
        ("garden_shrub", garden["shrub_spacing"], [4], garden["shrub_p"], np.inf, [.8, 1.25]),
    ):
        points = shifted_grid(origin, bounds, spacing, settings["jitter"], rng)
        distance_mask = keep.course_distance(points) <= distance
        kind = raster.at(points)
        eligible = distance_mask & np.isin(kind, is_class)
        land_open = distance_mask & (kind == 0) & (not name.startswith("garden"))
        in_zone, planted_zone = species_masks(points, zones)
        under = name in ("fern", "shrub", "garden_shrub")
        selected = eligible & (rng.random(len(points)) < p)
        blocked = keep.blocked(points)
        z = ground.z(points)
        valid_z = np.isfinite(z)
        placed = selected & ~blocked & valid_z
        allowed = placed & ~in_zone
        over = np.zeros(len(points), bool)
        if not under and allowed.any():
            pp = points[allowed]
            scale = rng.uniform(*size_range, len(pp))
            if name == "conifer":
                hinoki = fp._patch_rand(pp, settings["patch_m"], 1) < settings["hinoki_share"]
                species = np.where(hinoki, "hinoki", "sugi")
            elif name == "garden_tree":
                species = np.where(rng.random(len(pp)) < garden["conifer_share"], "hinoki", "broadleaf")
            else:
                species = np.full(len(pp), "broadleaf")
            model = [models[SPECIES[n][rng.integers(len(SPECIES[n]))]] for n in species]
            crown_radius = np.array([m.crown_radius for m in model]) * scale
            crown_bottom = np.array([m.crown_base * m.height for m in model]) * scale
            d = keep.road_distance(pp)
            reach = d < crown_radius
            if reach.any():
                height = fp._road_z(pp[reach], d[reach], ground, keep, surface=surface)
                rejected = z[allowed][reach] + crown_bottom[reach] < height + settings["overhang_clear_m"]
                over[np.flatnonzero(allowed)[reach][rejected]] = True
        yield name, points, {
            "map_open_candidates": land_open,
            "map_garden_candidates_on_forest_grid": distance_mask & (kind == 4) & (not name.startswith("garden")),
            "eligible_before_probability": eligible,
            "understory_distance_rejected": ~distance_mask & np.isin(kind, is_class)
            if name in ("fern", "shrub") else np.zeros(len(points), bool),
            "understory_probability_rejected": eligible & ~selected if name in ("fern", "shrub") else np.zeros(len(points), bool),
            "clearance_rejected": selected & blocked,
            "no_ground_rejected": selected & ~blocked & ~valid_z,
            "species_zones_removed": placed & in_zone,
            "species_zones_removed_outside_planting_polygons": placed & in_zone & ~planted_zone,
            "overhang_rejected": over,
            "retained_excluding_manual_planting_and_view": allowed & ~over,
            "understory_after_clearance_and_zones_before_probability": eligible & ~blocked & valid_z & ~in_zone,
        }


def local_totals(trials, bounds, whole, raster, ground, surface, keep, cfg, survey, locate, models, zones):
    scope_survey = scopes(survey, locate)
    understory_ground = np.isin(raster.at(survey), [1, 2]) & ~keep.blocked(survey)
    understory_ground &= (keep.course_distance(survey) <= cfg["understory"]["max_d"]) & np.isfinite(ground.z(survey))
    understory_ground &= ~species_masks(survey, zones)[0]
    statistics = {side: {} for side in scope_survey}
    for seed in range(1, trials + 1):
        points_by_stage = {}
        for layer, points, stages in replay(seed, bounds, whole, raster, ground, surface, keep, cfg, models, zones):
            point_scopes = scopes(points, locate)
            for side, inside in point_scopes.items():
                layer_stats = statistics[side].setdefault(layer, {})
                for stage, mask in stages.items():
                    layer_stats.setdefault(stage, []).append(int((inside & mask).sum()))
            for stage, mask in stages.items():
                group = "understory" if layer in ("fern", "shrub") else "garden_understory" if layer == "garden_shrub" else "trees"
                points_by_stage.setdefault((group, stage), []).append(points[mask])
        by = {key: np.concatenate(value) for key, value in points_by_stage.items()}
        full_u = coverage(by["understory", "understory_after_clearance_and_zones_before_probability"], survey)
        thinned_u = coverage(by["understory", "retained_excluding_manual_planting_and_view"], survey)
        lost_overhang = coverage(by["trees", "overhang_rejected"], survey)
        remaining = "retained_excluding_manual_planting_and_view"
        left_all = coverage(np.concatenate([by[group, remaining] for group in ("trees", "understory", "garden_understory")]), survey)
        for side, mask in scope_survey.items():
            a = statistics[side].setdefault("combined_m2", {})
            a.setdefault("eligible_understory_ground", []).append(int((mask & understory_ground).sum()))
            a.setdefault("understory_empty_due_to_spacing", []).append(int((mask & understory_ground & ~full_u).sum()))
            a.setdefault("understory_empty_due_to_probability", []).append(int((mask & understory_ground & full_u & ~thinned_u).sum()))
            a.setdefault("overhang_footprint", []).append(int((mask & lost_overhang).sum()))
            a.setdefault("overhang_newly_empty", []).append(int((mask & lost_overhang & ~left_all).sum()))

    result = {
        side: {
            layer: {k: {"mean": round(float(np.mean(v)), 2), "min": min(v), "max": max(v)} for k, v in values.items()}
            for layer, values in layers.items()
        }
        for side, layers in statistics.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--out", "--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="既存の出力ファイルへの書き込みを許可")
    args = parser.parse_args()
    args.out = config.assert_writable(args.out)
    if (args.out.exists() or args.out.is_symlink()) and not args.force:
        parser.error(f"{args.out} は既にあります。上書きするには --force を指定してください")
    if args.trials < 1:
        parser.error("--trials は 1 以上を指定してください")
    cfg = json.loads((config.DATA_MANUAL / "forest.json").read_text())
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    locate, bounds = band(sec)
    piece_bounds = bounds + [-160, -160, 160, 160]
    pieces, extent = mesh_inputs(piece_bounds)
    ground = terrain_meshes(pieces)
    surface = terrain_meshes(pieces, road=True)
    keep, nearest_kind = keepout(pieces, extent, piece_bounds, cfg["clearance"])
    raster, map_mask, whole = maps(piece_bounds, pieces, ground, cfg)
    survey = grid_xy(bounds)
    road_loc = scopes(survey, locate)
    game = keep.blocked_game(survey)
    blocked = keep.blocked(survey)
    map_open = raster.at(survey) == 0
    view_clear = circle_mask(survey, cfg["view_clear"])
    rr, cc, _ = keep.fine.rc(survey)
    diag = {
        **{key: item.at(survey).astype(bool) for key, item in map_mask.items()},
        "survey_m2": np.ones(len(survey), bool),
        "clearance_all": blocked,
        "clearance_manifest": game,
        "clearance_osm_only": blocked & ~game,
        "clearance_after_map": blocked & ~map_open,
        "clearance_manifest_after_map": game & ~map_open,
        "clearance_osm_only_after_map": blocked & ~game & ~map_open,
        "view_clear": view_clear,
        "requested_keep_map": circle_mask(survey, cfg["keep_map"]),
        **{f"clearance_manifest_{key}": game & (nearest_kind[rr, cc] == i)
           for i, key in enumerate(fg.CLASSES, 1)},
    }
    zones = local_species_zones(cfg["species_zones"], bounds)
    cleared_zone, planting_zone = species_masks(survey, zones)
    diag["species_zones_clear_circles"] = cleared_zone
    diag["species_zones_planting_polygons"] = planting_zone
    diag["species_zones_cleared_outside_planting_polygons"] = cleared_zone & ~planting_zone
    diag["species_zones_cleared_outside_polygons_after_map_and_clearance"] = (
        cleared_zone & ~planting_zone & ~map_open & ~blocked
    )
    observed = {side: {k: int((region & mask).sum()) for k, mask in diag.items()} for side, region in road_loc.items()}
    data = archives(bounds)
    models = crown_models()
    result = {
        "sources": {"mesh_for_local_reconstruction": str(MANIFEST.relative_to(config.PROJECT_ROOT)),
                    "archived_sight": str((BUILT / "removed_sight.npy").relative_to(config.PROJECT_ROOT)),
                    "archived_valley": str(VALLEY.relative_to(config.PROJECT_ROOT)),
                    "archived_live": "export/compare/route70/work/trees_54/forest/trees.npy"},
        "area_units": "XY 平面の 1 m 格子。舗装端から外向きに測り、左右は上り基準。1 マスは 1 m²。",
        "without_tree_or_understory_if_no_stem_within_m": RADIUS,
        "map_clearance_m2": observed,
        "archived_counts_and_footprint": archive_totals(data, survey, locate, models),
        "early_stages_local_estimate": {
            "trials": args.trials,
            "method": "旧設定の格子間隔・位置のばらつき・抽選を使い、乱数の種を変えて局所で繰り返した。"
                      "開地に当たる候補は種類ごとの格子で数えた。針葉樹と広葉樹などの候補は代替のため、単純合計は実本数を表さない。"
                      "枝の高さを判定する前に、樹種指定区域の円で通常候補を除いた。区域内に植え直した個体や手植えの個体は台帳から確認できる。"
                      "旧森を生成した当日の道路・建物メッシュと途中候補は見つからず、局所試行には谷を掘る前の控えを使用した。"
                      "最小・最大は今回の試行範囲。",
            "results": local_totals(args.trials, bounds, whole, raster, ground, surface, keep,
                                   {**fp.DEFAULT_PLACE, **cfg["place"]}, survey, locate, models, zones),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w" if args.force else "x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
