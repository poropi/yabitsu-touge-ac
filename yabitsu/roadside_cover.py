"""県道本線に沿う山の地面を、現在の森に追加する株・木で覆う。"""
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree

from . import forest_place
from .crosssection import surface_height


@dataclass
class Location:
    s: np.ndarray
    towards: np.ndarray
    z: np.ndarray
    bridge: np.ndarray


class Road:
    def __init__(self, section):
        self.s = np.asarray(section["s"])
        self.z = np.asarray(section["z"])
        self.bridge = np.asarray(section["bridge"], bool)
        wl, wr = np.asarray(section["hw_left"]), np.asarray(section["hw_right"])
        self.hw = float(max(wl.max(), wr.max()))
        slope, crown = np.asarray(section.get("slope", 0)), np.asarray(section.get("crown", 0))
        self.high_z = np.maximum.reduce((self.z, surface_height(wl, self.z, slope, crown),
                                        surface_height(-wr, self.z, slope, crown)))
        self.xy = np.c_[section["x"], section["y"]]
        heading = np.asarray(section["heading"])
        normal = np.c_[-np.sin(heading), np.cos(heading)]
        left = self.xy + normal * np.asarray(section["hw_left"])[:, None]
        right = self.xy - normal * np.asarray(section["hw_right"])[:, None]
        self.lines = shapely.STRtree(shapely.linestrings(np.stack((self.xy[:-1], self.xy[1:]), axis=1)))
        a, b, c, d = left[:-1], right[:-1], right[1:], left[1:]
        tri = np.stack([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=1).reshape(-1, 3, 2)
        tri = shapely.polygons(tri)
        self.pavement = shapely.STRtree([shapely.union_all(tri[i:i+32]) for i in range(0, len(tri), 32)])

    def locate(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        ids = self.lines.query_nearest(shapely.points(xy), all_matches=False)[1]
        start = self.xy[ids]
        delta = self.xy[ids+1] - start
        fraction = np.clip(np.einsum("ij,ij->i", xy-start, delta) / np.einsum("ij,ij->i", delta, delta), 0, 1)
        s = self.s[ids] + fraction * (self.s[ids+1]-self.s[ids])
        z = self.z[ids] + fraction * (self.z[ids+1]-self.z[ids])
        return Location(s, start + fraction[:, None]*delta - xy, z, self.bridge[ids] & self.bridge[ids+1])

    def edge_distance(self, xy):
        _, distance = self.pavement.query_nearest(shapely.points(xy), all_matches=False, return_distance=True)
        return distance

    def interval_mask(self, xy, span):
        """絞り込み後、道路全体の最近接線分から区間を特定する。"""
        a, b = np.searchsorted(self.s, span)
        route = self.xy[max(0,a-1):b+2]
        lo, hi = route.min(axis=0), route.max(axis=0)
        lower_bound = np.linalg.norm(np.maximum(0, np.maximum(lo-xy, xy-hi)), axis=1)
        upper_bound = cKDTree(self.xy).query(xy)[0]
        candidates = np.flatnonzero(lower_bound <= upper_bound + 1e-8)
        kept = np.zeros(len(xy), bool)
        s = self.locate(xy[candidates]).s
        kept[candidates] = (s >= span[0]) & (s <= span[1])
        return kept

    def grid(self, distance):
        """世界座標で固定した 1 m 格子。本線の付近だけを列挙する。"""
        blocks = []
        for poly in self.pavement.geometries:
            lo = np.floor(np.asarray(poly.bounds[:2]) - distance).astype(int)
            hi = np.ceil(np.asarray(poly.bounds[2:]) + distance).astype(int)
            x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
            xy = np.c_[x.ravel(), y.ravel()]
            blocks.append(xy[shapely.dwithin(shapely.points(xy), poly, distance)])
        return np.unique(np.concatenate(blocks), axis=0).astype(float)


def genus(patterns):
    return np.asarray([str(p).split("_", 1)[0] for p in patterns])


def bare(xy, trees, radius):
    if not len(trees):
        return np.ones(len(xy), bool)
    return cKDTree(np.c_[trees.x, trees.y]).query(xy, distance_upper_bound=radius + 1e-9)[0] > radius


def eligible(xy, road, fm, ground, exclusion, cfg, *, minimum=None, maximum=None):
    distance = road.edge_distance(xy)
    lo = cfg["min_edge_m"] if minimum is None else minimum
    hi = cfg["max_edge_m"] if maximum is None else maximum
    valid = (distance >= lo) & (distance <= hi) & (distance > 0)
    valid &= open_mask(xy, exclusion) & ~road.locate(xy).bridge
    classes = fm.at(xy)
    z = np.full(len(xy), np.nan)
    z[valid] = ground.z(xy[valid])
    valid &= np.isfinite(z)
    i = np.flatnonzero(valid & ~np.isin(classes, (1, 2)))
    if len(i):
        p = xy[i]
        dx = ground.z(p + [.5, 0]) - ground.z(p - [.5, 0])
        dy = ground.z(p + [0, .5]) - ground.z(p - [0, .5])
        valid[i] &= np.hypot(dx, dy) >= np.tan(np.deg2rad(cfg["min_slope_deg"]))
    return valid, distance, z


def open_mask(xy, exclusion):
    if isinstance(exclusion, list):
        ok = np.ones(len(xy), bool)
        points = shapely.points(xy)
        for tree, distance in exclusion:
            if distance:
                ok[tree.query(points, predicate="dwithin", distance=distance)[0]] = False
            else:
                ok[tree.query(points, predicate="intersects")[0]] = False
        return ok
    return ~shapely.intersects_xy(exclusion, *xy.T)


def _spacing(xy, order, minimum):
    """先に通した候補から指定の間隔を取る。"""
    pairs = cKDTree(xy).query_ball_point(xy, minimum)
    free = np.ones(len(xy), bool)
    chosen = []
    for i in order:
        if free[i]:
            chosen.append(i)
            free[pairs[i]] = False
    return np.asarray(chosen, dtype=int)


def _cover_plants(targets, positions, old, cfg):
    """1 m 格子の空白を多く埋める候補から採用する。同点は種付きの順序。"""
    radius = cfg["cover_radius_m"]
    targets = targets[bare(targets, old, radius)]
    if not len(targets) or not len(positions):
        return np.empty(0, int)
    neighbors = cKDTree(targets).query_ball_point(positions, radius)
    reverse = cKDTree(positions).query_ball_point(targets, radius)
    missing = np.ones(len(targets), bool)
    scores = np.array([len(n) for n in neighbors])
    chosen = []
    for score in range(int(scores.max()), 0, -1):
        for i in np.flatnonzero(scores == score):
            if scores[i] != score:
                continue
            chosen.append(i)
            for j in neighbors[i]:
                if missing[j]:
                    missing[j] = False
                    scores[reverse[j]] -= 1
    return np.asarray(chosen, dtype=int)


def _rows(xy, z, names, species, rng, scale, max_height=None, models=None):
    rows = np.zeros(len(xy), dtype=forest_place.DTYPE).view(np.recarray)
    rows.x, rows.y = xy.T
    rows.z = z
    rows.pattern = [rng.choice(species[name]) for name in names]
    rows.angle_deg = rng.uniform(0, 360, len(rows))
    rows.scale = rng.uniform(*scale, len(rows))
    if max_height is not None:
        rows.scale = np.minimum(rows.scale, max_height / np.array([models[p].height for p in rows.pattern]))
    return rows


def limit_ground_crown(rows, road, models):
    """株の葉を舗装の外へ収める。一覧に書く座標・倍率の丸めも含める。"""
    xy = np.round(np.c_[rows.x, rows.y], 2)
    edge = road.edge_distance(xy)
    radius = np.array([models[name].crown_radius for name in rows.pattern])
    cap = np.floor(np.maximum(edge - .015, 0) / np.maximum(radius, 1e-6) * 100) / 100
    rows.scale = np.minimum(rows.scale, cap)


def safe_trees(trees, road, models, cfg, sight):
    """既存の見通し判定と、道路に届く枝の最低高を追加候補に適用する。"""
    if not len(trees):
        return np.ones(0, bool)
    trees = trees.copy()
    for field in ("x", "y", "z", "scale"):
        trees[field] = np.round(np.asarray(trees[field], float), 2)
    xy = np.c_[trees.x, trees.y]
    height = np.array([models[p].height for p in trees.pattern]) * trees.scale
    understory = forest_place._under(trees.pattern)
    radius = np.array([models[p].crown_radius for p in trees.pattern]) * trees.scale
    edge = road.edge_distance(xy)
    pos = road.locate(xy)
    base = np.array([models[p].crown_base for p in trees.pattern]) * height
    # 斜め方向の道路にも張り出し得るので、枝が届く範囲内の高い路面を使う。
    over = ~understory & (edge < radius)
    highest = pos.z.copy()
    for i in np.flatnonzero(over):
        hits = road.lines.query(shapely.Point(xy[i]), predicate="dwithin", distance=radius[i] + road.hw)
        if len(hits):
            highest[i] = max(np.max(road.high_z[hits]), np.max(road.high_z[hits+1]))
    keep = ~over | (trees.z + base >= highest + cfg["overhang_clear_m"])
    checked = np.flatnonzero(keep & (~understory | (height >= 1)))
    if len(checked):
        # 旧来の細い若木にも、新規配置では樹冠の見通しを適用する。
        test = trees[checked].copy()
        alias = {f"young_{p}": models[p] for p in set(test.pattern) if p.startswith("sapling")}
        test.pattern = [f"young_{p}" if p.startswith("sapling") else p for p in test.pattern]
        keep[checked] &= forest_place.sightline_keep(test, {**models, **alias}, road.xy, road.z, sight,
                                                      include_shrubs=True, check_trunks=True)
    return keep


def plant(grid, old, *, road, fm, ground, exclusion, models, species, manual_xy, cfg, place_cfg, sight):
    """範囲の対象格子から、既存の配置を維持して4層の追加行を返す。"""
    rng = np.random.default_rng(cfg["seed"])
    valid, _, _ = eligible(grid, road, fm, ground, exclusion, cfg)
    targets = grid[valid]
    jitter = rng.uniform(-cfg["jitter_m"], cfg["jitter_m"], targets.shape)
    candidates = targets + jitter
    allowed, edge, z = eligible(candidates, road, fm, ground, exclusion, cfg)
    # 境目と幹の近くは格子中心へ戻してから判定し直す。
    stems = cKDTree(np.asarray(manual_xy).reshape(-1, 2))
    clear = cfg["manual_clear_m"] + .015  # 一覧の 1 cm 丸めで 1 m を割らない余裕
    crowded = stems.query(candidates)[0] < clear
    candidates[~allowed | crowded] = targets[~allowed | crowded]
    allowed, edge, z = eligible(candidates, road, fm, ground, exclusion, cfg)
    allowed &= stems.query(candidates)[0] >= clear
    allowed &= bare(candidates, old, .15)
    p, z, edge = candidates[allowed], z[allowed], edge[allowed]
    order = rng.permutation(len(p))
    p, z, edge = p[order], z[order], edge[order]

    orig_kind = genus(old.pattern)
    under = np.isin(orig_kind, ("fern", "grass", "shrub"))
    plant_spacing = {**cfg, "cover_radius_m": cfg.get("plant_ground_radius_m", cfg["cover_radius_m"])}
    ground_cover = _cover_plants(targets, p, old[under], plant_spacing)
    grass_share = cfg.get("grass_share", .10)
    names = np.where(rng.random(len(ground_cover)) < grass_share, "grass", "fern")
    low = _rows(p[ground_cover], z[ground_cover], names, species, rng, cfg.get("ground_size", (.85, 1.45)),
                cfg.get("ground_height_m", .95), models)
    layers = [low]
    for name, spacing, sizes, limit in (("shrub", cfg["shrub_spacing_m"], (.55, .95), .98),
                                        ("sapling", cfg["young_spacing_m"], (.38, .65), None)):
        sel = bare(p, old[orig_kind == name], spacing)
        if name == "sapling":
            sel &= bare(p, old[~under], 2.)
        sel &= bare(p, low, .16)
        idx = np.flatnonzero(sel)
        options = _rows(p[idx], z[idx], [name]*len(idx), species, rng, sizes, limit, models)
        good = safe_trees(options, road, models, cfg, sight)
        options = options[good]
        chosen = _spacing(np.c_[options.x, options.y], np.arange(len(options)), spacing)
        added = options[chosen]
        layers.append(added)
        # 同じ候補点を後の層で使わない。
        used = idx[good][chosen]
        p = np.delete(p, used, axis=0)
        z, edge = np.delete(z, used), np.delete(edge, used)

    tall_xy = np.c_[old.x, old.y][np.isin(orig_kind, ("sugi", "hinoki", "broadleaf"))]
    tall_kind = orig_kind[np.isin(orig_kind, ("sugi", "hinoki", "broadleaf"))]
    outer = (road.edge_distance(tall_xy) > cfg["max_edge_m"]) if len(tall_xy) else np.zeros(0, bool)
    ok = (edge >= cfg["tall_min_edge_m"]) & bare(p, low, .16)
    p, z = p[ok], z[ok]
    if outer.any() and len(p):
        distance, nearest = cKDTree(tall_xy[outer]).query(p)
        context = tall_kind[outer][nearest]
        tier = place_cfg.get("tiers", forest_place.DEFAULT_PLACE["tiers"])[0]
        spacing = tier["spacing"]
        for forest, kinds in (("conifer", ("sugi", "hinoki")), ("broadleaf", ("broadleaf",))):
            same = np.isin(tall_kind, kinds)
            i = (distance <= cfg.get("forest_neighbor_m", 27.)) & np.isin(context, kinds)
            if same.any():
                i &= cKDTree(tall_xy[same]).query(p)[0] >= spacing[forest]
            options = _rows(p[i], z[i], context[i], species, rng, tier["size"])
            options = options[safe_trees(options, road, models, cfg, sight)]
            chosen = _spacing(np.c_[options.x, options.y], np.arange(len(options)), spacing[forest])
            layers.append(options[chosen])
    for layer in layers[:2]:
        limit_ground_crown(layer, road, models)
    return np.concatenate(layers).view(np.recarray)
