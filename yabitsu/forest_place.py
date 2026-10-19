"""森の木を配置する。空中写真の地図（森／針葉樹／広葉樹）・地面の高さ・植えない範囲に従い、ゲームの中の道（コース）からの
距離の帯ごとに密度と大きさを変える（OSM の道は植えない範囲と張り出しの判定にだけ使う）。"""
import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

# data/manual/forest.json の "place" で上書きする
DEFAULT_PLACE = {
    # 道路からの距離 max_d m までの帯。spacing は木の間隔 m（ほぼ格子にばらつきを付ける）、size は大きさの倍率
    "tiers": [
        {"max_d": 60, "spacing": {"conifer": 4.0, "broadleaf": 6.0}, "size": [0.85, 1.15]},
        {"max_d": 250, "spacing": {"conifer": 6.0, "broadleaf": 8.0}, "size": [0.9, 1.2]},
        {"max_d": 800, "spacing": {"conifer": 9.0, "broadleaf": 11.0}, "size": [1.0, 1.3]},
        {"max_d": 1e9, "spacing": {"conifer": 13.0, "broadleaf": 15.0}, "size": [1.1, 1.4]},
    ],
    "blend_m": 30.0,          # 帯の境目をなめらかにする幅
    "hinoki_share": 0.4,      # 針葉樹のうち檜の割合
    "patch_m": 50.0,          # 植林のまとまり（同じ樹種にそろえる区画）の大きさ
    "jitter": 0.42,           # 格子からのずれ（間隔に対する割合）
    "understory": {"max_d": 30.0, "fern_spacing": 3.5, "fern_p": 0.45, "shrub_spacing": 6.0, "shrub_p": 0.35},
    "overhang_clear_m": 5.0,  # 道路の上に張り出す枝の、路面からの最低の高さ
}
DTYPE = [("x", "f8"), ("y", "f8"), ("z", "f8"), ("pattern", "U16"), ("angle_deg", "f4"), ("scale", "f4")]


def _grid(bounds, spacing, jitter, rng):
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0 + spacing / 2, x1, spacing); ys = np.arange(y0 + spacing / 2, y1, spacing)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel()]
    return P + rng.uniform(-jitter, jitter, P.shape) * spacing


def _ramp(d, lo, hi, blend):
    """帯 [lo, hi) の中で 1、外側 blend m で 0 へ下がる（隣の帯と半分ずつ重なる）。"""
    a = np.clip((d - (lo - blend / 2)) / blend, 0, 1) if lo > 0 else np.ones_like(d)
    b = np.clip(((hi + blend / 2) - d) / blend, 0, 1) if hi < 1e8 else np.ones_like(d)
    return np.minimum(a, b)


def _patch_rand(xy, size, seed):
    i = np.floor(xy[:, 0] / size).astype(np.int64); j = np.floor(xy[:, 1] / size).astype(np.int64)
    h = (i * 73856093) ^ (j * 19349663) ^ (seed * 83492791)
    return (h % 10007) / 10007.0


def _road_z(xy, d, ground, keep, h=1.0):
    """一番近い道路の路面の高さ（距離の勾配の逆向きへ d だけ進んだ所の地面）。"""
    gx = keep.road_distance(xy + [h, 0]) - keep.road_distance(xy - [h, 0])
    gy = keep.road_distance(xy + [0, h]) - keep.road_distance(xy - [0, h])
    g = np.c_[gx, gy]
    n = np.linalg.norm(g, axis=1, keepdims=True)
    g = np.where(n > 1e-6, g / np.maximum(n, 1e-6), 0)
    z = ground.z(xy - g * d[:, None])
    return np.where(np.isfinite(z), z, ground.z(xy))


def place(area, fm, ground, keep, models, cfg, seed=1, species=None):
    if species is None:
        from .treemodels import SPECIES as species
    cfg = {**DEFAULT_PLACE, **cfg}
    rng = np.random.default_rng(seed)
    if isinstance(area, BaseGeometry):
        bounds = area.bounds
    else:
        bounds = (fm.x0, fm.y1 - fm.cls.shape[0] * fm.cell, fm.x0 + fm.cls.shape[1] * fm.cell, fm.y1)
    out = []

    lo = 0.0
    for tier in cfg["tiers"]:
        hi = tier["max_d"]
        for kind, cls_id in (("conifer", 1), ("broadleaf", 2)):
            P = _grid(bounds, tier["spacing"][kind], cfg["jitter"], rng)
            if isinstance(area, BaseGeometry):
                P = P[shapely.contains_xy(area, P[:, 0], P[:, 1])]
            P = P[fm.at(P) == cls_id]
            if not len(P):
                continue
            d = keep.course_distance(P)
            w = _ramp(d, lo, hi, cfg["blend_m"])
            sel = rng.random(len(P)) < w
            P, d = P[sel], d[sel]
            bad = keep.blocked(P)
            P, d = P[~bad], d[~bad]
            z = ground.z(P)
            ok = np.isfinite(z)
            P, d, z = P[ok], d[ok], z[ok]
            if kind == "conifer":
                hin = _patch_rand(P, cfg["patch_m"], seed) < cfg["hinoki_share"]
                names = np.where(hin, "hinoki", "sugi")
            else:
                names = np.full(len(P), "broadleaf")
            pats = np.array([species[n][rng.integers(len(species[n]))] for n in names], dtype="U16")
            size = rng.uniform(*tier["size"], len(P)).astype(np.float32)
            out.append((P, z, pats, size, d))
        lo = hi

    # 下草（シダ・低木）：道路のそばの森の中だけ
    u = cfg["understory"]
    for name, spacing, p in (("fern", u["fern_spacing"], u["fern_p"]), ("shrub", u["shrub_spacing"], u["shrub_p"])):
        P = _grid(bounds, spacing, cfg["jitter"], rng)
        if isinstance(area, BaseGeometry):
            P = P[shapely.contains_xy(area, P[:, 0], P[:, 1])]
        P = P[(fm.at(P) > 0) & (fm.at(P) < 3)]
        d = keep.course_distance(P)
        sel = (d <= u["max_d"]) & (rng.random(len(P)) < p)
        P, d = P[sel], d[sel]
        bad = keep.blocked(P)
        P, d = P[~bad], d[~bad]
        z = ground.z(P)
        ok = np.isfinite(z)
        pats = np.array([species[name][rng.integers(len(species[name]))] for _ in range(ok.sum())], dtype="U16")
        out.append((P[ok], z[ok], pats, rng.uniform(0.8, 1.25, ok.sum()).astype(np.float32), d[ok]))

    P = np.concatenate([o[0] for o in out]) if out else np.zeros((0, 2))
    z = np.concatenate([o[1] for o in out]) if out else np.zeros(0)
    pats = np.concatenate([o[2] for o in out]) if out else np.zeros(0, "U16")
    size = np.concatenate([o[3] for o in out]) if out else np.zeros(0, np.float32)
    d = np.concatenate([o[4] for o in out]) if out else np.zeros(0)

    d = keep.road_distance(P) if len(P) else d      # 張り出しは OSM の道も含めた道路からの距離で見る
    # 道路の上に張り出す木：枝の最低の高さが路面＋overhang_clear_m 未満なら植えない
    cb = np.array([models[p].crown_base * models[p].height for p in pats]) if len(pats) else np.zeros(0)
    cr = np.array([models[p].crown_radius for p in pats]) if len(pats) else np.zeros(0)
    under = np.char.startswith(pats, "shrub") | np.char.startswith(pats, "fern") if len(pats) else np.zeros(0, bool)
    over = (~under) & (d < cr * size)
    if over.any():
        rz = _road_z(P[over], d[over], ground, keep)
        low = z[over] + cb[over] * size[over] < rz + cfg["overhang_clear_m"]
        drop = np.flatnonzero(over)[low]
        mask = np.ones(len(P), bool); mask[drop] = False
        P, z, pats, size = P[mask], z[mask], pats[mask], size[mask]

    t = np.zeros(len(P), DTYPE)
    t["x"], t["y"], t["z"], t["pattern"], t["scale"] = P[:, 0], P[:, 1], z, pats, size
    t["angle_deg"] = rng.uniform(0, 360, len(P))
    return t.view(np.recarray)
