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


def _road_z(xy, d, ground, keep, h=1.0, surface=None):
    """一番近い道路の路面の高さ（距離の勾配の逆向きへ d だけ進んだ所）。surface（1ROAD の面）があればその高さ。
    道路の真下の地形は削ってあり、路面より平均 6 m 低い（2026-10-19 の全線の比較）ので、地面では代わりにならない。"""
    gx = keep.road_distance(xy + [h, 0]) - keep.road_distance(xy - [h, 0])
    gy = keep.road_distance(xy + [0, h]) - keep.road_distance(xy - [0, h])
    g = np.c_[gx, gy]
    n = np.linalg.norm(g, axis=1, keepdims=True)
    g = np.where(n > 1e-6, g / np.maximum(n, 1e-6), 0)
    z = np.full(len(xy), np.nan)
    if surface is not None:     # 距離の格子は 1 m ほどずれるので、道路の内側へ少しずつ深く探す
        for extra in (0.5, 1.5, 3.0, 5.0):
            todo = ~np.isfinite(z)
            if not todo.any():
                break
            z[todo] = surface.z(xy[todo] - g[todo] * (d[todo, None] + extra))
    todo = ~np.isfinite(z)
    z[todo] = ground.z(xy[todo] - g[todo] * (d[todo, None] + 0.5))
    return np.where(np.isfinite(z), z, ground.z(xy))


def place(area, fm, ground, keep, models, cfg, seed=1, species=None, road_surface=None, zones=()):
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

    # 植林の道路沿いに若い雑木の層を足す。上層の杉・檜は残し、広葉樹林には重ねない。
    # 後段の張り出し・見通し判定にも通すので、低い樹冠で道路を塞がない。
    if young := cfg.get("young_broadleaf"):
        P = _grid(bounds, young["spacing"], cfg["jitter"], rng)
        if isinstance(area, BaseGeometry):
            P = P[shapely.contains_xy(area, P[:, 0], P[:, 1])]
        P = P[fm.at(P) == 1]
        d = keep.course_distance(P)
        sel = (d <= young["max_d"]) & (rng.random(len(P)) < young["p"]) & ~keep.blocked(P)
        P, d = P[sel], d[sel]
        z = ground.z(P)
        ok = np.isfinite(z)
        pats = np.array([species["broadleaf"][rng.integers(len(species["broadleaf"]))]
                         for _ in range(ok.sum())], dtype="U16")
        size = rng.uniform(*young["size"], ok.sum()).astype(np.float32)
        out.append((P[ok], z[ok], pats, size, d[ok]))

    # 樹種を決める場所（zones）：中にあった木を除き、森の判定に関わらず指定の樹種と低木を植え直す
    fixed = [zn["inside"] for zn in zones if "inside" in zn]   # 中身を決めてある所（島など）。ほかの場所の木は入れない
    for zn in zones:
        cx, cy = zn["center"]; R = zn["radius"]
        # 前に植えた木を植え直す。ただし位置や範囲を決めて植えた木（ほかの場所の分）は残す
        out = [(o[0][m_], o[1][m_], o[2][m_], o[3][m_], o[4][m_]) for o in out
               for m_ in [(np.hypot(o[0][:, 0] - cx, o[0][:, 1] - cy) >= R) | (o[4] >= 1e8)]]
        under_sp = zn.get("shrub_species", "shrub")   # 木の下に植えるもの（低木、または草むらの代わりのシダ）
        for name, spacing in ((zn["species"], zn["spacing"]), (under_sp, zn.get("shrub_spacing", zn["spacing"] * 0.8))):
            under = name == under_sp
            if not under and "points" in zn:   # 木の位置を決めてある（支柱を立てる若木など）
                Q = np.asarray(zn["points"], float).reshape(-1, 2)
                zq = ground.z(Q); pats_q = np.array([species[name][rng.integers(len(species[name]))] for _ in range(len(Q))], dtype="U16")
                sz = rng.uniform(*zn.get("size", [0.6, 1.0]), len(Q)).astype(np.float32)
                out.append((Q, zq, pats_q, sz, np.full(len(Q), 1e9)))
                continue
            Q = _grid((cx - R, cy - R, cx + R, cy + R), spacing, cfg["jitter"], rng)
            Q = Q[np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) < R]
            if "inside" in zn:   # 多角形の中だけに植える（島など）
                Q = Q[shapely.contains_xy(zn["inside"], Q[:, 0], Q[:, 1])]
            else:
                for g in fixed:
                    Q = Q[~shapely.contains_xy(g, Q[:, 0], Q[:, 1])]
            # road_clear：植えない範囲の代わりに道路からの距離だけで決める（幅の狭い分岐の島など）
            dist = keep.fine_road_distance if ("road_clear" in zn or zn.get("clear") == "game") else keep.road_distance
            if zn.get("clear") == "game":   # ゲームの中の道・建物だけを避ける（地図の小道の線では消さない）
                Q = Q[(dist(Q) >= zn.get("road_clear", 2.0)) & ~keep.blocked_game(Q)]
            else:
                Q = Q[dist(Q) >= zn["road_clear"]] if "road_clear" in zn else Q[~keep.blocked(Q)]
            zq = ground.z(Q); okq = np.isfinite(zq); Q, zq = Q[okq], zq[okq]
            pats_q = np.array([species[name][rng.integers(len(species[name]))] for _ in range(len(Q))], dtype="U16")
            sz = rng.uniform(*(zn.get("size", [0.6, 1.0]) if not under else zn.get("shrub_size", [0.8, 1.3])), len(Q)).astype(np.float32)
            if under and len(Q):   # 低木は駐車場の縁から 3.5 m 離す（車止めの外の草地）
                ok2 = dist(Q) >= zn.get("road_clear", 3.5)
                Q, zq, pats_q, sz = Q[ok2], zq[ok2], pats_q[ok2], sz[ok2]
            if not under and len(Q):   # 枝は駐車場の上に少し出てよいが、樹冠の半分より近くには植えない
                crr = np.array([models[pp].crown_radius for pp in pats_q]) * sz
                ok2 = dist(Q) >= crr * zn.get("overhang", 0.5)
                Q, zq, pats_q, sz = Q[ok2], zq[ok2], pats_q[ok2], sz[ok2]
            out.append((Q, zq, pats_q, sz, np.full(len(Q), 1e9)))   # 場所で決めた木は、張り出しの判定をしない（駐車場の上に枝が出てよい）

    P = np.concatenate([o[0] for o in out]) if out else np.zeros((0, 2))
    z = np.concatenate([o[1] for o in out]) if out else np.zeros(0)
    pats = np.concatenate([o[2] for o in out]) if out else np.zeros(0, "U16")
    size = np.concatenate([o[3] for o in out]) if out else np.zeros(0, np.float32)
    d = np.concatenate([o[4] for o in out]) if out else np.zeros(0)

    zone_tree = d >= 1e8 if len(P) else np.zeros(0, bool)
    d = keep.road_distance(P) if len(P) else d      # 張り出しは OSM の道も含めた道路からの距離で見る
    # 道路の上に張り出す木：枝の最低の高さが路面＋overhang_clear_m 未満なら植えない
    cb = np.array([models[p].crown_base * models[p].height for p in pats]) if len(pats) else np.zeros(0)
    cr = np.array([models[p].crown_radius for p in pats]) if len(pats) else np.zeros(0)
    under = np.char.startswith(pats, "shrub") | np.char.startswith(pats, "fern") if len(pats) else np.zeros(0, bool)
    over = (~under) & (d < cr * size) & ~zone_tree
    if over.any():
        rz = _road_z(P[over], d[over], ground, keep, surface=road_surface)
        low = z[over] + cb[over] * size[over] < rz + cfg["overhang_clear_m"]
        drop = np.flatnonzero(over)[low]
        mask = np.ones(len(P), bool); mask[drop] = False
        P, z, pats, size = P[mask], z[mask], pats[mask], size[mask]

    t = np.zeros(len(P), DTYPE)
    t["x"], t["y"], t["z"], t["pattern"], t["scale"] = P[:, 0], P[:, 1], z, pats, size
    t["angle_deg"] = rng.uniform(0, 360, len(P))
    return t.view(np.recarray)


def fill_bare(trees, ground, keep, species, zone, seed=7):
    """むき出しの地面を埋める（data/manual/forest.json の fill_bare）：円の中で、ゲームの道・建物・壁から離れていて、
    近くに木も茂みも無い所に、広葉樹と茂みを植える。open（円の並び）の中は草地のまま残す。足した木の配列を返す。"""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    cx, cy = zone["center"]; R = zone["radius"]; sp = zone.get("spacing", 3.0)
    Q = _grid((cx - R, cy - R, cx + R, cy + R), sp, 0.3, rng)
    Q = Q[np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) < R]
    for ox, oy, orad in zone.get("open", []):
        Q = Q[np.hypot(Q[:, 0] - ox, Q[:, 1] - oy) >= orad]
    Q = Q[(keep.fine_road_distance(Q) >= zone.get("road_clear", 2.5)) & ~keep.blocked_game(Q)]
    pats = np.asarray(trees.pattern)
    cover = ~np.char.startswith(pats, "fern")
    if cover.any() and len(Q):
        d, _ = cKDTree(np.c_[trees.x, trees.y][cover]).query(Q)
        Q = Q[d >= zone.get("bare_m", 3.5)]
    z = ground.z(Q); ok = np.isfinite(z); Q, z = Q[ok], z[ok]
    n = len(Q)
    tree = np.arange(n) % 2 == 0                     # 半分に広葉樹、全部の所に茂み（少しずらす）
    P = np.r_[Q[tree], Q + rng.uniform(-0.8, 0.8, Q.shape)]
    zz = np.r_[z[tree], ground.z(P[tree.sum():])]
    names = np.r_[[species["broadleaf"][rng.integers(len(species["broadleaf"]))] for _ in range(tree.sum())],
                  [species["shrub"][rng.integers(len(species["shrub"]))] for _ in range(n)]].astype("U16")
    sc = np.r_[rng.uniform(*zone.get("size", [0.8, 1.2]), tree.sum()), rng.uniform(1.2, 1.8, n)].astype(np.float32)
    ok = np.isfinite(zz)
    t = np.zeros(int(ok.sum()), DTYPE)
    t["x"], t["y"], t["z"], t["pattern"], t["scale"] = P[ok, 0], P[ok, 1], zz[ok], names[ok], sc[ok]
    t["angle_deg"] = rng.uniform(0, 360, len(t))
    return t.view(np.recarray)


# 視界：コースの上を走る目の高さから先の路面への見通しの線に、樹冠がかかる木を取り除く（data/manual/forest.json の "sight"）。
# 見通す距離は短め（道にはみ出した枝を取る。カーブの内側の森そのものは残す）。
DEFAULT_SIGHT = {
    "step_m": 2.0,                       # 中心線の点の間隔
    "ahead_m": [6, 12, 20, 28, 35],      # 先の路面までの距離
    "lateral_m": [-1.6, 0.0, 1.6],       # 目の位置（左右の車線・中央）
    "eye_h": 1.2, "target_h": [1.0, 3.0],
    "crown_shrink": 0.85,                # 樹冠の半径（葉の先端）を少し内側に
    "margin_m": 0.3,
    "near_m": 60.0,                      # これより道から遠い木は調べない
}


def _resample(cl, z, step):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cl, axis=0), axis=1))]
    ss = np.arange(0, s[-1], step)
    return np.c_[np.interp(ss, s, cl[:, 0]), np.interp(ss, s, cl[:, 1])], np.interp(ss, s, z)


def sightline_keep(trees, models, cl, cl_z, cfg) -> np.ndarray:
    """True＝残す。下草（シダ・低木）と、植えたばかりの細い若木（sapling）は調べない。"""
    from scipy.spatial import cKDTree
    cfg = {**DEFAULT_SIGHT, **cfg}
    P, Z = _resample(np.asarray(cl, float), np.asarray(cl_z, float), cfg["step_m"])
    d = np.gradient(P, axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.c_[-d[:, 1], d[:, 0]]
    keep = np.ones(len(trees), bool)
    pats = np.asarray(trees.pattern)
    big = ~(np.char.startswith(pats, "shrub") | np.char.startswith(pats, "fern") | np.char.startswith(pats, "sapling"))
    xy = np.c_[trees.x, trees.y]
    dist, i0 = cKDTree(P).query(xy)
    cand = np.flatnonzero(big & (dist < cfg["near_m"]))
    if not len(cand):
        return keep
    c = xy[cand]; i0 = i0[cand]
    sc = np.asarray(trees.scale, float)[cand]
    m = [models[p] for p in pats[cand]]
    H = np.array([x.height for x in m]) * sc
    zb = np.asarray(trees.z, float)[cand] + np.array([x.crown_base for x in m]) * H
    zt = np.asarray(trees.z, float)[cand] + H
    R = np.array([x.crown_radius for x in m]) * sc * cfg["crown_shrink"]
    cone = np.char.startswith(pats[cand], "sugi") | np.char.startswith(pats[cand], "hinoki")
    hit = np.zeros(len(cand), bool)
    n = len(P)
    reach = int(np.ceil((R.max() + cfg["margin_m"] + max(map(abs, cfg["lateral_m"]))) / cfg["step_m"])) + 1
    for ahead in cfg["ahead_m"]:
        k = max(1, int(round(ahead / cfg["step_m"])))
        for sgn in (1, -1):              # 上り・下り
            for off in range(-k - reach, reach + 1):
                i = i0 + off * sgn
                j = i + k * sgn
                ok = (i >= 0) & (i < n) & (j >= 0) & (j < n) & ~hit
                if not ok.any():
                    continue
                ii, jj = i[ok], j[ok]
                for lat in cfg["lateral_m"]:
                    a2 = P[ii] + nrm[ii] * lat * sgn
                    b2 = P[jj] + nrm[jj] * lat * sgn
                    ab = b2 - a2
                    t = np.clip(np.einsum("ij,ij->i", c[ok] - a2, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
                    dd = np.linalg.norm(a2 + ab * t[:, None] - c[ok], axis=1)
                    for th in cfg["target_h"]:
                        h = Z[ii] + cfg["eye_h"] + t * (Z[jj] + th - Z[ii] - cfg["eye_h"])
                        f = np.clip((zt[ok] - h) / np.maximum(zt[ok] - zb[ok], 1e-6), 0, 1) ** 0.9
                        r = np.where(cone[ok], R[ok] * f, R[ok])
                        inside = (h >= zb[ok]) & (h <= zt[ok]) & (dd < r + cfg["margin_m"])
                        idx = np.flatnonzero(ok)[inside]
                        hit[idx] = True
    keep[cand[hit]] = False
    return keep


def view_clear_keep(trees, models, zones) -> np.ndarray:
    """眺めの開けた場所（zones：中心・半径・展望の高さ z・余裕 margin）：その場所より低い地面に立ち、梢が z＋margin を
    超える木を取り除く。True＝残す。2026-10-20：菜の花台の駐車場は谷側の眺めが開けている（ストリートビュー）。"""
    keep = np.ones(len(trees), bool)
    pats = np.asarray(trees.pattern)
    H = np.array([models[p].height for p in pats]) * np.asarray(trees.scale, float) if len(pats) else np.zeros(0)
    for zn in zones:
        d = np.hypot(trees.x - zn["center"][0], trees.y - zn["center"][1])
        top = trees.z + H
        keep &= ~((d < zn["radius"]) & (trees.z < zn["z"]) & (top > zn["z"] + zn.get("margin", 0.0)))
    return keep
