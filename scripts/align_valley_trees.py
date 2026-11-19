"""控えと新地形を照合し、谷や擁壁で動いた地面に載る既存の木だけ補正する。"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

from scripts.align_s5_8960_trees import HEIGHT_THRESHOLD, apply_forest_changes, ground_changes
from scripts.clear_fubuki_trees import VALLEY_TREE_ORDER
from yabitsu import config, track, valleys, valley_fit
from yabitsu.crosssection import Section


@dataclass(frozen=True)
class Change:
    index: int
    old_ground: float
    new_ground: float
    reason: str


def ground_pairs(before_path, after_path, sec, rows, bounds):
    """範囲外の標高と Z 以外の配列が一致することを確かめ、各地点を切り出す。"""
    with (np.load(before_path) as before, np.load(after_path) as after):
        if set(before.files) != set(after.files):
            raise ValueError("新旧の近景地形の配列名が異なります")
        for key in before.files:
            if key == "Z":
                continue
            a, b = before[key], after[key]
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                raise ValueError(f"近景地形の {key} が変わっています")
        xs, ys, z_old, z_new = before["xs"], before["ys"], before["Z"], after["Z"]
        if z_old.shape != z_new.shape or z_old.dtype != z_new.dtype:
            raise ValueError("新旧の近景地形の Z の型または形状が異なります")
        old_bits = z_old.view(np.dtype(f"V{z_old.dtype.itemsize}"))
        new_bits = z_new.view(np.dtype(f"V{z_new.dtype.itemsize}"))
        r, c = np.nonzero(old_bits != new_bits)
        x, y = xs[c], ys[r]
        halo = np.hypot(np.diff(xs).max(), np.diff(ys).max())
        assigned = np.zeros(len(r), bool)
        grids = []
        d, holes = before["D"], before["H"]
        for section, center in dict.fromkeys((v.section, v.center_m) for v in rows):
            site = valleys.site_area(sec, bounds, section, center)
            inside = shapely.contains_xy(site.buffer(-halo), x, y)
            if not inside.any():
                continue
            if (inside & assigned).any():
                raise ValueError("地形の変更点が複数の谷の範囲に重なっています")
            assigned |= inside
            r0, c0 = max(r[inside].min()-1, 0), max(c[inside].min()-1, 0)
            r1, c1 = min(r[inside].max()+2, len(ys)), min(c[inside].max()+2, len(xs))
            axes = {"xs": xs[c0:c1], "ys": ys[r0:r1],
                    "D": d[r0:r1, c0:c1].copy(), "H": holes[r0:r1-1, c0:c1-1].copy()}
            sub_old = z_old[r0:r1, c0:c1].copy()
            sub_new = sub_old.copy()
            sub_new[r[inside]-r0, c[inside]-c0] = z_new[r[inside], c[inside]]
            grids.append(({**axes, "Z": sub_old}, {**axes, "Z": sub_new}))
        if not assigned.all():
            raise ValueError(f"3か所の外で Z が変わった点が {np.count_nonzero(~assigned)} 点あります")
    return grids


def plan_changes(trees, pairs, rows, *, waters=()):
    result = []
    for old, new in pairs:
        ids, before, after = ground_changes(old, new, trees.x, trees.y)
        changed = abs(after - before) > HEIGHT_THRESHOLD
        ids, before, after = ids[changed], before[changed], after[changed]
        x, y = trees.x[ids], trees.y[ids]
        bottom = np.zeros(len(ids), bool)
        under_water = np.zeros(len(ids), bool)
        for valley in rows:
            _, at, normal, arc = valley.project(x, y)
            bottom |= ((before > after) & (abs(normal) <= valley.width_m / 2) & (at > 0) & (at < arc[-1]))
        for polygon, height in waters:
            wet = shapely.intersects_xy(polygon, x, y)
            under_water[wet] |= after[wet] < height(x[wet], y[wet])
        reason = np.select([under_water, bottom], ["水面", "谷底"], "")
        result.extend(Change(int(i), float(a), float(b), str(why)) for i, a, b, why in zip(ids, before, after, reason))
    return sorted(result, key=lambda c: c.index)



def plan_forest_changes(trees, dug, rows, surfaces):
    """掘った面に根元がある個体だけ、描画三角形の標高へ合わせる。"""
    xy = np.c_[trees.x, trees.y]
    ids = np.flatnonzero(dug.covers(xy))
    points = xy[ids]
    before, after = dug.old.height(points), dug.new.height(points)
    if not (np.isfinite(before).all() and np.isfinite(after).all()):
        raise ValueError("地形の高さを取れない木があります")
    if dug.bed is not None:
        bottom, wet = dug.tree_exclusions(points, surfaces[1])
        reasons = np.select([wet, bottom], ["水面", "谷底"], "")
        changed = (trees.z[ids] != after.astype(trees.z.dtype)) | bottom | wet
        return [Change(int(i), float(a), float(b), str(why)) for i, a, b, why in
                zip(ids[changed], before[changed], after[changed], reasons[changed])]
    changed = before - after > HEIGHT_THRESHOLD
    bottom = valleys.tree_bottom(rows, *points.T, forest=True)
    wet = valleys.newly_wet(dug, *surfaces, points)
    reason = np.select([wet, bottom], ["水面", "谷底"], "")
    take = (changed & (trees.z[ids] != after)) | (wet | (bottom & changed))
    return [Change(int(i), float(a), float(b), str(why))
            for i, a, b, why in zip(ids[take], before[take], after[take], reason[take])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=VALLEY_TREE_ORDER)
    parser.add_argument("--before-near", required=True, type=Path)
    parser.add_argument("--after-near", type=Path, default=config.DATA_PROCESSED / "track_terrain_near.npz")
    parser.add_argument("--forest-dir", required=True, type=Path)
    parser.add_argument("--before-track", required=True, type=Path)
    parser.add_argument("--track", type=Path, default=config.TRACK_BUILD_DIR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.before_near.resolve() == args.after_near.resolve():
        parser.error("新旧の地形に別のファイルを指定してください")
    print(VALLEY_TREE_ORDER)
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    rows = valleys.load()
    with np.load(args.after_near) as data:
        new = dict(data)
    if "valley_faces" in new:
        old = valleys.read_before_near(args.before_near, new)
        sites = {r.id: valleys.site_area(sec, bounds, r.section, r.center_m) for r in rows}
        issues, _ = valley_fit.grid_checks(old, new, sites)
        if issues:
            raise ValueError(f"細分地形の範囲または接続に異常があります: {issues[0]}")
    else:
        if not ground_pairs(args.before_near, args.after_near, sec, rows, bounds):
            print("地形の変更はありません")
            return
        with np.load(args.before_near) as data:
            old = dict(data)
    dug = valleys.changed_ground(old, new)
    surfaces = tuple(valley_fit.river_surface(root) for root in (args.before_track, args.track))
    trees = np.load(args.forest_dir / "trees.npy").view(np.recarray)
    changes = plan_forest_changes(trees, dug, rows, surfaces)
    print(f"対象 {len(changes)} 本（除外 {sum(bool(c.reason) for c in changes)} 本）")
    for c in changes:
        i = c.index
        print(f"行 {i} XY {trees.x[i]:.2f}, {trees.y[i]:.2f}: {c.old_ground:.3f} → {c.new_ground:.3f} m {c.reason or '根元補正'}")
    if args.apply:
        for path in apply_forest_changes(args.forest_dir, trees, changes, "_before_valleys"):
            print(path)


if __name__ == "__main__":
    main()
