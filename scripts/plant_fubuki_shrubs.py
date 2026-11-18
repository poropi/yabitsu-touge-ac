"""吹風入口の右歩道外に低木を足す。既定では追加本数だけを表示する。"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from yabitsu import config, features, forest_ac, terrain, track
from yabitsu.crosssection import Section


def roadside_shrubs(sec, trees, feats, s5, ground):
    """歩道外の三列に、既存木の根元を避けて高さの異なる低木を植える。"""
    rng = np.random.default_rng(20261118)
    walk = features.sidewalk_offsets(sec, feats)[1]
    edge = sec.hw_right + features.GUTTER_WIDTH + walk
    plants = []
    # 低い前列で地面を隠し、その奥はガードパイプを越える高さにする。
    for row, (gap, pitch, pattern, scale) in enumerate((
        (1.25, 1.25, "shrub_a", (1.30, 1.55)),
        (2.65, 1.35, "shrub_b", (1.38, 1.65)),
        (4.10, 1.60, "shrub_b", (1.25, 1.50)),
    )):
        s = np.arange(10608.6 + row * 0.39, 10682.0, pitch) + s5
        s += rng.uniform(-0.13, 0.13, len(s))
        h = np.interp(s, sec.s, np.unwrap(sec.heading))
        right = np.interp(s, sec.s, edge) + gap + rng.uniform(-0.10, 0.10, len(s))
        x = np.interp(s, sec.s, sec.x) + right * np.sin(h)
        y = np.interp(s, sec.s, sec.y) - right * np.cos(h)
        z = ground(x, y)
        angles = rng.uniform(0, 360, len(s))
        scales = rng.uniform(*scale, len(s))
        road_z = np.interp(s, sec.s, sec.edge_z(-1))
        keep = np.isfinite(z) & (z >= road_z - 3.5)
        plants.extend((x[i], y[i], z[i], pattern, angles[i], scales[i]) for i in np.flatnonzero(keep))

    out = np.array(plants, dtype=trees.dtype).view(np.recarray)
    if not len(out):
        return out
    xy = np.c_[out.x, out.y]
    nearby = ((trees.x >= xy[:, 0].min() - 1) & (trees.x <= xy[:, 0].max() + 1)
              & (trees.y >= xy[:, 1].min() - 1) & (trees.y <= xy[:, 1].max() + 1))
    old = cKDTree(np.c_[trees.x[nearby], trees.y[nearby]])
    return out[old.query(xy)[0] >= 0.65]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forest_dir", type=Path, help="法面の木を整理した後の trees.npy がある既存の森")
    parser.add_argument("--apply", action="store_true", help="元の一覧を控え、trees.npy と CSP の一覧を更新する")
    args = parser.parse_args()

    current = args.forest_dir / "trees.npy"
    trees = np.load(current, mmap_mode="r").view(np.recarray)
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    feats = features.load_features_rows(track.load_section_rows("features", bounds))
    with (np.load(config.DATA_PROCESSED / "track_terrain_near.npz") as near,
          np.load(config.DATA_PROCESSED / "track_terrain_mid.npz") as mid):
        ground = terrain.terrain_mesh_z(near, mid)
        added = roadside_shrubs(sec, trees, feats, bounds["s5"][0], ground)
    print(f"{current}: 低木 {len(added)} 本追加、合計 {len(trees) + len(added)} 本")
    if args.apply and len(added):
        from scripts.forest_build import old_tree_meshes

        backup = current.with_name("trees_before_fubuki_shrubs.npy")
        if not backup.exists():
            shutil.copy2(current, backup)
        combined = np.concatenate((trees, added)).view(np.recarray)
        hidden = old_tree_meshes()
        lists = forest_ac.write_lists(combined, args.forest_dir / "extension")
        forest_ac.write_ext_config(args.forest_dir, lists, hidden)
        pending = current.with_name("trees_fubuki_shrubs_pending.npy")
        np.save(pending, combined)
        pending.replace(current)


if __name__ == "__main__":
    main()
