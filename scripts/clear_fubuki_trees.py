"""吹風入口の法面の木を外し、地形を作り直した後の根元を合わせる。既定では除外本数を表示する。

以前の一覧から復元して処理するため、追加の低木はこの処理の後に plant_fubuki_shrubs で植える。
"""
import argparse
import shlex
import shutil
from pathlib import Path

import numpy as np

from yabitsu import config, features, forest_ac, fubuki, terrain, track
from yabitsu.crosssection import Section


def _roadside_shrub_positions(trees, sec, s5):
    """橋から坑口の付近で、復元によって消える低木を見分ける。"""
    rows = (sec.s >= s5 + 10607) & (sec.s <= s5 + 10684)
    x, y = trees["x"], trees["y"]
    nearby = ((x >= sec.x[rows].min() - 20) & (x <= sec.x[rows].max() + 20)
              & (y >= sec.y[rows].min() - 20) & (y <= sec.y[rows].max() + 20))
    local = trees[nearby]
    shrubs = local[np.isin(local["pattern"], ("shrub_a", "shrub_b"))]
    return set(zip(shrubs["x"], shrubs["y"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forest_dir", type=Path, help="trees.npy のある既存の森のディレクトリ")
    parser.add_argument("--apply", action="store_true", help="バックアップを作り、trees.npy と CSP の一覧を書き出す")
    args = parser.parse_args()

    current = args.forest_dir / "trees.npy"
    backup = current.with_name("trees_before_fubuki.npy")
    source = backup if backup.exists() else current
    trees = np.load(source, mmap_mode="r")
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    feats = features.load_features_rows(track.load_section_rows("features", bounds))
    s5 = bounds["s5"][0]
    keep = fubuki.keep_slope_trees(sec, trees, feats, s5)
    print(f"{source}: {len(trees) - keep.sum()} 本除外、{keep.sum()} 本残る")
    if args.apply:
        from scripts.forest_build import old_tree_meshes

        filtered = trees[keep].view(np.recarray)
        with (np.load(config.DATA_PROCESSED / "track_terrain_near.npz") as near,
              np.load(config.DATA_PROCESSED / "track_terrain_mid.npz") as mid):
            ground = terrain.terrain_mesh_z({k: near[k] for k in ("xs", "ys", "Z", "D")}, mid)
        count = fubuki.ground_retained_trees(sec, filtered, s5, ground)
        print(f"新しい地形に合わせた根元 {count} 本")
        hidden_meshes = old_tree_meshes()
        if not backup.exists():
            shutil.copy2(current, backup)
            removed = 0
        else:
            active_shrubs = _roadside_shrub_positions(np.load(current, mmap_mode="r"), sec, s5)
            removed = len(active_shrubs - _roadside_shrub_positions(filtered, sec, s5))
        lists = forest_ac.write_lists(filtered, args.forest_dir / "extension")
        forest_ac.write_ext_config(args.forest_dir, lists, hidden_meshes)
        pending = current.with_name("trees_fubuki_pending.npy")
        np.save(pending, filtered)
        pending.replace(current)
        if removed:
            print(f"橋から坑口の低木 {removed} 本が復元で消えました。次のコマンドで植え直してください。")
            print(f".venv/bin/python -m scripts.plant_fubuki_shrubs {shlex.quote(str(args.forest_dir))} --apply")


if __name__ == "__main__":
    main()
