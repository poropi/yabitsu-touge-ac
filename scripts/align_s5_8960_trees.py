"""S5 8960 m 付近で、依頼25の地形変更にかかった木だけを補正する。

吹風も再適用する巡では、clear_fubuki_trees、plant_fubuki_shrubs、本コマンドの順に実行する。
clear_fubuki_trees は旧版の trees.npy から復元する。plant_fubuki_shrubs 単独は現行の木を引き継ぐ。
"""
import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from shapely import distance, points, union_all
from shapely.geometry import Polygon

from yabitsu import config, csptree, features, terrain, track
from yabitsu.crosssection import Section

HEIGHT_THRESHOLD = 0.01
BEFORE_DIR = config.PROJECT_ROOT / "export/compare/route70/work/before_25"
NEAR_FILE = "track_terrain_near.npz"
BACKUP_SUFFIX = "_before_s5_8960"


def load_ground_pair(before_dir, after_path):
    """変更点の控えを新旧の全格子で照合し、触れる三角形を全て含む長方形を返す。"""
    with (np.load(before_dir / NEAR_FILE) as before, np.load(after_path) as after):
        xs, ys = before["xs"], before["ys"]
        if not (np.array_equal(xs, after["xs"]) and np.array_equal(ys, after["ys"])):
            raise ValueError("新旧の近地形の格子軸が異なります")
        old_z, new_z = before["Z"], after["Z"]
        equal = (old_z == new_z) | (np.isnan(old_z) & np.isnan(new_z))
        changed = np.argwhere(~equal)
        if not np.array_equal(changed, np.load(before_dir / "changed_idx.npy")):
            raise ValueError("新旧の近地形の変更点が changed_idx.npy と一致しません")
        r0, c0 = np.maximum(changed.min(0) - 1, 0)
        r1, c1 = np.minimum(changed.max(0) + 2, old_z.shape)
        cells = (slice(r0, r1), slice(c0, c1))
        faces = (slice(r0, r1 - 1), slice(c0, c1 - 1))
        axes = {"xs": xs[c0:c1], "ys": ys[r0:r1]}
        grids = tuple({**axes, "Z": z[cells].copy(), "D": data["D"][cells].copy(),
                       "H": data["H"][faces].copy()} for z, data in ((old_z, before), (new_z, after)))
        return grids


def ground_changes(old, new, x, y):
    """変更頂点を持つ面上の行番号と、既存の三角形補間で求めた新旧の高さを返す。"""
    xs, ys = old["xs"], old["ys"]
    ids = np.flatnonzero((xs[0] <= x) & (x <= xs[-1]) & (ys[0] <= y) & (y <= ys[-1]))
    x, y = x[ids], y[ids]
    c = np.minimum(np.searchsorted(xs, x, side="right") - 1, len(xs) - 2)
    r = np.minimum(np.searchsorted(ys, y, side="right") - 1, len(ys) - 2)
    fx = (x - xs[c]) / (xs[c + 1] - xs[c])
    fy = (y - ys[r]) / (ys[r + 1] - ys[r])
    changed = old["Z"] != new["Z"]
    hit = (changed[r, c] | changed[r + 1, c + 1]
           | ((fx >= fy) & changed[r, c + 1]) | ((fy >= fx) & changed[r + 1, c]))
    r, c, x, y = r[hit], c[hit], x[hit], y[hit]
    heights = []
    for grid in (old, new):
        corners = (grid["D"][r, c], grid["D"][r, c + 1],
                   grid["D"][r + 1, c], grid["D"][r + 1, c + 1])
        if np.any(grid["H"][r, c]) or np.any(np.asarray(corners) > terrain.NEAR_BAND - 2):
            raise ValueError("変更面が近地形の描画範囲または補間範囲を外れています")
        # この範囲では中地形にフォールバックしない。
        z = terrain.terrain_mesh_z(grid, grid)(x, y)
        if not np.all(np.isfinite(z)):
            raise ValueError("変更面の根元高さを補間できません")
        heights.append(z)
    return ids[hit], *heights


@dataclass(frozen=True)
class Change:
    index: int
    s5_m: float
    lateral: float
    old_ground: float
    new_ground: float
    reason: str


def planned_changes(trees, old, new, sec, feats, s5, clearance):
    ids, before, after = ground_changes(old, new, trees.x, trees.y)
    selected = np.abs(after - before) > HEIGHT_THRESHOLD
    ids, before, after = ids[selected], before[selected], after[selected]
    if not len(ids):
        return []

    wall, = [f for f in feats if f.type == "retaining_wall" and f.side == "L"
             and np.isclose(f.s_from - s5, 8961, rtol=0, atol=1e-6)
             and np.isclose(f.s_to - s5, 8980, rtol=0, atol=1e-6)]
    left, right = features.sidewalk_offsets(sec, feats)
    sl = features.rows_of(sec, wall)
    vertices, faces, _ = features.retaining_wall_mesh(sec, wall, left)
    wall_area = union_all([Polygon(vertices[face, :2]) for face in faces])
    crest = vertices[1::2, :2]
    center = np.c_[sec.x[sl], sec.y[sl]]
    behind_road = Polygon(np.concatenate((center, crest[::-1])))

    x, y = trees.x[ids], trees.y[ids]
    frame = terrain.road_frame(sec, x, y)
    s = sec.s[frame["idx"]] + frame["along"]
    lat = frame["lat"]
    p = points(x, y)
    by_wall = distance(wall_area, p) <= clearance["wall"]
    between = behind_road.covers(p)
    # 造林では 1ROAD の側溝も道路に含む。幅が変わる区間は断面間で補間する。
    road_edge = np.interp(s, sec.s, sec.hw_right + features.GUTTER_WIDTH + right)
    by_road = (lat < 0) & (-lat - road_edge <= clearance["road"])
    reasons = np.select([by_wall, between, by_road], ["左擁壁", "左擁壁と道路の間", "右道路際"], default="")
    return [Change(int(i), float(d - s5), float(l), float(a), float(b), str(reason))
            for i, d, l, a, b, reason in zip(ids, s, lat, before, after, reasons)]


def _line(tree):
    return csptree.tree_line(f"{tree.pattern}.zip", tree.x, tree.z, -tree.y,
                            tree.angle_deg, tree.scale).encode("utf-8")


def _patch_tile(path, rows, replacements, removed):
    """対象タイルを現行の NPY と照合し、対象の行だけをバイト単位で置き換える。"""
    lines = path.read_bytes().splitlines(keepends=True)
    positions = [i for i, line in enumerate(lines) if line.startswith(b"tree: ")]
    original = [_line(row) for row in rows]
    updated = [replacements.get(i, line) for i, line in enumerate(original) if i not in removed]
    actual = [lines[i].rstrip(b"\r\n") for i in positions]
    if actual == updated:
        return None  # 前回、CSP の一覧を更新した直後に中断した場合も再開できる。
    if actual != original:
        raise ValueError(f"{path} の木の行と trees.npy が一致しません")
    for i, value in replacements.items():
        j = positions[i]
        lines[j] = value + lines[j][len(original[i]):]
    for i in removed:
        lines[positions[i]] = b""
    return b"".join(lines)


def update_forest(forest_dir, old, new, sec, feats, s5, clearance, *, apply=False):
    # Windows でも置換できるよう、保存先のファイルを mmap で開いたままにしない。
    trees = np.load(forest_dir / "trees.npy").view(np.recarray)
    plan = planned_changes(trees, old, new, sec, feats, s5, clearance)
    paths = apply_forest_changes(forest_dir, trees, plan, BACKUP_SUFFIX) if apply else []
    return plan, paths


def apply_forest_changes(forest_dir, trees, plan, backup_suffix):
    """全対象タイルを照合し、木の高さ変更と除外だけを一覧へ書く。"""
    current = forest_dir / "trees.npy"
    removed = {c.index for c in plan if c.reason}
    heights = {c.index: c.new_ground for c in plan if not c.reason and trees.z[c.index] != c.new_ground}
    edits = sorted(removed | heights.keys())
    if not edits:
        return []

    keep = np.ones(len(trees), bool)
    keep[list(removed)] = False
    tx = np.floor(trees.x / 1000).astype(int)
    ty = np.floor(trees.y / 1000).astype(int)
    tiles = sorted(set(zip(tx[edits], ty[edits])))
    output = {}
    for i, j in tiles:
        ids = np.flatnonzero((tx == i) & (ty == j))
        rows = trees[ids]
        updated = rows.copy()
        replacements = {}
        for pos, index in enumerate(ids):
            if index in heights:
                updated.z[pos] = heights[index]
                replacements[pos] = _line(updated[pos])
        drops = {pos for pos, index in enumerate(ids) if index in removed}
        path = forest_dir / "extension" / "trees" / f"forest_{i}_{j}.txt"
        patch = _patch_tile(path, rows, replacements, drops)
        if patch is not None:
            output[path] = patch

    # 先に全タイルの整合を検査し、控えと一時ファイルを揃えてから置換する。
    for path in [current, *output]:
        backup = path.with_name(path.stem + backup_suffix + path.suffix)
        if not backup.exists():
            shutil.copy2(path, backup)
    with TemporaryDirectory(dir=forest_dir) as tmp:
        pending = Path(tmp)
        for path, content in output.items():
            (pending / path.name).write_bytes(content)
        for i, z in heights.items():
            trees.z[i] = z
        np.save(pending / current.name, trees[keep])
        for path in output:
            (pending / path.name).replace(path)
        (pending / current.name).replace(current)
    return sorted(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("forest_dir", type=Path, help="trees.npy と extension/trees がある森のディレクトリ")
    parser.add_argument("--before-dir", type=Path, default=BEFORE_DIR, help="旧地形と changed_idx.npy の控え")
    parser.add_argument("--apply", action="store_true", help="差がある場合に一覧を更新する")
    args = parser.parse_args()

    old, new = load_ground_pair(args.before_dir, config.DATA_PROCESSED / NEAR_FILE)
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    feats = features.load_features_rows(track.load_section_rows("features", bounds))
    clearance = json.loads((config.DATA_MANUAL / "forest.json").read_text(encoding="utf-8"))["clearance"]
    plan, paths = update_forest(args.forest_dir, old, new, sec, feats, bounds["s5"][0], clearance, apply=args.apply)
    print(f"高さ差 > 1 cm: {len(plan)} 本（除外 {sum(bool(c.reason) for c in plan)} 本）")
    for c in plan:
        side = "左" if c.lateral >= 0 else "右"
        print(f"行 {c.index} | S5 {c.s5_m:.3f} m | {side} {abs(c.lateral):.3f} m | "
              f"{c.old_ground:.4f} → {c.new_ground:.4f} m | {c.reason or '根元補正'}")
    for path in paths:
        print(path)
    if args.apply:
        print("吹風も再適用する場合は clear_fubuki_trees → plant_fubuki_shrubs → 本コマンド --apply の順です。")


if __name__ == "__main__":
    main()
