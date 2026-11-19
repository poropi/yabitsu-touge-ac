"""組み立て済みのNPZ、マニフェスト、木を読み取り専用で点検する。"""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import shapely

from yabitsu import config, gsi, small_bridges, terrain, track, valleys, valley_fit, water
from yabitsu.crosssection import Section


def npz(path):
    with np.load(path) as data:
        return dict(data)


def meshes(track_dir, dress_dir):
    def settings(file, default):
        path = dress_dir / file
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    hide = set(settings("hidden.json", []))
    retex = settings("retex.json", {})
    for mesh in valley_fit.manifest_meshes(track_dir):
        mesh.visible &= mesh.name not in hide
        mesh.settings["retexture"] = retex.get(mesh.name)
        yield mesh
    yield from valley_fit.manifest_meshes(dress_dir)


def point_labels(issues, sec, bounds):
    data = [asdict(i) for i in issues]
    for row in data:
        row.pop("samples")
    placed = [row for row in data if row["xy"]]
    if placed:
        xy = np.array([row["xy"] for row in placed])
        frame = terrain.road_frame(sec, *xy.T)
        for row, s, lateral in zip(placed, sec.s[frame["idx"]] + frame["along"], frame["lat"]):
            for name, (a, b) in bounds.items():
                if a <= s <= b:
                    row.update(section=name, m=round(float(s - a), 3), lateral=round(float(lateral), 3))
                    break
    return data


def summarize(issues):
    counts = Counter()
    for issue in issues:
        counts[issue.category] += issue.count
    return dict(sorted(counts.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before-near", required=True, type=Path)
    p.add_argument("--after-near", type=Path, default=config.DATA_PROCESSED / "track_terrain_near.npz")
    p.add_argument("--mid", type=Path, default=config.DATA_PROCESSED / "track_terrain_mid.npz")
    p.add_argument("--section", type=Path, default=config.DATA_PROCESSED / "track_section.json")
    p.add_argument("--track", type=Path, default=config.TRACK_BUILD_DIR)
    p.add_argument("--dress", type=Path, default=config.DATA_PROCESSED / "dress_mountain")
    p.add_argument("--before-track", "--head-track", required=True, type=Path)
    p.add_argument("--before-dress", "--head-dress", required=True, type=Path)
    p.add_argument("--before-forest", required=True, type=Path)
    p.add_argument("--forest-dir", required=True, type=Path)
    args = p.parse_args()

    before, after, mid = [npz(path) for path in (args.before_near, args.after_near, args.mid)]
    sec, bounds, rows = Section.load_json(args.section), track.load_bounds(), valleys.load()
    keys = dict.fromkeys((r.section, r.center_m) for r in rows)
    sites = {f"{s} {c:g}": valleys.site_area(sec, bounds, s, c) for s, c in keys}
    area = shapely.union_all(list(sites.values()))
    grid, stats = valley_fit.grid_checks(before, after, sites)
    before_z, after_z = [terrain.terrain_mesh_z(n, mid) for n in (before, after)]
    spans = small_bridges.spans(sec, bounds, small_bridges.load())
    _, openings = valleys.road_areas(sec, area, (None, None),
                                    [spans[k] for k in small_bridges.SUPPORT_SPANS if k in spans],
                                    -valleys.SHOULDER_M)
    channels = [(shapely.LineString(r.line), r.width_m) for r in rows if r.water]
    streams, _, _ = water.load(sec, gsi.default_sampler(gsi.TileCache(offline=True)))
    def inspect(track_root, dress_root, forest_root, near, ground):
        local = []
        for mesh in meshes(track_root, dress_root):
            low, high = mesh.verts[:, :2].min(axis=0), mesh.verts[:, :2].max(axis=0)
            if shapely.intersects(shapely.box(*low, *high), area):
                local.append(mesh)
        levels = water.valley_surface_levels(sec, bounds, rows, streams, ground, near=near)
        trees = np.load(forest_root / "trees.npy", mmap_mode="r").view(np.recarray)
        issues = valley_fit.mesh_checks(local, area, ground, before_z, openings=openings, channels=channels,
                                       ground_tri=valley_fit.terrain_triangles(near, area))
        issues += valley_fit.bridge_checks(local, sec, spans, ground)
        issues += valley_fit.wall_checks(local, sec, ground)
        issues += valley_fit.water_join_checks(local, levels, ground)
        issues += valley_fit.tree_checks(trees, area, rows, ground, before_z, local)
        issues += valley_fit.card_checks(local, area, rows, ground, before_z)
        return local, issues

    old, previous = inspect(args.before_track, args.before_dress, args.before_forest, before, before_z)
    new, current = inspect(args.track, args.dress, args.forest_dir, after, after_z)
    existing, regressions = valley_fit.issue_changes(previous, current, old_meshes=old, old_ground=before_z)
    regressions = grid + regressions
    print(json.dumps({"terrain": stats, "meshes_in_window": {"before": len(old), "after": len(new)},
                      "existing_counts": summarize(existing), "regression_counts": summarize(regressions),
                      "existing": point_labels(existing, sec, bounds),
                      "regressions": point_labels(regressions, sec, bounds)},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return bool(regressions)


if __name__ == "__main__":
    raise SystemExit(main())
