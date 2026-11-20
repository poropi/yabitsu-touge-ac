"""Render the bridge A roadside pilot and count visible ground in the comparison frames."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yabitsu.geo import to_local


ROUTE = ROOT / "export/compare/route70"
INVENTORY = ROOT / "export/compare/route70_inventory_20261030"
WORK = ROUTE / "work/forest_56"
SITE = "A-s4-1724"
SIZE = (768, 480)
MODES = ("before", "after")
FONT_PATH = Path("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc")


def model_pose(pattern, angle):
    """Map the CSP instance and any baked rotation back to the Blender collection."""
    model = pattern
    baked = re.fullmatch(r"(mt_(?:sugi_edge_[ab]|roadside_[ab]))_r(\d\d)", model)
    if baked:
        model, index = baked.groups()
        angle += int(index) * 22.5
    return model, angle % 360


def foreground_roadside_mask(aov, road_xy, half_widths):
    """Label visible terrain left (+1) or right (-1) within 0–12 m of pavement."""
    coverage = aov[..., 2]
    ground = coverage > .5
    mask = np.zeros(coverage.shape, dtype=np.int8)
    if not ground.any():
        return mask
    points = aov[ground, :2] / coverage[ground, None]
    segment = np.diff(road_xy, axis=0)
    length_squared = np.einsum("ij,ij->i", segment, segment)
    midpoints = road_xy[:-1] + .5 * segment
    max_half_step = .5 * np.sqrt(length_squared.max())
    tree = cKDTree(midpoints)
    # Any segment closer than the pavement edge plus 12 m has its midpoint
    # within this search radius. The road points are sampled about 1 m apart.
    search = half_widths.max() + 12 + max_half_step
    distance, _ = tree.query(points)
    candidate_pixels = np.flatnonzero(distance <= search)
    measured = np.zeros(len(points), dtype=np.int8)
    if len(candidate_pixels):
        candidate_points = points[candidate_pixels]
        # A nearest midpoint is an upper bound on the nearest segment distance.
        # Include every midpoint whose segment could improve that bound.
        close = tree.query_ball_point(candidate_points, distance[candidate_pixels] + max_half_step + 1e-7,
                                      return_sorted=True)
        counts = np.fromiter(map(len, close), dtype=np.int64, count=len(close))
        ids = np.concatenate(close).astype(np.int64, copy=False)
        offsets = np.repeat(candidate_points, counts, axis=0) - road_xy[ids]
        vectors = segment[ids]
        along = np.clip(np.einsum("ij,ij->i", offsets, vectors) / length_squared[ids], 0, 1)
        delta = offsets - along[:, None] * vectors
        distance2 = np.einsum("ij,ij->i", delta, delta)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        minimum = np.repeat(np.minimum.reduceat(distance2, starts), counts)
        indices = np.where(distance2 == minimum, np.arange(len(ids)), len(ids))
        take = np.minimum.reduceat(indices, starts)
        steps = ids[take]
        left = vectors[take, 0] * delta[take, 1] - vectors[take, 1] * delta[take, 0] > 0
        side = (~left).astype(int)
        widths = ((1 - along[take]) * half_widths[steps, side] + along[take] * half_widths[steps + 1, side])
        from_edge = np.sqrt(distance2[take]) - widths
        within = (from_edge >= -1e-6) & (from_edge <= 12 + 1e-6)
        measured[candidate_pixels] = np.where(within, np.where(left, 1, -1), 0)
    mask[ground] = measured
    return mask


def prepare(site, forest, temporary, mode, view_filter):
    processed = ROOT / "data/processed"
    section = {key: np.asarray(value) for key, value in
               json.loads((processed / "track_section.json").read_text()).items()}
    with np.load(processed / "track_profile.npz", allow_pickle=True) as profile:
        start = profile["bounds"].item()["s4"][0]
    origin = np.asarray(site["origin"])

    def sample(station, name):
        return np.interp(station + start, section["s"], section[name])

    def center(station):
        return np.array([sample(station, name) for name in ("x", "y", "z")]) - origin

    views = []
    captures = {row["id"]: row for row in map(json.loads, (INVENTORY / "captures.jsonl").read_text().splitlines())}
    photos = {view["name"]: view for view in site["views"] if view["photo"]}
    for number, facing in (("0366", "f"), ("0366", "b"), ("0365", "f")):
        row = captures[f"R70-{number}"]
        station = row["section_s_m"]
        key = f"r70_{number}_{facing}"
        heading, pitch = np.deg2rad([row[f"heading_{facing}"], row["pitch_deg"]])
        direction = [np.sin(heading) * np.cos(pitch), np.cos(heading) * np.cos(pitch), np.sin(pitch)]
        if key in photos:
            location = np.asarray(photos[key]["position"]) - origin
            fixed_z = True
        else:
            location = center(station)
            location[:2] = to_local(row["lat"], row["lon"]) - origin[:2]
            fixed_z = False
        views.append(dict(name=key, station=station, center=location.tolist(), direction=direction,
                          h_fov=row["fov_deg"], fixed_z=fixed_z, height=None, photo=True))
    for station in range(1660, 1761, 20):
        for name, sign in (("up", -1), ("down", 1)):
            location = center(station)
            direction = (center(station + 2) - center(station - 2)) * sign
            heading = direction[:2] / np.linalg.norm(direction[:2])
            lane = sample(station, "hw_left" if sign > 0 else "hw_right") / 2
            location[:2] += np.array([-heading[1], heading[0]]) * lane
            views.append(dict(name=f"s4_{station}_{name}", station=station, center=location.tolist(),
                              direction=direction.tolist(), h_fov=56, fixed_z=False, height=1.2, photo=False))
    if view_filter:
        views = [view for view in views if view["name"] in view_filter]
        if {view["name"] for view in views} != set(view_filter):
            raise ValueError(f"視点が見つかりません: {view_filter}")
    reference = (np.asarray(photos["r70_0366_f"]["position"]) - origin).tolist()
    setup = dict(views=views, photo_height_reference=reference, lights=site["light"], size=SIZE)
    (temporary / "setup.json").write_text(json.dumps(setup), encoding="utf-8")
    rows = np.load(forest, mmap_mode="r", allow_pickle=False)
    inside = np.hypot(rows["x"] - origin[0], rows["y"] - origin[1]) <= site["radius"]
    rows = rows[inside].copy()
    for name, delta in zip(("x", "y", "z"), origin):
        rows[name] -= delta
    model_names = np.array([model_pose(str(pattern), angle) for pattern, angle in
                            zip(rows["pattern"], rows["angle_deg"])],
                           dtype=[("model", "U32"), ("angle", "f8")])
    np.savez(temporary / f"{mode}.npz", rows=rows, poses=model_names)

    points = np.column_stack([section[k] for k in ("x", "y")]) - origin[:2]
    close = np.hypot(*points.T) <= site["radius"] + 50
    indexes = np.flatnonzero(close)
    area = slice(indexes.min(), indexes.max() + 1)
    half = np.column_stack([section[k] for k in ("hw_left", "hw_right")])[area]
    return views, points[area], half


def count_and_save_masks(out, mode, views, road, widths):
    results = []
    for view in views:
        prefix = f"{view['name']}__{mode}"
        aov = np.load(out / f"{prefix}.npy")
        mask = foreground_roadside_mask(aov, road, widths)
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(out / f"{prefix}_ground.png")
        results.append(dict(view=view["name"], mode=mode, left=int(np.count_nonzero(mask > 0)),
                            right=int(np.count_nonzero(mask < 0)), total=int(np.count_nonzero(mask)),
                            frame=int(mask.size)))
        (out / f"{prefix}.npy").unlink()
    (out / f"ground_pixels_{mode}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def combine(out, views):
    counts = {}
    for mode in MODES:
        rows = json.loads((out / f"ground_pixels_{mode}.json").read_text())
        counts[mode] = {row["view"]: row for row in rows}
    all_results = []
    font = ImageFont.truetype(str(FONT_PATH), 14)
    for view in views:
        key = view["name"]
        before, after = (counts[mode][key] for mode in MODES)
        frames = [Image.open(out / f"{key}__{mode}.png").convert("RGB") for mode in MODES]
        pair = Image.new("RGB", (SIZE[0] * 2, SIZE[1] + 26), "#24272c")
        draw = ImageDraw.Draw(pair)
        for index, (frame, column, label) in enumerate(zip(frames, (before, after), ("今", "適用後"))):
            x = index * SIZE[0]
            pair.paste(frame, (x, 26))
            draw.text((x + 9, 4), f"{key}　{label}　見えている地形 {column['total']:,} 画素", fill="white", font=font)
        pair.save(out / f"{key}__compare.jpg", quality=94)
        all_results.append(dict(view=key, station=view["station"], h_fov=view["h_fov"],
                                before=before["total"], after=after["total"], delta=after["total"] - before["total"],
                                before_left=before["left"], before_right=before["right"],
                                after_left=after["left"], after_right=after["right"], frame=before["frame"]))
    (out / "ground_pixels.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    photo_views = [view for view in views if view["photo"]]
    if photo_views:
        sheet = Image.new("RGB", (SIZE[0] * 3, (SIZE[1] + 26) * len(photo_views)), "#24272c")
        draw = ImageDraw.Draw(sheet)
        for i, view in enumerate(photo_views):
            key = view["name"]
            photo = INVENTORY / "images/sv" / f"{key.replace('r70_', 'R70-', 1)}.jpg"
            y = i * (SIZE[1] + 26)
            sheet.paste(Image.open(photo).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS), (0, y + 26))
            sheet.paste(Image.open(out / f"{key}__compare.jpg").convert("RGB"), (SIZE[0], y))
            draw.text((9, y + 4), f"{key}　現地写真", fill="white", font=font)
        sheet.save(out / "photo_views.jpg", quality=93)
    drive = [view for view in views if not view["photo"]]
    if drive:
        pair_size = (SIZE[0], (SIZE[1] + 26) // 2)
        sheet = Image.new("RGB", (2 * pair_size[0], ((len(drive) + 1) // 2) * pair_size[1]), "#24272c")
        for i, view in enumerate(drive):
            pair = Image.open(out / f"{view['name']}__compare.jpg")
            sheet.paste(pair.resize(pair_size, Image.Resampling.LANCZOS), ((i % 2) * pair_size[0], (i // 2) * pair_size[1]))
        sheet.save(out / "drive_views.jpg", quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="橋 A を 160 m に切り出した一時ディレクトリ")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--out", type=Path, default=WORK / "views")
    parser.add_argument("--forest", type=Path)
    parser.add_argument("--view", action="append")
    opts = parser.parse_args()
    out = opts.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = opts.scene.resolve()
    site = json.loads((scene / "site.json").read_text())
    if site["site"] != SITE or site["radius"] != 160:
        parser.error("橋 A の半径 160 m の切り出しを指定してください")
    source = opts.forest or (ROUTE / "work/trees_54/forest/trees.npy" if opts.mode == "before" else WORK / "trees.npy")
    if not source.is_file():
        parser.error(f"trees.npy がありません: {source}")
    views, road, widths = prepare(site, source, scene, opts.mode, opts.view)
    with (out / f"render_{opts.mode}.log").open("w") as log:
        subprocess.run(["/opt/homebrew/bin/blender", "-b", "--factory-startup", str(scene / "site_original.blend"),
                        "--python-exit-code", "1", "--python", str(ROOT / "blender_scripts/render_roadside_pilot.py"),
                        "--", str(scene), str(out), opts.mode], stdout=log, stderr=subprocess.STDOUT, check=True)
    count_and_save_masks(out, opts.mode, views, road, widths)
    if opts.mode == "after" and (out / "ground_pixels_before.json").exists():
        combine(out, views)


if __name__ == "__main__":
    main()
