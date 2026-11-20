"""既存のゲーム撮影と並べて、谷の色や無地の領域を検査する。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yabitsu import config
from yabitsu.valley_site_gate import (FRAME, LUMA_WEIGHTS, MAX_CHANNEL_STD, MIN_FRACTION,
                                     MIN_SURROUND_CONTRAST, centre_frame, compare_regions,
                                     comparison_sheet, flat_regions, highlight_fraction, highlight_mask)

SITES = ("A-s4-1724", "B-s4-6354", "C-s5-8956")
MODES = ("no_trees", "proxy_trees")
BLENDER_SCRIPT = ROOT / "blender_scripts/valley_site_render.py"
BRIGHTNESS_REGIONS = {
    "r70_0366_f": [("A 路面・木陰", (254, 408, 274, 428))],
    "val_A_left": [
        ("A 路面・日なた", (344, 400, 376, 432)),
        ("A 岸の土・日なた", (494, 126, 522, 154)),
    ],
    "val_A_side_down": [("A 橋の袖壁", (290, 278, 306, 294))],
    "val38_A_under_up": [
        ("A 橋台・日陰", (62, 192, 110, 240)),
        ("A 床版下面・日陰", (342, 44, 378, 80)),
    ],
    "r70_0551_f": [("B 路面・日なた", (294, 396, 326, 428))],
    "val_B_side_up": [("B 橋台", (340, 276, 360, 296))],
    "val_B_side_down": [("B 擁壁・日陰側", (594, 246, 642, 294))],
    "val38_B_rail_to_river": [
        ("B 林床・日なた", (30, 393, 44, 407)),
        ("B 林床・日陰", (264, 158, 288, 182)),
        ("B 沢底の礫・日なた", (390, 378, 426, 414)),
    ],
    "r70_0913_b": [
        ("C 路面・木陰", (238, 416, 258, 436)),
        ("C 砂利の路肩", (488, 356, 493, 361)),
    ],
    "val_C_right": [
        ("C 路面・木陰", (420, 382, 448, 410)),
        ("C 欄干・日陰", (416, 234, 424, 242)),
    ],
    "val_C_side_down": [("C 岸の土・中間", (552, 324, 600, 372))],
    "val38_C_under_up": [("C 橋台・日陰", (74, 172, 122, 220))],
    "val38_C_under_down": [
        ("C 床版下面・日陰", (378, 72, 426, 120)),
        ("C 林床・日陰", (606, 216, 634, 244)),
    ],
    "val38_C_wall_367_foot": [("C 擁壁・中間", (506, 196, 546, 236))],
}
BRIGHT_SURFACES = {
    "r70_0912_f": [("C 367 擁壁・日なた", (386, 208, 540, 236))],
}


def prepare_views(views):
    if not views or len({view["name"] for view in views}) != len(views):
        raise ValueError("site.json に有効な視点がありません")
    for view in views:
        if not re.fullmatch(r"[A-Za-z0-9_]+", view["name"]):
            raise ValueError(f"視点名が不正です ({view['name']})")
        for key in ("game", "photo"):
            if key == "photo" and not view.get(key):
                continue
            path = ROOT / view[key]
            if not path.is_file():
                raise FileNotFoundError(path)
            view[key] = path
    return views


def brightness_reference(view, game, render, pixel_ids, labels, regions=BRIGHTNESS_REGIONS,
                         *, include_blender_clipping=False):
    """Sample uncompressed 8-bit frames in fixed comparison rectangles."""
    patches = []
    for name, (x0, y0, x1, y1) in regions.get(view, ()):
        box = np.s_[y0:y1, x0:x1]
        ids, pixels = np.unique(pixel_ids[box], return_counts=True)
        game_rgb, blend_rgb = [frame[box].mean(axis=(0, 1)) for frame in (game, render)]
        delta_rgb = blend_rgb - game_rgb
        patches.append({
            "surface": name,
            "bbox": [x0, y0, x1, y1],
            "dominant_surface": labels[ids[pixels.argmax()]],
            "game_rgb": game_rgb.round(2).tolist(),
            "blender_rgb": blend_rgb.round(2).tolist(),
            "delta_rgb": delta_rgb.round(2).tolist(),
            "game_luma": round(float(game_rgb @ LUMA_WEIGHTS / 1000), 2),
            "blender_luma": round(float(blend_rgb @ LUMA_WEIGHTS / 1000), 2),
            "delta": round(float(delta_rgb @ LUMA_WEIGHTS / 1000), 2),
            "delta_r_minus_b": round(float(delta_rgb[0] - delta_rgb[2]), 2),
            "game_clipped_percent": round(float(highlight_mask(game[box]).mean() * 100), 4),
        })
        if include_blender_clipping:
            patches[-1]["blender_clipped_percent"] = round(float(highlight_mask(render[box]).mean() * 100), 4)
    return patches


def draw_gate(meta, views, blend, blender, destination):
    if not blend.is_file():
        raise FileNotFoundError(blend)
    with blend.open("rb") as source:
        blend_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    request = {
        "frame_size": FRAME,
        "modes": MODES,
        "views": [{"name": view["name"], "camera": view["camera"]} for view in views],
    }

    with TemporaryDirectory(prefix="valley-site-gate-") as tmp:
        stage = Path(tmp).resolve()
        instructions = stage / "request.json"
        instructions.write_text(json.dumps(request), encoding="utf-8")
        subprocess.run([
            str(blender), "-b", "--factory-startup", "--disable-autoexec",
            str(blend.resolve()), "--python-exit-code", "1",
            "--python", str(BLENDER_SCRIPT), "--", str(instructions), str(stage),
        ], check=True)
        sky = tuple(json.loads((stage / "sky.json").read_text(encoding="utf-8")))
        rows = []
        for view in views:
            name = view["name"]
            with Image.open(view["game"]) as source:
                game = source.copy()
            game_rgb = np.asarray(centre_frame(game))
            photo = None
            if view.get("photo"):
                with Image.open(view["photo"]) as source:
                    photo = source.copy()
            for mode in MODES:
                image_id = f"{name}__{mode}"
                with Image.open(stage / f"{image_id}.png") as source:
                    rgba = source.convert("RGBA")
                with np.load(stage / f"{image_id}.npz", allow_pickle=False) as visibility:
                    pixel_ids = visibility["pixels"]
                    labels = json.loads(visibility["labels"].item())
                render = Image.new("RGB", rgba.size, sky)
                render.paste(rgba, mask=rgba.getchannel("A"))
                rgb = np.asarray(render)
                regions = flat_regions(rgb, pixel_ids=pixel_ids, labels=labels)
                sheet = comparison_sheet(game, render, photo, mode)
                output = f"{image_id}.jpg"
                sheet.save(stage / output, quality=95, subsampling=0)
                rows.append({
                    "view": name, "mode": mode, "sheet": output,
                    "regions": [row for row in regions if row["kind"] == "editable"
                                and row["overexposed_fraction"] < .5],
                    "overexposed_regions": [row for row in regions if row["kind"] == "editable"
                                            and row["overexposed_fraction"] >= .5],
                    "background_regions": [row for row in regions if row["kind"] != "editable"],
                    "brightness_reference": brightness_reference(name, game_rgb, rgb, pixel_ids, labels),
                    "bright_surfaces": brightness_reference(name, game_rgb, rgb, pixel_ids, labels,
                                                            BRIGHT_SURFACES, include_blender_clipping=True),
                    "highlights": highlight_fraction(game_rgb, rgb),
                })

        report = {
            "frame_size": FRAME,
            "min_fraction": MIN_FRACTION,
            "max_channel_std": MAX_CHANNEL_STD,
            "min_surround_contrast_rgb": MIN_SURROUND_CONTRAST,
            "blend_sha256": blend_sha256,
            "baseline_scene": blend_sha256 == meta.get("baseline_blend_sha256"),
            "light": meta.get("light", {}),
            "views": rows,
        }
        destination.mkdir(parents=True, exist_ok=True)
        for row in rows:
            (destination / row["sheet"]).write_bytes((stage / row["sheet"]).read_bytes())
    return report


def write_report(destination, report):
    (destination / "flat_regions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def render_site(site_dir, blender, *, original_only=False):
    meta = json.loads((site_dir / "site.json").read_text(encoding="utf-8"))
    views = prepare_views(meta["views"])
    original = site_dir / "site_original.blend"
    baseline = None
    before_dir = site_dir / "gate_original"
    if original_only:
        original = original if original.is_file() else site_dir / "site.blend"
        write_report(before_dir, draw_gate(meta, views, original, blender, before_dir))
        return before_dir
    destination = site_dir / "gate"
    report = draw_gate(meta, views, site_dir / "site.blend", blender, destination)
    if report["baseline_scene"]:
        baseline = report
    elif original.is_file():
        baseline_report = before_dir / "flat_regions.json"
        if baseline_report.is_file():
            candidate = json.loads(baseline_report.read_text(encoding="utf-8"))
            if (candidate["blend_sha256"] == meta["baseline_blend_sha256"]
                    and all("bright_surfaces" in row for row in candidate["views"])):
                baseline = candidate
        if baseline is None:
            baseline = draw_gate(meta, views, original, blender, before_dir)
            write_report(before_dir, baseline)
    if baseline is not None:
        report["baseline_gate_sha256"] = baseline["blend_sha256"]
        old_views = {(before["view"], before["mode"]): before for before in baseline["views"]}
        for row in report["views"]:
            before = old_views[row["view"], row["mode"]]
            row["difference"] = {
                "editable": compare_regions(before["regions"], row["regions"]),
                "overexposed": compare_regions(before["overexposed_regions"], row["overexposed_regions"]),
                "background": compare_regions(before["background_regions"], row["background_regions"]),
            }
    write_report(destination, report)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=SITES)
    parser.add_argument("--sites-dir", type=Path, default=ROOT / "export/blender_sites")
    parser.add_argument("--blender", type=Path, default=config.BLENDER_EXE)
    parser.add_argument("--original", action="store_true", help="切り出し時の控えの門だけを描く")
    args = parser.parse_args()
    try:
        destination = render_site(args.sites_dir / args.site, args.blender, original_only=args.original)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"門の描画に失敗しました。{error}\n")
    print(destination)


if __name__ == "__main__":
    main()
