"""Image comparisons for the bridge-site render gate."""

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

FRAME = (768, 480)
MIN_FRACTION = 0.005
MAX_CHANNEL_STD = 10.0
MIN_SURROUND_CONTRAST = 32.0
COLOR_STEP = 24
LUMA_WEIGHTS = np.array([299, 587, 114])
HIGHLIGHT_LUMA = 250
HIGHLIGHT_WARNING_PP = 5


def centre_frame(image, size=FRAME):
    """Keep the centre and aspect ratio of the Blender camera frame."""
    width, height = image.size
    target_width, target_height = size
    crop_width = min(width, round(height * target_width / target_height))
    crop_height = min(height, round(width * target_height / target_width))
    x, y = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((x, y, x + crop_width, y + crop_height)).convert("RGB").resize(
        size, Image.Resampling.LANCZOS
    )


def comparison_sheet(game, render, photo, mode):
    """Place the supplied photographs and render at the same panel size."""
    panels = [("Game", game), (f"Blender / {mode.replace('_', ' ')}", render)]
    if photo is not None:
        panels.insert(0, ("Photo", photo))
    width, height = render.size
    header = 23
    sheet = Image.new("RGB", (width * len(panels), height + header), (27, 29, 30))
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(panels):
        x = index * width
        sheet.paste(centre_frame(image, (width, height)), (x, header))
        draw.text((x + 8, 5), label, fill=(255, 255, 255))
    return sheet


def _colour_groups(rgb, surface):
    # Overlapping colour bins avoid splitting a patch at a single arbitrary bin edge.
    smooth = ndimage.gaussian_filter(rgb.astype(np.float32), (1, 1, 0))
    for offset in (0, COLOR_STEP / 2):
        bins = ((smooth + offset) / COLOR_STEP).astype(np.int16)
        groups = (bins @ np.array([1, 16, 256], dtype=np.int16)) + 1
        yield np.where(surface, groups, 0)


def _overlaps(left, right):
    a, b = left["bbox"], right["bbox"]
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height >= 0.7 * smaller and np.linalg.norm(
        np.subtract(left["mean_rgb"], right["mean_rgb"])
    ) < COLOR_STEP


def highlight_mask(rgb):
    return rgb @ LUMA_WEIGHTS >= HIGHLIGHT_LUMA * 1000


def highlight_fraction(game, render):
    game_percent, blender_percent = [float(highlight_mask(frame).mean() * 100) for frame in (game, render)]
    difference = blender_percent - game_percent
    return {
        "game_percent": round(game_percent, 4),
        "blender_percent": round(blender_percent, 4),
        "difference_pp": round(difference, 4),
        "warning": difference + 1e-10 >= HIGHLIGHT_WARNING_PP,
    }


def _surface_record(ids, labels):
    keys, counts = np.unique(ids, return_counts=True)
    rows = [{**labels[key], "pixels": int(count)} for key, count in zip(keys, counts)]
    for row in rows:
        collections = row.get("collections", [row.get("collection", "")])
        row["kind"] = ("editable" if any(c.startswith("EDIT_") for c in collections) else
                       "reference" if any(c.startswith("REF_") for c in collections) else
                       "context" if any(c.startswith("LOCK_") for c in collections) else
                       "unassigned" if row["object"] else "sky")
    kinds = {row["kind"] for row in rows}
    kind = ("editable" if "editable" in kinds else
            next(iter(kinds)) if len(kinds) == 1 else "background")
    return kind, sorted(rows, key=lambda row: (-row["pixels"], row["object"] or "", row["material"] or ""))


def compare_regions(before, after):
    """Pair by location and part, so a remapped material can stay in the same region."""
    def position_key(old, new):
        a, b = old["bbox"], new["bbox"]
        width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
        overlap = width * height / (area - width * height)
        old_parts = {item["object"] for item in old.get("surfaces", ())}
        new_parts = {item["object"] for item in new.get("surfaces", ())}
        if old_parts != new_parts and not old_parts & new_parts:
            return 0
        return overlap

    matches = sorted(((position_key(old, new), i, j)
                      for i, old in enumerate(before) for j, new in enumerate(after)),
                     reverse=True)
    used_before, used_after, retained = set(), set(), []
    for overlap, i, j in matches:
        if overlap < .35:
            break
        if i not in used_before and j not in used_after:
            retained.append({"before": before[i], "after": after[j]})
            used_before.add(i)
            used_after.add(j)
    return {"added": [row for j, row in enumerate(after) if j not in used_after],
            "removed": [row for i, row in enumerate(before) if i not in used_before],
            "retained": retained}


def flat_regions(rgb, surface=None, *, pixel_ids=None, labels=None):
    """Find large, flat colour patches distinct from their visible surroundings.

    Coordinates are measured in the gate frame. Contrast is the median RGB
    distance from the patch colour in a narrow ring outside the patch.
    """
    height, width, _ = rgb.shape
    if surface is None:
        surface = np.ones((height, width), dtype=bool)
    minimum = math.ceil(MIN_FRACTION * width * height)
    ring_radius = max(5, round(min(width, height) * 0.018))
    regions = []
    for groups in _colour_groups(rgb, surface):
        group_counts = np.bincount(groups.ravel())
        group_bounds = ndimage.find_objects(groups)
        for code in np.flatnonzero(group_counts[1:] >= minimum) + 1:
            bounds = group_bounds[code - 1]
            components, _ = ndimage.label(groups[bounds] == code)
            counts = np.bincount(components.ravel())
            for label, patch in enumerate(ndimage.find_objects(components), 1):
                count = int(counts[label])
                if count < minimum:
                    continue
                ys, xs = patch
                x0, y0 = bounds[1].start + xs.start, bounds[0].start + ys.start
                x1, y1 = bounds[1].start + xs.stop, bounds[0].start + ys.stop
                mask = components[patch] == label
                colors = rgb[y0:y1, x0:x1][mask]
                mean = colors.mean(axis=0)
                std = colors.std(axis=0).max()
                if std > MAX_CHANNEL_STD:
                    continue

                xlo, ylo = max(0, x0 - ring_radius), max(0, y0 - ring_radius)
                xhi, yhi = min(width, x1 + ring_radius), min(height, y1 + ring_radius)
                padded = np.zeros((yhi - ylo, xhi - xlo), dtype=bool)
                padded[y0 - ylo:y1 - ylo, x0 - xlo:x1 - xlo] = mask
                filled = ndimage.binary_fill_holes(padded)
                inner = ndimage.binary_dilation(filled, iterations=2)
                ring = ndimage.binary_dilation(filled, iterations=ring_radius) & ~inner
                nearby = rgb[ylo:yhi, xlo:xhi][ring & surface[ylo:yhi, xlo:xhi]]
                if len(nearby) == 0:
                    continue
                contrast = np.median(np.linalg.norm(nearby - mean, axis=1))
                if contrast >= MIN_SURROUND_CONTRAST:
                    region = {
                        "bbox": [x0, y0, x1, y1],
                        "pixels": count,
                        "fraction": round(count / (width * height), 6),
                        "mean_rgb": mean.round(2).tolist(),
                        "max_channel_std": round(float(std), 2),
                        "surround_contrast_rgb": round(float(contrast), 2),
                        "overexposed_fraction": round(float(highlight_mask(colors).mean()), 6),
                    }
                    if pixel_ids is not None:
                        region["kind"], region["surfaces"] = _surface_record(
                            pixel_ids[y0:y1, x0:x1][mask], labels)
                    regions.append(region)

    unique = []
    for region in sorted(regions, key=lambda r: (-r["pixels"], r["bbox"])):
        if not any(_overlaps(region, previous) for previous in unique):
            unique.append(region)
    return unique
