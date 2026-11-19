"""依頼41の、谷と橋の合否に使う数値条件。"""

import numpy as np


def embankment(station, right):
    return right & (station >= 8962.) & (station <= 9006.)


def fold_failures(degrees, hidden, station, right):
    limit = np.where(embankment(station, right), 60., 35.)
    return ~hidden & (degrees > limit)


def pit_failures(depth, under_pavement):
    return ~under_pavement & (depth > .001)


def shoulder_outside_height(actual, before, station):
    existing_drain_gap = (station >= 9009.) & (station <= 9010.) & (before < -.15)
    low_limit = np.where(existing_drain_gap, before - .001, -.15)
    return (~np.isfinite(actual) | (actual < low_limit) | (actual > .05)), existing_drain_gap


def local_triangles(root, bounds, select):
    import json

    from yabitsu import kn5

    x0, y0, x1, y1 = bounds
    manifest = json.loads((root / "manifest.json").read_text())
    for name, entry in manifest["meshes"].items():
        if not select(name, entry):
            continue
        with np.load(root / entry["file"]) as data:
            v = data["verts"]
            if not len(v) or v[:, 0].max() < x0 or v[:, 0].min() > x1 or v[:, 1].max() < y0 or v[:, 1].min() > y1:
                continue
            tri = v[kn5._triangles(data["faces"])]
        lo, hi = tri[..., :2].min(axis=1), tri[..., :2].max(axis=1)
        keep = (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= y0) & (lo[:, 1] <= y1)
        if keep.any():
            yield name, tri[keep]
