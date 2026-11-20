"""局所の谷の置換面。2 m 格子を変更せず、同一の三角形を描画と接地に渡す。"""

import numpy as np
import shapely
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from . import kn5, terrain, valleys


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def sample_grid(grid, x, y):
    xs, ys = grid["xs"], grid["ys"]
    c = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    r = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
    fx, fy = (x - xs[c]) / (xs[c + 1] - xs[c]), (y - ys[r]) / (ys[r + 1] - ys[r])
    z = grid["Z"]
    a, b, d, e = z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]
    return a + np.where(fx >= fy, fx * (b - a) + fy * (e - b), fy * (d - a) + fx * (e - d))


def allowed_area(grid, rows, site, protected, repair):
    strip = shapely.union_all([shapely.LineString(v.line).buffer(v.reach_m) for v in rows])
    # The B survey can extend beyond the designed banks.
    if repair is not None:
        line = shapely.LineString(repair.row.line)
        a, b = repair.start, repair.end
        from shapely.ops import substring

        bank = substring(line, a, b).buffer(repair.far + 3)
        bx, by = np.meshgrid(
            grid["xs"][(grid["xs"] >= bank.bounds[0] - 2) & (grid["xs"] <= bank.bounds[2] + 2)],
            grid["ys"][(grid["ys"] >= bank.bounds[1] - 2) & (grid["ys"] <= bank.bounds[3] + 2)],
        )
        hit = repair.contains(bx, by)
        strip = strip.union(
            shapely.union_all(shapely.box(bx[hit] - 2, by[hit] - 2, bx[hit] + 2, by[hit] + 2))
        )
    return strip.intersection(site).difference(protected)


def site_surface(sec, grid, rows, site, span, protected, repair):
    area = allowed_area(grid, rows, site, protected, repair)
    xs, ys = grid["xs"], grid["ys"]
    x0, y0, x1, y1 = area.bounds
    c0, c1 = np.clip(np.searchsorted(xs, [x0, x1]) + [-3, 3], 0, len(xs) - 1)
    r0, r1 = np.clip(np.searchsorted(ys, [y0, y1]) + [-3, 3], 0, len(ys) - 1)
    cell = xs[1] - xs[0]
    step = cell / 4
    x = xs[c0] + np.arange((c1 - c0) * 4 + 1) * step
    y = ys[r0] + np.arange((r1 - r0) * 4 + 1) * step
    X, Y = np.meshgrid(x, y)
    base = sample_grid(grid, X, Y)
    at = shapely.contains_xy(area, X, Y) & np.isfinite(base)
    px, py = X[at], Y[at]
    design = base.copy()
    cap = np.full(len(px), np.inf)
    bottom = np.zeros_like(base, bool)
    bed = np.zeros(len(px), bool)
    for row in rows:
        _, u, side, arc = row.project(px, py)
        floor = np.interp(u, arc, np.broadcast_to(row.bed["z_m"], arc.shape))
        dist = abs(side)
        # A rounded toe grows into a straight bank over a metre on either side.
        half_width = row.width_m / 2
        if repair is not None and row.water:
            # The narrower low-flow bed rejoins the existing 6 m bed beyond the survey.
            half_width = 3.0 + (half_width - 3.0) * repair.low_flow_blend(px, py)
        toe = dist - half_width
        rounded = np.where(toe >= 1, toe, 0.25 * np.maximum(0, toe + 1) ** 2)
        slope = np.where(side >= 0, row.bank_left, row.bank_right)
        slope = slope * (1 + 0.055 * np.sin(u / 7 + np.sign(side)))
        channel = floor + slope * rounded
        if row.id.startswith("A"):
            # Low-flow channel within the wider gravel bed.
            channel += 0.15 * smooth(dist / 0.85)
        w = smooth((row.reach_m - dist) / row.bed.get("taper_m", 5))
        ends = np.minimum(u, arc[-1] - u)
        taper = row.bed.get("end_taper_m", row.bed.get("taper_m", 5))
        if taper:
            w *= smooth(ends / taper)
        else:
            w *= ends > 0
        cap = np.minimum(cap, base[at] + w * np.minimum(channel - base[at], 0))
        bed |= (dist <= row.width_m * 0.64) & (ends > 0)
    if repair is not None:
        cap = repair.shape_bed(rows, cap, base, at, px, py, step)
    # The grade under approaches supports the gutters. Across the bridge it opens fully.
    frame = terrain.road_frame(sec, px, py)
    i, lat = frame["idx"], frame["lat"]
    s = sec.s[i] + frame["along"]
    outside_span = np.maximum(sec.s[span[0]] - s, s - sec.s[span[1]])
    width = np.where(lat > 0, sec.hw_left[i], sec.hw_right[i])
    weight = 1 - smooth((outside_span + 0.6) / 3.4) * (1 - smooth((abs(lat) - width + 0.15) / 2.2))
    design[at] = base[at] + weight * (cap - base[at])
    bottom[at] = bed
    # Select complete coarse cells with local relief, including the transition to
    # unchanged ground. Road, tunnel, unrelated pads and the outer reach stay fixed.
    oy, ox = np.indices((r1 - r0, c1 - c0))
    corners = np.stack((xs[c0 + ox], ys[r0 + oy], xs[c0 + ox + 1], ys[r0 + oy + 1]), axis=-1)
    inside = shapely.covers(area, shapely.box(*np.moveaxis(corners, -1, 0)))
    dist = grid["D"][r0 : r1 + 1, c0 : c1 + 1]
    inside &= ~grid["H"][r0:r1, c0:c1]
    inside &= (
        np.maximum.reduce([dist[:-1, :-1], dist[1:, :-1], dist[:-1, 1:], dist[1:, 1:]]) <= terrain.NEAR_BAND
    )
    changed = (abs(base - design) > 0.008) | (bottom & at)
    candidates = changed[:-1, :-1].reshape(r1 - r0, 4, c1 - c0, 4).any(axis=(1, 3))
    mask = binary_dilation(candidates, iterations=2) & inside
    cells = np.argwhere(mask) + [r0, c0]
    rc, faces = valleys.fine_topology(cells)
    rr, cc = (rc - [r0 * 4, c0 * 4]).T
    dense = np.zeros_like(base, bool)
    # Mark interior fine vertices, including shared edges, but never the outer boundary.
    tile = mask.repeat(4, 0).repeat(4, 1)
    dense[1:-1, 1:-1] = tile[:-1, :-1] & tile[:-1, 1:] & tile[1:, :-1] & tile[1:, 1:]
    edge_distance = distance_transform_edt(dense) * step
    # Filter only valid samples so that a missing DEM tile cannot generate a NaN seam.
    finite = np.isfinite(design)
    filtered = gaussian_filter(np.where(finite, design, 0), 1.6) / np.maximum(
        1e-9, gaussian_filter(finite.astype(float), 1.6)
    )
    z = base[rr, cc] + smooth(edge_distance[rr, cc] / 2.0) * (filtered[rr, cc] - base[rr, cc])
    verts = np.c_[x[cc], y[rr], z]
    verts[:, 2] = relax_banks(verts, faces, edge_distance[rr, cc] == 0, np.where(bottom[rr, cc], 0.9, 0.08))
    if repair is not None:
        reference, missing = repair.reference(*verts[:, :2].T)
        interior = missing & (edge_distance[rr, cc] > 0)
        verts[interior, 2] = np.clip(
            verts[interior, 2], reference[interior] - 0.97, reference[interior] + 0.97
        )
    # Preserve the sampled thalweg while softening exposed bank folds.
    centre = np.logical_or.reduce([abs(row.project(*verts[:, :2].T)[2]) < 1.6 for row in rows])
    fixed = (edge_distance[rr, cc] == 0) | centre
    verts[:, 2] = limit_folds(verts, faces, fixed)
    verts[:, 2] = blend_outer_slopes(sec, grid, verts, faces, fixed)
    if repair is not None:
        from .valley_drainage import downhill_grades

        verts[:, 2] = downhill_grades(verts, faces, edge_distance[rr, cc] == 0, rows, repair.reference)
    return cells, verts, faces, bottom[rr, cc][faces].sum(axis=1) >= 2


def blend_outer_slopes(sec, grid, vertices, faces, fixed):
    """露出する外辺の両側で法線を照合し、新しく付いた急な折れだけを内側で丸める。"""
    from .crosssection import GUTTER_WIDTH

    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, first, count = np.unique(np.sort(directed, axis=1), axis=0, return_index=True, return_counts=True)
    outer = first[count == 1]
    a, b = vertices[directed[outer]].transpose(1, 0, 2)
    tip = faces[outer // 3, (outer % 3 + 2) % 3]
    mid = (a + b) / 2
    edge = b - a
    length = np.linalg.norm(edge[:, :2], axis=1)
    along = edge[:, :2] / length[:, None]
    inward = along[:, ::-1] * [-1, 1]
    # 辺の中点を少し外へずらすと、隣接する粗い三角形の平面を一意に引ける。
    probe = mid[:, :2] - 0.10 * inward
    r, c = (np.searchsorted(grid[axis], probe[:, k], side="right") - 1 for k, axis in ((1, "ys"), (0, "xs")))
    rows, cols = grid["H"].shape
    valid = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
    r, c = np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)
    valid &= ~grid["H"][r, c]
    valid &= (
        np.maximum.reduce([grid["D"][r + dr, c + dc] for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))])
        <= terrain.NEAR_BAND
    )
    frame = terrain.road_frame(sec, *mid[:, :2].T)
    width = np.where(frame["lat"] > 0, sec.hw_left[frame["idx"]], sec.hw_right[frame["idx"]])
    valid &= (abs(frame["lat"]) > width + GUTTER_WIDTH + 0.20) & ~fixed[tip]

    # 辺に直交する平面で二面角を測る。辺自体の縦勾配も考慮する。
    parallel = edge[:, 2] / length
    scale = np.hypot(1, parallel)
    outside = np.arctan((mid[:, 2] - sample_grid(grid, *probe.T)) / (0.10 * scale))
    offset = vertices[tip, :2] - mid[:, :2]
    plane_z = mid[:, 2] + (offset * along).sum(axis=1) * parallel
    distance = (offset * inward).sum(axis=1) * scale
    old = np.arctan((sample_grid(grid, *vertices[tip, :2].T) - plane_z) / distance)
    drawn = np.arctan((vertices[tip, 2] - plane_z) / distance)
    old_fold = np.abs(old - outside)
    bad = valid & (abs(drawn - outside) > np.maximum(old_fold + np.deg2rad(0.25), np.deg2rad(34)))
    if not bad.any():
        return vertices[:, 2]
    tolerance = np.maximum(old_fold[bad], np.deg2rad(32))
    target = np.clip(drawn[bad], outside[bad] - tolerance, outside[bad] + tolerance)
    fitted = vertices.copy()
    z = plane_z[bad] + distance[bad] * np.tan(target)
    # 角では同じ内点が二つの辺に接する。大きな修正を優先する。
    order = np.argsort(abs(z - vertices[tip[bad], 2]), kind="stable")
    fitted[tip[bad][order], 2] = z[order]
    hold = fixed.copy()
    hold[tip[bad]] = True
    return limit_folds(fitted, faces, hold)


def build(sec, grid, rows, bounds, spans, *, pads=(), keep_rows=None, shoulder_section=None):
    from .valley_survey import bank_repair

    pieces = []
    offset = 0
    for key, bridge in (("A", 2), ("B", 16), ("C", 25)):
        group = [v for v in rows if v.id.startswith(key + "-")]
        if not group:
            continue
        site = valleys.site_area(sec, bounds, group[0].section, group[0].center_m)
        protected = valleys.protected_area(sec, site, pads, keep_rows)
        repair = bank_repair(group[0]) if key == "B" else None
        cells, v, f, bed = site_surface(sec, grid, group, site, spans[bridge], protected, repair)
        pieces.append((cells, v, f + offset, bed))
        offset += len(v)
    fine = dict(
        zip(
            ("valley_cells", "valley_vertices", "valley_faces", "valley_bed"),
            (np.concatenate(items) for items in zip(*pieces)),
        )
    )
    if shoulder_section is not None:
        ribbons = s5_shoulder_ribbons(shoulder_section, bounds, spans, grid=grid)
        ribbons.extend(bridge_approach_ribbons(shoulder_section, spans))
        fine = replace_shoulder_ribbons(grid, fine, ribbons)
        if ribbons:
            round_approach_folds(grid, fine, shoulder_section, bounds, spans)
    return fine


def keep_quads(verts, faces, near):
    r, c = near["valley_cells"].T
    width = len(near["xs"])
    xy = verts[faces[:, 0], :2]
    ids = np.searchsorted(near["ys"], xy[:, 1]) * width + np.searchsorted(near["xs"], xy[:, 0])
    return ~np.isin(ids, r * width + c)


def face_blocks(near):
    v, f = near["valley_vertices"], near["valley_faces"]
    blocks = np.arange(len(f)) // 16000
    if "valley_source_faces" in near:
        source = near["valley_source_faces"]
        origin_y = v[f[:, 0], 1]
        b = (source >= 0) & (origin_y >= 1500) & (origin_y < 6000)
        blocks[b] = source[b] // 16000
    return blocks


def mesh_parts(near):
    v, f, bed = (near[k] for k in ("valley_vertices", "valley_faces", "valley_bed"))
    # Smooth shading across UV and material seams. KN5 normals use game axes.
    normals = near.get("valley_edit_normals")
    if normals is None:
        normals = kn5._normals(v @ kn5.C.T, f)
    remainders = near.get("valley_coarse_remainder")
    gravel = near.get("valley_gravel", np.zeros(len(f), bool))
    blocks = near.get("valley_edit_blocks")
    if blocks is None:
        blocks = face_blocks(near)
    edited_materials = near.get("valley_edit_materials")
    edited_uv = near.get("valley_edit_uv")
    edited_normals = near.get("valley_edit_corner_normals")
    for block in np.unique(blocks):
        batch = blocks == block
        local = f[batch]
        active = True if remainders is None else remainders[batch, 0] < 0
        shoulder = gravel[batch]
        # Keep different sites in their own KN5 even when they fit into one packet.
        for site, select in (
            ("A", v[local[:, 0], 1] < 1500),
            ("B", (v[local[:, 0], 1] >= 1500) & (v[local[:, 0], 1] < 6000)),
            ("C", v[local[:, 0], 1] >= 6000),
        ):
            river = bed[batch]
            for tag, kind, material in (("bed", river & ~shoulder, "gravel_dark"),
                                         ("bank", ~river & ~shoulder, "valley_forest_floor"),
                                         ("shoulder", shoulder, "gravel_grey")):
                indices = np.flatnonzero(batch)[active & select & kind]
                if not len(indices):
                    continue
                used = f[indices]
                tri = v[used]
                n = abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
                axis = np.argmax(n, axis=1)
                planes = np.array([[1, 2], [0, 2], [0, 1]])[axis]
                uv = np.take_along_axis(tri, planes[:, None, :], axis=2) / (5.0 if tag == "bank" else 2.8)
                corner_normals = normals[used]
                stem = f"1GRASS_valley_{site}_{block:03d}"
                if edited_materials is None:
                    yield (f"{stem}_{tag}", tri.reshape(-1, 3), np.arange(used.size).reshape(-1, 3),
                           uv.reshape(-1, 2), material, corner_normals.reshape(-1, 3))
                    continue
                assigned = edited_materials[indices]
                edited = assigned != ""
                uv[edited] = edited_uv[indices[edited]]
                if edited_normals is not None:
                    explicit = np.isfinite(edited_normals[indices, 0, 0])
                    corner_normals[explicit] = edited_normals[indices[explicit]]
                assigned = np.where(edited, assigned, material)
                for chosen in np.unique(assigned):
                    selected = np.flatnonzero(assigned == chosen)
                    key = stem if chosen == material else f"{stem}_edit_{chosen}"
                    for packet, start in enumerate(range(0, len(selected), 16000)):
                        j = selected[start:start + 16000]
                        name = f"{key}_{tag}" if packet == 0 else f"{key}_{packet:02d}_{tag}"
                        yield (name, tri[j].reshape(-1, 3), np.arange(len(j) * 3).reshape(-1, 3),
                               uv[j].reshape(-1, 2), str(chosen), corner_normals[j].reshape(-1, 3))


def relax_banks(vertices, faces, pinned, fidelity):
    """各内辺の両側の横断勾配差を小さくする。外周と測量した沢床の高さは拘束する。"""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve

    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    first = np.cumsum(counts) - counts
    internal = first[counts == 2]
    pair = np.stack((order[internal], order[internal + 1]), axis=1) // 3
    edge = edges[order[internal]]
    xy = vertices[:, :2]
    delta = xy[edge[:, 1]] - xy[edge[:, 0]]
    normal = delta[:, ::-1] * [-1, 1] / np.linalg.norm(delta, axis=1)[:, None]
    tri = xy[faces]
    opposite = np.roll(tri, -1, axis=1) - np.roll(tri, -2, axis=1)
    area = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 2, 0] - tri[:, 0, 0]) * (
        tri[:, 1, 1] - tri[:, 0, 1]
    )
    grad = opposite[:, :, ::-1] * [1, -1] / area[:, None, None]
    values = np.einsum("neki,ni->nek", grad[pair], normal) * np.array([1.0, -1.0])[None, :, None]
    D = coo_matrix(
        (values.ravel(), (np.repeat(np.arange(len(pair)), 6), faces[pair].ravel())),
        shape=(len(pair), len(vertices)),
    ).tocsr()
    z = vertices[:, 2].copy()
    free = ~pinned
    A = D[:, free]
    fixed = D[:, pinned] @ z[pinned]
    w = np.ones(len(pair))
    target = z[free].copy()
    cost = fidelity[free].copy()
    graph_edges = np.unique(edges, axis=0)
    for iteration in range(32):
        weighted = A.T @ diags(w)
        z[free] = spsolve(weighted @ A + diags(cost), cost * target - weighted @ fixed)
        if not np.isfinite(z).all():
            raise ValueError("谷の法面計算で有限の高さが得られません")
        drained = drain_to_boundary(z, graph_edges, pinned)
        pooled = drained[free] - z[free] > 0.0001
        target[pooled] = np.maximum(target[pooled], drained[free][pooled])
        cost[pooled] = np.maximum(cost[pooled], 8.0)
        tri = np.column_stack((xy, drained))[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1)[:, None]
        dot = np.einsum("ni,ni->n", n[pair[:, 0]], n[pair[:, 1]])
        angle = np.rad2deg(np.arccos(np.clip(dot, -1, 1)))
        if angle.max(initial=0) <= 34.5:
            break
        w = np.minimum(2e4, w * np.clip((angle / 29.0) ** 2, 1, 5))
    return drained


def drain_to_boundary(z, edges, boundary, *, parents=False):
    """閉じた窪みだけを最低の流出口まで満たし、外縁を固定する。"""
    from heapq import heapify, heappop, heappush
    from scipy.sparse import csr_matrix

    a, b = edges.T
    graph = csr_matrix((np.ones(2 * len(edges), bool), (np.r_[a, b], np.r_[b, a])), shape=(len(z), len(z)))
    parent = np.arange(len(z))
    seen = boundary.copy()
    surface = z.copy()
    queue = list(zip(z[boundary], np.flatnonzero(boundary)))
    heapify(queue)
    while queue:
        h, i = heappop(queue)
        for j in graph.indices[graph.indptr[i] : graph.indptr[i + 1]]:
            if seen[j]:
                continue
            seen[j] = True
            parent[j] = i
            surface[j] = max(h, z[j])
            heappush(queue, (surface[j], j))
    return (surface, parent) if parents else surface


def limit_folds(vertices, faces, locked, *, shoulder=None):
    """大きな折れが残る内辺の周辺だけ、面法線の角度を直接制限する。"""
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, inv, n = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    starts = (np.cumsum(n) - n)[n == 2]
    order = np.argsort(inv)
    pair = np.stack((order[starts], order[starts + 1]), axis=1) // 3
    xy = vertices[:, :2]
    z = vertices[:, 2].copy()
    outer = np.zeros(len(z), bool)
    outer[unique[n == 1]] = True
    if shoulder is None:
        limits = np.full(len(pair), 34.7)
        relief = None
    else:
        limits, lower, upper = shoulder(vertices[unique[n == 2]].mean(1))
        relief = drain_to_boundary(z, unique, outer) - z
    for attempt in range(6):
        tri = vertices[faces].copy()
        tri[:, :, 2] = z[faces]
        norm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm /= np.linalg.norm(norm, axis=1)[:, None]
        dot = (norm[pair[:, 0]] * norm[pair[:, 1]]).sum(1)
        bad = pair[dot < np.cos(np.deg2rad(limits))]
        if not len(bad):
            break
        distance = cKDTree(xy[np.unique(faces[bad])]).query(xy)[0]
        move = (distance < 3.0 + attempt) & ~locked
        active = move[faces].any(1)
        candidates = active[pair].any(1) & (limits < 180.)
        pairs = pair[candidates]
        angle_limit = np.cos(np.deg2rad(limits[candidates] - (2.7 if shoulder is None else 8.)))
        ids, local = np.unique(faces[pairs], return_inverse=True)
        local = local.reshape(-1, 2, 3)
        x = xy[ids][local]
        d = x[:, :, 1:] - x[:, :, :1]
        area = d[:, :, 0, 0] * d[:, :, 1, 1] - d[:, :, 0, 1] * d[:, :, 1, 0]
        opp = np.roll(x, -1, axis=2) - np.roll(x, -2, axis=2)
        G = opp[:, :, :, ::-1] * [1, -1] / area[:, :, None, None]
        # Constrain the already draining paths through the optimisation region.
        # Otherwise flattening a newly formed pit introduces a second abrupt fold.
        _, parent = drain_to_boundary(z, unique, outer, parents=True)
        # Prefer the steepest downhill neighbour when a vertex is off a flat pool.
        e = np.concatenate((unique, unique[:, ::-1]))
        downhill = z[e[:, 1]] < z[e[:, 0]] - 1e-7
        e = e[downhill]
        if len(e):
            rank = np.lexsort((z[e[:, 1]], e[:, 0]))
            sorted_edges = e[rank]
            node, first = np.unique(sorted_edges[:, 0], return_index=True)
            parent[node] = sorted_edges[first, 1]
        affected = np.flatnonzero(move | move[parent])
        if relief is not None:
            affected = affected[relief[affected] <= .001]
        compare = np.c_[affected, parent[affected]]
        # Both endpoints must be in the regional vector, including fixed neighbours.
        old_ids = ids
        ids = np.union1d(ids, compare.ravel())
        local = np.searchsorted(ids, old_ids)[local]
        compare = np.searchsorted(ids, compare)
        initial = z[ids].copy()
        free = move[ids]
        if not free.any():
            break
        height_bounds = None if shoulder is None else list(zip(lower[ids[free]], upper[ids[free]]))

        drainage_weight = 8. if shoulder is None else 2.5

        def energy(values):
            h = initial.copy()
            h[free] = values
            grad = np.einsum("peki,pek->pei", G, h[local])
            vec = np.concatenate([-grad, np.ones((*grad.shape[:2], 1))], axis=2)
            length = np.linalg.norm(vec, axis=2)
            N = vec / length[:, :, None]
            dot = (N[:, 0] * N[:, 1]).sum(1)
            err = np.minimum(dot - angle_limit, 0)
            derivative = (dot[:, None, None] * N[:, :, :2] - N[:, ::-1, :2]) / length[:, :, None]
            coeff = np.einsum("pei,peki->pek", derivative, G) * err[:, None, None]
            jac = np.bincount(local.ravel(), weights=coeff.ravel(), minlength=len(ids))
            diff = np.minimum(h[compare[:, 0]] - h[compare[:, 1]], 0)
            force = drainage_weight * diff
            np.add.at(jac, compare[:, 0], force)
            np.add.at(jac, compare[:, 1], -force)
            shift = values - initial[free]
            return 0.5 * (err @ err + drainage_weight * (diff @ diff) + 1e-6 * (shift @ shift)), jac[free] + 1e-6 * shift

        solution = minimize(
            energy,
            initial[free],
            jac=True,
            method="L-BFGS-B",
            bounds=height_bounds,
            options={"maxiter": 350 if shoulder is None else 2500, "ftol": 1e-15, "gtol": 1e-9},
        )
        z[ids[free]] = solution.x
        drained = drain_to_boundary(z, unique, outer)
        if relief is None:
            z = drained
        else:
            z[move] = np.maximum(z[move], drained[move] - relief[move])
    return z


def s5_shoulder_ribbons(sec, bounds, spans, *, grid=None):
    """小橋25の開口から離れた側で、側溝と旧法面の間を細かくすりつける。"""
    from .crosssection import GUTTER_WIDTH

    if 25 not in spans:
        return []
    start = sec.s[spans[25][1]]
    if not np.any((sec.s > start) & (sec.s < start + 45.) & (sec.hw_left < 3.74)):
        return []
    end = bounds["s5"][0] + 9010.
    marks = np.arange(start + .25, end, .25)
    rejoin = end - .25
    marks = np.sort(np.r_[marks[abs(marks - rejoin) > .04], rejoin])
    next_row = np.searchsorted(sec.s, marks)
    apart = np.minimum(marks - sec.s[next_row - 1], sec.s[next_row] - marks) > .04
    s = np.sort(np.r_[start, marks[apart], sec.s[(sec.s > start) & (sec.s < end)], end])
    h = np.interp(s, sec.s, np.unwrap(sec.heading))
    normal = np.c_[-np.sin(h), np.cos(h)]
    offsets = np.array([-1.4, -1.2, -1., -.8, -.6, -.45, -.3, -.22, -.15, -.08,
                        0., .09, .24, .45, .7, 1.03, 1.18,
                        1.33, 1.48, 1.63, 1.78, 1.93, 2.08, 2.23, 2.39])
    across = smooth((offsets + 1.4) / 1.25) * smooth((2.39 - offsets) / 1.36)
    ribbons = []
    for sign, width in ((1, sec.hw_left), (-1, sec.hw_right)):
        # 左は欄干から擁壁へ続く砂利の路肩を、橋台の直後から受ける。
        nose = .42 if sign > 0 else .90
        if sign > 0:
            finish = smooth((end - s[:, None]) / 1.25)
        else:
            # The outer slope reaches the original bank sooner than the
            # shoulder does, leaving the old ditch cross section intact.
            tail = end - .25 - .28 * smooth((offsets - 1.08) / .47)
            finish = smooth((tail - s[:, None]) / .88)
        along = smooth((s - start) / nose)[:, None] * finish
        lip_rows = np.c_[sec.x, sec.y] + sign * (width + GUTTER_WIDTH)[:, None] * np.c_[
            -np.sin(sec.heading), np.cos(sec.heading)]
        lip_xy = np.stack([np.interp(s, sec.s, line) for line in lip_rows.T], axis=1)
        xy = lip_xy[:, None] + sign * offsets[None, :, None] * normal[:, None]
        edge = np.interp(s, sec.s, sec.edge_z(sign))
        drop = np.interp(offsets, [-1.4, -.60, -.30, -.22, -.15, -.08, 0., .09, 1.03, 2.39],
                         [.18, .17, .14, .125, .092, .048, .0005, -.001, -.005, .07])
        z = edge[:, None] - drop[None, :]
        local = s - bounds["s5"][0]
        if sign < 0 and grid is not None:
            # Keep the shoulder above the old grade where the old ditch
            # reappears and its floor crosses the shoulder height.
            old = sample_grid(grid, *np.moveaxis(xy, 2, 0)) - edge[:, None]
            at_end = smooth((local - 9008.65) / .35) * smooth((9009.63 - local) / .32)
            outer = smooth((offsets - .15) / .16) * smooth((1.30 - offsets) / .25)
            z += at_end[:, None] * outer * np.maximum(0., .043 - .32 * np.abs(old))
            # 9009 mの最後の横断は、窪みへ落とす前に横一列を路肩の
            # 高さへそろえる。外側は幅0.4 mで元の法面へすりつける。
            end_lip = smooth((local - 9009.) / .17) * smooth((9009.66 - local) / .26)
            lip_width = smooth((offsets - .63) / .24) * smooth((1.44 - offsets) / .34)
            z += .075 * end_lip[:, None] * lip_width
        gravel = (sign > 0) & ((s >= start) & (local <= 8980.))[:, None] & (offsets <= .45)
        # 壁の裏と、谷側の肩より外にある既存法面は、元より低く切らない。
        left_wall = (sign > 0) * smooth((local - 8960.5) / .5) * smooth((9009.5 - local) / .5)
        # 367の末端と370の始端の間は、壁の高さで法面を残さず肩をつなぐ。
        left_wall *= smooth((abs(local - 8980.5) - .70) / 1.25)
        face = 1. - smooth((offsets - .12) / .34)
        wider = (sign > 0) * (1 - smooth((abs(local - 8980.5) - .70) / 1.25))
        slope = 1. - smooth((offsets[None, :] - 1.03) / (.17 + 1.05 * wider[:, None]))
        cut = slope + left_wall[:, None] * (face - slope)
        drains = np.broadcast_to(offsets >= 1.03, z.shape)
        ribbons.append(np.dstack((xy, z, along * across, gravel, cut, drains)))
    return ribbons



def bridge_approach_ribbons(sec, spans):
    """Aの両端とCの手前で、欄干の外側に落ち込んだ左の路肩を受ける。"""
    from .crosssection import GUTTER_WIDTH

    offsets = np.array([-.24, -.14, -.06, .02, .10, .20, .35, .50, .62, .76, .98, 1.24, 1.55, 1.9])
    across = smooth((offsets + .24) / .32) * smooth((1.9 - offsets) / 1.26)
    lip = np.c_[sec.x, sec.y] + (sec.hw_left + GUTTER_WIDTH)[:, None] * np.c_[
        -np.sin(sec.heading), np.cos(sec.heading)]
    heading = np.unwrap(sec.heading)
    for bridge, end, direction in ((2, 0, -1), (2, 1, 1), (25, 0, -1)):
        if bridge not in spans:
            continue
        start = sec.s[spans[bridge][end]]
        marks = np.arange(-1.24, 5.01, .20)
        marks = np.sort(np.r_[marks, -.12, 0., .10])
        s = np.sort(start + direction * marks)
        h = np.interp(s, sec.s, heading)
        normal = np.c_[-np.sin(h), np.cos(h)]
        center = np.stack([np.interp(s, sec.s, line) for line in lip.T], axis=1)
        xy = center[:, None] + offsets[None, :, None] * normal[:, None]
        edge = np.interp(s, sec.s, sec.edge_z(1))
        drop = np.interp(offsets, [-.24, -.06, .10, 1.9], [.18, .14, .067, .10])
        z = edge[:, None] - drop
        away = (s - start) * direction
        # Under the deck, only the outer bank rises. The inner fill begins
        # below the gutter after passing the abutment.
        lead = 1.24 * smooth((offsets - .03) / .15)
        inner = 1. - smooth((offsets - .04) / .12)
        along = smooth((away[:, None] + lead - .02 * inner) / (lead + .055 + .14 * inner)) * smooth((5. - away[:, None]) / 1.85)
        weight = along * across
        yield np.dstack((xy, z, weight, np.zeros_like(weight), np.zeros_like(weight), np.ones_like(weight)))


def round_approach_folds(grid, fine, sec, bounds, spans):
    """路肩の高さと外周を拘束し、橋詰めと法面に残る急な折れをならす。"""
    from . import meshgen, small_bridges, track
    from .crosssection import GUTTER_WIDTH, parse_width_rows, width_taper_areas
    from .valley_fit import Surfaces

    v = fine["valley_vertices"]
    f = fine["valley_faces"]
    corners = {k: sec.s[np.array(spans[k])] for k in (2, 25) if k in spans}
    if 25 not in corners or 2 not in corners:
        return
    strip_ground = Surfaces(v[f])

    def ground_height(x, y):
        xy = np.c_[np.ravel(x), np.ravel(y)]
        return strip_ground.height(xy).reshape(np.shape(x))

    cover = []
    for k, (start, end) in corners.items():
        stop = end + (54. if k == 25 else 7.)
        rows = slice(np.searchsorted(sec.s, start - 7.) - 1, np.searchsorted(sec.s, stop) + 1)
        ribbons = [meshgen.road_surface(sec, rows), meshgen.gutter(sec, rows, 1),
                   meshgen.gutter(sec, rows, -1)]
        parts = small_bridges.support_meshes(
            sec, {k: spans[k]}, ground_height,
            graded_shoulder=True,
        )
        for part in parts:
            if part[0].endswith(("_deck", "_left", "_right")):
                ribbons.append(part[1:4])
        cover.extend(points[kn5._triangles(faces)] for points, faces, _ in ribbons)
    roof = Surfaces(np.concatenate(cover))

    def roadside(points):
        frame = terrain.road_frame(sec, *points[:, :2].T)
        i, lateral = frame["idx"], frame["lat"]
        s = sec.s[i] + frame["along"]
        left = lateral > 0
        width = np.where(left, np.interp(s, sec.s, sec.hw_left), np.interp(s, sec.s, sec.hw_right))
        return s, left, abs(lateral) - width - GUTTER_WIDTH

    s, left, offset = roadside(v)
    a, b = corners[2]
    c, d = corners[25]
    s5 = bounds["s5"][0]
    # 青線の沢床、手を入れない区間、置換セルの外周は固定する。
    zones = (
        (a - 5.5, b + 5.6, left),
        (c - 5.7, c + 2.1, left),
        (d - 2.5, s5 + 8963., np.ones_like(left)),
        (s5 + 8977.5, s5 + 8983., left),
        (s5 + 9003.8, s5 + 9011.2, ~left),
    )
    movable = np.logical_or.reduce([(s >= a) & (s <= b) & side for a, b, side in zones])
    movable &= (offset > -1.15) & (offset < 2.35)
    # Lock the original shoulder profile under gutters in the C taper.
    movable &= ~((s >= s5 + 8959.8) & (offset <= .025) & (left | (s >= s5 + 8961.35)))
    movable &= ~(~left & (s >= s5 + 9009.84) & (offset <= 1.35))
    remainder = fine["valley_coarse_remainder"][:, 0] >= 0
    movable[f[remainder]] = False
    sides = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(sides, axis=0, return_counts=True)
    movable[edges[counts == 1]] = False
    road_z = np.where(left, np.interp(s, sec.s, sec.edge_z(1)), np.interp(s, sec.s, sec.edge_z(-1)))
    # Keep the new bank outside the soffit projection and retain clearance
    # during height optimisation, including at the abutment seam.
    under_deck = (((s >= a - .035) & (s <= b + .035)) | ((s >= c - .035) & (s <= d + .035)))
    under_deck &= (offset <= .020) & left
    ceiling = road_z - small_bridges.DECK_DEPTH - small_bridges.DECK_TOP_DROP - 1.56
    v[under_deck, 2] = np.minimum(v[under_deck, 2], ceiling[under_deck])
    original = v[:, 2].copy()
    lower = original - 2.8
    upper = original + 2.8
    walk = (offset >= -.01) & (offset <= 1.18) & (s >= s5 + 8962.) & (s <= s5 + 9009.1)
    lower[walk] = np.minimum(original[walk], road_z[walk] - .13)
    upper[walk] = np.maximum(original[walk], road_z[walk] + .038)
    shoulder = (offset >= -.02) & (offset <= 1.30) & (s >= s5 + 8962.) & (s <= s5 + 9010.1)
    baseline = np.minimum(road_z[shoulder], sample_grid(grid, *v[shoulder, :2].T))
    lower[shoulder] = np.maximum(lower[shoulder], np.minimum(original[shoulder], baseline + .004))
    last_shoulder = (~left & (s >= s5 + 9009.12) & (s <= s5 + 9009.41)
                     & (offset >= .78) & (offset <= 1.18))
    lower[last_shoulder] = np.maximum(lower[last_shoulder], original[last_shoulder] - .004)
    edge = (offset > .06) & (offset < .68) & (s < s5 + 8957.)
    lower[edge] = np.minimum(original[edge], road_z[edge] - .20)
    top = roof.height(v[:, :2])
    covered = np.isfinite(top)
    upper[covered] = np.minimum(upper[covered], np.maximum(original[covered], top[covered] - .015))
    upper[under_deck] = np.minimum(upper[under_deck], ceiling[under_deck])
    lower[under_deck] = np.minimum(lower[under_deck], upper[under_deck])

    def limits(mid):
        s, left, offset = roadside(mid)
        top = roof.height(mid[:, :2])
        exposed = (top <= mid[:, 2] + .24) & (offset > -.45)
        bank = ~left & (s >= s5 + 8962.) & (s <= s5 + 9006.)
        # Preserve the backs of retaining walls. The opening between them is shaped from both ends.
        left_cut = left & (s >= s5 + 8961.5) & (s <= s5 + 9009.)
        opening = (s >= s5 + 8979.8) & (s <= s5 + 8981.3)
        exposed &= ~left_cut | opening | (offset < .27)
        return np.where(exposed, np.where(bank, 59.7, 34.7), 180.), lower, upper

    v[:, 2] = limit_folds(v, f, ~movable, shoulder=limits)

    # The tail joins coarse triangles along a skew grid line. Allow the
    # small remainder triangles on the inside of the road-width polygon to
    # soften together with the embankment. Pin every vertex on an incident
    # face that reaches beyond that polygon.
    rows = parse_width_rows(track.load_section_rows("widths", bounds))
    _, _, area = next(item for item in width_taper_areas(sec, rows) if item[0] <= d <= item[1])
    within = shapely.covers(area.buffer(1e-8), shapely.polygons(v[f, :2]))
    tail = (s > s5 + 9008.) & (s < s5 + 9010.) & ~left & (offset > 1.36) & (offset < 2.38)
    tail[f[~within]] = False
    tail[edges[counts == 1]] = False
    before_tail = v[:, 2].copy()

    def tail_limits(mid):
        station, left, _ = roadside(mid)
        active = (station > s5 + 9007.8) & (station < s5 + 9010.7) & ~left
        return np.where(active, 33.5, 180.), before_tail - .55, before_tail + .55

    finished = limit_folds(v, f, ~tail, shoulder=tail_limits)
    changed = abs(finished - before_tail) > 1e-8
    v[changed, 2] = finished[changed]
    fine["valley_coarse_remainder"][changed[f].any(axis=1)] = -1

    # Keep the crown at the end of the right shoulder below the height
    # limit; blend the cut back into the adjoining ground.
    station = s - s5
    apron = smooth((station - 9008.95) / .24) * smooth((9009.62 - station) / .18)
    apron *= smooth((offset - .72) / .07) * smooth((1.38 - offset) / .21)
    apron[left] = 0.
    apron[f[~within]] = 0.
    apron[edges[counts == 1]] = 0.
    trim = apron * np.clip(v[:, 2] - road_z - .037, 0., .06)
    v[:, 2] -= trim
    fine["valley_coarse_remainder"][(trim[f] > 0.).any(axis=1)] = -1

    # Round the two short joins independently. At the ditch, protect the
    # existing shoulder heights to within the 1 mm comparison tolerance.
    joins = (
        (c - .04, c + 1.15, True, .06, .90, -.09, .16),
        (s5 + 9009.52, s5 + 9010.03, False, .65, 1.70, -.001, .12),
    )
    for start, end, is_left, inner, outer, low, high in joins:
        move = (s > start) & (s < end) & (left == is_left) & (offset > inner) & (offset < outer)
        preserved_faces = fine["valley_coarse_remainder"][:, 0] >= 0 if is_left else ~within
        move[f[preserved_faces]] = False
        move[edges[counts == 1]] = False
        before_join = v[:, 2].copy()
        high_z = before_join + high
        if not is_left:
            high_z = np.maximum(before_join, np.minimum(high_z, road_z + .043))

        def join_limits(mid):
            station, mid_left, _ = roadside(mid)
            covered_above = roof.height(mid[:, :2]) > mid[:, 2] + .005
            active = (station > start - .5) & (station < end + .5) & (mid_left == is_left) & ~covered_above
            return np.where(active, 33.4, 180.), before_join + low, high_z

        z = limit_folds(v, f, ~move, shoulder=join_limits)
        changed = abs(z - before_join) > 1e-8
        v[changed, 2] = z[changed]
        fine["valley_coarse_remainder"][changed[f].any(axis=1)] = -1


def replace_shoulder_ribbons(grid, fine, ribbons):
    """肩の輪郭で既存の面を切り分け、輪郭上と輪郭の外には旧面の高さを残す。"""
    if not ribbons:
        return fine

    cells, vertices, faces, bed = (fine[k] for k in ("valley_cells", "valley_vertices",
                                                    "valley_faces", "valley_bed"))
    bands = []
    for ribbon in ribbons:
        ids = np.arange(ribbon.shape[0] * ribbon.shape[1]).reshape(ribbon.shape[:2])
        quads = np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]), axis=-1)
        if ribbon.shape[2] == 4:
            ribbon = np.pad(ribbon, ((0, 0), (0, 0), (0, 1)))
            ribbon = np.dstack((ribbon, np.ones(ribbon.shape[:2]), np.zeros(ribbon.shape[:2])))
        bands.append(ribbon.reshape(-1, 7)[quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)])
    bands = np.concatenate(bands)
    band_polys = shapely.polygons(bands[..., :2])
    region = shapely.union_all(band_polys)
    xs, ys = grid["xs"], grid["ys"]
    x0, y0, x1, y1 = region.bounds
    c0, c1 = np.searchsorted(xs, [x0, x1])
    r0, r1 = np.searchsorted(ys, [y0, y1])
    rr, cc = np.mgrid[max(0, r0 - 1):min(len(ys) - 1, r1),
                      max(0, c0 - 1):min(len(xs) - 1, c1)]
    rr, cc = rr.ravel(), cc.ravel()
    boxes = shapely.box(xs[cc], ys[rr], xs[cc + 1], ys[rr + 1])
    inside = shapely.area(shapely.intersection(boxes, region)) > 1e-10
    inside &= ~grid["H"][rr, cc]
    for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
        inside &= grid["D"][rr + dr, cc + dc] <= terrain.NEAR_BAND
        inside &= np.isfinite(grid["Z"][rr + dr, cc + dc])
    old_cells = set(map(tuple, cells))
    new_cells = np.array([(r, c) for r, c in zip(rr[inside], cc[inside])
                          if (r, c) not in old_cells], dtype=int).reshape(-1, 2)
    old = vertices[faces]
    affected = shapely.area(shapely.intersection(region, shapely.polygons(old[..., :2]))) > 1e-10
    source = np.concatenate((old[affected], valleys.original_triangles(grid, new_cells)))
    source_bed = np.r_[bed[affected], np.zeros(len(new_cells) * 2, bool)]
    polys = shapely.polygons(source[..., :2])
    outline = shapely.union_all(polys)
    # Node the linework together so intersections appear on both incident faces.
    seams = shapely.intersection(shapely.boundary(band_polys), outline)
    lines = shapely.get_parts(shapely.union_all(np.r_[shapely.boundary(polys), seams]))
    regions = shapely.get_parts(shapely.polygonize(lines))
    markers = shapely.point_on_surface(regions)
    ri, si = shapely.STRtree(polys).query(markers, predicate="within")
    owners = np.full(len(regions), -1)
    owners[ri] = si
    ri, bi = shapely.STRtree(band_polys).query(markers, predicate="within")
    design = np.full(len(regions), -1)
    design[ri] = bi
    ri = np.flatnonzero(owners >= 0)
    triangles, parent = [], []
    for i in ri:
        for part in shapely.get_parts(shapely.constrained_delaunay_triangles(regions[i])):
            xy = shapely.get_coordinates(part)[:3]
            if part.area > 1e-12:
                triangles.append(xy if shapely.is_ccw(part.exterior) else xy[::-1])
                parent.append(i)
    xy, parent = np.asarray(triangles).reshape(-1, 3, 2), np.asarray(parent, int)
    owner = owners[parent]
    ribbon = design[parent]
    in_ribbon = ribbon >= 0

    def interpolate(tri, xy):
        relative = tri[:, 1:, :2] - tri[:, :1, :2]
        weights = (xy - tri[:, :1, :2]) @ np.linalg.inv(relative)
        return tri[:, :1, 2:] + weights @ (tri[:, 1:, 2:] - tri[:, :1, 2:])

    baseline = interpolate(source[owner], xy)
    old_z = np.concatenate((old[~affected, :, 2], baseline[..., 0]))
    target = interpolate(bands[ribbon[in_ribbon]], xy[in_ribbon])
    delta = target[..., :1] - baseline[in_ribbon]
    delta = np.maximum(delta, 0) + np.minimum(delta, 0) * np.clip(target[..., 3:4], 0, 1)
    baseline[in_ribbon] += np.clip(target[..., 1:2], 0, 1) * delta
    all_tri = np.concatenate((old[~affected], np.concatenate((xy, baseline), axis=2)))
    # Keep the original vertex coordinates and weld sub-micrometre differences
    # introduced by interpolation through different incident triangles.
    raw = all_tri.reshape(-1, 3)
    _, ids, inverse = np.unique(raw[:, :2].round(8), return_index=True, return_inverse=True, axis=0)
    v = raw[ids]
    faces = inverse.reshape(-1, 3)
    distinct = (faces != np.roll(faces, 1, axis=1)).all(axis=1)
    drainage = np.zeros(len(v), bool)
    ribbon_faces = faces[len(old) - affected.sum():][in_ribbon]
    drainage[ribbon_faces[(target[..., 4] >= 1 - 1e-9) & (target[..., 1] > 1e-9)]] = True
    if drainage.any():
        # 肩より外の法面に今回生じた窪みだけを越流の高さまで戻す。
        # 粗格子の残片、路肩の上端、舗装下にあった窪みは移動させない。
        fixed_faces = faces[distinct]
        sides = np.sort(fixed_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        edges, counts = np.unique(sides, axis=0, return_counts=True)
        boundary = np.zeros(len(v), bool)
        boundary[edges[counts == 1]] = True
        old_z = old_z.ravel()[ids]
        old_depth = drain_to_boundary(old_z, edges, boundary) - old_z
        drained = drain_to_boundary(v[:, 2], edges, boundary)
        v[drainage, 2] = np.maximum(v[drainage, 2], drained[drainage] - old_depth[drainage])
    coarse = np.full((len(all_tri), 3), -1, dtype=int)
    origin = owner - affected.sum()
    active = in_ribbon.copy()
    active[in_ribbon] = np.any(target[..., 1] > 1e-12, axis=1)
    copy = (origin >= 0) & ~active
    coarse[len(old) - affected.sum() + np.flatnonzero(copy)] = np.c_[new_cells[origin[copy] // 2], origin[copy] % 2]
    gravel = np.zeros(len(all_tri), bool)
    gravel[len(old) - affected.sum() + np.flatnonzero(in_ribbon)] = bands[ribbon[in_ribbon], :, 4].mean(axis=1) >= .5
    return {
        "valley_cells": np.concatenate((cells, new_cells)),
        "valley_vertices": v,
        "valley_faces": faces[distinct],
        "valley_bed": np.r_[bed[~affected], source_bed[owner]][distinct],
        "valley_coarse_remainder": coarse[distinct],
        "valley_gravel": gravel[distinct],
        # Preserve the original chunk assignments of B, which the shoulder
        # patches do not touch.
        "valley_source_faces": np.r_[np.flatnonzero(~affected), np.full(len(xy), -1)][distinct],
    }


def copy_attributes(source, xy, uv, normals):
    """粗格子の残片には元の材質・UV・法線の補間を使う。"""
    delta = source[:, 1:, :2] - source[:, :1, :2]
    weights = (xy - source[:, :1, :2]) @ np.linalg.inv(delta)
    return tuple(a[:, :1] + weights @ (a[:, 1:] - a[:, :1]) for a in (uv, normals))
