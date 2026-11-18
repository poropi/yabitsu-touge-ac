"""S5 の吹風トンネル手前の形と部品配置。距離は S5 の起点からの m。"""
import json
from dataclasses import replace

import numpy as np
import shapely

from . import config, dress_mountain as dm, features, landmarks, terrain
from .meshgen import compact, ribbon
from .start_lot import _merge

ASSET = "r70_fubuki"
SLOPE_START = 10338.0
SLOPE_END = 10507.0
KAMATAGAWA_ID = 131561912


def crest_offset(s):
    """右法面の上端。橋側の端に向かって道の方へ寄る。"""
    return np.interp(s, [SLOPE_START, 10485.0, 10504.0], [14.0, 14.0, 12.0])


def walls_after_gate(feats, s5):
    return [(k, wall) for k, wall in enumerate(feats)
            if wall.side == "R" and wall.type == "retaining_wall"
            and s5 + 10435 <= wall.s_from < wall.s_to <= s5 + SLOPE_END]


def slope_footprint(sec, feats, s5, runs):
    """貼り替え表の各区間で、右擁壁の天端と法面上端が囲む平面範囲。"""
    walk = features.sidewalk_offsets(sec, feats)[1]
    foot = sec.hw_right + features.GUTTER_WIDTH + walk
    areas = []
    for a, b, row in runs:
        for wall in feats:
            if wall.type != "retaining_wall" or wall.side != row["side"]:
                continue
            start, end = max(a, wall.s_from), min(b, wall.s_to)
            if start >= end:
                continue
            rows = dm.rows_between(sec, start, end)
            s = np.r_[start, sec.s[rows][(sec.s[rows] > start) & (sec.s[rows] < end)], end]
            run = np.c_[s, np.zeros((len(s), 2))]
            top = foot[rows] + features.WALL_LEAN * wall.height_m
            xy0 = dm.bend_on_rail(run, sec, rows, -1, top, top * 0, lambda x: x)[:, :2]
            local_s = sec.s[rows] - s5
            crest = crest_offset(local_s)
            if a >= s5 + 10436:
                # フェンスが始まる 10445 m までは、白い家の敷地を避けて法面の幅を絞る。
                crest = np.minimum(crest, np.interp(local_s, [10442, 10445], [7.25, 14.0]))
            xy1 = dm.bend_on_rail(run, sec, rows, -1, crest, crest * 0, lambda x: x)[:, :2]
            areas.append(shapely.Polygon(np.r_[xy0, xy1[::-1]]))
    return shapely.union_all(areas)


def split_slope_triangles(v, faces, uv, footprint):
    """草地の境をまたぐ三角形を切る。新しい頂点の高さと UV は元の三角形上で補間する。"""
    triangles = shapely.polygons(v[faces, :2])
    candidates = np.flatnonzero(shapely.intersects(triangles, footprint))
    overlap = shapely.intersection(triangles[candidates], footprint)
    has_area = shapely.area(overlap) > 1e-9
    if not has_area.any():
        return []
    inside = np.zeros(len(faces), dtype=bool)
    inside[candidates] = shapely.covered_by(triangles[candidates], footprint)
    crosses = has_area & ~inside[candidates]
    border = candidates[crosses]
    attrs = np.c_[v, uv]
    additions = [[], []]
    for idx, grass in zip(border, overlap[crosses]):
        orig = attrs[faces[idx]]
        xy = orig[:, :2]
        basis = (xy[1:] - xy[0]).T
        pieces = [grass, triangles[idx].difference(footprint)]
        for out, piece in zip(additions, pieces):
            for tri in shapely.constrained_delaunay_triangles(piece).geoms:
                p = np.array(shapely.orient_polygons(tri).exterior.coords)[:3]
                weights = np.linalg.solve(basis, (p - xy[0]).T).T
                other = orig[0, 2:] + weights @ (orig[1:, 2:] - orig[0, 2:])
                out.append(np.c_[p, other])

    outside = ~inside
    outside[border] = False
    result = []
    for tag, key, keep, extra in zip(("grass", "terrain"), ("mt_grass", "mt_terrain"), (inside, outside), additions):
        f = faces[keep]
        data = attrs
        if extra:
            extra = np.concatenate(extra)
            _, first, ids = np.unique(np.round(np.vstack([attrs, extra]), 9), axis=0,
                                      return_index=True, return_inverse=True)
            f = np.vstack([f, first[ids[len(attrs):]].reshape(-1, 3)])
            data = np.vstack([attrs, extra])
        if len(f):
            result.append((tag, key, compact(data[:, :3], f, data[:, 3:])))
    return result


def wall_cap_meshes(sec, feats, s5, ground):
    """右擁壁の天端を閉じ、橋側の薄い背面を足元から地形まで延ばす。"""
    walk = features.sidewalk_offsets(sec, feats)[1]
    for k, wall in walls_after_gate(feats, s5):
        sl = features.rows_of(sec, wall)
        stations = sec.s[sl]
        # 橋側だけ行間を細かくし、地形の三角形をまたぐ垂れの下に隙間を残さない。
        terminal = stations > stations[-1] - 3.0 + 1e-4
        coarse = np.arange(len(stations))
        start = max(0, np.flatnonzero(terminal)[0] - 1)
        fine = np.r_[coarse[:start], np.arange(start, coarse[-1], 1 / 32), coarse[-1]]
        sample = lambda a: np.interp(fine, coarse, a)
        h = sample(np.unwrap(sec.heading[sl]))
        edge = sample(sec.edge_z(-1)[sl])
        coarse_foot = sec.hw_right[sl] + features.GUTTER_WIDTH + walk[sl]
        lean = features.WALL_LEAN * wall.height_m
        foot = sample(coarse_foot)
        top = foot + lean
        thickness = 0.04
        prior_taper = np.clip((stations - stations[-1]) / 3.0 + 1, 0, 1)
        back_width = np.where(terminal, thickness, 0.35 - 0.31 * prior_taper)
        back = top + sample(back_width)
        top_z = edge + wall.height_m
        foot_z = edge - features.WALL_BASE_DROP
        rear_foot = sample(coarse_foot + np.where(terminal, 0.0, lean) + back_width)
        x, y = sample(sec.x[sl]), sample(sec.y[sl])
        z_back = ground(x + rear_foot * np.sin(h), y - rear_foot * np.cos(h))
        bottom = np.minimum(top_z - 0.15, z_back - 0.05)
        rear_foot_z = np.where(sample(terminal) > 0, foot_z, bottom)
        bottom = np.minimum(bottom, rear_foot_z)
        skirt = bottom < rear_foot_z
        face_mask = np.ones((len(top) - 1, 3), dtype=bool)
        face_mask[:, 2] = skirt[:-1] | skirt[1:]
        v, faces, uv = ribbon(x, y, h, -np.c_[top, back, rear_foot, rear_foot],
                              np.c_[top_z, top_z, rear_foot_z, bottom], sample(stations),
                              u_scale=2.0, face_mask=face_mask, flip=True)
        # 垂れの始まりでは上下の点を共有し、四角形を三角形にする。
        faces = np.where((faces % 4 == 3) & ~skirt[faces // 4], faces - 1, faces)
        rise = wall.height_m + features.WALL_BASE_DROP
        uv[:, 0] += np.hypot(top - foot, rise).repeat(4) / 2.0
        cap_verts, cap_uvs = [], []
        for j in (0, len(top) - 1):
            toe_xy = [x[j] + foot[j] * np.sin(h[j]), y[j] - foot[j] * np.cos(h[j])]
            end = np.vstack([v[4*j:4*j+3], [*toe_xy, foot_z[j]]])
            if j:
                across = np.array([0.0, thickness, thickness, 0.0])
            else:
                across = np.array([top[j], back[j], rear_foot[j], foot[j]]) - foot[j]
            if j and skirt[j]:
                end = np.vstack([end, v[4*j+3], [*toe_xy, bottom[j]]])
                across = np.r_[across, thickness, 0.0]
            cap_verts.append(end)
            # 小口の横にも UV の幅を取る。帯の V は端の全頂点で同じ。
            cap_uvs.append(np.c_[(end[:, 2] - foot_z[j]) / 2.0, across / 4.0])
        caps = np.arange(len(v), len(v) + 8).reshape(2, 4)
        caps[0] = caps[0, ::-1]
        if skirt[-1]:
            front_foot, back_foot = caps[-1, [3, 2]]
            back_bottom, front_bottom = len(v) + 8, len(v) + 9
            caps = np.vstack([caps, [front_foot, back_foot, back_bottom, front_bottom]])
        mesh = compact(np.vstack([v, *cap_verts]), np.vstack([faces, caps]), np.vstack([uv, *cap_uvs]))
        yield (f"track_fubuki_wall_cap_{k:03d}", *mesh, "concrete")


def _tree_road_frame(sec, trees, s5_start, reach):
    rows = (sec.s >= s5_start + SLOPE_START - 1) & (sec.s <= s5_start + SLOPE_END + 1)
    x, y = trees["x"], trees["y"]
    local = ((x >= sec.x[rows].min() - reach) & (x <= sec.x[rows].max() + reach)
             & (y >= sec.y[rows].min() - reach) & (y <= sec.y[rows].max() + reach))
    frame = terrain.road_frame(sec, x[local], y[local])
    idx = frame["idx"]
    s = sec.s[idx] + frame["along"] - s5_start
    return local, idx, s, -frame["lat"]


def keep_slope_trees(sec, trees, feats, s5_start):
    """草の法面とフェンスのすぐ裏を空け、門前と橋側の谷の植生を残す。"""
    local, idx, s, right = _tree_road_frame(sec, trees, s5_start, 16.0)
    right_walk = features.sidewalk_offsets(sec, feats)[1]
    foot = sec.hw_right[idx] + features.GUTTER_WIDTH + right_walk[idx]
    slope = (((s >= SLOPE_START) & (s < 10405.0)) | ((s >= 10436.0) & (s <= SLOPE_END)))
    outer = crest_offset(s) + np.where((s >= 10496.0) & (s <= 10504.0), 2.0, 0.0)
    keep = np.ones(len(trees), dtype=bool)
    keep[local] = ~(slope & (right >= foot) & (right <= outer))
    return keep


def ground_retained_trees(sec, trees, s5, ground):
    """門と法面の補正を含む右側の範囲で、残す木を新しい地形メッシュに載せる。"""
    local, idx, s, right = _tree_road_frame(sec, trees, s5, 27.0)
    on_right = right > sec.hw_right[idx]
    # 2 m 格子の補間が編集範囲より外に届くため、格子の対角線分も含める。
    gate = (s >= 10402.0) & (s <= 10455.0) & (right <= 20.0)
    bridge = (s >= 10477.0) & (s <= SLOPE_END + 3.0) & (right <= 25.0)
    ids = np.flatnonzero(local)[on_right & (gate | bridge)]
    z = ground(trees.x[ids], trees.y[ids])
    valid = np.isfinite(z)
    trees.z[ids[valid]] = z[valid]
    return int(valid.sum())


def load():
    return json.loads((config.DATA_MANUAL / "s5_fubuki.json").read_text(encoding="utf-8"))


def part_ready(name):
    return landmarks.exists(ASSET, name) and (landmarks.part_dir(ASSET, name) / f"{name}.png").is_file()


def preferred_bridge_part(bridge):
    preferred = bridge.get("preferred_whole")
    if preferred:
        asset, name = preferred["asset"], preferred["name"]
        if landmarks.exists(asset, name) and (landmarks.part_dir(asset, name) / f"{name}.png").is_file():
            return asset, name
    return None


def kamatagawa_parts(feats, bridges, s5):
    """橋の v2 が揃ったときだけ、桁・欄干・歩道を同じ値で拡幅する。"""
    spec = config.DATA_MANUAL / "s45_route70_bridges.json"
    visual_bridges = json.loads(spec.read_text(encoding="utf-8"))["bridges"] if spec.is_file() else []
    visual = next((b for b in visual_bridges if b["section"] == "s5" and b.get("whole") == "red_arch_bridge"), {})
    bridge = next((b for b in bridges if b["osm_id"] == KAMATAGAWA_ID), {})
    width = float(bridge.get("right_extension_m", 0)) if preferred_bridge_part(visual) else 0.0

    out = []
    for f in feats:
        if f.type == "sidewalk" and f.side == "R" and (f.s_from, f.s_to) == (s5 + 10507, s5 + 10607):
            if width > 0:
                out.append(replace(f, width_m=width))
        else:
            out.append(f)
    return out, [dict(b, right_extension_m=width) if b is bridge else b for b in bridges]


def _run_mesh(sec, s0, s1, part, side, rows, offset, z, ground=None, end_post=False, *, stretch_to_ground=False):
    pv, pf, uv = part
    if end_post:
        starts, pitch = [s0 if side < 0 else s1], 2.0
    else:
        n, pitch = dm.r70_rail_layout(s1 - s0, 2.0)
        starts = (s0 + np.arange(n) * pitch if side > 0 else s1 - np.arange(n) * pitch)
    pieces = []
    for start in starts:
        def s_at(x):
            return start + side * x * pitch / 2.0

        v = dm.bend_on_rail(pv, sec, rows, side, offset, z, s_at)
        if stretch_to_ground:
            low, high = pv[:, 2].min(), pv[:, 2].max()
            base = v[:, 2] - pv[:, 2] + low
            terrain_z = np.minimum(ground(v[:, 0], v[:, 1]) - 0.02, base)
            v[:, 2] += (high - pv[:, 2]) / (high - low) * (terrain_z - base)
        elif ground is not None:
            # 土台の幅で高さを変えず、各支柱は鉛直のまま地形の三角形に置く。
            q = dm.bend_on_rail(pv * [1, 0, 0], sec, rows, side, offset, z, s_at)
            v[:, 2] += ground(q[:, 0], q[:, 1])
        pieces.append((v, pf, uv))
    return _merge(pieces)


def slope_fence_meshes(sec, s5, ground):
    """右法面のフェンスを、門の切れ目を空けて地形の上端に置く。"""
    out = []
    for k, run in enumerate(load()["slope_fences"]):
        if not part_ready(run["part"]):
            continue
        start, end = s5 + run["s_from"], s5 + run["s_to"]
        rows = dm.rows_between(sec, start, end)
        offset = crest_offset(sec.s[rows] - s5)
        z = np.zeros(len(rows))
        for name, is_end in ((run["part"], False), (run.get("end"), True)):
            if name and part_ready(name):
                part = landmarks.load(ASSET, name)
                out.append((f"slope_{k}_{name}", name, *_run_mesh(sec, start, end, part, -1,
                                                                rows, offset, z, ground, is_end)))
    return out


def _subdivide_gate_wall(part):
    """長い塀の底も地形に沿うよう、頂点と UV を各辺の中点で補う。"""
    v, faces, uv = part
    for _ in range(2):
        edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1)
        pairs, inverse = np.unique(np.sort(edges, axis=-1).reshape(-1, 2), axis=0, return_inverse=True)
        mid = inverse.reshape(faces.shape) + len(v)
        corners = np.stack([faces, mid, np.roll(mid, 1, axis=1)], axis=-1)
        faces = np.vstack([corners.reshape(-1, 3), mid])
        v = np.vstack([v, v[pairs].mean(axis=1)])
        uv = np.vstack([uv, uv[pairs].mean(axis=1)])
    return v, faces, uv


def gate_meshes(sec, s5, feats, ground):
    """門の開口と分かれ道を空け、歩道より奥に門柱と低い塀を置く。"""
    gate = load()["gate"]
    offset = sec.hw_right + features.GUTTER_WIDTH + features.sidewalk_offsets(sec, feats)[1] + gate["offset_from_walk_m"]
    walk_top = sec.edge_z(-1) + features.CURB_HEIGHT
    stations = s5 + np.asarray(gate["posts_m"])
    out = []
    post = gate["post"]
    if part_ready(post):
        v, faces, uv = landmarks.load(ASSET, post)
        heading = np.unwrap(sec.heading)
        foot = v[:, 2] == v[:, 2].min()
        # 歩道に勾配があるため、高い側の歩道面を両方の柱の基準にする。
        post_z = v[:, 2] - v[:, 2].min() + np.interp(stations, sec.s, walk_top).max()
        for k, s in enumerate(stations):
            h = np.interp(s, sec.s, heading)
            right = np.array([np.sin(h), -np.cos(h)])
            along = np.array([np.cos(h), np.sin(h)])
            center = np.array([np.interp(s, sec.s, sec.x), np.interp(s, sec.s, sec.y)])
            center += right * np.interp(s, sec.s, offset)
            xy = center - v[:, :1] * along + v[:, 1:2] * right
            z = post_z.copy()
            z[foot] = np.minimum(z[foot], ground(*xy[foot].T).min() - 0.02)
            out.append((f"gate_post_{k}", post, np.c_[xy, z], faces, uv))
    wall = gate["wall"]
    if part_ready(wall):
        part = landmarks.load(ASSET, wall)
        for k, (a, b) in enumerate(gate["walls_m"]):
            a, b = a + s5, b + s5
            rows = dm.rows_between(sec, a, b)
            far_side = a >= stations[-1]
            z = walk_top[rows] if far_side else np.full(len(rows), -0.02)
            mesh_part = _subdivide_gate_wall(part) if far_side else part
            out.append((f"gate_wall_{k}", wall, *_run_mesh(sec, a, b, mesh_part, -1, rows, offset[rows],
                                                          z, ground, stretch_to_ground=far_side)))
    return out


def walk_joins(s5):
    return {s5 + s for s in (10480.0, 10507.0, 10607.0)}


def wall_top_meshes(sec, s5, feats):
    """左の擁壁の傾きと高さを使って、白い壁の底を天端に合わせる。"""
    row = load()["wall_top"]
    name = row["part"]
    if not part_ready(name):
        return []

    start, end = s5 + row["s_from"], s5 + row["s_to"]
    part = landmarks.load(ASSET, name)
    metadata = landmarks.part_dir(ASSET, name) / "asset.json"
    if not metadata.is_file():
        metadata = config.PROJECT_ROOT / "assets" / "landmarks" / ASSET / "asset.json"
    spec = json.loads(metadata.read_text(encoding="utf-8"))
    depth = next(part["dimensions_m"][1] for part in spec["parts"] if part["name"] == name)
    walk = features.sidewalk_offsets(sec, feats)[0]
    out = []
    for k, wall in enumerate(feats):
        if wall.type != "retaining_wall" or wall.side != "L" or wall.s_to <= start or wall.s_from >= end:
            continue
        sl = features.rows_of(sec, wall)
        a, b = max(start, sec.s[sl.start]), min(end, sec.s[sl.stop - 1])
        if a >= b:
            continue
        rows = dm.rows_between(sec, a, b)
        offset = sec.hw_left[rows] + features.GUTTER_WIDTH + walk[rows] + features.WALL_LEAN * wall.height_m + depth / 2.0
        z = sec.edge_z(1)[rows] + wall.height_m
        out.append((f"wall_top_{k}", name, *_run_mesh(sec, a, b, part, 1, rows, offset, z)))
    return out
