"""橋タグのない県道70号の小橋。欄干の端を道路断面の行にそろえる。"""
import json

import numpy as np
from scipy.spatial import cKDTree

from . import config, dress_mountain, features, kn5, landmarks, meshgen
from .crosssection import GUTTER_DEPTH, GUTTER_WIDTH, SKIRT_DEPTH

RAIL_OFFSET = 0.37  # 舗装の縁から側溝側。最も幅広い square_09 も側溝の内側に収まる。
TAPER_M = 1.5
SUPPORT_SPANS = (2, 16, 25)
RIVER_RAIL = 281
RIVER_CUT_WALLS = (279, 280, 395)
DECK_DEPTH = 0.60
DECK_TOP_DROP = GUTTER_DEPTH + 0.045


def load():
    path = config.DATA_MANUAL / "s45_route70_bridges.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"bridges": [], "tunnels": []}


def spans(sec, bounds, spec):
    """割り当て表の行番号を保ち、明示区間だけ既存の側溝と端点を共有する。"""
    return {
        k: tuple(int(np.argmin(np.abs(sec.s - (bounds[b["section"]][0] + b[key]))))
                 for key in ("s_from", "s_to"))
        for k, b in enumerate(spec["bridges"]) if "s_from" in b and not b.get("skip")
    }


def rail_gaps(sec, feature, bridge_spans):
    return [(i, j) for i, j in bridge_spans.values()
            if sec.s[i] <= feature.s_to and feature.s_from <= sec.s[j]]


def wall_gaps(sec, feature, bridge_spans):
    # 旧表の S4 6351 m 右擁壁用。現行表は右 W ビームなので rail_gaps で切る。
    gap = bridge_spans.get(16)
    if gap and feature.side == "R" and feature.s_from <= sec.s[gap[0]] < sec.s[gap[1]] <= feature.s_to:
        return [gap]
    return []


def right_rail_join(sec, feature, bridge_spans):
    """S5 の右。橋の終端から後続の W ビームの最初の行までは道路断面が 1 行離れる。"""
    span = bridge_spans.get(25)
    if span is None or feature.side != "R":
        return None
    j = span[1]
    rows = features.rows_of(sec, feature)
    return j if rows.stop == j + 1 or rows.start == j + 1 else None


def tapered_rail_join(sec, row):
    """すりつけ後の連続区間では、つなぎに終端の外曲げを作らない。"""
    return abs(sec.hw_right[row + 1] - sec.hw_right[row]) <= .15 * (sec.s[row + 1] - sec.s[row])


def wall_name(k, label):
    return f"1WALL_track_smallspan_{k:02d}_{label}"


def cut_guardrail(sec, mesh, gaps, *, foot_surface=None):
    """既存の面を橋端の道路断面で切り、元の外側の頂点・UV・面を保つ。"""
    for i, j in gaps:
        v, f, uv = mesh
        if not len(f):
            break
        distance = []
        for row, direction in ((i, -1), (j, 1)):
            origin = [sec.x[row], sec.y[row]]
            axis = np.array([np.cos(sec.heading[row]), np.sin(sec.heading[row])])
            d = direction * ((v[:, :2] - origin) @ axis)
            d[np.abs(d) < 1e-8] = 0
            distance.append(d)
        if any(np.all(d >= 0) for d in distance):
            continue
        tris = kn5._triangles(f)
        outside = np.any([np.all(d[tris] >= 0, axis=1) for d in distance], axis=0)
        if outside.all():
            continue
        candidates = tris[~outside]
        vertices_used, inv = np.unique(candidates.ravel(), return_inverse=True)
        nearest = cKDTree(np.c_[sec.x, sec.y]).query(v[vertices_used, :2])[1]
        near = ((nearest >= i - 2) & (nearest <= j + 2))[inv].reshape(-1, 3)
        # 遠い頂点が 2 枚の無限平面の内側に投影されても、その面は切らない。
        in_gap = (distance[0][candidates] < 0) & (distance[1][candidates] < 0)
        local = near.any(axis=1) & (~in_gap | near).all(axis=1)
        if not local.any():
            continue
        keep = outside.copy()
        keep[~outside] = ~local
        faces = list(tris[keep])
        vertices, uvs = list(v), list(uv)
        for d in distance:
            crossing = (d[tris].min(axis=1) < 0) & (d[tris].max(axis=1) > 0)
            intersections = {}
            crossed = tris[crossing & ~keep]
            fitted = set()
            if foot_surface is not None and len(crossed):
                ids = np.unique(crossed)
                target = foot_surface.height(v[ids, :2]) - .12
                fitted = set(ids[np.isclose(v[ids, 2], target, atol=1e-9, rtol=0)])
            for tri in crossed:
                polygon = []
                for a, b in zip(tri, np.roll(tri, -1)):
                    if d[a] >= 0:
                        polygon.append(a)
                    if d[a] * d[b] < 0:
                        edge = (min(a, b), max(a, b))
                        if edge not in intersections:
                            t = d[a] / (d[a] - d[b])
                            intersections[edge] = len(vertices)
                            point = v[a] + t * (v[b] - v[a])
                            if a in fitted and b in fitted:
                                # 掘削面に沿った支柱底を切った場合、その新しいXYで実面に再度根入れする。
                                z = foot_surface.height(point[:2])
                                if np.isfinite(z):
                                    point[2] = z - .12
                            vertices.append(point)
                            uvs.append(uv[a] + t * (uv[b] - uv[a]))
                        polygon.append(intersections[edge])
                faces.extend((polygon[0], polygon[k], polygon[k + 1]) for k in range(1, len(polygon) - 1))
        mesh = meshgen.compact(np.asarray(vertices), np.asarray(faces, int).reshape(-1, 3), np.asarray(uvs))
    return mesh


def collision_mesh(sec, i, j, side, neighbors, face_offset, walk=None, join_after=False, *,
                   retaining_after=None):
    """欄干の道側の面に立て、両端を隣の衝突面から縦に 1.5 m かけて寄せる。"""
    hw = sec.hw_left if side > 0 else sec.hw_right
    edge_z = sec.edge_z(side)
    ends = []
    for row, step in ((i, -1), (j, 1)):
        bottom = top = GUTTER_WIDTH
        dz = 0.0
        adj = row + step
        for feature in neighbors:
            sl = features.rows_of(sec, feature)
            if not sl.start <= adj < sl.stop:
                continue
            if feature.type == "guardrail":
                if not (sl.start <= row < sl.stop or (step > 0 and join_after)):
                    continue
                bottom = top = GUTTER_WIDTH + features.GUARDRAIL_OFFSET + feature.width_m
                if step > 0 and join_after and tapered_rail_join(sec, j):
                    bottom = top = bottom + dress_mountain.guardrail_join_offset(sec.s[j], sec.s[j])
                dz = features.CURB_HEIGHT if feature.width_m > 0 else 0.0
                break
            if not sl.start <= row < sl.stop:
                row = adj
            bottom = GUTTER_WIDTH + (walk[row] if walk is not None else 0.0)
            # 上端を壁面の延長に置き、両方に高さがある所の断面を一致させる。
            top = bottom + (features.WALL_LEAN * feature.height_m * 1.5
                            / (feature.height_m + features.WALL_BASE_DROP))
            break
        ends.append((row, bottom, top, dz))
    if retaining_after is not None:
        stop, wall_edge = retaining_after
        ends[1] = (j, face_offset, face_offset, 0.)
    else:
        stop = ends[1][0]
    sl = slice(ends[0][0], stop + 1)
    s = sec.s[sl]
    bottom = np.full(len(s), face_offset)
    top = bottom.copy()
    height = np.zeros(len(s))
    for row, low, high, dz in ends:
        weight = np.clip(1.0 - abs(s - sec.s[row]) / TAPER_M, 0, 1)
        bottom += weight * (low - face_offset)
        top += weight * (high - face_offset)
        height += weight * dz
    z = edge_z[sl] + height
    v, f, uv = meshgen.wall_ribbon(sec, sl, side, hw[sl] + bottom, hw[sl] + top, z - 0.3, z + 1.2)
    if retaining_after is not None:
        # Connect to the actual collision face, including its fitted foot and lean.
        foot, crest = wall_edge
        fraction = (edge_z[stop] + 1.2 - foot[2]) / (crest[2] - foot[2])
        joint = np.stack((foot, foot + np.clip(fraction, 0, 1) * (crest - foot)))
        offset = j - sl.start
        t = (s[offset:] - sec.s[j]) / (sec.s[stop] - sec.s[j])
        remaining = v[2 * offset:].reshape(-1, 2, 3)
        remaining += t[:, None, None] * (joint - remaining[-1])
    return v, f, uv


def rail_mesh(sec, i, j, side, part, panel_m):
    """欄干の線の弧長を等分する。1 mm 未満の誤差で枚数が変わるのを防ぐ。"""
    rows = np.arange(i, j + 1)
    hw = sec.hw_left if side > 0 else sec.hw_right
    lat = hw[rows] + RAIL_OFFSET
    h = sec.heading[rows]
    ze = sec.edge_z(side)[rows]
    line = np.c_[sec.x[rows] - side * lat * np.sin(h),
                 sec.y[rows] + side * lat * np.cos(h), ze]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    n = max(1, int(np.floor(round(float(arc[-1]), 3) / panel_m + 0.5)))
    panel_length = arc[-1] / n
    pv, pf, uv = part
    verts, faces = [], []
    for k in range(n):
        distance = (k + 0.5 + pv[:, 0] / panel_m) * panel_length
        distance = distance if side > 0 else arc[-1] - distance
        s_at = np.interp(distance, arc, sec.s[rows])
        verts.append(dress_mountain.bend_on_rail(pv, sec, rows, side, lat, ze, lambda _: s_at))
        faces.append(pf + k * len(pv))
    return np.concatenate(verts), np.concatenate(faces), np.tile(uv, (n, 1))


def rail_face(part):
    return RAIL_OFFSET + float(landmarks.load("r70_bridges", part)[0][:, 1].min())


def w_beam_join(sec, j, feature, part):
    """368 番の本体は再配列せず、始端と橋の間に波板の断面を 1 行分延ばす。"""
    v, f, uv = part
    # +4 m まで通る波板だけを使う。支柱・金具・小口は延ばさない。
    f = f[(np.ptp(v[f, 0], axis=1) >= dress_mountain.R70_RAIL_SPAN - 1e-6)]
    v, f, uv = meshgen.compact(v, f, uv)
    _, side, lat, ze = dress_mountain.guardrail_line(
        sec, feature, joined_row=j + 1 if tapered_rail_join(sec, j) else None)
    rows = np.array([j, j + 1])
    near = sec.hw_right[j] + GUTTER_WIDTH + features.GUARDRAIL_OFFSET + feature.width_m
    if tapered_rail_join(sec, j):
        near += dress_mountain.guardrail_join_offset(sec.s[j], sec.s[j])
    start, stop = sec.s[rows]
    s_at = lambda x: stop - (stop - start) * x / dress_mountain.R70_RAIL_SPAN
    v = dress_mountain.bend_on_rail(v, sec, rows, side, np.array([near, lat[0]]),
                                    np.array([sec.edge_z(side)[j], ze[0]]), s_at)
    return v, f, uv


def w_beam_cut_post(sec, row, feature, ground_z=None, setback=0.0, *, outset=0.0, align_to_rail=False):
    side = features.SIDES[feature.side]
    hw = (sec.hw_left if side > 0 else sec.hw_right)[row]
    lat = side * (hw + GUTTER_WIDTH + features.GUARDRAIL_OFFSET + feature.width_m - 0.02 + outset)
    h = sec.heading[row]
    # 谷側の盛土より下まで入れる。波板に掛かる上端は道路縁から 0.825 m。
    sink = 1.5
    foot = [sec.x[row] - np.sin(h) * lat + setback * np.cos(h),
            sec.y[row] + np.cos(h) * lat + setback * np.sin(h),
            sec.edge_z(side)[row] - sink + (features.CURB_HEIGHT if feature.width_m > 0 else 0)]
    phase = h - side * np.pi / 2 if align_to_rail else 0.
    v, f, uv = dress_mountain.cylinder(foot, dress_mountain.POST_R, sink + 0.825, phase=phase)
    if ground_z is not None:
        low = v[:, 2] == foot[2]
        v[low, 2] = min(foot[2], np.min(ground_z(*v[low, :2].T)) - 0.15)
    return v, f, uv


def sill_mesh(sec, i, j, side, *, supported=False):
    """側溝の外縁と同じ高さで根元を受ける。側溝の底へ差し込む寸法。"""
    sl = slice(i, j + 1)
    hw = (sec.hw_left if side > 0 else sec.hw_right)[sl]
    ze = sec.edge_z(side)[sl]
    outer = GUTTER_WIDTH + 0.03  # 側溝のスカートと同一平面を避け、幅広の親柱も受ける。
    lat = side * (hw[:, None] + [0.12, 0.12, outer, outer])
    z = ze[:, None] + [-0.15, 0.0, 0.0, -0.15]
    v, f, uv = meshgen.ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, z, sec.s[sl],
                              u_scale=1.0, v_scale=1.0, flip=side < 0)
    caps = np.array([[0, 1, 2, 3], [len(v) - 1, len(v) - 2, len(v) - 3, len(v) - 4]])
    if supported:
        return v, f, uv  # 端は橋台の面で閉じる。台座と同じ平面の蓋は重ねない。
    return v, np.concatenate([f, caps if side > 0 else caps[:, ::-1]]), uv


def support_meshes(sec, bridge_spans, ground_z, *, graded_shoulder=False):
    """写真を参考にした床版と袖付き橋台。奥行きと厚さは見当。"""
    for k in SUPPORT_SPANS:
        if k not in bridge_spans:
            continue
        i, j = bridge_spans[k]
        sl = slice(i, j + 1)
        left, right = sec.hw_left[sl] + GUTTER_WIDTH, -sec.hw_right[sl] - GUTTER_WIDTH
        zl, zr = sec.edge_z(1)[sl], sec.edge_z(-1)[sl]
        lat = np.column_stack((right, left, left, right, right))
        top = np.column_stack((zr, zl, zl, zr, zr)) - DECK_TOP_DROP
        top[:, 2:4] -= DECK_DEPTH
        v, f, uv = meshgen.ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, top, sec.s[sl])
        # 上は既存の路面・側溝、端は橋台で閉じる。側面と下面を張る。
        yield f"smallspan_{k:02d}_deck", v, f.reshape(-1, 4, 4)[:, 1:].reshape(-1, 4), uv / 2, "valley_soffit", True
        for row, tag, direction in ((i, "start", -1), (j, "end", 1)):
            h = sec.heading[row]
            axis = np.array([np.cos(h), np.sin(h)])
            normal = np.array([-np.sin(h), np.cos(h)])
            a = -sec.hw_right[row] - GUTTER_WIDTH - 0.04
            b = sec.hw_left[row] + GUTTER_WIDTH + 0.04
            width = np.linspace(a, b, int(np.ceil((b - a) / 0.10)) + 1)
            edge_l, edge_r = sec.edge_z(1)[row], sec.edge_z(-1)[row]
            lhw, rhw = sec.hw_left[row], sec.hw_right[row]
            profile_lat = [a, -rhw - GUTTER_WIDTH, -rhw - GUTTER_WIDTH / 2, -rhw,
                           0., lhw, lhw + GUTTER_WIDTH / 2, lhw + GUTTER_WIDTH, b]
            profile_z = [edge_r, edge_r, edge_r - GUTTER_DEPTH, edge_r, sec.z[row],
                         edge_l, edge_l - GUTTER_DEPTH, edge_l, edge_l]
            lateral = np.unique(np.r_[width, profile_lat])
            xy = np.array([sec.x[row], sec.y[row]]) + lateral[:, None] * normal
            upper = np.interp(lateral, profile_lat, profile_z) - .008
            lower = min(np.min(ground_z(*xy.T)) - .65, upper.min() - DECK_DEPTH)
            v = np.r_[np.c_[xy, upper], np.c_[xy, np.full(len(xy), lower)]]
            n = len(xy)
            p = np.arange(n - 1)
            f = np.column_stack((p, p + 1, p + n + 1, p + n))
            uv = np.c_[np.tile(lateral, 2), v[:, 2]] / 2
            yield (f"smallspan_{k:02d}_abutment_{tag}", np.r_[v, v],
                   np.r_[f, f[:, ::-1] + len(v)], np.r_[uv, uv], "concrete_moss", True)
            for side, label, hw in ((1, "left", sec.hw_left), (-1, "right", sec.hw_right)):
                if k == 16 and side < 0:
                    continue  # 藤熊川側の袖はレールの土留めが続く。
                yield _approach_wall(sec, row, direction, side, hw, ground_z,
                                     f"smallspan_{k:02d}_approach_{tag}_{label}", graded_shoulder=graded_shoulder)
        if k == 16:
            from .old_river_bridge import meshes as old_bridge_meshes
            yield from old_bridge_meshes(sec, (i, j), ground_z)


def _approach_wall(sec, row, direction, side, hw, ground_z, name, *, graded_shoulder=False):
    """橋台から既存の盛土へ戻るまで、側溝の下に袖を通す。"""
    end = sec.s[row] + direction * 4.
    stop = np.clip(np.searchsorted(sec.s, end), 0, sec.n - 1)
    ids = np.arange(min(row, stop), max(row, stop) + 1)
    x, y, h = sec.x[ids], sec.y[ids], sec.heading[ids]
    profile = [-.06, .43, .43, -.06, -.06]
    upper = sec.edge_z(side)[ids] - .045
    if graded_shoulder and name in ("smallspan_02_approach_start_left", "smallspan_02_approach_end_left",
                                     "smallspan_25_approach_start_left"):
        # The new ground carries the shoulder. Keep the wing below the drain
        # instead of drawing a second horizontal surface through the fill.
        profile = [-.06, .080, .080, -.06, -.06]
        upper += .025
    if name == "smallspan_25_approach_end_right" or (graded_shoulder and name == "smallspan_25_approach_end_left"):
        # 橋台で受けた天端を、道路縁に沿って絞りながら盛土側で下げる。
        from .crosssection import smoothstep
        merge = smoothstep((sec.s[ids] - sec.s[row]) / 4.)
        cap = .057 - .017 * merge
        inner = np.full_like(cap, -.06)
        profile = np.column_stack((inner, cap, cap, inner, inner))
        upper -= .10 * merge
    lat = side * (hw[ids, None] + GUTTER_WIDTH + np.array(profile))
    gx = x[:, None] - lat * np.sin(h[:, None])
    gy = y[:, None] + lat * np.cos(h[:, None])
    floor = ground_z(gx, gy).min(axis=1)
    middle = ground_z((gx[1:] + gx[:-1]) / 2, (gy[1:] + gy[:-1]) / 2).min(axis=1)
    base = np.minimum(floor, np.minimum(np.r_[middle, np.inf], np.r_[np.inf, middle])) - .65
    base = np.minimum(base, upper - .25)
    z = np.stack((upper, upper, base, base, upper), axis=1)
    v, f, uv = meshgen.ribbon(x, y, h, lat, z, sec.s[ids], flip=side < 0)
    return name, v, f, uv / 2, "concrete_moss", True


def fit_sites(sec, spans):
    """袖壁・基礎を置く3地点の境界。形状を変える面は境界を越えない。"""
    import shapely
    from . import track, valleys

    if not any(k in spans for k in SUPPORT_SPANS):
        return shapely.GeometryCollection()
    bounds = track.load_bounds()
    keys = dict.fromkeys((v.section, v.center_m) for v in valleys.load())
    return shapely.union_all([valleys.site_area(sec, bounds, section, center) for section, center in keys])


def sink_foot(verts, bottom, ground, area):
    """支柱の上を保ち、部品全体が調整範囲にあるときだけ底面を地中へ入れる。"""
    import shapely

    feet = verts[bottom]
    if not len(feet) or area.is_empty:
        return
    if shapely.contains_xy(area, *verts[:, :2].T).all():
        z = np.min(ground(*feet[:, :2].T))
        if np.isfinite(z):
            verts[bottom, 2] = np.minimum(feet[:, 2], z - .12)


def skirt_depths(sec, deck_spans, ground_z=None, fit_area=None):
    """橋の区間だけ、床版の上面までスカートを短くする。"""
    depth = np.full(sec.n, SKIRT_DEPTH)
    for i, j in deck_spans:
        depth[i:j + 1] = DECK_TOP_DROP
    return {1: depth, -1: depth}


def rail_support_meshes(sec, feature, gap, ground_z):
    """藤熊川側の支柱の足元を側溝と一体の土留めで受ける。橋の下は開ける。"""
    rows, _, lat, z = dress_mountain.guardrail_line(sec, feature)
    for tag, keep in (("before", rows <= gap[0]), ("after", rows >= gap[1])):
        ids = np.flatnonzero(keep)
        # 端の鋼管（半径57mm）の底まで受ける。橋台側は橋端の断面で止める。
        ids = np.r_[ids[0], ids] if tag == "before" else np.r_[ids, ids[-1]]
        ds = np.zeros(len(ids))
        ds[0 if tag == "before" else -1] = -.08 if tag == "before" else .08
        rr = rows[ids]
        inner = -(sec.hw_right[rr] + GUTTER_WIDTH - .01)
        outer = -(lat[ids] + .46)
        ztop = z[ids] - .02
        direction = sec.heading[rr]
        cx = sec.x[rr] + ds * np.cos(direction)
        cy = sec.y[rr] + ds * np.sin(direction)
        nx, ny = -np.sin(direction), np.cos(direction)
        crown = ztop - .035
        # 石張りの護岸を谷へ傾ける。最小断面は地面のすぐ上にある低い縁石。
        initial_drop = crown - ground_z(cx + nx*outer, cy + ny*outer)
        toe = outer - np.clip(.38 * (initial_drop + .45), .15, 2.70)
        stations = sec.s[rr] + ds
        distances = np.column_stack((inner, outer, toe))
        x = cx[:, None] + nx[:, None] * distances
        y = cy[:, None] + ny[:, None] * distances
        ground = ground_z(x, y).min(axis=1)
        middle = ground_z((x[1:] + x[:-1])/2, (y[1:] + y[:-1])/2).min(axis=1)
        base = np.minimum(ground, np.minimum(np.r_[middle, np.inf], np.r_[np.inf, middle]))
        bottom = np.minimum(base - .50, crown - .28)
        # 天端・背・底・谷向きの石積みは稜線で頂点を分ける。縦断方向の
        # 隣り合う面だけが法線を共有し、天端が丸く光らない。
        bank_lat = np.column_stack((outer, inner, inner, outer, toe, outer))
        bank_z = np.column_stack((crown, ztop, bottom, bottom, bottom, crown))
        vertices, faces, uvs = [], [], []
        for band in range(bank_lat.shape[1]-1):
            cols = slice(band, band + 2)
            v, f, _ = meshgen.ribbon(cx, cy, direction, bank_lat[:, cols], bank_z[:, cols], stations)
            faces.append(f + sum(map(len, vertices)))
            vertices.append(v)
            if band == 0:
                uvs.append(v[:, :2] / .70)  # 天端の小割り石。側面の大きな目地とは向きを分ける。
            else:
                uvs.append(np.c_[np.repeat(stations-stations[0], 2), v[:, 2]-crown[0]] / 1.8)
        # 橋に向かない端だけ閉じ、橋台の端面と重ねない。
        end = 0 if tag == "before" else -1
        indices = [0, 1, 2, 4] if tag == "before" else [4, 2, 1, 0]
        lateral = bank_lat[end, indices]
        cap = np.c_[cx[end]+nx[end]*lateral, cy[end]+ny[end]*lateral, bank_z[end, indices]]
        faces.append(np.arange(4)[None, :] + sum(map(len, vertices)))
        vertices.append(cap)
        uvs.append(np.c_[lateral, cap[:, 2]-crown[0]] / 1.8)
        yield (f"smallspan_16_rail_support_{tag}", np.concatenate(vertices), np.concatenate(faces),
               np.concatenate(uvs), "kenchi_shade", True)


def wall_367_gravel(sec, feature, wall=None):
    """左側溝にたまった砂利の肩。車道の外で、擁壁の斜面の足元に差し込む。"""
    sl = features.rows_of(sec, feature)
    station = sec.s[sl]
    hw = sec.hw_left[sl]
    ze = sec.edge_z(1)[sl]
    s = station - station[0]
    lip = .22 + .035 * np.sin(1.4*s) + .02 * np.sin(.38*s)
    lat = hw[:,None] + np.column_stack((lip, np.full(len(s), .46),
                                       np.full(len(s), GUTTER_WIDTH + .11)))
    gutter = -GUTTER_DEPTH * (1. - abs(lip - GUTTER_WIDTH/2)/(GUTTER_WIDTH/2))
    z = ze[:,None] + np.column_stack((gutter + .015, -.012 + .008*np.sin(s), .035 + .008*np.sin(s)))
    if wall is not None:
        bottom, top = wall[::2], wall[1::2]
        fraction = (z[:, -1] - bottom[:, 2]) / (top[:, 2] - bottom[:, 2])
        contact = bottom[:, :2] + fraction[:, None] * (top[:, :2] - bottom[:, :2])
        delta = contact - np.c_[sec.x[sl], sec.y[sl]]
        lat[:, -1] = np.sum(delta * np.c_[-np.sin(sec.heading[sl]), np.cos(sec.heading[sl])], axis=1) + .015
    return meshgen.ribbon(sec.x[sl], sec.y[sl], sec.heading[sl], lat, z, station, u_scale=.95, v_scale=.95)



def add_valley_materials(m):
    """小橋専用の絵を登録。既設の橋や林床の絵は元の名前で使える。"""
    from shutil import copyfile
    from PIL import Image
    from . import textures

    folder = m.root / "textures"
    folder.mkdir(exist_ok=True)
    board = textures._board_concrete().astype(float)
    stains = textures._concrete_moss().astype(float)
    soffit = np.asarray(.17*board + .40*stains, np.uint8)
    Image.fromarray(soffit).save(folder/"valley_soffit.png")
    rng = np.random.default_rng(6354)
    oxides = textures.periodic_noise(256,12,rng)
    grain = textures.periodic_noise(256,1,rng)
    rust = np.array([37.,29.,25.]) + oxides[:,:,None]*[61.,32.,14.] + (grain[:,:,None]-.5)*17
    Image.fromarray(rust.astype(np.uint8)).save(folder/"valley_rusted_steel.png")
    copyfile(config.PROJECT_ROOT/"assets/dress/forest_floor.png", folder/"valley_forest_floor.png")
    for name in ("valley_soffit","valley_rusted_steel","valley_forest_floor"):
        m.add_material(name,texture=f"textures/{name}.png")
