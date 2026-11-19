from dataclasses import replace

import numpy as np

from . import features as ft
from . import meshgen as mg
from . import config, kn5, layouts as layouts_mod
from .crosssection import smoothstep, surface_height
from . import pass_area, small_bridges
from . import start_lot as lot_mod
from .start_lot import _merge
from . import trees as tree_mod
from .terrain import FAR_HOLE, NEAR_BAND, grid_mesh_chunks

GROUP = config.TRACK_GROUP


def _keep_spans(sec, skip):
    """skip（(s0, s1) か、その並び）の外の、帯を作る行の範囲 [(i0, i1), ...]。"""
    if not skip:
        return [(0, sec.n - 1)]
    ranges = [skip] if isinstance(skip[0], (int, float, np.floating)) else list(skip)
    keep = np.ones(sec.n, bool)
    for s0, s1 in ranges:
        keep &= ~((sec.s > s0 + 1e-9) & (sec.s < s1 - 1e-9))
    spans, i = [], 0
    while i < sec.n:
        if keep[i]:
            j = i
            while j + 1 < sec.n and keep[j + 1]:
                j += 1
            if j > i:
                spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


MOUTH_DASH = (1.0, 1.0)   # わき道の入口を横切る路肩の線（破線）の長さと間隔 m（本線の形は 1 m 刻みなので、これより短くできない）


def _dry_runs(sec, i0, i1, dry):
    """行 i0〜i1 の帯を、dry（(s0, s1) の並び：トンネルの中）にかかる所とかからない所の、続いた行の範囲 [(行の始め, 行の終わり, 中か)] に分ける。
    行の組（j, j+1）の真ん中の s で決める。"""
    s = sec.s[i0:i1 + 1]; mid = (s[:-1] + s[1:]) / 2
    inside = np.zeros(len(mid), bool)
    for s0, s1 in dry:
        inside |= (mid >= s0) & (mid <= s1)
    out, j = [], 0
    while j < len(mid):
        e = j
        while e + 1 < len(mid) and inside[e + 1] == inside[j]:
            e += 1
        out.append((i0 + j, i0 + e + 1, bool(inside[j])))
        j = e + 1
    return out


def add_road(m, sec, rows_per_chunk=200, skip=None, mouths=(), no_lines=(), dry=(), paving=(), deck_spans=(), ground_z=None, fit_area=None, dug=None, width_tapers=()):
    """skip の範囲（町の範囲・島で分かれる所：道は多角形で作る）は帯を作らない。skip は (s0, s1) かその並び。
    mouths（(side, s0, s1) の並び）：わき道が接している所。その側は側溝を作らずに路面を続け、路肩の線を破線にする。
    no_lines（(side, s0, s1) の並び）：路肩の白線を引かない所。
    dry（(s0, s1) の並び：トンネルの中）：その所の路面・側溝は名前に _tun を付けた別の部品にし、路面の材質を asphalt_tunnel にする
    （RainFX の水たまり・濡れの材質に入れない）。平らなコンクリート舗装の上では路肩線を 1 cm 上げる。
    舗装区間の最初と最後の断面は元の高さにして、隣の断面との間ですり付ける。"""
    skirt_depth = small_bridges.skirt_depths(sec, deck_spans, ground_z, fit_area)
    if dug is not None or width_tapers:
        from .valleys import fit_foot_chain
        roadside = taper_segments(sec.s, width_tapers)
        for side, hw in ((1,sec.hw_left),(-1,sec.hw_right)):
            lat = side*(hw+mg.GUTTER_WIDTH)
            z = sec.edge_z(side)
            foot = np.c_[sec.x-lat*np.sin(sec.heading),sec.y+lat*np.cos(sec.heading),z-mg.SKIRT_DEPTH]
            lowered = fit_foot_chain(foot, dug, 0., ground=ground_z, roadside=roadside)
            depth = skirt_depth[side].copy()
            fit = (depth == mg.SKIRT_DEPTH) & (lowered < foot[:,2])
            depth[fit] = z[fit]-lowered[fit]
            skirt_depth[side] = depth
    edge_raise = np.zeros(sec.n)
    for patch in paving:
        if patch.kind == "concrete_plain":
            first = np.searchsorted(sec.s, patch.s_from)
            last = np.searchsorted(sec.s, patch.s_to, side="right") - 1
            edge_raise[first + 1:last] = 0.01
    chunks = [(a + i0, a + i1) for a, b in _keep_spans(sec, skip) for i0, i1 in mg.split_rows(b - a + 1, rows_per_chunk)]
    for k, (i0, i1) in enumerate(chunks):
        sl = slice(i0, i1 + 1)
        runs = _dry_runs(sec, i0, i1, dry)
        for wet in (False, True):
            parts = [mg.road_surface(sec, slice(a, b + 1)) for a, b, inside in runs if inside == wet]
            if not parts:
                continue
            v, f, uv = _merge(parts)
            bad = mg.folded_faces(v, f)
            if len(bad):
                raise ValueError(f"路面が折り返しています: s={sec.s[i0]:.0f}〜{sec.s[i1]:.0f}m の部品（{len(bad)} 面）。中心線の補正か道幅を見直してください")
            m.add_mesh(GROUP, f"1ROAD_s1_{k:02d}" + ("_tun" if wet else ""), v, f, uv, "asphalt_tunnel" if wet else "asphalt")
        for side, tag in ((1, "L"), (-1, "R")):
            s = sec.s[sl]
            rows = np.zeros(len(s), bool)
            for sd, s0, s1 in mouths:
                rows |= (sd == side) & (s >= s0) & (s <= s1)
            seg = rows[:-1] & rows[1:]
            for wet in (False, True):
                parts = [mg.gutter(sec, slice(a, b + 1), side, skip=seg[a - i0:b - i0],
                                   skirt_depth=skirt_depth[side][a:b + 1] if deck_spans else None)
                         for a, b, inside in runs if inside == wet]
                if parts:
                    name = f"1ROAD_s1_gutter{tag}_{k:02d}" + ("_tun" if wet else "")
                    v, f, uv = _merge(parts)
                    if np.any(skirt_depth[side][sl] != mg.SKIRT_DEPTH):
                        old, _, _ = _merge([mg.gutter(sec, slice(a, b + 1), side, skip=seg[a - i0:b - i0])
                                            for a, b, inside in runs if inside == wet])
                        normals = kn5._normals(old @ kn5.C.T, kn5._triangles(f))
                        m.add_mesh(GROUP, name, v, f, uv, "concrete", normals=normals)
                    else:
                        m.add_mesh(GROUP, name, v, f, uv, "concrete")
            if seg.any():   # 側溝の代わりに路面をそのまま続ける
                m.add_mesh(GROUP, f"1ROAD_s1_mouth{tag}_{k:02d}", *mg.mouth_strip(sec, sl, side, seg), "asphalt")
            hw = (sec.hw_left if side > 0 else sec.hw_right)[sl]
            bare = np.zeros(len(s), bool)
            for sd, s0, s1 in no_lines:
                bare |= (sd == side) & (s >= s0) & (s <= s1)
            bare = bare[:-1] | bare[1:]
            lift = mg.LINE_LIFT + edge_raise[sl, None] if edge_raise[sl].any() else mg.LINE_LIFT
            if (~seg & ~bare).any():
                m.add_mesh(GROUP, f"LINES_s1_edge{tag}_{k:02d}", *mg.line(sec, sl, side * (hw - 0.25), lift=lift, row_mask=~seg & ~bare), "line_white")
            if (seg & ~bare).any():
                m.add_mesh(GROUP, f"LINES_s1_edge{tag}_mouth_{k:02d}",
                           *mg.line(sec, sl, side * (hw - 0.25), lift=lift, dash=MOUTH_DASH, row_mask=seg & ~bare), "line_white")


CENTRE_STYLES = {"white_dashed": ("line_white", (5.0, 5.0)), "white_solid": ("line_white", None),
                 "orange_solid": ("line_orange", None), "orange_flanked": ("line_orange", None),
                 "orange_median": ("line_orange", None)}
MEDIAN_W, MEDIAN_H = 0.6, 0.12   # orange_median：中央の盛り上がった帯（宮ヶ瀬湖ぞい）の幅・高さ。両側にオレンジの実線
CENTRE_LIFT = 0.008   # 中央線は路面の色（赤い滑り止めなど、LIFT 0.005）より上に出す。同じ高さだと重なる所でちらつく（菜の花台のヘアピン、2026-10-30）
FLANK_OFFSET, FLANK_DASH = 0.9, (2.0, 2.0)   # orange_flanked：オレンジの実線の両側の、白い破線（減速を促す線）


def _cut_rows(rows, gaps):
    """表の行（s_from〜s_to）から、gaps の範囲を切り取る。"""
    out = []
    for r in rows:
        pieces = [(float(r["s_from"]), float(r["s_to"]))]
        for g0, g1 in gaps:
            nxt = []
            for a, b in pieces:
                if b <= g0 or a >= g1:
                    nxt.append((a, b))
                else:
                    if a < g0:
                        nxt.append((a, g0))
                    if b > g1:
                        nxt.append((g1, b))
            pieces = nxt
        out += [{**r, "s_from": a, "s_to": b} for a, b in pieces]
    return out


def add_centrelines(m, sec, rows, s_min=0.0, gaps=()):
    """中央線の表（区間ごとの {区間}_centrelines.csv を通しの距離にしたもの）の範囲にだけ、中央線を引く。
    s_min より手前（町の範囲）と gaps（島で分かれる所）は引かない（多角形の路面の表示で描く）。"""
    rows = _cut_rows([{**r, "taper_start": float(r["s_from"])} for r in rows], list(gaps))
    for k, r in enumerate(rows):
        r = {**r, "s_from": max(float(r["s_from"]), s_min)}
        if float(r["s_from"]) >= float(r["s_to"]):
            continue
        style = str(r["style"]).strip()
        if style not in CENTRE_STYLES:
            raise ValueError(f"中央線の style は {list(CENTRE_STYLES)} のどれか: {r}")
        mat, dash = CENTRE_STYLES[style]
        i0 = int(np.searchsorted(sec.s, float(r["s_from"])))
        i1 = int(np.searchsorted(sec.s, float(r["s_to"]), side="right")) - 1
        if i1 - i0 < 1:
            continue
        sl = slice(i0, i1 + 1)
        if style == "orange_median":   # 盛り上がった帯（上面と両側の面）と、その両側のオレンジの実線
            taper = float(r.get("taper_m") or 0)
            if taper and i0 and sec.s[i0] > r["s_from"] and r["s_from"] == r["taper_start"]:
                sl = slice(i0 - 1, i1 + 1)
            station = sec.s[sl]
            profile = (smoothstep((station - r["taper_start"]) / taper)
                       if taper else np.ones(len(station)))
            half = MEDIAN_W / 2 * profile
            # 舗装上の落ち葉シートからも離す。鼻は幅0から立ち上がる。
            height = MEDIAN_H * profile + (.015 * (1 - profile) if taper else 0)
            # 先端では重なる頂点でできる三角形を除く。
            top = mg.line(sec, sl, np.zeros(len(station)), width=half * 2, lift=height)
            zc = lambda sgn: surface_height(sgn * half, sec.z[sl], sec.slope[sl], sec.crown[sl])
            sides = [mg.wall_ribbon(sec, sl, sgn, half, half, zc(sgn) - 0.02 * profile, zc(sgn) + height)
                     for sgn in (1, -1)]   # 縦の縁（外向き）
            v, f, uv = _merge([top, *sides])
            if taper:
                f = kn5._triangles(f)
                a, b, c = np.moveaxis(v[f], 1, 0)
                f = f[np.any(np.cross(b - a, c - a) != 0, axis=1)]
            m.add_mesh(GROUP, f"track_centre_{k:03d}_median", *mg.compact(v, f, uv), "concrete")
            # 先端では隣り合う塗線の内縁を合わせ、中央の白い帯に沿って開く。
            offset = np.where(profile == 1, MEDIAN_W / 2 + .12, 0.075 + (MEDIAN_W / 2 + .045) * profile)
            parts = [mg.compact(*mg.line(sec, sl, sgn * offset))
                     for sgn in (1, -1)]
            m.add_mesh(GROUP, f"LINES_track_centre_{k:03d}", *_merge(parts), mat)
            continue
        v, f, uv = mg.compact(*mg.line(sec, sl, np.zeros(i1 - i0 + 1), dash=dash, lift=CENTRE_LIFT))
        if len(f):
            m.add_mesh(GROUP, f"LINES_track_centre_{k:03d}", v, f, uv, mat)
        if style == "orange_flanked":
            parts = [mg.compact(*mg.line(sec, sl, np.full(i1 - i0 + 1, sgn * FLANK_OFFSET), dash=FLANK_DASH)) for sgn in (1, -1)]
            parts = [q for q in parts if len(q[1])]
            if parts:
                m.add_mesh(GROUP, f"LINES_track_centre_{k:03d}_flank", *_merge(parts), "line_white")


def add_bay_lines(m, sec, bays, dash=(3.0, 3.0)):
    """バス停の切り込みと車線の境の白い破線（広げる前の道の端の位置＝外側線の位置）。"""
    from .crosssection import apply_bays
    for k, b in enumerate(bays):
        if b["depth_m"] <= 0:   # 細くなる所（島の横など）には線を引かない
            continue
        add_l, add_r = apply_bays(sec.s, np.zeros(sec.n), np.zeros(sec.n), [b])
        i0 = int(np.searchsorted(sec.s, b["s_from"])); i1 = int(np.searchsorted(sec.s, b["s_to"], side="right")) - 1
        if i1 - i0 < 2:
            continue
        sl = slice(i0, i1 + 1)
        if b["side"] == "L":
            lat = (sec.hw_left - add_l)[sl] - 0.25
        else:
            lat = -((sec.hw_right - add_r)[sl] - 0.25)
        v, f, uv = mg.compact(*mg.line(sec, sl, lat, dash=dash))
        if len(f):
            m.add_mesh(GROUP, f"LINES_track_bay_{k:02d}", v, f, uv, "line_white")


def far_photo_uv(v, far):
    """遠い地形の長方形（far の xs・ys の端）を 0〜1 にする uv（空中写真を 1 枚貼る。v=0 が南＝画像の下）。"""
    x0, x1, y0, y1 = far["xs"][0], far["xs"][-1], far["ys"][0], far["ys"][-1]
    return np.c_[(v[:, 0] - x0) / (x1 - x0), (v[:, 1] - y0) / (y1 - y0)]


def add_valley_chunk(m, name, v, f, uv, near):
    """Only the old quads replaced by the fine surface disappear from this chunk."""
    from .valley_surface import keep_quads
    retain = keep_quads(v, f, near)
    if retain.all():
        m.add_mesh(GROUP, name, v, f, uv, "ground")
    else:
        normals = kn5._normals(v @ kn5.C.T, kn5._triangles(f))
        if "valley_coarse_remainder" in near:
            keys = np.c_[np.searchsorted(near["ys"], v[f[~retain, 0], 1]),
                         np.searchsorted(near["xs"], v[f[~retain, 0], 0])]
            rem = near["valley_coarse_remainder"]
            owners = {tuple(k): i for i, k in enumerate(keys)}
            select = np.array([i for i, row in enumerate(rem) if tuple(row[:2]) in owners], dtype=int)
            if len(select):
                from .valley_surface import copy_attributes

                old_faces = kn5._triangles(f[~retain])
                keys = rem[select]
                owner = np.array([owners[tuple(k[:2])] + len(owners) * k[2] for k in keys])
                vv = near["valley_vertices"][near["valley_faces"][select]]
                vu, vn = copy_attributes(v[old_faces[owner]], vv[..., :2],
                                         uv[old_faces[owner]], normals[old_faces[owner]])
                first = len(v)
                m.add_mesh(GROUP, name, np.r_[v, vv.reshape(-1, 3)],
                           np.r_[kn5._triangles(f[retain]), first + np.arange(vv.size // 3).reshape(-1, 3)],
                           np.r_[uv, vu.reshape(-1, 2)], "ground",
                           normals=np.r_[normals, vn.reshape(-1, 3)], removed_faces=old_faces)
                return
        if retain.any():
            m.add_mesh(GROUP, name, v, f[retain], uv, "ground", normals=normals, removed_faces=f[~retain])


def add_terrain(m, near, far, mid=None, far_photo=False, *, before_near=None):
    D = near["D"]
    keep = (D[:-1, :-1] <= NEAR_BAND) & (D[:-1, 1:] <= NEAR_BAND) & (D[1:, 1:] <= NEAR_BAND) & (D[1:, :-1] <= NEAR_BAND)
    if "H" in near:   # トンネルの筒の中に入る面は描かない（terrain.tunnel_holes）
        keep &= ~near["H"]
    original = near if before_near is None else before_near
    for k, (v, f, uv) in enumerate(grid_mesh_chunks(near["xs"], near["ys"], original["Z"], keep)):
        name = f"1GRASS_s1_terrain_{k:03d}"
        if "valley_cells" in near:
            add_valley_chunk(m, name, v, f, uv, near)
            continue
        if before_near is not None:
            c = np.searchsorted(near["xs"], v[:, 0])
            r = np.searchsorted(near["ys"], v[:, 1])
            v[:, 2] = near["Z"][r, c]
            normals = kn5.retain_grid_normals(v, f, before_near)
            if normals is not None:
                m.add_mesh(GROUP, name, v, f, uv, "ground", normals=normals)
                continue
        m.add_mesh(GROUP, name, v, f, uv, "ground")
    if "valley_faces" in near:
        from .valley_surface import mesh_parts
        for name, v, f, uv, mat, normals in mesh_parts(near):
            m.add_mesh(GROUP, name, v, f, uv, mat, normals=normals)
    if mid is not None:   # 道から 200m まで：4m 格子
        from .terrain import mid_keep
        for k, (v, f, uv) in enumerate(grid_mesh_chunks(mid["xs"], mid["ys"], mid["Z"], mid_keep(mid["D"]), block=80, uv_scale=8.0)):
            m.add_mesh(GROUP, f"1GRASS_track_mid_{k:03d}", v, f, uv, "ground")
    for k, (v, f, uv) in enumerate(grid_mesh_chunks(far["xs"], far["ys"], far["Z"], far["Dc"] >= FAR_HOLE, block=50, uv_scale=40.0)):
        if far_photo:     # 遠い地形は空中写真（菜の花台などから見下ろす街・畑が、くり返しの地面の絵にならない）
            m.add_mesh(GROUP, f"TERRAIN_FAR_s1_{k:02d}", v, f, far_photo_uv(v, far), "ground_far")
        else:
            m.add_mesh(GROUP, f"TERRAIN_FAR_s1_{k:02d}", v, f, uv, "ground")


def taper_segments(s, width_tapers):
    selected = np.zeros(len(s) - 1, bool)
    for a, b, _, _ in width_tapers:
        selected |= (s[:-1] < b) & (s[1:] > a)
    return selected


def add_features(m, sec, feats, right_walk_joins=(), bridge_spans=None, ground_z=None, dug=None, width_tapers=(),
                 *, graded_shoulder=False):
    off = dict(zip(("L", "R"), ft.sidewalk_offsets(sec, feats)))  # 壁・生け垣・欄干は歩道の外側に立てる
    shelf = ft.shelf_widths(sec, feats)
    bridge_spans = bridge_spans or {}
    c_retaining_join = None
    for k, f in enumerate(feats):
        if f.type == "guardrail":
            join = small_bridges.right_rail_join(sec, f, bridge_spans)
            line = replace(f, s_from=sec.s[join]) if join is not None and sec.s[join] < f.s_from else f
            lateral = 0.
            if line is not f and small_bridges.tapered_rail_join(sec, join):
                from .dress_mountain import guardrail_join_offset
                lateral = guardrail_join_offset(sec.s[ft.rows_of(sec, line)], sec.s[join])
            vis, col = ft.guardrail_meshes(sec, line, lateral_adjustment=lateral)
            gaps = small_bridges.rail_gaps(sec, f, bridge_spans)
            vis = small_bridges.cut_guardrail(sec, vis, gaps)
            col = small_bridges.cut_guardrail(sec, col, gaps)
            if len(vis[1]):
                m.add_mesh(GROUP, f"GUARDRAIL_s1_{k:03d}", *vis, "guardrail")
                m.add_mesh(GROUP, f"1WALL_s1_rail_{k:03d}", *col, "guardrail", renderable=False)
            if k == small_bridges.RIVER_RAIL and 16 in bridge_spans and ground_z is not None:
                add_meshes(m, small_bridges.rail_support_meshes(sec, f, bridge_spans[16], ground_z))
        elif f.type == "parapet":
            m.add_mesh(GROUP, f"1WALL_track_parapet_{k:03d}", *ft.parapet_mesh(sec, f, off[f.side]), "concrete")
        elif f.type == "fence":  # 透けて見える柵（欄干と同じ形に、棒の絵を貼る）
            f2 = ft.Feature(f.s_from, f.s_to, f.side, f.type, f.height_m or ft.FENCE_HEIGHT, f.note, f.width_m)
            m.add_mesh(GROUP, f"1WALL_track_fence_{k:03d}", *ft.parapet_mesh(sec, f2, off[f.side]), "railing_bars")
        elif f.type == "sidewalk":
            top, sides = ft.sidewalk_mesh(sec, f, join_previous=f.side == "R" and f.s_from in right_walk_joins)
            m.add_mesh(GROUP, f"1ROAD_track_sidewalk_{k:03d}", *top, f.material or "concrete")   # 行の material（黒っぽい舗装の路肩は asphalt）
            m.add_mesh(GROUP, f"track_sidewalk_{k:03d}_curb", *sides, "concrete")
        elif f.type == "hedge":
            m.add_mesh(GROUP, f"1WALL_track_hedge_{k:03d}", *ft.hedge_mesh(sec, f, off[f.side]), "hedge")
        elif f.type == "overpass":  # OSM に無い、道の上を渡る橋（OSM にある橋は add_site_meshes で overpasses から）
            deck, piers = ft.overpass_meshes(sec, f)
            m.add_mesh(GROUP, f"track_overpass_{k:03d}_deck", *deck, "concrete")
            m.add_mesh(GROUP, f"1WALL_track_overpass_{k:03d}_piers", *piers, "concrete")
        else:
            gaps = small_bridges.wall_gaps(sec, f, bridge_spans)
            wall_ground = ground_z if k in small_bridges.RIVER_CUT_WALLS and 16 in bridge_spans else None
            roadside = crest = None
            if width_tapers and k in (367, 370) and 25 in bridge_spans:
                station = sec.s[ft.rows_of(sec, f)]
                roadside = taper_segments(station, width_tapers)
                crest = np.zeros_like(station)
                for a, b, initial, final in width_tapers:
                    m_taper = (station >= a) & (station < b)
                    crest[m_taper] = .5 * (final - initial) * (1 - smoothstep((station[m_taper] - a) / (b - a)))
            wall = small_bridges.cut_guardrail(sec, ft.retaining_wall_mesh(sec, f, off[f.side], wall_ground,
                                                                           crest_offset=crest), gaps)
            normals = None
            if wall_ground is None and (dug is not None or roadside is not None):
                from .valleys import fit_foot_chain
                v, faces, _ = wall
                foot = v[::2]
                new_z = fit_foot_chain(foot, dug, .10, ground=ground_z, roadside=roadside)
                if np.any(foot[:,2] != new_z):
                    normals = kn5._normals(v @ kn5.C.T, kn5._triangles(faces))
                    foot[:,2] = new_z
            kw = {"normals":normals} if normals is not None else {}
            m.add_mesh(GROUP, f"1WALL_s1_retaining_{k:03d}", *wall, "concrete", **kw)
            if k == 367 and 25 in bridge_spans and f.side == "L" and graded_shoulder:
                c_retaining_join = (ft.rows_of(sec, f).start, wall[0][:2])
            if k == 367 and 25 in bridge_spans and f.side == "L" and not graded_shoulder:
                m.add_mesh(GROUP, "smallspan_25_wall_367_gravel", *small_bridges.wall_367_gravel(sec, f, wall[0] if crest is not None else None),
                           "gravel_grey")
            if shelf[ft.SIDES[f.side]][ft.rows_of(sec, f)].max() > 0.01:   # 壁を道から離して立てた所：手前の草地（行の material があればその材質、砂利の路肩など）
                wall_shelf = small_bridges.cut_guardrail(sec, ft.shelf_mesh(sec, f, off[f.side]), gaps)
                m.add_mesh(GROUP, f"1GRASS_track_shelf_{k:03d}", *wall_shelf, f.material or "ground")

    spec = small_bridges.load()["bridges"] if bridge_spans else []
    for k, (i, j) in bridge_spans.items():
        for side, label in (("L", "left"), ("R", "right")):
            neighbors = [f for f in feats if f.side == side and f.type in ("guardrail", "retaining_wall")
                         and f.s_from <= sec.s[j + 1] and sec.s[i - 1] <= f.s_to]
            face = small_bridges.rail_face(spec[k][label])
            m.add_mesh(GROUP, small_bridges.wall_name(k, label),
                       *small_bridges.collision_mesh(sec, i, j, ft.SIDES[side], neighbors,
                                                     face, off[side], join_after=(k == 25 and side == "R"),
                                                     retaining_after=c_retaining_join if k == 25 and side == "L" else None),
                       "guardrail", renderable=False)


def add_trees(m, trees, *, heights=None, remove=None):
    ids = tree_mod.chunk_ids(trees)
    for k, cid in enumerate(np.unique(ids)):
        sel = ids == cid
        v, f, uv = tree_mod.cards(trees[sel])
        name = f"TREES_s1_{k:03d}"
        if heights is not None:
            moved = np.isfinite(heights[sel])
            cards = v.reshape(-1, 8, 3)
            cards[moved, :, 2] += (heights[sel][moved] - cards[moved, 0, 2])[:, None]
        if remove is not None and remove[sel].any():
            f = f[~np.repeat(remove[sel], 2)]
            if not len(f):
                continue
            v, f, uv = mg.compact(v, f, uv)
        m.add_mesh(GROUP, name, v, f, uv, "tree")


def add_pads(m, pads):
    for k, p in enumerate(pads):
        if p.get("kind") in ("lot", "pass", "site", "stub", "branch", "motorway"):  # それぞれの部品で作る（地形合わせだけの土台）
            continue
        m.add_mesh(GROUP, f"1ROAD_s1_pad_{k:02d}", *mg.pad_mesh(p["polygon"], p["z"]), p.get("material", "asphalt"))   # 砂利の広場は gravel_grey（仕上げで貼り替えない）


def add_start_lot(m, spec, pads, sec=None):
    for p in pads:
        if p.get("kind") == "lot":
            for name, v, f, uv, mat, rend in lot_mod.meshes(spec, p, sec):
                m.add_mesh(GROUP, name, v, f, uv, mat, renderable=rend)


def add_meshes(m, items, group=GROUP):
    """(名前, 頂点, 面, UV, 材質, 描画するか) の並びをそのまま足す（分かれ道・高架橋・町の部品など）。"""
    for name, v, f, uv, mat, rend in items:
        m.add_mesh(group, name, v, f, uv, mat, renderable=rend)


def add_pass_area(m, site):
    for name, v, f, uv, mat, rend in site.meshes():
        m.add_mesh(GROUP, name, v, f, uv, mat, renderable=rend)


def add_layout_objects(m, sec, pads, layouts, bounds):
    for layout in layouts:
        for d in layouts_mod.layout_objects(sec, pads, layout, bounds):
            m.add_empty(f"{layout.name}_objects", d)


def road_end_closure(sec, s_end):
    """道の終わり（S4 の終わり。S5 はまだ）を、道幅＋側溝の全部で横切る見えない壁と、その 2m 手前の工事中の柵。"""
    import math
    from .crosssection import GUTTER_WIDTH
    from . import town_objects
    out = []
    for s, kind in ((s_end - 1.0, "wall"), (s_end - 3.0, "barricade")):
        i = int(np.argmin(np.abs(sec.s - s))); h = sec.heading[i]
        n = np.array([-math.sin(h), math.cos(h)]); c = np.array([sec.x[i], sec.y[i]])
        a = c - n * (sec.hw_right[i] + GUTTER_WIDTH + 1.5); b = c + n * (sec.hw_left[i] + GUTTER_WIDTH + 1.5)
        if kind == "wall":
            z = float(sec.z[i])
            v = np.array([[*a, z - 1.0], [*b, z - 1.0], [*b, z + 4.0], [*a, z + 4.0]], float)
            out.append(("1WALL_road_end_block", v, np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.int32), v[:, :2] / 2.0, "concrete", False))
        else:
            d = np.array([math.cos(h), math.sin(h)])
            out += town_objects.barricade_meshes(a + n * 1.2, b - n * 1.2, lambda x, y: np.full(np.shape(x), float(sec.z[i])), -d, k=900)
    return out
