import numpy as np

from . import features as ft
from . import meshgen as mg
from . import config, layouts as layouts_mod
from . import pass_area
from . import start_lot as lot_mod
from . import trees as tree_mod
from .terrain import FAR_HOLE, NEAR_BAND, grid_mesh_chunks

GROUP = config.TRACK_GROUP


def add_road(m, sec, rows_per_chunk=200, skip=None):
    """skip=(s0, s1) の範囲（町の範囲：道は town の多角形で作る）は帯を作らない。今は s0 = 0 だけを扱う。"""
    first = int(np.searchsorted(sec.s, skip[1] - 1e-9)) if skip else 0
    for k, (i0, i1) in enumerate(mg.split_rows(sec.n - first, rows_per_chunk)):
        i0, i1 = i0 + first, i1 + first
        sl = slice(i0, i1 + 1)
        v, f, uv = mg.road_surface(sec, sl)
        bad = mg.folded_faces(v, f)
        if len(bad):
            s_bad = sec.s[i0 + bad[0] // 4]
            raise ValueError(f"路面が折り返しています: s={s_bad:.0f}m 付近（{len(bad)} 面）。中心線の補正か道幅を見直してください")
        m.add_mesh(GROUP, f"1ROAD_s1_{k:02d}", v, f, uv, "asphalt")
        for side, tag in ((1, "L"), (-1, "R")):
            m.add_mesh(GROUP, f"1ROAD_s1_gutter{tag}_{k:02d}", *mg.gutter(sec, sl, side), "concrete")
            hw = (sec.hw_left if side > 0 else sec.hw_right)[sl]
            m.add_mesh(GROUP, f"LINES_s1_edge{tag}_{k:02d}", *mg.line(sec, sl, side * (hw - 0.25)), "line_white")


CENTRE_STYLES = {"white_dashed": ("line_white", (5.0, 5.0)), "white_solid": ("line_white", None),
                 "orange_solid": ("line_orange", None), "orange_flanked": ("line_orange", None)}
FLANK_OFFSET, FLANK_DASH = 0.9, (2.0, 2.0)   # orange_flanked：オレンジの実線の両側の、白い破線（減速を促す線）


def add_centrelines(m, sec, rows, s_min=0.0):
    """中央線の表（区間ごとの {区間}_centrelines.csv を通しの距離にしたもの）の範囲にだけ、中央線を引く。
    s_min より手前（町の範囲）は引かない（町の路面の表示で描く）。"""
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
        v, f, uv = mg.compact(*mg.line(sec, sl, np.zeros(i1 - i0 + 1), dash=dash))
        if len(f):
            m.add_mesh(GROUP, f"LINES_track_centre_{k:03d}", v, f, uv, mat)
        if style == "orange_flanked":
            parts = [mg.compact(*mg.line(sec, sl, np.full(i1 - i0 + 1, sgn * FLANK_OFFSET), dash=FLANK_DASH)) for sgn in (1, -1)]
            parts = [q for q in parts if len(q[1])]
            if parts:
                from .start_lot import _merge
                m.add_mesh(GROUP, f"LINES_track_centre_{k:03d}_flank", *_merge(parts), "line_white")


def add_terrain(m, near, far):
    D = near["D"]
    keep = (D[:-1, :-1] <= NEAR_BAND) & (D[:-1, 1:] <= NEAR_BAND) & (D[1:, 1:] <= NEAR_BAND) & (D[1:, :-1] <= NEAR_BAND)
    for k, (v, f, uv) in enumerate(grid_mesh_chunks(near["xs"], near["ys"], near["Z"], keep)):
        m.add_mesh(GROUP, f"1GRASS_s1_terrain_{k:03d}", v, f, uv, "ground")
    for k, (v, f, uv) in enumerate(grid_mesh_chunks(far["xs"], far["ys"], far["Z"], far["Dc"] >= FAR_HOLE, block=50, uv_scale=40.0)):
        m.add_mesh(GROUP, f"TERRAIN_FAR_s1_{k:02d}", v, f, uv, "ground")


def add_features(m, sec, feats):
    off = dict(zip(("L", "R"), ft.sidewalk_offsets(sec, feats)))  # 壁・生け垣・欄干は歩道の外側に立てる
    for k, f in enumerate(feats):
        if f.type == "guardrail":
            vis, col = ft.guardrail_meshes(sec, f)
            m.add_mesh(GROUP, f"GUARDRAIL_s1_{k:03d}", *vis, "guardrail")
            m.add_mesh(GROUP, f"1WALL_s1_rail_{k:03d}", *col, "guardrail", renderable=False)
        elif f.type == "parapet":
            m.add_mesh(GROUP, f"1WALL_track_parapet_{k:03d}", *ft.parapet_mesh(sec, f, off[f.side]), "concrete")
        elif f.type == "fence":  # 透けて見える柵（欄干と同じ形に、棒の絵を貼る）
            f2 = ft.Feature(f.s_from, f.s_to, f.side, f.type, f.height_m or ft.FENCE_HEIGHT, f.note, f.width_m)
            m.add_mesh(GROUP, f"1WALL_track_fence_{k:03d}", *ft.parapet_mesh(sec, f2, off[f.side]), "railing_bars")
        elif f.type == "sidewalk":
            top, sides = ft.sidewalk_mesh(sec, f)
            m.add_mesh(GROUP, f"1ROAD_track_sidewalk_{k:03d}", *top, "concrete")
            m.add_mesh(GROUP, f"track_sidewalk_{k:03d}_curb", *sides, "concrete")
        elif f.type == "hedge":
            m.add_mesh(GROUP, f"1WALL_track_hedge_{k:03d}", *ft.hedge_mesh(sec, f, off[f.side]), "hedge")
        elif f.type == "overpass":  # OSM に無い、道の上を渡る橋（OSM にある橋は add_site_meshes で overpasses から）
            deck, piers = ft.overpass_meshes(sec, f)
            m.add_mesh(GROUP, f"track_overpass_{k:03d}_deck", *deck, "concrete")
            m.add_mesh(GROUP, f"1WALL_track_overpass_{k:03d}_piers", *piers, "concrete")
        else:
            m.add_mesh(GROUP, f"1WALL_s1_retaining_{k:03d}", *ft.retaining_wall_mesh(sec, f, off[f.side]), "concrete")


def add_trees(m, trees):
    ids = tree_mod.chunk_ids(trees)
    for k, cid in enumerate(np.unique(ids)):
        m.add_mesh(GROUP, f"TREES_s1_{k:03d}", *tree_mod.cards(trees[ids == cid]), "tree")


def add_pads(m, pads):
    for k, p in enumerate(pads):
        if p.get("kind") in ("lot", "pass", "site", "stub"):  # 駐車場・峠・蓑毛・分かれ道は、それぞれの部品で作る
            continue
        m.add_mesh(GROUP, f"1ROAD_s1_pad_{k:02d}", *mg.pad_mesh(p["polygon"], p["z"]), "asphalt")


def add_start_lot(m, spec, pads):
    for p in pads:
        if p.get("kind") == "lot":
            for name, v, f, uv, mat, rend in lot_mod.meshes(spec, p):
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
