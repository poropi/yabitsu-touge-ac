import numpy as np

from . import features as ft
from . import meshgen as mg
from . import config, layouts as layouts_mod
from . import pass_area
from . import start_lot as lot_mod
from . import trees as tree_mod
from .terrain import FAR_HOLE, NEAR_BAND, grid_mesh_chunks

GROUP = config.TRACK_GROUP


def add_road(m, sec, rows_per_chunk=200):
    for k, (i0, i1) in enumerate(mg.split_rows(sec.n, rows_per_chunk)):
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
                 "orange_solid": ("line_orange", None)}


def add_centrelines(m, sec, rows):
    """中央線の表（区間ごとの {区間}_centrelines.csv を通しの距離にしたもの）の範囲にだけ、中央線を引く。"""
    for k, r in enumerate(rows):
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


def add_terrain(m, near, far):
    D = near["D"]
    keep = (D[:-1, :-1] <= NEAR_BAND) & (D[:-1, 1:] <= NEAR_BAND) & (D[1:, 1:] <= NEAR_BAND) & (D[1:, :-1] <= NEAR_BAND)
    for k, (v, f, uv) in enumerate(grid_mesh_chunks(near["xs"], near["ys"], near["Z"], keep)):
        m.add_mesh(GROUP, f"1GRASS_s1_terrain_{k:03d}", v, f, uv, "ground")
    for k, (v, f, uv) in enumerate(grid_mesh_chunks(far["xs"], far["ys"], far["Z"], far["Dc"] >= FAR_HOLE, block=50, uv_scale=40.0)):
        m.add_mesh(GROUP, f"TERRAIN_FAR_s1_{k:02d}", v, f, uv, "ground")


def add_features(m, sec, feats):
    for k, f in enumerate(feats):
        if f.type == "guardrail":
            vis, col = ft.guardrail_meshes(sec, f)
            m.add_mesh(GROUP, f"GUARDRAIL_s1_{k:03d}", *vis, "guardrail")
            m.add_mesh(GROUP, f"1WALL_s1_rail_{k:03d}", *col, "guardrail", renderable=False)
        elif f.type == "parapet":
            m.add_mesh(GROUP, f"1WALL_track_parapet_{k:03d}", *ft.parapet_mesh(sec, f), "concrete")
        else:
            m.add_mesh(GROUP, f"1WALL_s1_retaining_{k:03d}", *ft.retaining_wall_mesh(sec, f), "concrete")


def add_trees(m, trees):
    ids = tree_mod.chunk_ids(trees)
    for k, cid in enumerate(np.unique(ids)):
        m.add_mesh(GROUP, f"TREES_s1_{k:03d}", *tree_mod.cards(trees[ids == cid]), "tree")


def add_pads(m, pads):
    for k, p in enumerate(pads):
        if p.get("kind") in ("lot", "pass", "site"):  # 駐車場・峠・蓑毛は add_start_lot / add_pass_area で作る
            continue
        m.add_mesh(GROUP, f"1ROAD_s1_pad_{k:02d}", *mg.pad_mesh(p["polygon"], p["z"]), "asphalt")


def add_start_lot(m, spec, pads):
    for p in pads:
        if p.get("kind") == "lot":
            for name, v, f, uv, mat, rend in lot_mod.meshes(spec, p):
                m.add_mesh(GROUP, name, v, f, uv, mat, renderable=rend)


def add_pass_area(m, site):
    for name, v, f, uv, mat, rend in site.meshes():
        m.add_mesh(GROUP, name, v, f, uv, mat, renderable=rend)


def add_layout_objects(m, sec, pads, layouts, bounds):
    for layout in layouts:
        for d in layouts_mod.layout_objects(sec, pads, layout, bounds):
            m.add_empty(f"{layout.name}_objects", d)
