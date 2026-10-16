import numpy as np

from . import features as ft
from . import meshgen as mg
from . import trees as tree_mod
from .terrain import FAR_HOLE, NEAR_BAND, grid_mesh_chunks

GROUP = "yabitsu_s1"


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
        v, f, uv = mg.compact(*mg.line(sec, sl, np.zeros(i1 - i0 + 1), dash=(5.0, 5.0), row_mask=sec.lanes[sl][:-1] >= 2))
        if len(f):
            m.add_mesh(GROUP, f"LINES_s1_centre_{k:02d}", v, f, uv, "line_white")


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
        else:
            m.add_mesh(GROUP, f"1WALL_s1_retaining_{k:03d}", *ft.retaining_wall_mesh(sec, f), "concrete")


def add_trees(m, trees):
    ids = tree_mod.chunk_ids(trees)
    for k, cid in enumerate(np.unique(ids)):
        m.add_mesh(GROUP, f"TREES_s1_{k:03d}", *tree_mod.cards(trees[ids == cid]), "tree")
