import numpy as np

from . import meshgen as mg

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
