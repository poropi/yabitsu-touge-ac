import math

import numpy as np
from PIL import Image

from yabitsu import config
from yabitsu.acobjects import hotlap_start, pit_boxes, start_grid, timing_gate
from yabitsu.manifest import Manifest

ROOT = config.DATA_PROCESSED / "test"


def grid(x0, x1, y0, y1, z=0.0, nx=1, ny=20):
    X, Y = np.meshgrid(np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1))
    v = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
    i = np.arange(X.size).reshape(X.shape)
    f = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1).reshape(-1, 4)
    return v, f, v[:, :2] / 4.0


def wall(x, y0, y1, h, ny=8):
    ys = np.linspace(y0, y1, ny + 1)
    v = np.array([[x, y, z] for y in ys for z in (0.0, h)])
    i = np.arange(len(v)).reshape(-1, 2)
    f = np.stack([i[:-1, 0], i[:-1, 1], i[1:, 1], i[1:, 0]], -1)  # 面は -X（道路側）を向く
    return v, f, np.c_[v[:, 1] / 4.0, v[:, 2] / 4.0]


def box(cx, cy, size):
    h = size / 2
    v = np.array([[cx + dx, cy + dy, z] for z in (0, size) for dx, dy in ((-h, -h), (h, -h), (h, h), (-h, h))], float)
    f = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    return v, f, v[:, :2] / size


def main():
    (ROOT / "textures").mkdir(parents=True, exist_ok=True)
    m = Manifest(ROOT)
    for name, rgb in {"asphalt": (70, 70, 72), "grass": (60, 100, 40), "concrete": (160, 160, 155),
                      "marker_green": (0, 200, 0), "marker_red": (220, 0, 0)}.items():
        Image.new("RGB", (64, 64), rgb).save(ROOT / "textures" / f"{name}.png")
        m.add_material(name, texture=f"textures/{name}.png", two_sided=name.startswith("marker"))
    g = "yabitsu_test"
    m.add_mesh(g, "1ROAD_test", *grid(-3, 3, 0, 200, nx=2), "asphalt")
    m.add_mesh(g, "1GRASS_test_L", *grid(-23, -3, 0, 200, nx=4), "grass")
    m.add_mesh(g, "1GRASS_test_R", *grid(3, 23, 0, 200, nx=4), "grass")
    m.add_mesh(g, "1WALL_test_hidden", *wall(5.0, 60, 140, 1.2), "concrete", renderable=False)
    vw, fw, uw = wall(-5.0, 60, 140, 1.2)
    m.add_mesh(g, "1WALL_test_visible", vw, fw[:, ::-1], uw, "concrete")  # 西側の壁は +X を向ける
    m.add_mesh(g, "MARKER_NORTH", *box(0, 230, 4), "marker_green")
    m.add_mesh(g, "MARKER_EAST", *box(30, 100, 4), "marker_red")
    m.add_mesh(g, "MARKER_EAST_NEAR", *box(10, 25, 3), "marker_red")  # ピットのすぐ右前（見落とし防止）
    s = np.arange(0.0, 201.0)
    path = {"s": s, "x": np.zeros_like(s), "y": s, "z": np.zeros_like(s), "heading": np.full_like(s, math.pi / 2)}
    o = "test_objects"
    for d in [*start_grid(path, 40, 1, n=2), hotlap_start(path, 25, 1), *timing_gate(path, 50, 1, 4.0, "START"),
              *timing_gate(path, 180, 1, 4.0, "FINISH"), *pit_boxes((0.0, 15.0), 0.0, math.pi / 2, n=2)]:
        m.add_empty(o, d)
    print(m.save()); print(m.summary())


if __name__ == "__main__":
    main()
