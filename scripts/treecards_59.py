"""依頼59のFPS測定用。保存済みの木と舗装から板のKN5とCSP一覧を作る。"""
import argparse
import configparser
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from yabitsu import csptree, forest_light, kn5
from yabitsu.config import assert_writable


WORK = Path("export/compare/route70/work")
ATLAS_NAME = "treecards_59.dds"
ATLAS_SIZE = 2048
IMAGE_SIZE = 640
SPECIES = (
    "sugi_a", "sugi_b", "sugi_c", "hinoki_a", "hinoki_b",
    "broadleaf_a", "broadleaf_b", "broadleaf_c", "broadleaf_d",
)
SPECIES_ID = {name.encode(): i for i, name in enumerate(SPECIES)}
CELL_SIZE_M = 250
VERTICES_PER_TREE = 8
MAX_TREES_PER_MESH = 65535 // VERTICES_PER_TREE


def raster_tree(textures, submeshes):
    """正規化した木を+Z側から正射影し、葉の抜きを残して幹と葉を重ねる。"""
    rgba = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
    depth = np.full(rgba.shape[:2], -np.inf)
    last = IMAGE_SIZE - 1
    for mesh in submeshes:
        texture = textures[mesh.diffuse]
        h, w = texture.shape[:2]
        x, y, z = mesh.verts["p"].astype(float).T
        px = (x + 1) * last / 2
        py = (1 - y) * last / 1.02
        uv = mesh.verts["uv"]
        for a, b, c in mesh.indices.reshape(-1, 3):
            p, q, r = px[[a, b, c]]
            s, t, u = py[[a, b, c]]
            det = (q - p) * (u - s) - (r - p) * (t - s)
            if abs(det) < 1e-8:
                continue
            x0, x1 = max(0, int(np.ceil(min(p, q, r)))), min(last, int(np.floor(max(p, q, r))))
            y0, y1 = max(0, int(np.ceil(min(s, t, u)))), min(last, int(np.floor(max(s, t, u))))
            if x0 > x1 or y0 > y1:
                continue
            gy, gx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
            f = ((gx - p) * (u - s) - (gy - s) * (r - p)) / det
            g = ((gy - s) * (q - p) - (gx - p) * (t - s)) / det
            frame = np.s_[y0:y1 + 1, x0:x1 + 1]
            z_at = z[a] + f * (z[b] - z[a]) + g * (z[c] - z[a])
            dy, dx = np.nonzero((f >= 0) & (g >= 0) & (f + g <= 1) & (z_at > depth[frame]))
            coord = uv[a] + f[dy, dx, None] * (uv[b] - uv[a]) + g[dy, dx, None] * (uv[c] - uv[a])
            col = np.floor(coord[:, 0] * w).astype(int) % w
            row = np.floor((1 - coord[:, 1]) * h).astype(int) % h
            color = texture[row, col]
            solid = color[:, 3] >= 255 * kn5.ALPHA_REF
            dy, dx = dy[solid], dx[solid]
            depth[frame][dy, dx] = z_at[dy, dx]
            rgba[frame][dy, dx, :3] = color[solid, :3]
            rgba[frame][dy, dx, 3] = 255
    return Image.fromarray(rgba)


def make_atlas(models):
    atlas = Image.new("RGBA", (ATLAS_SIZE, ATLAS_SIZE))
    sizes = []
    uv_rects = []
    for i, name in enumerate(SPECIES):
        with zipfile.ZipFile(models / f"{name}.zip") as package:
            ini = configparser.ConfigParser()
            ini.read_string(package.read("tree.ini").decode("utf-8"))
            sizes.append([ini["BASIC"].getfloat("HEIGHT"), ini["BASIC"].getfloat("WIDTH")])
            names, meshes = csptree.read_mesh(package.read("LOD0.mesh"))
            textures = {
                m.diffuse: np.array(Image.open(io.BytesIO(package.read(names[m.diffuse][1]))).convert("RGBA"))
                for m in meshes
            }
        x, y = (i % 3) * 672 + 16, (i // 3) * 672 + 16
        image = raster_tree(textures, meshes)
        atlas.paste(image, (x, y))
        uv_rects.append(np.array([x + 0.5, y + 0.5, x + IMAGE_SIZE - 0.5, y + IMAGE_SIZE - 0.5]) / ATLAS_SIZE)
    return kn5.dds_bytes(atlas), np.array(sizes), np.array(uv_rects)


def split_lists(source, output, edges, settings):
    """一覧の元バイトを保持しつつ、座標・回転・倍率・樹種番号を集める。"""
    tiles = sorted(source.glob("forest_*.txt"))
    if not tiles:
        raise ValueError(f"木の一覧がありません: {source}")
    pattern = forest_light.tall_pattern(settings)
    all_rows, far_rows = [], []
    no_tall = output / "lists_no_tall"
    near = output / "lists_tall_near15"
    no_tall.mkdir(exist_ok=True)
    near.mkdir(exist_ok=True)
    for path in tiles:
        lines, indices, names, values = forest_light.read_tile(path, pattern)
        rows = np.column_stack((values, [SPECIES_ID[name] for name in names]))
        is_near = forest_light.near_mask(rows[:, (0, 2)] * [1, -1], edges, settings.near_edge_m)
        (no_tall / path.name).write_bytes(forest_light.keep_other_lines(lines, indices))
        (near / path.name).write_bytes(forest_light.keep_other_lines(lines, indices[~is_near]))
        all_rows.append(rows)
        far_rows.append(rows[~is_near])
    return np.concatenate(all_rows), np.concatenate(far_rows), len(tiles)


def grid_chunks(rows):
    grid = np.floor(rows[:, (0, 2)] / CELL_SIZE_M).astype(int)
    order = np.lexsort((grid[:, 1], grid[:, 0]))
    sorted_grid = grid[order]
    cuts = np.flatnonzero(np.any(sorted_grid[1:] != sorted_grid[:-1], axis=1)) + 1
    for cell in np.split(order, cuts):
        if len(cell) == 0:
            continue
        gx, gz = grid[cell[0]]
        for part, start in enumerate(range(0, len(cell), MAX_TREES_PER_MESH)):
            yield f"TREECARD_{gx}_{gz}_{part:02d}", cell[start:start + MAX_TREES_PER_MESH]


def card_geometry(rows, sizes, uv_rects):
    xyz = rows[:, :3]
    angle = np.deg2rad(rows[:, 3])
    kind = rows[:, 5].astype(int)
    height, width = (sizes[kind] * rows[:, 4, None]).T
    cos, sin = np.cos(angle), np.sin(angle)
    axis = np.column_stack((cos, np.zeros(len(rows)), -sin))
    cross = np.column_stack((-sin, np.zeros(len(rows)), -cos))
    axes = np.stack((axis, cross), axis=1)
    offsets = axes[:, :, None, :] * (width[:, None, None, None] / 2) * np.array([-1, 1, 1, -1])[None, None, :, None]
    pos = xyz[:, None, None, :] + offsets
    pos[:, :, :2, 1] -= 0.3
    pos[:, :, 2:, 1] += height[:, None, None]

    u0, v0, u1, v1 = uv_rects[kind].T
    quad_uv = np.stack((u0, v1, u1, v1, u1, v0, u0, v0), axis=1).reshape(-1, 4, 2)
    uv = np.repeat(quad_uv[:, None], 2, axis=1)
    normals = np.broadcast_to([0.0, 1.0, 0.0], pos.shape)  # 表裏で同じ明るさ
    tangents = np.broadcast_to(axes[:, :, None, :], pos.shape)
    front = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.uint16)
    both = np.concatenate((front, front[:, ::-1]))
    tris = both + (np.arange(len(rows), dtype=np.uint16) * VERTICES_PER_TREE)[:, None, None]
    return pos.reshape(-1, 3), normals.reshape(-1, 3), uv.reshape(-1, 2), tangents.reshape(-1, 3), tris.reshape(-1, 3)


def write_kn5(path, rows, sizes, uv_rects, atlas):
    groups = list(grid_chunks(rows))
    temp = path.with_suffix(".kn5.tmp")
    with temp.open("wb") as stream:
        writer = kn5._Out(stream)
        writer.raw(b"sc6969")
        writer.pack("iii", kn5.VERSION, 1, 1)  # version、テクスチャ数、有効
        writer.string(ATLAS_NAME)
        writer.pack("i", len(atlas))
        writer.raw(atlas)

        writer.pack("i", 1)
        writer.string("yabitsu_treecards_59")
        writer.string("ksPerPixelAT")
        writer.pack("B?i", 0, True, 0)
        props = {**kn5.MATERIAL_PROPS, "ksAlphaRef": kn5.ALPHA_REF, "ksSpecular": 0.0}
        writer.pack("i", len(props))
        for name, value in props.items():
            writer.string(name)
            writer.pack("10f", value, *([0.0] * 9))
        writer.pack("i", 1)
        writer.string("txDiffuse")
        writer.pack("i", 0)
        writer.string(ATLAS_NAME)

        writer.pack("i", kn5.NODE_BASE)
        writer.string(path.stem)
        writer.pack("i?", len(groups), True)
        writer.pack("16f", *np.eye(4).ravel())
        for name, indices in groups:
            geometry = card_geometry(rows[indices], sizes, uv_rects)
            kn5._write_mesh(writer, name, *geometry, 0, True, cast_shadows=False)
    temp.replace(path)
    return len(groups)


def build(source, models, manifest, output, *, settings=None):
    output = assert_writable(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = settings if settings is not None else forest_light.load_settings()
    atlas, sizes, uv_rects = make_atlas(models)
    all_rows, far_rows, tiles = split_lists(source, output, forest_light.road_edges(manifest), settings)
    (output / ATLAS_NAME).write_bytes(atlas)
    print(f"{tiles} タイル、高木 {len(all_rows):,} 本、道ぎわ {len(all_rows) - len(far_rows):,} 本", flush=True)
    for variant, rows in (("all", all_rows), ("far", far_rows)):
        path = output / f"yabitsu_treecards_{variant}.kn5"
        parts = write_kn5(path, rows, sizes, uv_rects, atlas)
        print(f"{path.name}: {parts:,} 部品、{8 * len(rows):,} 三角形、{path.stat().st_size:,} バイト", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORK / "forest_57_full/extension/trees")
    parser.add_argument("--models", type=Path, default=WORK / "supervisor_41.fmEVx6/deliver/forest/extension/trees")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/track/manifest.json"))
    parser.add_argument("--output", "--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="既存の出力先への書き込みを許可")
    args = parser.parse_args()
    output = assert_writable(args.output)
    try:
        output.mkdir(parents=True, exist_ok=args.force)
    except FileExistsError:
        parser.error(f"{output} は既にあります。上書きするには --force を指定してください")
    build(args.source, args.models, args.manifest, output)


if __name__ == "__main__":
    main()
