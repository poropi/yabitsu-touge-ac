import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yabitsu import csptree, kn5


WORK = Path("export/compare/route70/work")
OUTPUT = WORK / "treecards_59"
SOURCE = WORK / "forest_57_full/extension/trees"
MODELS = WORK / "supervisor_41.fmEVx6/deliver/forest/extension/trees"
TALL = re.compile(rb"tree: (sugi_|hinoki_|broadleaf_)")
TREE = re.compile(rb"tree: (\w+)\.zip; pos=([^;]+); angle = ([^;]+); size = ([^\s]+)")


def make_road(root):
    meshes = {}
    for name, (x, y), material in (
        ("1ROAD_a", (0, 0), "asphalt"),
        ("1ROAD_b", (50, 0), "asphalt"),
        ("1ROAD_side_stub", (0, 50), "asphalt"),
        ("1ROAD_concrete", (0, 50), "concrete"),
        ("2ROAD_other", (0, 50), "asphalt"),
    ):
        verts = np.array([[x, y, 0], [x + 50, y, 0], [x + 50, y + 50, 0], [x, y + 50, 0]])
        file = name + ".npz"
        np.savez(root / file, verts=verts, faces=[[0, 1, 2, 3]])
        meshes[name] = {"material": material, "file": file}
    path = root / "manifest.json"
    path.write_text(json.dumps({"meshes": meshes}))
    return path


def make_model(path, name):
    verts = np.zeros(8, dtype=csptree.VERT_DTYPE)
    verts["p"] = [
        (-0.12, -0.02, 0), (0.12, -0.02, 0), (0.12, 0.6, 0), (-0.12, 0.6, 0),
        (-0.9, 0.45, 0.1), (0.9, 0.45, 0.1), (0.9, 1, 0.1), (-0.9, 1, 0.1),
    ]
    verts["uv"] = [(0, 1), (1, 1), (1, 0), (0, 0)] * 2
    indices = np.array([0, 1, 2, 0, 2, 3])
    raw_mesh = csptree.write_mesh(["bark.dds", "leaf.dds"], [
        csptree.Submesh(0, 0, indices, verts[:4]),
        csptree.Submesh(1, 1, indices, verts[4:]),
    ])
    leaf = Image.new("RGBA", (8, 8), (30, 140, 50, 255))
    leaf.putpixel((3, 3), (30, 140, 50, 0))
    csptree.write_package(path / (name + ".zip"), {"BASIC": {"HEIGHT": 10, "WIDTH": 4}},
                          {"LOD0.mesh": raw_mesh},
                          {"bark.dds": kn5.dds_bytes(Image.new("RGBA", (8, 8), (120, 70, 25, 255))),
                           "leaf.dds": kn5.dds_bytes(leaf)})


def check_kn5(path, tree_count, atlas):
    saved = kn5.read_kn5(path)
    assert saved["textures"] == {"treecards_59.dds": atlas}
    material, = saved["materials"]
    assert material["shader"] == "ksPerPixelAT"
    assert material["blend"] == 0 and material["alpha_tested"]
    assert material["props"]["ksAlphaRef"] == pytest.approx(0.3)
    assert material["textures"] == [("txDiffuse", 0, "treecards_59.dds")]
    nodes = saved["root"]["children"]
    assert sum(len(node["indices"]) // 3 for node in nodes) == 8 * tree_count
    for node in nodes:
        assert len(node["vertices"]) <= 65535
        assert node["indices"].max() < len(node["vertices"])
        assert node["visible"] and node["renderable"] and not node["cast_shadows"]
        assert node["material"] == 0
    return nodes


def check_tree(nodes, xyz, angle, height, width):
    for node in nodes:
        boards = node["vertices"].reshape(-1, 8, 11)
        roots = boards[:, :2, :3].mean(axis=1)
        at_root = np.isclose(roots, (xyz[0], xyz[1] - 0.3, xyz[2]), atol=0.001, rtol=0).all(axis=1)
        for i in np.flatnonzero(at_root):
            board = boards[i, :, :3]
            axis = np.array([np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))])
            cross = np.array([axis[2], 0, -axis[0]])
            if not np.allclose(board[1] - board[0], width * axis, atol=0.001):
                continue
            np.testing.assert_allclose(board[5] - board[4], width * cross, atol=0.001)
            np.testing.assert_allclose(board[[2, 3, 6, 7], 1], xyz[1] + height, atol=0.001)
            np.testing.assert_allclose(board[[0, 1, 4, 5], 1], xyz[1] - 0.3, atol=0.001)
            np.testing.assert_allclose(board[4:6].mean(axis=0), roots[i], atol=0.001)
            tris = node["indices"].reshape(-1, 8, 3)[i] - i * 8
            assert {tuple(t) for t in tris[:4]} == {tuple(t[::-1]) for t in tris[4:]}
            assert len(set(tris.ravel())) == 8
            return boards[i, :, 6:8]
    pytest.fail(f"板の位置または向きが台帳と一致しません: {xyz}, {angle}°")


def test_build_treecards_59(tmp_path, light_settings):
    from scripts import treecards_59

    models = tmp_path / "models"
    for name in treecards_59.SPECIES:
        make_model(models, name)
    source = tmp_path / "source"
    source.mkdir()
    header = b"configure: size variance = 1, 1\r\n"
    near = b"tree: sugi_a.zip; pos=100, 206, -65; angle = 90; size = 2\n"
    far = b"tree: hinoki_b.zip; pos=75, 205, -65.01; angle = 33; size = 0.5\r\n"
    seam = b"tree: broadleaf_d.zip; pos=50, 201, -25; angle = 15; size = 1\n"
    low = b"tree: fern_tall_a.zip; pos=3, 4, 5; angle = 0; size = 0.56"
    name = "forest_0_0.txt"
    (source / name).write_bytes(header + near + far + seam + low)
    (source / "forest_1_0.txt").write_bytes(header + low)
    output = tmp_path / "treecards"
    treecards_59.build(source, models, make_road(tmp_path), output, settings=light_settings)

    assert (source / name).read_bytes() == header + near + far + seam + low
    assert (output / "lists_no_tall" / name).read_bytes() == header + low
    assert (output / "lists_tall_near15" / name).read_bytes() == header + near + low
    for list_dir in ("lists_no_tall", "lists_tall_near15"):
        assert (output / list_dir / "forest_1_0.txt").read_bytes() == header + low
    atlas = (output / "treecards_59.dds").read_bytes()
    all_nodes = check_kn5(output / "yabitsu_treecards_all.kn5", 3, atlas)
    far_nodes = check_kn5(output / "yabitsu_treecards_far.kn5", 2, atlas)
    assert len(all_nodes) == len(far_nodes) == 1
    near_uv = check_tree(all_nodes, (100, 206, -65), 90, 20, 8)
    far_uv = check_tree(far_nodes, (75, 205, -65.01), 33, 5, 2)
    check_tree(far_nodes, (50, 201, -25), 15, 10, 4)
    assert not np.array_equal(near_uv, far_uv)
    image = Image.open(io.BytesIO(atlas))
    assert image.mode == "RGBA" and max(image.size) <= 2048
    pixels = np.asarray(image)
    for uv in (near_uv, far_uv):
        u, v = uv.min(axis=0)
        u1, v1 = uv.max(axis=0)
        patch = pixels[int(v * image.height):int(v1 * image.height) + 1,
                       int(u * image.width):int(u1 * image.width) + 1]
        assert set(np.unique(patch[..., 3])) == {0, 255}
        assert np.any((patch[..., :3] == (30, 140, 50)).all(axis=-1) & (patch[..., 3] == 255))
        assert np.any((patch[..., :3] == (120, 70, 25)).all(axis=-1) & (patch[..., 3] == 255))


def test_grid_split_limits_vertices():
    from scripts import treecards_59

    rows = np.zeros((8194, 6))
    rows[-2, 0] = 250
    rows[-1, 2] = -0.01
    groups = list(treecards_59.grid_chunks(rows))
    assert sorted(len(indices) for _, indices in groups) == [1, 1, 1, 8191]
    assert len({name for name, _ in groups}) == 4


def test_delivered_treecards_59():
    for path in (OUTPUT / "yabitsu_treecards_far.kn5", OUTPUT / "yabitsu_treecards_all.kn5",
                 OUTPUT / "lists_no_tall", OUTPUT / "lists_tall_near15", SOURCE, MODELS):
        if not path.exists():
            pytest.skip(f"依頼59の全線試作の照合資料がありません: {path}")

    paths = sorted(SOURCE.glob("forest_*.txt"))
    assert len(paths) == 168
    counts = Counter()
    samples = {"all": {}, "far": {}}
    for path in paths:
        rows = path.read_bytes().splitlines(keepends=True)
        low = [row for row in rows if not TALL.match(row)]
        near_path = OUTPUT / "lists_tall_near15" / path.name
        near_rows = near_path.read_bytes().splitlines(keepends=True)
        assert (OUTPUT / "lists_no_tall" / path.name).read_bytes() == b"".join(low)
        assert [row for row in near_rows if not TALL.match(row)] == low
        remaining = Counter(row for row in near_rows if TALL.match(row))
        counts["near"] += remaining.total()
        for row in rows:
            if TALL.match(row):
                counts["all"] += 1
                kind = TREE.fullmatch(row.rstrip()).group(1)
                samples["all"].setdefault(kind, row)
                if remaining[row]:
                    remaining[row] -= 1
                else:
                    samples["far"].setdefault(kind, row)
        assert remaining.total() == 0

    atlas = (OUTPUT / "treecards_59.dds").read_bytes()
    image = Image.open(io.BytesIO(atlas))
    alpha = image.getchannel("A")
    assert image.size == (2048, 2048)
    for variant, count in (("all", counts["all"]), ("far", counts["all"] - counts["near"])):
        nodes = check_kn5(OUTPUT / f"yabitsu_treecards_{variant}.kn5", count, atlas)
        assert len(samples[variant]) == 9
        cells = set()
        for row in samples[variant].values():
            name, pos, angle, scale = TREE.fullmatch(row.rstrip()).groups()
            xyz = tuple(map(float, pos.split(b",")))
            with zipfile.ZipFile(MODELS / (name.decode() + ".zip")) as package:
                ini = package.read("tree.ini")
            height = float(re.search(rb"^HEIGHT=(.+)$", ini, re.M)[1]) * float(scale)
            width = float(re.search(rb"^WIDTH=(.+)$", ini, re.M)[1]) * float(scale)
            gx, gz = (int(np.floor(xyz[i] / 250)) for i in (0, 2))
            uv = check_tree([n for n in nodes if n["name"].startswith(f"TREECARD_{gx}_{gz}_")],
                            xyz, float(angle), height, width)
            cells.add(tuple(uv.min(axis=0)))
            x0, y0 = (uv.min(axis=0) * image.width).astype(int)
            x1, y1 = (uv.max(axis=0) * image.width).astype(int)
            assert alpha.crop((x0, y0, x1 + 1, y1 + 1)).getextrema() == (0, 255)
        assert len(cells) == 9
        del nodes
