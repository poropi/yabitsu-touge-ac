import io
import math

import numpy as np
import pytest
from PIL import Image

from yabitsu import kn5
from yabitsu.acobjects import Dummy
from yabitsu.config import AC_TRACKS, ProtectedPathError
from yabitsu.manifest import Manifest

QUAD_V = np.array([[0, 0, 1], [2, 0, 1], [2, 3, 1], [0, 3, 1]], float)  # 上向き（+Z）の四角形
QUAD_F = np.array([[0, 1, 2, 3]])
QUAD_UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)


def _manifest(tmp_path):
    (tmp_path / "textures").mkdir()
    for name, rgb in {"asphalt": (70, 70, 72), "tree": (10, 90, 10)}.items():
        Image.new("RGBA", (8, 4), (*rgb, 255)).save(tmp_path / "textures" / f"{name}.png")
    m = Manifest(tmp_path)
    m.add_material("asphalt", texture="textures/asphalt.png")
    m.add_material("tree", texture="textures/tree.png", alpha=True, two_sided=True)
    m.add_material("red", color=(1.0, 0.0, 0.0))
    m.add_mesh("g", "1ROAD_a", QUAD_V, QUAD_F, QUAD_UV, "asphalt")
    m.add_mesh("g", "1WALL_hidden", QUAD_V, QUAD_F, QUAD_UV, "asphalt", renderable=False)
    m.add_mesh("g", "TREE_a", QUAD_V, QUAD_F, QUAD_UV, "tree")
    m.add_mesh("g", "MARK", QUAD_V, QUAD_F, QUAD_UV, "red")
    m.add_empty("objs", Dummy("AC_START_0", (10.0, 20.0, 5.3), 0.0))      # 東向き
    m.add_empty("objs", Dummy("AC_PIT_0", (0.0, 0.0, 0.0), math.pi / 2))  # 北向き
    return m.save()


def _write_read(tmp_path, group):
    out = tmp_path / "out" / f"{group}.kn5"
    kn5.write_from_manifest(_manifest(tmp_path), group, out)
    return kn5.read_kn5(out)


def _nodes(k):
    return {n["name"]: n for n in k["root"]["children"]}


def test_dds_has_full_mip_chain_and_original_pixels():
    img = Image.new("RGBA", (8, 4), (10, 20, 30, 128))
    data = kn5.dds_bytes(img)
    assert data[:4] == b"DDS "
    assert int.from_bytes(data[28:32], "little") == 4  # 8x4 → 4x2 → 2x1 → 1x1
    back = Image.open(io.BytesIO(data)).convert("RGBA")
    assert back.size == (8, 4) and back.getpixel((3, 2)) == (10, 20, 30, 128)
    assert len(data) == 128 + 4 * (32 + 8 + 2 + 1)


def test_header_root_and_textures(tmp_path):
    k = _write_read(tmp_path, "g")
    assert k["version"] == 5
    assert k["root"]["name"] == "g" and k["root"]["cls"] == 1
    assert np.allclose(k["root"]["matrix"], np.eye(4))
    assert set(k["textures"]) == {"asphalt.dds", "tree.dds", "red.dds"}
    assert Image.open(io.BytesIO(k["textures"]["red.dds"])).convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_materials_shaders_and_defaults(tmp_path):
    mats = {m["name"]: m for m in _write_read(tmp_path, "g")["materials"]}
    a = mats["asphalt"]
    assert a["shader"] == "ksPerPixel" and a["blend"] == 0 and a["alpha_tested"] is False and a["depth"] == 0
    assert a["props"]["ksAmbient"] == pytest.approx(0.5) and a["props"]["ksDiffuse"] == pytest.approx(0.6)
    assert a["textures"] == [("txDiffuse", 0, "asphalt.dds")]
    t = mats["tree"]
    assert t["shader"] == "ksPerPixelAT" and t["alpha_tested"] is True and t["props"]["ksAlphaRef"] > 0


def test_mesh_is_in_ac_axes_with_flipped_v(tmp_path):
    n = _nodes(_write_read(tmp_path, "g"))["1ROAD_a"]
    pos = n["vertices"][:, 0:3]
    assert {tuple(p) for p in pos.round(3)} == {(0, 1, 0), (2, 1, 0), (2, 1, -3), (0, 1, -3)}  # (x, z, -y)
    uv = n["vertices"][:, 6:8]
    by_pos = {tuple(p): tuple(t) for p, t in zip(pos.round(3), uv.round(3))}
    assert by_pos[(0, 1, 0)] == (0, 1) and by_pos[(2, 1, -3)] == (1, 0)


def test_quad_becomes_two_front_facing_triangles(tmp_path):
    n = _nodes(_write_read(tmp_path, "g"))["1ROAD_a"]
    tri = n["indices"].reshape(-1, 3)
    assert len(tri) == 2
    p, nrm = n["vertices"][:, 0:3], n["vertices"][:, 3:6]
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    assert np.all(np.cross(b - a, c - a)[:, 1] > 0)  # 右手系の巻き順で上（+Y）を向く
    assert np.allclose(nrm, [0, 1, 0])


def test_mesh_flags_and_bounding_sphere(tmp_path):
    nodes = _nodes(_write_read(tmp_path, "g"))
    road, wall = nodes["1ROAD_a"], nodes["1WALL_hidden"]
    assert (road["cast_shadows"], road["visible"], road["transparent"], road["renderable"]) == (True, True, False, True)
    assert (wall["cast_shadows"], wall["visible"], wall["renderable"]) == (False, False, False)
    c, r = np.array(road["bsphere_center"]), road["bsphere_radius"]
    assert np.all(np.linalg.norm(road["vertices"][:, 0:3] - c, axis=1) <= r + 1e-4)
    assert road["lod"] == (0.0, 0.0) and road["layer"] == 0


def test_two_sided_material_adds_reversed_back_node(tmp_path):
    k = _write_read(tmp_path, "g"); nodes = _nodes(k)
    back = nodes["TREE_a_back"]
    assert back["material"] == nodes["TREE_a"]["material"]
    assert k["materials"][back["material"]]["name"] == "tree"
    p = back["vertices"][:, 0:3]; tri = back["indices"].reshape(-1, 3)
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    assert np.all(np.cross(b - a, c - a)[:, 1] < 0) and np.allclose(back["vertices"][:, 3:6], [0, -1, 0])
    assert "1ROAD_a_back" not in nodes


def test_dummy_matrix_forward_is_local_z(tmp_path):
    k = _write_read(tmp_path, "objs")
    assert k["textures"] == {} and k["materials"] == []
    nodes = _nodes(k)
    east = nodes["AC_START_0"]["matrix"]
    assert np.allclose(east[2, :3], [1, 0, 0], atol=1e-6)   # 前 = +X（東）
    assert np.allclose(east[1, :3], [0, 1, 0], atol=1e-6)   # 上 = +Y
    assert np.allclose(east[3, :3], [10.0, 5.3, -20.0])     # 位置 = (x, z, -y)
    north = nodes["AC_PIT_0"]["matrix"]
    assert np.allclose(north[2, :3], [0, 0, -1], atol=1e-6)  # 北 = AC の -Z


def test_refuses_protected_folder(tmp_path):
    with pytest.raises(ProtectedPathError):
        kn5.write_from_manifest(_manifest(tmp_path), "g", AC_TRACKS / "nanohanadai" / "x.kn5")
