import json
import numpy as np
import pytest
from yabitsu.manifest import Manifest
from yabitsu.acobjects import DUMMY_YAW_OFFSET, Dummy


def _quad():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    return v, np.array([[0, 1, 2, 3]]), v[:, :2].copy()


def test_add_and_save(tmp_path):
    m = Manifest(tmp_path)
    m.add_material("asphalt", texture="textures/asphalt.png")
    v, f, uv = _quad()
    m.add_mesh("g", "1ROAD_a", v, f, uv, "asphalt")
    m.add_mesh("g", "1WALL_a", v, f, uv, "asphalt", renderable=False)
    m.add_empty("o", Dummy("AC_START_0", (1.0, 2.0, 3.0), 0.5))
    d = json.loads(m.save().read_text(encoding="utf-8"))
    assert d["groups"]["g"]["meshes"] == ["1ROAD_a", "1WALL_a"]
    assert (tmp_path / d["meshes"]["1ROAD_a"]["file"]).exists()
    assert d["meshes"]["1WALL_a"]["renderable"] is False
    # 上り・下りで同じダミー名を使うため、ダミーは「グループ/名前」のキーで持つ
    assert d["groups"]["o"]["empties"] == ["o/AC_START_0"]
    assert d["empties"]["o/AC_START_0"]["name"] == "AC_START_0"
    assert d["empties"]["o/AC_START_0"]["rotation_z"] == pytest.approx(0.5 + DUMMY_YAW_OFFSET)


def test_same_dummy_name_in_two_groups(tmp_path):
    m = Manifest(tmp_path)
    m.add_empty("up", Dummy("AC_START_0", (0, 0, 0), 0.0))
    m.add_empty("down", Dummy("AC_START_0", (9, 9, 9), 3.1))
    with pytest.raises(ValueError):
        m.add_empty("up", Dummy("AC_START_0", (1, 1, 1), 0.0))


def test_rejects_too_many_vertices(tmp_path):
    m = Manifest(tmp_path)
    v = np.zeros((70000, 3)); f = np.array([[0, 1, 2, 3]]); uv = np.zeros((70000, 2))
    with pytest.raises(ValueError):
        m.add_mesh("g", "big", v, f, uv, "x")


def test_rejects_duplicate_names(tmp_path):
    m = Manifest(tmp_path)
    v, f, uv = _quad()
    m.add_mesh("g", "a", v, f, uv, "x")
    with pytest.raises(ValueError):
        m.add_mesh("g", "a", v, f, uv, "x")


def test_split_puts_every_mesh_in_exactly_one_group(tmp_path):
    """場所で部品を別のグループ（kn5）へ移す：どの部品も、ちょうど1つのグループに入る。"""
    m = Manifest(tmp_path)
    f = np.array([[0, 1, 2]], np.int32); uv = np.zeros((3, 2))
    for k, x in enumerate((0.0, 50.0, 100.0, 150.0)):
        m.add_mesh("main", f"part_{k}", np.array([[x, 0, 0], [x + 1, 0, 0], [x, 1, 0]], float), f, uv, "asphalt")
    moved = m.split_group("main", "s4", lambda c: c[0] >= 90.0)
    assert moved == 2
    assert m.groups["main"]["meshes"] == ["part_0", "part_1"] and m.groups["s4"]["meshes"] == ["part_2", "part_3"]
    allm = m.groups["main"]["meshes"] + m.groups["s4"]["meshes"]
    assert sorted(allm) == sorted(m.meshes) and len(allm) == len(set(allm))
