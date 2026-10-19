import numpy as np
import pytest

from yabitsu import forest_ac, forest_place


def _trees(rows):
    t = np.zeros(len(rows), forest_place.DTYPE)
    for i, (x, y, z, p) in enumerate(rows):
        t[i] = (x, y, z, p, 30.0, 1.1)
    return t.view(np.recarray)


def test_write_lists_tiles(tmp_path):
    t = _trees([(10.0, 20.0, 500.0, "sugi_a"), (1500.0, 20.0, 600.0, "hinoki_a"), (20.0, 30.0, 505.0, "fern_a")])
    lists = forest_ac.write_lists(t, tmp_path, tile_m=1000)
    assert sorted(lists) == ["trees\\forest_0_0.txt", "trees\\forest_1_0.txt"]
    a = (tmp_path / "trees" / "forest_0_0.txt").read_text(encoding="utf-8").splitlines()
    assert a[0] == "configure: angle variance = 0, 0" and a[1] == "configure: size variance = 1, 1"
    assert "configure: width variance = 1, 1" in a
    assert "tree: sugi_a.zip; pos=10.00, 500.00, -20.00; angle = 30; size = 1.10" in a
    assert len([l for l in a if l.startswith("tree:")]) == 2


def test_ext_config_contents(tmp_path):
    forest_ac.write_ext_config(tmp_path, ["trees\\forest_0_0.txt", "trees\\forest_1_0.txt"], ["TREES_s1_000", "TREES_s1_001"])
    s = (tmp_path / "extension" / "ext_config.ini").read_text(encoding="utf-8")
    assert s.startswith(forest_ac.MARK)
    assert "[TREES]" in s and "LIST_0=trees\\forest_0_0.txt" in s and "LIST_1=trees\\forest_1_0.txt" in s
    assert "MESHES=TREES_s1_000,TREES_s1_001" in s and "IS_ACTIVE=0" in s


def test_ext_config_refuses_foreign(tmp_path):
    (tmp_path / "extension").mkdir()
    (tmp_path / "extension" / "ext_config.ini").write_text("[LIGHTING]\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        forest_ac.write_ext_config(tmp_path, ["trees\\a.txt"], [])


def test_layout_disables_trees(tmp_path):
    forest_ac.write_layout_off(tmp_path, ["s1_landscape_up"])
    s = (tmp_path / "s1_landscape_up" / "extension" / "ext_config.ini").read_text(encoding="utf-8")
    assert s.startswith(forest_ac.MARK) and "[TREES]\nACTIVE=0" in s
