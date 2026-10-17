import pytest

from yabitsu import assembly
from yabitsu.manifest import Manifest
from tests.test_meshgen import _circle, _straight


def test_add_road_names_and_materials(tmp_path):
    m = Manifest(tmp_path)
    sec = _straight(401)
    assembly.add_road(m, sec)
    names = m.groups[assembly.GROUP]["meshes"]
    assert "1ROAD_s1_00" in names and "1ROAD_s1_01" in names
    assert "1ROAD_s1_gutterL_00" in names and "LINES_s1_edgeR_01" in names and "LINES_s1_centre_00" in names
    assert m.meshes["1ROAD_s1_00"]["material"] == "asphalt"


def test_add_road_rejects_folding(tmp_path):
    with pytest.raises(ValueError, match="s="):
        assembly.add_road(Manifest(tmp_path), _circle(2.5))


def test_add_pads_skips_pads_built_elsewhere(tmp_path):
    # 駐車場（lot）・峠（pass）・蓑毛（site）の pad は、それぞれのモジュールが路面を作る。平らな板を重ねない
    m = Manifest(tmp_path)
    sq = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]
    pads = [{"name": n, "role": "flat", "kind": k, "polygon": sq, "z": 1.0}
            for n, k in (("a", "lot"), ("b", "pass"), ("c", "site"))] + [{"name": "osm", "role": "end", "polygon": sq, "z": 1.0}]
    assembly.add_pads(m, pads)
    assert [n for n in m.meshes if n.startswith("1ROAD")] == ["1ROAD_s1_pad_03"]
