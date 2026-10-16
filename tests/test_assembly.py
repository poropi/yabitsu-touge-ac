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
