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
    assert "1ROAD_s1_gutterL_00" in names and "LINES_s1_edgeR_01" in names
    assert not any("centre" in n for n in names)  # 中央線は車線数ではなく、中央線の表（add_centrelines）で引く
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


def test_add_centrelines_draws_only_listed_ranges_with_style(tmp_path):
    import numpy as np
    m = Manifest(tmp_path)
    sec = _straight(401)
    rows = [{"s_from": 0.0, "s_to": 100.0, "style": "orange_solid", "note": ""},
            {"s_from": 200.0, "s_to": 300.0, "style": "white_dashed", "note": ""}]
    assembly.add_centrelines(m, sec, rows)
    names = m.groups[assembly.GROUP]["meshes"]
    assert names == ["LINES_track_centre_000", "LINES_track_centre_001"]
    assert m.meshes["LINES_track_centre_000"]["material"] == "line_orange"
    assert m.meshes["LINES_track_centre_001"]["material"] == "line_white"
    v0 = np.load(tmp_path / m.meshes["LINES_track_centre_000"]["file"])["verts"]
    v1 = np.load(tmp_path / m.meshes["LINES_track_centre_001"]["file"])["verts"]
    assert v0[:, 0].min() >= 0.0 and v0[:, 0].max() <= 100.0 and abs(v0[:, 1]).max() < 0.1   # 道の中央
    assert v1[:, 0].min() >= 200.0 and v1[:, 0].max() <= 300.0
    assert m.meshes["LINES_track_centre_001"]["faces"] < m.meshes["LINES_track_centre_000"]["faces"]  # 破線は実線より面が少ない


def test_unknown_centreline_style_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        assembly.add_centrelines(Manifest(tmp_path), _straight(101), [{"s_from": 0, "s_to": 50, "style": "pink", "note": ""}])


def test_orange_flanked_adds_white_dashes_both_sides(tmp_path):
    import numpy as np
    m = Manifest(tmp_path)
    sec = _straight(401)
    assembly.add_centrelines(m, sec, [{"s_from": 0.0, "s_to": 100.0, "style": "orange_flanked", "note": ""}])
    names = m.groups[assembly.GROUP]["meshes"]
    assert names == ["LINES_track_centre_000", "LINES_track_centre_000_flank"]
    assert m.meshes["LINES_track_centre_000"]["material"] == "line_orange"
    assert m.meshes["LINES_track_centre_000_flank"]["material"] == "line_white"
    v = np.load(tmp_path / m.meshes["LINES_track_centre_000_flank"]["file"])["verts"]
    lat = np.unique(np.round(np.abs(v[:, 1]), 2))
    assert lat.min() >= assembly.FLANK_OFFSET - 0.1 and lat.max() <= assembly.FLANK_OFFSET + 0.1
    assert (v[:, 1] > 0).any() and (v[:, 1] < 0).any()


def test_bus_bay_gets_dashed_separator_at_the_old_edge(tmp_path):
    import numpy as np
    from yabitsu import crosssection as cs
    m = Manifest(tmp_path)
    sec = _straight(401)
    bays = cs.parse_bay_rows([{"s_from": "100", "s_to": "160", "side": "L", "depth_m": "3", "taper_m": "10"}])
    hw0 = sec.hw_left.copy()
    sec.hw_left, sec.hw_right = cs.apply_bays(sec.s, sec.hw_left, sec.hw_right, bays)
    assembly.add_bay_lines(m, sec, bays)
    (name,) = m.groups[assembly.GROUP]["meshes"]
    v = np.load(tmp_path / m.meshes[name]["file"])["verts"]
    assert m.meshes[name]["material"] == "line_white"
    assert v[:, 0].min() >= 100 - 1e-6 and v[:, 0].max() <= 160 + 1e-6
    assert np.allclose(np.abs(v[:, 1]).mean(), hw0[0] - 0.25, atol=0.1)
