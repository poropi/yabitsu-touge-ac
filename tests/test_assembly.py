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


def test_narrowing_rows_get_no_bay_line(tmp_path):
    from yabitsu import crosssection as cs
    m = Manifest(tmp_path)
    sec = _straight(401)
    assembly.add_bay_lines(m, sec, cs.parse_bay_rows([{"s_from": "100", "s_to": "160", "side": "R", "depth_m": "-1.5", "taper_m": "10"}]))
    assert m.groups.get(assembly.GROUP, {"meshes": []})["meshes"] == []


def test_branch_pads_are_not_drawn(tmp_path):
    m = Manifest(tmp_path)
    pad = {"name": "branch_1_0_0", "role": "flat", "kind": "branch", "polygon": [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]], "z": 1.0}
    assembly.add_pads(m, [pad])
    assert not m.meshes


def test_road_end_is_closed_across_the_whole_road():
    """S4 の終わり（S5 はまだ）：道幅＋側溝の全部をふさぐ見えない壁と、その手前に工事中の柵。"""
    import numpy as np
    from yabitsu.crosssection import GUTTER_WIDTH
    sec = _straight()
    ms = assembly.road_end_closure(sec, 190.0)
    wall = next(m for m in ms if m[0] == "1WALL_road_end_block")
    v = wall[1]
    assert wall[5] is False and v[:, 0].min() > 186.0 and v[:, 0].max() < 192.0
    assert v[:, 1].min() <= -(sec.hw_right[190] + GUTTER_WIDTH) - 1.0 and v[:, 1].max() >= sec.hw_left[190] + GUTTER_WIDTH + 1.0
    bars = [m for m in ms if m[0] != "1WALL_road_end_block"]
    assert bars and max(m[1][:, 0].max() for m in bars) < v[:, 0].min()     # 柵は壁の手前（峠の側）


def test_raised_median_strip_with_double_orange_lines(tmp_path):
    """宮ヶ瀬湖ぞい（S5 9000〜9570m）の中央：幅 0.6m のコンクリートの盛り上がり（ゼブラ帯の縁石）と、その両側のオレンジの実線。"""
    import numpy as np
    m = Manifest(tmp_path)
    sec = _straight(401)
    assembly.add_centrelines(m, sec, [{"s_from": 50.0, "s_to": 150.0, "style": "orange_median", "note": ""}])
    names = m.groups[assembly.GROUP]["meshes"]
    strip = next(n for n in names if n.endswith("_median"))
    v = np.load(tmp_path / m.meshes[strip]["file"])["verts"]
    assert m.meshes[strip]["material"] == "concrete"
    assert np.abs(v[:, 1]).max() == pytest.approx(assembly.MEDIAN_W / 2, abs=1e-6)
    assert v[:, 2].max() >= float(sec.z[100]) + assembly.MEDIAN_H - 0.02
    lines = [n for n in names if m.meshes[n]["material"] == "line_orange"]
    lv = np.concatenate([np.load(tmp_path / m.meshes[n]["file"])["verts"] for n in lines])
    assert lv[:, 1].min() < -assembly.MEDIAN_W / 2 and lv[:, 1].max() > assembly.MEDIAN_W / 2


def test_road_inside_a_tunnel_is_a_separate_dry_piece_and_other_names_stay():
    """トンネルの中（dry）の路面・側溝は名前に _tun を付けた部品で、路面の材質は asphalt_tunnel（RainFX の水たまりに入れない）。
    トンネルの外の部品は今までどおりの名前（仕上げの貼り替え・雨の設定が名前で引く）。路面の面の数は分けても変わらない。"""
    import tempfile
    from pathlib import Path
    import numpy as np
    sec = _straight()
    with tempfile.TemporaryDirectory() as d0, tempfile.TemporaryDirectory() as d1:
        m0, m1 = Manifest(d0), Manifest(d1)
        assembly.add_road(m0, sec)
        assembly.add_road(m1, sec, dry=[(30.0, 60.0)])
        assert {n: e["material"] for n, e in m1.meshes.items() if n.endswith("_tun")} == {
            "1ROAD_s1_00_tun": "asphalt_tunnel", "1ROAD_s1_gutterL_00_tun": "concrete", "1ROAD_s1_gutterR_00_tun": "concrete"}
        assert set(m0.meshes) <= set(m1.meshes)
        faces = lambda m, names: sum(len(np.load(Path(m.root) / m.meshes[n]["file"])["faces"]) for n in names)
        assert faces(m0, ["1ROAD_s1_00"]) == faces(m1, ["1ROAD_s1_00", "1ROAD_s1_00_tun"])
