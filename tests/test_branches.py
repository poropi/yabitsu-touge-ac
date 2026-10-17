import numpy as np
import pytest
from shapely.geometry import Point

from yabitsu import branches as br
from yabitsu.meshgen import face_normals
from tests.test_meshgen import _straight
from tests.test_side_roads import _way


def _main_and_branch():
    sec = _straight()   # 本線：x 軸、幅 6、高さ 100
    main = _way([[0, 0], [200, 0]], id=1, ref="70")
    branch = _way([[60, 0], [70, 10], [130, 10], [140, 0]], id=2, ref="70", oneway="yes")
    return sec, [main, branch]


def test_parallel_route70_way_is_found_and_main_is_not():
    sec, ways = _main_and_branch()
    bs = br.parallel_branches(ways, sec, {})
    assert [b.osm_id for b in bs] == [2]


def test_branch_surface_at_main_road_edge_height_and_not_on_main():
    sec, ways = _main_and_branch()
    b, = br.parallel_branches(ways, sec, {"2": {"width_m": 4.0}})
    from yabitsu.crosssection import GUTTER_WIDTH
    assert b.surface.contains(Point(100, 10)) and not b.surface.contains(Point(100, 1.0))
    (name, v, f, uv, mat, rend), = [m for m in b.meshes() if m[0].startswith("1ROAD_")]
    assert np.allclose(v[:, 2], sec.edge_z(1)[100], atol=0.02)
    assert (face_normals(v, f)[:, 2] > 0).all()


def test_island_is_raised_and_curbed():
    sec, ways = _main_and_branch()
    island = [[*br._ll(x, y)] for x, y in [(72, 3.8), (128, 3.8), (128, 8.0), (72, 8.0)]]
    b, = br.parallel_branches(ways, sec, {"2": {"width_m": 4.0, "island": island}})
    ms = {m[0]: m for m in b.meshes()}
    top = next(m for n, m in ms.items() if n.startswith("branch_2_island") and m[4] == "gravel")
    assert np.allclose(top[1][:, 2], sec.edge_z(1)[100] + br.ISLAND_H, atol=0.03)


def test_torii_proportions():
    parts = br.torii_meshes(np.array([0.0, 0.0]), 100.0, facing_deg=0.0, span=3.0, height=4.2)
    v = np.concatenate([p[1] for p in parts])
    assert v[:, 2].min() == pytest.approx(100.0 - 0.2, abs=0.05)
    assert v[:, 2].max() == pytest.approx(100.0 + 4.2, abs=0.25)            # 笠木の反りの先まで
    assert np.ptp(v[:, 0]) > 3.0 + 2 * 0.6                                   # 笠木は柱より左右に出る（東西に広がる）
    assert np.ptp(v[:, 1]) < 1.2                                             # 南北（くぐる向き）には薄い


def test_route70_beyond_the_track_end_is_not_a_branch():
    sec, ways = _main_and_branch()
    ways.append(_way([[200, 0], [260, 30], [400, 40]], id=9, ref="70"))     # 道の終わりの先へ続く県道70号
    assert [b.osm_id for b in br.parallel_branches(ways, sec, {})] == [2]


def test_split_area_has_no_gap_between_road_island_and_branch():
    sec, ways = _main_and_branch()
    island = [[*br._ll(x, y)] for x, y in [(75, 5.0), (125, 5.0), (125, 8.0), (75, 8.0)]]
    b, = br.parallel_branches(ways, sec, {"2": {"width_m": 4.0, "island": island}})
    from shapely.geometry import box
    from yabitsu.start_lot import road_formation
    whole = b.surface.union(b.island).union(road_formation(sec, 0.0, 200.0))
    gap = box(70, 0, 130, 12).intersection(b.line.buffer(2.0).union(box(75, 0, 125, 10))).difference(whole)
    assert gap.area < 1.0                                     # 本線・島・東側の道の間に地面が出ない
    assert b.surface.contains(Point(66, 4.5))                  # 分かれる所の三角の所も路面（本線の側溝の外）


def test_branch_pads_follow_a_slope():
    from yabitsu.site import plane_z
    sec, ways = _main_and_branch()
    sec.z[:] = 100.0 + 0.06 * sec.s
    b, = br.parallel_branches(ways, sec, {"2": {"width_m": 4.0}})
    pads = b.pads()
    assert len(pads) >= 5 and all("plane" in p for p in pads)
    for p in pads:
        c = np.asarray(p["polygon"])[:-1]
        dz = b.z(c[:, 0], c[:, 1]) - plane_z(p["plane"], c[:, 0], c[:, 1])
        assert dz.min() >= 0.0 and dz.max() < 0.3              # 平面は路面のすぐ下
