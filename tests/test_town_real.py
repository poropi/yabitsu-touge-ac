"""町の範囲（名古木交差点）を、本当のデータで確かめる（処理済みデータがあるときだけ）。"""
import numpy as np
import pytest
from shapely.geometry import Point

from yabitsu import config

_SEC = config.DATA_PROCESSED / "track_section.json"
pytestmark = pytest.mark.skipif(not _SEC.exists(), reason="処理済みデータがまだない")


@pytest.fixture(scope="module")
def town():
    from yabitsu import nagoki
    from yabitsu.crosssection import Section
    return nagoki.load_town(Section.load_json(_SEC))


def test_closed_streets_cannot_be_driven_beyond_the_barricades(town):
    """封鎖した道は、柵から 8m 先がもう走れる範囲の外（柵の横を回って奥へ行けない。最終レビューの指摘 1）。"""
    from yabitsu.town import _xy
    d = town.drivable(); c0 = np.array(town.center)
    roads = {str(w.get("id")): ln for (w, ln, wd, sl, sr) in town._roads}
    from yabitsu.town import main_formation
    main = main_formation(town.sec, 0.0, town.main_skip[1])
    checked = 0
    for c in town.closures:
        rid = c["name"].split("_", 1)[1]
        if rid not in roads:
            continue
        a, b = _xy([c["a"], c["b"]]); m = (a + b) / 2
        ln = roads[rid]; t = ln.project(Point(*m))
        p, q = (np.array(ln.interpolate(t + k).coords[0]) for k in (-8.0, 8.0))
        if c["name"].startswith("side_"):   # 本線ぞいの横道：本線から遠い方が奥
            far = lambda r: main.distance(Point(*r))
        else:                               # 交差点の腕：交差点の中心から遠い方が奥
            far = lambda r: float(np.linalg.norm(r - c0))
        beyond = p if far(p) > far(q) else q
        assert not d.contains(Point(*beyond)), c["name"]
        checked += 1
    assert checked >= 5


def test_drivable_area_stays_near_route_70_and_the_junction(town):
    """走れる範囲は、県道70号（＋歩道）と交差点の中だけ。交差点から 150m 先の横道などは入らない。"""
    d = town.drivable(); c0 = Point(*town.center)
    from yabitsu.town import main_formation
    zone = main_formation(town.sec, 0.0, town.main_skip[1]).buffer(town.main_walk + 0.5).union(c0.buffer(60.0))
    assert d.difference(zone).area < 1.0


def test_no_hidden_wall_or_roadside_wall_on_the_split_or_town_paving():
    """分かれ道の入口の見えない壁・柵も、道ぞいの壁・生け垣も、島の路面と町の路面の上には立たない（最終レビューの指摘 3）。"""
    import json
    from shapely.ops import unary_union
    from yabitsu import features as ft, nagoki, track
    from yabitsu.crosssection import Section
    sec = Section.load_json(_SEC)
    pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    raw = ft.load_features_rows(track.load_section_rows("features", track.load_bounds()))
    stubs, feats = nagoki.track_stubs(sec, pads, raw)
    tw = nagoki.load_town(sec)
    split_road = unary_union([z.carriageway for z in nagoki.load_splits(sec)])
    assert not [st.osm_id for st in stubs if st.mouth.intersects(split_road)]
    paved = unary_union([split_road, tw.carriageway])
    assert len(ft.cut_off(feats, sec, paved)) == len(feats)


def test_s4_branches_are_long_and_sealed():
    """S4（峠の先）の車の入れる分かれ道は、入口から 200m（短ければ道の終わり・本線へ戻る手前）まで作られ、入口と奥に見えない壁がある。"""
    import json
    from yabitsu import features as ft, nagoki, side_roads as sr, track
    from yabitsu.crosssection import Section
    sec = Section.load_json(_SEC)
    pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    stubs, _ = nagoki.track_stubs(sec, pads, ft.load_features_rows(track.load_section_rows("features", track.load_bounds())))
    s4 = sr.s4_start(sec)
    mine = [st for st in stubs if sr._mouth_s(sec, st) >= s4]
    assert len(mine) >= 5
    ways = {w["id"]: w for w in sr.load_elements() if w.get("type") == "way"}
    long_ = [st for st in mine if st.axis.length > sr.STUB_LENGTH_M + 1.0]
    assert len(long_) >= 3
    names = [m[0] for m in sr.stub_meshes(mine)]
    assert sum(n.startswith("1WALL_side_block_") for n in names) == len(mine)
    assert sum(n.startswith("1WALL_side_end_") for n in names) == len(long_)
    assert all(st.axis.length <= sr.S4_BRANCH_M + 1.0 for st in mine)


def test_nagoki_town_unchanged_after_generalising(town):
    """町の範囲の作りを「交差点ごとの設定」にしても、名古木の町は今と同じ（data/manual/s3_town_reference.json）。"""
    import json
    ref = json.loads((config.DATA_MANUAL / "s3_town_reference.json").read_text(encoding="utf-8"))
    d = town.drivable()
    for name, g in (("carriageway", town.carriageway), ("sidewalk", town.sidewalk), ("drivable", d)):
        a, x, y = ref[name]
        assert abs(g.area - a) < 0.5 and abs(g.centroid.x - x) < 0.01 and abs(g.centroid.y - y) < 0.01, name
    assert abs(town.region.area - ref["region"][0]) < 0.5
    assert list(town.main_skip) == pytest.approx(ref["main_skip"], abs=1e-6)
    assert [[c["name"], c["a"], c["b"]] for c in town.closures] == ref["closures"]
    assert abs(sum(g.length for g in town.boundary_walls()) - ref["walls_len"]) < 0.05
    assert len(town.markings) == ref["markings"] and [p["name"] for p in town.pads()] == ref["pads"]


_KITA = config.DATA_MANUAL / "s5_kitahara.json"


@pytest.mark.skipif(not _KITA.exists(), reason="宮ヶ瀬北原の spec がまだない")
def test_toriibara_arm_is_built_past_the_barricade():
    """宮ヶ瀬北原：県道64号の鳥居原方面（北西）は、封鎖の柵の先 50m も道（将来の延長）。柵の先へは走れない。"""
    from yabitsu import geo, nagoki
    from yabitsu.crosssection import Section
    from yabitsu.town import _xy
    j = nagoki.JUNCTIONS["kitahara"]
    tw = nagoki.load_town(Section.load_json(_SEC), junction=j)
    tx, ty = geo.to_local(35.538915, 139.222051)          # 鳥居原ふれあいの館
    c0 = np.array(tw.center); goal = np.array([float(tx), float(ty)])
    arms = [c for c in tw.closures if c["name"].startswith("arm_")]
    def toward(c):
        a, b = _xy([c["a"], c["b"]]); m = (a + b) / 2
        return float(np.dot((m - c0) / np.linalg.norm(m - c0), (goal - c0) / np.linalg.norm(goal - c0)))
    c = max(arms, key=toward)
    a, b = _xy([c["a"], c["b"]]); m = (a + b) / 2
    ln = next(l for (w, l, wd, sl, sr) in tw._roads if str(w.get("id")) == c["name"].split("_", 1)[1])   # 道に沿って 50m 先
    t0 = ln.project(Point(*m))
    q0, q1 = ln.interpolate(t0 - 50.0), ln.interpolate(t0 + 50.0)
    p = max((q0, q1), key=lambda q: q.distance(Point(*c0)))
    assert tw.carriageway.buffer(0.5).contains(p) and not tw.drivable().contains(p)


def test_second_junction_mesh_names_get_the_junction_name_once():
    from yabitsu import nagoki
    items = [("town_obj_concrete", 1), ("1WALL_town_block_000", 2), ("barricade_000", 3), ("1WALL_kitahara_polesign_poles", 4),
             ("kitahara_polesign_0", 5)]
    assert [n for n, _ in nagoki.rename_meshes(items, "kitahara")] == [
        "kitahara_obj_concrete", "1WALL_kitahara_block_000", "kitahara_barricade_000", "1WALL_kitahara_polesign_poles", "kitahara_polesign_0"]
