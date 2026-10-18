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
