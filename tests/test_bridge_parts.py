import numpy as np
import pytest
from shapely.geometry import Point

from yabitsu import bridge_parts as bp, terrain as tr
from yabitsu.crosssection import GUTTER_WIDTH
from yabitsu.start_lot import road_formation
from tests.test_meshgen import _straight

SPEC = [{"osm_id": 5, "name": "釜田川橋", "kana": "かまたがわばし", "rail": "steel", "rail_color": "#C0302A",
         "deck_depth_m": 1.5, "pier_every_m": 30.0}]


def _ground(x, y):
    return np.full(np.shape(x), 80.0)     # 谷底（路面 100 の 20m 下）


def test_bridge_posts_stand_at_both_ends_outside_the_road():
    sec = _straight()
    ms = bp.meshes(sec, SPEC, {5: (60.0, 150.0)}, _ground)
    posts = np.concatenate([m[1] for m in ms if "_post" in m[0]])
    xs = sorted(set(np.round(posts[:, 0] / 10) * 10))
    assert min(posts[:, 0]) < 62 and max(posts[:, 0]) > 148
    hw = float(sec.hw_left[100]) + GUTTER_WIDTH
    assert np.abs(posts[:, 1]).min() >= hw - 1e-6
    plates = [m for m in ms if "_plate" in m[0]]
    assert len(plates) == 4 and all(m[4].startswith("bridge_plate_") for m in plates)


def test_piers_reach_the_ground_and_stay_off_the_road():
    sec = _straight()
    ms = bp.meshes(sec, SPEC, {5: (60.0, 150.0)}, _ground)
    piers = [m for m in ms if m[0].startswith("1WALL_bridge_5_pier")]
    assert len(piers) == 2                                    # 90m に 30m おき（両端の台を除く）
    for m in piers:
        v = m[1]
        assert v[:, 2].min() <= 80.0 + 1e-6 and v[:, 2].max() <= 100.0 - 1.0
    deck = next(m for m in ms if m[0] == "bridge_5_girder")
    assert deck[1][:, 2].min() == pytest.approx(100.0 - 1.5, abs=0.2)


def test_bridges_before_s5_keep_todays_terrain():
    """S5 の橋の下だけ谷を残す（地形を盛らない）。それより前の橋の下は今まで通り。"""
    sec = _straight()
    sec.bridge[40:61] = True; sec.bridge[120:151] = True
    X = np.array([[50.0, 135.0]]); Y = np.array([[0.0, 0.0]])
    nat = np.full(X.shape, 70.0)
    no_fill = bp.no_fill_mask(sec, s_from=100.0)
    Z, _ = tr.carve(X, Y, nat, sec, no_fill_rows=no_fill)
    Z0, _ = tr.carve(X, Y, nat, sec)
    assert Z[0, 0] == pytest.approx(Z0[0, 0])                 # s=50 の橋（S5 より前）は今まで通り盛る
    assert Z[0, 1] == pytest.approx(70.0)                     # s=135 の橋（S5）の下は谷のまま


def test_rails_follow_the_bridge_style_and_block_cars():
    sec = _straight()
    spec = [dict(SPEC[0], rail="concrete"), dict(SPEC[0], osm_id=6, rail="steel", rail_color="#7A5A3A")]
    ms = {m[0]: m for m in bp.meshes(sec, spec, {5: (20.0, 60.0), 6: (100.0, 150.0)}, _ground)}
    assert ms["bridge_5_rail_L"][4] == "concrete" and ms["bridge_6_rail_L"][4] == "bridge_rail_7A5A3A"
    walls = [m for n, m in ms.items() if n.startswith("1WALL_bridge_6_rail")]
    assert len(walls) == 2 and all(m[5] is False for m in walls)                      # 当たり判定（見えない壁）
    assert np.abs(ms["bridge_6_rail_L"][1][:, 1]).min() >= float(sec.hw_left[120]) + GUTTER_WIDTH - 1e-6


def test_red_arch_frames_span_over_the_road():
    sec = _straight()
    ms = {m[0]: m for m in bp.meshes(sec, [dict(SPEC[0], arch=True, arch_color="#C0302A")], {5: (60.0, 150.0)}, _ground)}
    a = ms["bridge_5_arch"]
    assert a[4] == "bridge_rail_C0302A"
    v = a[1]
    assert v[:, 2].max() >= 100.0 + bp.ARCH_CLEAR - 1e-6 and v[:, 1].min() < -sec.hw_right[100] and v[:, 1].max() > sec.hw_left[100]
