"""区間をつなぐ作り替えのあとも、S1 が変わっていないことを確かめる（参照：data/manual/s1_reference.json）。
処理済みデータ（data/processed/track_*.json）がまだ無いときは skip する。"""
import json
import math

import numpy as np
import pytest

from yabitsu import config, geo

REF = config.DATA_MANUAL / "s1_reference.json"
SEC = config.DATA_PROCESSED / "track_section.json"
PADS = config.DATA_PROCESSED / "track_pads.json"
PROFILE = config.DATA_PROCESSED / "track_profile.npz"
JOIN_M = 30.0  # S1 の始点からこの距離までは、つなぎ目として形が変わってよい（設計書 8 章）
END_JOIN_M = 30.0  # S1 の終わり（峠の先、S4 とのつなぎ目）からこの距離も、形が変わってよい（S4 設計書 6 章）

pytestmark = pytest.mark.skipif(not (SEC.exists() and PADS.exists() and PROFILE.exists()),
                                reason="track_* の処理済みデータがまだない")


def _ref():
    return json.loads(REF.read_text(encoding="utf-8"))


def _track():
    from yabitsu.crosssection import Section
    sec = Section.load_json(SEC)
    pads = json.loads(PADS.read_text(encoding="utf-8"))
    pr = np.load(PROFILE, allow_pickle=True)
    bounds = pr["bounds"].item()
    return sec, pads, bounds


def _xy(lat, lon):
    x, y = geo.to_local(lat, lon)
    return float(x), float(y)


def test_s1_road_unchanged_beyond_join():
    # 中心線の点（1m ごと）どうしではなく、今の中心線（線）までの距離で比べる。区間をつなぐと点の位置は道に沿ってずれる
    from yabitsu.acobjects import frame_at
    from yabitsu.layouts import s_of
    sec, pads, bounds = _track()
    path = sec.path()
    length = bounds["s1"][1] - bounds["s1"][0]
    for s_local, lat, lon, z in _ref()["centerline"]:
        if s_local < JOIN_M or s_local > length - END_JOIN_M:
            continue
        x, y, zc, _ = frame_at(path, s_of(sec, (lat, lon), bounds, "s1"))
        xr, yr = _xy(lat, lon)
        assert math.hypot(x - xr, y - yr) <= 0.1, (s_local, math.hypot(x - xr, y - yr))
        assert abs(zc - z) <= 0.1, (s_local, zc - z)


@pytest.mark.skipif(not (config.DATA_MANUAL / "layouts.json").exists(), reason="layouts.json（Task 4）がまだない")
def test_s1_layout_objects_unchanged():
    from yabitsu import layouts
    sec, pads, bounds = _track()
    s0 = bounds["s1"][0]
    current = {l.name: {d.name: d for d in layouts.layout_objects(sec, pads, l, bounds)}
               for l in layouts.load_layouts() if l.name in ("s1_up", "s1_down")}
    for name, objs in _ref()["layouts"].items():
        for dname, (lat, lon, z, h) in objs.items():
            d = current[name][dname]
            x, y = _xy(lat, lon)
            near = int(np.argmin(np.hypot(sec.x - x, sec.y - y)))
            in_join = sec.s[near] - s0 < JOIN_M  # つなぎ目の中はヘアピンの形が変わってよい（位置 1m・向き 0.15rad まで）
            assert math.hypot(d.location[0] - x, d.location[1] - y) <= (1.0 if in_join else 0.1), (name, dname)
            assert abs(math.remainder(d.heading - h, 2 * math.pi)) <= (0.15 if in_join else 0.01), (name, dname)


def test_s1_sites_unchanged():
    from yabitsu import start_lot
    sec, pads, bounds = _track()
    ref = _ref()["sites"]
    lot = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    f = start_lot.frame(lot)
    assert math.dist(_xy(*ref["lot_origin"]), lot["origin"]) <= 0.05
    assert math.dist(_xy(*ref["tower"]), f.xy(lot["tower"]["a"], lot["tower"]["b"])) <= 0.05
    lot_pad = next(p for p in pads if p.get("kind") == "lot")
    assert abs(lot_pad["z"] - ref["lot_z"]) <= 0.1
    lane = next(p for p in pads if p["name"] == "pass_lane")
    for (lat, lon, z, h), (x, y, zc, hc) in zip(ref["pass_lane_pits"], lane["pit_points"]):
        assert math.dist(_xy(lat, lon), (x, y)) <= 0.1 and abs(zc - z) <= 0.1
