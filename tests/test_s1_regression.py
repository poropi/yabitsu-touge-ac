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
    sec, pads, bounds = _track()
    s0 = bounds["s1"][0]
    m = sec.s >= s0 - 5.0
    for s_local, lat, lon, z in _ref()["centerline"]:
        if s_local < JOIN_M:
            continue
        x, y = _xy(lat, lon)
        d = np.hypot(sec.x[m] - x, sec.y[m] - y)
        i = int(np.argmin(d))
        assert d[i] <= 0.1, (s_local, d[i])
        assert abs(sec.z[m][i] - z) <= 0.1, (s_local, sec.z[m][i] - z)


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
            tol = 1.0 if sec.s[near] - s0 < JOIN_M else 0.1
            assert math.hypot(d.location[0] - x, d.location[1] - y) <= tol, (name, dname)
            assert abs(math.remainder(d.heading - h, 2 * math.pi)) <= 0.01, (name, dname)


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
