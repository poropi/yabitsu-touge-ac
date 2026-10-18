"""S5 を札掛の先に足したあとも、S4 が変わっていないことを確かめる（参照：data/manual/s4_reference.json）。
処理済みデータ（data/processed/track_*.json）がまだ無いときは skip する。"""
import json
import math

import numpy as np
import pytest

from yabitsu import config, geo

REF = config.DATA_MANUAL / "s4_reference.json"
SEC = config.DATA_PROCESSED / "track_section.json"
PADS = config.DATA_PROCESSED / "track_pads.json"
PROFILE = config.DATA_PROCESSED / "track_profile.npz"
BRIDGE_NEAR_M, BRIDGE_TOL_M = 10.0, 0.2  # 橋（とその前後 5m）から 10m 以内の点は 0.2m まで

pytestmark = pytest.mark.skipif(not (SEC.exists() and PADS.exists() and PROFILE.exists() and REF.exists()),
                                reason="track_* の処理済みデータか参照がまだない")


def _ref():
    return json.loads(REF.read_text(encoding="utf-8"))


def _track():
    from yabitsu.crosssection import Section
    sec = Section.load_json(SEC)
    pads = json.loads(PADS.read_text(encoding="utf-8"))
    bounds = np.load(PROFILE, allow_pickle=True)["bounds"].item()
    return sec, pads, bounds


def _xy(lat, lon):
    x, y = geo.to_local(lat, lon)
    return float(x), float(y)


END_JOIN_M = 30.0  # S4 の終わり（S5 とのつなぎ目）からこの距離は、形が変わってよい


def test_s4_road_unchanged_before_the_joint():
    from yabitsu.acobjects import frame_at
    from yabitsu.layouts import s_of
    sec, pads, bounds = _track()
    path = sec.path()
    pr = np.load(PROFILE, allow_pickle=True)
    length = bounds["s4"][1] - bounds["s4"][0]
    for s_local, lat, lon, z in _ref()["centerline"]:
        if s_local > length - END_JOIN_M:
            continue
        s = s_of(sec, (lat, lon), bounds, "s4")
        x, y, zc, _ = frame_at(path, s)
        xr, yr = _xy(lat, lon)
        assert math.hypot(x - xr, y - yr) <= 0.1, (s_local, math.hypot(x - xr, y - yr))
        on_bridge = not pr["valid"][(pr["s"] >= s - BRIDGE_NEAR_M) & (pr["s"] <= s + BRIDGE_NEAR_M)].all()
        assert abs(zc - z) <= (BRIDGE_TOL_M if on_bridge else 0.1), (s_local, zc - z)


@pytest.mark.skipif(not (config.DATA_MANUAL / "layouts.json").exists(), reason="layouts.json がまだない")
def test_s4_layout_objects_unchanged():
    from yabitsu import layouts
    sec, pads, bounds = _track()
    current = {l.name: {d.name: d for d in layouts.layout_objects(sec, pads, l, bounds)}
               for l in layouts.load_layouts() if l.name in _ref()["layouts"]}
    for name, objs in _ref()["layouts"].items():
        for dname, (lat, lon, z, h) in objs.items():
            d = current[name][dname]
            x, y = _xy(lat, lon)
            assert math.hypot(d.location[0] - x, d.location[1] - y) <= 0.1, (name, dname)
            assert abs(d.location[2] - z) <= 0.1 and abs(math.remainder(d.heading - h, 2 * math.pi)) <= 0.01, (name, dname)


def test_s4_pits_unchanged():
    sec, pads, bounds = _track()
    for name, ref in _ref()["sites"].items():
        pits = next(p for p in pads if p["name"] == name)["pit_points"]
        for (lat, lon, z, h), (x, y, zc, hc) in zip(ref, pits):
            assert math.dist(_xy(lat, lon), (x, y)) <= 0.1 and abs(zc - z) <= 0.1, name
