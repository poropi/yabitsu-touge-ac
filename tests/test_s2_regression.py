"""S3 を下に足したあとも、S2 が変わっていないことを確かめる（参照：data/manual/s2_reference.json）。
処理済みデータ（data/processed/track_*.json）がまだ無いときは skip する。"""
import json
import math

import numpy as np
import pytest

from yabitsu import config, geo

REF = config.DATA_MANUAL / "s2_reference.json"
SEC = config.DATA_PROCESSED / "track_section.json"
PADS = config.DATA_PROCESSED / "track_pads.json"
PROFILE = config.DATA_PROCESSED / "track_profile.npz"
JOIN_M = 30.0  # S2 の始点からこの距離までは、つなぎ目として形が変わってよい
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


def test_s2_road_unchanged_beyond_join():
    from yabitsu.acobjects import frame_at
    from yabitsu.layouts import s_of
    sec, pads, bounds = _track()
    path = sec.path()
    pr = np.load(PROFILE, allow_pickle=True)
    for s_local, lat, lon, z in _ref()["centerline"]:
        if s_local < JOIN_M:
            continue
        s = s_of(sec, (lat, lon), bounds, "s2")
        x, y, zc, _ = frame_at(path, s)
        xr, yr = _xy(lat, lon)
        assert math.hypot(x - xr, y - yr) <= 0.1, (s_local, math.hypot(x - xr, y - yr))
        # 橋の上（標高データを使わず前後から補う所）は、補う範囲の端が格子1つぶん動くだけで 十数cm 変わる
        on_bridge = not pr["valid"][(pr["s"] >= s - BRIDGE_NEAR_M) & (pr["s"] <= s + BRIDGE_NEAR_M)].all()
        assert abs(zc - z) <= (BRIDGE_TOL_M if on_bridge else 0.1), (s_local, zc - z)


def test_s2_layout_objects_unchanged():
    from yabitsu import layouts
    sec, pads, bounds = _track()
    s0 = bounds["s2"][0]
    current = {l.name: {d.name: d for d in layouts.layout_objects(sec, pads, l, bounds)}
               for l in layouts.load_layouts() if l.name in ("s2_up", "s2_down")}
    for name, objs in _ref()["layouts"].items():
        for dname, (lat, lon, z, h) in objs.items():
            d = current[name][dname]
            x, y = _xy(lat, lon)
            near = int(np.argmin(np.hypot(sec.x - x, sec.y - y)))
            in_join = sec.s[near] - s0 < JOIN_M
            assert math.hypot(d.location[0] - x, d.location[1] - y) <= (1.0 if in_join else 0.1), (name, dname)
            assert abs(math.remainder(d.heading - h, 2 * math.pi)) <= (0.15 if in_join else 0.01), (name, dname)


def test_minoge_sites_unchanged():
    sec, pads, bounds = _track()
    ref = _ref()["sites"]
    turn = next(p for p in pads if p["name"] == "minoge_turnaround")
    for (lat, lon, z, h), (x, y, zc, hc) in zip(ref["turnaround_pits"], turn["pit_points"]):
        assert math.dist(_xy(lat, lon), (x, y)) <= 0.1 and abs(zc - z) <= 0.1
    houses = [(np.asarray(p["polygon"])[:-1].mean(0), p["z"]) for p in pads if p["name"].startswith("minoge_house_")]
    assert len(houses) == len(ref["houses"])
    for lat, lon, z in ref["houses"]:
        c, zc = min(houses, key=lambda h: math.dist(h[0], _xy(lat, lon)))
        assert math.dist(c, _xy(lat, lon)) <= 0.1 and abs(zc - z) <= 0.1
