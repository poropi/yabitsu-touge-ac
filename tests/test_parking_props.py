"""駐車場への侵入・向き反転・部品欠落・書出し破損を検出する。"""
import importlib
import importlib.util
import json

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from yabitsu import config, kn5, start_lot


def _module():
    assert importlib.util.find_spec("yabitsu.parking_props") is not None, "駐車小物の生成器が必要"
    return importlib.import_module("yabitsu.parking_props")


def test_car_and_person_have_nondegenerate_real_size_meshes_and_distinct_parts():
    pp = _module()
    for parts, min_h, max_h in ((pp.car_meshes("car", "body_silver"), 1.35, 1.65),
                                ([p[:4] for p in pp.person_meshes("person", "Male_Adult_05")], 1.65, 1.85)):
        vertices = np.concatenate([v for _, v, _, _ in parts])
        assert min_h < np.ptp(vertices[:, 2]) < max_h
        for name, v, f, mat in parts:
            assert np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v), name
            tris = v[f]
            assert (np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) > 1e-9).all(), name
    names = [p[0] for p in pp.car_meshes("car", "body_silver")]
    assert sum("tyre" in n for n in names) == 4
    assert sum(n.startswith(("car_wheel_-1_", "car_wheel_1_")) for n in names) == 4
    assert sum("mirror" in n for n in names) == 2
    assert any("window" in n for n in names) and any("headlamp" in n for n in names)
    parts = pp.person_meshes("person", "Male_Adult_05")
    assert {p[3] for p in parts} == {"person_m009_body", "person_m009_head", "person_m009_opacity"}
    v = np.concatenate([p[1] for p in parts]); uv = np.concatenate([p[4] for p in parts])
    assert (uv >= -0.01).all() and (uv <= 1.01).all()
    assert v[:, 2].min() == 0.0 and np.ptp(v[:, 0]) < 0.8   # 足の裏が 0、腕を下ろしている（両腕を広げると 1.1 m）
    assert pp.person_textures("Male_Adult_05")["person_m009_opacity"][1] is True


def test_real_placements_stay_in_stalls_and_clear_driving_and_pit_space():
    pp = _module()
    spec = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    settings = json.loads((config.DATA_MANUAL / "parking_props.json").read_text())
    frame = start_lot.frame(spec)
    pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text())
    pad = next(p for p in pads if p["name"] == "nanohanadai_lot")
    lot = Polygon(pad["polygon"])
    parts = pp.placed_meshes(spec, pad["z"], settings)
    assert len(settings["cars"]) == 3 and 2 <= len(settings["people"]) <= 3
    for idx, car in enumerate(settings["cars"]):
        if car.get("ac_car"):
            continue
        v = np.concatenate([v for name, v, *_ in parts if name.startswith(f"park_car_{idx}_")])
        a, b = frame.ab(v[:, :2])
        assert a.max() - a.min() < spec["stalls"]["pitch"] - 0.1
        assert b.min() > -2.5 and b.max() < 2.5
        assert lot.covers(MultiPoint(v[:, :2]).convex_hull)
        assert a.max() < -30  # ピット列の車体だけでなく出入りも空ける
        assert abs(v[:, 2].min() - pad["z"]) < 0.025
    for idx, person in enumerate(settings["people"]):
        v = np.concatenate([v for name, v, *_ in parts if name.startswith(f"park_person_{idx}_")])
        a, b = frame.ab(v[:, :2])
        assert b.min() > spec["fence"]["b"] + 0.25 and b.max() < 0
        assert lot.covers(MultiPoint(v[:, :2]).convex_hull)
    # 3台すべてを詰め込んだり、人物を車内へ置いたりしない。
    footprints = []
    for kind, items in (("car", settings["cars"]), ("person", settings["people"])):
        for idx in range(len(items)):
            if kind == "car" and items[idx].get("ac_car"):   # AC の車は全長 4.5 m の箱で見る
                (x, _, mz), (lx, _, lz) = (lambda c: (c["pos"], c["look"]))(pp.ac_cars(spec, pad["z"], {"cars": [items[idx]]})[0])
                c = np.array([x, -mz]); fwd = np.array([lx, -lz]); n = np.array([-fwd[1], fwd[0]])
                footprints.append(MultiPoint([c + fwd * sa * 2.25 + n * sb * 0.9 for sa in (1, -1) for sb in (1, -1)]).convex_hull)
                continue
            v = np.concatenate([v for name, v, *_ in parts if name.startswith(f"park_{kind}_{idx}_")])
            footprints.append(MultiPoint(v[:, :2]).convex_hull)
    assert all(a.distance(b) > 0.35 for i, a in enumerate(footprints) for b in footprints[i + 1:])


def test_manifest_and_kn5_roundtrip_preserve_all_parts(tmp_path):
    pp = _module()
    out = pp.build(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    model = kn5.read_kn5(out)
    names = [c["name"] for c in model["root"]["children"]]
    two_sided = [n for n, e in manifest["meshes"].items() if manifest["materials"][e["material"]].get("two_sided")]
    assert sorted(names) == sorted(list(manifest["meshes"]) + [n + "_back" for n in two_sided])   # 両面の材質は裏面の部品も書く
    assert not manifest["empties"]
    assert json.loads((tmp_path / "retex.json").read_text()) == {}
    assert json.loads((tmp_path / "hidden.json").read_text()) == []
    assert all(not name[0].isdigit() for name in manifest["meshes"])  # 非衝突の景観物
    assert "SCRIPT=parked_cars.lua" in (tmp_path / "extension" / "parts" / "parked_cars.ini").read_text()
    assert "loadKN5LOD" in (tmp_path / "extension" / "parked_cars.lua").read_text(encoding="utf-8")
    assert not any(n.startswith("park_car_") for n in manifest["meshes"])   # 車は全部 AC の車（Lua）


def test_side_glazing_is_planar():
    pp = _module()
    car = pp.car_meshes("car", "body_silver")
    for name, v, _, _ in car:
        if name.startswith(("car_window_-1_", "car_window_1_")):
            # 表側の窓面が平面であり、丸く膨らんだガラスにならない。
            assert np.linalg.svd(v[:4] - v[:4].mean(0), compute_uv=False)[-1] < 1e-8
    assert sum("spoke" in p[0] for p in car) >= 20


def test_ac_cars_are_placed_in_the_lot_by_lua_facing_the_valley():
    """ac_car の車は kn5 に入れず、Lua が各自の AC の車の LOD を読み込んで、駐車場の枠の中に谷の方を向けて置く。"""
    pp = _module()
    spec = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    settings = json.loads((config.DATA_MANUAL / "parking_props.json").read_text())
    pad = next(p for p in json.loads((config.DATA_PROCESSED / "track_pads.json").read_text()) if p["name"] == "nanohanadai_lot")
    cars = pp.ac_cars(spec, pad["z"], settings)
    assert len(cars) == sum(1 for c in settings["cars"] if c.get("ac_car"))
    frame = start_lot.frame(spec); lot = Polygon(pad["polygon"])
    for c, item in zip(cars, [c for c in settings["cars"] if c.get("ac_car")]):
        x, y, z = c["pos"]
        assert lot.contains(Point(x, -z)) and abs(y - pad["z"]) < 0.5
        assert abs(np.linalg.norm(c["look"]) - 1) < 1e-9 and c["look"][1] == 0
        a, b = frame.ab(np.array([[x, -z]]))
        assert abs(a[0] - item["a"]) < 1e-6 and abs(b[0] - item["b"]) < 1e-6
    lua = pp.cars_lua(cars)
    assert lua.count("{id=") == len(cars) and "loadKN5LOD" in lua and "pcall" in lua


def test_settle_puts_all_four_tyres_on_a_sloped_lot():
    import pytest
    import numpy as np
    from yabitsu import parking_props as pp
    slope = lambda x, y: 100.0 + 0.04 * np.asarray(x, float) - 0.02 * np.asarray(y, float)   # 東へ 4 % 上り、北へ 2 % 下り
    wheels = {"front": [0.77, -0.24, 1.47], "rear": [0.77, -0.27, -1.04]}                    # 原点がタイヤの底より上にある車（RAV4）
    for h in (0.0, 1.2, -2.5):
        z0, look, up = pp.settle(slope, 3.0, -2.0, h, wheels)
        L, U = np.array(look), np.array(up); R = np.cross(L, U)
        assert abs(L @ U) < 1e-9 and np.linalg.norm(U) == pytest.approx(1.0)
        d = np.array([np.cos(h), np.sin(h)]); n = np.array([-np.sin(h), np.cos(h)])
        for half, bottom, axle in wheels.values():
            for s in (1.0, -1.0):
                p = np.array([3.0, -2.0, z0]) + L * axle + U * bottom - R * s * half   # 傾けた車のタイヤの底
                assert p[2] - float(slope(p[0], p[1])) == pytest.approx(0.0, abs=0.003)
