import json

import numpy as np

from scripts.track_check import check

BOUNDS = {"s2": (0.0, 50.0), "s1": (50.0, 99.0)}
HEAD = "chunk,s_center,max_offset_m,ok,note\n"


def _setup(tmp_path, offset_ok=True, residual=0.5, s2_ok=True):
    p = tmp_path / "processed"; man = tmp_path / "manual"; (p / "track" / "meshes").mkdir(parents=True); man.mkdir()
    s = np.arange(0.0, 100.0); z = 100 + 0.05 * s
    np.savez(p / "track_profile.npz", s=s, z_raw=z, z_ref=z + residual, z=z, valid=np.ones(100, bool),
             bounds=np.array(BOUNDS, dtype=object))
    (man / "s1_position_review.csv").write_text(
        HEAD + "0,0,1.0,yes,\n1,150," + ("1.5,yes" if offset_ok else "3.0,no") + ",\n", encoding="utf-8")
    (man / "s2_position_review.csv").write_text(HEAD + "0,0," + ("1.0,yes" if s2_ok else "2.5,no") + ",\n", encoding="utf-8")
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    np.savez(p / "track" / "meshes" / "1ROAD_s1_00.npz", verts=v, faces=np.array([[0, 1, 2, 3]]), uvs=v[:, :2])
    (p / "track" / "manifest.json").write_text(json.dumps({"meshes": {"1ROAD_s1_00": {"file": "meshes/1ROAD_s1_00.npz", "faces": 1}},
                                                            "groups": {"yabitsu_main": {"meshes": ["1ROAD_s1_00"]}}}), encoding="utf-8")
    return p, man


def test_all_ok(tmp_path):
    r = check(*_setup(tmp_path))
    assert r["elevation_ok"] and r["position_ok"] and r["folds"] == 0 and r["tris"] == 2
    assert r["position_rows"] == 3 and set(r["sections"]) == {"s2", "s1"}


def test_position_ng(tmp_path):
    assert not check(*_setup(tmp_path, offset_ok=False))["position_ok"]


def test_position_ng_in_second_section(tmp_path):
    assert not check(*_setup(tmp_path, s2_ok=False))["position_ok"]


def test_elevation_ng(tmp_path):
    r = check(*_setup(tmp_path, residual=2.0))
    assert not r["elevation_ok"] and r["sections"]["s1"]["max_residual_m"] > 1.5


def test_check_counts_folds_in_both_groups(tmp_path):
    p, man = _setup(tmp_path)
    v = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], float)   # 下向き（折り返し）
    np.savez(p / "track" / "meshes" / "1ROAD_town_00.npz", verts=v, faces=np.array([[0, 1, 2, 3]]), uvs=v[:, :2])
    m = json.loads((p / "track" / "manifest.json").read_text(encoding="utf-8"))
    m["meshes"]["1ROAD_town_00"] = {"file": "meshes/1ROAD_town_00.npz", "faces": 1}
    m["groups"]["yabitsu_nagoki"] = {"meshes": ["1ROAD_town_00"]}
    (p / "track" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    r = check(p, man)
    assert r["folds"] == 1 and r["tris"] == 4
