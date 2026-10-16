import json

import numpy as np

from scripts.s1_check import check


def _setup(tmp_path, offset_ok=True, residual=0.5):
    p = tmp_path / "processed"; man = tmp_path / "manual"; (p / "s1" / "meshes").mkdir(parents=True); man.mkdir()
    s = np.arange(0.0, 100.0); z = 100 + 0.05 * s
    np.savez(p / "s1_profile.npz", s=s, z_raw=z, z_ref=z + residual, z=z, valid=np.ones(100, bool))
    (man / "s1_position_review.csv").write_text(
        "chunk,s_center,max_offset_m,ok,note\n0,0,1.0,yes,\n1,150," + ("1.5,yes" if offset_ok else "3.0,no") + ",\n", encoding="utf-8")
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    np.savez(p / "s1" / "meshes" / "1ROAD_s1_00.npz", verts=v, faces=np.array([[0, 1, 2, 3]]), uvs=v[:, :2])
    (p / "s1" / "manifest.json").write_text(json.dumps({"meshes": {"1ROAD_s1_00": {"file": "meshes/1ROAD_s1_00.npz", "faces": 1}},
                                                         "groups": {"yabitsu_s1": {"meshes": ["1ROAD_s1_00"]}}}), encoding="utf-8")
    return p, man


def test_all_ok(tmp_path):
    r = check(*_setup(tmp_path))
    assert r["elevation_ok"] and r["position_ok"] and r["folds"] == 0 and r["tris"] == 2


def test_position_ng(tmp_path):
    assert not check(*_setup(tmp_path, offset_ok=False))["position_ok"]


def test_elevation_ng(tmp_path):
    assert not check(*_setup(tmp_path, residual=2.0))["elevation_ok"]
