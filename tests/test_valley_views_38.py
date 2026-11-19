"""撮影視点の位置と、レビュー用の光線が手前の描画面を選ぶこと。"""

import importlib.util
from pathlib import Path
import re
import zipfile

import numpy as np
import pytest

from yabitsu.valley_fit import Mesh

WORK = Path(__file__).resolve().parents[1] / "export/compare/route70/work"


def load(path):
    assert path.is_file(), f"{path.name} がまだない"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_capture_definition_keeps_all_xy_directions_and_tri_views():
    mod = load(WORK / "cap/make_views_valleys_38.py")
    lua = "YabitsuFpsProbe/YabitsuFpsProbe.lua"
    with (
        zipfile.ZipFile(WORK / "cap/harness-cap-valleys.zip") as old,
        zipfile.ZipFile(WORK / "cap/harness-cap-valleys-38.zip") as new,
    ):
        original, updated = old.read(lua), new.read(lua)
        original_line = next(line for line in original.splitlines() if line.startswith(b"local views="))
        line = next(line for line in updated.splitlines() if line.startswith(b"local views="))
        old_rows = list(mod.VIEW.finditer(mod.original_views_line(original_line)))
        rows = list(mod.VIEW.finditer(line))
        assert len({r[1] for r in rows}) == len(old_rows) == len(rows) == 25
        for i, (a, b) in enumerate(zip(old_rows, rows)):
            assert a.group(1, 2, 4) == b.group(1, 2, 4)
            assert a.string[a.end(3):a.end()] == b.string[b.end(3):b.end()]
            if i < 6:
                assert a[0] == b[0]
        assert updated.replace(line, original_line) == original
        assert new.read("YabitsuFpsProbe/manifest.ini") == old.read("YabitsuFpsProbe/manifest.ini")
        dirs = np.array([
            np.fromstring(re.search(rb"dir=vec3\(([^)]+)\)", r[0])[1].decode(), sep=",")
            for r in rows[-7:]
        ])
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1, atol=0.0002)
        assert dirs[1, 1] < -0.75  # Bの欄干ぎわの視線が川へ向かう

    # 確定した水面water_012_valley_01の第8203面。追加カメラの中央が水へ届く。
    preview = load(WORK / "valley_preview_38.py")
    water_triangle = np.array(
        [[[-429.75, 4105.75, 440.5613618], [-429.75, 4106.00, 440.5200070], [-430.00, 4105.75, 440.5055444]]]
    )
    _, eye, target = mod.extra_views()[1]
    ray = (target - eye) / np.linalg.norm(target - eye)
    index, distance = preview.cast(eye, ray[None], water_triangle, np.ones(1, bool))
    assert index[0] == 0 and 11.0 < distance[0] < 11.5


def test_first_hit_uses_depth_and_culls_backface():
    preview = load(WORK / "valley_preview_38.py")
    # カメラは原点から北を見る。後方と裏面を除外し、奥の壁より手前の床版を選ぶ。
    front = np.array([[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [0.0, 1.0, 1.0]])
    tri = np.stack([front * 2, front * 3, front[::-1] * 0.5, front * -1])
    p = np.zeros(3)
    d = np.array([0.0, 1.0, 0.0])
    dirs = np.array([d, [1.0, 0.0, 0.0]])
    who, dist = preview.cast(p, dirs, tri, np.zeros(4, dtype=bool))
    assert who.tolist() == [0, -1]
    assert dist[0] == 2.0
    who, _ = preview.cast(p, dirs[:1], tri, np.array([False, False, True, False]))
    assert who.tolist() == [2]


def test_capture_collector_requests_all_25_views():
    # 実撮影を起動せず、取得対象を一覧できる入口を使う。
    import subprocess

    cap = WORK / "cap/cap_valleys_38.sh"
    assert cap.is_file()
    listed = subprocess.check_output(["bash", cap, "--list"], text=True).splitlines()
    assert len(listed) == len(set(listed)) == 25
    with zipfile.ZipFile(WORK / "cap/harness-cap-valleys-38.zip") as z:
        lua = z.read("YabitsuFpsProbe/YabitsuFpsProbe.lua").decode()
    assert listed == re.findall(r'name="([^"]+)",pos=vec3', lua)


def test_capture_b_side_down_preserves_its_eye_height_above_the_new_bank():
    # 地面の441.722 mと目高の1.608 mは依頼38の補助図と同じ箇所の測定値。
    with zipfile.ZipFile(WORK / "cap/harness-cap-valleys-38.zip") as z:
        lua = z.read("YabitsuFpsProbe/YabitsuFpsProbe.lua").decode()
    match = re.search(r'name="val_B_side_down",pos=vec3\([^,]+,([^,]+),', lua)
    assert float(match[1]) == 443.33


def test_camera_adjustment_keeps_original_eye_height_and_uses_final_road_surface():
    mod = load(WORK / "cap/make_views_valleys_38.py")
    line = (
        b'local views={{name="val_B_side_down",pos=vec3(1,1.61,-1),dir=vec3(0,0,1),fov=64.0},'
        b'{name="val_C_right",pos=vec3(5,5.6,-2),dir=vec3(1,0,0),fov=64.0},'
        b'{name="r70_0912_f",pos=vec3(4,7.5,-1),dir=vec3(0,0,1),fov=64.0}}'
    )
    road = Mesh(
        "1ROAD_s1_134",
        np.array([[4, 1, 5.1], [6, 1, 5.1], [6, 3, 5.1], [4, 3, 5.1]]),
        np.array([[0, 1, 2, 3]]),
    )
    updated, result = mod.adjusted_views(line, lambda x, y: x * 0., lambda x, y: x * 0. + 2.63, [road])
    np.testing.assert_allclose([r["z_m"] for r in result], [4.24, 6.7, 7.5])
    assert [r["support"] for r in result] == ["terrain", "road", "terrain"]
    assert updated.endswith(line[line.index(b'{name="r70_0912_f"'):])


def test_camera_adjustment_checks_ground_and_solid_structure():
    mod = load(WORK / "cap/make_views_valleys_38.py")
    line = b'local views={{name="val_C_side_down",pos=vec3(0,1.6,0),dir=vec3(0,0,1),fov=64.0}}'
    verts = np.array([
        [-1, -1, 1.5], [1, -1, 1.5], [1, 1, 1.5], [-1, 1, 1.5],
        [-1, -1, 1.7], [1, -1, 1.7], [1, 1, 1.7], [-1, 1, 1.7],
    ])
    faces = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    box = Mesh("smallspan_25_deck", verts, faces)
    ground = lambda x, y: x * 0.
    with pytest.raises(ValueError, match="地面までの高さ"):
        mod.adjusted_views(line, lambda x, y: x * 0. + 2., ground, [])
    with pytest.raises(ValueError, match="smallspan_25_deck"):
        mod.adjusted_views(line, ground, ground, [box])
    assert mod.structure_intersection(np.array([1., 0., 1.6]), box.triangles)
    assert not mod.structure_intersection(np.array([0., 0., 1.8]), box.triangles)
    _, rows = mod.adjusted_views(line, ground, ground, [Mesh("ceiling", verts, faces[1:2])])
    assert rows[0]["z_m"] == 1.6


def test_reproduction_requires_an_empty_directory_outside_inputs(tmp_path):
    tool = load(WORK / "reproduce_valleys_38.py")
    source = tmp_path / "read_only"
    source.mkdir()
    with pytest.raises(ValueError, match="入力"):
        tool.prepare_output(source / "empty", [source])
    occupied = tmp_path / "previous"
    occupied.mkdir()
    marker = occupied / "keep"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="空"):
        tool.prepare_output(occupied, [source])
    assert marker.read_text() == "preserve"
    link = tmp_path / "pointing_at_input"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="入力"):
        tool.prepare_output(link, [source])
    empty = tmp_path / "new"
    assert tool.prepare_output(empty, [source]) == empty
    assert list(empty.iterdir()) == []


def test_above_ground_supplement_keeps_xy_aim_and_original_eye_height():
    preview = load(WORK / "valley_preview_38.py")
    p = np.array([10.0, 20.0, 101.608])
    direction = np.array([1.0, 0.0, -0.1])
    camera = (p.copy(), direction.copy(), 64.0)
    new = preview.above_ground_view(camera, 100.0, 102.63)
    np.testing.assert_allclose(new[0], [10.0, 20.0, 104.238])
    np.testing.assert_array_equal(new[1], direction)
    assert new[2] == camera[2]
    np.testing.assert_array_equal(camera[0], p)
