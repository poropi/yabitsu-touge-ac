"""軽い木の正式出力と、CSP 高木へ戻したときの互換性。"""
from collections import Counter
from dataclasses import replace
import filecmp
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from scripts import forest_build
from yabitsu import acdata, config, kn5, layouts

WORK = config.COMPARE_DIR / "route70/work"
PACKAGES = WORK / "supervisor_41.fmEVx6/deliver/forest/extension/trees"
KN5 = "yabitsu_trees_light.kn5"


FIXTURE = Path(__file__).parent / "fixtures/forest_csp_before_67"


def write_models(track_dir):
    for layout in layouts.load_layouts():
        path = track_dir / f"models_{layout.name}.ini"
        acdata.write_models_ini(path, layouts.kn5_files(layout.name, track_dir=track_dir))
        yield layout.name, path


@pytest.mark.parametrize("enabled", [True, False])
def test_models_ini_follows_destination_kn5(tmp_path, monkeypatch, light_settings, enabled):
    from yabitsu import forest_light

    monkeypatch.setattr(forest_light, "load_settings", lambda: replace(light_settings, enabled=enabled))
    for present in (False, True, False):
        (tmp_path / KN5).unlink(missing_ok=True)
        if present:
            (tmp_path / KN5).touch()
        for layout, path in write_models(tmp_path):
            actual = path.read_text(encoding="utf-8")
            assert actual.count(f"FILE={KN5}") == int(present)
            if present:
                assert actual.index(f"FILE={layout}_objects.kn5") < actual.index(f"FILE={KN5}")


def small_forest(root):
    out = root / "forest"
    trees = out / "extension/trees"
    trees.mkdir(parents=True)
    header = b"configure: size variance = 1, 1\r\n"
    low = b"tree: fern_tall_a.zip; pos=3, 4, 5; angle = 0; size = 0.56"
    first = header + b"tree: sugi_a.zip; pos=20, 203, -65; angle = 90; size = 1\n" + low
    second = header + b"tree: broadleaf_d.zip; pos=20, 207, -65.01; angle = 5; size = 1\r\n"
    originals = {"forest_0_0.txt": first, "forest_0_1.txt": second}
    for name, data in originals.items():
        (trees / name).write_bytes(data)
    for name in ("sugi_a", "broadleaf_d", "fern_tall_a"):
        with ZipFile(trees / f"{name}.zip", "w") as package:
            package.writestr("tree.ini", "[BASIC]\nHEIGHT=22\n")
    np.savez(root / "road.npz", verts=[[0, 0, 0], [50, 0, 0], [50, 50, 0], [0, 50, 0]], faces=[[0, 1, 2, 3]])
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"meshes": {"1ROAD_a": {"material": "asphalt", "file": "road.npz"}}}),
                        encoding="utf-8")
    return out, trees, manifest, originals, {name: header + data for name, data in zip(originals, (low, b""))}


@pytest.mark.parametrize("existing_track", [True, False])
def test_light_forest_output_keeps_csp_understory_and_installs_model(tmp_path, existing_track, light_settings):
    out, trees, manifest, originals, filtered = small_forest(tmp_path)
    lists = [f"trees\\{name}" for name in originals]
    low_zip = (trees / "fern_tall_a.zip").read_bytes()

    forest_build.finish_forest(out, lists, ["old_tree"], light=light_settings, manifest=manifest)

    assert {path.name: path.read_bytes() for path in trees.glob("*.txt")} == filtered
    assert {path.name: path.read_bytes() for path in trees.glob("*.zip")} == {"fern_tall_a.zip": low_zip}
    ini = (out / "extension/ext_config.ini").read_text(encoding="utf-8")
    assert "[TREES]\nACTIVE=1\nLIST_0=trees\\forest_0_0.txt\nLIST_1=trees\\forest_0_1.txt" in ini
    decoded = kn5.read_kn5(out / KN5)
    assert list(decoded["textures"]) == ["trees_light.dds"]
    mat, = decoded["materials"]
    assert mat["props"]["ksAlphaRef"] == pytest.approx(0.30)
    assert mat["props"]["ksAmbient"] == pytest.approx(0.45)
    assert mat["props"]["ksDiffuse"] == pytest.approx(0.35)
    near, far = decoded["root"]["children"]
    assert [len(mesh["indices"]) // 3 for mesh in (near, far)] == [244, 4]
    for mesh in (near, far):
        assert mesh["cast_shadows"]
        assert np.all(mesh["vertices"][:, 3:6] == [0, 1, 0])

    installed = tmp_path / "track"
    if existing_track:
        for layout in layouts.load_layouts():
            acdata.write_models_ini(installed / f"models_{layout.name}.ini",
                                    layouts.kn5_files(layout.name, track_dir=installed))
    forest_build.install(out, installed)
    assert filecmp.cmp(out / KN5, installed / KN5, shallow=False)
    assert not (installed / "extension/trees/sugi_a.zip").exists()
    if existing_track:
        for path in installed.glob("models_*.ini"):
            assert path.read_bytes().count(f"FILE={KN5}".encode()) == 1


def test_csp_mode_keeps_output_bytes_and_removes_installed_light_model(tmp_path, light_settings):
    out, trees, manifest, originals, _ = small_forest(tmp_path)
    lists = [f"trees\\{name}" for name in originals]
    prior = {path: path.read_bytes() for path in trees.iterdir()}

    forest_build.finish_forest(out, lists, ["old_tree"], light=replace(light_settings, enabled=False),
                               manifest=manifest)
    assert {path: path.read_bytes() for path in trees.iterdir()} == prior
    for relative in ("extension/ext_config.ini", "extension/parts/forest.ini"):
        assert (out / relative).read_bytes() == (FIXTURE / relative).read_bytes()

    installed = tmp_path / "track"
    installed.mkdir()
    (installed / KN5).touch()
    for _ in write_models(installed):
        pass
    forest_build.install(out, installed)
    assert not (installed / KN5).exists()
    assert (installed / "extension/trees/sugi_a.zip").is_file()
    assert (installed / "models_s1_up.ini").read_bytes() == (FIXTURE / "models_s1_up.ini").read_bytes()
    for path in installed.glob("models_*.ini"):
        assert f"FILE={KN5}".encode() not in path.read_bytes()


def test_near_threshold_keeps_original_csp_row_and_needed_zip(tmp_path, light_settings):
    out, trees, manifest, originals, filtered = small_forest(tmp_path)
    tile = trees / "forest_0_0.txt"
    short = b"tree: sugi_a.zip; pos=20, 203, -65; angle = 45; size = 0.20\r\n"
    boundary = b"tree: sugi_a.zip; pos=20, 203, -65; angle = 1; size = 0.35\n"
    tile.write_bytes(originals[tile.name].replace(b"tree: fern", short + boundary + b"tree: fern"))
    sugi_zip = (trees / "sugi_a.zip").read_bytes()
    lists = [f"trees\\{name}" for name in originals]

    forest_build.finish_forest(out, lists, [], light=replace(light_settings, near_scale=(0.35, 1.3)),
                               manifest=manifest)

    assert tile.read_bytes() == filtered[tile.name].replace(b"tree: fern", short + b"tree: fern")
    assert (trees / "sugi_a.zip").read_bytes() == sugi_zip
    assert not (trees / "broadleaf_d.zip").exists()
    near, far = kn5.read_kn5(out / KN5)["root"]["children"]
    assert len(near["indices"]) // 3 == 2 * 244
    assert len(far["indices"]) // 3 == 4


def test_csp_limit_stops_before_writing_a_kn5(tmp_path, light_settings):
    out, trees, manifest, _, _ = small_forest(tmp_path)
    tile = trees / "forest_0_0.txt"
    short = b"tree: sugi_a.zip; pos=20, 203, -65; angle = 45; size = 0.30\n"
    tile.write_bytes(short * 301)

    with pytest.raises(ValueError, match=r"301.*300.*6"):
        forest_build.finish_forest(out, [r"trees\forest_0_0.txt"], [],
                                   light=replace(light_settings, near_scale=(0.35, 1.3)), manifest=manifest)
    assert not (out / KN5).exists()
    assert tile.read_bytes() == short * 301


def test_forest_is_installed_before_failing_render(tmp_path, monkeypatch, light_settings):
    out, trees, manifest, originals, _ = small_forest(tmp_path)
    lists = [f"trees\\{name}" for name in originals]
    installed = tmp_path / "track"
    monkeypatch.setattr(forest_build, "MANIFEST", manifest.parent)
    monkeypatch.setattr(config, "TRACK_DIR", installed)
    monkeypatch.setattr(forest_build, "view_specs", lambda area, ground: [])

    def fail_render(command, **options):
        assert (installed / KN5).is_file()
        assert (installed / "extension/trees/forest_0_0.txt").is_file()
        assert not (installed / "extension/trees/sugi_a.zip").exists()
        assert (Path(command[-1]) / "sugi_a.zip").is_file()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(forest_build.subprocess, "run", fail_render)
    with pytest.raises(subprocess.CalledProcessError):
        forest_build.publish_forest(out, lists, light_settings, None, 0,
                                    SimpleNamespace(area="pilot", no_install=False, render=True))
    assert (installed / KN5).is_file()
    assert not list(out.glob(".render-trees-*"))


def rendered_near(nodes, placements, models, settings):
    from yabitsu import forest_light

    batches = forest_light.mesh_batches(placements, models, settings)
    trees = {}
    for node, (name, batch) in zip(nodes, batches, strict=True):
        assert node["name"] == name
        assert node["cast_shadows"] and node["active"] and node["visible"] and node["renderable"]
        assert not node["transparent"] and node["material"] == 0 and node["lod"] == (0., 0.)
        vertex = index = 0
        for item in batch:
            model = models[item.model]
            nverts = len(model.verts)
            nindices = model.faces.size
            vertices = node["vertices"][vertex:vertex + nverts]
            indices = node["indices"][index:index + nindices] - vertex
            np.testing.assert_array_equal(indices, model.faces.ravel())
            trees[id(item)] = vertices
            vertex += nverts
            index += nindices
        assert vertex == len(node["vertices"]) and index == len(node["indices"])
    return trees


def verify_approved_geometry_and_lists(out, settings):
    from yabitsu import forest_light

    source = WORK / "forest_57_full/extension/trees"
    approved = WORK / "treecards_59/lists_no_tall"
    reference = WORK / "treelight_66/yabitsu_treelight_r5.kn5"
    manifest = config.TRACK_BUILD_DIR / "manifest.json"
    for path in (source, approved, reference, manifest, PACKAGES):
        if not path.exists():
            pytest.skip(f"合格時の比較資料がありません: {path}")

    delivery = out / KN5
    lists = out / "lists"
    lists.mkdir(parents=True)
    stats = forest_light.build(source, PACKAGES, manifest, {settings.texture: delivery},
                               settings, filtered_lists=lists)
    print(f"KN5 書き出し完了: {delivery.stat().st_size:,} バイト", flush=True)

    models, _ = forest_light.load_assets(forest_light.CATALOG)
    edges = forest_light.road_edges(manifest)
    unclamped = replace(settings, near_scale=(0., settings.near_scale[1]))
    near, far, retained, count = forest_light.read_trees(source, PACKAGES, edges, unclamped, models)
    assert retained == set() and count == 0
    base = forest_light.place_near(near, edges, models, unclamped)
    before = [replace(item, scale=max(0.7, item.scale)) for item in base]
    kept = [item for item in base if item.scale >= 0.35]
    expected_csp = Counter(
        (tuple(item.xyz), settings.tall_families[family], height)
        for item, family, height in zip(base, near.family, near.height, strict=True) if item.scale < 0.35
    )
    assert len(base) == 33675 and len(far.xyz) == 965001
    assert stats["csp_trees"] == len(base) - len(kept) == 20

    def split_nodes(track):
        nodes = track["root"]["children"]
        return ([node for node in nodes if node["name"].startswith("treelight_near_")],
                [node for node in nodes if node["name"].startswith("treelight_far_")])

    old, new = kn5.read_kn5(reference), kn5.read_kn5(delivery)
    for key in ("version", "textures", "materials"):
        assert old[key] == new[key]
    np.testing.assert_array_equal(old["root"]["matrix"], new["root"]["matrix"])
    near_old, far_old = split_nodes(old)
    near_new, far_new = split_nodes(new)
    assert len(near_old) + len(far_old) == len(old["root"]["children"])
    assert len(near_new) + len(far_new) == len(new["root"]["children"])
    far_hash = hashlib.sha256()
    for a, b in zip(far_old, far_new, strict=True):
        for key in a:
            if key in ("vertices", "indices"):
                assert a[key].tobytes() == b[key].tobytes(), (a["name"], key)
                far_hash.update(a[key])
            else:
                assert a[key] == b[key], (a["name"], key)
    print(f"遠景 {len(far_new)} 部品・{sum(len(node['indices']) // 12 for node in far_new):,} 枚、頂点・面 SHA-256 {far_hash.hexdigest()}", flush=True)

    old_trees = rendered_near(near_old, before, models, settings)
    new_trees = rendered_near(near_new, kept, models, settings)
    changed = 0
    for old_item, item in zip(before, base, strict=True):
        if item.scale < 0.35:
            continue
        a, b = old_trees[id(old_item)], new_trees[id(item)]
        if old_item.scale == item.scale:
            assert a.tobytes() == b.tobytes()
        else:
            points = models[item.model].verts
            c, s = math.cos(item.angle), math.sin(item.angle)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            for placed, vertices in ((old_item, a), (item, b)):
                positions = ((points @ rotation.T * placed.scale + placed.xyz) @ kn5.C.T).astype("<f4")
                np.testing.assert_array_equal(vertices[:, :3], positions)
            np.testing.assert_array_equal(a[:, 3:8], b[:, 3:8])
            changed += 1
    assert changed == 7738
    print(f"近景 {len(old_trees):,} → {len(new_trees):,} 本、{changed:,} 本の倍率だけ変更、"
          f"{len(new_trees) - changed:,} 本の頂点は全属性一致", flush=True)

    pattern = forest_light.tall_pattern(settings)
    tiles = sorted(lists.glob("forest_*.txt"))
    assert {tile.name for tile in tiles} == {p.name for p in approved.glob("forest_*.txt")}
    assert len(tiles) == 168
    actual_csp = Counter()
    package_names = set()
    changed_tiles = 0
    for tile in tiles:
        lines, indices, names, rows = forest_light.read_tile(tile, pattern)
        assert forest_light.keep_other_lines(lines, indices) == (approved / tile.name).read_bytes(), tile.name
        if not names:
            continue
        changed_tiles += 1
        wanted = Counter(lines[i] for i in indices)
        expected, missing = [], wanted.copy()
        for line in (source / tile.name).read_bytes().splitlines(keepends=True):
            if not pattern.match(line):
                expected.append(line)
            elif missing[line]:
                missing[line] -= 1
                expected.append(line)
        assert not +missing and b"".join(expected) == tile.read_bytes()
        for name, (x, z, minus_y, angle, scale) in zip(names, rows, strict=True):
            model = name.decode()
            package = f"{model}.zip"
            height = forest_light.package_height(PACKAGES / package) * scale
            actual_csp[(x, -minus_y, z), model.split("_", 1)[0], height] += 1
            package_names.add(package)
    assert actual_csp == expected_csp
    assert package_names == set(stats["retained_packages"])
    print(f"CSP {len(tiles)} タイル、追記 {changed_tiles} タイル・{stats['csp_trees']} 行、"
          f"下草・設定行の全バイト一致。ZIP: {', '.join(sorted(package_names))}", flush=True)
    return stats


@pytest.mark.skipif(os.environ.get("YABITSU_VERIFY_LIGHT_TREES") != "1",
                    reason="全線のKN5照合は YABITSU_VERIFY_LIGHT_TREES=1 のときだけ実行")
def test_full_light_forest_matches_approved_geometry_and_lists(tmp_path, light_settings):
    verify_approved_geometry_and_lists(tmp_path, light_settings)
