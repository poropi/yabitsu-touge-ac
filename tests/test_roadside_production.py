"""合格済みの道ぎわの成果物と正式な出力を照合する。"""
import importlib
import json
import os
import re
import runpy
import sys

import numpy as np
from PIL import Image
import pytest

from scripts import dress_mountain, forest_build, roadside_forest
from yabitsu import config, ext_config


WORK = config.COMPARE_DIR / "route70/work"


def reference(path):
    if not path.exists():
        pytest.skip(f"合格時の比較資料がありません: {path}")
    return path


def test_roadside_trial_help_does_not_enter_the_build(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(roadside_forest, "WORK", tmp_path)
    monkeypatch.setattr(sys, "argv", ["roadside_57.py", "--help"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(config.PROJECT_ROOT / "scripts/roadside_57.py"), run_name="__main__")
    assert result.value.code == 0
    assert "--help" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


TRIAL_COMMANDS = [
    ("roadside_57", "--out", True, "inputs"),
    ("treecards_59", "--output", True, "make_atlas"),
    ("roadside_forest", "--out", True, "inputs"),
    ("audit_roadside_gap", "--out", False, "band"),
]


@pytest.mark.parametrize("module, flag, directory, entry", TRIAL_COMMANDS)
@pytest.mark.parametrize("args", ["no-output", "existing", "forced"])
def test_trial_output_guard(module, flag, directory, entry, args, tmp_path, monkeypatch, capsys):
    command = importlib.import_module(f"scripts.{module}")
    output = tmp_path / ("output" if directory else "output.json")
    if directory:
        output.mkdir()
        sentinel = output / "sentinel"
    else:
        sentinel = output
    sentinel.write_bytes(b"approved")
    monkeypatch.setattr(command, "OUTPUT" if hasattr(command, "OUTPUT") else "OUT", output, raising=False)
    argv = [module]
    if args != "no-output":
        argv.extend((flag, str(output)))
    if args == "forced":
        argv.append("--force")
    monkeypatch.setattr(sys, "argv", argv)

    class EnteredBuild(Exception):
        pass

    def stop(*args, **kwargs):
        raise EnteredBuild

    target = roadside_forest if module == "roadside_57" else command
    monkeypatch.setattr(target, entry, stop)
    if args == "forced":
        with pytest.raises(EnteredBuild):
            command.main()
    else:
        with pytest.raises(SystemExit) as result:
            command.main()
        assert result.value.code == 2
        error = capsys.readouterr().err
        assert "usage:" in error and flag in error
        if args == "existing":
            assert "--force" in error
    assert sentinel.read_bytes() == b"approved"


def test_production_roadside_can_enter_without_trial_module(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "scripts.roadside_forest", None)
    monkeypatch.delattr(sys.modules["scripts"], "roadside_forest", raising=False)

    class EnteredForest(Exception):
        pass

    def load_section(*args):
        raise EnteredForest

    monkeypatch.setattr(forest_build.Section, "load_json", load_section)
    with pytest.raises(EnteredForest):
        forest_build.roadside_additions(None, tmp_path, {"roadside_vegetation": {}})

    monkeypatch.setattr(dress_mountain.roadside_vegetation, "grass_selection",
                        lambda *args: ([], [], [], {}))
    assert dress_mountain.write_roadside_parts(tmp_path, config.DATA_PROCESSED, "") == (0, 0)


@pytest.mark.skipif(os.environ.get("YABITSU_VERIFY_ROADSIDE_FOREST") != "1",
                    reason="全線の森を再計算するため YABITSU_VERIFY_ROADSIDE_FOREST=1 が必要")
def test_forest_snapshot_matches_approved_roadside_lists(tmp_path):
    source = reference(WORK / "trees_54/forest")
    accepted = reference(WORK / "forest_57_full/extension/trees")
    trial = reference(WORK / "forest_57/extension/trees")
    packages = reference(roadside_forest.PACKAGES)
    cfg = json.loads(forest_build.CFG.read_text(encoding="utf-8"))
    output = tmp_path / "forest"
    forest_build.extend_snapshot(source, output, packages, cfg)

    old_tiles = roadside_forest.add_extra_trees.list_files(source)
    new_tiles = roadside_forest.add_extra_trees.list_files(output)
    assert new_tiles.keys() == old_tiles.keys()
    changed = added = 0
    for name, base in old_tiles.items():
        old = base.read_bytes()
        new = new_tiles[name].read_bytes()
        assert new == (accepted / name).read_bytes(), name
        assert new.startswith(old), name
        if new != old:
            changed += 1
            appended = new[len(old):]
            assert b"tree: mt_" not in appended
            added += appended.count(b"\n")
    assert (changed, added) == (26, 109_913)
    output_trees = output / "extension/trees"
    assert not list(output_trees.glob("mt_*.zip"))
    for name in ("fern_tall_a.zip", "fern_tall_b.zip"):
        assert (output_trees / name).read_bytes() == (trial / name).read_bytes()
    registered = set(re.findall(r"^LIST_\d+=trees\\(.+)$",
                                (output / "extension/ext_config.ini").read_text(encoding="utf-8"), re.M))
    assert registered == set(new_tiles)
    print(f"木の一覧 {len(new_tiles)} タイル一致、追記 {changed} タイル・{added:,} 行、削除 0、大きいシダ 2 ZIP 一致")


def test_mountain_roadside_parts_match_approved_sections(tmp_path):
    grass = reference(WORK / "forest_57/extension/parts/grass_fx.ini")
    stage = reference(WORK / "stage1_run3/ext_config.ini")
    finish = reference(config.DATA_PROCESSED / "dress_mountain/extension/dress_mountain.ini")
    mountain_finish = finish.read_text(encoding="utf-8")
    ext_config.write_part(tmp_path, "dress_mountain", mountain_finish)
    counts = dress_mountain.write_roadside_parts(tmp_path, config.DATA_PROCESSED, mountain_finish)
    bundled = ext_config.assemble(tmp_path).read_text(encoding="utf-8")
    bank = stage.read_text(encoding="utf-8").rsplit("[SHADER_REPLACEMENT_...]", 1)[-1]
    bank = "[SHADER_REPLACEMENT_...]" + bank
    parts = tmp_path / "extension/parts"
    assert counts == (225, 41)
    assert (parts / "grass_fx.ini").read_bytes() == grass.read_bytes()
    assert (parts / "valley_bank_ground.ini").read_text(encoding="utf-8").strip() == bank.strip()
    assert mountain_finish in bundled
    assert grass.read_text(encoding="utf-8") in bundled
    assert bank.strip() in bundled
    assert bundled.count("[GRASS_FX]") == 1
    assert bundled.count("MESHES=1GRASS_valley_?bank?") == 1
    print(f"GrassFX: {counts[0]} 対象、{counts[1]} 除外、草と谷岸の部品・統合設定が一致")


@pytest.mark.parametrize("key", ["mt_terrain", "mt_grass"])
def test_mountain_diffuse_matches_approved_dds(key, tmp_path):
    name = f"dress_{key}.dds"
    accepted = reference(WORK / "stage1_run3" / name)
    dress_mountain.write_retex_textures([key], tmp_path)
    actual = tmp_path / name
    if actual.read_bytes() == accepted.read_bytes():
        print(f"{name}: {actual.stat().st_size:,} バイト一致、最大画素差 0")
        return
    with Image.open(actual) as image, Image.open(accepted) as expected:
        assert image.size == expected.size
        diff = np.abs(np.asarray(image.convert("RGBA"), dtype=np.int16) - np.asarray(expected.convert("RGBA")))
    print(f"{name}: 最大画素差 {diff.max()}")
    assert diff.max() <= 1
