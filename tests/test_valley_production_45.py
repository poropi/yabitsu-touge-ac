"""The production gate and deploy list can be exercised without a production build."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from yabitsu import config, valley_edits
from yabitsu.manifest import Manifest
from yabitsu.valley_site_import import source_fingerprint


WORK = config.PROJECT_ROOT / "export/compare/route70/work"
SITE = "B-s4-6354"
NAME = f"valley_edit_{SITE}_stone"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(WORK))

    def load(name):
        assert (WORK / f"{name}.py").is_file(), f"依頼45の道具がありません: {name}"
        return importlib.import_module(name)

    return load


def test_freshness_matches_the_blend_and_fingerprints_all_edit_files(modules, tmp_path):
    gate = modules("validate_valleys_45")
    sites, edits = tmp_path / "sites", tmp_path / "edits"
    (sites / SITE).mkdir(parents=True)
    (edits / SITE).mkdir(parents=True)
    site = dict(site=SITE, fingerprint={"terrain": "source"}, parts=[])
    (sites / SITE / "site.json").write_text(json.dumps(site))
    blend = sites / SITE / "site.blend"
    blend.write_bytes(b"edited blend")
    (edits / SITE / "edit.json").write_text(json.dumps(
        dict(fingerprint=source_fingerprint(site), blend_sha256=valley_edits.file_digest(blend))))
    terrain = edits / SITE / "terrain.npz"
    terrain.write_bytes(b"terrain")
    first = gate.edit_snapshot(sites, edits)
    terrain.write_bytes(b"changed terrain")
    assert gate.edit_snapshot(sites, edits) != first
    unedited = sites / "A-s4-1724"
    unedited.mkdir()
    unedited_blend = unedited / "site.blend"
    unedited_blend.write_bytes(b"original")
    (unedited / "site.json").write_text(json.dumps(
        {"baseline_blend_sha256": valley_edits.file_digest(unedited_blend)}))
    gate.edit_snapshot(sites, edits)
    unedited_blend.write_bytes(b"first edit, before import")
    with pytest.raises(ValueError, match="A-s4-1724.*site.blend"):
        gate.edit_snapshot(sites, edits)
    unedited_blend.write_bytes(b"original")
    blend.write_bytes(b"saved again")
    with pytest.raises(ValueError, match="site.blend"):
        gate.edit_snapshot(sites, edits)


def part(root, name, material, *, group="yabitsu_s4"):
    m = Manifest(root)
    m.add_material(material, texture=f"textures/{material}.png")
    p = root / m.materials[material]["texture"]
    p.parent.mkdir()
    p.write_bytes(b"original pixels")
    m.add_mesh(group, name, [[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 1, 2]],
               [[0, 0], [1, 0], [1, 1]], material)
    m.save()
    return p


def test_final_inventory_requires_exact_dedicated_material_and_image(modules, tmp_path):
    gate = modules("validate_valleys_45")
    track = tmp_path / "track"
    image = part(track, "smallspan_16_deck_edit_stone", NAME)
    expected = gate.edit_inventory(track)
    assert expected["meshes"]["smallspan_16_deck_edit_stone"]["groups"] == ["yabitsu_s4"]
    assert expected["materials"][NAME]["texture"].endswith(".png")
    assert gate.inventory_mismatches(expected, gate.edit_inventory(track)) == {}
    image.write_bytes(b"changed pixels")
    mismatch = gate.inventory_mismatches(expected, gate.edit_inventory(track))
    assert mismatch == {"textures": [f"textures/{NAME}.png"]}
    image.with_suffix(".dds").write_bytes(b"unapproved texture")
    assert f"textures/{NAME}.dds" in gate.inventory_mismatches(expected, gate.edit_inventory(track))["textures"]


def test_deploy_list_marks_embedded_texture_changes_in_its_kn5_only(modules, tmp_path):
    deploy = modules("deploy_list_45")
    a, b = tmp_path / "old", tmp_path / "new"
    part(a, "existing", NAME)
    changed = part(b, "existing", NAME)
    assert deploy.kn5_changes(a, b) == {}
    changed.write_bytes(b"changed pixels")
    groups = deploy.kn5_changes(a, b)
    assert set(groups) == {"yabitsu_s4.kn5"}
    assert groups["yabitsu_s4.kn5"]["textures"] == [f"textures/{NAME}.png"]


def test_deploy_list_compares_payload_arrays_instead_of_npz_container(modules, tmp_path):
    deploy = modules("deploy_list_45")
    a, b = tmp_path / "old", tmp_path / "new"
    part(a, "existing", "ground")
    part(b, "existing", "ground")
    mesh = b / "meshes/existing.npz"
    with np.load(mesh) as src:
        arrays = dict(src)
    np.savez(mesh, **arrays)
    assert deploy.kn5_changes(a, b) == {}
    arrays["verts"][0, 2] = .1
    np.savez(mesh, **arrays)
    assert deploy.kn5_changes(a, b)["yabitsu_s4.kn5"]["meshes"] == ["existing"]


def test_deploy_list_excludes_forest_regeneration_backups(modules, tmp_path):
    deploy = modules("deploy_list_45")
    before, after = tmp_path / "before", tmp_path / "after"
    for root in (before, after):
        trees = root / "extension/trees"
        trees.mkdir(parents=True)
        (trees / "forest_-1_0.txt").write_bytes(b"tree row")
    (after / "extension/trees/forest_-1_0_before_valleys.txt").write_bytes(b"backup")
    old = deploy.files(before / "extension/trees", "forest_*.txt")
    new = deploy.files(after / "extension/trees", "forest_*.txt")
    assert deploy.files_changed(old, new, prefix="extension/trees/") == {"send": [], "remove": []}
