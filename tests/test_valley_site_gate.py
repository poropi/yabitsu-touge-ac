import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def baseline_gate_reports(tmp_path_factory):
    import hashlib

    from scripts.valley_site_render import ROOT, draw_gate, prepare_views
    from yabitsu import config

    reports = {}
    for site in ("A-s4-1724", "B-s4-6354", "C-s5-8956"):
        directory = ROOT / "export" / "blender_sites" / site
        meta = json.loads((directory / "site.json").read_text(encoding="utf-8"))
        original = directory / "site_original.blend"
        with original.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        assert expected == meta["baseline_blend_sha256"], site
        for variant in ("gate_original", "gate"):
            path = directory / variant / "flat_regions.json"
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                if report["blend_sha256"] == expected and all("highlights" in row for row in report["views"]):
                    reports[site] = report
                    break
        else:
            reports[site] = draw_gate(
                meta, prepare_views(meta["views"]), original, config.BLENDER_EXE,
                tmp_path_factory.mktemp("valley_original_gate"),
            )
    return reports


def test_crop_game_frame_preserves_centre_and_full_vertical_field():
    from yabitsu import valley_site_gate as gate

    pixels = np.zeros((50, 200, 3), dtype=np.uint8)
    pixels[:, 60:140] = (42, 74, 106)
    result = gate.centre_frame(Image.fromarray(pixels), (80, 50))
    np.testing.assert_array_equal(np.asarray(result), pixels[:, 60:140])


def test_regions_require_flat_colour_area_contrast_and_visible_geometry():
    from yabitsu import valley_site_gate as gate

    rgb = np.full((120, 160, 3), (110, 72, 40), dtype=np.uint8)
    surface = np.ones(rgb.shape[:2], dtype=bool)
    rgb[:14] = (135, 178, 225)
    surface[:14] = False
    rgb[18:41, 8:49] = (161, 163, 53)    # yellow-green embankment
    rgb[51:77, 55:120] = (120, 123, 125)  # grey stream bed
    rgb[83:114, 103:150] = (238, 233, 221)  # white retaining wall
    rgb[84:92, 8:16] = (250, 25, 180)    # below 0.5%
    rgb[95:114, 20:70] = (113, 74, 42)   # same colour as surrounding ground

    regions = gate.flat_regions(rgb, surface)
    assert [region["mean_rgb"] for region in sorted(regions, key=lambda r: r["bbox"][1])] == [
        [161.0, 163.0, 53.0], [120.0, 123.0, 125.0], [238.0, 233.0, 221.0],
    ]
    assert all(region["fraction"] >= 0.005 for region in regions)


def test_sheet_has_two_or_three_ordered_panels():
    from yabitsu import valley_site_gate as gate

    game = Image.new("RGB", (80, 50), (255, 0, 0))
    render = Image.new("RGB", (80, 50), (0, 255, 0))
    photo = Image.new("RGB", (160, 100), (0, 0, 255))
    for optional in (None, photo):
        sheet = gate.comparison_sheet(game, render, optional, "no_trees")
        assert sheet.width == 80 * (2 if optional is None else 3)
        colors = [sheet.getpixel((x, sheet.height - 1)) for x in range(40, sheet.width, 80)]
        assert colors == ([(0, 0, 255)] if optional else []) + [(255, 0, 0), (0, 255, 0)]


def test_baseline_gate_attributes_flat_patches(baseline_gate_reports):
    b = baseline_gate_reports["B-s4-6354"]["views"]
    candidates = next(row["regions"] for row in b
                      if row["view"] == "val38_B_rail_to_river" and row["mode"] == "no_trees")
    assert any(label["object"] == "valley_terrain" and label["material"] == "gravel_dark"
               for row in candidates for label in row["surfaces"])
    for report in baseline_gate_reports.values():
        for view in report["views"]:
            for region in view["regions"]:
                assert any(row["kind"] == "editable" for row in region["surfaces"])
                assert sum(label["pixels"] for label in region["surfaces"]) == region["pixels"]
            for region in view["background_regions"]:
                assert all(row["kind"] != "editable" for row in region["surfaces"])


def test_baseline_gate_measures_unclipped_surfaces_and_highlights(baseline_gate_reports):
    frame_count = 0
    patches = []
    for report in baseline_gate_reports.values():
        for view in report["views"]:
            if view["mode"] != "no_trees":
                continue
            frame_count += 1
            patches.extend(view["brightness_reference"])
            row = view["highlights"]
            assert row["warning"] == (row["difference_pp"] >= 5), view["view"]
    assert frame_count == 25
    assert len(patches) >= 12
    assert all(patch["game_clipped_percent"] == 0 for patch in patches)
    assert abs(float(np.median([patch["delta"] for patch in patches]))) <= 10


def test_brightness_report_uses_luma_and_warns_at_five_percentage_points():
    from yabitsu.valley_site_gate import highlight_fraction

    game = np.full((10, 20, 3), 100, dtype=np.uint8)
    render = game.copy()
    render[0, :10] = (250, 250, 250)
    render[1, :10] = (255, 0, 255)
    assert highlight_fraction(game, render) == {
        "game_percent": 0.0, "blender_percent": 5.0, "difference_pp": 5.0, "warning": True,
    }


def test_regions_attribute_patch_pixels_and_compare_by_surface():
    from yabitsu.valley_site_gate import compare_regions, flat_regions

    rgb = np.full((100, 100, 3), 70, dtype=np.uint8)
    rgb[25:65, 25:65] = 200
    pixels = np.zeros((100, 100), dtype=int)
    pixels[25:65, 25:50] = 1
    pixels[25:65, 50:65] = 2
    labels = [
        {"object": "road", "material": "asphalt", "collection": "LOCK_context"},
        {"object": "bank", "material": "soil", "collection": "EDIT_terrain"},
        {"object": "tree", "material": "leaves", "collection": "REF_trees"},
    ]
    patch = next(region for region in flat_regions(rgb, pixel_ids=pixels, labels=labels)
                 if region["mean_rgb"] == [200., 200., 200.])
    assert patch["kind"] == "editable"
    assert {row["object"] for row in patch["surfaces"]} == {"bank", "tree"}
    assert sum(row["pixels"] for row in patch["surfaces"]) == patch["pixels"]
    assert compare_regions([patch], [patch]) == {"added": [], "removed": [], "retained": [
        {"before": patch, "after": patch},
    ]}


def test_relight_rolls_back_if_publication_fails(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path

    from scripts import valley_site_relight

    before = {
        "site_original.blend": b"original scene",
        "site.blend": b"edited scene",
        "baseline_scene.npz": b"original baseline",
        "site.json": json.dumps({"baseline_blend_sha256":
                                 hashlib.sha256(b"original scene").hexdigest()}).encode(),
    }
    for name, data in before.items():
        (tmp_path / name).write_bytes(data)

    def fake_blender(command, *, check):
        out = command.index("--") + 1
        Path(command[out]).write_bytes(b"updated scene")
        for path in command[out + 2:]:
            Path(path).write_bytes(b"updated baseline")

    replace = Path.replace

    def interrupted(source, target):
        if source.name == "site.json" and source.parent.name.startswith(".relight-"):
            raise OSError("publication interrupted")
        return replace(source, target)

    monkeypatch.setattr(valley_site_relight.subprocess, "run", fake_blender)
    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="publication interrupted"):
        valley_site_relight.relight(tmp_path, Path("blender"))
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
