"""保存済みの道路断面で、小橋の端部と仕上げの順番を確かめる。"""
import json

import numpy as np
import pytest

from scripts import dress_mountain, track_06_terrain
from yabitsu import (
    config, features, gsi, kn5, nagoki, side_roads, small_bridges, start_lot,
    terrain, track, tunnel, valleys, valley_fit,
)
from yabitsu.crosssection import Section

BEFORE = config.PROJECT_ROOT / "export/compare/route70/work/before_34"
PARTS = (
    ("278", "mt_rail_278_0_r70_cut_end"),
    ("281", "mt_rail_281_0_r70end_1"),
    ("281", "mt_rail_281_0_r70end_2"),
    ("281", "mt_smallspan_16_right_start_post"),
    ("281", "mt_smallspan_16_right_end_post"),
    ("394", "mt_rail_394_0_r70end_1"),
    ("394", "mt_rail_394_0_r70end_2"),
    ("394", "mt_smallspan_16_left_start_post"),
    ("394", "mt_smallspan_16_left_end_post"),
    ("366", "mt_smallspan_25_right_start_post"),
)
pytestmark = pytest.mark.skipif(
    not all((BEFORE / p).exists() for p in ("dress_mountain/manifest.json", "track_terrain_near.npz")),
    reason="組み立て前の控えが必要",
)


class Parts:
    def __init__(self):
        self.meshes = {}

    def add_mesh(self, group, name, verts, faces, uvs, material, renderable=True):
        self.meshes[name] = verts[kn5._triangles(faces)]


@pytest.fixture(scope="module")
def generated():
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    bounds = track.load_bounds()
    feats = dress_mountain.track_features(sec, bounds)
    rows = dress_mountain.table_rows()
    spans = small_bridges.spans(sec, bounds, small_bridges.load())
    with np.load(BEFORE / "track_terrain_near.npz") as saved:
        arrays = dict(saved)
    pads = json.loads((config.DATA_PROCESSED / "track_pads.json").read_text(encoding="utf-8"))
    pads += start_lot.flats_for(config.DATA_MANUAL / "s1_start_lot.json", pads)
    sampler = gsi.default_sampler(gsi.TileCache(offline=True))
    initial = features.load_features_rows(track.load_section_rows("features", bounds))
    _, initial = nagoki.track_stubs(sec, pads, initial, sampler=sampler)
    keep = tunnel.carve_mask(sec.s, tunnel.from_osm(side_roads.load_elements(), sec))
    track_06_terrain.update_valley_ground(
        sec, arrays, valleys.load(), bounds, sampler,
        track_06_terrain.terrain_features(initial, bounds), pads=pads, keep_rows=keep,
    )
    with np.load(config.DATA_PROCESSED / "track_terrain_mid.npz") as mid:
        ground = terrain.terrain_mesh_z(arrays, dict(mid))
    result = Parts()
    dress_mountain.build_parts(sec, bounds, feats, rows, result, spans, ground)
    return result.meshes


@pytest.mark.parametrize(("number", "end"), PARTS)
def test_valley_rail_terminal_has_no_overlap(generated, number, end):
    body = f"mt_rail_{number}_0_r70span_0"
    checks = valley_fit.overlap_checks([(body, generated[body]), (end, generated[end])])
    count = sum(p.count for p in checks if p.mesh != p.other)
    assert count == 0, (body, end, count)


def test_bridge_16_existing_posts_keep_their_kn5_order(generated):
    before = json.loads((BEFORE / "dress_mountain/manifest.json").read_text(encoding="utf-8"))
    old = before["groups"][dress_mountain.GROUP]["meshes"]
    common = set(old) & set(generated)
    assert [name for name in old if name in common] == [name for name in generated if name in common]
