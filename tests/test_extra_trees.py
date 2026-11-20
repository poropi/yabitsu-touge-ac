import csv
import json
from zipfile import ZipFile

import numpy as np
import pytest
import shapely

from yabitsu import csptree, forest_place, geo, terrain
from yabitsu.crosssection import Section


FIELDS = "site,section,s_m,side,edge_offset_m,species,height_m,group,note".split(",")


@pytest.fixture
def inputs(tmp_path):
    from scripts import add_extra_trees as tool

    s = np.array([0., 10., 20., 30.])
    zero = np.zeros(len(s))
    sec = Section(s, s + 10, zero + 10, zero, zero, zero, zero, zero.astype(bool),
                  zero + 2, zero + 3, zero, zero)
    sec.save_json(tmp_path / "track_section.json")
    np.savez(tmp_path / "track_profile.npz", bounds={"s4": (10., 30.)})
    axis = np.arange(0., 51.)
    grid = dict(xs=axis, ys=axis, Z=np.full((51, 51), 10.), D=np.zeros((51, 51)))
    np.savez(tmp_path / "track_terrain_near.npz", **grid,
             valley_vertices=[[20, 14, 20], [26, 14, 26], [20, 20, 20]], valley_faces=[[0, 1, 2]])
    np.savez(tmp_path / "track_terrain_mid.npz", **grid)
    forest = tmp_path / "forest"
    lists = forest / "extension/trees"
    lists.mkdir(parents=True)
    trees = np.array([(20, 15, 10, "broadleaf_b", 21, .8), (29, 30, 10, "broadleaf_b", 37, 1.2)],
                     dtype=forest_place.DTYPE)
    np.save(forest / "trees.npy", trees)
    original = b"# comment\r\n" + b"".join(
        (csptree.tree_line("broadleaf_b.zip", x, z, -y, angle, scale) + "\r\n").encode()
        for x, y, z, _, angle, scale in trees)
    (lists / "forest_0_0.txt").write_bytes(original)
    (lists / "forest_1_1.txt").write_bytes(b"# another tile\n")
    with ZipFile(lists / "broadleaf_b.zip", "w") as pkg:
        pkg.writestr("tree.ini", "[BASIC]\nHEIGHT=10\n")
    for name in ("track", "dress_mountain"):
        root = tmp_path / name
        root.mkdir()
        (root / "manifest.json").write_text('{"meshes":{}}')
    building_file = tmp_path / "track_buildings.json"
    building_file.write_text("[]")
    table = tmp_path / "extra.csv"
    return tool, forest, table, building_file, original


def row(**changes):
    return dict(site="A", section="s4", s_m=2.5, side="L", edge_offset_m=3.5,
                species="broadleaf_b", height_m=10, group="always", note="") | changes


def write_csv(path, *rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare(inputs):
    tool, forest, table, buildings, _ = inputs
    return tool.prepare(forest, table, table.parent, buildings)


def test_appends_original_bytes_and_uses_drawn_valley_surface(inputs, tmp_path):
    tool, forest, table, _, original = inputs
    write_csv(table, row(), row(s_m=3.5, side="R", height_m=15))
    with pytest.warns(UserWarning, match="3 行目.*倍率"):
        patch = prepare(inputs)
    tool.write_copy(forest, tmp_path / "one", patch)
    tool.write_copy(forest, tmp_path / "two", patch)
    trees = np.load(tmp_path / "one/trees.npy").view(np.recarray)
    assert trees[:2].tobytes() == np.load(forest / "trees.npy").tobytes()
    assert (trees.x[2], trees.y[2], trees.z[2]) == pytest.approx((22.5, 15.5, 22.5))
    assert (trees.x[3], trees.y[3], trees.z[3], trees.scale[3]) == pytest.approx((23.5, 3.5, 10, 1.2))
    for relative in ("extension/trees/forest_0_0.txt", "extension/trees/forest_1_1.txt", "trees.npy"):
        assert (tmp_path / "one" / relative).read_bytes() == (tmp_path / "two" / relative).read_bytes()
    tile = tmp_path / "one/extension/trees/forest_0_0.txt"
    added = tile.read_bytes()[len(original):].splitlines()
    assert len(added) == 2 and added[0].startswith(b"tree: broadleaf_b.zip; pos=22.50, ")
    assert abs(float(added[0].split(b", ")[1]) - 22.5) <= .02
    assert tile.read_bytes().startswith(original)
    assert (forest / "extension/trees/forest_0_0.txt").read_bytes() == original
    assert (tmp_path / "one/extension/trees/forest_1_1.txt").read_bytes() == b"# another tile\n"
    assert list(patch.tiles) == ["forest_0_0.txt"]


@pytest.mark.parametrize(("name", "mat", "verts", "reason"), [
    ("1ROAD_side_stub_00", "asphalt", [(22, 15, 20), (23, 15, 20), (22.5, 16, 20)], "舗装"),
    ("smallspan_02_deck", "valley_soffit", [(22, 15, 20), (23, 15, 20), (22.5, 16, 20)], "橋"),
    ("smallspan_02_abutment_start", "concrete_moss", [(22, 15, 20), (23, 15, 20), (22.5, 16, 23)], "壁"),
    ("water_012_valley_00", "water", [(22, 15, 23), (23, 15, 23), (22.5, 16, 23)], "水面"),
    ("GUARDRAIL_s1_003", "guardrail", [(22, 15, 20), (23, 15, 20), (23, 15, 22)], "ガードレール"),
    ("1WALL_track_fence_018", "railing_bars", [(22, 15, 20), (23, 15, 20), (23, 15, 22)], "柵"),
])
def test_rejects_meshes(inputs, tmp_path, name, mat, verts, reason):
    _, _, table, _, _ = inputs
    y = 15.49 if "rail" in mat else 15.5
    write_csv(table, row(edge_offset_m=y - 12))
    root = tmp_path / "track"
    np.savez(root / "mesh.npz", verts=verts, faces=[[0, 1, 2]])
    (root / "manifest.json").write_text(json.dumps({"meshes": {name: {"file": "mesh.npz", "material": mat}}}))
    patch = prepare(inputs)
    assert not len(patch.additions)
    assert len(patch.errors) == 1 and "2 行目" in patch.errors[0] and reason in patch.errors[0]


def test_reports_bad_rows_and_checks_building_condition(inputs, tmp_path):
    _, _, table, building_file, _ = inputs
    xs, ys = zip(*shapely.box(22, 15, 23, 16).exterior.coords)
    lat, lon = geo.from_local(np.array(xs), np.array(ys))
    spec = dict(id="vm873_477", footprint=np.c_[lat, lon].tolist())
    building_file.write_text(json.dumps([spec]))
    write_csv(table, row(edge_offset_m=1.49), row(s_m=.5, edge_offset_m=3.),
              row(species="unknown"), row(), row(group="if_building_removed:vm873_477"))
    patch = prepare(inputs)
    assert not len(patch.additions)
    for ix, reason in enumerate(("路肩", "木の幹", "種類", "建物"), 2):
        assert f"{ix} 行目" in patch.errors[ix - 2] and reason in patch.errors[ix - 2]
    assert len(patch.errors) == 4 and patch.skipped == 1
    building_file.write_text(json.dumps([spec | {"gone": True, "keep_tree_clear": True}]))
    write_csv(table, row(group="if_building_removed:vm873_477"))
    patch = prepare(inputs)
    assert len(patch.additions) == 1 and not patch.errors


def test_scope_permits_only_csv_additions(inputs, tmp_path):
    from scripts.check_valley_scope import forest_changes

    tool, forest, table, _, _ = inputs
    write_csv(table, row())
    patch = prepare(inputs)
    out = tmp_path / "out"
    tool.write_copy(forest, out, patch)

    class Scope:
        def __init__(self):
            self.issues = []

        def issue(self, *issue):
            self.issues.append(issue)

    scope = Scope()
    forest_changes(scope, forest, out, extra_trees=patch)
    assert not scope.issues
    forest_changes(scope, forest, out)
    assert {"tree_added", "tree_text"} <= {issue[2] for issue in scope.issues}
    scope.issues.clear()
    path = out / "extension/trees/forest_0_0.txt"
    file = path.read_bytes()
    path.write_bytes(file.replace(b"# comment", b"# changed"))
    orig = np.load(out / "trees.npy")
    for index in (0, -1):
        data = orig.copy()
        data["angle_deg"][index] += 1
        np.save(out / "trees.npy", data)
        forest_changes(scope, forest, out, extra_trees=patch)
        assert {i[1] for i in scope.issues} == {"trees.npy", "forest_0_0.txt"}
        scope.issues.clear()


def test_plan_converter_keeps_both_groups(tmp_path):
    from scripts.extra_trees_from_plan import convert

    plan = tmp_path / "plan.json"
    out = tmp_path / "out.csv"
    values = {k: v for k, v in row(note="木, 左側").items() if k not in ("group", "site")}
    plan.write_text(json.dumps({"site": "A-s4-1724", "trees": [values], "if_building_removed": [values]}))
    convert(plan, out)
    with out.open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    assert [(r["site"], r["group"], r["note"]) for r in rows] == [
        ("A-s4-1724", "always", "木, 左側"),
        ("A-s4-1724", "if_building_removed:vm873_477", "木, 左側")]


def test_csv_error_uses_physical_line_number(inputs):
    _, _, table, _, _ = inputs
    write_csv(table, row(note="一行目\n二行目"), row(side="不明"))
    content = table.read_text().replace("\nA,s4,2.5,不明", "\n\nA,s4,2.5,不明")
    table.write_text(content)
    assert prepare(inputs).errors[0].startswith("5 行目:")


def test_ground_uses_actual_mid_triangles_and_rejects_undrawn_ground():
    axis = np.arange(2.)
    near = dict(xs=axis, ys=axis, Z=np.full((2, 2), 20.), D=np.full((2, 2), 106.),
                H=np.array([[True]]))
    mid = dict(xs=axis, ys=axis, Z=np.array([[10., 10.], [10., 14.]]), D=np.full((2, 2), 106.))
    ground = terrain.terrain_mesh_z(near, mid, drawn_only=True)
    assert ground(.75, .25) == pytest.approx(11.)
    assert np.isnan(ground(10., 10.))
    near["H"][:] = False
    assert ground(.75, .25) == pytest.approx(20.)
