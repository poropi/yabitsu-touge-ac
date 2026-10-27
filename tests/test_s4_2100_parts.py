import numpy as np
import pytest

from yabitsu import assembly, buildings as bd, geo, textures as tx, town_objects as to


def _ll(x, y):
    lat, lon = geo.from_local(float(x), float(y))
    return [float(lat), float(lon)]


def test_bollard_colour_cone_and_thick_bars():
    flat = lambda x, y: np.full(np.shape(x), 100.0)
    ms = to.object_meshes([{"kind": "bollard", "material": "pole_orange", "points": [_ll(0, 0)]},
                           {"kind": "bollard", "points": [_ll(2, 0)]},
                           {"kind": "cone", "points": [_ll(4, 0)]},
                           {"kind": "bar_fence", "material": "pole_brown", "bar_every_m": 0.25, "bar_w_m": 0.06, "points": [_ll(0, 5), _ll(2, 5)]}],
                          flat, prefix="o")
    by = {m[0]: m for m in ms}
    assert {"o_pole_orange", "o_post_red", "o_cone", "o_pole_brown"} <= set(by)   # 色を変えない車止めは今までどおり赤
    v = by["o_pole_brown"][1]
    first = v[np.isclose(v[:, 2], 100.12) & (v[:, 0] > 0.05) & (v[:, 0] < 0.2)][:, 0]   # 端から 0.125 m の最初の縦の棒の下端
    assert np.ptp(first) == pytest.approx(0.06, abs=0.002)                            # 縦の棒の太さ 6 cm


def test_plaza_material_passes_to_the_pad_mesh():
    class M:
        def __init__(self): self.mats = {}
        def add_mesh(self, g, name, v, f, uv, mat, renderable=True): self.mats[name] = mat
    m = M()
    sq = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]
    assembly.add_pads(m, [{"name": "a", "kind": "plaza", "polygon": sq, "z": 1.0},
                          {"name": "b", "kind": "plaza", "polygon": sq, "z": 1.0, "material": "gravel_grey"}])
    assert sorted(m.mats.values()) == ["asphalt", "gravel_grey"]
    assert "gravel_grey" in tx.MATERIALS


@pytest.mark.parametrize("fn", [tx.cafe_front_texture, tx.cafe_attic_texture, tx.rail_band_texture, tx.hall_front_texture, tx.hall_sign_texture])
def test_front_pictures_have_the_asked_size(fn):
    assert fn(320, 160).size == (320, 160)


def test_gable_end_colours_only_the_end_facing_the_bearing(tmp_path):
    import json
    from yabitsu import buildings
    fp = [_ll(0, 0), _ll(20, 0), _ll(20, 10), _ll(0, 10), _ll(0, 0)]   # 長い辺は東西、妻壁は東と西
    rec = {"id": "t", "footprint": fp, "storeys": 1, "height_m": 5.0, "wall": "#5E5444", "windows": True,
           "roof": {"type": "gable", "rise_m": 2.0, "color": "#3E3A36", "gable_end": {"bearing": 90, "wall": "#ECE8E0"}}}
    (tmp_path / "b.json").write_text(json.dumps([rec]), encoding="utf-8")
    b = buildings.load(tmp_path / "b.json")
    by = {m[0]: m for m in buildings.building_meshes(b, lambda x, y: np.zeros(np.shape(x)))}
    end, rest = by["bld_t_gable_end"], by["bld_t_gables"]
    assert end[4] == "facade_ECE8E0" and len(end[2]) == 1 and len(rest[2]) == 1        # 三角は 1 枚ずつ
    c = np.asarray(geo.to_local(*fp[0]))
    assert end[1][:, 0].mean() - c[0] == pytest.approx(20.0, abs=0.01)                 # 東の端（方位 90°）だけ白
    assert end[1][:, 2].max() == pytest.approx(7.0 + bd.BASE_H, abs=0.01)              # 基礎＋軒 5 m ＋ 棟まで 2 m
    mats = buildings.write_textures(b, tmp_path / "tex")
    assert "facade_ECE8E0" in mats
