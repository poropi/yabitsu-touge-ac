import zipfile

import numpy as np
import pytest

from yabitsu import csptree, treemodels

pytestmark = pytest.mark.skipif(not treemodels.sources_available(), reason="Poly Haven の素材がない（data/raw/polyhaven）")

ALL = [n for names in treemodels.SPECIES.values() for n in names]


@pytest.fixture(scope="module")
def models():
    return {n: treemodels.build_pattern(n, seed=1) for n in ALL}


def _tris(subs):
    return sum(len(s.indices) // 3 for s in subs)


def test_species_cover_plan():
    assert set(treemodels.SPECIES) == {"sugi", "hinoki", "broadleaf", "sapling", "shrub", "fern", "grass"}   # sapling：植えたばかりの若木（峠の島）、grass：ススキ（寺山富士見橋）
    assert all(len(v) >= 2 for k, v in treemodels.SPECIES.items() if k in ("sugi", "hinoki", "broadleaf"))


def test_pattern_bounds_normalized(models):
    for name, m in models.items():
        assert len(m.lods) == 3, name
        for lod in m.lods:
            p = np.concatenate([s.verts["p"] for s in lod])
            assert p[:, 1].min() >= -0.02 and p[:, 1].max() <= 1.0 + 1e-4, name
            assert np.abs(p[:, [0, 2]]).max() <= 1.0 + 1e-4, name


def test_lod_triangle_budget(models):
    for name, m in models.items():
        t = [_tris(lod) for lod in m.lods]
        assert t[0] > t[1] > t[2], (name, t)
        assert t[0] <= 12000 and t[2] <= 1000, (name, t)


def test_trunk_at_origin(models):
    for name, m in models.items():
        if name.startswith("fern"):
            continue
        bark = m.lods[0][-1].verts["p"]
        low = bark[bark[:, 1] < 0.01]
        assert np.abs(low[:, [0, 2]].mean(0)).max() < 0.05, name


def test_sugi_has_clear_lower_trunk(models):
    for name in treemodels.SPECIES["sugi"]:
        leaves = models[name].lods[0][0].verts["p"]
        assert leaves[:, 1].min() >= 0.3, name


def test_packages_written(tmp_path):
    out = treemodels.write_all(tmp_path, names=["sugi_a", "fern_a"])
    for n in ("sugi_a", "fern_a"):
        z = zipfile.ZipFile(tmp_path / f"{n}.zip")
        names = z.namelist()
        assert "tree.ini" in names and all(f"LOD{i}.mesh" in names for i in range(3))
        tex, subs = csptree.read_mesh(z.read("LOD0.mesh"))
        assert all(t[1] in names and t[1].endswith(".dds") for t in tex)
        assert out[n].height > 0


def test_crown_covers_all_lods(models):
    """視界・張り出しの判定に使う樹冠は、遠くで使う形（板が大きい）も含めていちばん広く低いもの。"""
    for name, m in models.items():
        if name.startswith("fern"):
            continue
        H, hw = m.height, m.width / 2
        for lod in m.lods:
            p = lod[0].verts["p"]
            assert np.hypot(p[:, 0], p[:, 2]).max() * hw <= m.crown_radius + 1e-3, name
            assert p[:, 1].min() * H >= m.crown_base * H - 1e-3, name
