import zipfile
from pathlib import Path

import numpy as np
import pytest

from yabitsu import csptree

REF = Path(r"C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/content/tracks/"
           r"hakoneTurnpike_wmmt5/extension/trees/treePine1.zip")


def _sub(diffuse, normal, n):
    v = np.zeros(n, csptree.VERT_DTYPE)
    v["p"] = np.random.default_rng(n).uniform(-1, 1, (n, 3))
    v["n"][:, 1] = 1.0
    v["b0"] = (127, 254, 127, 0)
    v["b2"] = (200, 255, 255, 0)
    idx = np.arange(n - n % 3, dtype=np.uint16)
    return csptree.Submesh(diffuse, normal, idx, v)


@pytest.mark.skipif(not REF.exists(), reason="AC の参考コースがない")
def test_roundtrip_reference():
    data = zipfile.ZipFile(REF).read("Pine_1_LOD1.mesh")
    tex, subs = csptree.read_mesh(data)
    assert csptree.write_mesh(tex, subs) == data


def test_write_read_synthetic():
    subs = [_sub(0, 1, 6), _sub(2, 3, 9)]
    data = csptree.write_mesh(["a.png", "a_n.png", "b.png", "b_n.png"], subs)
    tex, back = csptree.read_mesh(data)
    assert [t if isinstance(t, str) else t[1] for t in tex] == ["a.png", "a_n.png", "b.png", "b_n.png"]
    for s, b in zip(subs, back):
        assert (s.diffuse, s.normal) == (b.diffuse, b.normal)
        assert np.array_equal(s.indices, b.indices)
        assert s.verts.tobytes() == b.verts.tobytes()


def test_rejects_over_65535_verts():
    with pytest.raises(ValueError):
        csptree.write_mesh(["a.png"], [_sub(0, 0, 65536)])


def test_tree_line_format():
    assert csptree.tree_line("sugi_a.zip", 1, 2, 3, 45, 1.2) == \
        "tree: sugi_a.zip; pos=1.00, 2.00, 3.00; angle = 45; size = 1.20"


def test_to_ac():
    assert np.allclose(csptree.to_ac(np.array([[1.0, 2.0, 3.0]])), [[1.0, 3.0, -2.0]])


def test_write_package(tmp_path):
    p = tmp_path / "t.zip"
    csptree.write_package(p, {"BASIC": {"HEIGHT": 10, "WIDTH": 4}}, {"LOD0.mesh": b"x"}, {"a.png": b"y"})
    z = zipfile.ZipFile(p)
    assert sorted(z.namelist()) == ["LOD0.mesh", "a.png", "tree.ini"]
    assert "[BASIC]\nHEIGHT=10\nWIDTH=4\n" in z.read("tree.ini").decode()
