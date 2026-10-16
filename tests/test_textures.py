import numpy as np
from PIL import Image

from yabitsu import textures
from yabitsu.manifest import Manifest


def test_noise_is_tileable():
    n = textures.periodic_noise(128, 8, np.random.default_rng(0))
    inner = np.abs(np.diff(n, axis=1)).mean()
    wrap = np.abs(n[:, 0] - n[:, -1]).mean()
    assert wrap < 3 * inner and n.min() >= 0 and n.max() <= 1


def test_write_textures(tmp_path):
    paths = textures.write_textures(tmp_path)
    assert set(paths) == set(textures.MATERIALS)
    assert Image.open(paths["tree"]).mode == "RGBA"
    a = np.array(Image.open(paths["tree"]))[..., 3]
    assert a.min() == 0 and a.max() == 255


def test_register_materials(tmp_path):
    m = Manifest(tmp_path)
    textures.register_materials(m)
    assert m.materials["tree"]["alpha"] and m.materials["tree"]["two_sided"]
    assert m.materials["asphalt"]["texture"] == "textures/asphalt.png"
