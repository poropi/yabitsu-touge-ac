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


def test_new_textures_are_written(tmp_path):
    paths = textures.write_textures(tmp_path)
    assert {"paint_red", "concrete_circles"} <= set(paths)
    red = np.array(Image.open(paths["paint_red"]).convert("RGB")).reshape(-1, 3).mean(0)
    assert red[0] > red[1] + 0.2 * 255


def test_roadworks_and_signal_textures(tmp_path):
    paths = textures.write_textures(tmp_path)
    assert {"barricade", "cone", "sign_koji", "signal_head", "signal_ped", "pole_grey", "sign_bus", "text_tomare"} <= set(paths)
    bar = np.array(Image.open(paths["barricade"]).convert("RGB")).reshape(-1, 3)
    yellow = (bar[:, 0] > 180) & (bar[:, 1] > 150) & (bar[:, 2] < 90); black = bar.max(1) < 60
    assert yellow.mean() > 0.3 and black.mean() > 0.3          # 黄と黒のしま
    sign = np.array(Image.open(paths["sign_koji"]).convert("RGB")).reshape(-1, 3)
    assert (sign.max(1) < 60).mean() > 0.03                       # 文字（黒）がある
    head = np.array(Image.open(paths["signal_head"]).convert("RGB"))
    left = head[:, : head.shape[1] // 3].reshape(-1, 3).mean(0)
    assert left[1] > left[0] + 40                                  # 左のランプ（青）が点いている


def test_sign_texture_renders_japanese_text():
    im = textures.sign_texture("名古木駐在所", "#FFFFFF", "#1B3A8C", 512, 128)
    a = np.array(im.convert("RGB")).reshape(-1, 3)
    assert ((np.abs(a.astype(int) - [255, 255, 255]).sum(1)) > 100).mean() > 0.05
