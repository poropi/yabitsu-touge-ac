import numpy as np
import pytest
from PIL import Image

from yabitsu import dress_retex, ext_config
from tests.test_forest_ground import _manifest, _square


@pytest.fixture
def man(tmp_path):
    road_in = _square(40, -3, 60, 3, 0.0); road_out = _square(240, -3, 260, 3, 0.0)
    wall = ([(50, 6, 0), (60, 6, 0), (60, 6, 2), (50, 6, 2)], [(0, 1, 2), (0, 2, 3)])
    m = _manifest(tmp_path, {"1ROAD_s1_00": (*road_in, "asphalt"), "1ROAD_s1_01": (*road_out, "asphalt"),
                             "LINES_s1_edgeL_00": (*road_in, "line_white"), "1WALL_s1_retaining_000": (*wall, "concrete"),
                             "1GRASS_s1_terrain_000": (*road_in, "ground")})
    cl = {"s": np.arange(0, 301.0), "x": np.arange(0, 301.0), "y": np.zeros(301)}
    return m, cl


def test_meshes_limited_to_s3(man):
    m, cl = man
    kinds = dress_retex.s3_meshes(m, cl, (0.0, 100.0))
    assert kinds == {"1ROAD_s1_00": "road", "LINES_s1_edgeL_00": "line_white", "1WALL_s1_retaining_000": "wall"}


def test_wall_finish_from_csv(man, tmp_path):
    m, cl = man
    csv = tmp_path / "f.csv"
    csv.write_text("s_from,s_to,side,type,height_m,width_m,note\n40,70,L,retaining_wall,2,,間知石の石積み\n"
                   "40,70,R,retaining_wall,2,,ブロック塀\n", encoding="utf-8")
    assert dress_retex.wall_finish(m, cl, csv, ["1WALL_s1_retaining_000"]) == {"1WALL_s1_retaining_000": "kenchi"}


def test_retex_never_deactivates():
    s = dress_retex.section_text({"1ROAD_s1_00": "road_town", "1WALL_s1_retaining_000": "wall_kenchi"})
    assert "IS_ACTIVE" not in s and "[SHADER_REPLACEMENT_...]" in s
    assert "MESHES=1ROAD_s1_00,1ROAD_s1_00_back" in s and "RESOURCE_FILE_1=dress_wall_kenchi_nm.dds" in s


def test_assemble_keeps_forest_part(tmp_path):
    ext_config.write_part(tmp_path, "forest", "[TREES]\nACTIVE=1\n")
    ext_config.write_part(tmp_path, "dress_s3", "[SHADER_REPLACEMENT_...]\nMESHES=a\n")
    ext_config.assemble(tmp_path)
    s = (tmp_path / "extension" / "ext_config.ini").read_text(encoding="utf-8")
    assert s.startswith(ext_config.MARK) and "[TREES]" in s and "MESHES=a" in s


def test_compose_wall_rotates_to_uv_layout():
    """壁の UV は横＝高さ（1 = 2 m）、縦＝長さ（1 = 4 m）。元の画像（横＝長さ）を回して 2 m×4 m の並びにする。"""
    base = Image.new("RGB", (100, 100))
    base.paste((255, 0, 0), (0, 0, 50, 100))        # 左半分が赤（長さ方向に変わる模様）
    img = dress_retex.compose(base, base_m=(2, 2), u_m=2, v_m=4, rotate=True)
    a = np.asarray(img)
    assert a.shape[0] == 2 * a.shape[1]               # 縦（長さ 4 m）が横（高さ 2 m）の 2 倍
    assert (a[: a.shape[0] // 4, :, 0] > 200).all() or (a[: a.shape[0] // 4, :, 0] < 50).all()   # 長さ方向（縦）に模様が変わる


def test_assemble_migrates_old_single_file(tmp_path):
    """部品方式より前に書いた ext_config.ini（森だけ）は、forest の部品へ移して残す。"""
    (tmp_path / "extension").mkdir()
    (tmp_path / "extension" / "ext_config.ini").write_text(ext_config.MARK + "\n\n[TREES]\nLIST_0=trees\a.txt\n", encoding="utf-8")
    ext_config.write_part(tmp_path, "dress_s3", "[SHADER_REPLACEMENT_...]\nMESHES=a\n")
    ext_config.assemble(tmp_path)
    s = (tmp_path / "extension" / "ext_config.ini").read_text(encoding="utf-8")
    assert "LIST_0=trees\a.txt" in s and "MESHES=a" in s


def test_split_red_becomes_plain_road():
    """ストリートビュー 2025 では下り側の道は灰色（空中写真の赤は古い）。"""
    assert dress_retex.kind_of("1ROAD_split_red_00", "paint_red") == "road_plane"


def test_enlarge_keeps_size_tiles_and_scales_the_pattern():
    """enlarge(img, k)：同じ大きさの画像のまま、模様を k 倍に（目地の数が 1/k）。端は反対側とつながる（並べても継ぎ目がない）。"""
    import numpy as np
    from PIL import Image
    n = 256
    y, x = np.mgrid[0:n, 0:n]
    a = 128 + 60 * np.sign(np.sin(2 * np.pi * 4 * (x + y) / n)) + 40 * np.cos(2 * np.pi * x / n)   # 斜めの目地 4 本＋横に 1 周するむら
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
    out = dress_retex.enlarge(img, 2)
    assert out.size == img.size
    g = np.asarray(out.convert("L"), float); g -= g.mean()
    P = np.abs(np.fft.fft2(g)) ** 2; P[0, 0] = 0
    i, j = np.unravel_index(P.argmax(), P.shape)
    assert {int(i), int(j)} <= {2, n - 2}                       # 目地の周期が 4 → 2
    edge = np.abs(g[:, 0] - g[:, -1]).mean(); inner = np.abs(g[:, 100] - g[:, 101]).mean()
    assert edge < inner * 1.5                                  # 右端と左端の差は、となりの列どうしの差と同じくらい
    plain = np.asarray(img.resize((2 * n, 2 * n)).convert("L"), float)[:n, :n]
    assert np.abs(plain[:, 0] - plain[:, -1]).mean() > inner * 3   # 切り出しただけなら、むらの段差が端に出る（この確かめ方が効いている）
