"""県道 70 号（S4・S5）の Codex の部品の配置（data/manual/s45_route70.json、export/compare/route70/work/gen_route70_landmarks.py が作る）。"""
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from yabitsu import config, geo, landmarks, track
from yabitsu.crosssection import GUTTER_WIDTH, Section

SPEC = config.DATA_MANUAL / "s45_route70.json"
pytestmark = pytest.mark.skipif(not SPEC.exists(), reason="s45_route70.json が無い")


def _items():
    return json.loads(SPEC.read_text(encoding="utf-8"))["landmarks"]


def test_every_part_and_image_exists_and_names_are_ascii():
    """部品の npz と貼る画像があり、材質の名前は英字だけ（kn5 に日本語の名前を入れない）。"""
    for it in _items():
        assert landmarks.exists(it["asset"], it["name"]), it
        assert (landmarks.DIR / it["asset"] / f"{it.get('tex') or it['name']}.png").exists(), it
        assert landmarks.material(it["asset"], it["name"], it.get("tex")).isascii(), it


def test_items_are_in_s4_s5_and_off_the_road():
    """置いた物は S4・S5 の範囲で、路面と側溝の上に無い（根元が道の縁＋側溝より外）。
    広場の上に手で置いてあった物（今のコースの物の差し替えで z_from＝ground。護摩屋敷の広場の紅白ドラムなど）は、広場の形なので除く。
    擁壁・崖の前の物（at_road_edge）は、側溝の上（路面の端より外）にあればよい。"""
    sec = Section.load_json(config.DATA_PROCESSED / "track_section.json")
    s4_0 = float(track.load_bounds()["s4"][0])
    tree = cKDTree(np.c_[sec.x, sec.y])
    for it in _items():
        x, y = (float(v) for v in geo.to_local(*it["ll"]))
        i = int(tree.query([x, y])[1])
        assert sec.s[i] >= s4_0 - 5, it
        if it["src"].startswith(("s4_objects:", "s5_objects:")) and it.get("z_from") == "ground":
            continue
        h = sec.heading[i]
        d = -(x - sec.x[i]) * np.sin(h) + (y - sec.y[i]) * np.cos(h)
        hw = sec.hw_left[i] if d > 0 else sec.hw_right[i]
        if it.get("at_road_edge"):                    # 擁壁・崖の前：道の縁（側溝の上）。路面にはかからない
            assert abs(d) >= hw + 0.1, (it, d, hw)
            continue
        assert abs(d) >= hw + GUTTER_WIDTH - 0.05 or sec.bridge[i], (it, d, hw)


def test_items_do_not_overlap_each_other_or_stand_inside_buildings():
    """建物を除く置いた物どうしは 0.8 m 以上離れ（差し替えた今のコースの物どうしは元の位置のまま）、残っている建物の床の中に無い。"""
    from shapely.geometry import Point, Polygon
    items = [it for it in _items() if it["asset"] != "r70_buildings"]
    xy = np.array([geo.to_local(*it["ll"]) for it in items], float)
    added = [k for k, it in enumerate(items) if not it["src"].startswith(("s4_objects:", "s5_objects:"))]
    for k in added:
        d = np.hypot(*(xy - xy[k]).T)
        d[k] = 1e9
        assert d.min() >= 0.8, (items[k], items[int(d.argmin())])
    tb = json.loads((config.DATA_MANUAL / "track_buildings.json").read_text(encoding="utf-8"))
    polys = [Polygon([geo.to_local(*p) for p in b["footprint"]]) for b in tb if not b.get("gone")]
    for it, q in zip(items, xy):
        assert not any(pg.contains(Point(*q)) for pg in polys), it


def test_replaced_course_objects_are_not_drawn_twice():
    """Codex の部品に差し替えた今のコースのミラー・標識などには replaced_by が付き、組み立てで作らない。
    建物の床の中にあった今のコースの置き違い（グレース・マーヤ ホールの中のミラー）は r70_removed（どちらも作らない）。"""
    srcs = [it["src"] for it in _items() if it["src"].startswith(("s4_objects:", "s5_objects:"))]
    n = 0
    for sn in ("s4", "s5"):
        so = json.loads((config.DATA_MANUAL / f"{sn}_objects.json").read_text(encoding="utf-8"))
        n += sum(1 for o in so["objects"] if o.get("replaced_by") == "r70")
        assert not [o for o in so["objects"] if o["kind"] == "mirror" and not o.get("replaced_by")]
    assert n == len(srcs) > 0
