"""S3（名古木 → 蓑毛）の道ぞいの物をまとめて読む：町の範囲（交差点）、町の建物、S3 の道ぞいの家、道ぞいの信号・小物。
track_04（pads）・track_06（地形）・track_build_manifest（部品）が同じものを使う。"""
import json

import numpy as np
from shapely.geometry import Point

from . import buildings as bd
from . import config, geo, side_roads, town, town_ground
from .houses import Houses
from .start_lot import road_formation

P = config.DATA_PROCESSED
HOUSE_CLEAR_M = 2.5   # S3 の道ぞいの家は、道（路面＋側溝）からこれより近いものを置かない（歩道の上に建てない）


def center():
    x, y = geo.to_local(*config.NAGOKI)
    return float(x), float(y)


def circle():
    return Point(*center()).buffer(config.TOWN_RADIUS_M, 128)


def load_ground():
    return town_ground.TownGround.load(P / "track_town_ground.npz")


def load_town(sec, ground=None):
    spec = json.loads((config.DATA_MANUAL / "s3_nagoki.json").read_text(encoding="utf-8"))
    return town.Town(spec, side_roads.load_elements(), sec, ground or load_ground(), center(), config.TOWN_RADIUS_M)


def load_buildings():
    return [bd.Building.from_spec(d) for d in json.loads((config.DATA_MANUAL / "s3_buildings.json").read_text(encoding="utf-8"))]


def roadside_houses(sec, ground_z):
    spec = json.loads((config.DATA_MANUAL / "s3_houses.json").read_text(encoding="utf-8"))
    avoid = [road_formation(sec, 0.0, float(sec.s[-1])).buffer(HOUSE_CLEAR_M), circle()]
    return Houses(spec["buildings"], sec, ground_z, prefix="s3_houses", pad_name="s3_house_{k}", avoid=avoid)


def roadside_objects():
    return json.loads((config.DATA_MANUAL / "s3_objects.json").read_text(encoding="utf-8"))


def tree_free_areas(sec, bounds, buildings):
    """木を置かない所：町の範囲、町の建物のまわり 2m、S3 の道から 60m（家の多い町なか。木の作り込みは仕上げで）。"""
    from shapely.geometry import LineString
    m = sec.s <= bounds["s3"][1]
    return [circle().buffer(10.0), LineString(np.c_[sec.x[m], sec.y[m]]).buffer(60.0)] + [b.footprint.buffer(2.0) for b in buildings]


def load_branches(sec):
    """県道70号が2本に分かれる所の、もう1本と島（data/manual/s3_branches.json）。"""
    from . import branches
    specs = json.loads((config.DATA_MANUAL / "s3_branches.json").read_text(encoding="utf-8"))
    return branches.parallel_branches(side_roads.load_elements(), sec, specs)


def load_splits(sec):
    """県道70号が島をはさんで2本に分かれる所（data/manual/s3_split.json）。地図の線どおりに作る（split.py）。"""
    from . import split
    specs = json.loads((config.DATA_MANUAL / "s3_split.json").read_text(encoding="utf-8"))
    return [split.Split(sp, side_roads.load_elements(), sec) for sp in specs.values()]


def sidewalk_polygons(sec, feats):
    """道ぞいの歩道（features の sidewalk）の範囲の多角形。"""
    from .crosssection import GUTTER_WIDTH
    from .town import side_band
    out = []
    for f in feats:
        if f.type != "sidewalk":
            continue
        sd = 1 if f.side == "L" else -1
        i = int(np.argmin(np.abs(sec.s - (f.s_from + f.s_to) / 2)))
        hw = float(sec.hw_left[i] if sd > 0 else sec.hw_right[i]) + GUTTER_WIDTH
        out.append(side_band(sec, f.s_from, f.s_to, sd, hw - 0.05, hw + f.width_m))
    return out


def track_stubs(sec, pads, feats, sampler=None):
    """分かれ道（入口の柵・見えない壁は、本線の路面と歩道から離して置く）と、道ぞいの壁などを入口で切った features。"""
    from shapely.ops import unary_union
    from . import features as ft
    tw = load_town(sec); sps = load_splits(sec)
    base = [p for p in pads if p.get("kind") != "stub"]
    stubs = side_roads.for_track(sec, base, sampler, exclude=unary_union([tw.region] + [z.area() for z in sps]))
    walks = sidewalk_polygons(sec, feats)
    for st in stubs:
        side_roads.set_back_mouth(st, sec, walks)
    return stubs, ft.cut_at(feats, side_roads.branch_gaps(stubs, sec))
