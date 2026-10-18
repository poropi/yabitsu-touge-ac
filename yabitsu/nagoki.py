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


from dataclasses import dataclass


@dataclass(frozen=True)
class Junction:
    """町の範囲（半径 TOWN_RADIUS_M）として作る交差点：中心、町の spec、建物、駐車場、地面の保存先、kn5 のグループ。"""
    name: str
    latlon: tuple
    spec: str
    buildings: str
    lots: str
    ground: str
    group: str


JUNCTIONS = {
    "nagoki": Junction("nagoki", config.NAGOKI, "s3_nagoki.json", "s3_buildings.json", "s3_lots.json", "track_town_ground.npz", config.TOWN_GROUP),
    "kitahara": Junction("kitahara", config.KITAHARA, "s5_kitahara.json", "s5_kitahara_buildings.json", "s5_kitahara_lots.json",
                         "track_kitahara_ground.npz", config.KITAHARA_GROUP),
}
NAGOKI = JUNCTIONS["nagoki"]


def available():
    """spec のファイルがある交差点（作る交差点）。"""
    return [j for j in JUNCTIONS.values() if (config.DATA_MANUAL / j.spec).exists()]


def center(junction=NAGOKI):
    x, y = geo.to_local(*junction.latlon)
    return float(x), float(y)


def circle(junction=NAGOKI):
    return Point(*center(junction)).buffer(config.TOWN_RADIUS_M, 128)


def load_ground(junction=NAGOKI):
    return town_ground.TownGround.load(P / junction.ground)


def load_town(sec, ground=None, junction=NAGOKI):
    spec = json.loads((config.DATA_MANUAL / junction.spec).read_text(encoding="utf-8"))
    return town.Town(spec, side_roads.load_elements(), sec, ground or load_ground(junction), center(junction), config.TOWN_RADIUS_M)


def load_buildings(junction=NAGOKI):
    p = config.DATA_MANUAL / junction.buildings
    return [bd.Building.from_spec(d) for d in json.loads(p.read_text(encoding="utf-8"))] if p.exists() else []


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
    from . import water
    lake, _ = water.load_lake()   # 宮ヶ瀬湖の中にも木を置かない
    return ([circle().buffer(10.0), LineString(np.c_[sec.x[m], sec.y[m]]).buffer(60.0)] + [b.footprint.buffer(2.0) for b in buildings]
            + ([lake.buffer(2.0)] if lake is not None else []))


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


def side_crosswalk_points():
    """横道の上の横断歩道（OSM、data/manual/s3_side_crosswalks.json）の位置（ローカル座標）。"""
    pts = json.loads((config.DATA_MANUAL / "s3_side_crosswalks.json").read_text(encoding="utf-8"))
    return [tuple(float(v) for v in geo.to_local(la, lo)) for la, lo in pts]


def track_stubs(sec, pads, feats, sampler=None):
    """分かれ道（入口の柵・見えない壁は、本線の路面と歩道から離して置く）と、道ぞいの壁などを入口で切った features。"""
    from shapely.ops import unary_union
    from . import features as ft
    tw = load_town(sec); sps = load_splits(sec)
    base = [p for p in pads if p.get("kind") != "stub"]
    stubs = side_roads.for_track(sec, base, sampler, exclude=unary_union([tw.region] + [z.area() for z in sps]))
    walks = sidewalk_polygons(sec, feats)
    cws = side_crosswalk_points()
    for st in stubs:   # 分かれ道の上の横断歩道があれば、柵はその後ろ
        own = [st.crosswalk_area(p) for p in cws if st.surface.buffer(1.0).contains(Point(*p))]
        side_roads.set_back_mouth(st, sec, walks + own + [z.area() for z in sps])   # 島の路面の上にも柵を立てない
    paved = unary_union([z.area() for z in sps] + [tw.carriageway, tw.sidewalk])   # 別に作る舗装の上の壁・生け垣は作らない
    return stubs, ft.cut_off(ft.cut_at(feats, side_roads.branch_gaps(stubs, sec)), sec, paved)
