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
