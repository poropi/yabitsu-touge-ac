"""蓑毛のバス停まわり：バスの折り返し場（ピット）、家、お堂、石碑。位置は緯度経度（data/manual/s2_minoge.json）。

折り返し場は、まわりの路面に合わせた傾いた平面にそろえる。家・お堂・石碑は、それぞれの場所の地面の高さ（地理院の標高）に
平らな土台を作って建てる（道から離れた斜面の家が、浮いたり埋まったりしないように）。
"""
import math

import numpy as np
from shapely.geometry import Polygon

from . import site
from .acobjects import Dummy
from .start_lot import Frame, _gable_ends, _gable_roof, _merge, box, road_formation

TWO_STOREY_M2 = 60.0
WALL_H = {1: 3.0, 2: 5.5}
PAD_MARGIN = 1.0


def _compass_to_heading(deg):
    """方位（北=0、東=90、時計回り）→ 進行方向（+x から反時計回りのラジアン）。"""
    return math.radians(90.0 - deg)


def _footprint(f, size):
    L, W = size
    return f.rect((-L / 2, L / 2), (-W / 2, W / 2))


class Site:
    def __init__(self, spec, sec, ground_z):
        self.spec = spec
        road = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
        turn = Polygon(site.ll_to_xy(spec["turnaround"]["points"])).buffer(0)
        self.turnaround = site.largest(turn.difference(road))
        c = self.turnaround.centroid
        self.plane = site.fit_plane(sec, (c.x, c.y))

        sh = spec.get("shrine")
        shrine_poly = _footprint(Frame(site.ll_to_xy([sh["center"]])[0], sh["axis_deg"]), sh["size"]) if sh else None
        self.houses = []  # (Frame, size, 床の高さ, 壁の高さ, 屋根の高さ, 軒の出)
        for b in spec.get("buildings", []):
            f = Frame(site.ll_to_xy([b["center"]])[0], b["axis_deg"])
            fp = _footprint(f, b["size"])
            # 道・折り返し場・待合所（お堂）と重なる建物は置かない（建物データには待合所そのものも入っている）
            if fp.intersects(road) or fp.intersects(self.turnaround) or (shrine_poly is not None and fp.intersects(shrine_poly)):
                continue
            storeys = 2 if b["size"][0] * b["size"][1] >= TWO_STOREY_M2 else 1
            self.houses.append((f, b["size"], float(ground_z(*f.o)), WALL_H[storeys], b["roof_h"], b["eaves"]))
        self.shrine = None
        if sh:
            f = Frame(site.ll_to_xy([sh["center"]])[0], sh["axis_deg"])
            self.shrine = (f, sh["size"], float(ground_z(*f.o)), 2.4, sh["roof_h"], sh["eaves"])
        self.monuments = []
        for m in spec.get("monuments", []):
            p = site.ll_to_xy([m["center"]])[0]
            self.monuments.append((p, m["size"], float(ground_z(*p))))

    def z(self, x, y):
        return site.plane_z(self.plane, x, y)

    def pits(self):
        out = []
        for i, (lat, lon, deg) in enumerate(self.spec["pits"]):
            x, y = site.ll_to_xy([[lat, lon]])[0]
            out.append((float(x), float(y), float(self.z(x, y)), _compass_to_heading(deg)))
        return out

    def pads(self):
        t = self.turnaround; c = t.centroid
        out = [{"name": "minoge_turnaround", "role": "flat", "kind": "site", "polygon": np.asarray(t.exterior.coords).tolist(),
                "z": float(self.z(c.x, c.y)), "plane": self.plane, "pit_points": self.pits()}]
        for k, (f, size, z, *_ ) in enumerate(self.houses):
            poly = _footprint(f, size).buffer(PAD_MARGIN, join_style="mitre")
            out.append({"name": f"minoge_house_{k}", "role": "flat", "kind": "site",
                        "polygon": np.asarray(poly.exterior.coords).tolist(), "z": z})
        if self.shrine:
            f, size, z, *_ = self.shrine
            poly = _footprint(f, size).buffer(PAD_MARGIN, join_style="mitre")
            out.append({"name": "minoge_shrine", "role": "flat", "kind": "site",
                        "polygon": np.asarray(poly.exterior.coords).tolist(), "z": z})
        return out

    def meshes(self):
        """(名前, 頂点, 面, UV, 材質, 描画するか)。中身のない部品は出さない。"""
        out = [("1ROAD_s2_minoge_turnaround", *site.draped(self.turnaround, self.z), "asphalt", True)]
        walls, gables, roofs = [], [], {"brown": [], "grey": []}
        for k, (f, (L, W), z, wh, rh, ev) in enumerate(self.houses):
            a, b = (-L / 2, L / 2), (-W / 2, W / 2)
            walls.append(box(f, a, b, z - 0.3, z + wh, 2.0))
            gables.append(_gable_ends(f, a, b, z + wh, rh))
            roofs["brown" if k % 2 == 0 else "grey"].append(_gable_roof(f, a, b, z + wh, rh, ev))
        if walls:
            out.append(("1WALL_s2_houses", *_merge(walls), "wall_white", True))
            out.append(("s2_houses_gables", *_merge(gables), "wall_white", True))
        if roofs["brown"]:
            out.append(("s2_houses_roof_brown", *_merge(roofs["brown"]), "roof", True))
        if roofs["grey"]:
            out.append(("s2_houses_roof_grey", *_merge(roofs["grey"]), "roof_metal", True))
        if self.shrine:
            f, (L, W), z, wh, rh, ev = self.shrine
            a, b = (-L / 2, L / 2), (-W / 2, W / 2)
            out.append(("1WALL_s2_shrine", *box(f, a, b, z - 0.3, z + wh, 1.0), "wood", True))
            out.append(("s2_shrine_gables", *_gable_ends(f, a, b, z + wh, rh), "wood", True))
            out.append(("s2_shrine_roof", *_gable_roof(f, a, b, z + wh, rh, ev), "roof", True))
        if self.monuments:
            parts = [box(Frame(p, 0.0), (-w / 2, w / 2), (-d / 2, d / 2), z - 0.3, z + h, 1.0)
                     for p, (w, d, h), z in self.monuments]
            out.append(("1WALL_s2_monuments", *_merge(parts), "stone", True))
        return out


def build(spec, sec, ground_z):
    return Site(spec, sec, ground_z)


def load(path):
    from .start_lot import load_spec
    return load_spec(path)


def dem_ground():
    """地理院の標高で地面の高さを返す関数（家・お堂の土台に使う）。"""
    from . import geo, gsi
    sampler = gsi.default_sampler()
    return lambda x, y: float(sampler.sample(*geo.from_local(np.atleast_1d(x), np.atleast_1d(y)))[0])
