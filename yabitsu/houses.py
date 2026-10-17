"""道ぞいの家（蓑毛・S3 の集落）：外周線から作った向きのついた箱（center, axis_deg, size）に、壁・妻・切妻屋根。
家ごとに、その場所の地面の高さ（ground_z）に平らな土台を作って建てる（斜面の家が浮いたり埋まったりしないように）。

buildings = [{"center": [lat, lon], "axis_deg", "size": [L, W], "roof_h", "eaves", "roof": brown | grey | blue | red（なければ1軒おき）}]"""
import numpy as np

from . import site
from .start_lot import Frame, _gable_ends, _gable_roof, _merge, box, road_formation

TWO_STOREY_M2 = 60.0
WALL_H = {1: 3.0, 2: 5.5}
PAD_MARGIN = 1.0
ROOF_MATERIALS = {"brown": "roof", "grey": "roof_metal", "blue": "roof_blue", "red": "roof_red"}


def footprint(f, size):
    L, W = size
    return f.rect((-L / 2, L / 2), (-W / 2, W / 2))


class Houses:
    def __init__(self, buildings, sec, ground_z, prefix, pad_name, avoid=()):
        self.prefix, self.pad_name = prefix, pad_name
        road = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
        self.items = []  # (Frame, size, 床の高さ, 壁の高さ, 屋根の高さ, 軒の出, 屋根の色)
        k = 0
        for b in buildings:
            f = Frame(site.ll_to_xy([b["center"]])[0], b["axis_deg"])
            fp = footprint(f, b["size"])
            if fp.intersects(road) or any(fp.intersects(a) for a in avoid):
                continue
            storeys = 2 if b["size"][0] * b["size"][1] >= TWO_STOREY_M2 else 1
            colour = b.get("roof") or ("brown" if k % 2 == 0 else "grey")
            self.items.append((f, b["size"], float(ground_z(*f.o)), WALL_H[storeys], b["roof_h"], b["eaves"], colour))
            k += 1

    def pads(self):
        out = []
        for k, (f, size, z, *_) in enumerate(self.items):
            poly = footprint(f, size).buffer(PAD_MARGIN, join_style="mitre")
            out.append({"name": self.pad_name.format(k=k), "role": "flat", "kind": "site",
                        "polygon": np.asarray(poly.exterior.coords).tolist(), "z": z})
        return out

    def meshes(self):
        walls, gables, roofs = [], [], {}
        for f, (L, W), z, wh, rh, ev, colour in self.items:
            a, b = (-L / 2, L / 2), (-W / 2, W / 2)
            walls.append(box(f, a, b, z - 0.3, z + wh, 2.0))
            gables.append(_gable_ends(f, a, b, z + wh, rh))
            roofs.setdefault(colour, []).append(_gable_roof(f, a, b, z + wh, rh, ev))
        out = []
        if walls:
            out.append((f"1WALL_{self.prefix}", *_merge(walls), "wall_white", True))
            out.append((f"{self.prefix}_gables", *_merge(gables), "wall_white", True))
        for colour in ("brown", "grey", "blue", "red"):
            if roofs.get(colour):
                out.append((f"{self.prefix}_roof_{colour}", *_merge(roofs[colour]), ROOF_MATERIALS[colour], True))
        return out
