"""浅間神社（S2、ヘアピンの内側の山側）の入口：舗装の広場、奥の間知石の石積み、山の上へ上る石段と手すり、
石段の右の低いコンクリートの壁、社号標（石の柱）、台座の上の石碑、壁の掲示板。

位置は緯度経度（data/manual/s2_asama.json）。高さは、まわりの路面に合わせた傾いた平面にそろえる（pass_area と同じ）。
参考：nanohanadai コース（形を見ただけで、データは使わない）・ストリートビュー 2025・地理院の空中写真。
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from . import config, site
from .start_lot import Frame, _merge, _quad, box, road_formation

STEP_TOP = 0.02      # 石段の地形の平面は、段の先端よりこれだけ下


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class Site:
    def __init__(self, spec, sec, ground=None):
        self.spec, self.sec = spec, sec
        o = site.ll_to_xy([spec["origin"]])[0]
        self.f = Frame(o, spec["axis_deg"])
        pa = spec["pad"]
        road = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
        rect = self.f.rect(pa["a"], pa["b"])
        self._pad = site.largest(rect.difference(road.buffer(0.05)))
        self.plane = site.fit_plane(sec, np.asarray(self._pad.centroid.coords[0]))
        self.ground = ground       # 石段の上の端の地形の高さ（無ければ平均の傾き）

    def z(self, x, y):
        return float(site.plane_z(self.plane, x, y))

    def pad_polygon(self):
        return self._pad

    # 石段：下の端は広場の高さ、上の端は地形（spec の top_z か、地形の高さ）
    def _stairs(self):
        st = self.spec["stairs"]
        a0, a1 = st["a"]; b0, b1 = st["b"]
        bottom = self.f.xy((a0 + a1) / 2, b0)
        top = self.f.xy((a0 + a1) / 2, b1)
        z0 = self.z(*bottom)
        z1 = st.get("top_z") or (float(self.ground(top[0], top[1])) if self.ground else z0 + 0.65 * (b1 - b0))
        n = max(1, int(round((z1 - z0) / st["rise"])))
        return a0, a1, b0, b1, z0, z1, n

    def pads(self):
        c = self._pad.centroid
        a0, a1, b0, b1, z0, z1, n = self._stairs()
        poly = self.f.rect((a0 - 0.3, a1 + 0.3), (b0, b1 + 1.0))
        bottom, top = self.f.xy(0.0, b0), self.f.xy(0.0, b1)
        g = (z1 - z0) / (b1 - b0)
        gx, gy = g * self.f.n[0], g * self.f.n[1]          # 北（b）の向きへ上る平面
        return [{"name": "asama_pad", "role": "flat", "kind": "site", "polygon": np.asarray(self._pad.exterior.coords).tolist(),
                 "z": self.z(c.x, c.y), "plane": self.plane},
                {"name": "asama_stairs", "role": "flat", "kind": "site", "polygon": np.asarray(poly.exterior.coords).tolist(),
                 "z": z0, "plane": [float(bottom[0]), float(bottom[1]), float(z0 - STEP_TOP), float(gx), float(gy)]}]

    def meshes(self):
        sp, f = self.spec, self.f
        out = []
        v, faces = _drape(self._pad, self.z)
        out.append(("1ROAD_asama_pad", v, faces, v[:, :2] / 4.0, "asphalt", True))
        za = self.z(*f.xy(np.mean(sp["pad"]["a"]), 0.0))
        w = sp["wall"]
        out.append(("1WALL_asama_wall", *box(f, w["a"], w["b"], za - 0.5, za + w["height"], 2.0), "kenchi", True))
        # 石段（段ごとの箱）と、その横の手すり
        a0, a1, b0, b1, z0, z1, n = self._stairs()
        run = (b1 - b0) / n; rise = (z1 - z0) / n
        steps = [box(f, (a0, a1), (b0 + k * run, b1), z0 + k * rise - 0.3 if k else z0 - 0.3, z0 + (k + 1) * rise, 1.0)
                 for k in range(n)]
        out.append(("1ROAD_asama_stairs", *_merge(steps), "concrete", True))
        hr = sp["handrails"]; h = hr["height"]
        rails = []
        for a_ in hr["a"]:
            pts = np.arange(b0 + 0.3, b1, hr["post_every"])
            for b_ in pts:
                zb = z0 + (b_ - b0) / run * rise
                rails.append(box(f, (a_ - 0.025, a_ + 0.025), (b_ - 0.025, b_ + 0.025), zb, zb + h, 0.5))
            for b_, b2 in zip(pts[:-1], pts[1:]):      # 上の横棒（坂に沿って段々に）
                zb = z0 + (b_ - b0) / run * rise
                rails.append(box(f, (a_ - 0.025, a_ + 0.025), (b_, b2), zb + h - 0.05, zb + h, 0.5))
        out.append(("asama_handrails", *_merge(rails), "pole_grey", True))
        pr = sp["parapet"]
        out.append(("1WALL_asama_parapet", *box(f, pr["a"], pr["b"], z0 - 0.3, z0 + pr["height"], 1.0), "concrete", True))
        pl = sp["pillar"]; L, W, H = pl["size"]
        pv, pf, _ = box(f, (pl["a"] - L / 2, pl["a"] + L / 2), (pl["b"] - W / 2, pl["b"] + W / 2), z0 - 0.2, z0 + H, 1.0)
        pa, pb = f.ab(pv[:, :2])
        pu = np.c_[((pa - pl["a"] + L / 2) + (pb - pl["b"] + W / 2)) / L % 1.0001, 1.0 - (pv[:, 2] - z0 + 0.2) / (H + 0.2)]
        out.append(("asama_pillar", pv, pf, pu, "asama_pillar", True))   # 4面とも縦書きの字が1つずつ入る
        out.append(("1WALL_asama_pillar", *box(f, (pl["a"] - L / 2, pl["a"] + L / 2), (pl["b"] - W / 2, pl["b"] + W / 2),
                                               z0 - 0.2, z0 + H, 1.0), "concrete", False))
        mo = sp["monument"]; PL, PW, PH = mo["pedestal"]; SL, SW, SH = mo["stone"]
        zm = self.z(*f.xy(mo["a"], mo["b"]))
        ped = box(f, (mo["a"] - PL / 2, mo["a"] + PL / 2), (mo["b"] - PW / 2, mo["b"] + PW / 2), zm - 0.2, zm + PH, 1.0)
        stone = _slab(f, mo["a"], mo["b"], SL, SW, zm + PH, SH)
        out.append(("1WALL_asama_monument", *_merge([ped, stone]), "stone", True))
        bo = sp["board"]; zb = self.z(*f.xy(bo["a"], 0.0))
        out.append(("asama_board", *_quad(f, [bo["a"] - bo["w"] / 2, bo["a"] + bo["w"] / 2] * 1 + [bo["a"] + bo["w"] / 2, bo["a"] - bo["w"] / 2],
                                          [bo["b"]] * 4, [zb + bo["z"][0]] * 2 + [zb + bo["z"][1]] * 2, 1.0), "asama_board", True))
        return out


def _drape(poly, zf):
    tri = __import__("shapely").constrained_delaunay_triangles(poly)
    vs, fs = [], []
    for t in tri.geoms:
        c = np.asarray(t.exterior.coords)[:3]
        if (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - (c[1, 1] - c[0, 1]) * (c[2, 0] - c[0, 0]) < 0:
            c = c[::-1]
        b = len(vs); vs += [(p[0], p[1], zf(p[0], p[1])) for p in c]; fs.append((b, b + 1, b + 2))
    return np.array(vs, float), np.array(fs, np.int64)


def _slab(f, a, b, L, W, z0, H):
    """上へ少し細くなり、上の角が丸い自然石の石碑（8角の断面を上へすぼめる）。"""
    rings = []
    for t, s in ((0.0, 1.0), (0.55, 0.97), (0.85, 0.88), (1.0, 0.62)):
        ang = np.linspace(0, 2 * np.pi, 9)[:-1] + np.pi / 8
        aa = a + np.cos(ang) * L / 2 * s * (1.0 if t < 0.9 else 0.85)
        bb = b + np.sin(ang) * W / 2 * s
        rings.append(np.c_[f.xy(aa, bb), np.full(8, z0 + H * t)])
    v = np.concatenate(rings)
    faces = [(i * 8 + j, i * 8 + (j + 1) % 8, (i + 1) * 8 + (j + 1) % 8, (i + 1) * 8 + j) for i in range(3) for j in range(8)]
    top = len(v); v = np.r_[v, [[*f.xy(a, b), z0 + H * 1.02]]]
    faces += [(3 * 8 + j, 3 * 8 + (j + 1) % 8, top, top) for j in range(8)]
    uv = np.c_[np.arange(len(v)) % 8 / 8.0, v[:, 2] - z0]
    return v, np.array(faces, np.int64), uv


def pillar_texture(text, path, w=64, h=640):
    """社号標：白っぽい石に縦書きの黒い字。"""
    img = Image.new("RGB", (w, h), (214, 210, 200))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(config.font_file("mincho"), int(w * 0.8))
    for k, ch in enumerate(text):
        d.text((w / 2, h * 0.08 + k * w * 0.95), ch, fill=(40, 38, 36), font=font, anchor="mt")
    img.save(path)


def board_texture(path, w=280, h=140):
    """掲示板：白い板に赤い見出しの帯と黒い細い行（文字は読めない大きさの線）。"""
    img = Image.new("RGB", (w, h), (236, 234, 228))
    d = ImageDraw.Draw(img)
    d.rectangle((8, 8, w - 8, 34), fill=(190, 40, 36))
    for y in range(48, h - 12, 14):
        d.rectangle((14, y, w - 14 - (y * 7) % 60, y + 4), fill=(70, 70, 70))
    d.rectangle((0, 0, w - 1, h - 1), outline=(120, 120, 120), width=3)
    img.save(path)


def build(spec, sec, ground=None):
    return Site(spec, sec, ground)
