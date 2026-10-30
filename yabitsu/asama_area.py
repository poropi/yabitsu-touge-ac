"""浅間神社（S2、ヘアピンの内側の山側）の入口の駐車場：アスファルトの広場（区画線なし）、石段の前のコンクリートの土間、
奥の間知ブロックの石積み（のりで奥へ傾き、上に蔓草）、山の上へ上る石段と両側の手すり、石段の右の袖壁、社号標、
台座の上の石碑、壁の看板（専用駐車場・小さな掲示）、草の株。駐車場の前の本線は側溝を作らず、路肩の線を破線にする（mouths）。

位置は data/manual/s2_asama.json（原点の緯度経度と、そこからの東 a・北 b の m）。高さは、まわりの路面に合わせた傾いた平面にそろえる（pass_area と同じ）。
基準：Google 航空写真 z21・ストリートビュー 2025-07（export/compare/asama、生成 work/gen_asama.py）。
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
        road = road_formation(sec, s_min=0.0, s_max=float(sec.s[-1]))
        rect = Polygon(self.f.xy(*np.asarray(spec["lot"]["outline"], float).T)).buffer(0)
        self._pad = site.largest(rect.difference(road.buffer(0.05)))
        self.plane = site.fit_plane(sec, np.asarray(self._pad.centroid.coords[0]))
        self.ground = ground       # 石段の上の端の地形の高さ（無ければ平均の傾き）

    def z(self, x, y):
        return float(site.plane_z(self.plane, x, y))

    def pad_polygon(self):
        return self._pad

    def _terrain_pad(self):
        """地形を駐車場の高さに平らにする範囲：駐車場と、壁の根元から裏へ behind_m（既定 2.2 m）の帯。
        地形の網目は 2 m 間隔なので、駐車場の形のままだと壁の裏の斜面が手前へにじみ出て、壁の下と石碑の台座を埋めた（2026-10-30）。
        壁はのりで奥へ傾き、上には蔓草がかぶさるので、壁の裏の低い所は道から見えない。"""
        from shapely.geometry import Polygon as _P
        xy, nv = self._line()
        back = self.spec["wall"].get("terrain_behind_m", 2.2)
        band = _P(np.r_[xy, (xy + nv * back)[::-1]]).buffer(0)
        road = road_formation(self.sec, s_min=0.0, s_max=float(self.sec.s[-1]))
        band = band.buffer(0.3).difference(road.buffer(0.6))   # 道の縁から 0.6 m は帯に入れない（東の端で道ぞいの擁壁の線に重なった）
        return site.largest(self._pad.union(band).difference(road.buffer(0.05)))

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
        # 舗装する面は kind "pass"：面にかかる地形の格子（2 m）の角を全部 面より下げ、隣の石段の下の上り勾配の三角形が
        # 駐車場の角（袖壁の東・石碑の後ろ）に出ないようにする（terrain._sink_pass_cells。2026-10-30 に 1.2 m 出ていた）
        return [{"name": "asama_pad", "role": "flat", "kind": "pass", "polygon": np.asarray(self._terrain_pad().exterior.coords).tolist(),
                 "z": self.z(c.x, c.y), "plane": self.plane},
                {"name": "asama_stairs", "role": "flat", "kind": "site", "polygon": np.asarray(poly.exterior.coords).tolist(),
                 "z": z0, "plane": [float(bottom[0]), float(bottom[1]), float(z0 - STEP_TOP), float(gx), float(gy)]}]

    def meshes(self):
        sp, f = self.spec, self.f
        out = []
        v, faces = _drape(self._pad, self.z)
        out.append(("1ROAD_asama_pad", v, faces, v[:, :2] / 4.0, "asphalt", True))
        if sp["lot"].get("apron"):        # 石段の前のコンクリートの土間（駐車場の上に 1.5 cm）
            ap = site.largest(Polygon(f.xy(*np.asarray(sp["lot"]["apron"], float).T)).intersection(self._pad))
            v, faces = _drape(ap, lambda x, y: self.z(x, y) + 0.015)
            out.append(("asama_apron", v, faces, v[:, :2] / 2.0, "concrete_grey", True))
        out += self._wall()
        # 石段（段ごとの箱）と、両側の手すり
        a0, a1, b0, b1, z0, z1, n = self._stairs()
        run = (b1 - b0) / n; rise = (z1 - z0) / n
        steps = [box(f, (a0, a1), (b0 + k * run, b1), z0 - 0.4, z0 + (k + 1) * rise, 1.0)   # 側面は下まで（下の地形が下がっても隙間を見せない）
                 for k in range(n)]
        out.append(("1ROAD_asama_stairs", *_merge(steps), "concrete_grey", True))
        hr = sp["handrails"]; h = hr["height"]
        rails = []
        for a_ in (a0 + hr["inset"], a1 - hr["inset"]):
            pts = np.r_[np.arange(b0 + 0.3, b1, hr["post_every"]), b1 - 0.2]
            for b_ in pts:
                zb = z0 + (b_ - b0) / run * rise
                rails.append(box(f, (a_ - 0.024, a_ + 0.024), (b_ - 0.024, b_ + 0.024), zb, zb + h, 0.5))
            for b_, b2 in zip(pts[:-1], pts[1:]):      # 上の横棒（坂に沿って段々に）
                zb = z0 + (b_ - b0) / run * rise
                rails.append(box(f, (a_ - 0.024, a_ + 0.024), (b_, b2), zb + h - 0.05, zb + h, 0.5))
        out.append(("asama_handrails", *_merge(rails), "steel_galv", True))
        if sp.get("stair_wall"):
            out += self._stair_wall(a0, a1, b0, b1, z0, z1)
        pr = sp["parapet"]   # 石段の右の袖壁：手前が低く、奥（壁の線）へ上がる
        out.append(("1WALL_asama_parapet", *_wedge(f, (a1, a1 + pr["thick"]), (b0, max(0.3, b0 + 0.5)), z0 - 0.3,
                                                     z0 + pr["h_front"], z0 + pr["h_back"]), "stone", True))
        pl = sp["pillar"]; L, W, H = pl["size"]
        pv, pf, _ = box(f, (pl["a"] - L / 2, pl["a"] + L / 2), (pl["b"] - W / 2, pl["b"] + W / 2), z0 - 0.2, z0 + H, 1.0)
        pa, pb = f.ab(pv[:, :2])
        pu = np.c_[((pa - pl["a"] + L / 2) + (pb - pl["b"] + W / 2)) / L % 1.0001, 1.0 - (pv[:, 2] - z0 + 0.2) / (H + 0.2)]
        lm = sp.get("landmarks", {}).get("post")
        if lm and _landmark_exists(lm):   # Codex 制作の立札（原点＝根元の中央、正面 −Y＝南）
            v, fc, uv = _load_landmark(lm, lm["names"][0])
            out.append(("asama_pillar", np.c_[f.xy(pl["a"] + v[:, 0], pl["b"] + v[:, 1]), z0 - 0.02 + v[:, 2]], fc, uv, lm["material"], True))
        else:
            out.append(("asama_pillar", pv, pf, pu, "asama_pillar", True))   # 4面とも縦書きの字が1つずつ入る
        out.append(("1WALL_asama_pillar", *box(f, (pl["a"] - L / 2, pl["a"] + L / 2), (pl["b"] - W / 2, pl["b"] + W / 2),
                                               z0 - 0.2, z0 + H, 1.0), "concrete", False))
        out += self._monument()
        for bo in sp.get("boards", []):   # 壁の看板（壁の下の端から off 手前、少し傾ける）
            out.append((bo["tex"], *self._board(bo), bo["tex"], True))
        tufts = []
        lm = sp.get("landmarks", {}).get("grass")
        lm = lm if lm and _landmark_exists(lm) else None
        for k, g in enumerate(sp.get("grass", [])):
            a_, b_ = (g["a"], g["b"]) if "b" in g else self.wall_point(g["a"], g["off"])
            x, y = f.xy(a_, b_)
            if lm:   # Codex 制作の草の株（高さの近いものを選び、向きを変える）
                name = min(lm["names"], key=lambda n_: abs(lm["heights"][n_] - g["size"]))
                v, fc, uv = _load_landmark(lm, name)
                r = np.radians(k * 137.5); sc_ = g["size"] / lm["heights"][name]
                ca, sa = np.cos(r) * sc_, np.sin(r) * sc_
                tufts.append((lm["material"] + name[-2:], (np.c_[x + v[:, 0] * ca - v[:, 1] * sa, y + v[:, 0] * sa + v[:, 1] * ca, self.z(x, y) - 0.02 + v[:, 2] * sc_], fc, uv)))
            else:
                tufts.append(("hedge", _tuft(float(x), float(y), self.z(x, y), g["size"], seed=k)))
        for mat in sorted({m_ for m_, _ in tufts}):
            out.append(("asama_grass" + ("" if mat == "hedge" else "_" + mat.split("_")[-1]), *_merge([t for m_, t in tufts if m_ == mat]), mat, True))
        return out

    def _monument(self):
        """石碑（Codex 制作の assets/landmarks/asama_monument を blender_scripts/bake_asama_monument.py で焼き付けた部品）。
        正面（模型の −Y）を駐車場へ向け、壁の下の端から off m 手前に置く。傾いた路面で浮かないよう、台座の 4 隅の低い所に合わせる。
        当たり判定は台座と碑の 2 つの箱。部品が無ければ（焼き付け前）、前の簡単な形で代わりに置く。"""
        mo, f = self.spec["monument"], self.f
        a_, b_ = self.wall_point(mo["a"], mo["off"])
        c = f.xy(a_, b_)
        xy, nv = self._line()
        L = np.asarray(self.spec["wall"]["line"], float)
        k = int(np.argmin(np.abs(L[:, 0] - mo["a"])))
        back = nv[k] / np.linalg.norm(nv[k]); right = np.array([back[1], -back[0]])
        (x0, x1), (y0, y1) = mo["footprint"]
        corners = [c + right * x + back * y for x in (x0, x1) for y in (y0, y1)]
        zb = min(self.z(*q) for q in corners) - mo.get("sink", 0.02)
        to_w = lambda v: np.c_[c[0] + v[:, 0] * right[0] + v[:, 1] * back[0], c[1] + v[:, 0] * right[1] + v[:, 1] * back[1], zb + v[:, 2]]
        out = []
        npz = config.PROJECT_ROOT / mo["model"] / "monument.npz"
        if npz.exists():
            d = np.load(npz)
            out.append(("asama_monument", to_w(d["v"]), d["f"], d["uv"], "asama_monument", True))
        for (bx0, bx1), (by0, by1), h in mo["collision"]:
            q = np.array([[bx0, by0, -0.3], [bx1, by0, -0.3], [bx1, by1, -0.3], [bx0, by1, -0.3]])
            v = to_w(np.r_[q, q + [0, 0, h + 0.3]])
            fc = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])
            out.append((f"1WALL_asama_monument_{len(out)}", v, fc, v[:, :2], "stone", False))
        if not npz.exists():   # 焼き付け前の代わり：台座の箱と自然石
            ped = box(f, (a_ - 1.2, a_ + 1.2), (b_ - 0.75, b_ + 0.75), zb - 0.2, zb + 0.5, 1.0)
            out.append(("asama_monument", *_merge([ped, _slab(f, a_, b_, 1.35, 0.38, zb + 0.5, 2.0)]), "stone", True))
        return out

    def _stair_wall(self, a0, a1, b0, b1, z0, z1):
        """石段の左（西）の大きな間知の石積み（ストリートビュー 000・270・315）。石段の西の縁に沿って石段に向いた面（上の端は段から
        above_steps m、along_stairs m から先は low_end m まで下がる）と、石段の下の角から道ぞいに西へ road_len m の道に向いた面
        （高さ heights_road＝[西の端, 石段の角]）。どちらも のり batter で土の側へ傾け、厚み thick の中の詰まった壁にする（裏から見ても抜けない）。"""
        sw, f, sec = self.spec["stair_wall"], self.f, self.sec
        from .crosssection import GUTTER_WIDTH
        bt, T = sw["batter"], sw["thick"]
        slope = (z1 - z0) / (b1 - b0)
        # 石段ぞい：b0 → b0 + along_stairs + fade（土の側＝西＝−a）
        L1, fade = sw["along_stairs"], sw.get("fade", 2.5)
        bb = np.r_[np.arange(b0, b0 + L1 + fade, 1.0), b0 + L1 + fade]
        base1 = np.c_[f.xy(np.full(len(bb), a0 - 0.03), bb), z0 + (bb - b0) * slope]
        h1 = np.where(bb - b0 <= L1, sw["above_steps"], sw["above_steps"] + (sw["low_end"] - sw["above_steps"]) * (bb - b0 - L1) / fade)
        back1 = np.tile(-f.u, (len(bb), 1))
        # 道ぞい：石段の下の角から本線の右の縁ぞいに西（s が増える向き）へ
        c = f.xy(a0 - 0.03, b0)
        i0 = int(np.argmin(np.hypot(sec.x - c[0], sec.y - c[1])))
        ss = sec.s[i0] + np.arange(0.0, sw["road_len"] + 0.01, 1.0)
        idx = np.clip(np.searchsorted(sec.s, ss), 0, len(sec.s) - 1)
        nrm = np.c_[-np.sin(sec.heading[idx]), np.cos(sec.heading[idx])]           # 左（南）
        off = (sec.hw_right[idx] + GUTTER_WIDTH + 0.05)[:, None]
        p2 = np.c_[sec.x[idx], sec.y[idx]] - nrm * off
        p2[0] = c                                                                   # 石段の角から始める
        base2 = np.c_[p2, sec.z[idx] - 0.05]
        hw, hc = sw["heights_road"]
        t = (ss - ss[0]) / max(ss[-1] - ss[0], 1e-6)
        h2 = hc + (hw - hc) * t
        h2[0] = max(h2[0], z0 + sw["above_steps"] - base2[0, 2])
        back2 = -nrm                                                                # 土の側＝北（道の右の外）
        out = []
        for k, (base, h, back) in enumerate(((base1, h1, back1), (base2[::-1], h2[::-1], back2[::-1]))):
            out.append((f"1WALL_asama_stairwall_{k}", *_solid_wall(base, h, back, bt, T), sw.get("material", "kenchi"), True))
        return out

    # 奥の石積み：線に沿って、のり（batter）で奥へ傾く面。上には蔓草と低木がかぶさる
    def _line(self):
        L = np.asarray(self.spec["wall"]["line"], float)
        xy = self.f.xy(L[:, 0], L[:, 1])
        d = np.diff(xy, axis=0); d /= np.linalg.norm(d, axis=1)[:, None]
        nseg = np.c_[-d[:, 1], d[:, 0]]                     # 進む向きの左＝奥（山側）
        nv = np.r_[nseg[:1], nseg[:-1] + nseg[1:], nseg[-1:]]
        nv /= np.linalg.norm(nv, axis=1)[:, None]
        cos = np.r_[1.0, np.einsum("ij,ij->i", nv[1:-1], nseg[1:]), 1.0]
        return xy, nv / np.maximum(cos, 0.5)[:, None]

    def wall_point(self, a, off):
        """壁の下の端の、東の位置 a（m）での点から手前（駐車場側）へ off m の (a, b)。"""
        L = np.asarray(self.spec["wall"]["line"], float)
        b = float(np.interp(a, L[:, 0], L[:, 1]))
        xy, nv = self._line()
        i = int(np.argmin(np.abs(L[:, 0] - a)))
        p = self.f.xy(a, b) - nv[i] * off
        return self.f.ab(p)

    def _wall(self):
        w = self.spec["wall"]; H, bt = w["height"], w["batter"]
        xy, nv = self._line()
        zb = np.array([self.z(*p) for p in xy])
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        slant = H * np.hypot(1.0, bt)
        n = len(xy)
        bot = np.c_[xy, zb - 0.4]; top = np.c_[xy + nv * bt * H, zb + H]; back = np.c_[xy + nv * (bt * H + 0.5), zb + H]
        v = np.r_[bot, top, back]
        uv = np.r_[np.c_[s, np.full(n, -0.4)], np.c_[s, np.full(n, slant)], np.c_[s, np.full(n, slant + 0.5)]] / 2.0
        faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)] + [(n + i, n + i + 1, 2 * n + i + 1, 2 * n + i) for i in range(n - 1)]
        out = [("1WALL_asama_wall", v, np.array(faces, np.int64), uv, w.get("material", "kenchi"), True)]
        # 蔓草：下の端はぎざぎざ（vine_from の範囲）、壁の面の少し手前から上の端を越えて奥へ vine_over m、こんもり盛り上がる
        rng = np.random.default_rng(7)
        m = max(2, int(s[-1] / 0.6) + 1)
        ss = np.linspace(0.0, s[-1], m)
        P = np.c_[np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])]
        N = np.c_[np.interp(ss, s, nv[:, 0]), np.interp(ss, s, nv[:, 1])]
        Z = np.interp(ss, s, zb)
        lo, hi = w["vine_from"]
        hv = lo + (hi - lo) * np.clip(0.5 + 0.5 * np.sin(ss * 0.9 + 1.3) * np.cos(ss * 0.37) + rng.normal(0, 0.18, m), 0, 1)
        rows = [np.c_[P + N * (bt * hv - 0.18)[:, None], Z + hv],
                np.c_[P + N * (bt * (H * 0.5 + hv * 0.5) - 0.32)[:, None], Z + H * 0.5 + hv * 0.5],
                np.c_[P + N * (bt * H - 0.35), Z + H + 0.25 + rng.uniform(0, 0.25, m)],
                np.c_[P + N * (bt * H + w["vine_over"] * 0.5), Z + H + 0.75 + rng.uniform(0, 0.4, m)],
                np.c_[P + N * (bt * H + w["vine_over"]), Z + H + 0.6 + rng.uniform(0, 0.3, m)]]
        V = np.concatenate(rows)
        F = [(r * m + i, r * m + i + 1, (r + 1) * m + i + 1, (r + 1) * m + i) for r in range(len(rows) - 1) for i in range(m - 1)]
        UV = np.c_[np.tile(ss, len(rows)), np.repeat(np.arange(len(rows), dtype=float), m)] / 1.5
        lm = self.spec.get("landmarks", {}).get("vines")
        if lm and _landmark_exists(lm):
            out += self._vine_segments(lm, xy, nv, zb)
        else:
            out.append(("asama_vines", V, np.array(F, np.int64), UV, "hedge", True))
        return out

    def _vine_segments(self, lm, xy, nv, zb):
        """Codex 制作の蔓草の区切り（長さ seg m、原点＝壁の根元の線の真ん中、+X＝壁ぞい、+Y＝奥、のり 1:0.33 の壁の形に合わせてある）を
        壁の線に沿って並べる。種類は決まった順で混ぜ、頂点の上限（65535）に収まる数ずつ 1 つの部品にする。"""
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        seg = lm["seg"]; n = int(np.ceil(s[-1] / seg - 0.2))
        start = (s[-1] - n * seg) / 2
        parts, out = [], []
        for k in range(n):
            c_s = start + (k + 0.5) * seg
            j = int(np.clip(np.searchsorted(s, c_s) - 1, 0, len(xy) - 2))
            t = (c_s - s[j]) / (s[j + 1] - s[j])
            c = xy[j] + (xy[j + 1] - xy[j]) * t
            ux = (xy[j + 1] - xy[j]) / np.linalg.norm(xy[j + 1] - xy[j]); uy = np.array([-ux[1], ux[0]])   # 左＝奥
            z0 = zb[j] + (zb[j + 1] - zb[j]) * t
            v, f, uv = _load_landmark(lm, lm["names"][(k * 7 + 3) % len(lm["names"])])
            if parts and sum(len(q[0]) for q in parts) + len(v) > 65535:   # 部品 1 つの頂点の上限
                out.append((f"asama_vines_{len(out):02d}", *_merge(parts), lm["material"], True)); parts = []
            parts.append((np.c_[c + v[:, :1] * ux + v[:, 1:2] * uy, z0 + v[:, 2]], f, uv))
        if parts:
            out.append((f"asama_vines_{len(out):02d}", *_merge(parts), lm["material"], True))
        return out

    def _board(self, bo):
        """壁の面に貼った看板（壁ののりと同じだけ奥へ傾け、面から off m 手前。tilt_deg で面の中で回す）。"""
        bt = self.spec["wall"]["batter"]
        L = np.asarray(self.spec["wall"]["line"], float)
        i = int(np.clip(np.searchsorted(L[:, 0], bo["a"]), 1, len(L) - 1))
        xy, nv = self._line()
        d = xy[i] - xy[i - 1]; d /= np.linalg.norm(d); nb = nv[i] / np.linalg.norm(nv[i])
        c = self.f.xy(*self.wall_point(bo["a"], 0.0)); z0 = self.z(*c)
        t = np.radians(bo.get("tilt_deg", 0.0)); hw = bo["w"] / 2; zc = sum(bo["z"]) / 2; hh = (bo["z"][1] - bo["z"][0]) / 2
        v = []
        for u, q in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
            uu, qq = u * np.cos(t) - q * np.sin(t), u * np.sin(t) + q * np.cos(t)
            zz = zc + qq
            v.append([*(c + d * uu + nb * (bt * zz - bo["off"])), z0 + zz])
        return np.array(v), np.array([[0, 1, 2, 3]]), np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)

    def mouths(self, reach=0.6):
        """駐車場が本線に接している区間 [(side, s0, s1)]（この側は側溝を作らずに路面を続け、路肩の線を破線にする）。"""
        import shapely
        from .crosssection import GUTTER_WIDTH
        sec, edge = self.sec, self._pad.buffer(reach)
        out = []
        for side in (1, -1):
            hw = sec.hw_left if side > 0 else sec.hw_right
            nn = np.c_[-np.sin(sec.heading), np.cos(sec.heading)] * side
            p = np.c_[sec.x, sec.y] + nn * (hw + GUTTER_WIDTH * 0.5)[:, None]
            hit = shapely.contains_xy(edge, p[:, 0], p[:, 1])
            if hit.any():
                s = sec.s[hit]; out.append((side, float(s.min()), float(s.max())))
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


def _solid_wall(base, h, back, batter, thick):
    """根元の線 base（n×3、x y z）・高さ h（n）・土の側の向き back（n×2）から、のりで土の側へ傾いた表の面・上の面・裏の垂直な面・
    底・両端の面を持つ中の詰まった壁。面は外向き（表が見える側に頂点が反時計回り）。向きは面ごとに決める（曲がった壁では壁全体の
    中心が壁の外に出て、中心からの向きでは表の面が裏返った。2026-10-30）。uv は断面を一周する長さで付け、2 m で 1 枚。"""
    n = len(base)
    top = np.c_[base[:, :2] + back * (batter * h)[:, None], base[:, 2] + h]
    btop = np.c_[top[:, :2] + back * thick, top[:, 2]]
    bbot = np.c_[base[:, :2] + back * (batter * h + thick)[:, None], base[:, 2] - 0.4]
    fbot = np.c_[base[:, :2], base[:, 2] - 0.4]
    rings = [fbot, top, btop, bbot]
    v = np.concatenate(rings)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1))]
    edge = [np.linalg.norm(rings[(r + 1) % 4] - rings[r], axis=1) for r in range(4)]   # 断面の各辺の長さ
    acc = np.cumsum([np.zeros(n)] + edge[:3], axis=0)                                   # 断面を一周する長さ
    uv = np.concatenate([np.c_[s, acc[r]] for r in range(4)]) / 2.0
    bk = np.c_[back, np.zeros(n)]
    want = [-bk, np.tile([0.0, 0.0, 1.0], (n, 1)), bk, np.tile([0.0, 0.0, -1.0], (n, 1))]   # 表・上・裏・底の外向き
    faces, outs = [], []
    for r in range(4):
        r2 = (r + 1) % 4
        for i in range(n - 1):
            faces.append([r * n + i, r * n + i + 1, r2 * n + i + 1, r2 * n + i]); outs.append((want[r][i] + want[r][i + 1]) / 2)
    tan0 = np.r_[base[1, :2] - base[0, :2], 0.0]; tan1 = np.r_[base[-1, :2] - base[-2, :2], 0.0]
    for i, t in ((0, -tan0), (n - 1, tan1)):   # 両端の面は頂点を別に持ち、土の側への距離と高さで平らに uv を付ける
        q = np.array([fbot[i], top[i], btop[i], bbot[i]])
        k = len(v); v = np.r_[v, q]
        uv = np.r_[uv, np.c_[(q[:, :2] - fbot[i, :2]) @ back[i], q[:, 2] - fbot[i, 2]] / 2.0]
        faces.append([k, k + 1, k + 2, k + 3]); outs.append(t)
    faces = np.array(faces, np.int64)
    a_, b_, c_ = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    nrm = np.cross(b_ - a_, c_ - a_)
    flip = np.einsum("ij,ij->i", nrm, np.asarray(outs)) < 0
    faces[flip] = faces[flip][:, ::-1]
    return v, faces, uv


def _landmark_exists(lm):
    return all((config.PROJECT_ROOT / lm["dir"] / f"{n}.npz").exists() for n in lm["names"])


def _load_landmark(lm, name):
    """Codex 制作の部品（data/processed/landmarks/<id>/<name>.npz：v m・+Z 上・正面 −Y、f、uv）。
    uv_top＝true のものは v=0 が画像の上で書かれているので、このコースの決まり（v=0 が画像の下。kn5 で 1−v）へ直す。"""
    d = np.load(config.PROJECT_ROOT / lm["dir"] / f"{name}.npz")
    uv = np.array(d["uv"], float)
    if lm.get("uv_top"):
        uv[:, 1] = 1.0 - uv[:, 1]
    return np.asarray(d["v"], float), np.asarray(d["f"], np.int64), uv


def _wedge(f, a, b, z0, z_front, z_back):
    """上の面が手前（b0）から奥（b1）へ上がる箱（袖壁）。底なし。"""
    (a0, a1), (b0, b1) = a, b
    q = lambda aa, bb, zz: np.c_[f.xy(np.asarray(aa, float), np.asarray(bb, float)), np.asarray(zz, float)]
    zf, zk = z_front, z_back
    parts = [q([a0, a1, a1, a0], [b0] * 4, [z0, z0, zf, zf]), q([a1, a0, a0, a1], [b1] * 4, [z0, z0, zk, zk]),
             q([a1] * 4, [b0, b1, b1, b0], [z0, z0, zk, zf]), q([a0] * 4, [b1, b0, b0, b1], [z0, z0, zf, zk]),
             q([a0, a1, a1, a0], [b0, b0, b1, b1], [zf, zf, zk, zk])]
    out = []
    for v in parts:
        w = np.linalg.norm(v[1] - v[0]); h = max(np.linalg.norm(v[3] - v[0]), np.linalg.norm(v[2] - v[1]))
        out.append((v, np.array([[0, 1, 2, 3]]), np.array([[0, 0], [w, 0], [w, h], [0, h]]) / 1.0))
    return _merge(out)


def _tuft(x, y, z, size, seed=0, n=26):
    """ススキのような草の株：細い三角の葉を外へ反らせて束ねる（表裏とも描く）。"""
    rng = np.random.default_rng(100 + seed)
    vs, fs, us = [], [], []
    for k in range(n):
        ang = rng.uniform(0, 2 * np.pi); lean = rng.uniform(0.15, 0.75); L = size * rng.uniform(0.55, 1.0)
        d = np.array([np.cos(ang), np.sin(ang)]); side = np.array([-d[1], d[0]]) * 0.018 * size
        base = np.array([x, y]) + d * rng.uniform(0, 0.12) * size
        mid = base + d * L * lean * 0.45; tip = base + d * L * lean
        b = len(vs)
        vs += [[*(base - side), z - 0.03], [*(base + side), z - 0.03], [*(mid + side * 0.6), z + L * 0.55], [*(mid - side * 0.6), z + L * 0.55],
               [*tip, z + L * (0.95 - lean * 0.35)]]
        us += [[0, 0], [0.1, 0], [0.1, 0.5], [0, 0.5], [0.05, 1]]
        fs += [(b, b + 1, b + 2, b + 3), (b + 3, b + 2, b + 1, b), (b + 3, b + 2, b + 4, b + 4), (b + 4, b + 4, b + 2, b + 3)]
    return np.array(vs, float), np.array(fs, np.int64), np.array(us, float)


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


def board_texture(path, w=480, h=190):
    """「専用駐車場」の看板：白い板の上に青い小さな字の行、赤い大きな字、下に黒い小さな字の行（ストリートビュー 000）。"""
    img = Image.new("RGB", (w, h), (238, 238, 232))
    d = ImageDraw.Draw(img)
    small = ImageFont.truetype(config.font_file("gothic"), 17)
    big = ImageFont.truetype(config.font_file("gothic"), 74)
    d.text((w / 2, 10), "東田原ふれあい農園利用者", fill=(40, 60, 150), font=small, anchor="mt")
    d.text((w / 2, 34), "専用駐車場", fill=(205, 30, 30), font=big, anchor="mt")
    for y in (128, 152):
        d.rectangle((40, y, w - 40 - (y % 3) * 20, y + 9), fill=(60, 60, 70))
    d.rectangle((0, 0, w - 1, h - 1), outline=(150, 150, 150), width=4)
    img.save(path)


def notice_texture(path, w=96, h=128):
    """石碑の右の小さな白い掲示：上に赤い帯、下に黒い細い行。"""
    img = Image.new("RGB", (w, h), (236, 236, 230))
    d = ImageDraw.Draw(img)
    d.rectangle((6, 6, w - 6, 26), fill=(200, 36, 32))
    for y in range(38, h - 10, 12):
        d.rectangle((10, y, w - 10 - (y * 5) % 24, y + 4), fill=(70, 70, 70))
    d.rectangle((0, 0, w - 1, h - 1), outline=(140, 140, 140), width=2)
    img.save(path)


def build(spec, sec, ground=None):
    return Site(spec, sec, ground)
