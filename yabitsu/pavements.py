"""路面に重ねる模様（見た目だけ。走りは下の路面のまま）：ヘアピンの赤い滑り止め、丸い模様のコンクリート舗装、横断歩道。
表（data/manual/{区間}_pavements.csv）の列：s_from, s_to, kind, note。"""
from dataclasses import dataclass

import numpy as np

from .crosssection import surface_height
from .meshgen import compact, ribbon

KINDS = ("red_stripes", "concrete_circles", "crosswalk")
LIFT = 0.005
STRIPE_LEN, STRIPE_EVERY = 0.6, 1.8
BAR_W, BAR_EVERY, CURB_GAP = 0.45, 0.9, 0.5
MATERIALS = {"red_stripes": "paint_red", "concrete_circles": "concrete_circles", "crosswalk": "line_white"}


@dataclass
class Pavement:
    s_from: float
    s_to: float
    kind: str


def load_rows(rows):
    out = []
    for r in rows:
        if r.get("s_from") in (None, ""):
            continue
        kind = str(r["kind"]).strip()
        if kind not in KINDS:
            raise ValueError(f"kind は {KINDS} のどれか: {r}")
        out.append(Pavement(float(r["s_from"]), float(r["s_to"]), kind))
    return out


def _rows(sec, a, b, ends_only=False):
    """a〜b の s の並び（区間の端を必ず含む。ends_only なら両端だけ）と、その位置の道の値。"""
    s = np.array([a, b]) if ends_only else np.unique(np.r_[a, sec.s[(sec.s > a) & (sec.s < b)], b])
    at = lambda arr: np.interp(s, sec.s, arr)
    h = np.unwrap(np.arctan2(at(np.sin(sec.heading)), at(np.cos(sec.heading))))
    return s, at(sec.x), at(sec.y), h, at(sec.z), at(sec.slope), at(sec.crown), at(sec.hw_left), at(sec.hw_right)


def _patch(sec, a, b, lat_lo, lat_hi, n_lat=5, mask=None, ends_only=False):
    s, x, y, h, z, sl, cr, hl, hr = _rows(sec, a, b, ends_only)
    lo = lat_lo(hl, hr)[:, None]; hi = lat_hi(hl, hr)[:, None]
    lat = lo + (hi - lo) * np.linspace(0, 1, n_lat)[None, :]
    zz = surface_height(lat, z[:, None], sl[:, None], cr[:, None]) + LIFT
    return ribbon(x, y, h, lat, zz, s, u_scale=2.0, v_scale=2.0, face_mask=mask)


def meshes(sec, pavements):
    out = []
    for k, p in enumerate(pavements):
        name = f"LINES_track_pave_{k:03d}"
        if p.kind == "concrete_circles":
            v, f, uv = _patch(sec, p.s_from, p.s_to, lambda hl, hr: -hr, lambda hl, hr: hl)
        elif p.kind == "red_stripes":
            parts = [_patch(sec, a, a + STRIPE_LEN, lambda hl, hr: -hr, lambda hl, hr: hl, n_lat=5, ends_only=True)
                     for a in np.arange(p.s_from, p.s_to - STRIPE_LEN + 1e-9, STRIPE_EVERY)]
            v, f, uv = _merge(parts)
        else:  # crosswalk：道の向きに平行な白い帯を、路肩から CURB_GAP 内側まで横に並べる
            parts = []
            hl = float(np.interp((p.s_from + p.s_to) / 2, sec.s, sec.hw_left))
            hr = float(np.interp((p.s_from + p.s_to) / 2, sec.s, sec.hw_right))
            for c in np.arange(-hr + CURB_GAP + BAR_W / 2, hl - CURB_GAP - BAR_W / 2 + 1e-9, BAR_EVERY):
                parts.append(_patch(sec, p.s_from, p.s_to, lambda a, b, c=c: np.full_like(a, c - BAR_W / 2),
                                    lambda a, b, c=c: np.full_like(a, c + BAR_W / 2), n_lat=2, ends_only=True))
            v, f, uv = _merge(parts)
        v, f, uv = compact(v, f, uv)
        out.append((name, v, f, uv, MATERIALS[p.kind]))
    return out


def _merge(parts):
    vs, fs, us, base = [], [], [], 0
    for v, f, uv in parts:
        vs.append(v); fs.append(np.asarray(f) + base); us.append(uv); base += len(v)
    return np.concatenate(vs), np.concatenate(fs), np.concatenate(us)
