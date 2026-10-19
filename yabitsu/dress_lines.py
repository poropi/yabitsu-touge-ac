"""S3 の路面の線（中央線・外側線・緑の路側帯）を、表（data/manual/s3_lines.csv）どおりに引き直す。
今の kn5 の線は CSP で隠し（表示だけ）、この線を仕上げの kn5 に入れる。2026-10-19 にストリートビュー 166 枚で全区間を確かめた。"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import meshgen as mg
from .crosssection import Section

STEP = 0.25            # 線を引く細かさ m（区切れた線のため、1 m より細かくする）
EDGE_INSET = 0.25      # 外側線の中心：道の端から内側へ
WIDTH = 0.15
LIFT = 0.012           # 今の線（0.005）より少し上
FLANK_OFFSET, FLANK_DASH = 0.9, (2.0, 2.0)
# 中央線の種類 → (材質, 区切り (線, すき間) m)
CENTRE = {"orange_solid": ("dress_line_orange", None), "orange_dotted": ("dress_line_orange", (0.75, 0.25)),
          "orange_flanked": ("dress_line_orange", None), "white_dashed": ("dress_line_white", (5.0, 5.0))}
EDGE_DASH = {"solid": None, "dashed": (2.0, 1.5)}


def resample(sec: Section, s_from: float, s_to: float, step=STEP) -> Section:
    """s_from〜s_to を step m ごとに並べ直した Section（線形補間、向きは unwrap して補間）。"""
    s = np.arange(max(s_from, sec.s[0]), min(s_to, sec.s[-1]) + 1e-9, step)
    f = lambda a: np.interp(s, sec.s, np.asarray(a, float))
    return replace(sec, s=s, x=f(sec.x), y=f(sec.y), z=f(sec.z), heading=np.interp(s, sec.s, np.unwrap(sec.heading)),
                   curvature=f(sec.curvature), lanes=np.round(f(sec.lanes)).astype(np.int32),
                   bridge=np.interp(s, sec.s, sec.bridge.astype(float)) > 0.5, hw_left=f(sec.hw_left),
                   hw_right=f(sec.hw_right), slope=f(sec.slope), crown=f(sec.crown))


def _green(rows, a, b):
    """a〜b に重なる緑の路側帯の幅（無ければ 0）。"""
    for r in rows:
        if r["item"] == "green" and float(r["s_from"]) < b and float(r["s_to"]) > a:
            return float(r["value"])
    return 0.0


def _pieces(rows, a, b):
    """a〜b を、緑の路側帯の境で分ける。[(a', b', 緑の幅)]"""
    cuts = sorted({a, b} | {float(r[k]) for r in rows if r["item"] == "green" for k in ("s_from", "s_to")
                            if a < float(r[k]) < b})
    return [(x, y, _green(rows, x, y)) for x, y in zip(cuts[:-1], cuts[1:])]


def line_meshes(sec: Section, rows) -> list:
    """表の行 → [(名前, verts, faces, uvs, 材質)]。"""
    out = []

    def add(name, sub, lat, dash, mat, width=WIDTH):
        v, f, uv = mg.compact(*mg.line(sub, slice(None), lat, width=width, lift=LIFT, dash=dash))
        if len(f):
            out.append((name, v, f, uv, mat))

    for k, r in enumerate(rows):
        a, b = float(r["s_from"]), float(r["s_to"])
        if r["item"] == "centre":
            sub = resample(sec, a, b)
            if sub.n < 2:
                continue
            mat, dash = CENTRE[r["value"]]
            add(f"dress_centre_{k:03d}", sub, np.zeros(sub.n), dash, mat)
            if r["value"] == "orange_flanked":
                for sgn, tag in ((1, "L"), (-1, "R")):
                    add(f"dress_flank{tag}_{k:03d}", sub, np.full(sub.n, sgn * FLANK_OFFSET), FLANK_DASH, "dress_line_white")
        elif r["item"] == "edge":
            for j, (x, y, g) in enumerate(_pieces(rows, a, b)):
                sub = resample(sec, x, y)
                if sub.n < 2:
                    continue
                for sgn, tag in ((1, "L"), (-1, "R")):
                    hw = sub.hw_left if sgn > 0 else sub.hw_right
                    add(f"dress_edge{tag}_{k:03d}_{j}", sub, sgn * (hw - EDGE_INSET - g), EDGE_DASH[r["value"]], "dress_line_white")
        elif r["item"] == "green":
            sub = resample(sec, a, b)
            g = float(r["value"])
            for sgn, tag in ((1, "L"), (-1, "R")):
                hw = sub.hw_left if sgn > 0 else sub.hw_right
                inner = hw - EDGE_INSET - g + WIDTH / 2          # 外側線の外から道の端まで
                add(f"dress_green{tag}_{k:03d}", sub, sgn * (inner + hw) / 2, None, "dress_green", width=float(np.mean(hw - inner)))
    return out


def hidden_line_meshes(manifest_dir, cl, s_range, near_m=40.0):
    """隠す今の線（外側線・中央線）の名前と、隠す範囲の s（S3 より外にはみ出した分も引き直すため）。"""
    manifest_dir = Path(manifest_dir)
    m = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    tree = cKDTree(np.c_[cl["x"], cl["y"]])
    names, lo, hi = [], s_range[1], s_range[0]
    for name, e in m["meshes"].items():
        if not name.startswith(("LINES_s1_edge", "LINES_track_centre")):
            continue
        v = np.load(manifest_dir / e["file"])["verts"]
        d, i = tree.query(v[:, :2])
        s = cl["s"][i][d < near_m]
        if len(s) and s.min() <= s_range[1] and s.max() >= s_range[0]:
            names += [name, f"{name}_back"]
            lo, hi = min(lo, s.min()), max(hi, s.max())
    return names, (lo, hi)
