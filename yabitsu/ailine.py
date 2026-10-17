"""区間データから AI ライン（ai/fast_lane.ai）を直接作る（AC の中で走って記録しない）。

形式は Kunos の fast_lane.ai（spa / magione / imola）を最後まで読んで確かめた：
ヘッダ (7, 点の数, 0, 0) → 点（位置・累積距離・番号）→ 点ごとの 18 個の値 → 10m 格子の近傍表。
"""
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .config import assert_writable
from .kn5 import C

VERSION = 7
GRID_MARGIN = 350.0  # 実在の AI ラインは点の範囲 ±350m を格子にしている
GRID_STEP = 10.0
GRID_NEIGHBOURS = 10
G = 9.81
EXTRA = ["speed", "gas", "brake", "lat_g", "radius", "side_left", "side_right", "camber", "dir",
         "normal", "normal", "normal", "seg_length", "forward", "forward", "forward", "tag", "grade"]
POINT = np.dtype([("pos", "<f4", 3), ("length", "<f4"), ("id", "<i4")])


def speed_profile(curvature, step, v_max=27.8, a_lat=7.0, a_acc=3.0, a_brake=7.0):
    """曲がりやすさ（横 G）と加速・減速の限界から、点ごとの目安の速さ・アクセル・ブレーキを決める。"""
    k = np.abs(np.asarray(curvature, float))
    lim = np.minimum(v_max, np.sqrt(a_lat / np.maximum(k, 1e-9)))
    fwd, bwd = lim.copy(), lim.copy()
    for i in range(1, len(k)):
        fwd[i] = min(fwd[i], np.sqrt(fwd[i - 1] ** 2 + 2 * a_acc * step))
    for i in range(len(k) - 2, -1, -1):
        bwd[i] = min(bwd[i], np.sqrt(bwd[i + 1] ** 2 + 2 * a_brake * step))
    v = np.minimum(fwd, bwd)
    decel = np.r_[-(np.diff(v ** 2) / 2) / step, 0.0]
    brake = np.clip(decel / a_brake, 0.0, 1.0)
    gas = np.where(brake > 1e-9, 0.0, 1.0)
    return v, gas, brake


def _driving(sec, direction):
    if direction > 0:
        return (sec.x, sec.y, sec.z, sec.hw_left, sec.hw_right, sec.slope, sec.curvature)
    r = slice(None, None, -1)
    return (sec.x[r], sec.y[r], sec.z[r], sec.hw_right[r], sec.hw_left[r], -sec.slope[r], -sec.curvature[r])


def _grid(xz):
    mx = xz.max(0) + GRID_MARGIN; mn = xz.min(0) - GRID_MARGIN
    cols, rows = (int((mx - mn)[0] // GRID_STEP), int((mx - mn)[1] // GRID_STEP))
    ci, cj = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")
    centres = np.c_[mn[0] + (ci.ravel() + 0.5) * GRID_STEP, mn[1] + (cj.ravel() + 0.5) * GRID_STEP]
    k = min(GRID_NEIGHBOURS, len(xz))
    _, idx = cKDTree(xz).query(centres, k=k)
    return mx, mn, cols, rows, np.asarray(idx, np.int32).reshape(cols, rows, k)


def write_fast_lane(path, sec, direction, **speed_kw) -> Path:
    path = assert_writable(path)
    x, y, z, hl, hr, slope, kb = _driving(sec, direction)
    pb = np.c_[x, y, z].astype(float)
    pos = pb @ C.T
    n = len(pos)
    seg = np.r_[np.linalg.norm(np.diff(pos, axis=0), axis=1), np.linalg.norm(pos[0] - pos[-1])]
    length = np.r_[0.0, np.cumsum(seg[:-1])]
    tb = np.diff(pb, axis=0); tb = np.vstack([tb, tb[-1]]); tb /= np.linalg.norm(tb, axis=1, keepdims=True)
    left = np.c_[-tb[:, 1], tb[:, 0], np.zeros(n)]; left /= np.linalg.norm(left, axis=1, keepdims=True)
    nb = np.cross(tb, left + slope[:, None] * np.array([0.0, 0.0, 1.0]))
    nb /= np.linalg.norm(nb, axis=1, keepdims=True)
    fwd, nrm = tb @ C.T, nb @ C.T
    v, gas, brake = speed_profile(kb, float(np.median(seg[:-1])), **speed_kw)
    turn = np.where(kb >= 0, 1.0, -1.0)  # 左カーブ = +1（fwd_i × fwd_i+1 の Y の符号と同じ）
    extra = np.c_[v, gas, brake, v ** 2 * kb / G, 1.0 / np.maximum(np.abs(kb), 1e-7), hl, hr, np.arctan(slope),
                  turn, nrm, seg, fwd, np.zeros(n), fwd[:, 1]].astype("<f4")

    pts = np.zeros(n, POINT); pts["pos"] = pos; pts["length"] = length; pts["id"] = np.arange(n)
    mx, mn, cols, rows, cells = _grid(pos[:, [0, 2]])
    out = [struct.pack("<4i", VERSION, n, 0, 0), pts.tobytes(), struct.pack("<i", n), extra.tobytes(),
           struct.pack("<i3f3fif", 1, mx[0], 0.0, mx[1], mn[0], 0.0, mn[1], GRID_NEIGHBOURS, GRID_STEP),
           struct.pack("<i", cols)]
    k = cells.shape[2]
    for i in range(cols):
        out.append(struct.pack("<i", rows))
        out.append(np.c_[np.full(rows, k, np.int32), cells[i]].astype("<i4").tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(out))
    return path


def read_fast_lane(path) -> dict:
    d = Path(path).read_bytes()
    version, n, _, _ = struct.unpack_from("<4i", d, 0); off = 16
    pts = np.frombuffer(d, POINT, count=n, offset=off); off += n * POINT.itemsize
    off += 4
    raw = np.frombuffer(d, "<f4", count=n * 18, offset=off).reshape(n, 18).astype(float); off += n * 72
    extra = {}
    for j, name in enumerate(EXTRA):
        if name in ("normal", "forward"):
            extra.setdefault(name, raw[:, j:j + 3])
        else:
            extra[name] = raw[:, j]
    _, mxx, mxy, mxz, mnx, mny, mnz, nbh, step = struct.unpack_from("<i3f3fif", d, off); off += 36
    cols, = struct.unpack_from("<i", d, off); off += 4
    cells = []
    for _ in range(cols):
        rows, = struct.unpack_from("<i", d, off); off += 4
        col = []
        for _ in range(rows):
            c, = struct.unpack_from("<i", d, off); off += 4
            col.append(np.frombuffer(d, "<i4", count=c, offset=off)); off += 4 * c
        cells.append(col)
    if off != len(d):
        raise ValueError(f"{path}: 末尾に {len(d) - off} バイト余りがあります")
    return {"version": version, "pos": pts["pos"].astype(float), "length": pts["length"].astype(float),
            "id": pts["id"].copy(), "extra": extra,
            "grid": {"max": (mxx, mxy, mxz), "min": (mnx, mny, mnz), "step": step, "neighbours": nbh,
                     "cols": cols, "rows": len(cells[0]) if cells else 0, "cells": cells}}
