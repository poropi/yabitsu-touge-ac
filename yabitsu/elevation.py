from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d, median_filter

from . import geo

SIGMAS = (30, 25, 20, 16, 12, 10, 8, 6, 5, 4, 3)


def raw_profile(cl, sampler, unproject=geo.from_local):
    lat, lon = unproject(cl.x, cl.y)
    return sampler.sample(lat, lon)


def valid_mask(z_raw, bridge, step, margin_m=5.0):
    b = np.asarray(bridge, bool)
    if b.any():
        b = binary_dilation(b, iterations=max(1, int(round(margin_m / step))))
    return ~np.isnan(z_raw) & ~b


def _fill(s, z, valid):
    return np.interp(s, s[valid], z[valid])


def despike(z, valid, step, window_m=9.0):
    w = max(3, int(round(window_m / step)) | 1)
    return median_filter(z, size=w, mode="nearest")


def fit_profile(s, z_raw, valid, step, tol=1.5, sigmas=SIGMAS):
    zref = despike(_fill(s, z_raw, valid), valid, step)
    z = zref
    for sg in sigmas:
        z = gaussian_filter1d(zref, sg / step, mode="nearest")
        if np.abs(z - zref)[valid].max() <= tol:
            return z, zref, sg
    return z, zref, sigmas[-1]


def fit_sections(s, z_raw, valid, step, bounds, blend_m=60.0, sample=None):
    """区間ごとに fit_profile する。上の区間の値はそのまま使い、境目の下側 blend_m の間だけ、下の区間の値を
    「上の区間の境目の高さと勾配で延ばした直線」へなめらかに寄せる（高さと勾配が境目でそろう）。

    sample(s) -> 標高 を渡すと、区間ごとに「区間の始点からの格子」で標高を取り直して合わせ、通しの格子へ戻す。
    下に区間を足して通しの格子が区間に対してずれても、その区間の高さは変わらない。"""
    s = np.asarray(s, float); z_raw = np.asarray(z_raw, float); valid = np.asarray(valid, bool)
    z = np.empty_like(s); z_ref = np.empty_like(s); sigmas = []
    for n, (a, b) in enumerate(bounds):
        m = (s >= a - 1e-6) & (s <= b + 1e-6)
        # 下の区間は境目の先 blend_m まで含めて合わせる（区間の端でなめらか化が平らになるのを避ける）。
        # 一番上の区間は、自分の範囲だけで合わせる（つないでも結果が変わらない）
        top = min(b + blend_m, s[-1]) if n < len(bounds) - 1 else b
        if sample is None:
            ext = m | ((s > b) & (s <= top))
            zk, rk, sg = fit_profile(s[ext], z_raw[ext], valid[ext], step)
            keep = m[ext]
            z[m], z_ref[m] = zk[keep], rk[keep]
        else:
            g = a + np.arange(0.0, top - a + 1e-6, 1.0)  # 中心線の step（L/n）ではなく 1m ちょうど：区間を足しても格子が同じ
            g = np.append(g, top) if top - g[-1] > 1e-6 else g
            gi = np.clip(np.rint((g - s[0]) / step).astype(int), 0, len(s) - 1)
            zg = np.asarray(sample(g), float)
            vg = valid[gi] & ~np.isnan(zg)
            zk, rk, sg = fit_profile(g, np.where(np.isnan(zg), z_raw[gi], zg), vg, 1.0)
            z[m], z_ref[m] = np.interp(s[m], g, zk), np.interp(s[m], g, rk)
        sigmas.append(sg)
    k = max(1, int(round(10.0 / step)))
    for (_, b), _ in zip(bounds[:-1], bounds[1:]):
        i = int(np.argmin(np.abs(s - b)))
        j = min(i + k, len(s) - 1)
        g = (z[j] - z[i]) / max(s[j] - s[i], 1e-9)
        w_m = (s >= b - blend_m) & (s < s[i])
        t = (s[w_m] - (b - blend_m)) / blend_m; w = t * t * (3 - 2 * t)
        z[w_m] = (1 - w) * z[w_m] + w * (z[i] + g * (s[w_m] - s[i]))
    return z, z_ref, sigmas


def profile_stats(s, z, z_ref, valid, step, tol=1.5):
    k = max(1, int(round(10 / step)))
    grade = (z[k:] - z[:-k]) / (s[k:] - s[:-k]) * 100
    curv = np.gradient(np.gradient(z, step), step)
    r = np.abs(z - z_ref)[valid]
    return {"length_m": float(s[-1] - s[0]), "z_start": float(z[0]), "z_end": float(z[-1]),
            "rise_m": float(z[-1] - z[0]), "max_grade_pct": float(np.abs(grade).max()),
            "mean_grade_pct": float((z[-1] - z[0]) / (s[-1] - s[0]) * 100),
            "max_vertical_curvature": float(np.abs(curv).max()), "max_residual_m": float(r.max()),
            "pct_within_tol": float((r <= tol).mean() * 100)}


def plot_profile(png, s, z_ref, z, valid, title):
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    a.plot(s, z_ref, color="0.6", lw=1, label="DEM (despiked)")
    a.plot(s, z, color="C0", lw=1.5, label="road profile")
    a.plot(s[~valid], z_ref[~valid], ".", color="C3", ms=2, label="excluded (bridge / no data)")
    a.set_ylabel("elevation [m]"); a.legend(); a.set_title(title); a.grid(alpha=0.3)
    k = 10
    b.plot(s[k:], (z[k:] - z[:-k]) / (s[k:] - s[:-k]) * 100, color="C1")
    b.set_ylabel("grade over 10 m [%]"); b.set_xlabel("distance s [m]"); b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)


def plot_plan(png, cl):
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 12))
    ax.plot(cl.x, cl.y, color="C0")
    for sv in range(0, int(cl.s[-1]) + 1, 250):
        i = int(np.argmin(np.abs(cl.s - sv)))
        ax.plot(cl.x[i], cl.y[i], "o", color="C3", ms=3); ax.annotate(f"{sv}", (cl.x[i], cl.y[i]), fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]")
    fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
