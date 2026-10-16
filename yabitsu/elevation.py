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
