import csv
import json
import sys
from pathlib import Path

import numpy as np

from yabitsu import config, elevation
from yabitsu.meshgen import folded_faces


def check(processed_dir, manual_dir):
    P, M = Path(processed_dir), Path(manual_dir)
    pr = np.load(P / "s1_profile.npz")
    st = elevation.profile_stats(pr["s"], pr["z"], pr["z_ref"], pr["valid"], float(pr["s"][1] - pr["s"][0]))
    with (M / "s1_position_review.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    position_ok = bool(rows) and all(r["ok"].strip().lower() == "yes" and float(r["max_offset_m"]) <= 2.0 for r in rows)
    man = json.loads((P / "s1" / "manifest.json").read_text(encoding="utf-8"))
    folds = tris = 0
    for name in man["groups"]["yabitsu_s1"]["meshes"]:
        d = np.load(P / "s1" / man["meshes"][name]["file"])
        tris += len(d["faces"]) * (d["faces"].shape[1] - 2)
        if name.startswith("1ROAD_s1_") and "gutter" not in name:
            folds += len(folded_faces(d["verts"], d["faces"]))
    return {"elevation_ok": st["max_residual_m"] <= 1.5, "max_residual_m": st["max_residual_m"],
            "max_grade_pct": st["max_grade_pct"], "position_ok": position_ok, "position_rows": len(rows),
            "folds": folds, "tris": tris}


if __name__ == "__main__":
    r = check(config.DATA_PROCESSED, config.DATA_MANUAL)
    print(json.dumps(r, indent=1))
    kn5 = sorted(config.TRACK_DIR.glob("*.kn5"))
    print("kn5:", {p.name: f"{p.stat().st_size / 1e6:.0f} MB" for p in kn5})
    sys.exit(0 if r["elevation_ok"] and r["position_ok"] and r["folds"] == 0 else 1)
