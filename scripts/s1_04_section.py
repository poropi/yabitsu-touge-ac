import json

import numpy as np

from yabitsu import config, crosssection as cs, start_lot
from yabitsu.centerline import Centerline

P = config.DATA_PROCESSED


def main():
    cl = Centerline.load(P / "s1_centerline.npz")
    prof = np.load(P / "s1_profile.npz")
    sec = cs.build_section(cl, prof["z"], cs.load_widths_csv(config.DATA_MANUAL / "s1_widths.csv"))
    sec.save_json(P / "s1_section.json")
    pads = cs.pads_from_parking(json.loads((P / "s1_parking.json").read_text(encoding="utf-8")), sec)
    lot = start_lot.load_spec(config.DATA_MANUAL / "s1_start_lot.json")
    if lot:
        pads = start_lot.apply(pads, lot, sec)
    (P / "s1_pads.json").write_text(json.dumps(pads, ensure_ascii=False), encoding="utf-8")
    print(f"section rows={sec.n} width {2 * sec.hw_left.min():.1f}-{2 * sec.hw_left.max():.1f} m, "
          f"max slope {np.abs(sec.slope).max() * 100:.1f}%, pads={[(p['name'], p['role']) for p in pads]}")


if __name__ == "__main__":
    main()
