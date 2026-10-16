import json

import numpy as np

from yabitsu import centerline, config, elevation, gsi
from yabitsu.osm import Route

P = config.DATA_PROCESSED


def main():
    route = Route.from_json(P / "s1_route.json")
    offsets = centerline.load_offsets_csv(config.DATA_MANUAL / "s1_offsets.csv")
    cl = centerline.build_centerline(route, offsets=offsets)
    cl.save(P / "s1_centerline.npz")
    z_raw = elevation.raw_profile(cl, gsi.default_sampler())
    valid = elevation.valid_mask(z_raw, cl.bridge, cl.step)
    z, z_ref, sigma = elevation.fit_profile(cl.s, z_raw, valid, cl.step)
    np.savez(P / "s1_profile.npz", s=cl.s, z_raw=z_raw, z_ref=z_ref, z=z, valid=valid)
    st = elevation.profile_stats(cl.s, z, z_ref, valid, cl.step)
    st["sigma_m"] = sigma; st["offset_rows"] = int(len(offsets))
    elevation.plot_profile(P / "review" / "s1_profile.png", cl.s, z_ref, z, valid, "S1 Nanohanadai -> Yabitsu pass")
    elevation.plot_plan(P / "review" / "s1_plan.png", cl)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
