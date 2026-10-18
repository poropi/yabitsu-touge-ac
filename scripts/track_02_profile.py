import json

import numpy as np

from yabitsu import centerline, config, elevation, geo, gsi, track
from yabitsu.osm import Route

P = config.DATA_PROCESSED


def load_route():
    d = json.loads((P / "track_route.json").read_text(encoding="utf-8"))
    return Route(d["lat"], d["lon"], d["seg_lanes"], d["seg_bridge"]), [tuple(j) for j in d["joints"]]


def main():
    route, joints = load_route()
    cl = centerline.build_centerline(route)
    bounds = track.section_bounds(cl, config.SECTIONS, joints)
    rows = track.load_section_rows("offsets", bounds)
    offsets = np.array([(float(r["s_m"]), float(r["lateral_m"])) for r in rows]).reshape(-1, 2)
    if len(offsets):
        cl = centerline.build_centerline(route, offsets=offsets)
        bounds = track.section_bounds(cl, config.SECTIONS, joints)
    cl.save(P / "track_centerline.npz")
    sampler = gsi.default_sampler()
    z_raw = elevation.raw_profile(cl, sampler)
    valid = elevation.valid_mask(z_raw, cl.bridge, cl.step)

    def sample(q):  # 区間ごとの格子（区間の始点から 1m ごと）で標高を取る
        return sampler.sample(*geo.from_local(np.interp(q, cl.s, cl.x), np.interp(q, cl.s, cl.y)))

    z, z_ref, sigmas = elevation.fit_sections(cl.s, z_raw, valid, cl.step, list(bounds.values()), sample=sample,
                                              fixed=len(config.ACCEPTED_SECTIONS), robust=True)   # 受け入れ済みの区間は今の高さのまま。新しい区間は標高データの短い谷を外す
    np.savez(P / "track_profile.npz", s=cl.s, z_raw=z_raw, z_ref=z_ref, z=z, valid=valid,
             bounds=np.array(bounds, dtype=object))
    st = elevation.profile_stats(cl.s, z, z_ref, valid, cl.step)
    st["sigmas"] = dict(zip(bounds, sigmas)); st["bounds"] = bounds; st["offset_rows"] = int(len(offsets))
    elevation.plot_profile(P / "review" / "track_profile.png", cl.s, z_ref, z, valid, "Yabitsu track (all sections)")
    elevation.plot_plan(P / "review" / "track_plan.png", cl)
    print(json.dumps(st, indent=1, default=float))


if __name__ == "__main__":
    main()
