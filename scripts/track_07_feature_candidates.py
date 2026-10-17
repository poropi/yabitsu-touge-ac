import csv

from yabitsu import config, features, geo, gsi, track
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "track_section.json")
    bounds = track.load_bounds()
    dem = gsi.default_sampler()
    rows = features.suggest_candidates(sec, lambda x, y: dem.sample(*geo.from_local(x, y)))
    for name, (s0, s1) in bounds.items():  # 区間ごとに、区間の始点からの距離で書く
        mine = [{**r, "s_from": round(r["s_from"] - s0), "s_to": round(min(r["s_to"], s1) - s0)}
                for r in rows if s0 <= r["s_from"] < s1]
        out = P / "review" / f"{name}_feature_candidates.csv"
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, ["s_from", "s_to", "side", "type", "height_m", "streetview"]); w.writeheader(); w.writerows(mine)
        print(f"{name}: {len(mine)} candidates -> {out}")


if __name__ == "__main__":
    main()
