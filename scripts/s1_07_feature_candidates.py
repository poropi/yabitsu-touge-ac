import csv

from yabitsu import config, features, geo, gsi
from yabitsu.crosssection import Section

P = config.DATA_PROCESSED


def main():
    sec = Section.load_json(P / "s1_section.json")
    dem = gsi.default_sampler()
    rows = features.suggest_candidates(sec, lambda x, y: dem.sample(*geo.from_local(x, y)))
    out = P / "review" / "s1_feature_candidates.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["s_from", "s_to", "side", "type", "height_m", "streetview"]); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} candidates -> {out}")


if __name__ == "__main__":
    main()
