"""植栽計画 JSON を追植用 CSV へ写す。"""
import argparse
import csv
import json
from pathlib import Path


COLUMNS = ("site", "section", "s_m", "side", "edge_offset_m", "species", "height_m", "group", "note")
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data/manual/extra_trees.csv"
SITE_BUILDINGS = {"A-s4-1724": "vm873_477"}


def convert(plan, output, building_id=None):
    data = json.loads(plan.read_text(encoding="utf-8"))
    site = data["site"]
    conditional = data.get("if_building_removed", [])
    building_id = building_id or data.get("building_id") or SITE_BUILDINGS.get(site)
    if conditional and not building_id:
        raise ValueError("条件付きの木があるため --building-id を指定してください")
    rows = []
    for group, trees in (("always", data["trees"]), (f"if_building_removed:{building_id}", conditional)):
        for tree in trees:
            rows.append([site, *(tree[key] for key in COLUMNS[1:7]), group, tree.get("note", "")])
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--building-id")
    args = parser.parse_args()
    try:
        count = convert(args.plan, args.output, args.building_id)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{args.output}: {count} 本分")


if __name__ == "__main__":
    main()
