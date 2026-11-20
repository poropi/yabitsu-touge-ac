"""依頼61の試作。保存済みの高木を立体木と遠景板のKN5に置き換える。"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from yabitsu import forest_light
from yabitsu.config import assert_writable

WORK = Path("export/compare/route70/work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=WORK / "forest_57_full/extension/trees")
    parser.add_argument("--near-lists", type=Path, help="保存済みの道ぎわ一覧を使う場合に指定")
    parser.add_argument("--original-models", type=Path,
                        default=WORK / "supervisor_41.fmEVx6/deliver/forest/extension/trees")
    parser.add_argument("--catalog", type=Path, default=forest_light.CATALOG)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/track/manifest.json"))
    parser.add_argument("--shadows", choices=("none", "near", "all"), default="none")
    parser.add_argument("--normals", choices=("mesh", "up"), default="mesh")
    parser.add_argument("--ambient", type=float, default=forest_light.DEFAULT_AMBIENT)
    parser.add_argument("--diffuse", type=float, default=forest_light.DEFAULT_DIFFUSE)
    parser.add_argument("--texture", choices=("middle", "dark"), help="省略時は両方を書き出す")
    parser.add_argument("--mip-alpha", choices=("box", "coverage"), default="box")
    parser.add_argument("--output", type=Path, help="KN5の出力ファイル名。--textureと併用")
    args = parser.parse_args()
    if args.output is not None and args.texture is None:
        parser.error("--output には --texture の指定が必要です")
    if not (math.isfinite(args.ambient) and math.isfinite(args.diffuse)):
        parser.error("--ambient と --diffuse には有限の値を指定してください")
    output_file = assert_writable(args.output) if args.output is not None else None
    output = output_file.parent if output_file else assert_writable(WORK / "treelight_61")
    output.mkdir(parents=True, exist_ok=True)
    settings = replace(forest_light.load_settings(), shadows=args.shadows, normals=args.normals,
                       ambient=args.ambient, diffuse=args.diffuse, mip_alpha=args.mip_alpha)
    variants = (args.texture,) if args.texture else ("middle", "dark")
    outputs = {variant: output_file or output / f"yabitsu_treelight_{variant}.kn5" for variant in variants}
    stats = forest_light.build(args.source, args.original_models, args.manifest, outputs, settings,
                               catalog=args.catalog, near_lists=args.near_lists, verify=True)
    if args.texture is None:
        (output / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
