"""現行の橋まわりを、手直し用の Blender シーンへ切り出す。"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yabitsu import config, valley_site_source


def main():
    route = config.COMPARE_DIR / "route70"
    captured_forest = route / "work" / "supervisor_41.fmEVx6" / "deliver" / "forest" / "trees.npy"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=valley_site_source.SITES)
    parser.add_argument("--radius", type=float, default=60.0, help="半径（メートル、最大160）")
    parser.add_argument("--output-root", type=Path, default=ROOT / "export" / "blender_sites")
    parser.add_argument("--edit-root", type=Path, default=config.DATA_MANUAL / "valley_edits")
    parser.add_argument("--processed", type=Path, default=config.DATA_PROCESSED)
    parser.add_argument("--forest", type=Path, default=captured_forest, help="表示用の木の trees.npy")
    parser.add_argument("--blender", type=Path, default=config.BLENDER_EXE)
    args = parser.parse_args()
    if not 0 < args.radius <= 160:
        parser.error("半径は 0 より大きく 160 m 以下にしてください")
    output = args.output_root.resolve() / args.site
    if output.exists() or (args.edit_root / args.site).exists():
        parser.error(f"既存の作業を保護するため site.blend または書き戻し成果を上書きできません: {output}"
                     "。退避するか --output-root と --edit-root に別の場所を指定してください")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 生成の途中や失敗したシーンを作業用フォルダに残さない。
        with TemporaryDirectory(prefix=".valley_site_", dir=output.parent) as temporary:
            workspace = Path(temporary)
            data = valley_site_source.write_bundle(workspace, args.processed.resolve(), config.DATA_MANUAL,
                                                    route, args.forest.resolve(), args.site, args.radius)
            subprocess.run([str(args.blender), "-b", "--factory-startup", "--python-exit-code", "1",
                            "--python", str(ROOT / "blender_scripts" / "valley_site_export.py"), "--",
                            str(workspace)], check=True)
            shutil.copyfile(workspace / "site.blend", workspace / "site_original.blend")
            data["baseline_blend_sha256"] = valley_site_source.sha256(workspace / "site.blend")
            (workspace / "site.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            (workspace / "render.npz").unlink()
            (workspace / "render.json").unlink()
            workspace.rename(output)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"{output / 'site.blend'} {data['stats']}")


if __name__ == "__main__":
    main()
