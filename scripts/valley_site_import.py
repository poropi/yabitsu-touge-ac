"""Write checked Blender edits to data/manual/valley_edits/<site>."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yabitsu import config, valley_edits
from yabitsu.valley_site_import import source_fingerprint, stage_edit


SITES = ("A-s4-1724", "B-s4-6354", "C-s5-8956")
SITES_ROOT = config.PROJECT_ROOT / "export" / "blender_sites"
EDITS_ROOT = config.DATA_MANUAL / "valley_edits"
BLENDER_SCRIPT = config.PROJECT_ROOT / "blender_scripts" / "valley_site_import.py"


def import_site(site, site_dir, blend, edits_root, blender):
    site_dir = site_dir.resolve()
    edits_root = edits_root.resolve()
    if (edits_root.is_relative_to(config.PROJECT_ROOT)
            and not edits_root.is_relative_to(EDITS_ROOT)):
        raise ValueError("--edits-root は標準の手直し先か、リポジトリ外を指定してください")
    if not blend.is_file():
        raise ValueError(f"Blender ファイルがありません: {blend}")
    metadata = json.loads((site_dir / "site.json").read_text(encoding="utf-8"))
    if metadata["site"] != site:
        raise ValueError("指定した橋と site.json の橋が違います")
    destination = edits_root / site
    edits_root.mkdir(parents=True, exist_ok=True)
    lock = edits_root / f".{site}.importing"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(f"この橋の書き戻しは実行中です: {lock}") from exc
    try:
        if destination.exists():
            prior = json.loads((destination / "edit.json").read_text(encoding="utf-8"))
            if prior["fingerprint"] != source_fingerprint(metadata):
                raise ValueError("元データの指紋が異なる手直しがあります。保存先を分けてください")
        with TemporaryDirectory(prefix=f".{site}.", dir=edits_root) as temporary:
            staging = Path(temporary)
            capture = staging / "scene.npz"
            blend_sha256 = valley_edits.file_digest(blend)
            subprocess.run(
                [str(blender), "-b", "--factory-startup", "--disable-autoexec", str(blend.resolve()),
                 "--python-exit-code", "1", "--python", str(BLENDER_SCRIPT), "--", str(capture), site],
                check=True,
            )
            result = staging / "edit"
            result.mkdir()
            edit = stage_edit(site_dir, capture, result)
            if blend_sha256 != valley_edits.file_digest(blend):
                raise ValueError("書き戻し中に Blender ファイルが更新されました")
            edit["blend_sha256"] = blend_sha256
            edit["source_sha256"] = valley_edits.file_digest(site_dir / "source.npz")
            (result / "edit.json").write_text(json.dumps(edit, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
            backup = lock / "previous"
            try:
                if destination.exists():
                    destination.rename(backup)
                result.rename(destination)
            except BaseException:
                if backup.exists() and not destination.exists():
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
    finally:
        # Keep the old edit accessible if publishing or rollback was interrupted.
        if not any(lock.iterdir()):
            lock.rmdir()
    return destination, edit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=SITES)
    parser.add_argument("--site-dir", type=Path, help="site.json と元データがあるディレクトリ")
    parser.add_argument("--blend", type=Path, help="開く .blend ファイル。既定は site.blend")
    parser.add_argument("--edits-root", type=Path, default=EDITS_ROOT, help="書き戻し先の親ディレクトリ")
    parser.add_argument("--blender", type=Path, default=config.BLENDER_EXE)
    args = parser.parse_args(argv)
    source = args.site_dir or SITES_ROOT / args.site
    try:
        target, edit = import_site(args.site, source, args.blend or source / "site.blend",
                                   args.edits_root, args.blender)
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"書き戻し失敗　{exc}\n")
    print(f"{target} （地形変更 {edit['terrain_changed']}、部品変更 {len(edit['changed_parts'])} 件）")


if __name__ == "__main__":
    main()
