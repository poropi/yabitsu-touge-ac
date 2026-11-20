"""既存の谷シーンの照明を更新する。切り出し直さず、元シーンの検査用控えも更新する。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yabitsu import config, valley_site_source
from yabitsu.valley_site_light import SITE_LIGHT

BLENDER_SCRIPT = ROOT / "blender_scripts/valley_site_relight.py"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def relight(site, blender, settings=SITE_LIGHT):
    blend = site / "site.blend"
    original = site / "site_original.blend"
    metadata = site / "site.json"
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    source = original if original.is_file() else blend
    source_sha = digest(source)
    if source_sha != meta["baseline_blend_sha256"]:
        raise ValueError(f"{source}: 切り出し直後のシーンと指紋が一致しません")
    installed = digest(blend)
    with TemporaryDirectory(prefix=".relight-", dir=site) as temporary:
        stage = Path(temporary)

        def update(before, after, *extra):
            subprocess.run([
                str(blender), "-b", "--factory-startup", "--disable-autoexec", str(before),
                "--python-exit-code", "1", "--python", str(BLENDER_SCRIPT), "--",
                str(after), json.dumps(settings), *(str(path) for path in extra),
            ], check=True)

        update(source, stage / original.name, stage / "baseline_scene.npz")
        if installed == source_sha:
            shutil.copyfile(stage / original.name, stage / blend.name)
        else:
            update(blend, stage / blend.name)
        # Publish only once both scene copies and the new baseline are ready.
        if digest(blend) != installed or digest(source) != source_sha:
            raise ValueError("処理中にシーンが保存されました。再実行してください")
        meta["baseline_blend_sha256"] = digest(stage / original.name)
        meta["light"] = settings
        (stage / metadata.name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        names = (original.name, "baseline_scene.npz", metadata.name, blend.name)
        previous = stage / "previous"
        previous.mkdir()
        for name in names:
            target = site / name
            if target.exists():
                (previous / name).hardlink_to(target)
        installed_names = []
        try:
            for name in names:
                (stage / name).replace(site / name)
                installed_names.append(name)
        except BaseException:
            for name in reversed(installed_names):
                saved = previous / name
                if saved.exists():
                    saved.replace(site / name)
                else:
                    (site / name).unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", choices=valley_site_source.SITES)
    parser.add_argument("--sites-dir", type=Path, default=ROOT / "export/blender_sites")
    parser.add_argument("--blender", type=Path, default=config.BLENDER_EXE)
    args = parser.parse_args()
    try:
        directory = (args.sites_dir / args.site).resolve()
        relight(directory, args.blender)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"谷の照明を更新できません。{error}\n")
    print(directory / "site.blend")


if __name__ == "__main__":
    main()
