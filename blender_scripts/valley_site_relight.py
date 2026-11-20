"""背景モードで既存の谷シーンの照明・画像パスを更新し、別名で保存する。"""

import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from valley_site_io import capture
from valley_site_light import lighting, localize_images, material_coefficients


def main():
    out, light, *baseline = sys.argv[sys.argv.index("--") + 1:]
    settings = json.loads(light)
    scene = bpy.context.scene
    if not scene.get("valley_site") or not scene.objects.get("Game Sun") or scene.world is None:
        raise ValueError("切り出した谷の照明が見つかりません")
    lighting(scene, None, settings)
    material_coefficients(settings)
    localize_images()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.ops.wm.save_as_mainfile(filepath=out, copy=True, relative_remap=False)
    if baseline:
        capture(scene, Path(baseline[0]))


if __name__ == "__main__":
    main()
