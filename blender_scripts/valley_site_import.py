"""Run with Blender in background, after opening the site blend."""

from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from valley_site_io import capture


if __name__ == "__main__":
    output, site = sys.argv[sys.argv.index("--") + 1:]
    target = Path(output)
    capture(bpy.context.scene, target, texture_dir=target.parent, site=site)
