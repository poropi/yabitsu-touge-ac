#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
B="/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
[ -x "$B" ] || B="$(command -v blender)"   # Mac など
M=data/processed/track/manifest.json
"$B" --background --factory-startup --python blender_scripts/build_from_manifest.py -- --manifest "$M" --group yabitsu_main --blend blender/track.blend
"$B" --background blender/track.blend --python blender_scripts/add_underlay.py -- data/processed/track_underlay blender/track_review.blend
