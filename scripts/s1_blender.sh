#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
B="/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
M=data/processed/s1/manifest.json
FBX_ARGS=()
[ "${1:-}" = "fbx" ] && FBX_ARGS=(--fbx export/s1/yabitsu_s1.fbx)
"$B" --background --factory-startup --python blender_scripts/build_from_manifest.py -- --manifest "$M" --group yabitsu_s1 --blend blender/s1_track.blend "${FBX_ARGS[@]}"
"$B" --background blender/s1_track.blend --python blender_scripts/add_underlay.py -- data/processed/s1_underlay blender/s1_review.blend
if [ "${1:-}" = "fbx" ]; then
  for g in s1_up_objects s1_down_objects; do
    if grep -q "\"$g\"" "$M"; then
      "$B" --background --factory-startup --python blender_scripts/build_from_manifest.py -- --manifest "$M" --group "$g" --fbx "export/s1/$g.fbx"
    fi
  done
fi
