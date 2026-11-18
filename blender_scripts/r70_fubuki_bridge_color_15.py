"""依頼10の橋アトラスから、朱赤の鋼材だけを更新する。通常のPythonで実行。"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from route70_common import KEYS


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'assets/landmarks/r70_fubuki/red_arch_bridge_v2.png'
OLD_FILE_SHA = '6c7d9ded246bdd9e761614ddc06d2e851c3a60c9667065fd975697ef2877da89'
NEW_PIXEL_SHA = '781d26d925ae7968cc7e450b1c4d87c246c6d1ae5455f3896d0f8fc5c9afa1ac'
# 依頼10の実画素中央値からの差分。鋼材表面の明暗を保つ。
DELTAS = {'red': (-66, -35, -44), 'rust': (-55, -27, -31)}


def update_texture(path):
    with Image.open(path) as original:
        pixels = np.array(original)
    if hashlib.sha256(pixels.tobytes()).hexdigest() == NEW_PIXEL_SHA:
        return
    if hashlib.sha256(path.read_bytes()).hexdigest() != OLD_FILE_SHA:
        raise ValueError(f'更新元のアトラスが依頼10の版と異なります: {path}')
    for name, delta in DELTAS.items():
        tile = KEYS.index(name)
        x = 800 + (tile % 2) * 112
        y = (tile // 2) * 112
        patch = pixels[y:y+108, x:x+108, :3]
        patch[:] = np.clip(patch.astype(np.int16) + delta, 0, 255)
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == NEW_PIXEL_SHA
    Image.fromarray(pixels).save(path, optimize=True)


if __name__ == '__main__':
    update_texture(ATLAS)
