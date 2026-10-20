import re
from pathlib import Path

BLENDER_SCRIPTS = Path(__file__).resolve().parents[1] / "blender_scripts"
SAVE = re.compile(r"bpy\.ops\.wm\.save(_as)?_mainfile\(")
NO_THUMB = 'bpy.context.preferences.filepaths.file_preview_type = "NONE"'


def test_every_blend_save_disables_thumbnail():
    # Codexのサンドボックスなどホームを取得できない環境では、Blenderが.blend保存時の
    # サムネイル用に文字化けしたフォルダを作業フォルダへ作る。保存の直前で必ず止める。
    missing = []
    for path in sorted(BLENDER_SCRIPTS.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if SAVE.search(line) and not any(NO_THUMB in prev for prev in lines[max(0, i - 3):i]):
                missing.append(f"{path.relative_to(BLENDER_SCRIPTS.parent)}:{i + 1}")
    assert not missing, missing
