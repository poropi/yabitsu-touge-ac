import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MANUAL = PROJECT_ROOT / "data" / "manual"

AC_ROOT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
AC_TRACKS = AC_ROOT / "content" / "tracks"
TRACK_DIR = AC_TRACKS / "yabitsu"
TEST_TRACK_DIR = AC_TRACKS / "yabitsu_test"
BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

S1_START = (35.41025, 139.20926)  # 菜の花台休憩所（駐車場）
S1_PASS = (35.42782, 139.21934)   # ヤビツ峠（OSM の峠ノード）。上りのゴールと下りの計測開始はここを基準にする
S1_END = (35.4282221, 139.2190026)  # 道の終わり：峠の先へ約55m（下りのスタート列が並ぶ場所）



@dataclass(frozen=True)
class SectionDef:
    """区間：OSM の県道70号を start から end までたどる。並びは道の下（秦野側）から上。"""
    name: str
    start: tuple
    end: tuple


SECTIONS = (SectionDef("s1", S1_START, S1_END),)
TRACK_GROUP = "yabitsu_main"
TRACK_BUILD_DIR = DATA_PROCESSED / "track"

PROTECTED_DIRS = (AC_TRACKS / "nanohanadai",)


class ProtectedPathError(RuntimeError):
    pass


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(p))


def assert_writable(path) -> Path:
    p = _norm(path)
    for d in PROTECTED_DIRS:
        dn = _norm(d)
        if p == dn or p.startswith(dn + os.sep):
            raise ProtectedPathError(f"{path} は変更禁止のフォルダ（{d}）の中です")
    return Path(path)
