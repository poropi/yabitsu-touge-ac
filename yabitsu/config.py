import os
import sys
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
if sys.platform == "win32":
    BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
else:   # Mac で描画だけ確かめるとき（AC はない）
    BLENDER_EXE = Path("/opt/homebrew/bin/blender")

# 絵に文字を描くフォント。Windows は今までのファイル、Mac は近い書体で代える（同じ絵にはならない）
_WIN_FONTS = "C:/Windows/Fonts/"
_MAC_FONTS = "/System/Library/Fonts/"
FONTS = {
    "gothic": (_WIN_FONTS + "meiryo.ttc", _MAC_FONTS + "ヒラギノ角ゴシック W3.ttc"),
    "gothic_bold": (_WIN_FONTS + "BIZ-UDGothicB.ttc", _WIN_FONTS + "meiryob.ttc", _WIN_FONTS + "YuGothB.ttc",
                    _MAC_FONTS + "ヒラギノ角ゴシック W6.ttc"),
    "mincho": (_WIN_FONTS + "msmincho.ttc", _MAC_FONTS + "ヒラギノ明朝 ProN.ttc"),
    "arial_bold": (_WIN_FONTS + "arialbd.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
}


def font_file(kind) -> str:
    """kind のフォントのうち、このマシンにある最初のファイル。"""
    for f in FONTS[kind]:
        if Path(f).exists():
            return f
    raise FileNotFoundError(f"フォント {kind} が見つかりません: {FONTS[kind]}")

NAGOKI = (35.3796436, 139.2358251)  # 名古木交差点の中心（OSM node 293317044）。S3 の道の始まり
TOWN_RADIUS_M = 200.0               # 交差点から、この半径の中を地図どおりの道と建物で作る（町の範囲）
S2_START = (35.4150771, 139.2289678)  # 道の始まり：蓑毛バス停（OSM node 1312645669）から名古木側へ道に沿って 約95m 下
S1_START = (35.41025, 139.20926)  # 菜の花台休憩所（駐車場）
S1_PASS = (35.42782, 139.21934)   # ヤビツ峠（OSM の峠ノード）。上りのゴールと下りの計測開始はここを基準にする
S1_END = (35.4282221, 139.2190026)  # 道の終わり：峠の先へ約55m（下りのスタート列が並ぶ場所）



@dataclass(frozen=True)
class SectionDef:
    """区間：OSM の県道70号を start から end までたどる。並びは道の下（秦野側）から上。"""
    name: str
    start: tuple
    end: tuple


S4_END = (35.4651366, 139.2137214)  # S4 の終わり：札掛橋（OSM way 131561915）の北の端から、県道70号に沿って 250m

KITAHARA = (35.5187747, 139.2309644)  # 宮ヶ瀬北原交差点の中心（OSM の信号 node 293598433）。S5 の終わり、県道70号の終わり

SECTIONS = (SectionDef("s3", NAGOKI, S2_START), SectionDef("s2", S2_START, S1_START), SectionDef("s1", S1_START, S1_END),
            SectionDef("s4", S1_END, S4_END), SectionDef("s5", S4_END, KITAHARA))
ACCEPTED_SECTIONS = ("s3", "s2", "s1")   # 受け入れ済みの区間：高さの合わせ直しで、今の値をそのまま使う（elevation.fit_sections の fixed）
TRACK_GROUP = "yabitsu_main"
TOWN_GROUP = "yabitsu_nagoki"  # 町の範囲の部品は別の kn5 にする
S4_GROUP = "yabitsu_s4"  # S4（峠 → 札掛）の部品。全部のレイアウトが読む
S5_GROUP = "yabitsu_s5"  # S5（札掛 → 宮ヶ瀬北原）の部品
KITAHARA_GROUP = "yabitsu_kitahara"  # 宮ヶ瀬北原交差点の町の範囲
TRACK_BUILD_DIR = DATA_PROCESSED / "track"

PROTECTED_DIRS = (AC_TRACKS / "nanohanadai",)


class ProtectedPathError(RuntimeError):
    pass


def _norm(p) -> str:
    return os.path.abspath(p).casefold()   # AC のフォルダ名は大文字小文字を区別しない（Windows・Mac とも）


def assert_writable(path) -> Path:
    p = _norm(path)
    for d in PROTECTED_DIRS:
        dn = _norm(d)
        if p == dn or p.startswith(dn + os.sep):
            raise ProtectedPathError(f"{path} は変更禁止のフォルダ（{d}）の中です")
    return Path(path)
DRESS_KN5 = ("yabitsu_s3_dress.kn5", "yabitsu_mountain_dress.kn5")   # 道路まわりの仕上げ（scripts.dress_build・scripts.dress_mountain が書く。全部のレイアウトが読む）
