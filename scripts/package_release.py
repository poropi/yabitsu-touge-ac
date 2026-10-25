"""配布用の zip（Assetto Corsa の MOD）を export/release/ に作る。
入っているコース（config.TRACK_DIR）をそのまま content/tracks/yabitsu/ の形で入れ、どのレイアウトの ui_track.json にも
version と author を書く（入っているコースのファイルは変えない）。zip のいちばん上に説明書 README.txt。
使い方: uv run python -m scripts.package_release 0.0.1"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

from yabitsu import config

AUTHOR = "poropi"
TRACK_NAME = "yabitsu"
EXCLUDE = ("s1_landscape",    # Codex の S1 景観確認（試作。1.6 GB）は配布しない
           "extension/parked_cars/")   # 菜の花台の駐車車両：組み込んだ PC の AC の車のコピー（他人の車のデータなので配布しない）
README = """ヤビツ峠（Yabitsu Touge）for Assetto Corsa  v{version}

神奈川県の県道70号（秦野清川線）、名古木からヤビツ峠を越えて宮ヶ瀬北原までの約 29.9km を、
公開されている地図・標高データをもとに再現したコースです。

■ 入れ方
  ・Content Manager：この zip をそのまま Content Manager の画面に落とすと入ります。
  ・手で入れる：zip の中の content フォルダを、Assetto Corsa のフォルダ
    （例：C:\\Program Files (x86)\\Steam\\steamapps\\common\\assettocorsa）に上書きでコピーします。
  ・コース選択で「ヤビツ峠」を選び、レイアウトを選んでください。

■ レイアウト（12）
  S1 名古木→蓑毛 / S2 蓑毛→菜の花台 / S3 菜の花台→ヤビツ峠 / S4 ヤビツ峠→札掛 / S5 札掛→宮ヶ瀬北原 の上り・下り、
  全線（名古木→宮ヶ瀬北原）の上り・下り。
  どれも A 地点から B 地点まで走るコース（point to point）です。

■ わかっていること（v{version}）
  ・見た目（路面・ガードレール・擁壁・土手・落ち葉・森）はストリートビューに合わせて作り込み中です。
    Custom Shaders Patch（CSP）があると、森と路面・擁壁の模様が表示されます。
  ・S5 の宮ヶ瀬湖のまわりで、湖の縁が少し浮いて見える所、吹風トンネルの横の湖の一部が陸になっている所があります。
  ・分かれ道は工事中の柵でふさいであります（柵の向こうは走れません）。

■ 出典
  ・国土地理院：標高タイル・シームレス空中写真・基盤地図情報（ベクトルタイル）を加工して作成
  ・© OpenStreetMap contributors（ODbL https://opendatacommons.org/licenses/odbl/）
  ・建物・標識などの見た目は Google ストリートビューを見て確かめました（画像は入っていません）。
  Google・国土地理院・OpenStreetMap・Kunos Simulazioni とは関係ありません。

■ 作った人
  {author}  https://github.com/poropi/yabitsu-touge-ac

----------------------------------------------------------------
Yabitsu Touge for Assetto Corsa  v{version}
Route 70 (Hadano-Kiyokawa line), Kanagawa, Japan: Nagoki - Yabitsu Pass - Miyagase-Kitahara, about 29.9 km,
built from public map and elevation data. 12 point-to-point layouts. Custom Shaders Patch recommended.
Install: drop the zip into Content Manager, or copy the "content" folder into your assettocorsa folder.
Credits: GSI Japan (elevation tiles, seamless aerial photos, basic map vector tiles), (c) OpenStreetMap contributors (ODbL).
Not affiliated with Google, GSI, OpenStreetMap or Kunos Simulazioni.
"""


def package(track_dir, out_dir, version, author=AUTHOR):
    track_dir, out_dir = Path(track_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"yabitsu-touge-ac-v{version}.zip"
    base = f"content/tracks/{TRACK_NAME}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.txt", README.format(version=version, author=author).replace("\n", "\r\n").encode("utf-8-sig"))
        for p in sorted(track_dir.rglob("*")):
            if not p.is_file() or any(k in p.relative_to(track_dir).as_posix() for k in EXCLUDE):
                continue
            arc = f"{base}/{p.relative_to(track_dir).as_posix()}"
            if p.name == "ui_track.json":   # 入っているコースは変えず、zip の中だけ version と author を書く
                d = json.loads(p.read_text(encoding="utf-8"))
                d["version"] = version; d["author"] = author
                z.writestr(arc, json.dumps(d, ensure_ascii=False, indent=2).encode("utf-8"))
            else:
                z.write(p, arc)
    return path


def main():
    version = sys.argv[1]
    p = package(config.TRACK_DIR, config.PROJECT_ROOT / "export" / "release", version)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"{p}  {p.stat().st_size / 1e6:.0f} MB  sha256 {h}")


if __name__ == "__main__":
    main()
