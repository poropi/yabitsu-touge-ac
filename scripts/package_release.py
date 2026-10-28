"""配布用の zip（Assetto Corsa の MOD）を export/release/ に作る。
入っているコース（config.TRACK_DIR）をそのまま content/tracks/yabitsu/ の形で入れ、どのレイアウトの ui_track.json にも
version と author を書く（入っているコースのファイルは変えない）。zip のいちばん上に説明書 README.txt。
使い方: uv run python -m scripts.package_release 0.0.3 [コースのフォルダ]"""
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
  ・前の版から更新するときは、content/tracks/yabitsu を消してから入れると、古いファイルが残りません。
  ・コース選択で「ヤビツ峠」を選び、レイアウトを選んでください。

■ 必要なもの・推奨環境
  ・Custom Shaders Patch（CSP）が必要です。森・路面や擁壁の模様・夜の明かり・雨・環境音は CSP で表示します。
    動作を確かめた版：CSP 0.3.0-preview619。これより古い版では確かめていません。
  ・天気：Pure 2.57 で確かめました（なくても走れます）。
  ・Content Manager を使うと入れるのが簡単です。
  ・確かめた PC：Core i7-14700K / GeForce RTX 5070 / メモリ 64GB / Windows 11。
    5120×1440 の全画面で、全線を走って平均 約 64 FPS・最低 約 48 FPS（町の区間がいちばん重い）。
    これより弱い PC では、解像度・影・ワールドの細かさを下げてください。
  ・ディスク：展開後 約 2.2GB。

■ 確かめたときの AC の設定
  ・画面：5120×1440 全画面、垂直同期 なし、アンチエイリアス 4x、異方性フィルタ 16x
  ・影 2048、ワールドの細かさ 3、煙 3、モーションブラー なし
  ・CSP：Extra FX・Rain FX・Reflections FX・Particles FX・Windscreen FX を入れた状態
  ・夜は CSP の明かり（家の窓・街灯）、雨は Rain FX（水たまりは出しません）で表示します。

■ レイアウト（12）
  S1 名古木→蓑毛 / S2 蓑毛→菜の花台 / S3 菜の花台→ヤビツ峠 / S4 ヤビツ峠→札掛 / S5 札掛→宮ヶ瀬北原 の上り・下り、
  全線（名古木→宮ヶ瀬北原）の上り・下り。
  どれも A 地点から B 地点まで走るコース（point to point）です。

■ v0.0.2 からの変更
  ・コース全体の家・店・工場・物置を、日本の家の部品（窓・ベランダ・軒・雨樋・室外機など）で建て直し
  ・名古木のコンビニの店構えと看板、規格の大きさの電柱（腕木・変圧器・電線）、町の石積みの擁壁
  ・ヤビツ峠のトイレと駐車場、寺山富士見橋・護摩屋敷の水・門戸口橋・きまぐれ喫茶・青山荘・塩水橋・トンネルをストリートビューに合わせて作り込み
  ・山道の路面標示（30/40・急カーブ・ダイヤ）、すべり止めの赤・緑の舗装、S2〜S5 の電柱
  ・遠くの地形に空中写真、地平線（菜の花台から秦野の町・丘・相模湾が見える）
  ・夜（家の窓・街灯の明かり）・雨・環境音（蝉・鳥と小川）・季節
  ・S5 で木やシダにさえぎられて先が見えにくかった所を整理

■ わかっていること（v{version}）
  ・家の形・色、電線、庭の地面などは、まだストリートビューと違う所があります。
  ・菜の花台・ヤビツ峠の駐車の車は、作った PC の AC の車を使っているため、この zip には入っていません（車は表示されません）。
  ・分かれ道は工事中の柵でふさいであります（柵の向こうは走れません）。

■ 出典
  ・国土地理院：標高タイル・シームレス空中写真・基盤地図情報（ベクトルタイル）を加工して作成
  ・© OpenStreetMap contributors（ODbL https://opendatacommons.org/licenses/odbl/）
  ・建物の輪郭の一部：Overture Maps Foundation（ODbL）
  ・森の木の写真素材：Poly Haven（CC0）、壁・屋根の写真素材：ambientCG（CC0）
  ・環境音：Freesound の cartoonrob（日本の蝉）、tomasloos（日本の鳥と小川）の録音（CC0）を加工
  ・建物・標識などの見た目は Google ストリートビューを見て確かめました（画像は入っていません）。
  Google・国土地理院・OpenStreetMap・Overture Maps・Kunos Simulazioni とは関係ありません。

■ 作った人
  {author}  https://github.com/poropi/yabitsu-touge-ac

----------------------------------------------------------------
Yabitsu Touge for Assetto Corsa  v{version}
Route 70 (Hadano-Kiyokawa line), Kanagawa, Japan: Nagoki - Yabitsu Pass - Miyagase-Kitahara, about 29.9 km,
built from public map and elevation data. 12 point-to-point layouts.
Requires Custom Shaders Patch (tested with CSP 0.3.0-preview619; Pure 2.57 weather tested, optional).
Tested PC: i7-14700K / RTX 5070 / 64 GB, 5120x1440 fullscreen: about 64 FPS average, 48 FPS minimum over the full route.
Install: drop the zip into Content Manager, or copy the "content" folder into your assettocorsa folder
(delete content/tracks/yabitsu first when updating). About 2.2 GB unpacked.
Credits: GSI Japan (elevation tiles, seamless aerial photos, basic map vector tiles), (c) OpenStreetMap contributors (ODbL),
Overture Maps Foundation buildings (ODbL), Poly Haven and ambientCG textures (CC0), Freesound recordings by cartoonrob and tomasloos (CC0).
Not affiliated with Google, GSI, OpenStreetMap, Overture Maps or Kunos Simulazioni.
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
    src = sys.argv[2] if len(sys.argv) > 2 else config.TRACK_DIR   # Mac で作るときは、ゲーム PC から写したコースのフォルダ
    p = package(src, config.PROJECT_ROOT / "export" / "release", version)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"{p}  {p.stat().st_size / 1e6:.0f} MB  sha256 {h}")


if __name__ == "__main__":
    main()
