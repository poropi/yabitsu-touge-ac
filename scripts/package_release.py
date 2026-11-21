"""配布用の zip（Assetto Corsa の MOD）を export/release/ に作る。
入っているコース（config.TRACK_DIR）をそのまま content/tracks/yabitsu/ の形で入れ、どのレイアウトの ui_track.json にも
version と author を書く（入っているコースのファイルは変えない）。zip のいちばん上に説明書 README.txt。
使い方: uv run python -m scripts.package_release 0.0.4 [コースのフォルダ]"""
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
    5120×1440 の全画面・DLSS なしで、全線の決まった 17 か所で止めて測って、平均 約 67 FPS・最低 約 50 FPS。
    DLSS Balanced（0.58）では、最初の 6km を走って平均 約 76 FPS・最低 約 61 FPS。
    全線を走り通しての測定はまだしていません。
    これより弱い PC では、解像度・影・ワールドの細かさを下げてください。
  ・ディスク：展開後 約 4.4GB。

■ 確かめたときの AC の設定
  ・画面：5120×1440 全画面、垂直同期 なし、アンチエイリアス 4x、異方性フィルタ 16x
  ・影 2048、ワールドの細かさ 3、煙 3、モーションブラー なし
  ・CSP：Extra FX・Rain FX・Reflections FX・Particles FX・Windscreen FX を入れた状態
  ・夜は CSP の明かり（家の窓・街灯）、雨は Rain FX（水たまりは出しません）で表示します。

■ レイアウト（12）
  S1 名古木→蓑毛 / S2 蓑毛→菜の花台 / S3 菜の花台→ヤビツ峠 / S4 ヤビツ峠→札掛 / S5 札掛→宮ヶ瀬北原 の上り・下り、
  全線（名古木→宮ヶ瀬北原）の上り・下り。
  どれも A 地点から B 地点まで走るコース（point to point）です。

■ v0.0.3 からの変更
  ・浅間神社の駐車場（S2）を、空中写真とストリートビューから作り直し
  ・ヤビツ峠の売店・トイレ・まわりの物、菜の花台の駐車場と展望台を作り込み
  ・県道70号沿いの物（カーブミラー・電柱・km の標・標識・ガードレール・擁壁・側溝・ゲートなど）を、ストリートビューを見て作った部品に置き換え
  ・トンネルの入口と、吹風トンネル入口のまわり（擁壁・草の斜面・れんがの門・赤いアーチ橋・柵・低木）
  ・トンネルの中で路面が消えることがあった不具合を直した
  ・小さな橋と谷（S4 1724m・S4 6354m・S5 8956m）：沢の谷を掘り直し、橋を作り込み
  ・道ぎわの斜面を草と下草（全線で約 11 万株）で覆い、むき出しの土をなくした
  ・山の高い木（約 100 万本）を、道ぎわは立体の木・奥は絵を貼った板の軽い作りに置き換え（DLSS なしで 約 40 → 約 67 FPS）
  ・山の路面を写真に合わせて暗めの灰色に、遠くの山を夏の森の色に

■ わかっていること（v{version}）
  ・S2 324m 付近の丸い模様の舗装が白いまま
  ・S5 7899m 付近で、電線が運転席の高さを通る
  ・S5 10149m 付近の欄干の陰影がおかしい
  ・集落との境（S1 3159m・S4 73m 付近）の斜面が黄緑
  ・道のそばの奥の森の板が平たく見える所がある（S4 5573m・S5 4399m 付近）
  ・日なたの草が少し蛍光色ぎみ
  ・家の形・色、電線、庭の地面などは、まだストリートビューと違う所があります。
  ・菜の花台・ヤビツ峠の駐車の車は、作った PC の AC の車を使っているため、この zip には入っていません（車は表示されません）。
  ・分かれ道は工事中の柵でふさいであります（柵の向こうは走れません）。

■ 出典
  ・国土地理院：標高タイル・シームレス空中写真・基盤地図情報（ベクトルタイル）を加工して作成
  ・© OpenStreetMap contributors（ODbL https://opendatacommons.org/licenses/odbl/）
  ・建物の輪郭の一部：Overture Maps Foundation（ODbL）
  ・森の木の写真素材：Poly Haven（CC0）、壁・屋根の写真素材：ambientCG（CC0）
  ・環境音：Freesound の cartoonrob（日本の蝉）、tomasloos（日本の鳥と小川）の録音（CC0）を加工
  ・菜の花台の見物客：Microsoft Rocketbox Avatar Library（MIT、Copyright (c) 2020 Microsoft。LICENSE-Rocketbox.txt）
  ・建物・標識などの見た目は Google ストリートビューを見て確かめました（画像は入っていません）。
  Google・国土地理院・OpenStreetMap・Overture Maps・Kunos Simulazioni とは関係ありません。

■ ライセンス（自由に使ってください）
  このコースとデータは、自由に使ってかまいません。走る・改造する・別のコースや MOD に組み込む・再配布する・
  研究や学習に使う、どれも作者への連絡や許可は要りません。
  ただし、作者（{author}）は著作権とライセンスを放棄していません（パブリックドメインや CC0 ではありません）。
  CC BY-NC 4.0（クリエイティブ・コモンズ 表示-非営利 4.0 国際）で使えるようにしています。
  https://creativecommons.org/licenses/by-nc/4.0/deed.ja
  使うときは、次のことだけ守ってください。
    ・作者（{author}）とリポジトリの URL を表示する
    ・販売しない・有料で配布しない
    ・上の出典の表示を、改造版・再配布でも残す（地図データは ODbL、見物客の人物は MIT に従います）

■ 作った人
  {author}  https://github.com/poropi/yabitsu-touge-ac

----------------------------------------------------------------
Yabitsu Touge for Assetto Corsa  v{version}
Route 70 (Hadano-Kiyokawa line), Kanagawa, Japan: Nagoki - Yabitsu Pass - Miyagase-Kitahara, about 29.9 km,
built from public map and elevation data. 12 point-to-point layouts.
Requires Custom Shaders Patch (tested with CSP 0.3.0-preview619; Pure 2.57 weather tested, optional).
Tested PC: i7-14700K / RTX 5070 / 64 GB, 5120x1440 fullscreen without DLSS: about 67 FPS average, 50 FPS minimum
(17 fixed spots along the route). With DLSS Balanced: about 76 FPS average, 61 FPS minimum over the first 6 km.
Install: drop the zip into Content Manager, or copy the "content" folder into your assettocorsa folder
(delete content/tracks/yabitsu first when updating). About 4.4 GB unpacked.
Credits: GSI Japan (elevation tiles, seamless aerial photos, basic map vector tiles), (c) OpenStreetMap contributors (ODbL),
Overture Maps Foundation buildings (ODbL), Poly Haven and ambientCG textures (CC0), Freesound recordings by cartoonrob and tomasloos (CC0).
Spectators at Nanohanadai: Microsoft Rocketbox Avatar Library (MIT, Copyright (c) 2020 Microsoft, see LICENSE-Rocketbox.txt).
Not affiliated with Google, GSI, OpenStreetMap, Overture Maps or Kunos Simulazioni.
License: feel free to use this course and its data - drive it, modify it, build it into your own tracks or mods, share it,
use it for research - no need to ask. The author ({author}) does NOT waive copyright or the license (this is not public domain
or CC0): it is licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Credit {author} and this
repository, do not sell it or distribute it for a fee, and keep the credits above (map data ODbL, spectators MIT).
"""
ROCKETBOX_LICENSE = config.PROJECT_ROOT / "assets" / "free" / "rocketbox" / "LICENSE.md"   # 見物客の人物（MIT）：表示を同梱する


def package(track_dir, out_dir, version, author=AUTHOR):
    track_dir, out_dir = Path(track_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"yabitsu-touge-ac-v{version}.zip"
    base = f"content/tracks/{TRACK_NAME}"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.txt", README.format(version=version, author=author).replace("\n", "\r\n").encode("utf-8-sig"))
        z.writestr("LICENSE-Rocketbox.txt", ROCKETBOX_LICENSE.read_text(encoding="utf-8").replace("\n", "\r\n"))
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
