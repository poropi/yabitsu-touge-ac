import json
import zipfile

from scripts import package_release as pr


def _fake_track(root):
    (root / "ui" / "s1_up").mkdir(parents=True)
    (root / "ui" / "s1_up" / "ui_track.json").write_text(json.dumps({"name": "ヤビツ峠 S1", "tags": ["touge"]}, ensure_ascii=False), encoding="utf-8")
    (root / "yabitsu_main.kn5").write_bytes(b"kn5")
    (root / "models_s1_up.ini").write_text("[MODEL_0]\nFILE=yabitsu_main.kn5\n", encoding="utf-8")


def test_release_zip_has_the_ac_folder_layout_and_versioned_ui(tmp_path):
    """MOD の zip：content/tracks/yabitsu/ の形で、どのレイアウトの ui_track.json にも version と author。説明書は zip のいちばん上"""
    track = tmp_path / "yabitsu"; _fake_track(track)
    z = pr.package(track, tmp_path / "out", "0.0.1", author="poropi")
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        ui = json.loads(f.read("content/tracks/yabitsu/ui/s1_up/ui_track.json").decode("utf-8"))
    assert "content/tracks/yabitsu/yabitsu_main.kn5" in names and "content/tracks/yabitsu/models_s1_up.ini" in names
    assert ui["version"] == "0.0.1" and ui["author"] == "poropi" and ui["name"] == "ヤビツ峠 S1"
    assert any(n.startswith("README") and "/" not in n for n in names)
    assert z.name == "yabitsu-touge-ac-v0.0.1.zip"
    assert json.loads((track / "ui" / "s1_up" / "ui_track.json").read_text(encoding="utf-8")).get("version") is None   # 入っているコースは変えない


def test_release_excludes_codex_test(tmp_path):
    from scripts import package_release
    track = tmp_path / "yabitsu"
    for rel in ("yabitsu_main.kn5", "yabitsu_s1_landscape_test.kn5", "models_s1_landscape_up.ini",
                "s1_landscape_up/data/map.ini", "ui/s1_landscape_up/ui_track.json", "models_s1_up.ini"):
        p = track / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    import zipfile
    z = zipfile.ZipFile(package_release.package(track, tmp_path / "out", "9.9.9"))
    names = z.namelist()
    assert not [n for n in names if "s1_landscape" in n]
    assert "content/tracks/yabitsu/yabitsu_main.kn5" in names and "content/tracks/yabitsu/models_s1_up.ini" in names


def test_release_excludes_copied_parked_cars(tmp_path):
    """菜の花台の駐車車両は、組み込んだ PC の AC の車のコピー（extension/parked_cars）なので配布しない。Lua は配布する。"""
    from scripts import package_release
    import zipfile
    track = tmp_path / "yabitsu"
    for rel in ("yabitsu_main.kn5", "extension/parked_cars.lua", "extension/parked_cars/ks_toyota_gt86/toyota_gt86.kn5"):
        p = track / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    names = zipfile.ZipFile(package_release.package(track, tmp_path / "out", "9.9.9")).namelist()
    assert "content/tracks/yabitsu/extension/parked_cars.lua" in names
    assert not [n for n in names if "parked_cars/" in n]
