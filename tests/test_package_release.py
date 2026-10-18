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
