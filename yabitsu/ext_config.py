"""content/tracks/yabitsu/extension/ext_config.ini を部品から組み立てる。
森（forest）・S3 の仕上げ（dress_s3）などが extension/parts/<name>.ini を書き、assemble で1つにする（互いに消さない）。"""
from pathlib import Path

MARK = "; yabitsu forest（scripts.forest_build が書いたファイル。手で直さない）"


def _check_ours(p: Path):
    if p.exists() and not p.read_text(encoding="utf-8").startswith(MARK):
        raise RuntimeError(f"{p} はこのスクリプトが書いたものではないので、上書きしません")


def _migrate(track_dir):
    """部品方式より前の ext_config.ini（森だけ）があれば、forest の部品にして残す。"""
    ext = Path(track_dir) / "extension"
    old = ext / "ext_config.ini"
    if old.exists() and not (ext / "parts").exists():
        _check_ours(old)
        body = old.read_text(encoding="utf-8")[len(MARK):].lstrip("\n")
        (ext / "parts").mkdir(parents=True)
        (ext / "parts" / "forest.ini").write_text(body, encoding="utf-8", newline="\n")


def write_part(track_dir, name: str, text: str):
    _migrate(track_dir)
    p = Path(track_dir) / "extension" / "parts" / f"{name}.ini"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")


def assemble(track_dir):
    ext = Path(track_dir) / "extension"
    out = ext / "ext_config.ini"
    _check_ours(out)
    parts = sorted((ext / "parts").glob("*.ini"))
    body = "\n".join(f"; ---- {p.stem} ----\n" + p.read_text(encoding="utf-8") for p in parts)
    out.write_text(f"{MARK}\n\n{body}", encoding="utf-8", newline="\n")
    return out
