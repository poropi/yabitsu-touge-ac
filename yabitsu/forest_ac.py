"""森を AC（CSP の木の機能）用のファイルに書き出す：木の一覧 txt（1 km 四方ごと）と extension/ext_config.ini。"""
from pathlib import Path
import shutil

import numpy as np

from . import csptree, ext_config
from .ext_config import MARK  # noqa: F401（ほかのモジュールが forest_ac.MARK を使う）


def write_lists(trees, ext_dir: Path, tile_m=1000.0, *, base_lists=None) -> list:
    """タイル一覧を作り、ext_config 用のパスを返す。既存タイルを渡したときは末尾に追記する。"""
    ext_dir = Path(ext_dir)
    folder = ext_dir / "trees"
    folder.mkdir(parents=True, exist_ok=True)
    base_lists = {} if base_lists is None else base_lists
    ac = csptree.to_ac(np.c_[trees.x, trees.y, trees.z])
    i = np.floor(trees.x / tile_m).astype(int); j = np.floor(trees.y / tile_m).astype(int)
    written = set()
    for key in sorted(set(zip(i.tolist(), j.tolist()))):
        sel = np.flatnonzero((i == key[0]) & (j == key[1]))
        name = f"forest_{key[0]}_{key[1]}.txt"
        lines = [csptree.tree_line(f"{trees.pattern[k]}.zip", *ac[k], trees.angle_deg[k], trees.scale[k]) for k in sel]
        if name in base_lists:
            base = base_lists[name].read_bytes()
            if not base.endswith(b"\n"):
                raise ValueError(f"{name}: 末尾改行がなく元の行を保って追加できません")
            newline = "\r\n" if b"\r\n" in base else "\n"
            (folder / name).write_bytes(base + (newline.join(lines) + newline).encode("utf-8"))
        else:
            # ばらつきは各行の angle/size で指定済み。CSP が横幅や色を変えないようにする。
            header = ["configure: angle variance = 0, 0", "configure: size variance = 1, 1",
                      "configure: width variance = 1, 1", "configure: color variance = 0, 0, 0"]
            (folder / name).write_text("\n".join(header + lines) + "\n", encoding="utf-8", newline="\n")
        written.add(name)
    for name in base_lists.keys() - written:
        shutil.copyfile(base_lists[name], folder / name)
    return [f"trees\\{name}" for name in sorted(written | base_lists.keys(),
                                                 key=lambda n: tuple(map(int, Path(n).stem.split("_")[1:])))]


def _check_ours(p: Path):
    if p.exists() and not p.read_text(encoding="utf-8").startswith(MARK):
        raise RuntimeError(f"{p} はこのスクリプトが書いたものではないので、上書きしません")


def write_ext_config(track_dir: Path, lists, hidden_meshes):
    """森の部品（extension/parts/forest.ini）を書き、ext_config.ini を組み立て直す（S3 の仕上げなど他の部品は残る）。"""
    _check_ours(Path(track_dir) / "extension" / "ext_config.ini")
    s = ["[TREES]", "ACTIVE=1"] + [f"LIST_{k}={l}" for k, l in enumerate(lists)]
    if hidden_meshes:   # 古い板の木（kn5 の中）を隠す
        s += ["", "[MESH_ADJUSTMENT_...]", "MESHES=" + ",".join(hidden_meshes), "IS_ACTIVE=0", "IS_RENDERABLE=0"]
    ext_config.write_part(track_dir, "forest", "\n".join(s))
    ext_config.assemble(track_dir)


def write_layout_off(track_dir: Path, layouts):
    """森を入れないレイアウト（Codex の S1 景観確認）。"""
    for lay in layouts:
        p = Path(track_dir) / lay / "extension" / "ext_config.ini"
        _check_ours(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{MARK}\n\n[TREES]\nACTIVE=0\n", encoding="utf-8", newline="\n")
