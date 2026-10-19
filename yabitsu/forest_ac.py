"""森を AC（CSP の木の機能）用のファイルに書き出す：木の一覧 txt（1 km 四方ごと）と extension/ext_config.ini。"""
from pathlib import Path

import numpy as np

from . import csptree

MARK = "; yabitsu forest（scripts.forest_build が書いたファイル。手で直さない）"


def write_lists(trees, ext_dir: Path, tile_m=1000.0) -> list:
    """ext_dir/trees/forest_<i>_<j>.txt を書き、ext_config の LIST に書く相対パスを返す。"""
    ext_dir = Path(ext_dir)
    (ext_dir / "trees").mkdir(parents=True, exist_ok=True)
    ac = csptree.to_ac(np.c_[trees.x, trees.y, trees.z])
    i = np.floor(trees.x / tile_m).astype(int); j = np.floor(trees.y / tile_m).astype(int)
    out = []
    for key in sorted(set(zip(i.tolist(), j.tolist()))):
        sel = np.flatnonzero((i == key[0]) & (j == key[1]))
        lines = ["configure: angle variance = 0, 0", "configure: size variance = 1, 1"]
        lines += [csptree.tree_line(f"{trees.pattern[k]}.zip", *ac[k], trees.angle_deg[k], trees.scale[k]) for k in sel]
        name = f"forest_{key[0]}_{key[1]}.txt"
        (ext_dir / "trees" / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        out.append(f"trees\{name}")
    return out


def _check_ours(p: Path):
    if p.exists() and not p.read_text(encoding="utf-8").startswith(MARK):
        raise RuntimeError(f"{p} はこのスクリプトが書いたものではないので、上書きしません")


def write_ext_config(track_dir: Path, lists, hidden_meshes):
    p = Path(track_dir) / "extension" / "ext_config.ini"
    _check_ours(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    s = [MARK, "", "[TREES]", "ACTIVE=1"] + [f"LIST_{k}={l}" for k, l in enumerate(lists)]
    if hidden_meshes:   # 古い板の木（kn5 の中）を隠す
        s += ["", "[MESH_ADJUSTMENT_...]", "MESHES=" + ",".join(hidden_meshes), "IS_ACTIVE=0"]
    p.write_text("\n".join(s) + "\n", encoding="utf-8", newline="\n")


def write_layout_off(track_dir: Path, layouts):
    """森を入れないレイアウト（Codex の S1 景観確認）。"""
    for lay in layouts:
        p = Path(track_dir) / lay / "extension" / "ext_config.ini"
        _check_ours(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{MARK}\n\n[TREES]\nACTIVE=0\n", encoding="utf-8", newline="\n")
