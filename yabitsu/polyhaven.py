"""Poly Haven（CC0）の素材を data/raw/polyhaven/ へダウンロードする（済んでいれば何もしない）。"""
import json
import urllib.request

from .config import DATA_RAW

ROOT = DATA_RAW / "polyhaven"
ASSETS = ("fir_tree_01", "tree_small_02", "fern_02")
UA = {"User-Agent": "yabitsu-track-builder/1.0"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def fetch(assets=ASSETS, res="2k", log=print):
    for a in assets:
        d = json.loads(_get(f"https://api.polyhaven.com/files/{a}"))
        g = d["gltf"][res]["gltf"]
        items = [(v["url"], k) for k, v in g["include"].items() if not k.endswith(".bin")]   # 形（bin）は使わない
        for key, fmts in d.items():
            if "alpha" in key.lower():
                f = fmts[res].get("png") or fmts[res].get("jpg")
                items.append((f["url"], "textures/" + f["url"].rsplit("/", 1)[1]))
        for url, rel in items:
            p = ROOT / a / rel
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_get(url))
            log(f"downloaded {a}/{rel}")
