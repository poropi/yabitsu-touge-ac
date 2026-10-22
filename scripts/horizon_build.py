"""遠景（遠い地形の外の、地平線までの地形と海）を作って kn5 にする。
使い方: uv run python -m scripts.horizon_build [--no-install]
標高は地理院 dem_png（z=12）、写真は seamlessphoto（z=13）。出力は data/processed/horizon/。"""
import argparse
import json
import shutil

import numpy as np
from PIL import Image

from yabitsu import acdata, config, geo, gsi, horizon, kn5, layouts, photo
from yabitsu.manifest import Manifest

OUT = config.DATA_PROCESSED / "horizon"
GROUP = "yabitsu_horizon"
PX = 11.0     # 写真を引く 1 画素 m（44 km 四方で 4000 画素）。kn5 には TEX_PX に縮めて入れる（遠景なので 1 画素 約 21 m で足りる）
TEX_PX = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-install", action="store_true")
    a = ap.parse_args()
    far = dict(np.load(config.DATA_PROCESSED / "track_terrain_far.npz"))
    xs, ys = horizon.grid(far)
    cache = gsi.TileCache()
    dem = gsi.DemSampler(cache, layer="dem_png", z=12)

    def sample(X, Y):
        lat, lon = geo.from_local(X.ravel(), Y.ravel())
        return dem.sample(lat, lon)

    Z = horizon.heights(xs, ys, sample)
    keep = horizon.face_keep(xs, ys, far)
    if OUT.exists():
        shutil.rmtree(OUT)
    m = Manifest(OUT)
    tex = OUT / "textures"; tex.mkdir(parents=True)
    size = float(xs[-1] - xs[0])
    img = photo.render_local_image(cache, geo.from_local, float(xs[0]), float(ys[0]), size, px_size=PX, z=13)
    Image.fromarray(img).resize((TEX_PX, TEX_PX), Image.LANCZOS).save(tex / "horizon.jpg", quality=90)
    m.add_material("horizon", texture="textures/horizon.jpg")
    parts = horizon.meshes(xs, ys, Z, keep)
    for name, v, f, uv in parts:
        m.add_mesh(GROUP, name, v, f, uv, "horizon")
    m.save()
    (OUT / "retex.json").write_text("{}", encoding="utf-8")      # 確認描画（render_forest の DRESS_DIR）用
    (OUT / "hidden.json").write_text("[]", encoding="utf-8")
    kn5_path = OUT / f"{GROUP}.kn5"
    kn5.write_from_manifest(OUT / "manifest.json", GROUP, kn5_path)
    print(f"遠景：{len(parts)} メッシュ、格子 {Z.shape}、標高 {Z.min():.0f}〜{Z.max():.0f} m → {kn5_path} {kn5_path.stat().st_size / 1e6:.1f} MB")
    if a.no_install:
        return
    T = config.TRACK_DIR
    shutil.copy2(kn5_path, T / kn5_path.name)
    for lay in layouts.load_layouts():
        acdata.write_models_ini(T / f"models_{lay.name}.ini", layouts.kn5_files(lay.name))
    print("AC へ入れました：", T)


if __name__ == "__main__":
    main()
