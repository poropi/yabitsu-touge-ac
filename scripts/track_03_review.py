import json

import numpy as np
from PIL import Image

from yabitsu import config, geo, gsi, photo, track
from yabitsu.centerline import Centerline

P = config.DATA_PROCESSED
SIZE, PX = 250.0, 0.5


def main():
    cl = Centerline.load(P / "track_centerline.npz")
    bounds = track.load_bounds()
    cache = gsi.TileCache(); dem = gsi.default_sampler(cache)
    (P / "review").mkdir(parents=True, exist_ok=True); (P / "track_underlay").mkdir(parents=True, exist_ok=True)
    sec = None
    if (P / "track_section.json").exists():
        from yabitsu.crosssection import Section
        sec = Section.load_json(P / "track_section.json")
    under = []
    for name in bounds:
        v = track.section_view(cl, bounds, name)
        hw = None if sec is None else np.interp(v.s + bounds[name][0], sec.s, (sec.hw_left + sec.hw_right) / 2)
        for k, (sc, x0, y0) in enumerate(photo.review_chunks(v, 150.0, SIZE)):
            img = photo.render_local_image(cache, geo.from_local, x0, y0, SIZE, PX)
            xs = x0 + np.arange(0.5, SIZE, 1.0); ys = y0 + SIZE - np.arange(0.5, SIZE, 1.0)
            X, Y = np.meshgrid(xs, ys)
            Z = dem.sample(*geo.from_local(X.ravel(), Y.ravel())).reshape(X.shape)
            hs = (photo.hillshade(np.nan_to_num(Z, nan=float(np.nanmean(Z))), 1.0) * 255).astype(np.uint8)
            hs_img = np.array(Image.fromarray(hs).resize(img.shape[1::-1])); hs_img = np.stack([hs_img] * 3, -1)
            left = photo.draw_overlay(img, v, x0, y0, SIZE, PX, hw); right = photo.draw_overlay(hs_img, v, x0, y0, SIZE, PX, hw)
            both = Image.new("RGB", (left.width * 2, left.height)); both.paste(left, (0, 0)); both.paste(right, (left.width, 0))
            both.save(P / "review" / f"{name}_chunk_{k:02d}_s{int(sc):04d}.png")
            fn = f"{name}_photo_{k:02d}.jpg"
            Image.fromarray(img).save(P / "track_underlay" / fn, quality=90)
            under.append({"file": fn, "x0": x0, "y0": y0, "size_m": SIZE})
    (P / "track_underlay" / "underlay.json").write_text(json.dumps(under), encoding="utf-8")
    print(f"{len(under)} chunks written for {list(bounds)} (width overlay: {'yes' if sec is not None else 'no'})")


if __name__ == "__main__":
    main()
