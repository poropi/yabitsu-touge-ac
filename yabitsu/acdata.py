import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import assert_writable


def blender_to_ac(x, y, z):
    """Blender（X=東, Y=北, Z=上）→ AC（X=東, Y=上, Z=南）。Task 2 の試験で確認する規約。"""
    return (x, z, -y)


def write_text(path, text: str) -> Path:
    p = assert_writable(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\r\n")
    return p


def ini_text(sections: dict) -> str:
    out = []
    for name, kv in sections.items():
        out.append(f"[{name}]")
        out += [f"{k}={v}" for k, v in kv.items()]
        out.append("")
    return "\n".join(out)


_SURFACE_BASE = dict(FRICTION=0.99, DAMPING=0, WAV="", WAV_PITCH=0, FF_EFFECT="NULL", DIRT_ADDITIVE=0,
                     IS_VALID_TRACK=1, BLACK_FLAG_TIME=0, SIN_HEIGHT=0, SIN_LENGTH=0, IS_PITLANE=0,
                     VIBRATION_GAIN=0, VIBRATION_LENGTH=0)
SURFACES = [
    {"KEY": "ROAD", **_SURFACE_BASE},
    {"KEY": "GRASS", **{**_SURFACE_BASE, "FRICTION": 0.6, "DAMPING": 0.1, "DIRT_ADDITIVE": 1,
                        "VIBRATION_GAIN": 0.1, "VIBRATION_LENGTH": 1.0}},
    {"KEY": "WALL", **{**_SURFACE_BASE, "FRICTION": 0.3}},
]


def write_surfaces_ini(path) -> Path:
    return write_text(path, ini_text({f"SURFACE_{i}": s for i, s in enumerate(SURFACES)}))


def write_models_ini(path, kn5_files) -> Path:
    secs = {f"MODEL_{i}": {"FILE": f, "POSITION": "0,0,0", "ROTATION": "0,0,0"} for i, f in enumerate(kn5_files)}
    return write_text(path, ini_text(secs))


@dataclass
class MapParams:
    width: int
    height: int
    margin: int
    scale: float
    x_offset: float
    z_offset: float
    drawing_size: int


def map_params(ac_xz, max_px=1024, margin=20, drawing_size=10) -> MapParams:
    xz = np.asarray(ac_xz, float)
    mn, mx = xz.min(0), xz.max(0)
    span = float(max(mx - mn))
    scale = span / (max_px - 2 * margin)
    w = int(math.ceil((mx[0] - mn[0]) / scale)) + 2 * margin
    h = int(math.ceil((mx[1] - mn[1]) / scale)) + 2 * margin
    return MapParams(w, h, margin, scale, -mn[0] + margin * scale, -mn[1] + margin * scale, drawing_size)


def to_map_px(ac_xz, p: MapParams):
    xz = np.asarray(ac_xz, float)
    return (xz[:, 0] + p.x_offset) / p.scale, (xz[:, 1] + p.z_offset) / p.scale


def write_map(png_path, ini_path, ac_xz, max_px=1024, margin=20, drawing_size=10) -> MapParams:
    p = map_params(ac_xz, max_px, margin, drawing_size)
    px, py = to_map_px(ac_xz, p)
    img = Image.new("RGBA", (p.width, p.height), (0, 0, 0, 0))
    ImageDraw.Draw(img).line(list(zip(px, py)), fill=(255, 255, 255, 255), width=drawing_size, joint="curve")
    out = assert_writable(png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    write_text(ini_path, ini_text({"PARAMETERS": {
        "WIDTH": f"{p.width:.6f}", "HEIGHT": f"{p.height:.6f}", "MARGIN": p.margin,
        "SCALE_FACTOR": f"{p.scale:.6f}", "MAX_SIZE": max_px, "X_OFFSET": f"{p.x_offset:.6f}",
        "Z_OFFSET": f"{p.z_offset:.6f}", "DRAWING_SIZE": p.drawing_size}}))
    return p


@dataclass
class Camera:
    name: str
    position: tuple
    forward: tuple
    in_point: float
    out_point: float
    fov: float = 40.0


def _vec(v):
    return ",".join(f"{c:.6f}" for c in v)


def write_cameras_ini(path, cams) -> Path:
    secs = {"HEADER": {"VERSION": 3, "CAMERA_COUNT": len(cams), "SET_NAME": ""}}
    world_up = np.array([0.0, 1.0, 0.0])
    for i, c in enumerate(cams):
        f = np.asarray(c.forward, float); f /= np.linalg.norm(f)
        u = world_up - (world_up @ f) * f; u /= np.linalg.norm(u)
        secs[f"CAMERA_{i}"] = {
            "NAME": c.name, "POSITION": _vec(c.position), "FORWARD": _vec(f), "UP": _vec(u),
            "MIN_FOV": c.fov, "MAX_FOV": c.fov, "IN_POINT": round(c.in_point, 4), "OUT_POINT": round(c.out_point, 4),
            "SHADOW_SPLIT0": 1.8, "SHADOW_SPLIT1": 20, "SHADOW_SPLIT2": 180, "NEAR_PLANE": 0.1, "FAR_PLANE": 5000,
            "MIN_EXPOSURE": 0, "MAX_EXPOSURE": 10000, "DOF_FACTOR": 10, "DOF_RANGE": 10000, "DOF_FOCUS": 0,
            "DOF_MANUAL": 0, "SPLINE": "", "SPLINE_ROTATION": 0, "FOV_GAMMA": 1, "SPLINE_ANIMATION_LENGTH": 15,
            "IS_FIXED": 0}
    return write_text(path, ini_text(secs))


def write_ui_track_json(path, *, name, description, length_m, pitboxes, geotags, tags, credits) -> Path:
    d = {"name": name, "description": f"{description}\n{credits}".strip(), "tags": list(tags),
         "geotags": [f"{geotags[0]:.5f}", f"{geotags[1]:.5f}"], "country": "Japan",
         "city": "Hadano, Kanagawa", "length": f"{round(length_m)} m", "width": "4-6 m",
         "pitboxes": str(pitboxes), "run": "A to B"}
    return write_text(path, json.dumps(d, ensure_ascii=False, indent=2))


UI_W, UI_H = 1920, 1080


def write_ui_images(ui_dir, xy, photo, margin=0.08, line_px=10):
    """コース選択画面の絵：outline.png（そのレイアウトで走る道だけ、白い線、透明の背景）と preview.png（同じ枠の空中写真）。
    xy＝走る道の点（ローカル座標、東・北）。photo(x0, y0, w_m, h_m, w_px, h_px) → (h_px, w_px, 3) の画像。"""
    xy = np.asarray(xy, float)
    lo, hi = xy.min(0), xy.max(0); c = (lo + hi) / 2; span = (hi - lo) * (1 + 2 * margin)
    w_m = max(span[0], span[1] * UI_W / UI_H); h_m = w_m * UI_H / UI_W
    x0, y0 = c[0] - w_m / 2, c[1] - h_m / 2
    px = UI_W / w_m
    pts = [((x - x0) * px, (y0 + h_m - y) * px) for x, y in xy]
    out = Path(ui_dir); assert_writable(out / "outline.png"); out.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (UI_W, UI_H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    d.line(pts, fill=(0, 0, 0, 200), width=line_px + 6, joint="curve")      # 縁取り（写真の上でも見えるように）
    d.line(pts, fill=(255, 255, 255, 255), width=line_px, joint="curve")
    img.save(out / "outline.png")
    Image.fromarray(np.asarray(photo(x0, y0, w_m, h_m, UI_W, UI_H), np.uint8)).convert("RGB").save(out / "preview.png")
