"""保存済みの舗装面から運転席22視点を作り、写真2視点と一緒に収める。"""

import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np

from yabitsu import acdata, kn5, track
from yabitsu.crosssection import Section
from yabitsu.valley_fit import Surfaces

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "export/compare/route70/work/cap"
LUA_PATH = "YabitsuFpsProbe/YabitsuFpsProbe.lua"
PHOTO_NAMES = ("r70_0366_f", "r70_0366_b")
ASPECT = 5120 / 1440  # launch-cap.ps1
HORIZONTAL_FOV = 56.


def drive_views(section, start, surface_height):
    result = []
    heading = np.unwrap(section.heading)
    for station in range(1600, 1801, 20):
        s = start + station
        angle = np.interp(s, section.s, heading)
        forward = np.array([math.cos(angle), math.sin(angle)])
        left = np.array([-forward[1], forward[0]])
        center = np.array([np.interp(s, section.s, a) for a in (section.x, section.y)])
        nominal_z = np.interp(s, section.s, section.z)
        for name, sign, width in (("up", -1, section.hw_right), ("down", 1, section.hw_left)):
            lateral = sign * np.interp(s, section.s, width) / 2
            xy = (center + lateral * left).round(4)
            floor = float(surface_height(xy, nominal_z))
            if not math.isfinite(floor):
                raise ValueError(f"S4 {station} {name}: 配置先の舗装面がありません")
            result.append(dict(name=f"s4_{station}_{name}", station=station,
                               pos=[*xy, floor + 1.2], dir=[*(sign * forward), 0.],
                               surface_z=floor, lateral_m=lateral, horizontal_fov=HORIZONTAL_FOV))
    return result


def road_surfaces(track_dir, low, high):
    entries = json.loads((track_dir / "manifest.json").read_text())["meshes"]
    triangles = []
    for name, mesh in entries.items():
        if not mesh["renderable"] or not re.fullmatch(r"1ROAD_s1_\d+|smallspan_\d+_deck", name):
            continue
        with np.load(track_dir / mesh["file"]) as array:
            verts, faces = array["verts"], array["faces"]
        if ((verts[:, :2].min(axis=0) > high) | (verts[:, :2].max(axis=0) < low)).any():
            continue
        tris = verts[kn5._triangles(faces)]
        local = ((tris[..., :2].min(axis=1) <= high) & (tris[..., :2].max(axis=1) >= low)).all(axis=1)
        triangles.append(tris[local])
    return Surfaces(np.concatenate(triangles))


def vec3(values):
    return "vec3(" + ",".join(f"{v:.6f}" for v in acdata.blender_to_ac(*values)) + ")"


def write_harness(views, dest, details):
    vertical_fov = math.degrees(2 * math.atan(math.tan(math.radians(HORIZONTAL_FOV) / 2) / ASPECT))
    with ZipFile(CAP / "harness-cap-valleys-38.zip") as source:
        old_lua = source.read(LUA_PATH).decode()
        original = re.search(r"^local views=(.+)$", old_lua, re.M)
        entries = {m[1]: m[0] for m in re.finditer(r'\{name="([^"]+)"[^}]+\}', original[1])}
        rows = [
            f'{{name="{v["name"]}",pos={vec3(v["pos"])},dir={vec3(v["dir"])},fov={vertical_fov:.6f},drive=true}}'
            for v in views
        ] + [entries[n] for n in PHOTO_NAMES]
        lua = old_lua[:original.start()] + "local views={" + ",".join(rows) + "}" + old_lua[original.end():]
        # 写真は既存の自由カメラ、運転席は垂直画角がSDKに明記されたカメラを使う。
        a = lua.index("local probes=")
        b = lua.index("if enabled then record", a)
        lua = lua[:a] + """local camera
local function releaseCamera()
  if camera then camera:dispose(); camera = nil end
end
""" + lua[b:]
        a = lua.index("function script.frameBegin()")
        b = lua.index("function script.update(dt)", a)
        lua = lua[:a] + """function script.frameBegin()
  if not enabled then return end
  if not current or not current.drive then
    releaseCamera()
    if current then
      ac.setCurrentCamera(ac.CameraMode.Free)
      ac.setCameraPosition(current.pos)
      ac.setCameraDirection(current.dir)
      ac.setCameraFOV(current.fov)
    end
    return
  end
  if not camera then
    local err
    camera, err = ac.grabCamera('Yabitsu drive capture')
    if not camera then record('camera error '..tostring(err)); finish(); return end
  end
  camera.transform.position:set(current.pos)
  camera.transform.look:set(current.dir)
  camera.transform.up:set(0, 1, 0)
  camera.fov = current.fov
end
""" + lua[b:]
        lua = lua.replace(" or ac.getTrackID() == 'yabitsu'", "").replace("; runProbes()", "")
        lua = lua.replace("    if v.name=='portal_near_k_a' then runProbes() end\n", "")
        lua = lua.replace("output..'cap_'..v.name", "output..runTag..'_cap_'..v.name")
        lua = lua.replace("if not enabled then return end\n  if not current",
                          "if not enabled or done then releaseCamera(); return end\n  if not current")
        with ZipFile(dest, "w") as target:
            for item in source.infolist():
                target.writestr(item, lua.encode() if item.filename == LUA_PATH else source.read(item))
    details.write_text(json.dumps(dict(
        aspect=ASPECT, vertical_fov=vertical_fov, drive=views,
        photos=[entries[name] for name in PHOTO_NAMES],
    ), indent=2) + "\n")


def main():
    data = ROOT / "data/processed"
    section = Section.load_json(data / "track_section.json")
    start = track.load_bounds(data)["s4"][0]
    nearby = (section.s >= start + 1595) & (section.s <= start + 1805)
    xy = np.c_[section.x[nearby], section.y[nearby]]
    pavement = road_surfaces(data / "track", xy.min(axis=0) - 8, xy.max(axis=0) + 8)

    def height(point, nominal):
        return pavement.height(point[None, :], ceiling=nominal + 2.)[0]

    views = drive_views(section, start, height)
    write_harness(views, CAP / "harness-cap-drive-57.zip", CAP.parent / "forest_57/drive_views.json")


if __name__ == "__main__":
    main()
