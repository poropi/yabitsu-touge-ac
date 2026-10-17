import json
import shutil
from pathlib import Path

import numpy as np

MAX_VERTS = 65535


class Manifest:
    def __init__(self, root, reset=True):
        self.root = Path(root)
        if reset and (self.root / "meshes").exists():
            shutil.rmtree(self.root / "meshes")
        (self.root / "meshes").mkdir(parents=True, exist_ok=True)
        self.materials, self.meshes, self.empties, self.groups = {}, {}, {}, {}

    def _group(self, g):
        return self.groups.setdefault(g, {"meshes": [], "empties": []})

    def add_material(self, name, texture=None, color=None, alpha=False, two_sided=False):
        self.materials[name] = {"texture": texture, "color": color, "alpha": alpha, "two_sided": two_sided}

    def add_mesh(self, group, name, verts, faces, uvs, material, renderable=True):
        verts = np.asarray(verts, np.float64); faces = np.asarray(faces, np.int32); uvs = np.asarray(uvs, np.float32)
        if name in self.meshes:
            raise ValueError(f"名前が重複しています: {name}")
        if len(verts) > MAX_VERTS:
            raise ValueError(f"{name}: 頂点数 {len(verts)} が上限 {MAX_VERTS} を超えています")
        if len(faces) == 0:
            raise ValueError(f"{name}: 面がありません")
        rel = f"meshes/{name}.npz"
        np.savez_compressed(self.root / rel, verts=verts, faces=faces, uvs=uvs)
        self.meshes[name] = {"file": rel, "material": material, "renderable": bool(renderable),
                             "verts": int(len(verts)), "faces": int(len(faces))}
        self._group(group)["meshes"].append(name)

    def add_empty(self, group, dummy):
        key = f"{group}/{dummy.name}"
        if key in self.empties:
            raise ValueError(f"名前が重複しています: {key}")
        self.empties[key] = {"name": dummy.name, "location": list(dummy.location), "rotation_z": dummy.rotation_z}
        self._group(group)["empties"].append(key)

    def save(self) -> Path:
        p = self.root / "manifest.json"
        p.write_text(json.dumps({"materials": self.materials, "meshes": self.meshes, "empties": self.empties,
                                 "groups": self.groups}, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def summary(self) -> str:
        lines = []
        for g, c in self.groups.items():
            v = sum(self.meshes[n]["verts"] for n in c["meshes"]); f = sum(self.meshes[n]["faces"] for n in c["meshes"])
            lines.append(f"{g}: meshes={len(c['meshes'])} empties={len(c['empties'])} verts={v} faces={f}")
        return "\n".join(lines)
