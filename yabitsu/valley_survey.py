"""B の対岸で 10 m DEM に切り替わる孔を、周囲の実測 5 m 画素でつなぐ。"""

from dataclasses import replace

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, maximum_filter

from . import geo, gsi


def bank_repair(row):
    survey = row.bed.get("dem5_bank")
    return None if survey is None else BankRepair(row, survey)


class BankRepair:
    def __init__(self, row, survey):
        # 調査した範囲は水系線に固定する。低水路をその中で曲げても同じ孔を埋める。
        self.row = row = replace(row, line=np.asarray(survey.get("reference_xy", row.line)))
        self.start, self.end = survey["along_m"]
        self.near, self.far = survey["right_m"]
        self.blend = survey["end_blend_m"]

        # EPSG:3857 のタイル画素中心が測量値。コース格子に再標本化した値を
        # 根拠にすると、孔の縁で有効な実測点まで除外してしまう。
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(row.line, axis=0), axis=1))]
        points = row.line[(arc >= self.start - 10.0) & (arc <= self.end + 10.0)]
        margin = self.far + 15.0
        lo, hi = points.min(axis=0) - margin, points.max(axis=0) + margin
        gx, gy = self.pixels(np.array([lo[0], lo[0], hi[0], hi[0]]), np.array([lo[1], hi[1], lo[1], hi[1]]))
        ix = np.arange(np.floor(gx.min()), np.ceil(gx.max()) + 1, dtype=int)
        iy = np.arange(np.floor(gy.min()), np.ceil(gy.max()) + 1, dtype=int)
        x, y = np.meshgrid(ix, iy)
        layers = gsi.default_sampler(gsi.TileCache(offline=True)).samplers
        self.layers = layers
        self.coarse = layers[2]
        raw = layers[0]._pixels(x, y)
        missing = ~np.isfinite(raw)
        raw[missing] = layers[1]._pixels(x[missing], y[missing])
        missing = ~np.isfinite(raw)
        valid = ~missing
        if valid.sum() < 3:
            raise ValueError(f"{row.id}: 対岸を補う保存済み5 m標高が不足しています")
        native = LinearNDInterpolator(np.c_[x[valid], y[valid]], raw[valid])
        raw[missing] = native(x[missing], y[missing])
        self.ground = RegularGridInterpolator((iy, ix), raw, bounds_error=False, fill_value=np.nan)

        # 欠けの周囲の1画素も含め、2 m粗格子の高い頂点から隣へ張り出した
        # 三角形をカバーする。この縁では元の5 m DEMの双線形補間に一致する。
        halo = maximum_filter(missing.astype(float), size=3)
        self.hole = RegularGridInterpolator((iy, ix), halo, bounds_error=False, fill_value=0.0)
        self.laser = native
        if "laser_pixels_z15" in survey:
            # 同じ5 m標高でも正方形の対角の張り方で三角網が変わる。
            # 測量に使ったネイティブ画素の矩形を固定して、再現可能にする。
            x0, y0, x1, y1 = survey["laser_pixels_z15"]
            x, y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            laser = layers[0]._pixels(x, y)
            absent = ~np.isfinite(laser)
            laser[absent] = layers[1]._pixels(x[absent], y[absent])
            finite = np.isfinite(laser)
            self.laser = LinearNDInterpolator(np.c_[x[finite], y[finite]], laser[finite])

    @staticmethod
    def pixels(x, y):
        fx, fy = gsi.tile_coords(*geo.from_local(x, y), 15)
        return fx * 256 - 0.5, fy * 256 - 0.5

    def _coordinates(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        gx, gy = self.pixels(x.ravel(), y.ravel())
        coordinates = np.c_[gy, gx]
        # 呼出し側は複数の橋をまとめて扱うことがある。谷筋へ投影する前に
        # 保存済み欠損の近傍へ絞る。
        ids = np.flatnonzero(self.hole(coordinates) > 0)
        _, along, left, _ = self.row.project(x.ravel()[ids], y.ravel()[ids])
        right = -left
        edge = np.minimum.reduce(
            [
                (along - self.start) / self.blend,
                (self.end - along) / self.blend,
                (right - self.near) / 3.0,
                (self.far - right) / 3.0,
            ]
        )
        active = edge > 0
        w = np.minimum(edge[active], 1.0)
        return x.shape, ids[active], coordinates[ids[active]], w * w * (3 - 2 * w)

    def contains(self, x, y):
        shape, ids, _, _ = self._coordinates(x, y)
        mask = np.zeros(shape, dtype=bool)
        mask.flat[ids] = True
        return mask

    def reference(self, x, y):
        """実測点の三角網と、調査範囲内で5 m DEMが欠けて10 mに切り替わった点。"""
        ll = geo.from_local(x, y)
        missing = self.contains(x, y)
        for layer in self.layers[:2]:
            missing &= ~np.isfinite(layer.sample(*ll))
        missing &= np.isfinite(self.coarse.sample(*ll))
        return self.laser(*self.pixels(x, y)), missing

    def low_flow_blend(self, x, y):
        """既存川の幅6 mから調査帯の低水路へ連続に移る縦断重み。"""
        from .valley_surface import smooth

        along = self.row.project(x, y)[1]
        return smooth((along - self.start + 30.0) / 13.0) * smooth((self.end + 20.0 - along) / 13.0)

    def shape_bed(self, rows, cap, base, at, x, y, step):
        """実測点からの上下限を保ちながら低水路、砂利州、支流の合流口をつなぐ。"""
        from .valley_surface import smooth

        river = next(row for row in rows if row.water)
        _, along, side, arc = river.project(x, y)
        floor = np.interp(along, arc, river.bed["z_m"])
        distance = abs(side)
        blend = self.low_flow_blend(x, y)
        slope = np.where(side >= 0, river.bank_left, river.bank_right)
        slope *= 1.0 + 0.055 * np.sin(along / 7 + np.sign(side))
        low = floor + slope * np.maximum(0.0, distance - river.width_m / 2) * 0.35
        raised = blend * smooth((6.0 - distance) / 3.5) * np.maximum(0.0, low - base[at])
        for tributary in rows:
            if tributary is river:
                continue
            _, station, lateral, knots = tributary.project(x, y)
            level = np.interp(station, knots, tributary.bed["z_m"])
            mouth = smooth((station - knots[-1] + 10.0) / 6.0) * smooth((7.0 - abs(lateral)) / 4.0)
            raised = np.maximum(raised, mouth * np.maximum(0.0, level - base[at]))
        cap = np.minimum(cap, self.height(x, y))
        cap = np.where(raised > 1e-6, np.maximum(cap, base[at] + raised), cap)
        bar = floor + 0.10 + 0.76 * smooth((distance - 0.7) / 2.0)
        bar_weight = blend * smooth((10.0 - distance) / 3.0) * smooth((-side - 0.45) / 1.8)
        cap += bar_weight * np.maximum(0.0, bar - cap)
        reference, missing = self.reference(x, y)
        hole = np.zeros_like(base, bool)
        hole[at] = missing
        border = smooth(1.0 - step * distance_transform_edt(~hole)[at] / 3.0)
        target = np.clip(cap, reference - 0.73, reference + 0.87)
        return cap + np.where(np.isfinite(reference), border * (target - cap), 0.0)

    def height(self, x, y):
        """絶対標高の上限。調査範囲の端では旧10 m標高へ滑らかに戻す。"""
        shape, ids, pixels, weight = self._coordinates(x, y)
        z = np.full(shape, np.inf)
        reference = self.ground(pixels)
        if not np.isfinite(reference).all():
            raise ValueError(f"{self.row.id}: 対岸の補間点が5 m標高の凸包の外です")
        # 端では、実際に置き換える 10 m DEM との差だけを補間する。
        target = reference.copy()
        ramp = weight < 1.0
        if ramp.any():
            edge = ids[ramp]
            lat, lon = geo.from_local(
                np.broadcast_to(x, shape).ravel()[edge], np.broadcast_to(y, shape).ravel()[edge]
            )
            source = self.coarse.sample(lat, lon)
            if not np.isfinite(source).all():
                raise ValueError(f"{self.row.id}: 端の保存済み10 m標高が欠測しています")
            target[ramp] += (1.0 - weight[ramp]) * np.maximum(0.0, source - reference[ramp])
        z.flat[ids] = target
        return z
