import numpy as np
import pytest

from yabitsu import tunnel
from tests.test_meshgen import _straight
from tests.test_side_roads import _way


def test_tunnel_range_on_the_main_road():
    sec = _straight()
    ws = [_way([[100, 0], [160, 0]], id=7, highway="primary", ref="70", tunnel="yes", layer="-1", name="吹風トンネル"),
          _way([[20, 0], [40, 0]], id=8, highway="primary", ref="70", tunnel="yes")]
    t = sorted(tunnel.from_osm(ws, sec), key=lambda q: q.s0)
    assert [q.osm_id for q in t] == [8, 7]
    assert t[1].s0 == pytest.approx(100.0, abs=0.6) and t[1].s1 == pytest.approx(160.0, abs=0.6) and t[1].name == "吹風トンネル"
    assert t[0].name == ""


def test_tunnels_off_the_course_are_ignored():
    sec = _straight()
    ws = [_way([[100, 50], [160, 50]], id=9, highway="primary", tunnel="yes")]
    assert tunnel.from_osm(ws, sec) == []


def test_row_mask_covers_tunnel_and_margin():
    t = [tunnel.Tunnel(1, "", 100.0, 160.0)]
    s = np.arange(0.0, 201.0)
    m = tunnel.row_mask(s, t, margin_m=5.0)
    assert m[95] and m[165] and not m[94] and not m[166] and m.sum() == 71
