import numpy as np
from shapely.geometry import Point, box

from yabitsu import scenery as sc
from tests.test_side_roads import _way


def test_scenery_roads_skip_route70_and_excluded_areas():
    ways = [_way([[0, 50], [200, 50]], id=1), _way([[0, 80], [200, 80]], id=2, ref="70"),
            _way([[0, 120], [200, 120]], id=3, highway="footway"), _way([[0, 0], [10, 0]], id=4, highway="motorway")]
    keep_out = box(90, 0, 110, 200)
    g = sc.road_polygons(ways, keep_out)
    assert g.intersection(keep_out).area < 0.01
    assert g.contains(Point(50, 50)) and not g.contains(Point(50, 80)) and not g.contains(Point(5, 0))
    assert g.contains(Point(50, 120)) and not g.contains(Point(50, 121.0))     # 歩道は細い（1.5m）


def test_building_spec_defaults_by_size():
    small = sc.default_spec("a", [[35.0, 139.0]] * 4, area=30.0, rgb=np.array([0.5, 0.3, 0.2]))
    big = sc.default_spec("b", [[35.0, 139.0]] * 4, area=400.0, rgb=np.array([0.6, 0.6, 0.6]))
    assert small["storeys"] == 1 and small["roof"]["type"] == "gable"
    assert big["roof"]["type"] == "flat" and big["storeys"] >= 2
