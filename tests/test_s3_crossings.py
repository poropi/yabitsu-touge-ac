from scripts.s3_crossings import clusters, stop_rows


def test_two_crosswalks_of_one_junction_get_one_stop_line_each_way():
    rows = stop_rows([767.0, 782.0, 1574.0])
    up = [r for r in rows if r["kind"] == "stop_line"]; down = [r for r in rows if r["kind"] == "stop_line_down"]
    assert len(up) == 2 and len(down) == 2
    assert up[0]["s_to"] == 767.0 - 2.0 - 2.0 and down[0]["s_from"] == 782.0 + 2.0 + 2.0
    assert clusters([1.0, 20.0, 100.0]) == [[1.0, 20.0], [100.0]]
