from scripts import dress_build


def test_dress_table_fence_overrides_features(tmp_path):
    feats = tmp_path / "f.csv"
    feats.write_text("s_from,s_to,side,type,height_m,width_m,note\n2330,2480,L,guardrail,,,白いガードレール\n"
                     "100,200,R,guardrail,,,白いガードパイプ\n", encoding="utf-8")
    rows = [{"s_from": "2430", "s_to": "2510", "side": "L", "item": "fence", "value": "wbeam_redwhite", "note": ""}]
    runs = sorted(dress_build.fence_runs(feats, rows))
    assert (2430.0, 2510.0, "L", "wbeam_redwhite") in runs
    assert (2330.0, 2430.0, "L", "wbeam") in runs                    # 重ならない所は表（features）のまま
    assert not any(k == "wbeam" and a < 2470 < b and s == "L" for a, b, s, k in runs)
    assert (100.0, 200.0, "R", "white_pipe") in runs


def test_parts_skip_split_gap():
    """島で道が分かれる所（gap）には、本線の左右を基準にした柵・マンホールなどを置かない（分かれた道の中に立つ）。"""
    runs = [(3430.0, 3520.0, "R", "wbeam"), (3000.0, 3100.0, "L", "white_bar")]
    out = dress_build.cut_gaps(runs, [(3360.0, 3462.0)])
    assert (3462.0, 3520.0, "R", "wbeam") in out and (3000.0, 3100.0, "L", "white_bar") in out
    assert not any(a < 3440 < b for a, b, *_ in out)
