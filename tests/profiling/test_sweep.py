from syssim.profiling.sweep import margin_token_points


def test_margin_points_extend_just_outside_observed_range():
    pts = margin_token_points(observed_min=512, observed_max=4096, n=3)
    assert all(p > 0 for p in pts)
    assert max(pts) > 4096          # extends above
    assert min(pts) < 512           # extends below
    assert len(pts) == 6            # n below + n above
