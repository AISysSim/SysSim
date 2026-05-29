from syssim.profiling.measure import measure_worklist


def test_worklist_sequential_runs_all():
    items = [{"family": "x", "v": i} for i in range(5)]
    rows = measure_worklist(items, num_workers=1, runner=lambda it: {"v": it["v"]})
    assert len(rows) == 5
    assert {r["v"] for r in rows} == {0, 1, 2, 3, 4}


def test_worklist_skips_failing_items():
    items = [{"family": "x", "v": i} for i in range(5)]

    def runner(it):
        if it["v"] == 2:
            raise RuntimeError("kernel/alignment failure")
        return {"v": it["v"]}

    rows = measure_worklist(items, num_workers=1, runner=runner)
    assert len(rows) == 4                     # the failing item is skipped, not fatal
    assert {r["v"] for r in rows} == {0, 1, 3, 4}


def test_worklist_empty():
    assert measure_worklist([], num_workers=1, runner=lambda it: it) == []
