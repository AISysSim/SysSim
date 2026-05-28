from syssim.profiling.spec import load_profiling_spec, DEFAULT_SPEC_PATH
from syssim.profiling.catalog import gemm_worklist


def test_gemm_worklist_from_spec_is_nonempty_and_well_formed():
    spec = load_profiling_spec(DEFAULT_SPEC_PATH)
    wl = gemm_worklist(spec, token_points=[512, 4096])
    assert len(wl) > 0
    item = wl[0]
    assert set(item) >= {"M", "K", "N", "dtype", "direction", "tp"}
    assert item["M"] in (512, 4096)
    # TP sharding present (N or K divided by a tp factor for some entry)
    assert any(it["tp"] > 1 for it in wl)
    # dedup: no exact duplicates
    keys = {(it["M"], it["K"], it["N"], it["dtype"], it["direction"], it["tp"]) for it in wl}
    assert len(keys) == len(wl)
