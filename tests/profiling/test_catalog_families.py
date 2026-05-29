from syssim.profiling.spec import load_profiling_spec, DEFAULT_SPEC_PATH
from syssim.profiling.catalog import (
    attention_worklist, norm_worklist, elementwise_worklist, reduction_worklist, worklist,
)

SPEC = load_profiling_spec(DEFAULT_SPEC_PATH)


def test_attention_worklist():
    wl = attention_worklist(SPEC, seq_points=[1024, 4096])
    assert len(wl) > 0
    assert set(wl[0]) >= {"family", "B", "S", "H_q", "H_kv", "D", "causal", "dtype"}
    assert all(w["family"] == "attention" for w in wl)
    assert all(w["H_kv"] <= w["H_q"] for w in wl)        # GQA: kv groups <= query heads
    keys = {(w["B"], w["S"], w["H_q"], w["H_kv"], w["D"], w["causal"], w["dtype"]) for w in wl}
    assert len(keys) == len(wl)                          # deduped


def test_norm_worklist():
    wl = norm_worklist(SPEC, token_points=[512, 4096])
    assert len(wl) > 0
    assert set(wl[0]) >= {"family", "tokens", "hidden", "op_subtype", "dtype"}
    assert all(w["family"] == "normalization" for w in wl)


def test_elementwise_worklist():
    wl = elementwise_worklist(SPEC, token_points=[512])
    assert len(wl) > 0
    assert set(wl[0]) >= {"family", "total_elements", "op_subtype", "dtype"}
    assert all(w["family"] == "elementwise" for w in wl)


def test_reduction_worklist():
    wl = reduction_worklist(SPEC, seq_points=[1024])
    assert len(wl) > 0
    assert set(wl[0]) >= {"family", "op_subtype", "dtype"}
    assert all(w["family"] == "reduction" for w in wl)


def test_worklist_dispatcher_covers_families():
    wl = worklist(SPEC, families=["gemm", "attention", "normalization", "elementwise", "reduction"],
                  token_points=[512])
    fams = {it["family"] for it in wl}
    assert fams == {"gemm", "attention", "normalization", "elementwise", "reduction"}
    assert all("dtype" in it for it in wl)
