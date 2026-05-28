import pytest

from syssim.profiling.spec import load_profiling_spec, ProfilingSpec, DEFAULT_SPEC_PATH


def test_load_default_spec():
    spec = load_profiling_spec(DEFAULT_SPEC_PATH)
    assert isinstance(spec, ProfilingSpec)
    assert spec.hidden_sizes and spec.dtypes
    assert "fp8_e4m3" in spec.dtypes and "fp8_e5m2" in spec.dtypes   # FP8 required


def test_disallowed_key_raises(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("hidden_sizes: [2048]\nbogus_key: 1\n")
    with pytest.raises(ValueError, match="disallowed"):
        load_profiling_spec(str(p))
