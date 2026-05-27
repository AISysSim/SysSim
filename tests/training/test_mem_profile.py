"""Unit tests for MemoryProfile footprint decomposition."""

from syssim.training.mem_profile import MemoryProfile


def _profile():
    return MemoryProfile(
        persistent_by_type={"Parameter": 100, "Buffer": 0, "Gradient": 100, "Optstate": 600},
        act_bytes_per_mb=50,
        temp_bytes=20,
        peak_module="GPTModel.layers.0",
    )


def test_peak_bytes_scales_activation_only():
    mp = _profile()
    # persistent = 800; + in_flight*50 + 20
    assert mp.peak_bytes(in_flight=1) == 800 + 50 + 20
    assert mp.peak_bytes(in_flight=4) == 800 + 4 * 50 + 20


def test_peak_by_type_scales_activation_only():
    mp = _profile()
    d = mp.peak_by_type(in_flight=3)
    assert d["Parameter"] == 100
    assert d["Buffer"] == 0
    assert d["Gradient"] == 100
    assert d["Optstate"] == 600
    assert d["Activation"] == 3 * 50
    assert d["Temp"] == 20


def test_peak_gb_uses_in_flight():
    mp = _profile()
    assert abs(mp.peak_gb(in_flight=1) - (870 / 1e9)) < 1e-12


def test_to_dict_from_dict_roundtrip():
    mp = _profile()
    d = mp.to_dict()
    mp2 = MemoryProfile.from_dict(d)
    assert mp2.persistent_by_type == mp.persistent_by_type
    assert mp2.act_bytes_per_mb == mp.act_bytes_per_mb
    assert mp2.temp_bytes == mp.temp_bytes
    assert mp2.peak_module == mp.peak_module


def test_empty_profile_is_zero():
    mp = MemoryProfile()
    assert mp.peak_bytes(in_flight=8) == 0
    assert mp.peak_by_type(in_flight=8) == {"Activation": 0, "Temp": 0}


def test_from_mem_tracker_buckets_persistent_vs_per_mb():
    from syssim.training.mem_tracker import _MemRefType

    class _FakeModStats:
        def __init__(self, fqn, peak):
            self.mod_fqn = fqn
            self.local_peak = {"cuda:0": peak}

    class _FakeMT:
        def get_tracker_snapshot(self, kind):
            assert kind == "peak"
            return {"cuda:0": {
                _MemRefType.PARAM: 100,
                _MemRefType.BUFFER: 10,
                _MemRefType.GRAD: 100,
                _MemRefType.OPT: 600,
                _MemRefType.ACT: 50,
                _MemRefType.TEMP: 20,
                _MemRefType.OTH: 5,
                "Total": 885,
            }}
        memory_tracking = {
            0: _FakeModStats("GPTModel", 200),
            1: _FakeModStats("GPTModel.decoder.layers.0", 885),
        }

    mp = MemoryProfile.from_mem_tracker(_FakeMT())
    # Persistent = Parameter + Buffer + Gradient + Optstate + Other
    assert mp.persistent_by_type["Parameter"] == 100
    assert mp.persistent_by_type["Buffer"] == 10
    assert mp.persistent_by_type["Gradient"] == 100
    assert mp.persistent_by_type["Optstate"] == 600
    assert mp.persistent_by_type["Other"] == 5
    # Per-microbatch
    assert mp.act_bytes_per_mb == 50
    assert mp.temp_bytes == 20
    # peak_module = highest local_peak
    assert mp.peak_module == "GPTModel.decoder.layers.0"
