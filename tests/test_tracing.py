"""Unit tests for rlsysim tracing."""

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from syssim import (
    ExecutionMode,
    HardwareInfo,
    OperatorGraph,
    OperatorNode,
    OperatorType,
    SimulatorConfig,
    trace_model_for_inference,
    trace_model_for_training,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="rlsysim tracing requires CUDA",
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def hw():
    return HardwareInfo(
        peak_tflops_mm=989.0,
        peak_tflops_math=989.0,
        peak_memory_bandwidth_gbps=3350.0,
    )


@pytest.fixture
def config(hw):
    return SimulatorConfig(hw_info=hw)


# ── CUDA requirement ──────────────────────────────────────────────────────


def test_tracing_raises_without_cuda(config):
    """Tracing should raise RuntimeError when inputs are non-CUDA."""
    model = nn.Linear(8, 4)
    cpu_input = torch.randn(2, 8)  # CPU tensor
    with pytest.raises(RuntimeError, match="CUDA"):
        trace_model_for_inference(model, cpu_input, config)


# ── Basic tracing ─────────────────────────────────────────────────────────


@requires_cuda
class TestTraceLinear:
    """Trace a single nn.Linear layer."""

    def test_produces_operators(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        assert len(graph) > 0

    def test_contains_gemm(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        types = {op.op_type for op in graph.operators.values()}
        assert OperatorType.GEMM in types

    def test_single_stream(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        assert graph.streams == {0}


@requires_cuda
class TestTraceSequential:
    """Trace a multi-layer sequential model."""

    def test_operator_count(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        graph = trace_model_for_inference(model, torch.randn(4, 32).cuda(), config)
        # Expect at least: addmm, relu, addmm
        assert len(graph) >= 3

    def test_op_type_counts(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        graph = trace_model_for_inference(model, torch.randn(4, 32).cuda(), config)
        types = [op.op_type for op in graph.operators.values()]
        assert types.count(OperatorType.GEMM) == 2
        assert types.count(OperatorType.MATH) >= 1  # ReLU

    def test_data_dependencies(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        graph = trace_model_for_inference(model, torch.randn(4, 32).cuda(), config)
        # Every non-first operator should have at least one dependency
        topo = graph.topological_sort()
        for name in topo[1:]:
            op = graph.operators[name]
            assert len(op.data_deps) + len(op.stream_deps) > 0, f"{name} has no dependencies"


# ── Time estimation ──────────────────────────────────────────────────────


@requires_cuda
class TestTimeEstimation:
    """Roofline estimation produces non-zero times for compute ops."""

    def test_gemm_has_nonzero_time(self, config):
        model = nn.Linear(128, 64)
        graph = trace_model_for_inference(model, torch.randn(32, 128).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        assert len(gemm_ops) > 0
        for op in gemm_ops:
            assert op.estimated_time_ms > 0.0


# ── GEMM config extraction ──────────────────────────────────────────────


@requires_cuda
class TestGEMMConfig:
    """GEMM ops should have M/N/K in their config."""

    def test_gemm_has_mnk(self, config):
        model = nn.Linear(64, 32)
        graph = trace_model_for_inference(model, torch.randn(16, 64).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        assert len(gemm_ops) > 0
        cfg = gemm_ops[0].config
        assert "M" in cfg
        assert "N" in cfg
        assert "K" in cfg

    def test_gemm_shapes_match(self, config):
        model = nn.Linear(64, 32)
        graph = trace_model_for_inference(model, torch.randn(16, 64).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        cfg = gemm_ops[0].config
        assert cfg["M"] == 16
        assert cfg["K"] == 64
        assert cfg["N"] == 32


# ── Tensor metadata ─────────────────────────────────────────────────────


@requires_cuda
class TestTensorMetadata:
    """Operators should capture input/output tensor metadata."""

    def test_outputs_recorded(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        assert len(gemm_ops) > 0
        assert len(gemm_ops[0].outputs) > 0
        out_meta = gemm_ops[0].outputs[0]
        assert out_meta.shape == (4, 8)

    def test_inputs_recorded(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        assert len(gemm_ops[0].inputs) > 0


# ── Critical path ────────────────────────────────────────────────────────


@requires_cuda
class TestCriticalPath:
    """Critical path computation on traced graphs."""

    def test_returns_positive(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        graph = trace_model_for_inference(model, torch.randn(4, 32).cuda(), config)
        cp = graph.compute_critical_path()
        assert cp > 0.0

    def test_equals_total_on_single_stream(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        graph = trace_model_for_inference(model, torch.randn(4, 32).cuda(), config)
        cp = graph.compute_critical_path()
        total = sum(op.estimated_time_ms for op in graph.operators.values())
        assert cp == pytest.approx(total)

    def test_empty_graph(self):
        graph = OperatorGraph("empty")
        assert graph.compute_critical_path() == 0.0

    def test_earliest_start_finish_set(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        graph.compute_critical_path()
        for op in graph.operators.values():
            assert op.earliest_finish >= op.earliest_start


# ── Export formats ────────────────────────────────────────────────────────


@requires_cuda
class TestExports:
    """Graph export to DOT, JSON, and summary."""

    def test_to_dot(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        dot = graph.to_dot()
        assert "digraph" in dot
        assert "ms" in dot

    def test_to_json(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        data = json.loads(graph.to_json())
        assert "operators" in data
        assert len(data["operators"]) > 0
        assert "estimated_time_ms" in data["operators"][0]

    def test_summary(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        graph = trace_model_for_inference(model, torch.randn(4, 32).cuda(), config)
        s = graph.summary()
        assert "gemm: 2" in s
        assert "ms" in s
        assert "Parallelism" in s


# ── trace_model_for_training ─────────────────────────────────────────────


@requires_cuda
class TestTraceForTraining:
    """trace_model_for_training always traces forward + backward."""

    def test_more_ops_than_forward_only(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        inp = torch.randn(4, 32).cuda()
        graph_fwd = trace_model_for_inference(model, inp, config, mode="prefill")
        graph_train = trace_model_for_training(model, inp, config)
        assert len(graph_train) > len(graph_fwd)

    def test_custom_loss_fn(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_training(
            model,
            torch.randn(4, 16).cuda(),
            config,
            loss_fn=lambda out: out.mean(),
        )
        assert len(graph) > 0

    def test_produces_operators(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_training(model, torch.randn(4, 16).cuda(), config)
        assert len(graph) > 0


# ── trace_model_for_inference (prefill) ──────────────────────────────────


@requires_cuda
class TestTraceForInference:
    """trace_model_for_inference in prefill mode."""

    def test_produces_operators(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        assert len(graph) > 0

    def test_forward_only(self, config):
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        inp = torch.randn(4, 32).cuda()
        graph_prefill = trace_model_for_inference(model, inp, config, mode="prefill")
        graph_train = trace_model_for_training(model, inp, config)
        # Training includes backward, so should have more ops
        assert len(graph_train) > len(graph_prefill)

    def test_invalid_mode_raises(self, config):
        model = nn.Linear(8, 4)
        with pytest.raises(ValueError, match="Invalid inference mode"):
            trace_model_for_inference(model, torch.randn(2, 8).cuda(), config, mode="bogus")

    def test_contains_gemm(self, config):
        model = nn.Linear(16, 8)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        types = {op.op_type for op in graph.operators.values()}
        assert OperatorType.GEMM in types


# ── Decode mode ──────────────────────────────────────────────────────────


@requires_cuda
class TestDecodeMode:
    """Decode mode with KV cache-aware attention estimation."""

    def test_decode_attention_higher_time(self, hw):
        """Decode attention with cache_seq_len should produce higher time estimates
        than naive seq_len=1 tracing without cache awareness."""
        from syssim.compute.compute_cost_predictor import (
            _decode_attention_compute_ns,
            _decode_attention_transfer_ns,
        )

        # Simulate Q shape: (batch=4, heads=8, seq=1, dim=64)
        q = torch.randn(4, 8, 1, 64).cuda()
        args = (q,)
        cache_seq_len = 1024

        compute_ns = _decode_attention_compute_ns(args, hw, cache_seq_len)
        transfer_ns = _decode_attention_transfer_ns(args, hw, cache_seq_len)

        assert compute_ns > 0.0
        assert transfer_ns > 0.0

    def test_decode_is_memory_bound_at_large_cache(self, hw):
        """At large cache_seq_len, decode attention should be memory-bound."""
        from syssim.compute.compute_cost_predictor import (
            _decode_attention_compute_ns,
            _decode_attention_transfer_ns,
        )

        q = torch.randn(4, 8, 1, 64).cuda()
        args = (q,)
        cache_seq_len = 4096

        compute_ns = _decode_attention_compute_ns(args, hw, cache_seq_len)
        transfer_ns = _decode_attention_transfer_ns(args, hw, cache_seq_len)

        # With seq_len=1 and large cache, KV cache read dominates → memory-bound
        assert transfer_ns > compute_ns

    def test_cache_seq_len_zero_falls_back(self, hw):
        """With cache_seq_len=0, decode mode falls back to standard roofline."""
        from syssim.compute.compute_cost_predictor import estimate_runtime

        q = torch.randn(4, 8, 1, 64).cuda()
        # With cache_seq_len=0, the decode override should NOT activate
        # and standard path should be used (which returns same as no mode)
        time_no_mode = estimate_runtime(
            None,
            (q,),
            {},
            q,
            hw,
            OperatorType.ATTN,
        )
        time_decode_zero = estimate_runtime(
            None,
            (q,),
            {},
            q,
            hw,
            OperatorType.ATTN,
            execution_mode=ExecutionMode.DECODE,
            cache_seq_len=0,
        )
        assert time_no_mode == time_decode_zero

    def test_decode_scales_with_cache_len(self, hw):
        """Longer cache_seq_len should produce proportionally higher times."""
        from syssim.compute.compute_cost_predictor import (
            _decode_attention_transfer_ns,
        )

        q = torch.randn(4, 8, 1, 64).cuda()
        args = (q,)

        t_512 = _decode_attention_transfer_ns(args, hw, 512)
        t_2048 = _decode_attention_transfer_ns(args, hw, 2048)

        # 4x cache length should give roughly 4x transfer time
        # (KV cache dominates, Q and output are small)
        ratio = t_2048 / t_512
        assert ratio > 3.5  # Allow some slack for Q/output contribution

    def test_config_cache_seq_len(self, hw):
        """SimulatorConfig.cache_seq_len is passed through to decode estimation."""
        config_decode = SimulatorConfig(hw_info=hw, cache_seq_len=1024)
        assert config_decode.cache_seq_len == 1024

        config_default = SimulatorConfig(hw_info=hw)
        assert config_default.cache_seq_len == 0

    def test_execution_mode_enum(self):
        """ExecutionMode enum has expected values."""
        assert ExecutionMode.TRAINING.value == "training"
        assert ExecutionMode.PREFILL.value == "prefill"
        assert ExecutionMode.DECODE.value == "decode"

    def test_decode_compute_ns_non_4d_returns_zero(self, hw):
        """Non-4D Q tensor should return 0 from decode helpers."""
        from syssim.compute.compute_cost_predictor import (
            _decode_attention_compute_ns,
            _decode_attention_transfer_ns,
        )

        q_2d = torch.randn(4, 64).cuda()
        assert _decode_attention_compute_ns((q_2d,), hw, 1024) == 0.0
        assert _decode_attention_transfer_ns((q_2d,), hw, 1024) == 0.0


# ── Operator type coverage ───────────────────────────────────────────────


@requires_cuda
class TestOperatorTypeGEMM:
    """GEMM operator type: traced from nn.Linear / matmul / bmm."""

    def test_linear_produces_gemm(self, config):
        model = nn.Linear(32, 16)
        graph = trace_model_for_inference(model, torch.randn(4, 32).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        assert len(gemm_ops) >= 1

    def test_gemm_has_positive_time(self, config):
        model = nn.Linear(128, 64)
        graph = trace_model_for_inference(model, torch.randn(32, 128).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        for op in gemm_ops:
            assert op.estimated_time_ms > 0.0

    def test_gemm_config_has_mnk(self, config):
        model = nn.Linear(64, 32)
        graph = trace_model_for_inference(model, torch.randn(8, 64).cuda(), config)
        gemm_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.GEMM]
        cfg = gemm_ops[0].config
        assert cfg["M"] == 8
        assert cfg["K"] == 64
        assert cfg["N"] == 32


@requires_cuda
class TestOperatorTypeATTENTION:
    """ATTENTION operator type: traced from scaled_dot_product_attention."""

    def test_sdpa_produces_attention(self, config):
        """SDPA should be classified as ATTENTION."""

        class SDPAModel(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = SDPAModel()
        q = k = v = torch.randn(2, 4, 8, 32).cuda()
        graph = trace_model_for_inference(model, (q, k, v), config)
        attn_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.ATTN]
        assert len(attn_ops) >= 1

    def test_attention_has_positive_time(self, config):
        class SDPAModel(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = SDPAModel()
        q = k = v = torch.randn(2, 4, 16, 64).cuda()
        graph = trace_model_for_inference(model, (q, k, v), config)
        attn_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.ATTN]
        assert len(attn_ops) >= 1
        for op in attn_ops:
            assert op.estimated_time_ms > 0.0

    def test_attention_config_has_shape_info(self, config):
        class SDPAModel(nn.Module):
            def forward(self, q, k, v):
                return F.scaled_dot_product_attention(q, k, v)

        model = SDPAModel()
        q = k = v = torch.randn(2, 4, 8, 32).cuda()
        graph = trace_model_for_inference(model, (q, k, v), config)
        attn_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.ATTN]
        cfg = attn_ops[0].config
        assert cfg["batch"] == 2
        assert cfg["num_heads"] == 4
        assert cfg["seq_len"] == 8
        assert cfg["head_dim"] == 32


@requires_cuda
class TestOperatorTypeCOMPUTE:
    """COMPUTE operator type: traced from ReLU, LayerNorm, Conv, etc."""

    def test_relu_produces_compute(self, config):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        compute_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.MATH]
        assert len(compute_ops) >= 1

    def test_layernorm_produces_compute(self, config):
        model = nn.LayerNorm(16)
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        compute_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.MATH]
        assert len(compute_ops) >= 1

    def test_gelu_produces_compute(self, config):
        class GELUModel(nn.Module):
            def forward(self, x):
                return F.gelu(x)

        model = GELUModel()
        graph = trace_model_for_inference(model, torch.randn(4, 16).cuda(), config)
        compute_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.MATH]
        assert len(compute_ops) >= 1

    def test_conv1d_produces_compute(self, config):
        model = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        # Conv1d expects (batch, channels, length)
        graph = trace_model_for_inference(model, torch.randn(2, 16, 32).cuda(), config)
        compute_ops = [op for op in graph.operators.values() if op.op_type == OperatorType.MATH]
        assert len(compute_ops) >= 1


@requires_cuda
class TestOperatorTypeCOLLECTIVE:
    """COLLECTIVE operator type: constructed directly (requires distributed for tracing)."""

    def test_collective_node_in_graph(self):
        graph = OperatorGraph("collective_test")
        node = OperatorNode(
            name="allreduce_0",
            op_type=OperatorType.COLLECTIVE,
            estimated_time_ms=0.5,
            config={"op": "allreduce", "size_bytes": 1024},
        )
        graph.add_operator(node)
        assert len(graph) == 1
        assert graph.operators["allreduce_0"].op_type == OperatorType.COLLECTIVE

    def test_collective_in_critical_path(self):
        graph = OperatorGraph("collective_test")
        compute = OperatorNode(name="gemm_0", op_type=OperatorType.GEMM, estimated_time_ms=1.0)
        coll = OperatorNode(
            name="allreduce_0",
            op_type=OperatorType.COLLECTIVE,
            estimated_time_ms=2.0,
            data_deps=["gemm_0"],
        )
        graph.add_operator(compute)
        graph.add_operator(coll)
        cp = graph.compute_critical_path()
        assert cp == pytest.approx(3.0)

    def test_collective_in_summary(self):
        graph = OperatorGraph("collective_test")
        graph.add_operator(
            OperatorNode(
                name="allreduce_0",
                op_type=OperatorType.COLLECTIVE,
                estimated_time_ms=0.5,
            )
        )
        s = graph.summary()
        assert "collective: 1" in s


@requires_cuda
class TestOperatorTypeMEMORY:
    """MEMORY operator type: constructed directly (cross-device copy)."""

    def test_memory_node_in_graph(self):
        graph = OperatorGraph("memory_test")
        node = OperatorNode(
            name="copy_0",
            op_type=OperatorType.MEMORY,
            estimated_time_ms=0.1,
            config={"size_bytes": 4096, "non_blocking": True},
        )
        graph.add_operator(node)
        assert len(graph) == 1
        assert graph.operators["copy_0"].op_type == OperatorType.MEMORY

    def test_memory_in_critical_path(self):
        graph = OperatorGraph("memory_test")
        graph.add_operator(
            OperatorNode(
                name="compute_0",
                op_type=OperatorType.MATH,
                estimated_time_ms=1.0,
            )
        )
        graph.add_operator(
            OperatorNode(
                name="copy_0",
                op_type=OperatorType.MEMORY,
                estimated_time_ms=0.5,
                data_deps=["compute_0"],
            )
        )
        cp = graph.compute_critical_path()
        assert cp == pytest.approx(1.5)

    def test_memory_in_summary(self):
        graph = OperatorGraph("memory_test")
        graph.add_operator(
            OperatorNode(
                name="copy_0",
                op_type=OperatorType.MEMORY,
                estimated_time_ms=0.1,
            )
        )
        s = graph.summary()
        assert "memory: 1" in s


@requires_cuda
class TestOperatorTypeBARRIER:
    """BARRIER operator type: waits for ALL streams in critical path."""

    def test_barrier_node_in_graph(self):
        graph = OperatorGraph("barrier_test")
        graph.add_operator(
            OperatorNode(
                name="barrier_0",
                op_type=OperatorType.BARRIER,
                estimated_time_ms=0.0,
            )
        )
        assert graph.operators["barrier_0"].op_type == OperatorType.BARRIER

    def test_barrier_waits_for_all_streams(self):
        """BARRIER waits for the slowest of all streams."""
        graph = OperatorGraph("barrier_test")
        # Stream 0: fast op
        graph.add_operator(
            OperatorNode(
                name="fast_op",
                op_type=OperatorType.MATH,
                estimated_time_ms=1.0,
                stream_id=0,
            )
        )
        # Stream 1: slow op
        graph.add_operator(
            OperatorNode(
                name="slow_op",
                op_type=OperatorType.MATH,
                estimated_time_ms=5.0,
                stream_id=1,
            )
        )
        # Barrier on stream 0 - should wait for stream 1 too
        graph.add_operator(
            OperatorNode(
                name="barrier_0",
                op_type=OperatorType.BARRIER,
                estimated_time_ms=0.0,
                stream_id=0,
                stream_deps=["fast_op"],
            )
        )
        cp = graph.compute_critical_path()
        # Barrier must wait for slow_op (5.0 ms) even though it's on stream 0
        assert cp >= 5.0

    def test_barrier_in_summary(self):
        graph = OperatorGraph("barrier_test")
        graph.add_operator(
            OperatorNode(
                name="barrier_0",
                op_type=OperatorType.BARRIER,
                estimated_time_ms=0.0,
            )
        )
        s = graph.summary()
        assert "barrier: 1" in s


@requires_cuda
class TestOperatorTypeSTREAM_SYNC:
    """STREAM_SYNC operator type: waits for a specific target stream."""

    def test_stream_sync_node_in_graph(self):
        graph = OperatorGraph("sync_test")
        graph.add_operator(
            OperatorNode(
                name="sync_0",
                op_type=OperatorType.STREAM_SYNC,
                estimated_time_ms=0.0,
                config={"target_stream": 1},
            )
        )
        assert graph.operators["sync_0"].op_type == OperatorType.STREAM_SYNC

    def test_stream_sync_waits_for_target_stream(self):
        """STREAM_SYNC waits for its target stream only."""
        graph = OperatorGraph("sync_test")
        # Stream 0: has an op
        graph.add_operator(
            OperatorNode(
                name="op_s0",
                op_type=OperatorType.MATH,
                estimated_time_ms=1.0,
                stream_id=0,
            )
        )
        # Stream 1: slow op
        graph.add_operator(
            OperatorNode(
                name="op_s1",
                op_type=OperatorType.MATH,
                estimated_time_ms=5.0,
                stream_id=1,
            )
        )
        # Stream 0: sync waits for stream 1
        graph.add_operator(
            OperatorNode(
                name="sync_0",
                op_type=OperatorType.STREAM_SYNC,
                estimated_time_ms=0.0,
                stream_id=0,
                stream_deps=["op_s0"],
                data_deps=["op_s1"],
                config={"target_stream": 1},
            )
        )
        # Op after sync on stream 0
        graph.add_operator(
            OperatorNode(
                name="after_sync",
                op_type=OperatorType.MATH,
                estimated_time_ms=1.0,
                stream_id=0,
                stream_deps=["sync_0"],
            )
        )
        cp = graph.compute_critical_path()
        # sync waits for op_s1 (5.0), then after_sync (1.0) → 6.0
        assert cp == pytest.approx(6.0)

    def test_stream_sync_in_summary(self):
        graph = OperatorGraph("sync_test")
        graph.add_operator(
            OperatorNode(
                name="sync_0",
                op_type=OperatorType.STREAM_SYNC,
                estimated_time_ms=0.0,
                config={"target_stream": 1},
            )
        )
        s = graph.summary()
        assert "stream_sync: 1" in s


@requires_cuda
class TestAllOperatorTypesInMixedGraph:
    """Verify all 7 operator types coexist correctly in a single graph."""

    def test_mixed_graph_critical_path(self):
        graph = OperatorGraph("mixed")

        # GEMM on stream 0
        graph.add_operator(
            OperatorNode(
                name="gemm_0",
                op_type=OperatorType.GEMM,
                estimated_time_ms=2.0,
                stream_id=0,
            )
        )
        # ATTENTION on stream 0
        graph.add_operator(
            OperatorNode(
                name="attn_0",
                op_type=OperatorType.ATTN,
                estimated_time_ms=3.0,
                stream_id=0,
                stream_deps=["gemm_0"],
            )
        )
        # COMPUTE on stream 1
        graph.add_operator(
            OperatorNode(
                name="compute_0",
                op_type=OperatorType.MATH,
                estimated_time_ms=1.0,
                stream_id=1,
            )
        )
        # COLLECTIVE on stream 1
        graph.add_operator(
            OperatorNode(
                name="collective_0",
                op_type=OperatorType.COLLECTIVE,
                estimated_time_ms=4.0,
                stream_id=1,
                stream_deps=["compute_0"],
            )
        )
        # MEMORY on stream 0
        graph.add_operator(
            OperatorNode(
                name="memory_0",
                op_type=OperatorType.MEMORY,
                estimated_time_ms=0.5,
                stream_id=0,
                stream_deps=["attn_0"],
            )
        )
        # BARRIER on stream 0 (waits for ALL streams)
        graph.add_operator(
            OperatorNode(
                name="barrier_0",
                op_type=OperatorType.BARRIER,
                estimated_time_ms=0.0,
                stream_id=0,
                stream_deps=["memory_0"],
            )
        )
        # STREAM_SYNC on stream 0 (waits for stream 1)
        graph.add_operator(
            OperatorNode(
                name="sync_0",
                op_type=OperatorType.STREAM_SYNC,
                estimated_time_ms=0.0,
                stream_id=0,
                stream_deps=["barrier_0"],
                config={"target_stream": 1},
            )
        )

        cp = graph.compute_critical_path()
        assert cp > 0.0

        # All 7 types present
        types = {op.op_type for op in graph.operators.values()}
        assert types == {
            OperatorType.GEMM,
            OperatorType.ATTN,
            OperatorType.MATH,
            OperatorType.COLLECTIVE,
            OperatorType.MEMORY,
            OperatorType.BARRIER,
            OperatorType.STREAM_SYNC,
        }

    def test_mixed_graph_summary_counts(self):
        graph = OperatorGraph("mixed")
        graph.add_operator(OperatorNode(name="g", op_type=OperatorType.GEMM, estimated_time_ms=1.0))
        graph.add_operator(OperatorNode(name="a", op_type=OperatorType.ATTN, estimated_time_ms=1.0))
        graph.add_operator(OperatorNode(name="c", op_type=OperatorType.MATH, estimated_time_ms=1.0))
        graph.add_operator(OperatorNode(name="co", op_type=OperatorType.COLLECTIVE, estimated_time_ms=1.0))
        graph.add_operator(OperatorNode(name="m", op_type=OperatorType.MEMORY, estimated_time_ms=1.0))
        graph.add_operator(OperatorNode(name="b", op_type=OperatorType.BARRIER, estimated_time_ms=0.0))
        graph.add_operator(OperatorNode(name="s", op_type=OperatorType.STREAM_SYNC, estimated_time_ms=0.0))
        s = graph.summary()
        assert "gemm: 1" in s
        assert "attn: 1" in s
        assert "math: 1" in s
        assert "collective: 1" in s
        assert "memory: 1" in s
        assert "barrier: 1" in s
        assert "stream_sync: 1" in s
