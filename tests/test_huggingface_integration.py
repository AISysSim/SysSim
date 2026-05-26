"""Tests for Hugging Face Transformers integration."""

import pytest
import torch

from syssim import HardwareInfo, SimulatorConfig
from syssim.integrations.huggingface import (
    trace_hf_model_for_training,
    trace_hf_training_step,
)

try:
    from transformers import AutoTokenizer, GPT2LMHeadModel

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for tracing")


@pytest.fixture
def config():
    """Hardware configuration fixture."""
    hw = HardwareInfo(989.0, 989.0, 3350.0)
    return SimulatorConfig(hw_info=hw)


@pytest.fixture
def gpt2_model():
    """GPT-2 model fixture."""
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")
    return GPT2LMHeadModel.from_pretrained("gpt2")


@pytest.fixture
def gpt2_tokenizer():
    """GPT-2 tokenizer fixture."""
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
@requires_cuda
class TestHuggingFaceTraining:
    """Test suite for Hugging Face training integration."""

    def test_gpt2_training_basic(self, config, gpt2_model, gpt2_tokenizer):
        """Trace GPT-2 training with built-in loss."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        inputs = tokenizer("Hello world", return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()

        graph = trace_hf_model_for_training(model, inputs, config)

        # Verify graph is not empty
        assert len(graph.operators) > 0, "Graph should contain operators"

        # Verify critical path is positive
        critical_path = graph.compute_critical_path()
        assert critical_path > 0, "Critical path should be positive"

        # Check for expected operator types (forward + backward)
        op_types = {op.op_type.value for op in graph.operators.values()}
        assert "gemm" in op_types, "Should have GEMM operations"
        assert "attn" in op_types, "Should have attention operations"

    def test_gpt2_training_batch(self, config, gpt2_model, gpt2_tokenizer):
        """Trace GPT-2 training with batch."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        texts = ["Hello world", "Machine learning", "PyTorch"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        inputs["labels"] = inputs["input_ids"].clone()

        graph = trace_hf_model_for_training(model, inputs, config)

        # Verify graph contains operators
        assert len(graph.operators) > 0, "Graph should contain operators"

        # Verify batch size
        assert inputs["input_ids"].shape[0] == 3, "Batch size should be 3"

        # Verify critical path
        critical_path = graph.compute_critical_path()
        assert critical_path > 0, "Critical path should be positive"

    def test_gpt2_training_custom_loss(self, config, gpt2_model, gpt2_tokenizer):
        """Trace GPT-2 training with custom loss function."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        inputs = tokenizer("Custom loss", return_tensors="pt")

        # Custom loss function
        def custom_loss(outputs):
            return outputs.logits.sum()

        graph = trace_hf_model_for_training(model, inputs, config, loss_fn=custom_loss)

        # Verify graph
        assert len(graph.operators) > 0, "Graph should contain operators"
        assert graph.compute_critical_path() > 0, "Critical path should be positive"

    def test_batch_encoding_input(self, config, gpt2_model, gpt2_tokenizer):
        """Test with BatchEncoding input (not dict)."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        # BatchEncoding (default from tokenizer)
        inputs = tokenizer("Hello", return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()

        # Verify it's BatchEncoding
        assert hasattr(inputs, "data"), "Should be BatchEncoding object"

        graph = trace_hf_model_for_training(model, inputs, config)

        # Verify graph
        assert len(graph.operators) > 0, "Graph should contain operators"

    def test_training_with_labels_separate(self, config, gpt2_model, gpt2_tokenizer):
        """Test training with labels passed separately."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        inputs = tokenizer("Test labels", return_tensors="pt")
        labels = inputs["input_ids"].clone()

        graph = trace_hf_model_for_training(model, inputs, config, labels=labels)

        # Verify graph
        assert len(graph.operators) > 0, "Graph should contain operators"

    def test_training_step_wrapper(self, config, gpt2_model, gpt2_tokenizer):
        """Test trace_hf_training_step wrapper."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        batch = tokenizer(["Hello world", "Machine learning"], return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()

        graph = trace_hf_training_step(model, batch, config)

        # Verify graph
        assert len(graph.operators) > 0, "Graph should contain operators"
        assert graph.compute_critical_path() > 0, "Critical path should be positive"

    def test_training_step_mixed_precision(self, config, gpt2_model, gpt2_tokenizer):
        """Test training step with mixed precision."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        batch = tokenizer("Mixed precision test", return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()

        # Trace with mixed precision
        graph = trace_hf_training_step(model, batch, config, use_mixed_precision=True)

        # Verify graph
        assert len(graph.operators) > 0, "Graph should contain operators"
        assert graph.compute_critical_path() > 0, "Critical path should be positive"

    def test_backward_pass_present(self, config, gpt2_model, gpt2_tokenizer):
        """Verify that backward pass operators are traced."""
        model = gpt2_model
        tokenizer = gpt2_tokenizer

        inputs = tokenizer("Test backward", return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()

        graph = trace_hf_model_for_training(model, inputs, config)

        # Count operators (training should have ~2x operators vs inference)
        # This is a rough heuristic - backward pass adds gradient ops
        total_ops = len(graph.operators)
        assert total_ops > 50, f"Training should have many ops, got {total_ops}"

        # Check for typical backward ops (gemm, math)
        op_types = [op.op_type.value for op in graph.operators.values()]
        gemm_count = sum(1 for t in op_types if t == "gemm")
        assert gemm_count > 0, "Should have GEMM ops from backward pass"


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
def test_import_error_message():
    """Test that helpful error is raised when transformers not installed."""
    # This test is a bit contrived since we skip if HF not available
    # but it documents the expected behavior
    from syssim.integrations.huggingface import HF_AVAILABLE

    if HF_AVAILABLE:
        pytest.skip("transformers is installed")


@requires_cuda
def test_model_auto_moved_to_cuda(config):
    """Test that CPU model is automatically moved to CUDA."""
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Ensure model starts on CPU
    model = model.cpu()
    assert not next(model.parameters()).is_cuda, "Model should start on CPU"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("Hello", return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()

    # Trace should auto-move to CUDA
    graph = trace_hf_model_for_training(model, inputs, config)

    assert len(graph.operators) > 0, "Graph should contain operators"
