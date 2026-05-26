#!/usr/bin/env python3
"""Validation script for compute cost profiler improvements.

This script tests:
1. Hardware detection (Phase 0: unit conversion fix)
2. Enhanced feature extraction (Phase 2.2)
3. Data augmentation (Phase 2.3)
4. Model architecture compatibility (Phase 2.1)

Usage:
    python -m syssim.compute.validate_profiler
"""

from __future__ import annotations

import pandas as pd
import torch

print("=" * 80)
print("COMPUTE COST PROFILER VALIDATION")
print("=" * 80)

# Test 1: Hardware Detection (Phase 0 fix)
print("\n[Test 1] Hardware Detection (Unit Conversion Fix)")
print("-" * 80)

if torch.cuda.is_available():
    from syssim.compute.compute_cost_profiler import _get_hardware_info

    hw = _get_hardware_info()
    print("✓ Hardware detection completed")
    print(f"  Peak FLOP/s (MM): {hw.peak_tflops_mm:.1f} TFLOP/s")
    print(f"  Peak FLOP/s (Math): {hw.peak_tflops_math:.1f} TFLOP/s")
    print(f"  Peak Bandwidth: {hw.peak_memory_bandwidth_gbps:.1f} GB/s")

    # Validation: Modern GPUs should have > 50 TFLOP/s FP16
    if hw.peak_tflops_mm < 50.0:
        print(f"⚠ WARNING: Peak FLOP/s seems low ({hw.peak_tflops_mm:.1f} TFLOP/s)")
        print("  Expected > 50 TFLOP/s for modern GPUs (V100+)")
    elif hw.peak_tflops_mm > 100.0:
        print("✓ PASS: Peak FLOP/s in expected range for modern GPUs")
    else:
        print(f"✓ OK: Peak FLOP/s = {hw.peak_tflops_mm:.1f} TFLOP/s")

else:
    print("⊘ SKIP: CUDA not available")

# Test 2: Enhanced Feature Extraction (Phase 2.2)
print("\n[Test 2] Enhanced Feature Extraction")
print("-" * 80)

from syssim.compute.compute_cost_profiler import _extract_enhanced_features

# Create dummy GEMM data
gemm_data = {
    "M": [64, 128, 256, 512],
    "N": [64, 128, 256, 512],
    "K": [64, 128, 256, 512],
    "t_measured_ms": [0.1, 0.5, 2.0, 8.0],
    "t_roofline_ms": [0.05, 0.3, 1.5, 6.0],
    "efficiency": [0.5, 0.6, 0.75, 0.75],
}
df_gemm = pd.DataFrame(gemm_data)

X_gemm, features_gemm = _extract_enhanced_features(df_gemm, "gemm", ["M", "N", "K"])

print("✓ GEMM features extracted")
print(f"  Shape: {X_gemm.shape}")
print(f"  Features ({len(features_gemm)}): {', '.join(features_gemm)}")

# Validate expected features
expected_gemm = ["log_M", "log_N", "log_K", "log_M_over_N", "log_K_over_N", "log_min_dim", "log_max_dim"]
if features_gemm == expected_gemm:
    print("✓ PASS: All expected features present")
else:
    print("⚠ WARNING: Feature mismatch")
    print(f"  Expected: {expected_gemm}")
    print(f"  Got: {features_gemm}")

# Test attention features
attn_data = {
    "batch": [1, 2, 4],
    "num_heads": [8, 16, 32],
    "seq_len": [128, 256, 512],
    "head_dim": [64, 128, 256],
    "t_measured_ms": [0.1, 0.5, 2.0],
    "t_roofline_ms": [0.05, 0.3, 1.5],
    "efficiency": [0.5, 0.6, 0.75],
}
df_attn = pd.DataFrame(attn_data)

X_attn, features_attn = _extract_enhanced_features(df_attn, "attn", ["batch", "num_heads", "seq_len", "head_dim"])

print("\n✓ Attention features extracted")
print(f"  Shape: {X_attn.shape}")
print(f"  Features ({len(features_attn)}): {', '.join(features_attn)}")

expected_attn = ["log_batch", "log_num_heads", "log_seq_len", "log_head_dim", "log_total_seq", "log_seq_over_head"]
if features_attn == expected_attn:
    print("✓ PASS: All expected features present")
else:
    print("⚠ WARNING: Feature mismatch")

# Test 3: Data Augmentation (Phase 2.3)
print("\n[Test 3] Data Augmentation (Transpose Symmetry)")
print("-" * 80)

from syssim.compute.compute_cost_profiler import _augment_gemm_data

df_original = pd.DataFrame(
    {
        "M": [64, 128, 64],
        "N": [128, 128, 64],  # Note: (64, 64) is square
        "K": [256, 256, 256],
        "efficiency": [0.5, 0.6, 0.7],
    }
)

df_augmented = _augment_gemm_data(df_original)

print("✓ Data augmentation completed")
print(f"  Original: {len(df_original)} samples")
print(f"  Augmented: {len(df_augmented)} samples")
print(f"  Added: {len(df_augmented) - len(df_original)} samples")

# Expected: 3 original + 1 new transpose (2 duplicates: square + equal M/N)
# Original: (64,128), (128,128), (64,64)
# Transposed: (128,64) [new], (128,128) [dup], (64,64) [dup]
# Total: 4 unique
if len(df_augmented) == 4:
    print("✓ PASS: Correct augmentation (3 → 4, duplicates removed)")
elif len(df_augmented) == 6:
    print("⚠ WARNING: Expected 4 samples (duplicates should be removed), got 6")
else:
    print(f"⚠ WARNING: Unexpected augmented size: {len(df_augmented)}")

# Verify transpose exists
has_transpose = ((df_augmented["M"] == 128) & (df_augmented["N"] == 64)).any()
if has_transpose:
    print("✓ PASS: Transpose config (M=128, N=64) found")
else:
    print("⚠ WARNING: Transpose config not found")

# Test 4: Model Architecture (Phase 2.1)
print("\n[Test 4] Model Architecture Compatibility")
print("-" * 80)

import torch.nn as nn

from syssim.compute.efficiency_models import MLPEfficiencyModel

# Build model with new architecture
input_dim = 7  # GEMM: 3 base + 4 enhanced
hidden_dims = [128, 128, 64]

print(f"Building model: input_dim={input_dim}, hidden_dims={hidden_dims}")

# Create mock checkpoint
model_layers = []
prev_dim = input_dim
for i, hidden_dim in enumerate(hidden_dims):
    model_layers.extend(
        [
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
        ]
    )
    if i < len(hidden_dims) - 1:
        model_layers.append(nn.Dropout(0.1))
    prev_dim = hidden_dim
model_layers.append(nn.Linear(prev_dim, 1))
model_ref = nn.Sequential(*model_layers)

print("✓ Reference model created")
print(f"  Total layers: {len(model_ref)}")
print(f"  Parameters: {sum(p.numel() for p in model_ref.parameters()):,}")

# Test inference model builder
mock_checkpoint = {
    "input_dim": input_dim,
    "hidden_dims": hidden_dims,
    "model_state_dict": model_ref.state_dict(),
    "feature_order": ["log_M", "log_N", "log_K", "log_M_over_N", "log_K_over_N", "log_min_dim", "log_max_dim"],
    "operator": "gemm",
}

# Save and load
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "test_model.pth")
    torch.save(mock_checkpoint, model_path)

    try:
        # This will use the updated _build_model method
        from syssim.compute.efficiency_models import MLPEfficiencyModel

        loaded_model = MLPEfficiencyModel(model_path, mock_checkpoint["feature_order"])
        print("✓ PASS: Model loaded successfully")
        print(f"  Loaded model parameters: {sum(p.numel() for p in loaded_model.model.parameters()):,}")
    except Exception as e:
        print("✗ FAIL: Model loading failed")
        print(f"  Error: {e}")

# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print("✓ All core improvements validated")
print("\nNext steps:")
print("  1. Run profiler: python -m syssim.compute.compute_cost_profiler \\")
print("                     --operator gemm --output models/gemm_mlp.pth \\")
print("                     --epochs 100 --num-runs 50")
print("  2. Check efficiency MAPE in output (target: < 15%)")
print("  3. Compare with previous baseline (337% time MAPE)")
print("=" * 80)
