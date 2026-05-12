# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial open-source release
- Operator-level tracing via `TorchDispatchMode` + `FakeTensorMode`
- Hybrid roofline model (analytical + ML efficiency predictor) for GEMM, ATTN, MATH ops
- Network simulator with LogGP model, 5 topology types, 8 collective operations
- Device mesh abstraction for hierarchical cluster profiling
- Critical path analysis on multi-stream operator DAGs
- HuggingFace Transformers integration
- Megatron-Core tensor parallel example
- Hardware auto-detection (GH200, H100, A100, V100, A40, RTX 4090, MI250, MI300)
- DOT (Graphviz) and JSON graph export
- **RTX PRO 6000 (Blackwell, GB202)** hardware support in `get_hardware_info()` with FP16 / FP8 / FP4 tensor-unit peaks (3,752 / 7,504 / 15,008 TFLOP/s) and GDDR7 bandwidth (1,792 GB/s)
- **Low-precision profiling** for GEMM (FP16/FP8/FP4) and attention (FP16/FP8) — FP8 via `torch._scaled_mm` and FlashInfer prefill, FP4 via FlashInfer NVFP4 GEMM (`flashinfer.gemm.mm_fp4`)
- **Per-dtype peak FLOPs** on `HardwareInfo` (`peak_tflops_mm_fp8`, `peak_tflops_mm_fp4`) and dtype-aware roofline selection via `HardwareInfo.get_peak_tflops_mm_for_dtype()`
- **Per-`(op, dtype)` efficiency model routing** in `BackendManager`, with automatic fallback to the FP16 model when a per-dtype model is unavailable (e.g., FP8 attention on consumer Blackwell sm_120 with FlashInfer 0.6)
- New examples: `examples/profile_pro6000.py` (low-precision profiling driver) and `examples/train_pro6000_models.py` (per-(op, dtype) XGBoost trainer)
- New tests: `tests/test_pro6000_hw_detect.py`, `tests/test_low_precision_profiling.py`, and dtype-routing coverage in `tests/test_efficiency_models.py`
- Pro 6000 profiling CSVs under `data/profiling/{op}_pro6000_{dtype}_data.csv` (GEMM FP16/FP8/FP4, attention FP16/FP8, RMSNorm FP16, SiLU FP16)
- Engineering report: `docs/docs/2026-04-30-pro6000-low-precision-profiling.md`

### Changed
- `requirements.txt` adds `flashinfer-python>=0.6` (required for FP8/FP4 GEMM and FP8 attention profiling)
