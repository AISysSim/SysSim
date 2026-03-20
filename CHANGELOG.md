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
