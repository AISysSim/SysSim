# SysSim official container image — Isambard-AI Phase 2 (GH200, aarch64).
#
# Derived from the NVIDIA NGC PyTorch container (custom CUDA build of PyTorch +
# numpy + CUDA toolkit). Installs the full SysSim dependency stack so the image
# covers everything: CPU unit tests, the profiler, AND the Megatron tracer /
# training-simulation path on a GPU node.
#
#   * torch is pinned to the base build via a constraints file so no transitive
#     dependency can swap the NGC build (a torch/CUDA mismatch breaks the
#     forward-compat path at runtime).
#   * xgboost is intentionally omitted (the predictor replaced the efficiency
#     stack); lightgbm + pyarrow are added for the predictor / profiler.
#   * The heavy training deps (megatron-core, megatron-bridge, flashinfer-python)
#     pull source builds (e.g. mamba-ssm) whose setup.py does `import torch`.
#     Those fail under pip's default build isolation, so they are installed with
#     --no-build-isolation against the ambient NGC torch; TORCH_CUDA_ARCH_LIST
#     targets GH200/H100 (sm_90) so CUDA extensions build without a GPU present.
#
# The project itself is NOT installed — mount the repo and set PYTHONPATH=<repo>
# at `podman-hpc run` time so source edits stay live.
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Freeze the base image's exact torch so pip can never upgrade/downgrade it.
RUN python -c "import torch; print('torch==' + torch.__version__)" > /tmp/torch-constraint.txt \
 && cat /tmp/torch-constraint.txt

# Lightweight, torch-independent deps (aarch64 wheels; fast).
RUN pip install --no-cache-dir -c /tmp/torch-constraint.txt \
        pytest \
        lightgbm \
        pyarrow \
        pandas \
        "pyyaml>=6.0" \
        scikit-learn

# Build tools required by --no-build-isolation source builds (mamba-ssm via megatron-bridge).
RUN pip install --no-cache-dir -c /tmp/torch-constraint.txt ninja packaging setuptools wheel

# Heavy training / tracer stack. --no-build-isolation so source builds see the
# ambient NGC torch; sm_90 target so CUDA extensions compile without a build GPU.
# MAX_JOBS caps parallel nvcc so the source build (mamba-ssm) uses the
# node's many cores without OOMing (each nvcc is multi-GB).
ENV TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=16
RUN pip install --no-cache-dir --no-build-isolation -c /tmp/torch-constraint.txt \
        transformers \
        megatron-core \
        megatron-bridge

# Build-time sanity: torch is still the NGC build and the CPU-safe deps import.
# (No GPU at build, so megatron CUDA paths are exercised at runtime.)
RUN python -c "import torch, pytest, lightgbm, pyarrow, pandas, yaml, sklearn, transformers; \
print('OK torch', torch.__version__, '| lightgbm', lightgbm.__version__)"
