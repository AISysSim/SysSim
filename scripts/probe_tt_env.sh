#!/bin/bash
# =============================================================================
# Tenstorrent N300 Environment Probe Script
# Run this on the N300 device via VS Code tunnel terminal.
# Copy the FULL output and paste it back.
# =============================================================================

set -e

echo "========== SYSTEM INFO =========="
echo "hostname: $(hostname)"
echo "uname: $(uname -a)"
echo "date: $(date)"
echo ""

echo "========== PYTHON =========="
which python3 2>/dev/null && python3 --version || echo "python3 not found"
which python 2>/dev/null && python --version || echo "python not found"
echo ""

echo "========== PIP PACKAGES (torch-related) =========="
pip list 2>/dev/null | grep -iE "torch|xla|tenstorrent|tt-" || echo "no torch packages found via pip"
pip3 list 2>/dev/null | grep -iE "torch|xla|tenstorrent|tt-" || echo "no torch packages found via pip3"
echo ""

echo "========== TENSTORRENT DEVICES =========="
ls -la /dev/tenstorrent* 2>/dev/null || echo "no /dev/tenstorrent* devices"
echo ""

echo "========== TT-SMI (Tenstorrent System Management Interface) =========="
which tt-smi 2>/dev/null && tt-smi -h 2>&1 | head -5 || echo "tt-smi not found"
echo ""

echo "========== ENVIRONMENT VARIABLES (TT-related) =========="
env | grep -iE "TT_|TENSTORRENT|XLA" 2>/dev/null || echo "no TT/XLA env vars"
echo ""

echo "========== PYTHON TORCH + XLA TEST =========="
python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
except ImportError:
    print('PyTorch: NOT INSTALLED')

try:
    import torch_xla
    print(f'torch_xla: {torch_xla.__version__}')
except ImportError:
    print('torch_xla: NOT INSTALLED')

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f'XLA device: {device}')
    print(f'Device type: {device.type}')
    
    # Simple test: create tensor on device
    import torch
    x = torch.randn(2, 2, device=device)
    xm.mark_step()
    xm.wait_device_ops()
    print(f'Tensor on device: OK (shape={x.shape})')
except Exception as e:
    print(f'XLA device test failed: {e}')

try:
    # Check for tt-xla specifically
    import tt_xla
    print(f'tt-xla: {tt_xla.__version__}')
except ImportError:
    print('tt-xla: NOT INSTALLED (may be bundled with torch_xla)')

# Check available backends
try:
    import torch
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'MPS available: {hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available()}')
except Exception as e:
    print(f'Backend check error: {e}')
" 2>&1 || echo "Python test script failed"
echo ""

echo "========== DISK / WORKING DIRECTORY =========="
pwd
df -h . 2>/dev/null | tail -1
echo ""

echo "========== DONE =========="
