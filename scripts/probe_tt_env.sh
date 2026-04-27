#!/bin/bash
# =============================================================================
# Tenstorrent Environment Probe Script
# Works with Wormhole (N300/N150) and Blackhole (P150A/P150B/P100A) hosts.
# Run on the device host and copy the FULL output back.
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

echo "========== PIP PACKAGES (TT / torch / xla) =========="
pip list 2>/dev/null | grep -iE "torch|xla|tenstorrent|tt-|ttnn|metalium|pjrt" \
    || echo "no TT/torch packages found via pip"
pip3 list 2>/dev/null | grep -iE "torch|xla|tenstorrent|tt-|ttnn|metalium|pjrt" \
    || echo "no TT/torch packages found via pip3"
echo ""

echo "========== TENSTORRENT DEVICES =========="
ls -la /dev/tenstorrent* 2>/dev/null || echo "no /dev/tenstorrent* devices"
echo ""

echo "========== TT-SMI (Tenstorrent System Management Interface) =========="
if command -v tt-smi >/dev/null 2>&1; then
    tt-smi -h 2>&1 | head -5 || true
    echo ""
    echo "----- tt-smi -ls (chip listing) -----"
    tt-smi -ls 2>&1 | head -40 || echo "tt-smi -ls failed"
else
    echo "tt-smi not found"
fi
echo ""

echo "========== TT-METAL HOME / FABRIC DESCRIPTORS =========="
echo "TT_METAL_HOME=${TT_METAL_HOME:-<unset>}"
echo "TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-<unset>}"
echo "TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-<unset>}"
if [ -n "$TT_METAL_HOME" ] && [ -d "$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors" ]; then
    echo "Available mesh graph descriptors:"
    ls "$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors" 2>/dev/null | head -20
fi
echo ""

echo "========== ENVIRONMENT VARIABLES (TT-related) =========="
env | grep -iE "TT_|TENSTORRENT|XLA" 2>/dev/null || echo "no TT/XLA env vars"
echo ""

echo "========== PYTHON: ttnn / torch / xla =========="
python3 - <<'PY' 2>&1 || echo "Python probe script failed"
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except ImportError:
    print("PyTorch: NOT INSTALLED")

try:
    import torch_xla, torch_xla.core.xla_model as xm
    print(f"torch_xla: {torch_xla.__version__}")
    try:
        device = xm.xla_device()
        print(f"XLA device: {device}, type: {device.type}")
        import torch
        x = torch.randn(2, 2, device=device)
        xm.mark_step()
        xm.wait_device_ops()
        print(f"XLA tensor on device: OK shape={tuple(x.shape)}")
    except Exception as e:
        print(f"XLA device test failed: {e}")
except ImportError:
    print("torch_xla: NOT INSTALLED")

try:
    import ttnn
    print(f"ttnn: {getattr(ttnn, '__version__', 'unknown')}")
    for dev_id in (0, 1):
        try:
            d = ttnn.open_device(device_id=dev_id)
            print(f"ttnn device {dev_id}: opened -> repr={d!r}")
            print(f"  str(device) = {str(d)!r}")  # syssim auto-detect uses str()
            try:
                arch = ttnn.get_arch_name(d)
                print(f"  ttnn.get_arch_name = {arch}")
            except Exception as e:
                print(f"  ttnn.get_arch_name unavailable: {e}")
            ttnn.close_device(d)
        except Exception as e:
            print(f"ttnn device {dev_id}: FAILED - {e}")
except ImportError:
    print("ttnn: NOT INSTALLED")

try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
except Exception as e:
    print(f"Backend check error: {e}")
PY
echo ""

echo "========== DISK / WORKING DIRECTORY =========="
pwd
df -h . 2>/dev/null | tail -1
echo ""

echo "========== DONE =========="
