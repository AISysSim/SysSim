#!/bin/bash
# =============================================================================
# Install the Python TT stack used by syssim's Tenstorrent profiler.
# Works on Wormhole (N300/N150) and Blackhole (P150A/P150B/P100A) hosts.
#
# On Blackhole hosts the wheel ecosystem may differ (TT-Metalium is the
# preferred low-level entry point). Set TT_METAL_HOME first if you want
# the script to point ttnn at an existing Blackhole-built tree.
# =============================================================================
set -e

echo "===== Step 1: System check ====="
python3 --version
(tt-smi -ls 2>/dev/null || tt-smi -l 2>/dev/null || echo "tt-smi listing failed")
ls -la /dev/tenstorrent/ 2>/dev/null || echo "no /dev/tenstorrent devices visible"
echo "TT_METAL_HOME=${TT_METAL_HOME:-<unset>}"
echo "TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-<unset>}"

echo ""
echo "===== Step 2: Install PyTorch (CPU build) ====="
pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3

echo ""
echo "===== Step 3: Install ttnn ====="
pip install ttnn 2>&1 | tail -5
echo "ttnn exit: $?"

echo ""
echo "===== Step 4: Install pjrt-plugin-tt ====="
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/ 2>&1 | tail -5
echo "pjrt exit: $?"

echo ""
echo "===== Step 5: Install torch_xla ====="
pip install torch_xla 2>&1 | tail -5
echo "torch_xla exit: $?"

echo ""
echo "===== Step 6: Verify packages ====="
pip list 2>/dev/null | grep -iE "torch|xla|ttnn|pjrt|tenstorrent|tt-|metalium"

echo ""
echo "===== Step 7: Device test ====="
python3 - <<'PY'
print('--- Test A: ttnn ---')
try:
    import ttnn
    opened = []
    for dev_id in (0, 1):
        try:
            d = ttnn.open_device(device_id=dev_id)
            opened.append((dev_id, d))
            print(f'ttnn device {dev_id} opened: str={str(d)!r}')
        except Exception as e:
            print(f'ttnn device {dev_id}: FAILED - {e}')
    if opened:
        import torch
        dev_id, d = opened[0]
        x = torch.rand(32, 32, dtype=torch.float32)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
        print(f'Tensor on device {dev_id}: OK shape={xt.shape}')
    for _, d in opened:
        try:
            ttnn.close_device(d)
        except Exception:
            pass
    print('ttnn: WORKING' if opened else 'ttnn: NO DEVICE')
except Exception as e:
    print(f'ttnn: FAILED - {e}')

print()
print('--- Test B: torch_xla ---')
try:
    import torch, torch_xla, torch_xla.core.xla_model as xm
    device = xm.xla_device()
    x = torch.randn(2, 2, device=device)
    xm.mark_step()
    xm.wait_device_ops()
    print(f'XLA device: {device}, tensor shape: {x.shape}')
    print('torch_xla: WORKING')
except Exception as e:
    print(f'torch_xla: FAILED - {e}')
PY

echo ""
echo "===== DONE ====="
