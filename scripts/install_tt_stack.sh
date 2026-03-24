#!/bin/bash
set -e
echo "===== Step 1: System check =====" && python3 --version && (tt-smi -l 2>/dev/null || echo "tt-smi -l failed") && ls -la /dev/tenstorrent/
echo "" && echo "===== Step 2: Install PyTorch =====" && pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
echo "" && echo "===== Step 3: Install ttnn =====" && pip install ttnn 2>&1 | tail -5; echo "ttnn exit: $?"
echo "" && echo "===== Step 4: Install pjrt-plugin-tt =====" && pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/ 2>&1 | tail -5; echo "pjrt exit: $?"
echo "" && echo "===== Step 5: Install torch_xla =====" && pip install torch_xla 2>&1 | tail -5; echo "torch_xla exit: $?"
echo "" && echo "===== Step 6: Verify packages =====" && pip list 2>/dev/null | grep -iE "torch|xla|ttnn|pjrt|tenstorrent|tt-"
echo "" && echo "===== Step 7: Device test =====" && python3 -c "
print('--- Test A: ttnn ---')
try:
    import ttnn
    device = ttnn.open_device(device_id=0)
    print(f'ttnn device opened: {device}')
    import torch
    x = torch.rand(32, 32, dtype=torch.float32)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f'Tensor on device: OK shape={xt.shape}')
    ttnn.close_device(device)
    print('ttnn: WORKING')
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
"
echo "" && echo "===== DONE ====="
