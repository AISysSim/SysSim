"""Context that redirects CUDA tensor allocations to the meta device.

PyTorch tensor factory calls like `torch.empty(..., device="cuda:0")` and
`tensor.cuda()` normally allocate real GPU memory. For the simulator we want
the model graph (parameter shapes, dtypes, device labels) without any real
allocation. This context monkey-patches the relevant entry points so any
cuda-targeted allocation lands on the meta device instead.

Use as a `with`-statement wrapper around model construction.
"""

from __future__ import annotations

import contextlib

import torch


_FACTORY_NAMES = (
    "empty", "zeros", "ones", "full",
    "rand", "randn", "randint", "arange",
    "tensor",
)


def _is_cuda_device(device) -> bool:
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, int):
        # torch.cuda.current_device() returns int; bare integer means cuda index
        return True
    return "cuda" in str(device)


@contextlib.contextmanager
def redirect_cuda_alloc_to_meta():
    """Within this context, any tensor allocation targeting cuda goes to meta.

    Patches:
      - top-level factory functions on `torch` (empty, zeros, ones, ...) so a
        `device=cuda:0` (or `device=0`) kwarg is rewritten to `device="meta"`.
      - `torch.Tensor.cuda(...)` so it returns a meta copy instead of moving
        to a real GPU.
    """
    saved_factories = {name: getattr(torch, name) for name in _FACTORY_NAMES}
    saved_tensor_cuda = torch.Tensor.cuda

    def _wrap_factory(orig):
        def wrapper(*args, **kwargs):
            if _is_cuda_device(kwargs.get("device")):
                kwargs["device"] = "meta"
            return orig(*args, **kwargs)
        return wrapper

    def _meta_cuda(self, device=None, non_blocking=False, memory_format=None):
        return self.to("meta")

    for name, orig in saved_factories.items():
        setattr(torch, name, _wrap_factory(orig))
    torch.Tensor.cuda = _meta_cuda
    try:
        yield
    finally:
        for name, orig in saved_factories.items():
            setattr(torch, name, orig)
        torch.Tensor.cuda = saved_tensor_cuda
