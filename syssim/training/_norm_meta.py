"""Meta/fake kernels for fused LayerNorm / RMSNorm custom kernels (apex + Transformer Engine).

megatron-core's local layer spec uses fused norm kernels whose custom CUDA ops have no
fake/meta registration, so SysSim's ``FakeTensorMode`` tracer crashes on them:

* **apex** ``FusedLayerNormAffineFunction.apply`` (and RMS / plain variants) call the
  ``fused_layer_norm_cuda`` C kernels directly -> ``RuntimeError: ... data is not allocated yet``
  in the forward pass.
* **Transformer Engine** norm ops import ``layernorm_fwd/bwd`` (and ``rmsnorm_fwd/bwd``) from the
  ``transformer_engine_torch`` C ext; the forward has a fake path but the **backward** kernel runs
  for real -> ``CUDA Error: an illegal memory access`` under fake tensors.

We monkeypatch the affected functions to return correctly-shaped tensors for fake/meta inputs and
delegate to the real kernel otherwise (a no-op during actual training). The big ops (GEMMs via
ColumnParallelLinear/RowParallelLinear, attention via DotProductAttention) are torch-native and
trace fine; only the fused norms need this. Shapes mirror the libraries' own returns.
"""
import importlib

import torch


def _is_fake(t) -> bool:
    from torch._subclasses.fake_tensor import FakeTensor

    return isinstance(t, torch.Tensor) and (t.is_meta or isinstance(t, FakeTensor))


def _rows(input_, normalized_shape) -> int:
    k = len(normalized_shape) if hasattr(normalized_shape, "__len__") else 1
    n1 = 1
    for d in input_.shape[: input_.dim() - k]:
        n1 *= int(d)
    return n1


def _install_apex() -> None:
    """apex legacy autograd.Functions call fused_layer_norm_cuda.<fn> on the module object."""
    try:
        import fused_layer_norm_cuda as C
    except ImportError:
        return
    if getattr(C, "_syssim_meta_installed", False):
        return

    def _stat(input_, normalized_shape):
        return input_.new_empty((_rows(input_, normalized_shape),), dtype=torch.float32)

    def fwd_affine(orig):  # forward_affine(input, nshape, weight, bias, eps) -> (out, mean, invvar)
        def f(input_, normalized_shape, weight, bias, eps, *a):
            if _is_fake(input_):
                s = _stat(input_, normalized_shape)
                return torch.empty_like(input_), s, s.clone()
            return orig(input_, normalized_shape, weight, bias, eps, *a)
        return f

    def rms_fwd_affine(orig):  # rms_forward_affine(input, nshape, weight, eps) -> (out, invvar)
        def f(input_, normalized_shape, weight, eps, *a):
            if _is_fake(input_):
                return torch.empty_like(input_), _stat(input_, normalized_shape)
            return orig(input_, normalized_shape, weight, eps, *a)
        return f

    def fwd(orig):  # forward(input, nshape, eps) -> (out, mean, invvar)
        def f(input_, normalized_shape, eps, *a):
            if _is_fake(input_):
                s = _stat(input_, normalized_shape)
                return torch.empty_like(input_), s, s.clone()
            return orig(input_, normalized_shape, eps, *a)
        return f

    def rms_fwd(orig):  # rms_forward(input, nshape, eps) -> (out, invvar)
        def f(input_, normalized_shape, eps, *a):
            if _is_fake(input_):
                return torch.empty_like(input_), _stat(input_, normalized_shape)
            return orig(input_, normalized_shape, eps, *a)
        return f

    def bwd_affine(orig):  # backward_affine(grad, mean, invvar, input, nshape, weight, bias, ...) -> (gi, gw, gb)
        def f(grad_output, mean, invvar, input_, normalized_shape, weight, bias, *a):
            if _is_fake(grad_output) or _is_fake(input_):
                return torch.empty_like(input_), torch.empty_like(weight), torch.empty_like(bias)
            return orig(grad_output, mean, invvar, input_, normalized_shape, weight, bias, *a)
        return f

    def rms_bwd_affine(orig):  # rms_backward_affine(grad, invvar, input, nshape, weight, ...) -> (gi, gw)
        def f(grad_output, invvar, input_, normalized_shape, weight, *a):
            if _is_fake(grad_output) or _is_fake(input_):
                return torch.empty_like(input_), torch.empty_like(weight)
            return orig(grad_output, invvar, input_, normalized_shape, weight, *a)
        return f

    def bwd(orig):  # backward(grad, mean, invvar, input, nshape, eps, ...) -> gi
        def f(grad_output, mean, invvar, input_, *a):
            if _is_fake(grad_output) or _is_fake(input_):
                return torch.empty_like(input_)
            return orig(grad_output, mean, invvar, input_, *a)
        return f

    def rms_bwd(orig):  # rms_backward(grad, invvar, input, nshape, eps, ...) -> gi
        def f(grad_output, invvar, input_, *a):
            if _is_fake(grad_output) or _is_fake(input_):
                return torch.empty_like(input_)
            return orig(grad_output, invvar, input_, *a)
        return f

    wrappers = {
        "forward_affine": fwd_affine, "forward_affine_mixed_dtypes": fwd_affine,
        "rms_forward_affine": rms_fwd_affine, "rms_forward_affine_mixed_dtypes": rms_fwd_affine,
        "forward": fwd, "rms_forward": rms_fwd,
        "backward_affine": bwd_affine, "rms_backward_affine": rms_bwd_affine,
        "backward": bwd, "rms_backward": rms_bwd,
    }
    for name, wrap in wrappers.items():
        orig = getattr(C, name, None)
        if orig is not None:
            setattr(C, name, wrap(orig))
    C._syssim_meta_installed = True


def _install_te() -> None:
    """TE op modules do `from transformer_engine_torch import ...` -> patch the bound name in each
    op module. Both forward and backward run real CUDA kernels on fake data (the forward error
    surfaces asynchronously at the next sync), so both need meta wrappers. x is pre-viewed to
    (rows, inner_dim), so the per-row stats (means/rstdevs) are shape (rows,)."""

    def _rows1d(x):
        return int(x.shape[0]) if x.dim() >= 1 else 1

    def ln_fwd(orig):  # layernorm_fwd(x, w, b, eps, ...) -> (y, means, rstdevs)
        def f(x, *a, **k):
            if _is_fake(x):
                s = x.new_empty((_rows1d(x),), dtype=torch.float32)
                return torch.empty_like(x), s, s.clone()
            return orig(x, *a, **k)
        return f

    def rms_fwd(orig):  # rmsnorm_fwd(x, w, eps, ...) -> (y, _, rstdevs)
        def f(x, *a, **k):
            if _is_fake(x):
                return torch.empty_like(x), None, x.new_empty((_rows1d(x),), dtype=torch.float32)
            return orig(x, *a, **k)
        return f

    def ln_bwd(orig):  # layernorm_bwd(dy, x, means, rstdevs, w, sm_margin, zcg) -> (dx, dw, db)
        def f(dy, x, means, rstdevs, w, *a, **k):
            if _is_fake(dy) or _is_fake(x):
                return torch.empty_like(x), torch.empty_like(w), torch.empty_like(w)
            return orig(dy, x, means, rstdevs, w, *a, **k)
        return f

    def rms_bwd(orig):  # rmsnorm_bwd(dy, x, rstdevs, w, sm_margin, zcg) -> (dx, dw)
        def f(dy, x, rstdevs, w, *a, **k):
            if _is_fake(dy) or _is_fake(x):
                return torch.empty_like(x), torch.empty_like(w)
            return orig(dy, x, rstdevs, w, *a, **k)
        return f

    targets = {
        "transformer_engine.pytorch.ops.basic.layer_norm": [("layernorm_fwd", ln_fwd), ("layernorm_bwd", ln_bwd)],
        "transformer_engine.pytorch.ops.basic.rmsnorm": [("rmsnorm_fwd", rms_fwd), ("rmsnorm_bwd", rms_bwd)],
    }
    for modname, fns in targets.items():
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if getattr(mod, "_syssim_meta_installed", False):
            continue
        for attr, wrap in fns:
            orig = getattr(mod, attr, None)
            if orig is not None:
                setattr(mod, attr, wrap(orig))
        mod._syssim_meta_installed = True


def install_norm_meta_kernels() -> None:
    """Idempotently make apex + TE fused norm kernels fake/meta-safe for tracing. No-op if absent."""
    _install_apex()
    _install_te()
