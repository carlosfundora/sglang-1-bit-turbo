"""ROCm-safe linear algebra: route CPU LAPACK ops to the GPU.

The ROCm PyTorch build ships no CPU LAPACK, so `torch.linalg.{qr,svd,eig,eigh,
cholesky,solve,lstsq,inv,...}` on a CPU tensor raises e.g.
  "Calling torch.geqrf on a CPU tensor requires compiling PyTorch with LAPACK".
rocSOLVER provides these on the GPU, so run the op on CUDA/HIP when the inputs are
on CPU and a device is available, then move the result back. This is a no-op (runs
as-is) on non-HIP builds, or when the inputs are already on the GPU.

Usage:
    from sglang.srt.hardware_backend.rocm.linalg import linalg_on_device
    Q, R = linalg_on_device(torch.linalg.qr, G)        # G may be a CPU tensor
    x = linalg_on_device(torch.linalg.solve, A, b)
"""

import torch


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def linalg_on_device(fn, *tensors, **kwargs):
    """Run `fn(*tensors, **kwargs)` on the GPU if its CPU inputs would otherwise
    hit the (absent) ROCm CPU LAPACK; results are returned on the original device."""
    if not (_is_rocm() and torch.cuda.is_available()):
        return fn(*tensors, **kwargs)
    cpu_inputs = [t for t in tensors if isinstance(t, torch.Tensor) and not t.is_cuda]
    if not cpu_inputs:
        return fn(*tensors, **kwargs)

    orig_device = cpu_inputs[0].device
    moved = [
        t.cuda() if isinstance(t, torch.Tensor) and not t.is_cuda else t
        for t in tensors
    ]
    out = fn(*moved, **kwargs)

    if isinstance(out, torch.Tensor):
        return out.to(orig_device)
    if isinstance(out, (tuple, list)):  # incl. torch.return_types.* named tuples
        return tuple(o.to(orig_device) if isinstance(o, torch.Tensor) else o for o in out)
    return out
