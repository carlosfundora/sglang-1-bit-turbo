#!/usr/bin/env python3
"""Deterministic preflight for hip_zero_patch on ROCm gfx1030 override."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _one_run(strict_repo: Path) -> int:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("SGLANG_STRICT_PROVENANCE", "1")

    import sglang  # noqa: F401
    import torch
    import universal_kv.hip_zero_patch as hip_zero_patch

    print(f"python={sys.executable}")
    print(f"sglang={Path(sys.modules['sglang'].__file__).resolve()}")
    print(f"universal_kv={Path(sys.modules['universal_kv'].__file__).resolve()}")
    print(f"hip_zero_patch={Path(hip_zero_patch.__file__).resolve()}")
    print(f"torch={Path(torch.__file__).resolve()}")
    print(f"arch_list={torch.cuda.get_arch_list()}")

    sglang_path = Path(sys.modules["sglang"].__file__).resolve()
    patch_path = Path(hip_zero_patch.__file__).resolve()
    if not str(sglang_path).startswith(str(strict_repo)):
        raise RuntimeError(f"sglang path outside repo root: {sglang_path}")
    if not str(patch_path).startswith(str(strict_repo)):
        raise RuntimeError(f"hip_zero_patch path outside repo root: {patch_path}")

    if not hip_zero_patch.is_applied():
        raise RuntimeError("hip_zero_patch is not applied")

    x = torch.zeros(10, 10, device="cuda", dtype=torch.float16)
    y = torch.ones(5, 5, device="cuda", dtype=torch.float16)
    z = torch.empty(7, 7, device="cuda", dtype=torch.float16)
    z.fill_(2.0)
    nc = torch.empty((4, 4), device="cuda", dtype=torch.float32).t()
    noncontig_guard_ok = False
    try:
        nc.zero_()
    except RuntimeError as exc:
        noncontig_guard_ok = "non-contiguous CUDA tensors is unsupported" in str(exc)
    if not noncontig_guard_ok:
        raise RuntimeError("Expected non-contiguous zero_ guard RuntimeError")
    torch.cuda.synchronize()

    assert x.shape == (10, 10)
    assert y.shape == (5, 5)
    assert z.shape == (7, 7)
    assert nc.shape == (4, 4)
    return 0


def _spawned_runs(repeats: int, strict_repo: Path) -> int:
    cmd = [sys.executable, __file__, "--single", "--strict-repo", str(strict_repo)]
    for i in range(repeats):
        print(f"[run {i + 1}/{repeats}]")
        proc = subprocess.run(cmd, env=os.environ.copy())
        if proc.returncode != 0:
            print(f"FAILED on run {i + 1}")
            return proc.returncode
    print("All runs passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--strict-repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    if args.single:
        return _one_run(args.strict_repo.resolve())
    return _spawned_runs(args.repeats, args.strict_repo.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
