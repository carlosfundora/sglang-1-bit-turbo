#!/usr/bin/env python3
"""Microbenchmarks for the optional PyO3 utility path."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path


def _python_trim_overlap(existing_text: str, new_chunk: str) -> str:
    max_overlap = 0
    max_possible = min(len(existing_text), len(new_chunk))
    for i in range(max_possible, 0, -1):
        if existing_text.endswith(new_chunk[:i]):
            max_overlap = i
            break
    return new_chunk[max_overlap:]


def _python_sha256_manifest(model_path: Path, filenames: list[str]) -> dict[str, tuple[str, int]]:
    out = {}
    for filename in filenames:
        path = model_path / filename
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(64 * 1024):
                digest.update(chunk)
        out[filename] = (digest.hexdigest(), path.stat().st_size)
    return out


def _load_rust_utils():
    try:
        from sglang.sglang_rust_utils import find_files, sha256_manifest, trim_overlap
    except Exception:
        from sglang_rust_utils import find_files, sha256_manifest, trim_overlap
    return trim_overlap, find_files, sha256_manifest


def _time(label: str, fn, iterations: int = 1):
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.6f}s total, {elapsed / iterations:.9f}s/op")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=int, default=256)
    parser.add_argument("--file-bytes", type=int, default=64 * 1024)
    parser.add_argument("--trim-iters", type=int, default=100_000)
    args = parser.parse_args()

    rust_trim, rust_find_files, rust_sha256_manifest = _load_rust_utils()

    existing = "prefix " + ("abc " * 32) + "shared boundary"
    chunk = "shared boundary and new text"
    _time("python trim_overlap", lambda: _python_trim_overlap(existing, chunk), args.trim_iters)
    _time("rust trim_overlap", lambda: rust_trim(existing, chunk), args.trim_iters)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        payload = b"x" * args.file_bytes
        filenames = []
        for i in range(args.files):
            name = f"shard_{i:05d}.bin"
            (model_dir / name).write_bytes(payload)
            filenames.append(name)

        _time("python sha256_manifest", lambda: _python_sha256_manifest(model_dir, filenames))
        _time("rust sha256_manifest", lambda: rust_sha256_manifest(str(model_dir), filenames, 4))
        _time("rust find_files", lambda: rust_find_files(str(model_dir)))


if __name__ == "__main__":
    main()
