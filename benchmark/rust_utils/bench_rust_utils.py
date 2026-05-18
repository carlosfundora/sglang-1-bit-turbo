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


def _python_hicache_page_hashes(
    token_ids: list[int | tuple[int, int]], page_size: int
) -> list[str]:
    out = []
    parent_hash = None
    for start in range(0, len(token_ids), page_size):
        digest = hashlib.sha256()
        if parent_hash:
            digest.update(bytes.fromhex(parent_hash))
        for token in token_ids[start : start + page_size]:
            if isinstance(token, tuple):
                for elem in token:
                    digest.update(elem.to_bytes(4, byteorder="little", signed=False))
            else:
                digest.update(token.to_bytes(4, byteorder="little", signed=False))
        parent_hash = digest.hexdigest()
        out.append(parent_hash)
    return out


def _python_saguaro_prefix_hash(tokens: list[int], window: int) -> str:
    suffix = tokens[-window:] if len(tokens) >= window else tokens
    raw = ",".join(str(t) for t in suffix)
    return hashlib.sha256(raw.encode()).hexdigest()


class _SamplingParams:
    def __init__(self, i: int):
        self.temperature = 0.0 if i % 4 == 0 else 0.7
        self.top_p = 1.0 if i % 3 == 0 else 0.9
        self.top_k = 1 if i % 5 == 0 else 64
        self.min_p = 0.0 if i % 7 else 0.05
        self.sampling_seed = None if i % 2 else i


class _Req:
    def __init__(self, i: int):
        self.sampling_params = _SamplingParams(i)
        self.custom_logit_processor = None


def _python_pack_sampling_params(reqs: list[_Req], top_k_all: int):
    return (
        [r.sampling_params.temperature for r in reqs],
        [r.sampling_params.top_p for r in reqs],
        [r.sampling_params.top_k for r in reqs],
        [r.sampling_params.min_p for r in reqs],
        [
            r.sampling_params.sampling_seed
            if r.sampling_params.sampling_seed is not None
            else 42
            for r in reqs
        ],
        all(r.sampling_params.top_k <= 1 for r in reqs),
        any(r.sampling_params.top_p != 1.0 for r in reqs),
        any(r.sampling_params.top_k != top_k_all for r in reqs),
        any(r.sampling_params.min_p > 0 for r in reqs),
        any(r.custom_logit_processor for r in reqs),
    )


def _load_rust_utils():
    try:
        from sglang.sglang_rust_utils import (
            find_files,
            hicache_page_hashes,
            pack_sampling_params,
            saguaro_prefix_hash,
            sha256_manifest,
            trim_overlap,
        )
    except Exception:
        from sglang_rust_utils import (
            find_files,
            hicache_page_hashes,
            pack_sampling_params,
            saguaro_prefix_hash,
            sha256_manifest,
            trim_overlap,
        )
    return (
        trim_overlap,
        find_files,
        sha256_manifest,
        hicache_page_hashes,
        saguaro_prefix_hash,
        pack_sampling_params,
    )


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

    (
        rust_trim,
        rust_find_files,
        rust_sha256_manifest,
        rust_hicache_page_hashes,
        rust_saguaro_prefix_hash,
        rust_pack_sampling_params,
    ) = _load_rust_utils()

    existing = "prefix " + ("abc " * 32) + "shared boundary"
    chunk = "shared boundary and new text"
    _time("python trim_overlap", lambda: _python_trim_overlap(existing, chunk), args.trim_iters)
    _time("rust trim_overlap", lambda: rust_trim(existing, chunk), args.trim_iters)

    tokens = list(range(4096))
    _time(
        "python hicache_page_hashes",
        lambda: _python_hicache_page_hashes(tokens, 64),
        1_000,
    )
    _time("rust hicache_page_hashes", lambda: rust_hicache_page_hashes(tokens, 64), 1_000)
    _time(
        "python saguaro_prefix_hash",
        lambda: _python_saguaro_prefix_hash(tokens, 32),
        100_000,
    )
    _time(
        "rust saguaro_prefix_hash",
        lambda: rust_saguaro_prefix_hash(tokens, 32),
        100_000,
    )
    reqs = [_Req(i) for i in range(512)]
    _time(
        "python pack_sampling_params",
        lambda: _python_pack_sampling_params(reqs, 1 << 30),
        10_000,
    )
    _time(
        "rust pack_sampling_params",
        lambda: rust_pack_sampling_params(reqs, 1 << 30, True, True),
        10_000,
    )

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
