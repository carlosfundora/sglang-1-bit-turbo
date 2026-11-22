#!/usr/bin/env python3
"""CPU-only radix cache and schedule-policy profiling gate.

This benchmark does not enable a Rust scheduler. It measures whether the Python
prefix-cache scheduling path is hot enough to justify a later Rust core.
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

torch = None
SchedulePolicy = None
InsertParams = None
MatchPrefixParams = None
RadixCache = None
RadixKey = None


def _load_sglang_cache_modules() -> None:
    global InsertParams, MatchPrefixParams, RadixCache, RadixKey, SchedulePolicy, torch

    try:
        import torch as torch_module

        from sglang.srt.managers.schedule_policy import SchedulePolicy as policy_cls
        from sglang.srt.mem_cache.base_prefix_cache import (
            InsertParams as insert_params_cls,
            MatchPrefixParams as match_prefix_params_cls,
        )
        from sglang.srt.mem_cache.radix_cache import (
            RadixCache as radix_cache_cls,
            RadixKey as radix_key_cls,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Unable to import SGLang cache modules. Install the full SGLang "
            f"runtime dependencies before running this benchmark. {exc}"
        ) from exc

    torch = torch_module
    SchedulePolicy = policy_cls
    InsertParams = insert_params_cls
    MatchPrefixParams = match_prefix_params_cls
    RadixCache = radix_cache_cls
    RadixKey = radix_key_cls


def _build_tree(num_prefixes: int, prefix_len: int) -> RadixCache:
    tree = RadixCache.create_simulated()
    for i in range(num_prefixes):
        key = [i, *range(prefix_len - 1)]
        tree.insert(
            InsertParams(
                key=RadixKey(token_ids=key),
                value=torch.arange(prefix_len, dtype=torch.int64),
            )
        )
    return tree


def _build_reqs(num_reqs: int, prefix_len: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            rid=i,
            origin_input_ids=[i, *range(prefix_len - 1), 999],
            output_ids=[],
            extra_key=None,
            prefix_indices=[],
            last_node=None,
            last_host_node=None,
            host_hit_length=0,
            priority=0,
            sampling_params=SimpleNamespace(max_new_tokens=128),
            time_stats=SimpleNamespace(wait_queue_entry_time=float(i)),
            routing_key=str(i % 8),
        )
        for i in range(num_reqs)
    ]


def _time(label: str, fn, iterations: int):
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.6f}s total, {elapsed / iterations:.9f}s/op")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", type=int, default=512)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    _load_sglang_cache_modules()

    tree = _build_tree(args.prefixes, args.prefix_len)
    probe_key = RadixKey(token_ids=[0, *range(args.prefix_len - 1), 999])
    _time(
        "RadixCache.match_prefix",
        lambda: tree.match_prefix(MatchPrefixParams(key=probe_key)),
        args.iters,
    )

    policy = SchedulePolicy(
        "lpm",
        tree_cache=tree,
        enable_hierarchical_cache=False,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
    )

    def calc_priority():
        waiting_queue = _build_reqs(args.requests, args.prefix_len)
        policy.calc_priority(waiting_queue)

    _time("SchedulePolicy.calc_priority(lpm)", calc_priority, args.iters)


if __name__ == "__main__":
    main()
