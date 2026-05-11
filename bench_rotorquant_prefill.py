#!/usr/bin/env python3
"""RotorQuant long-context prefill benchmark: pp2048/4096/8192 scaling tests.

Measures prefill throughput (tokens/sec) across different prompt lengths
to evaluate RotorQuant KV cache compression scaling characteristics.

Usage:
    # Start server first with different KV cache modes:
    python -m sglang.launch_server --model-path MODEL --kv-cache-dtype <mode>

    # Then run benchmark:
    python bench_rotorquant_prefill.py [--warmup N] [--runs N]
"""
import argparse
import json
import sys
import time
import urllib.request
from typing import List, Tuple

# Prompt lengths to test (prefill token counts)
PROMPT_LENGTHS = [512, 1024, 2048, 4096, 8192]

# Base URL for SGLang server
URL = "http://localhost:30000"

def get_server_info() -> dict:
    """Get server info including KV cache dtype."""
    try:
        req = urllib.request.Request(f"{URL}/get_model_info")
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt that will be approximately target_tokens long.

    Uses a repeating pattern that tokenizes predictably.
    Most tokenizers give ~1.3 tokens per word, so we aim for ~0.75 words per token.
    """
    # Use a mix of common words that tokenize consistently
    words = "The quick brown fox jumps over the lazy dog. " * 50
    # Estimate: ~4 chars per token average for English text
    chars_needed = target_tokens * 4
    prompt = (words * (chars_needed // len(words) + 1))[:chars_needed]
    return f"Please summarize the following text:\n\n{prompt}\n\nSummary:"

def run_prefill_test(prompt: str, max_tokens: int = 1) -> Tuple[int, float, float]:
    """Run a single prefill test, returns (prompt_tokens, prefill_time, ttft)."""
    data = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=300)
    t1 = time.perf_counter()

    result = json.loads(resp.read())
    prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
    # For single-token generation, total time ≈ TTFT ≈ prefill time
    prefill_time = t1 - t0

    return prompt_tokens, prefill_time, prefill_time

def run_benchmark(warmup: int = 2, runs: int = 3) -> List[dict]:
    """Run the full benchmark suite."""
    results = []

    print(f"\n{'='*70}")
    print("RotorQuant Long-Context Prefill Benchmark")
    print(f"{'='*70}")

    # Get server info
    info = get_server_info()
    kv_dtype = info.get("kv_cache_dtype", "unknown")
    model = info.get("model_path", "unknown")
    print(f"Model: {model}")
    print(f"KV Cache Dtype: {kv_dtype}")
    print(f"Warmup runs: {warmup}, Benchmark runs: {runs}")
    print(f"{'='*70}\n")

    for target_len in PROMPT_LENGTHS:
        prompt = generate_prompt(target_len)

        # Warmup
        print(f"[pp{target_len}] Warming up...", end=" ", flush=True)
        for _ in range(warmup):
            run_prefill_test(prompt)
        print("done")

        # Benchmark runs
        times = []
        actual_tokens = 0
        for i in range(runs):
            tokens, prefill_time, _ = run_prefill_test(prompt)
            times.append(prefill_time)
            actual_tokens = tokens
            tps = tokens / prefill_time
            print(f"  Run {i+1}: {tokens} tok / {prefill_time*1000:.1f}ms = {tps:.1f} tok/s")

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_tps = actual_tokens / avg_time

        result = {
            "target_len": target_len,
            "actual_tokens": actual_tokens,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "avg_tps": avg_tps,
            "kv_dtype": kv_dtype,
        }
        results.append(result)

        print(f"  ── AVG: {actual_tokens} tok / {avg_time*1000:.1f}ms = {avg_tps:.1f} tok/s")
        print()

    return results

def print_summary(results: List[dict]):
    """Print summary table."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Prompt Len':<12} {'Actual Tok':<12} {'Avg (ms)':<12} {'Tok/s':<12}")
    print(f"{'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for r in results:
        print(f"{r['target_len']:<12} {r['actual_tokens']:<12} {r['avg_time_ms']:<12.1f} {r['avg_tps']:<12.1f}")
    print(f"{'='*70}\n")

    # Print JSON for easy parsing
    print("JSON Results:")
    print(json.dumps(results, indent=2))

def main():
    parser = argparse.ArgumentParser(description="RotorQuant prefill benchmark")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per length")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark runs per length")
    args = parser.parse_args()

    try:
        results = run_benchmark(warmup=args.warmup, runs=args.runs)
        print_summary(results)
    except urllib.error.URLError as e:
        print(f"Error: Cannot connect to SGLang server at {URL}")
        print(f"Make sure the server is running: python -m sglang.launch_server ...")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
