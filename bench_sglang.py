#!/usr/bin/env python3
"""Reusable SGLang benchmark: 3 prompts × max_tokens, reports t/s."""
import time, json, urllib.request, sys

URL = "http://localhost:30000/v1/chat/completions"
PROMPTS = [
    "Explain the theory of relativity in detail.",
    "Write a Python function to compute the nth Fibonacci number using dynamic programming.",
    "What are the main differences between TCP and UDP protocols?",
]
MAX_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 256

results = []
for i, prompt in enumerate(PROMPTS):
    data = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=300)
    r = json.loads(resp.read())
    elapsed = time.time() - t0
    tok = r.get("usage", {}).get("completion_tokens", 0)
    tps = tok / elapsed if elapsed > 0 else 0
    text = r["choices"][0]["message"]["content"][:60]
    results.append(tps)
    print(f"  Run {i+1}: {tok} tok / {elapsed:.2f}s = {tps:.1f} t/s | {text}...")

avg = sum(results) / len(results)
print(f"  ── AVG: {avg:.1f} t/s  (min={min(results):.1f}, max={max(results):.1f})")
