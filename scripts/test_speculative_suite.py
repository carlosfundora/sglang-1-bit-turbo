#!/usr/bin/env python3
"""Smoke test for the full speculative decoding suite.

Usage:
    python scripts/test_speculative_suite.py [--model MODEL_PATH] [--draft DRAFT_PATH]

Tests code paths only — does NOT validate acceptance rates (requires trained models).
Each algorithm is launched in a subprocess, sent a probe request, and checked for crash-free execution.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "/home/local/ai/models/local/Bonsai-4B-gguf/Bonsai-4B.gguf"
DEFAULT_DRAFT = "/home/local/ai/models/local/Bonsai-4B-EAGLE3/"
DEFAULT_PORT = 30099

ALGORITHMS = [
    {
        "name": "NGRAM",
        "args": [
            "--speculative-algorithm", "NGRAM",
            "--speculative-ngram-max-trie-depth", "4",
            "--speculative-ngram-match-type", "BFS",
        ],
        "needs_draft": False,
    },
    {
        "name": "EAGLE3",
        "args": [
            "--speculative-algorithm", "EAGLE3",
            "--speculative-eagle-topk", "10",
            "--speculative-num-steps", "4",
        ],
        "needs_draft": True,
    },
    {
        "name": "P_EAGLE",
        "args": [
            "--speculative-algorithm", "P_EAGLE",
            "--speculative-eagle-topk", "10",
            "--speculative-num-steps", "4",
        ],
        "needs_draft": True,
    },
    {
        "name": "P_CASCADE",
        "args": [
            "--speculative-algorithm", "P_CASCADE",
            "--speculative-eagle-topk", "10",
            "--speculative-num-steps", "4",
        ],
        "needs_draft": True,
    },
]

PROBE_PAYLOAD = json.dumps({
    "model": "test",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 16,
    "temperature": 0.0,
}).encode()


def wait_for_server(port: int, timeout: int = 120) -> bool:
    """Poll /health until server is ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            resp = urllib.request.urlopen(req, timeout=3)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def send_probe(port: int) -> dict:
    """Send a chat completion request and return response."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=PROBE_PAYLOAD,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def test_algorithm(algo: dict, model: str, draft: str, port: int) -> dict:
    """Launch server with algorithm, probe, return result."""
    name = algo["name"]
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--tp", "1",
        "--port", str(port),
        "--trust-remote-code",
        "--disable-overlap-schedule",
    ]

    if algo["needs_draft"]:
        cmd.extend(["--speculative-draft-model-path", draft])

    cmd.extend(algo["args"])

    env = os.environ.copy()
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    env["PYTORCH_ROCM_ARCH"] = "gfx1030"
    env["SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE"] = "1"

    result = {"algorithm": name, "status": "UNKNOWN", "error": None, "output": ""}

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )

    try:
        print(f"  PID {proc.pid}, waiting for server...")
        if not wait_for_server(port, timeout=120):
            result["status"] = "TIMEOUT"
            result["error"] = "Server did not become ready within 120s"
            return result

        print(f"  Server ready, sending probe...")
        resp = send_probe(port)
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = resp.get("usage", {})
        print(f"  Response: {content[:80]!r}")
        print(f"  Usage: {usage}")
        result["status"] = "PASS"
        result["output"] = content[:200]

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        print(f"  ERROR: {e}")

    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        print(f"  Server stopped.")

    return result


def main():
    parser = argparse.ArgumentParser(description="Speculative suite smoke test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Target model path")
    parser.add_argument("--draft", default=DEFAULT_DRAFT, help="Draft model path")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--algorithm", type=str, default=None,
                        help="Test only this algorithm (e.g., NGRAM)")
    args = parser.parse_args()

    algos = ALGORITHMS
    if args.algorithm:
        algos = [a for a in ALGORITHMS if a["name"] == args.algorithm.upper()]
        if not algos:
            print(f"Unknown algorithm: {args.algorithm}")
            sys.exit(1)

    results = []
    for algo in algos:
        r = test_algorithm(algo, args.model, args.draft, args.port)
        results.append(r)
        time.sleep(3)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status_emoji = {"PASS": "✅", "FAIL": "❌", "TIMEOUT": "⏱️"}.get(r["status"], "❓")
        err = f" ({r['error']})" if r["error"] else ""
        print(f"  {status_emoji} {r['algorithm']}: {r['status']}{err}")

    failed = sum(1 for r in results if r["status"] != "PASS")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
