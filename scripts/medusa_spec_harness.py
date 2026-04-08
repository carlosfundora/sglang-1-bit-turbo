#!/usr/bin/env python3
"""
medusa_spec_harness.py — Automatic Medusa speculative decoding tuner.

Phase 1: Offset Probe — determines which head predicts which future offset.
Phase 2: TopK Sweep  — for each head, finds the optimal topk for acceptance.
Phase 3: Draft Depth  — simulates tree verify at different draft depths to
                        find where marginal acceptance drops below overhead.
Phase 4: Dial-In      — binary-searches around the best config to tighten.

Outputs the optimal --medusa-num-heads, --speculative-num-draft-tokens,
and --medusa-topk for the SGLang launch command.

Usage:
  python scripts/medusa_spec_harness.py \
    --base-model <path> --heads-path <path> \
    [--num-heads 3] [--hidden-size 2048] [--vocab-size 151669]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


# ── Head Architecture ───────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.linear = nn.Linear(hs, hs)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaHeads(nn.Module):
    def __init__(self, num_heads, hidden_size, vocab_size):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                ResBlock(hidden_size),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states):
        return [head(hidden_states) for head in self.heads]


# ── Prompt Bank ─────────────────────────────────────────────────────────

PROMPTS = [
    "The capital of France is",
    "Write a Python function to compute the factorial of a number:",
    "Explain how photosynthesis works in simple terms.",
    "The quick brown fox jumps over the",
    "In machine learning, gradient descent is an optimization algorithm that",
    "To make a peanut butter and jelly sandwich, first",
    "The three branches of the United States government are",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "A linked list is a data structure that",
    "The speed of light in a vacuum is approximately",
]

STEPS = 50  # tokens to generate per prompt for ground truth


# ── Data Collection ─────────────────────────────────────────────────────

def collect_hidden_and_tokens(model, tokenizer, prompts, steps, device):
    """Run base model on prompts, collect hidden states + ground truth tokens."""
    samples = []
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.inference_mode():
            # Generate ground truth tokens
            gen_ids = [input_ids]
            out = model(input_ids, output_hidden_states=True)
            past = out.past_key_values
            cur = out.logits[:, -1:, :].argmax(dim=-1)
            gen_ids.append(cur)
            for _ in range(steps):
                out2 = model(cur, past_key_values=past, output_hidden_states=True)
                past = out2.past_key_values
                cur = out2.logits[:, -1:, :].argmax(dim=-1)
                gen_ids.append(cur)

            full_ids = torch.cat(gen_ids, dim=1)  # [1, prompt_len + steps + 1]

            # Rerun full sequence to get all hidden states in one pass
            out_full = model(full_ids, output_hidden_states=True)
            all_hidden = out_full.hidden_states[-1]  # [1, total_len, hidden]

        samples.append({
            "prompt": prompt,
            "prompt_len": input_ids.shape[1],
            "full_ids": full_ids,          # [1, total_len]
            "all_hidden": all_hidden,      # [1, total_len, hidden]
        })
        print(f"  Collected prompt {pi+1}/{len(prompts)}: "
              f"{full_ids.shape[1]} tokens", flush=True)

    return samples


# ── Phase 1: Offset Probe ──────────────────────────────────────────────

def phase1_offset_probe(heads, samples, num_heads, max_offset=6):
    """Test each head against offsets t+1..t+max_offset to find alignment."""
    print("\n" + "=" * 65)
    print("PHASE 1: HEAD OFFSET ALIGNMENT PROBE")
    print("=" * 65)

    hits = [[0] * max_offset for _ in range(num_heads)]
    counts = [[0] * max_offset for _ in range(num_heads)]

    for s in samples:
        all_h = s["all_hidden"]
        full_ids = s["full_ids"]
        total_len = full_ids.shape[1]

        with torch.inference_mode():
            for pos in range(total_len - max_offset - 1):
                h_state = all_h[:, pos, :]  # [1, hidden]
                m_logits = heads(h_state)    # list of [1, vocab]
                for hi in range(num_heads):
                    pred = m_logits[hi].argmax(dim=-1).item()
                    for off in range(max_offset):
                        gt_pos = pos + off + 1
                        if gt_pos < total_len:
                            gt = full_ids[0, gt_pos].item()
                            hits[hi][off] += int(pred == gt)
                            counts[hi][off] += 1

    # Print matrix
    hdr = "      | " + " | ".join(f" t+{o+1} " for o in range(max_offset))
    print(hdr)
    print("-" * len(hdr))

    best_offset = []
    for hi in range(num_heads):
        accs = [hits[hi][o] / max(counts[hi][o], 1) for o in range(max_offset)]
        best_o = max(range(max_offset), key=lambda o: accs[o])
        best_offset.append(best_o)
        row = " | ".join(f"{a*100:5.1f}%" for a in accs)
        print(f"Hd {hi}  | {row}  ← best: t+{best_o+1} ({accs[best_o]*100:.1f}%)")

    # Detect duplicate / clone heads
    clones = []
    for hi in range(num_heads):
        for hj in range(hi + 1, num_heads):
            if best_offset[hi] == best_offset[hj]:
                acc_i = hits[hi][best_offset[hi]] / max(counts[hi][best_offset[hi]], 1)
                acc_j = hits[hj][best_offset[hj]] / max(counts[hj][best_offset[hj]], 1)
                if abs(acc_i - acc_j) < 0.02:
                    clones.append((hi, hj, best_offset[hi]))
                    print(f"\n⚠ Head {hi} and Head {hj} are clones "
                          f"(both predict t+{best_offset[hi]+1}, "
                          f"Δacc={abs(acc_i-acc_j)*100:.2f}%)")

    # Build ordered mapping: sorted by offset
    mapping = sorted(range(num_heads), key=lambda h: best_offset[h])
    print(f"\nOptimal head order: {mapping}")
    print(f"Offset mapping: " +
          ", ".join(f"Head {h}→t+{best_offset[h]+1}" for h in mapping))

    return best_offset, clones, hits, counts


# ── Phase 2: TopK Sweep ────────────────────────────────────────────────

def phase2_topk_sweep(heads, samples, num_heads, best_offset,
                      topk_range=(1, 2, 3, 5, 8, 10, 15, 20)):
    """For each head, measure acceptance at different topk values."""
    print("\n" + "=" * 65)
    print("PHASE 2: TOPK ACCEPTANCE SWEEP")
    print("=" * 65)

    results = {}  # results[head][topk] = acceptance_rate

    for hi in range(num_heads):
        off = best_offset[hi]
        results[hi] = {}

        for topk in topk_range:
            hit = 0
            total = 0
            for s in samples:
                all_h = s["all_hidden"]
                full_ids = s["full_ids"]
                total_len = full_ids.shape[1]
                with torch.inference_mode():
                    for pos in range(total_len - off - 2):
                        h_state = all_h[:, pos, :]
                        logits = heads.heads[hi](h_state)  # [1, vocab]
                        topk_ids = logits.topk(topk, dim=-1).indices.squeeze()
                        gt = full_ids[0, pos + off + 1].item()
                        if topk == 1:
                            hit += int(topk_ids.item() == gt)
                        else:
                            hit += int(gt in topk_ids)
                        total += 1

            acc = hit / max(total, 1)
            results[hi][topk] = acc

        # Print
        row = " | ".join(f"k={k}:{results[hi][k]*100:5.1f}%" for k in topk_range)
        print(f"Head {hi} (t+{off+1}): {row}")

    # Find optimal topk per head (highest acceptance with diminishing returns)
    optimal_topk = {}
    for hi in range(num_heads):
        # Pick the smallest topk where going higher gains < 3% absolute
        sorted_k = sorted(topk_range)
        prev_acc = 0
        best_k = sorted_k[0]
        for k in sorted_k:
            acc = results[hi][k]
            if acc - prev_acc < 0.03 and prev_acc > 0:
                break
            best_k = k
            prev_acc = acc
        optimal_topk[hi] = best_k

    print(f"\nOptimal topk per head: " +
          ", ".join(f"Head {h}→k={optimal_topk[h]}" for h in range(num_heads)))

    return results, optimal_topk


# ── Phase 3: Draft Depth Simulation ────────────────────────────────────

def phase3_draft_depth(heads, samples, num_heads, best_offset, optimal_topk,
                       clones, max_depth=6):
    """Simulate sequential tree verify at increasing draft depths.
    
    For a linear chain tree:
      depth=1 → only head at t+1
      depth=2 → head t+1, if accepted → head t+2
      depth=N → sequential chain

    Returns expected tokens per verify step at each depth.
    """
    print("\n" + "=" * 65)
    print("PHASE 3: DRAFT DEPTH SIMULATION (linear chain)")
    print("=" * 65)

    # Build usable head list: sorted by offset, skip clones
    used_offsets = set()
    usable_heads = []
    for hi in sorted(range(num_heads), key=lambda h: best_offset[h]):
        off = best_offset[hi]
        is_clone = any((hi == c[1] and off == c[2]) for c in clones)
        if off not in used_offsets and not is_clone:
            usable_heads.append((hi, off))
            used_offsets.add(off)

    print(f"Usable heads (after dedup): "
          + ", ".join(f"Head {h} (t+{o+1})" for h, o in usable_heads))

    # For each depth, simulate acceptance chain
    depth_results = []
    for depth in range(1, min(max_depth, len(usable_heads)) + 1):
        chain = usable_heads[:depth]
        total_steps = 0
        total_accepted = 0
        total_bonus = 0  # 1 bonus token always from target verify

        for s in samples:
            all_h = s["all_hidden"]
            full_ids = s["full_ids"]
            total_len = full_ids.shape[1]

            with torch.inference_mode():
                for pos in range(total_len - max(o for _, o in chain) - 2):
                    total_steps += 1
                    accepted = 0
                    for hi, off in chain:
                        k = optimal_topk.get(hi, 1)
                        h_state = all_h[:, pos + accepted, :]
                        logits = heads.heads[hi](h_state)
                        if k == 1:
                            pred = logits.argmax(dim=-1).item()
                            gt = full_ids[0, pos + off + 1].item()
                            if pred == gt:
                                accepted += 1
                            else:
                                break
                        else:
                            topk_ids = logits.topk(k, dim=-1).indices.squeeze()
                            gt = full_ids[0, pos + off + 1].item()
                            if gt in topk_ids:
                                accepted += 1
                            else:
                                break

                    total_accepted += accepted
                    total_bonus += 1  # target verify always gives 1 token

        avg_accepted = total_accepted / max(total_steps, 1)
        avg_total = (total_accepted + total_bonus) / max(total_steps, 1)
        # Cost model: 1 target forward for verify batch of (1 + depth) tokens
        # Baseline: 1 target forward for 1 token
        # Effective speedup: avg_total / cost_ratio
        cost_ratio = 1.0 + depth * 0.15  # each draft adds ~15% overhead
        effective = avg_total / cost_ratio

        depth_results.append({
            "depth": depth,
            "chain": [(h, o) for h, o in chain],
            "avg_accepted_drafts": avg_accepted,
            "avg_tokens_per_step": avg_total,
            "cost_ratio": cost_ratio,
            "effective_speedup": effective,
        })

        heads_str = "+".join(f"H{h}" for h, _ in chain)
        print(f"  depth={depth} [{heads_str}]: "
              f"accepted={avg_accepted:.2f} "
              f"tokens/step={avg_total:.2f} "
              f"cost={cost_ratio:.2f}× "
              f"effective={effective:.2f}×")

    # Find optimal depth
    best_depth = max(depth_results, key=lambda d: d["effective_speedup"])
    print(f"\n✓ Optimal depth: {best_depth['depth']} "
          f"({best_depth['effective_speedup']:.2f}× effective speedup)")

    return depth_results, usable_heads


# ── Phase 4: Dial-In (fine-grained search) ──────────────────────────────

def phase4_dial_in(heads, samples, num_heads, best_offset, usable_heads,
                   depth_results):
    """Fine-tune around the best config with tighter parameter sweeps."""
    print("\n" + "=" * 65)
    print("PHASE 4: DIAL-IN (fine-grained parameter search)")
    print("=" * 65)

    best = max(depth_results, key=lambda d: d["effective_speedup"])
    opt_depth = best["depth"]
    chain = usable_heads[:opt_depth]

    # Test topk more finely around the best values
    fine_topk_range = [1, 2, 3, 4, 5, 6, 8]
    best_config = None
    best_effective = 0

    print(f"Testing topk combinations for depth={opt_depth} "
          f"chain={[f'H{h}' for h,_ in chain]}...")

    # For single head, test topk directly
    if opt_depth == 1:
        hi, off = chain[0]
        for k in fine_topk_range:
            hit = 0
            total = 0
            for s in samples:
                all_h = s["all_hidden"]
                full_ids = s["full_ids"]
                total_len = full_ids.shape[1]
                with torch.inference_mode():
                    for pos in range(total_len - off - 2):
                        h_state = all_h[:, pos, :]
                        logits = heads.heads[hi](h_state)
                        if k == 1:
                            pred = logits.argmax(dim=-1).item()
                            gt = full_ids[0, pos + off + 1].item()
                            hit += int(pred == gt)
                        else:
                            topk_ids = logits.topk(k, dim=-1).indices.squeeze()
                            gt = full_ids[0, pos + off + 1].item()
                            hit += int(gt in topk_ids)
                        total += 1

            acc = hit / max(total, 1)
            # With topk>1, verify batch is larger: k candidates per position
            cost = 1.0 + k * 0.05  # each candidate adds ~5% overhead
            tokens_per_step = 1.0 + acc  # 1 bonus + acc drafts
            effective = tokens_per_step / cost
            marker = " ◄" if effective > best_effective else ""
            if effective > best_effective:
                best_effective = effective
                best_config = {"depth": 1, "topk": [k], "heads": [hi],
                               "acc": acc, "effective": effective}
            print(f"  topk={k}: acc={acc*100:5.1f}% "
                  f"tokens/step={tokens_per_step:.2f} "
                  f"cost={cost:.2f}× effective={effective:.2f}×{marker}")

    else:
        # For multi-head, test each head's topk independently
        # (full grid would be exponential; use greedy per-head)
        final_topk = []
        for idx, (hi, off) in enumerate(chain):
            best_k = 1
            best_k_eff = 0
            for k in fine_topk_range:
                hit = 0
                total = 0
                for s in samples:
                    all_h = s["all_hidden"]
                    full_ids = s["full_ids"]
                    total_len = full_ids.shape[1]
                    with torch.inference_mode():
                        for pos in range(total_len - off - 2):
                            h_state = all_h[:, pos, :]
                            logits = heads.heads[hi](h_state)
                            if k == 1:
                                pred = logits.argmax(dim=-1).item()
                                gt = full_ids[0, pos + off + 1].item()
                                hit += int(pred == gt)
                            else:
                                topk_ids = logits.topk(k, dim=-1).indices.squeeze()
                                gt = full_ids[0, pos + off + 1].item()
                                hit += int(gt in topk_ids)
                            total += 1
                acc = hit / max(total, 1)
                eff = acc / (1 + k * 0.05)
                if eff > best_k_eff:
                    best_k_eff = eff
                    best_k = k
            final_topk.append(best_k)
            print(f"  Head {hi} (t+{off+1}): best topk={best_k}")

        # Simulate final config
        total_steps = 0
        total_tokens = 0
        for s in samples:
            all_h = s["all_hidden"]
            full_ids = s["full_ids"]
            total_len = full_ids.shape[1]
            max_off = max(o for _, o in chain)
            with torch.inference_mode():
                for pos in range(total_len - max_off - 2):
                    total_steps += 1
                    accepted = 0
                    for ci, (hi, off) in enumerate(chain):
                        k = final_topk[ci]
                        h_state = all_h[:, pos + accepted, :]
                        logits = heads.heads[hi](h_state)
                        gt = full_ids[0, pos + off + 1].item()
                        if k == 1:
                            ok = logits.argmax(dim=-1).item() == gt
                        else:
                            ok = gt in logits.topk(k, dim=-1).indices.squeeze()
                        if ok:
                            accepted += 1
                        else:
                            break
                    total_tokens += accepted + 1

        avg_tokens = total_tokens / max(total_steps, 1)
        cost = 1.0 + sum(final_topk) * 0.03
        effective = avg_tokens / cost
        best_config = {
            "depth": opt_depth,
            "topk": final_topk,
            "heads": [h for h, _ in chain],
            "effective": effective,
            "tokens_per_step": avg_tokens,
        }
        print(f"\n  Final: tokens/step={avg_tokens:.2f} "
              f"cost={cost:.2f}× effective={effective:.2f}×")

    return best_config


# ── Report ──────────────────────────────────────────────────────────────

def print_report(best_offset, clones, topk_results, optimal_topk,
                 depth_results, best_config, num_heads, heads_path):
    """Print the final recommendation."""
    print("\n" + "=" * 65)
    print("═══ MEDUSA SPEC HARNESS — FINAL REPORT ═══")
    print("=" * 65)

    print(f"\nHeads path: {heads_path}")
    print(f"Total heads: {num_heads}")

    print("\nHead Alignment:")
    for h in range(num_heads):
        clone_note = ""
        for c in clones:
            if h == c[1]:
                clone_note = f" ⚠ CLONE of Head {c[0]}"
        print(f"  Head {h} → predicts t+{best_offset[h]+1}{clone_note}")

    usable = [h for h in range(num_heads)
              if not any(h == c[1] for c in clones)]
    print(f"\nUsable heads (after dedup): {usable}")

    if clones:
        print(f"\n⚠ {len(clones)} clone head(s) detected — "
              "these waste verify compute.")
        print("  Consider retraining cloned heads for their target offset.")

    if best_config:
        depth = best_config["depth"]
        topk = best_config.get("topk", [1])
        max_topk = max(topk)
        head_list = best_config.get("heads", usable[:depth])

        print(f"\n{'─'*65}")
        print(f"OPTIMAL CONFIGURATION:")
        print(f"  --medusa-num-heads {depth}")
        print(f"  --speculative-num-draft-tokens {depth}")
        print(f"  --medusa-topk {max_topk}")
        print(f"  Effective speedup: {best_config['effective']:.2f}×")
        print(f"{'─'*65}")

        print(f"\nFull launch command:")
        print(f"  python -m sglang.launch_server \\")
        print(f"    --model-path <BASE_MODEL> \\")
        print(f"    --speculative-algorithm MEDUSA \\")
        print(f"    --medusa-model-path {heads_path} \\")
        print(f"    --medusa-num-heads {depth} \\")
        print(f"    --speculative-num-draft-tokens {depth} \\")
        print(f"    --medusa-topk {max_topk} \\")
        print(f"    --mem-fraction-static 0.50 --tp 1 \\")
        print(f"    --disable-cuda-graph --attention-backend torch_native")

    # Verdict
    best_head0_acc = topk_results.get(0, {}).get(1, 0)
    if best_head0_acc >= 0.6:
        verdict = "HEADS_GOOD"
        detail = ("Head 0 has strong accuracy. If SGLang throughput is low, "
                   "check integration code.")
    elif best_head0_acc >= 0.3:
        verdict = "HEADS_MODERATE"
        detail = ("Heads have moderate accuracy. Expect 1.2-1.5× speedup. "
                   "More training would help.")
    else:
        verdict = "HEADS_UNDERTRAINED"
        detail = "Heads need more training for useful speculation."

    print(f"\nVERDICT: {verdict}")
    print(f"  {detail}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Medusa speculative decoding auto-tuner")
    parser.add_argument("--base-model", required=True, help="Base model path")
    parser.add_argument("--heads-path", required=True,
                        help="Path to medusa_lm_head.safetensors")
    parser.add_argument("--num-heads", type=int, default=None,
                        help="Number of heads (auto-detected from config)")
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=8,
                        help="Number of prompts to test (max 10)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Tokens to generate per prompt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    heads_dir = Path(args.heads_path)
    safetensors_file = (heads_dir / "medusa_lm_head.safetensors"
                        if heads_dir.is_dir()
                        else heads_dir)
    config_file = heads_dir / "medusa_config.json" if heads_dir.is_dir() else None

    # Auto-detect from config
    if config_file and config_file.exists():
        cfg = json.loads(config_file.read_text())
        num_heads = args.num_heads or cfg.get("medusa_num_heads", 3)
        hidden_size = args.hidden_size or cfg.get("hidden_size", 2048)
        vocab_size = args.vocab_size or cfg.get("vocab_size", 151669)
        print(f"Config: {num_heads} heads, hidden={hidden_size}, "
              f"vocab={vocab_size}")
    else:
        num_heads = args.num_heads or 3
        hidden_size = args.hidden_size or 2048
        vocab_size = args.vocab_size or 151669

    device = args.device
    dtype = torch.bfloat16

    # Load base model
    print("Loading base model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map=device, trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Load Medusa heads
    print("Loading Medusa heads...")
    heads = MedusaHeads(num_heads, hidden_size, vocab_size)
    heads = heads.to(device=device, dtype=dtype)
    with safe_open(str(safetensors_file), framework="pt") as f:
        state = {k: f.get_tensor(k).to(device=device, dtype=dtype)
                 for k in f.keys()}
    heads.load_state_dict(state)
    heads.eval()
    print(f"  {num_heads} heads loaded")

    # Collect data
    prompts = PROMPTS[:args.num_prompts]
    print(f"\nCollecting hidden states ({len(prompts)} prompts × "
          f"{args.steps} steps)...")
    samples = collect_hidden_and_tokens(model, tokenizer, prompts,
                                        args.steps, device)

    # Phase 1
    best_offset, clones, hits, counts = phase1_offset_probe(
        heads, samples, num_heads)

    # Phase 2
    topk_results, optimal_topk = phase2_topk_sweep(
        heads, samples, num_heads, best_offset)

    # Phase 3
    depth_results, usable_heads = phase3_draft_depth(
        heads, samples, num_heads, best_offset, optimal_topk, clones)

    # Phase 4
    best_config = phase4_dial_in(
        heads, samples, num_heads, best_offset, usable_heads, depth_results)

    # Report
    print_report(best_offset, clones, topk_results, optimal_topk,
                 depth_results, best_config, num_heads, args.heads_path)

    # Save JSON if requested
    if args.json:
        output = {
            "best_offset": best_offset,
            "clones": clones,
            "optimal_topk": {str(k): v for k, v in optimal_topk.items()},
            "depth_results": [
                {k: v for k, v in d.items() if k != "chain"}
                for d in depth_results
            ],
            "best_config": best_config,
        }
        Path(args.json).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
