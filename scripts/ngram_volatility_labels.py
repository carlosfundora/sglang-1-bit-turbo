#!/usr/bin/env python3
"""
ngram_volatility_labels.py — Build n-gram trie from training data and compute
volatility labels for each token position.

Volatility = how unpredictable a position is, measured by:
  1. N-gram entropy (how spread out the n-gram distribution is)
  2. Base model confidence (softmax top-1 probability)
  3. Combined score = w1 * norm(ngram_entropy) + w2 * (1 - base_conf)

Outputs:
  - volatility_scores.pt: tensor of [total_positions] volatility scores
  - ngram_trie.pt: serialized trie for inference-time lane routing
  - stats: distribution of volatile vs confident positions

Usage:
  python scripts/ngram_volatility_labels.py \
    --data-path /path/to/ShareGPT.json \
    --output-dir /path/to/output \
    [--ngram-order 4] [--threshold 0.5]
"""

import argparse
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path

import torch


class NgramTrie:
    """Memory-efficient n-gram trie for volatility scoring."""

    def __init__(self, max_order=4):
        self.max_order = max_order
        # counts[order][context_tuple] = {token: count}
        self.counts = [defaultdict(lambda: defaultdict(int))
                       for _ in range(max_order + 1)]
        self.totals = [defaultdict(int) for _ in range(max_order + 1)]

    def add_sequence(self, token_ids):
        """Add a token sequence to the trie at all n-gram orders."""
        for i in range(len(token_ids) - 1):
            next_tok = token_ids[i + 1]
            for order in range(1, self.max_order + 1):
                start = max(0, i - order + 1)
                context = tuple(token_ids[start:i + 1])
                if len(context) == order:
                    self.counts[order][context][next_tok] += 1
                    self.totals[order][context] += 1

    def entropy(self, context_ids, backoff=True):
        """Compute entropy of next-token distribution given context.

        Uses backoff: tries longest context first, falls back to shorter.
        Returns (entropy, order_used).
        """
        for order in range(min(self.max_order, len(context_ids)), 0, -1):
            context = tuple(context_ids[-order:])
            total = self.totals[order][context]
            if total >= 3:  # minimum count for reliable estimate
                probs = [c / total for c in self.counts[order][context].values()]
                ent = -sum(p * math.log2(p) for p in probs if p > 0)
                return ent, order

            if not backoff:
                break

        # Unknown context: maximum entropy estimate
        return 15.0, 0  # log2(vocab_size) ≈ 17 for 151K vocab

    def confidence(self, context_ids, next_token):
        """P(next_token | context) with backoff."""
        for order in range(min(self.max_order, len(context_ids)), 0, -1):
            context = tuple(context_ids[-order:])
            total = self.totals[order][context]
            if total >= 3:
                return self.counts[order][context].get(next_token, 0) / total
        return 0.0

    def stats(self):
        """Print trie statistics."""
        for order in range(1, self.max_order + 1):
            n_contexts = len(self.counts[order])
            total_counts = sum(self.totals[order].values())
            print(f"  {order}-gram: {n_contexts:,} contexts, "
                  f"{total_counts:,} observations")


def build_trie_from_sharegpt(data_path, tokenizer, max_samples=500,
                              max_length=512, ngram_order=4):
    """Build n-gram trie from ShareGPT conversation data."""
    print(f"Building {ngram_order}-gram trie from {data_path}...")

    with open(data_path) as f:
        raw = json.load(f)

    trie = NgramTrie(max_order=ngram_order)
    n_tokens = 0

    for i, item in enumerate(raw[:max_samples]):
        convs = item.get("conversations", [])
        if not convs:
            continue
        text = " ".join(c.get("value", "") for c in convs[:2])
        if len(text) < 50:
            continue

        token_ids = tokenizer(text, max_length=max_length,
                              truncation=True).input_ids
        trie.add_sequence(token_ids)
        n_tokens += len(token_ids)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1} samples, {n_tokens:,} tokens")

    print(f"  Final: {n_tokens:,} tokens from {min(len(raw), max_samples)} samples")
    trie.stats()
    return trie


def compute_volatility_scores(trie, token_sequences, base_confidences=None,
                               w_ngram=0.5, w_model=0.5):
    """Compute per-position volatility scores.

    Args:
        trie: NgramTrie
        token_sequences: list of token ID lists
        base_confidences: optional list of tensors (base model softmax top-1)
        w_ngram: weight for n-gram entropy component
        w_model: weight for (1 - base_model_confidence) component

    Returns:
        volatility: tensor [total_positions]
        ngram_entropies: tensor [total_positions]
    """
    all_vol = []
    all_ent = []

    for seq_i, tokens in enumerate(token_sequences):
        tokens = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        seq_len = len(tokens)

        for pos in range(seq_len):
            context = tokens[max(0, pos - trie.max_order):pos + 1]
            ent, _order = trie.entropy(context)
            all_ent.append(ent)

            # Combine with base model confidence if available
            if base_confidences is not None and seq_i < len(base_confidences):
                bc = base_confidences[seq_i]
                if pos < len(bc):
                    model_unc = 1.0 - bc[pos].item()
                else:
                    model_unc = 0.5
            else:
                model_unc = 0.5

            # Normalize entropy to [0, 1] range (15 bits ≈ max for 151K vocab)
            norm_ent = min(ent / 15.0, 1.0)
            vol = w_ngram * norm_ent + w_model * model_unc
            all_vol.append(vol)

    return torch.tensor(all_vol), torch.tensor(all_ent)


def analyze_distribution(volatility, threshold=0.5):
    """Print volatility distribution analysis."""
    print(f"\n  Volatility Distribution:")
    print(f"    Total positions: {len(volatility):,}")
    print(f"    Mean: {volatility.mean():.3f}")
    print(f"    Std:  {volatility.std():.3f}")
    print(f"    Min:  {volatility.min():.3f}")
    print(f"    Max:  {volatility.max():.3f}")

    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        n_volatile = (volatility > t).sum().item()
        pct = n_volatile / len(volatility) * 100
        marker = " ◄" if abs(t - threshold) < 0.01 else ""
        print(f"    Volatile (>{t:.1f}): {n_volatile:,} ({pct:.1f}%){marker}")

    # Distribution in deciles
    print(f"\n  Decile breakdown:")
    for d in range(10):
        lo = d * 0.1
        hi = (d + 1) * 0.1
        n = ((volatility >= lo) & (volatility < hi)).sum().item()
        bar = "█" * int(n / len(volatility) * 50)
        print(f"    [{lo:.1f}-{hi:.1f}): {n:5d} {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--base-model", default=None,
                        help="If provided, also compute base model confidence")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ngram-order", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--w-ngram", type=float, default=0.5)
    parser.add_argument("--w-model", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer_path = args.base_model or \
        "/mnt/ai/models/huggingface/models--prism-ml--Bonsai-1.7B-unpacked/snapshots/8d2f546ce33b5572d6fa6df0df0c10cd4948908c"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True)

    # Build trie
    t0 = time.time()
    trie = build_trie_from_sharegpt(
        args.data_path, tokenizer,
        max_samples=args.max_samples,
        ngram_order=args.ngram_order)
    print(f"  Trie built in {time.time()-t0:.1f}s")

    # Tokenize data for volatility scoring
    print("\nTokenizing data for volatility scoring...")
    with open(args.data_path) as f:
        raw = json.load(f)

    token_sequences = []
    for item in raw[:args.max_samples]:
        convs = item.get("conversations", [])
        if convs:
            text = " ".join(c.get("value", "") for c in convs[:2])
            if len(text) > 50:
                ids = tokenizer(text, max_length=512,
                                truncation=True).input_ids
                token_sequences.append(ids)

    # Compute base model confidence if model provided
    base_confs = None
    if args.base_model:
        print("Computing base model confidence...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            device_map=args.device, trust_remote_code=True)
        model.eval()

        base_confs = []
        with torch.inference_mode():
            for i, ids in enumerate(token_sequences):
                inp = torch.tensor([ids], device=args.device)
                out = model(inp)
                probs = torch.softmax(out.logits.squeeze(0), dim=-1)
                conf = probs.max(dim=-1).values.cpu()
                base_confs.append(conf)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(token_sequences)}")

        del model
        torch.cuda.empty_cache()

    # Compute volatility
    print("\nComputing volatility scores...")
    volatility, entropies = compute_volatility_scores(
        trie, token_sequences, base_confs,
        w_ngram=args.w_ngram, w_model=args.w_model)

    analyze_distribution(volatility, args.threshold)

    # Save outputs
    torch.save(volatility, str(output_dir / "volatility_scores.pt"))
    torch.save(entropies, str(output_dir / "ngram_entropies.pt"))
    with open(output_dir / "ngram_trie.pkl", "wb") as f:
        pickle.dump(trie, f)

    # Save config
    config = {
        "ngram_order": args.ngram_order,
        "threshold": args.threshold,
        "w_ngram": args.w_ngram,
        "w_model": args.w_model,
        "n_positions": len(volatility),
        "n_volatile": int((volatility > args.threshold).sum().item()),
        "n_confident": int((volatility <= args.threshold).sum().item()),
    }
    with open(output_dir / "volatility_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  volatility_scores.pt ({len(volatility):,} positions)")
    print(f"  ngram_entropies.pt")
    print(f"  ngram_trie.pkl")
    print(f"  volatility_config.json")


if __name__ == "__main__":
    main()
