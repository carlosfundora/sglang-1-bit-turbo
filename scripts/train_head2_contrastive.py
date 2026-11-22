#!/usr/bin/env python3
"""
train_head2_contrastive.py — Fine-tune a single Medusa head for t+3 offset
using contrastive loss against the clone (which predicts t+2).

Strategy:
  1. Initialize head 2 from checkpoint-250's head 1 (mid-training, more plastic)
  2. Train on t+3 offset with standard CE loss
  3. Add contrastive penalty: pushes head 2 AWAY from clone's t+2 predictions
  4. Save as proper 3-head safetensors

Loss = CE(head2_logits, label_t+3) + α * KL(head2_probs ‖ clone_probs)
       ↑ learn correct t+3            ↑ diverge from clone's t+2

Usage:
  python scripts/train_head2_contrastive.py \
    --base-model /path/to/Bonsai-1.7B \
    --existing-heads /path/to/pristine-2head \
    --clone-heads /path/to/3head-production \
    --output-dir /path/to/output \
    --data-path /path/to/ShareGPT.json \
    [--epochs 2] [--lr 1e-3] [--alpha 0.1] [--steps 500]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from torch.utils.data import Dataset, DataLoader


# ── Head Architecture (must match inference) ────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.linear = nn.Linear(hs, hs)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class SingleHead(nn.Module):
    """One Medusa head: ResBlock → Linear(hidden→vocab)."""
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.block = ResBlock(hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(self.block(x))


# ── Dataset: collect hidden states + ground truth ───────────────────────

class HiddenStateDataset(Dataset):
    """Pre-computed hidden states from base model with ground truth tokens."""

    def __init__(self, hidden_states, token_ids, offset=3):
        """
        hidden_states: [total_tokens, hidden_size] — last layer hidden
        token_ids: [total_tokens] — ground truth token IDs
        offset: which future token this head should predict (3 = t+3)
        """
        self.hidden_states = hidden_states
        self.token_ids = token_ids
        self.offset = offset
        # Valid range: position i predicts token at i+offset
        self.valid_len = len(token_ids) - offset

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        return (
            self.hidden_states[idx],          # input hidden state at position t
            self.token_ids[idx + self.offset], # target token at t+offset
        )


def collect_hidden_states(model, tokenizer, data_path, max_samples=200,
                          max_length=512, device="cuda"):
    """Run base model on ShareGPT data, collect hidden states."""
    print(f"Loading dataset from {data_path}...")

    with open(data_path) as f:
        raw = json.load(f)

    # Extract conversation texts
    texts = []
    for item in raw:
        convs = item.get("conversations", [])
        if convs:
            text = " ".join(c.get("value", "") for c in convs[:2])
            if len(text) > 100:
                texts.append(text[:2000])  # cap per-sample length
        if len(texts) >= max_samples:
            break

    print(f"  Collected {len(texts)} text samples")

    all_hidden = []
    all_ids = []

    model.eval()
    with torch.inference_mode():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt",
                               max_length=max_length, truncation=True)
            input_ids = inputs.input_ids.to(device)

            out = model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1].squeeze(0)  # [seq_len, hidden]
            ids = input_ids.squeeze(0)                  # [seq_len]

            all_hidden.append(hidden.cpu())
            all_ids.append(ids.cpu())

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(texts)} samples")

    # Concatenate all
    hidden_cat = torch.cat(all_hidden, dim=0)  # [total, hidden]
    ids_cat = torch.cat(all_ids, dim=0)        # [total]
    print(f"  Total: {hidden_cat.shape[0]} hidden states")

    return hidden_cat, ids_cat


# ── Training Loop ──────────────────────────────────────────────────────

def train_head(head, clone_head, dataset, args, device):
    """Train head 2 with contrastive loss against clone."""

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    total_steps = args.steps or (args.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    ce_loss_fn = nn.CrossEntropyLoss()

    head.train()
    clone_head.eval()

    step = 0
    best_loss = float("inf")
    best_state = None

    print(f"\nTraining head for t+3 offset")
    print(f"  Steps: {total_steps}, LR: {args.lr}, α(contrastive): {args.alpha}")
    print(f"  Batch size: {args.batch_size}, Dataset: {len(dataset)} samples")
    print()

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_ce = 0
        epoch_contra = 0
        epoch_correct = 0
        epoch_total = 0

        for batch_i, (hidden, target) in enumerate(loader):
            hidden = hidden.to(device)
            target = target.to(device)

            # Forward through new head
            logits = head(hidden)  # [batch, vocab]

            # CE loss: predict the correct t+3 token
            loss_ce = ce_loss_fn(logits, target)

            # Contrastive loss: diverge from clone's predictions
            loss_contra = torch.tensor(0.0, device=device)
            if args.alpha > 0:
                with torch.no_grad():
                    clone_logits = clone_head(hidden)  # [batch, vocab]
                    clone_probs = F.softmax(clone_logits, dim=-1)

                head_log_probs = F.log_softmax(logits, dim=-1)
                # Reverse KL: we want head to DIVERGE from clone
                # Use negative KL = -KL(clone ‖ head) as a penalty
                # = -sum(clone * log(clone/head))
                # = -sum(clone * (log_clone - log_head))
                # = sum(clone * log_head) - sum(clone * log_clone)
                # Maximizing this pushes head probs where clone probs are high
                # We want the OPPOSITE: minimize agreement
                # So: loss_contra = sum(clone_probs * head_log_probs)
                # = cross-entropy of head w.r.t. clone as "labels"
                # Minimizing this would match clone — we MAXIMIZE it (negate)
                # But we want gradient to push AWAY, so:
                # loss_contra = -KL(head ‖ uniform) + KL(head ‖ clone)
                # Simpler: just use dot product of probs as agreement penalty
                loss_contra = (clone_probs * F.softmax(logits, dim=-1)).sum(dim=-1).mean()

            loss = loss_ce + args.alpha * loss_contra

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == target).sum().item()
                epoch_total += target.shape[0]

            epoch_loss += loss.item()
            epoch_ce += loss_ce.item()
            epoch_contra += loss_contra.item()
            step += 1

            if step % 50 == 0:
                avg_loss = epoch_loss / (batch_i + 1)
                avg_ce = epoch_ce / (batch_i + 1)
                avg_contra = epoch_contra / (batch_i + 1)
                acc = epoch_correct / max(epoch_total, 1)
                lr = scheduler.get_last_lr()[0]
                print(f"  step {step:4d} | loss={avg_loss:.4f} "
                      f"(ce={avg_ce:.4f} contra={avg_contra:.4f}) "
                      f"| acc={acc*100:.1f}% | lr={lr:.2e}")

            if step >= total_steps:
                break

        # Epoch summary
        avg_loss = epoch_loss / max(len(loader), 1)
        acc = epoch_correct / max(epoch_total, 1)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} acc={acc*100:.1f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if step >= total_steps:
            break

    if best_state:
        head.load_state_dict(best_state)
    return head


# ── Save 3-head bundle ─────────────────────────────────────────────────

def save_3head_bundle(existing_path, new_head, output_dir, config):
    """Combine existing 2 heads + new head 2 into 3-head safetensors."""
    os.makedirs(output_dir, exist_ok=True)

    # Load existing 2 heads
    existing = load_file(str(Path(existing_path) / "medusa_lm_head.safetensors"))

    # Build 3-head state dict
    state = {}
    for k, v in existing.items():
        state[k] = v  # heads.0.* and heads.1.*

    # Add new head as heads.2.*
    new_sd = new_head.state_dict()
    state["heads.2.0.linear.weight"] = new_sd["block.linear.weight"]
    state["heads.2.0.linear.bias"] = new_sd["block.linear.bias"]
    state["heads.2.1.weight"] = new_sd["proj.weight"]

    save_file(state, str(Path(output_dir) / "medusa_lm_head.safetensors"))

    # Update config
    config["medusa_num_heads"] = 3
    config["head_2_note"] = ("Contrastive-trained for t+3 offset. "
                              "Init from ckpt-250 head 1, diverged from clone.")
    with open(Path(output_dir) / "medusa_config.json", "w") as f:
        json.dump(config, f, indent=2)

    size = (Path(output_dir) / "medusa_lm_head.safetensors").stat().st_size
    print(f"\nSaved 3-head bundle to {output_dir} ({size/1e6:.1f} MB)")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--existing-heads", required=True,
                        help="Path to pristine 2-head dir")
    parser.add_argument("--clone-heads", required=True,
                        help="Path to 3-head-production (has the clone)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", required=True,
                        help="ShareGPT JSON training data")
    parser.add_argument("--init-from", default="checkpoint-250",
                        choices=["checkpoint-250", "checkpoint-500", "random"],
                        help="Initialize head 2 weights from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=None,
                        help="Max steps (overrides epochs)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Contrastive loss weight (0=no contrastive)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    # Load config
    cfg_path = Path(args.existing_heads) / "medusa_config.json"
    config = json.loads(cfg_path.read_text())
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]

    # Load base model for hidden state collection
    print("=" * 65)
    print("PHASE 0: Collecting hidden states from base model")
    print("=" * 65)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device,
        trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)

    hidden_cat, ids_cat = collect_hidden_states(
        model, tokenizer, args.data_path,
        max_samples=args.max_samples, device=device)

    # Free base model VRAM
    del model
    torch.cuda.empty_cache()
    print("  Base model freed from VRAM")

    # Build dataset for t+3 offset
    dataset = HiddenStateDataset(hidden_cat.to(dtype), ids_cat, offset=3)
    print(f"  Dataset: {len(dataset)} training pairs for t+3")

    # Initialize new head
    print("\n" + "=" * 65)
    print("PHASE 1: Initializing head 2")
    print("=" * 65)

    new_head = SingleHead(hidden_size, vocab_size).to(device=device, dtype=dtype)

    if args.init_from == "random":
        print("  Random initialization")
    else:
        # Load from checkpoint's head 1 (t+2 predictor — closest to t+3)
        ckpt_dir = Path(args.existing_heads).parent / args.init_from
        ckpt_file = ckpt_dir / "medusa_lm_head.safetensors"
        if ckpt_file.exists():
            ckpt_weights = load_file(str(ckpt_file))
            # Map head 1 weights → new head
            new_head.block.linear.weight.data = ckpt_weights["heads.1.0.linear.weight"].to(device=device, dtype=dtype)
            new_head.block.linear.bias.data = ckpt_weights["heads.1.0.linear.bias"].to(device=device, dtype=dtype)
            new_head.proj.weight.data = ckpt_weights["heads.1.1.weight"].to(device=device, dtype=dtype)
            print(f"  Initialized from {args.init_from} head 1 (t+2 predictor)")
        else:
            print(f"  ⚠ {ckpt_file} not found, using random init")

    # Load clone head (for contrastive loss)
    clone_head = SingleHead(hidden_size, vocab_size).to(device=device, dtype=dtype)
    clone_weights = load_file(str(Path(args.clone_heads) / "medusa_lm_head.safetensors"))
    clone_head.block.linear.weight.data = clone_weights["heads.2.0.linear.weight"].to(device=device, dtype=dtype)
    clone_head.block.linear.bias.data = clone_weights["heads.2.0.linear.bias"].to(device=device, dtype=dtype)
    clone_head.proj.weight.data = clone_weights["heads.2.1.weight"].to(device=device, dtype=dtype)
    clone_head.eval()
    print(f"  Clone head loaded (contrastive target, α={args.alpha})")

    # Move hidden states to device
    hidden_cat = hidden_cat.to(device=device, dtype=dtype)
    dataset = HiddenStateDataset(hidden_cat, ids_cat.to(device), offset=3)

    # Train
    print("\n" + "=" * 65)
    print("PHASE 2: Training with contrastive divergence")
    print("=" * 65)

    t0 = time.time()
    new_head = train_head(new_head, clone_head, dataset, args, device)
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")

    # Quick eval
    print("\n" + "=" * 65)
    print("PHASE 3: Evaluation")
    print("=" * 65)

    new_head.eval()
    correct = 0
    total = 0
    correct_top5 = 0
    with torch.inference_mode():
        for i in range(min(len(dataset), 5000)):
            h, target = dataset[i]
            logits = new_head(h.unsqueeze(0))
            pred = logits.argmax(dim=-1).item()
            top5 = logits.topk(5, dim=-1).indices.squeeze()
            correct += int(pred == target.item())
            correct_top5 += int(target.item() in top5)
            total += 1

    print(f"  Head 2 (t+3) accuracy:")
    print(f"    Top-1: {correct/total*100:.1f}%")
    print(f"    Top-5: {correct_top5/total*100:.1f}%")

    # Also test what clone gets on same data
    clone_correct = 0
    with torch.inference_mode():
        for i in range(min(len(dataset), 5000)):
            h, target = dataset[i]
            logits = clone_head(h.unsqueeze(0))
            pred = logits.argmax(dim=-1).item()
            clone_correct += int(pred == target.item())

    print(f"  Clone (untrained) at t+3: {clone_correct/total*100:.1f}%")
    print(f"  Improvement: +{(correct-clone_correct)/total*100:.1f}% absolute")

    # Save
    print("\n" + "=" * 65)
    print("PHASE 4: Saving 3-head bundle")
    print("=" * 65)
    save_3head_bundle(args.existing_heads, new_head, args.output_dir, config)

    print("\n✓ Done! Launch with:")
    print(f"  --medusa-model-path {args.output_dir} "
          f"--medusa-num-heads 3 --speculative-num-draft-tokens 3")


if __name__ == "__main__":
    main()
