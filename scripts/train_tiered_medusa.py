#!/usr/bin/env python3
"""
train_tiered_medusa.py — Train a 7-head Tiered Medusa bundle.

Architecture:
  HEAD 0:   Screen head (broken clone, will be quantized post-training)
  HEAD 1-2: Easy heads  (trained on high-confidence positions only)
  HEAD 3-6: Precision heads (full-spectrum, contrastive against screen)

Collects hidden states + base logits once, then trains all heads sequentially
with the base model freed from VRAM.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from torch.utils.data import Dataset, DataLoader


# ── Head Architecture ───────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.linear = nn.Linear(hs, hs)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class SingleHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.block = ResBlock(hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(self.block(x))


# ── Datasets ────────────────────────────────────────────────────────────

class FullDataset(Dataset):
    """All positions, standard training."""
    def __init__(self, hidden_states, token_ids, offset):
        self.hidden = hidden_states
        self.ids = token_ids
        self.offset = offset
        self.valid = len(token_ids) - offset

    def __len__(self):
        return self.valid

    def __getitem__(self, idx):
        return self.hidden[idx], self.ids[idx + self.offset]


class EasyDataset(Dataset):
    """Only high-confidence positions (base model top-1 prob > threshold)."""
    def __init__(self, hidden_states, token_ids, base_confidence, offset,
                 threshold=0.5):
        self.hidden = hidden_states
        self.ids = token_ids
        self.offset = offset
        # Build index of easy positions
        self.easy_idx = []
        valid = len(token_ids) - offset
        for i in range(valid):
            if base_confidence[i] >= threshold:
                self.easy_idx.append(i)
        self.easy_idx = torch.tensor(self.easy_idx, dtype=torch.long)

    def __len__(self):
        return len(self.easy_idx)

    def __getitem__(self, idx):
        pos = self.easy_idx[idx].item()
        return self.hidden[pos], self.ids[pos + self.offset]


# ── Data Collection ─────────────────────────────────────────────────────

def collect_hidden_and_logits(model, tokenizer, data_path, max_samples=200,
                               max_length=512, device="cuda"):
    """Collect hidden states AND base model confidence (for easy head training)."""
    print(f"Loading dataset from {data_path}...")
    with open(data_path) as f:
        raw = json.load(f)

    texts = []
    for item in raw:
        convs = item.get("conversations", [])
        if convs:
            text = " ".join(c.get("value", "") for c in convs[:2])
            if len(text) > 100:
                texts.append(text[:2000])
        if len(texts) >= max_samples:
            break

    print(f"  {len(texts)} text samples")

    all_hidden = []
    all_ids = []
    all_conf = []  # base model top-1 probability at each position

    model.eval()
    with torch.inference_mode():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt",
                               max_length=max_length, truncation=True)
            input_ids = inputs.input_ids.to(device)
            out = model(input_ids, output_hidden_states=True)

            hidden = out.hidden_states[-1].squeeze(0).cpu()  # [seq, hidden]
            ids = input_ids.squeeze(0).cpu()                 # [seq]
            # Base model confidence: max softmax prob at each position
            probs = F.softmax(out.logits.squeeze(0), dim=-1)  # [seq, vocab]
            conf = probs.max(dim=-1).values.cpu()              # [seq]

            all_hidden.append(hidden)
            all_ids.append(ids)
            all_conf.append(conf)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(texts)}")

    hidden_cat = torch.cat(all_hidden, dim=0)
    ids_cat = torch.cat(all_ids, dim=0)
    conf_cat = torch.cat(all_conf, dim=0)
    print(f"  Total: {hidden_cat.shape[0]} positions")
    print(f"  Easy positions (conf>0.5): "
          f"{(conf_cat > 0.5).sum().item()} "
          f"({(conf_cat > 0.5).float().mean()*100:.1f}%)")
    print(f"  Hard positions (conf<0.5): "
          f"{(conf_cat <= 0.5).sum().item()} "
          f"({(conf_cat <= 0.5).float().mean()*100:.1f}%)")

    return hidden_cat, ids_cat, conf_cat


# ── Training ────────────────────────────────────────────────────────────

def train_single_head(head, dataset, clone_head, args, device,
                      label="", alpha=0.0, steps=500):
    """Train one head. Optional contrastive against clone_head."""
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=args.lr * 0.01)
    ce_fn = nn.CrossEntropyLoss()

    head.train()
    step = 0
    best_loss = float("inf")
    best_state = None

    for epoch in range(20):  # max epochs, step-limited
        ep_loss = 0
        ep_correct = 0
        ep_total = 0

        for hidden, target in loader:
            hidden = hidden.to(device)
            target = target.to(device)

            logits = head(hidden)
            loss = ce_fn(logits, target)

            # Contrastive: diverge from clone
            if alpha > 0 and clone_head is not None:
                with torch.no_grad():
                    clone_probs = F.softmax(clone_head(hidden), dim=-1)
                head_probs = F.softmax(logits, dim=-1)
                agreement = (clone_probs * head_probs).sum(dim=-1).mean()
                loss = loss + alpha * agreement

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                ep_correct += (logits.argmax(-1) == target).sum().item()
                ep_total += target.shape[0]
            ep_loss += loss.item()
            step += 1

            if step % 100 == 0:
                acc = ep_correct / max(ep_total, 1)
                print(f"  [{label}] step {step:4d}/{steps} "
                      f"loss={ep_loss/(step%len(loader) or len(loader)):.4f} "
                      f"acc={acc*100:.1f}%")

            if step >= steps:
                break

        avg = ep_loss / max(len(loader), 1)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if step >= steps:
            break

    if best_state:
        head.load_state_dict(best_state)
    return head


def evaluate_head(head, dataset, device, max_eval=5000):
    """Quick top-1 and top-5 eval."""
    head.eval()
    correct1 = correct5 = total = 0
    with torch.inference_mode():
        for i in range(min(len(dataset), max_eval)):
            h, t = dataset[i]
            logits = head(h.unsqueeze(0).to(device))
            pred = logits.argmax(-1).item()
            top5 = logits.topk(5, dim=-1).indices.squeeze()
            target = t.item() if isinstance(t, torch.Tensor) else t
            correct1 += int(pred == target)
            correct5 += int(target in top5)
            total += 1
    return correct1 / total, correct5 / total


# ── Quantize Screen Head ───────────────────────────────────────────────

def quantize_screen_head(head, device):
    """Aggressively quantize the screen head to INT8 dynamic."""
    head_cpu = head.cpu().float()
    quantized = torch.quantization.quantize_dynamic(
        head_cpu, {nn.Linear}, dtype=torch.qint8)
    return quantized


# ── Save Bundle ─────────────────────────────────────────────────────────

def save_7head_bundle(heads, screen_head, output_dir, config):
    """Save all heads into a single safetensors + config."""
    os.makedirs(output_dir, exist_ok=True)

    state = {}
    # Screen head = heads.0
    screen_sd = screen_head.state_dict()
    for k, v in screen_sd.items():
        # Quantized heads have different key format, dequantize for storage
        if hasattr(v, 'dequantize'):
            v = v.dequantize()
        mapped = k.replace("block.", "0.").replace("proj.", "1.")
        state[f"heads.0.{mapped}"] = v.to(torch.bfloat16)

    # Trained heads = heads.1 through heads.6
    for i, head in enumerate(heads):
        sd = head.state_dict()
        state[f"heads.{i+1}.0.linear.weight"] = sd["block.linear.weight"]
        state[f"heads.{i+1}.0.linear.bias"] = sd["block.linear.bias"]
        state[f"heads.{i+1}.1.weight"] = sd["proj.weight"]

    save_file(state, str(Path(output_dir) / "medusa_lm_head.safetensors"))

    config["medusa_num_heads"] = 7
    config["tiered_architecture"] = {
        "screen_heads": [0],
        "easy_heads": [1, 2],
        "precision_heads": [3, 4, 5, 6],
        "screen_quantized": True,
        "easy_confidence_threshold": 0.5,
    }
    config["head_offsets"] = {
        "0": "screen (negative filter, predicts t+2 at all positions)",
        "1": "easy t+1 (high-confidence positions)",
        "2": "easy t+2 (high-confidence positions)",
        "3": "precision t+1",
        "4": "precision t+2",
        "5": "precision t+3",
        "6": "precision t+4",
    }
    with open(Path(output_dir) / "medusa_config.json", "w") as f:
        json.dump(config, f, indent=2)

    size = (Path(output_dir) / "medusa_lm_head.safetensors").stat().st_size
    print(f"\n  Saved 7-head bundle: {size/1e6:.1f} MB")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--existing-heads", required=True,
                        help="pristine-2head dir")
    parser.add_argument("--clone-heads", required=True,
                        help="3head-production dir (has clone for screen)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--easy-steps", type=int, default=400)
    parser.add_argument("--precision-steps", type=int, default=500)
    parser.add_argument("--easy-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    cfg_path = Path(args.existing_heads) / "medusa_config.json"
    config = json.loads(cfg_path.read_text())
    hs = config["hidden_size"]
    vs = config["vocab_size"]

    # ── Phase 0: Collect data ──────────────────────────────────────────
    print("=" * 65)
    print("PHASE 0: Collecting hidden states + base model confidence")
    print("=" * 65)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device,
        trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True)

    hidden_cat, ids_cat, conf_cat = collect_hidden_and_logits(
        model, tokenizer, args.data_path,
        max_samples=args.max_samples, device=device)

    del model
    torch.cuda.empty_cache()
    print("  Base model freed")

    # Move to device
    hidden_gpu = hidden_cat.to(device=device, dtype=dtype)
    ids_gpu = ids_cat.to(device)
    conf_gpu = conf_cat.to(device)

    # ── Load screen head (clone) ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 1: Screen Head (quantized clone)")
    print("=" * 65)

    screen = SingleHead(hs, vs).to(device=device, dtype=dtype)
    clone_w = load_file(str(Path(args.clone_heads) / "medusa_lm_head.safetensors"))
    # Use head 2 from clone (the broken one)
    screen.block.linear.weight.data = clone_w["heads.2.0.linear.weight"].to(device=device, dtype=dtype)
    screen.block.linear.bias.data = clone_w["heads.2.0.linear.bias"].to(device=device, dtype=dtype)
    screen.proj.weight.data = clone_w["heads.2.1.weight"].to(device=device, dtype=dtype)
    screen.eval()

    # Eval screen at various offsets
    for off in [1, 2, 3, 4]:
        ds = FullDataset(hidden_gpu, ids_gpu, offset=off)
        top1, top5 = evaluate_head(screen, ds, device)
        print(f"  Screen at t+{off}: top-1={top1*100:.1f}% top-5={top5*100:.1f}%")

    # Quantize
    screen_q = quantize_screen_head(screen, device)
    print("  ✓ Screen head quantized to INT8")

    # ── Train Easy Heads ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 2: Easy Heads (high-confidence specialists)")
    print("=" * 65)

    easy_heads = []
    for offset in [1, 2]:
        print(f"\n  Training Easy Head for t+{offset} "
              f"(threshold={args.easy_threshold})...")

        # Init from existing trained head if available
        head = SingleHead(hs, vs).to(device=device, dtype=dtype)
        init_key = f"heads.{offset-1}" if offset <= 2 else None
        if init_key:
            existing_w = load_file(
                str(Path(args.existing_heads) / "medusa_lm_head.safetensors"))
            head.block.linear.weight.data = existing_w[f"{init_key}.0.linear.weight"].to(device=device, dtype=dtype)
            head.block.linear.bias.data = existing_w[f"{init_key}.0.linear.bias"].to(device=device, dtype=dtype)
            head.proj.weight.data = existing_w[f"{init_key}.1.weight"].to(device=device, dtype=dtype)
            print(f"    Init from existing head {offset-1}")

        ds = EasyDataset(hidden_gpu, ids_gpu, conf_gpu, offset=offset,
                         threshold=args.easy_threshold)
        print(f"    Easy positions: {len(ds)} "
              f"({len(ds)/len(ids_gpu)*100:.1f}% of data)")

        head = train_single_head(
            head, ds, screen.to(device=device, dtype=dtype), args, device,
            label=f"Easy-t+{offset}", alpha=args.alpha,
            steps=args.easy_steps)

        # Eval on ALL positions (not just easy)
        full_ds = FullDataset(hidden_gpu, ids_gpu, offset=offset)
        top1_all, top5_all = evaluate_head(head, full_ds, device)
        top1_easy, top5_easy = evaluate_head(head, ds, device)
        print(f"    Result: all={top1_all*100:.1f}% "
              f"easy={top1_easy*100:.1f}% top5={top5_all*100:.1f}%")

        easy_heads.append(head)

    # ── Train Precision Heads ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3: Precision Heads (full-spectrum, contrastive)")
    print("=" * 65)

    precision_heads = []
    # Load ckpt-250 for init (more plastic than final)
    ckpt_path = Path(args.existing_heads).parent / "checkpoint-250"
    ckpt_file = ckpt_path / "medusa_lm_head.safetensors"
    ckpt_w = None
    if ckpt_file.exists():
        ckpt_w = load_file(str(ckpt_file))
        print(f"  Using checkpoint-250 for initialization")

    for offset in [1, 2, 3, 4]:
        print(f"\n  Training Precision Head for t+{offset}...")

        head = SingleHead(hs, vs).to(device=device, dtype=dtype)

        # Init: use existing head for offsets 1-2, ckpt-250 head 1 for 3-4
        if offset <= 2:
            existing_w = load_file(
                str(Path(args.existing_heads) / "medusa_lm_head.safetensors"))
            head.block.linear.weight.data = existing_w[f"heads.{offset-1}.0.linear.weight"].to(device=device, dtype=dtype)
            head.block.linear.bias.data = existing_w[f"heads.{offset-1}.0.linear.bias"].to(device=device, dtype=dtype)
            head.proj.weight.data = existing_w[f"heads.{offset-1}.1.weight"].to(device=device, dtype=dtype)
            print(f"    Init from existing head {offset-1}")
        elif ckpt_w is not None:
            head.block.linear.weight.data = ckpt_w["heads.1.0.linear.weight"].to(device=device, dtype=dtype)
            head.block.linear.bias.data = ckpt_w["heads.1.0.linear.bias"].to(device=device, dtype=dtype)
            head.proj.weight.data = ckpt_w["heads.1.1.weight"].to(device=device, dtype=dtype)
            print(f"    Init from ckpt-250 head 1")

        ds = FullDataset(hidden_gpu, ids_gpu, offset=offset)
        head = train_single_head(
            head, ds, screen.to(device=device, dtype=dtype), args, device,
            label=f"Prec-t+{offset}", alpha=args.alpha,
            steps=args.precision_steps)

        top1, top5 = evaluate_head(head, ds, device)
        print(f"    Result: top-1={top1*100:.1f}% top-5={top5*100:.1f}%")

        precision_heads.append(head)

    # ── Save ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 4: Saving 7-head tiered bundle")
    print("=" * 65)

    all_trained = easy_heads + precision_heads  # 6 trained heads
    save_7head_bundle(all_trained, screen, args.output_dir, config)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("═══ TIERED MEDUSA TRAINING COMPLETE ═══")
    print("=" * 65)
    print(f"""
  Head 0: Screen (INT8 quantized clone, negative filter)
  Head 1: Easy t+1 (high-confidence specialist)
  Head 2: Easy t+2 (high-confidence specialist)
  Head 3: Precision t+1 (full-spectrum, contrastive)
  Head 4: Precision t+2 (full-spectrum, contrastive)
  Head 5: Precision t+3 (full-spectrum, contrastive)
  Head 6: Precision t+4 (full-spectrum, contrastive)

  Output: {args.output_dir}

  Launch:
    --medusa-model-path {args.output_dir}
    --medusa-num-heads 7
    --speculative-num-draft-tokens 4
""")


if __name__ == "__main__":
    main()
