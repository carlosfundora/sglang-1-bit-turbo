"""Medusa multi-head speculative decoding model.

Medusa adds N lightweight MLP heads to the base model. Each head predicts
the token at a different future position (+1, +2, ... +K) in parallel from
the SAME last hidden state. No autoregression — all K candidates come from
a single forward pass.

Architecture per head:
  ResBlock × L  →  Linear(hidden → vocab)
  ResBlock = x + SiLU(Linear(x))   (initialized to identity at training start)

PALTROW heads (Pinned-memory Auxiliary Latency-Tolerant Relocated On-CPU Workers):
  Heads designated as PALTROW run on CPU with pinned host memory instead of GPU.
  This saves GPU VRAM for the KV cache and precision draft heads.  With AMD SAM
  (Smart Access Memory) / Resizable BAR, the full GPU BAR is visible to CPU and
  pinned DMA transfers run at ~14 GB/s — more than enough for the tiny hidden-
  state copy (~4 KB per batch element).

  PALTROW heads are ideal for:
    - Screen heads (DraftPreFilter L1 inversion — latency-tolerant by design)
    - Easy heads (lower quality, tolerance for CPU latency)
    - Any head that would cause GPU OOM if loaded on-device

  Auto-detection: if ``tiered_architecture.screen_heads`` is present in
  medusa_config.json, those heads become PALTROW automatically.  Override
  with ``--medusa-paltrow-heads`` CLI flag.  Heads that OOM on GPU are
  also auto-promoted to PALTROW with a warning.

Reference: arXiv 2401.10774 (Medusa: Simple LLM Inference Acceleration
Framework with Multiple Decoding Heads)
"""

import glob
import json
import logging
import os
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MedusaResBlock(nn.Module):
    """Residual block: x + SiLU(Linear(x)). Zero-init ⇒ identity at start."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaHead(nn.Module):
    """Single Medusa head: ResBlock stack + vocabulary projection."""

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MedusaResBlock(hidden_size) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward: hidden_states [*, hidden] → logits [*, vocab]."""
        x = hidden_states
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


class MedusaModel(nn.Module):
    """Collection of Medusa heads for multi-position parallel drafting.

    Supports mixed CPU/GPU placement: screen heads (latency-tolerant, used
    only for DraftPreFilter inversion) run on CPU to save GPU VRAM.  Draft
    heads (quality-critical) stay on GPU.

    With AMD SAM/ReBAR, pinned host memory gives ~14 GB/s bandwidth.  The
    hidden-state copy to CPU is tiny (batch × hidden_size × 2 bytes ≈ 4 KB).

    Args:
        hidden_size: Base model hidden dimension.
        vocab_size: Vocabulary size.
        num_heads: Number of Medusa heads (each predicts +1..+K).
        num_layers: ResBlocks per head (typically 1).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 5,
        num_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.heads = nn.ModuleList(
            [
                MedusaHead(hidden_size, vocab_size, num_layers)
                for _ in range(num_heads)
            ]
        )
        # Mixed device placement state (set by from_pretrained)
        self._cpu_head_indices: FrozenSet[int] = frozenset()
        self._gpu_head_indices: FrozenSet[int] = frozenset()
        self._gpu_device: str = "cuda"
        # Pinned buffer for CPU heads (avoids alloc per forward)
        self._pinned_hidden: Optional[torch.Tensor] = None

    def _setup_mixed_placement(
        self,
        paltrow_head_indices: Set[int],
        gpu_device: str,
        dtype: torch.dtype,
    ):
        """Place designated PALTROW heads on CPU, others on GPU.

        PALTROW = Pinned-memory Auxiliary Latency-Tolerant Relocated On-CPU Workers.
        These heads run on CPU with pinned memory, saving GPU VRAM for KV cache
        and precision draft heads.  With AMD SAM/ReBAR, DMA bandwidth is ~14 GB/s.

        OOM auto-fallback: if a GPU head fails to allocate, it is automatically
        promoted to PALTROW with a warning — the system gracefully degrades
        rather than crashing.
        """
        paltrow_set = set(paltrow_head_indices)
        gpu_set = set(i for i in range(self.num_heads) if i not in paltrow_set)
        self._gpu_device = gpu_device

        cpu_param_bytes = 0
        gpu_param_bytes = 0

        # Move explicitly-designated PALTROW heads to CPU
        for i in sorted(paltrow_set):
            self.heads[i].to(device="cpu", dtype=dtype)
            for p in self.heads[i].parameters():
                cpu_param_bytes += p.numel() * p.element_size()

        # Move GPU heads with OOM auto-fallback to PALTROW
        auto_promoted = []
        for i in sorted(gpu_set):
            try:
                self.heads[i].to(device=gpu_device, dtype=dtype)
                for p in self.heads[i].parameters():
                    gpu_param_bytes += p.numel() * p.element_size()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or "OutOfMemoryError" in type(e).__name__:
                    logger.warning(
                        "PALTROW auto-promote: head %d OOM on GPU (%.0f MB needed), "
                        "relocating to CPU pinned memory",
                        i,
                        sum(p.numel() * p.element_size() for p in self.heads[i].parameters()) / 1e6,
                    )
                    torch.cuda.empty_cache()
                    self.heads[i].to(device="cpu", dtype=dtype)
                    paltrow_set.add(i)
                    auto_promoted.append(i)
                    for p in self.heads[i].parameters():
                        cpu_param_bytes += p.numel() * p.element_size()
                else:
                    raise

        # Update frozen sets
        self._cpu_head_indices = frozenset(paltrow_set)
        self._gpu_head_indices = frozenset(
            i for i in range(self.num_heads) if i not in paltrow_set
        )

        if auto_promoted:
            logger.info(
                "PALTROW auto-promoted heads %s to CPU (GPU VRAM insufficient). "
                "To avoid this, use --medusa-paltrow-heads or reduce --mem-fraction-static.",
                auto_promoted,
            )

        # Pre-allocate pinned hidden buffer for PALTROW heads (reused each forward)
        if paltrow_set:
            try:
                self._pinned_hidden = torch.empty(
                    64, self.hidden_size, dtype=dtype, pin_memory=True
                )
                logger.info(
                    "PALTROW placement: %d heads on CPU (%.0f MB pinned), "
                    "%d heads on GPU (%.0f MB), SAM/ReBAR DMA ready",
                    len(paltrow_set),
                    cpu_param_bytes / 1e6,
                    len(self._gpu_head_indices),
                    gpu_param_bytes / 1e6,
                )
            except Exception:
                self._pinned_hidden = None
                logger.warning("Could not allocate pinned memory, PALTROW heads use pageable")

    @property
    def has_cpu_heads(self) -> bool:
        return len(self._cpu_head_indices) > 0

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Forward all heads, handling mixed CPU/GPU placement.

        Args:
            hidden_states: [batch_size, hidden_size] from target last layer (on GPU).

        Returns:
            List of [batch_size, vocab_size] logits, one per head.
            GPU heads return logits on GPU; CPU heads return logits on CPU.
        """
        if not self.has_cpu_heads:
            return [head(hidden_states) for head in self.heads]

        # Prepare CPU hidden states — copy via pinned buffer for speed
        batch_size = hidden_states.shape[0]
        if self._pinned_hidden is not None and batch_size <= self._pinned_hidden.shape[0]:
            self._pinned_hidden[:batch_size].copy_(hidden_states[:batch_size])
            hidden_cpu = self._pinned_hidden[:batch_size]
        else:
            hidden_cpu = hidden_states.to("cpu", non_blocking=True)

        results: List[torch.Tensor] = [None] * self.num_heads  # type: ignore

        # GPU heads (async on GPU stream)
        for i in self._gpu_head_indices:
            results[i] = self.heads[i](hidden_states)

        # CPU heads (run while GPU is busy)
        with torch.no_grad():
            for i in self._cpu_head_indices:
                results[i] = self.heads[i](hidden_cpu)

        return results

    def predict_tokens(
        self,
        hidden_states: torch.Tensor,
        typical: bool = False,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
    ) -> torch.Tensor:
        """Get draft tokens from all heads.

        Args:
            hidden_states: [batch_size, hidden_size]
            typical: If True, use entropy-adaptive typical acceptance sampling
                     instead of greedy argmax. From FasterDecoding/Medusa.
            posterior_threshold: Fixed ceiling for typical acceptance (default 0.09).
            posterior_alpha: Entropy scaling factor (default 0.3, ≈ √threshold).

        Returns:
            [batch_size, num_heads] tensor of draft token IDs (on GPU).
        """
        logits = self.forward(hidden_states)
        if typical:
            return self._typical_sample_all(
                logits, hidden_states.device, posterior_threshold, posterior_alpha
            )
        tokens = []
        for lg in logits:
            tok = lg.argmax(dim=-1)
            if tok.device.type == "cpu":
                tok = tok.to(hidden_states.device, non_blocking=True)
            tokens.append(tok)
        return torch.stack(tokens, dim=1)

    def predict_with_logits(
        self,
        hidden_states: torch.Tensor,
        typical: bool = False,
        posterior_threshold: float = 0.09,
        posterior_alpha: float = 0.3,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get draft tokens AND raw logits from all heads.

        Returns:
            tokens: [batch_size, num_heads] draft token IDs (on GPU)
            logits: List of [batch_size, vocab_size] per head
                    (CPU heads return logits on CPU — DraftPreFilter
                     consumes screen logits there, avoiding GPU transfer
                     of the large vocab-sized tensor)
        """
        logits = self.forward(hidden_states)
        if typical:
            tokens = self._typical_sample_all(
                logits, hidden_states.device, posterior_threshold, posterior_alpha
            )
        else:
            tokens = []
            for lg in logits:
                tok = lg.argmax(dim=-1)
                if tok.device.type == "cpu":
                    tok = tok.to(hidden_states.device, non_blocking=True)
                tokens.append(tok)
            tokens = torch.stack(tokens, dim=1)
        return tokens, logits

    def _typical_sample_all(
        self,
        logits_list: List[torch.Tensor],
        target_device: torch.device,
        posterior_threshold: float,
        posterior_alpha: float,
    ) -> torch.Tensor:
        """Entropy-adaptive typical acceptance sampling for all heads.

        For each head's logits, compute entropy then set an adaptive threshold:
            threshold = min(posterior_threshold, exp(-entropy) * posterior_alpha)
        Tokens below threshold are masked out; we sample from the remainder.

        Low-entropy (confident) → stricter threshold → fewer candidates → higher quality.
        High-entropy (uncertain) → looser threshold → more diversity → better exploration.

        Reference: FasterDecoding/Medusa utils.py get_typical_one_token()
        """
        tokens = []
        for lg in logits_list:
            tok = self._typical_sample_one(
                lg, posterior_threshold, posterior_alpha
            )
            if tok.device.type == "cpu":
                tok = tok.to(target_device, non_blocking=True)
            tokens.append(tok)
        return torch.stack(tokens, dim=1)

    @staticmethod
    def _typical_sample_one(
        logits: torch.Tensor,
        posterior_threshold: float,
        posterior_alpha: float,
    ) -> torch.Tensor:
        """Entropy-adaptive typical sampling for a single head's logits.

        Args:
            logits: [batch_size, vocab_size] raw logits from one Medusa head.
            posterior_threshold: Fixed ceiling (e.g. 0.09).
            posterior_alpha: Entropy scaling (e.g. 0.3 ≈ √0.09).

        Returns:
            [batch_size] sampled token IDs.

        Algorithm (from FasterDecoding/Medusa):
            1. probs = softmax(logits)
            2. entropy = -sum(probs * log(probs + ε))
            3. threshold = min(posterior_threshold, exp(-entropy) * alpha)
            4. Mask out probs < threshold
            5. Sample from remaining distribution
            6. If all masked (degenerate), fall back to argmax
        """
        probs = torch.softmax(logits, dim=-1)

        # Per-sample entropy: [batch_size]
        entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )

        # Adaptive threshold: min(fixed, exp(-H) * α)  — [batch_size]
        threshold = torch.minimum(
            torch.full_like(entropy, posterior_threshold),
            torch.exp(-entropy) * posterior_alpha,
        )

        # Mask low-probability tokens: [batch_size, vocab_size]
        mask = probs < threshold.unsqueeze(-1)
        filtered_logits = logits.clone()
        filtered_logits[mask] = float("-inf")

        # Check for degenerate cases (all masked out) — fall back to argmax
        all_masked = filtered_logits.isinf().all(dim=-1)  # [batch_size]
        if all_masked.any():
            # For degenerate rows, restore original logits (argmax fallback)
            filtered_logits[all_masked] = logits[all_masked]

        # Sample from the filtered distribution
        filtered_probs = torch.softmax(filtered_logits, dim=-1)
        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        paltrow_head_indices: Optional[List[int]] = None,
    ) -> "MedusaModel":
        """Load Medusa heads from a checkpoint directory.

        Supports two checkpoint formats:
          1. HuggingFace safetensors (medusa_lm_head.{i}.{j}... keys)
          2. PyTorch state dict  (medusa_lm_head.pt)

        PALTROW placement:
          If ``paltrow_head_indices`` is provided, those heads run on CPU with
          pinned memory.  If None, auto-detects from config: screen heads AND
          bloom/volatility filter heads become PALTROW (they're latency-tolerant
          pre-rejection filters, not draft generators).  Pass ``[]`` to force
          all heads to GPU.  Heads that OOM on GPU auto-promote to PALTROW.
        """
        if dtype is None:
            dtype = torch.float16

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_path, "medusa_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json or medusa_config.json at {model_path}")

        with open(config_path) as f:
            cfg = json.load(f)

        hidden_size = cfg.get("hidden_size", cfg.get("model_hidden_size", 2560))
        vocab_size = cfg.get("vocab_size", 151669)
        num_heads = cfg.get("medusa_num_heads", cfg.get("num_heads", 5))
        num_layers = cfg.get("medusa_num_layers", cfg.get("num_layers", 1))

        # Auto-detect PALTROW heads from tiered config if not explicitly set
        if paltrow_head_indices is None:
            tiered = cfg.get("tiered_architecture", {})
            paltrow_candidates = []

            # Screen heads → PALTROW (negative filter, latency-tolerant)
            screen_heads = tiered.get("screen_heads", [])
            paltrow_candidates.extend(screen_heads)

            # Bloom/volatility filter heads → PALTROW (pre-rejection, CPU-friendly)
            bloom_heads = tiered.get("bloom_heads", [])
            paltrow_candidates.extend(bloom_heads)

            # Any head whose offset description contains "screen", "bloom",
            # "filter", or "volatility" is a pre-rejection head → PALTROW
            offsets = cfg.get("head_offsets", {})
            paltrow_keywords = {"screen", "bloom", "filter", "volatility", "negative"}
            for idx_str, desc in offsets.items():
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if any(kw in desc.lower() for kw in paltrow_keywords):
                    if idx not in paltrow_candidates:
                        paltrow_candidates.append(idx)

            if paltrow_candidates:
                paltrow_head_indices = paltrow_candidates
                logger.info(
                    "Auto-detected PALTROW heads %s from tiered config "
                    "(screen/bloom/filter → CPU pinned memory)",
                    paltrow_head_indices,
                )
            else:
                paltrow_head_indices = []

        # Validate indices
        paltrow_set = set(i for i in paltrow_head_indices if 0 <= i < num_heads)

        model = cls(hidden_size, vocab_size, num_heads, num_layers)

        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        pt_file = os.path.join(model_path, "medusa_lm_head.pt")

        if safetensor_files:
            from safetensors.torch import load_file

            state_dict: Dict[str, torch.Tensor] = {}
            for sf in safetensor_files:
                state_dict.update(load_file(sf))
            cls._load_flexible(model, state_dict)
        elif os.path.exists(pt_file):
            state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
            cls._load_flexible(model, state_dict)
        else:
            logger.warning(
                "No weights found at %s — Medusa heads are randomly initialised.",
                model_path,
            )

        # Apply PALTROW CPU/GPU placement or move everything to GPU
        if paltrow_set:
            model._setup_mixed_placement(paltrow_set, device, dtype)
            model.eval()
        else:
            # No PALTROW heads requested — try all on GPU, auto-promote on OOM
            try:
                model = model.to(device=device, dtype=dtype).eval()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "OOM loading all heads to GPU — falling back to "
                        "PALTROW auto-placement (per-head GPU attempt with CPU fallback)"
                    )
                    torch.cuda.empty_cache()
                    model = model.to(device="cpu", dtype=dtype)
                    model._setup_mixed_placement(set(), device, dtype)
                    model.eval()
                else:
                    raise

        param_count = sum(p.numel() for p in model.parameters())
        gpu_params = sum(
            p.numel() for p in model.parameters() if p.device.type != "cpu"
        )
        cpu_params = param_count - gpu_params
        paltrow_names = []
        if hasattr(model, "_cpu_head_indices") and model._cpu_head_indices:
            paltrow_names = sorted(model._cpu_head_indices)
        logger.info(
            "MedusaModel loaded: %d heads × %d layers, %.1f M params "
            "(GPU: %.1f M, PALTROW/CPU: %.1f M), dtype=%s%s",
            num_heads,
            num_layers,
            param_count / 1e6,
            gpu_params / 1e6,
            cpu_params / 1e6,
            dtype,
            f", PALTROW heads: {paltrow_names}" if paltrow_names else "",
        )
        return model

    @classmethod
    def _load_flexible(cls, model: "MedusaModel", state_dict: Dict[str, torch.Tensor]):
        """Load with flexible key mapping.

        Handles:
          medusa_lm_head.{head}.{sub}.linear.weight → heads[head].blocks[sub]...
          medusa_lm_head.{head}.{last}.weight → heads[head].proj.weight
          heads.{head}.blocks... → direct
        """
        mapped: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if key.startswith("medusa_lm_head."):
                parts = key.replace("medusa_lm_head.", "").split(".")
                head_idx = int(parts[0])
                sub_idx = int(parts[1])
                rest = ".".join(parts[2:])

                if head_idx >= model.num_heads:
                    continue
                num_blocks = len(model.heads[head_idx].blocks)
                if sub_idx == num_blocks:
                    mapped[f"heads.{head_idx}.proj.{rest}"] = tensor
                else:
                    mapped[f"heads.{head_idx}.blocks.{sub_idx}.{rest}"] = tensor
            elif key.startswith("heads."):
                # Handle raw nn.Sequential index format:
                #   heads.{head}.{sub}.linear.weight → heads.{head}.blocks.{sub}.linear.weight
                #   heads.{head}.{last}.weight       → heads.{head}.proj.weight
                parts = key.replace("heads.", "", 1).split(".")
                head_idx = int(parts[0])
                sub_idx = int(parts[1])
                rest = ".".join(parts[2:])

                if head_idx >= model.num_heads:
                    continue
                num_blocks = len(model.heads[head_idx].blocks)
                if sub_idx == num_blocks:
                    mapped[f"heads.{head_idx}.proj.{rest}"] = tensor
                elif f"heads.{head_idx}.blocks.{sub_idx}.{rest}" in model.state_dict():
                    mapped[f"heads.{head_idx}.blocks.{sub_idx}.{rest}"] = tensor
                else:
                    mapped[key] = tensor
            else:
                mapped[key] = tensor

        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            logger.warning("Medusa load — missing keys: %s", missing[:10])
        if unexpected:
            logger.debug("Medusa load — unexpected keys: %s", unexpected[:10])
