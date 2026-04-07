"""Medusa multi-head speculative decoding model.

Medusa adds N lightweight MLP heads to the base model. Each head predicts
the token at a different future position (+1, +2, ... +K) in parallel from
the SAME last hidden state. No autoregression — all K candidates come from
a single forward pass.

Architecture per head:
  ResBlock × L  →  Linear(hidden → vocab)
  ResBlock = x + SiLU(Linear(x))   (initialized to identity at training start)

Reference: arXiv 2401.10774 (Medusa: Simple LLM Inference Acceleration
Framework with Multiple Decoding Heads)
"""

import glob
import json
import logging
import os
from typing import Dict, List, Optional

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

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Forward all heads.

        Args:
            hidden_states: [batch_size, hidden_size] from target last layer.

        Returns:
            List of [batch_size, vocab_size] logits, one per head.
        """
        return [head(hidden_states) for head in self.heads]

    def predict_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get argmax draft tokens from all heads.

        Args:
            hidden_states: [batch_size, hidden_size]

        Returns:
            [batch_size, num_heads] tensor of draft token IDs.
        """
        logits = self.forward(hidden_states)
        return torch.stack([lg.argmax(dim=-1) for lg in logits], dim=1)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "MedusaModel":
        """Load Medusa heads from a checkpoint directory.

        Supports two checkpoint formats:
          1. HuggingFace safetensors (medusa_lm_head.{i}.{j}... keys)
          2. PyTorch state dict  (medusa_lm_head.pt)
        """
        if dtype is None:
            dtype = torch.float16

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config.json at {model_path}")

        with open(config_path) as f:
            cfg = json.load(f)

        hidden_size = cfg.get("hidden_size", cfg.get("model_hidden_size", 2560))
        vocab_size = cfg.get("vocab_size", 151669)
        num_heads = cfg.get("medusa_num_heads", cfg.get("num_heads", 5))
        num_layers = cfg.get("medusa_num_layers", cfg.get("num_layers", 1))

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

        model = model.to(device=device, dtype=dtype).eval()
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "MedusaModel loaded: %d heads × %d layers, %.1f M params, dtype=%s",
            num_heads,
            num_layers,
            param_count / 1e6,
            dtype,
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
                mapped[key] = tensor
            else:
                mapped[key] = tensor

        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            logger.warning("Medusa load — missing keys: %s", missing[:10])
        if unexpected:
            logger.debug("Medusa load — unexpected keys: %s", unexpected[:10])
