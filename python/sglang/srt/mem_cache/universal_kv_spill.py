from __future__ import annotations

import torch


class UniversalKVSpillManager:
    """Pinned-memory spill manager for warm universal KV blocks."""

    def __init__(self, pin_memory: bool = True):
        self.pin_memory = pin_memory
        self.store: dict[str, torch.Tensor] = {}

    def spill(self, key: str, value: torch.Tensor) -> str:
        host = value.detach().to("cpu", non_blocking=True)
        if self.pin_memory:
            host = host.pin_memory()
        self.store[key] = host
        return key

    def restore(self, key: str) -> torch.Tensor:
        return self.store[key]
