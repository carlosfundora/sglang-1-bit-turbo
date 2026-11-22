from __future__ import annotations

from typing import Any

import torch


class UniversalKVSpillManager:
    """Pinned-memory spill manager for warm universal KV blocks."""

    def __init__(self, pin_memory: bool = True):
        self.pin_memory = pin_memory
        self.store: dict[str, Any] = {}

    def spill(self, key: str, value: Any) -> str:
        self.store[key] = self._to_host(value)
        return key

    def restore(self, key: str) -> Any:
        return self.store[key]

    def discard(self, key: str) -> None:
        self.store.pop(key, None)

    def _to_host(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            host = value.detach().to("cpu", non_blocking=True)
            if self.pin_memory:
                host = host.pin_memory()
            return host
        if isinstance(value, dict):
            return {k: self._to_host(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_host(v) for v in value)
        if isinstance(value, list):
            return [self._to_host(v) for v in value]
        return value
