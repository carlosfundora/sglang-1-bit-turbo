from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelKVShape:
    model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int


class ModelShapeRegistry:
    """Model metadata registry used by the universal broker materialization path."""

    def __init__(self) -> None:
        self._registry: dict[str, ModelKVShape] = {}

    def register(
        self, model_id: str, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> None:
        self._registry[model_id] = ModelKVShape(
            model_id=model_id,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )

    def lookup(self, model_id: str) -> ModelKVShape:
        if model_id not in self._registry:
            raise KeyError(f"Model {model_id!r} is not registered")
        return self._registry[model_id]
