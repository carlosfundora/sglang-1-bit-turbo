from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from universal_kv.model_registry import ModelShapeRegistry
from universal_kv.types import TierKind, UniversalKVBlockHeader


@dataclass(frozen=True)
class AllocationHandle:
    block_id: int
    model_id: str
    layer: int
    seq_len: int


@dataclass
class UniversalKVRecord:
    header: UniversalKVBlockHeader
    compressed_hot: torch.Tensor
    residual_warm: torch.Tensor | None
    tier: TierKind
    metadata: dict[str, Any]


class UniversalKVBroker:
    """
    Broker skeleton for model-agnostic compressed KV ownership.

    Current implementation focuses on broker contracts:
    - allocate()
    - compress_and_store()
    - materialize_for_model()
    """

    def __init__(self, gpu_capacity_mb: int, ram_capacity_mb: int):
        self.gpu_capacity_mb = gpu_capacity_mb
        self.ram_capacity_mb = ram_capacity_mb
        self.registry = ModelShapeRegistry()
        self._next_id = 0
        self.handles: dict[int, AllocationHandle] = {}
        self.records: dict[int, UniversalKVRecord] = {}

    def allocate(self, model_id: str, layer: int, seq_len: int) -> int:
        block_id = self._next_id
        self._next_id += 1
        self.handles[block_id] = AllocationHandle(block_id, model_id, layer, seq_len)
        return block_id

    def compress_and_store(
        self, block_id: int, kv_tensor: torch.Tensor, metadata: dict[str, Any]
    ) -> None:
        handle = self.handles[block_id]
        importance = float(metadata.get("importance", 1.0))
        tier = TierKind.VRAM_HOT if importance >= 0.7 else TierKind.RAM_WARM
        bit_width = int(metadata.get("bit_width", 3))
        block_size = int(metadata.get("block_size", 16))
        rotor_id = int(metadata.get("rotor_id", 0))
        model_tag = int(metadata.get("origin_model_tag", 0))
        scale = float(metadata.get("scale", 1.0))

        compressed_hot = self._compress_rotor_hot(kv_tensor, bit_width=bit_width)
        residual_warm = (
            self._compress_turbo_residual(kv_tensor) if tier == TierKind.RAM_WARM else None
        )

        self.records[block_id] = UniversalKVRecord(
            header=UniversalKVBlockHeader(
                block_size=block_size,
                bit_width=bit_width,
                rotor_id=rotor_id,
                origin_model_tag=model_tag,
                turbo_residual_flag=residual_warm is not None,
                scale=scale,
            ),
            compressed_hot=compressed_hot,
            residual_warm=residual_warm,
            tier=tier,
            metadata={"model_id": handle.model_id, "layer": handle.layer, **metadata},
        )

    def materialize_for_model(self, model_id: str, layer: int, block_id: int) -> torch.Tensor:
        record = self.records[block_id]
        x = self._decompress_rotor_hot(record.compressed_hot)
        if record.residual_warm is not None:
            x = x + self._decompress_turbo_residual(record.residual_warm)
        return x

    def get_record_tier(self, block_id: int) -> TierKind:
        return self.records[block_id].tier

    def _compress_rotor_hot(self, kv_tensor: torch.Tensor, bit_width: int) -> torch.Tensor:
        scale = kv_tensor.abs().max().clamp(min=1e-8)
        normalized = (kv_tensor / scale).clamp(-1, 1)
        levels = (1 << bit_width) - 1
        q = torch.round((normalized + 1.0) * 0.5 * levels).to(torch.uint8)
        return q

    def _decompress_rotor_hot(self, code: torch.Tensor) -> torch.Tensor:
        # Skeleton decode path; exact scale/rotation restoration comes in kernel tranche.
        return code.to(torch.float32)

    def _compress_turbo_residual(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        return torch.sign(kv_tensor).to(torch.int8)

    def _decompress_turbo_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.to(torch.float32)
