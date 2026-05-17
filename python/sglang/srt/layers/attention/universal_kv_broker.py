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
    compressed_hot: torch.Tensor | None
    residual_warm: torch.Tensor | None
    tier: TierKind
    size_hot_bytes: int
    size_warm_bytes: int
    importance: float
    last_access_tick: int
    spill_key: str | None
    metadata: dict[str, Any]


class UniversalKVBroker:
    """
    Broker skeleton for model-agnostic compressed KV ownership.

    Current implementation focuses on broker contracts:
    - allocate()
    - compress_and_store()
    - materialize_for_model()
    """

    def __init__(
        self,
        gpu_capacity_mb: int,
        ram_capacity_mb: int,
        hot_importance_threshold: float = 0.7,
        spill_manager: Any | None = None,
    ):
        self.gpu_capacity_mb = gpu_capacity_mb
        self.ram_capacity_mb = ram_capacity_mb
        self.hot_importance_threshold = hot_importance_threshold
        self.spill_manager = spill_manager
        self.registry = ModelShapeRegistry()
        self._next_id = 0
        self._tick = 0
        self._spilled_blocks = 0
        self._evicted_blocks = 0
        self._gpu_used_bytes = 0
        self._ram_used_bytes = 0
        self._gpu_capacity_bytes = max(0, gpu_capacity_mb) * 1024 * 1024
        self._ram_capacity_bytes = max(0, ram_capacity_mb) * 1024 * 1024
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
        preferred_hot = importance >= self.hot_importance_threshold
        bit_width = int(metadata.get("bit_width", 3))
        block_size = int(metadata.get("block_size", 16))
        rotor_id = int(metadata.get("rotor_id", 0))
        model_tag = int(metadata.get("origin_model_tag", 0))
        scale = float(metadata.get("scale", 1.0))
        self._tick += 1

        compressed_hot = self._compress_rotor_hot(kv_tensor, bit_width=bit_width)
        residual_warm = self._compress_turbo_residual(kv_tensor) if not preferred_hot else None
        size_hot_bytes = self._tensor_nbytes(compressed_hot)

        record = UniversalKVRecord(
            header=UniversalKVBlockHeader(
                block_size=block_size,
                bit_width=bit_width,
                rotor_id=rotor_id,
                origin_model_tag=model_tag,
                turbo_residual_flag=not preferred_hot,
                scale=scale,
            ),
            compressed_hot=compressed_hot,
            residual_warm=residual_warm,
            tier=TierKind.VRAM_HOT if preferred_hot else TierKind.RAM_WARM,
            size_hot_bytes=size_hot_bytes if preferred_hot else 0,
            size_warm_bytes=0,
            importance=importance,
            last_access_tick=self._tick,
            spill_key=None,
            metadata={"model_id": handle.model_id, "layer": handle.layer, **metadata},
        )
        self.records[block_id] = record

        if preferred_hot:
            self._gpu_used_bytes += size_hot_bytes
        else:
            self._move_record_to_warm(block_id, count_spill=False)
        self._rebalance_capacity()

    def materialize_for_model(self, model_id: str, layer: int, block_id: int) -> torch.Tensor:
        handle = self.handles[block_id]
        if handle.model_id != model_id or handle.layer != layer:
            raise ValueError(
                f"Universal KV block {block_id} belongs to ({handle.model_id}, layer {handle.layer}), "
                f"not ({model_id}, layer {layer})"
            )
        record = self.records[block_id]
        self._tick += 1
        record.last_access_tick = self._tick
        payload = self._load_payload(record)

        compressed_hot = payload["compressed_hot"]
        if compressed_hot is None:
            raise RuntimeError(f"Universal KV block {block_id} has no compressed payload")

        x = self._decompress_rotor_hot(compressed_hot)
        residual_warm = payload["residual_warm"]
        if residual_warm is not None:
            x = x + self._decompress_turbo_residual(residual_warm)
        return x

    def get_record_tier(self, block_id: int) -> TierKind:
        return self.records[block_id].tier

    def get_metrics(self) -> dict[str, int]:
        hot_blocks = 0
        warm_blocks = 0
        for record in self.records.values():
            if record.tier == TierKind.VRAM_HOT:
                hot_blocks += 1
            else:
                warm_blocks += 1
        return {
            "gpu_used_bytes": self._gpu_used_bytes,
            "gpu_capacity_bytes": self._gpu_capacity_bytes,
            "ram_used_bytes": self._ram_used_bytes,
            "ram_capacity_bytes": self._ram_capacity_bytes,
            "hot_blocks": hot_blocks,
            "warm_blocks": warm_blocks,
            "spilled_blocks": self._spilled_blocks,
            "evicted_blocks": self._evicted_blocks,
        }

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

    def _load_payload(self, record: UniversalKVRecord) -> dict[str, torch.Tensor | None]:
        if record.spill_key is None:
            return {
                "compressed_hot": record.compressed_hot,
                "residual_warm": record.residual_warm,
            }
        if self.spill_manager is None:
            raise RuntimeError("UniversalKVBroker record is spilled but no spill manager exists")
        payload = self.spill_manager.restore(record.spill_key)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"UniversalKVBroker expected dict spill payload for {record.spill_key}"
            )
        return {
            "compressed_hot": payload.get("compressed_hot"),
            "residual_warm": payload.get("residual_warm"),
        }

    def _tensor_nbytes(self, value: torch.Tensor | None) -> int:
        if value is None:
            return 0
        return value.element_size() * value.numel()

    def _move_record_to_warm(self, block_id: int, count_spill: bool) -> None:
        record = self.records[block_id]
        payload = {
            "compressed_hot": record.compressed_hot,
            "residual_warm": record.residual_warm,
        }
        record.size_warm_bytes = self._tensor_nbytes(record.compressed_hot) + self._tensor_nbytes(
            record.residual_warm
        )
        record.size_hot_bytes = 0
        record.tier = TierKind.RAM_WARM
        if self.spill_manager is not None:
            spill_key = f"ukv-{block_id}-{self._tick}"
            self.spill_manager.spill(spill_key, payload)
            record.spill_key = spill_key
            record.compressed_hot = None
            record.residual_warm = None
        else:
            if record.compressed_hot is not None:
                record.compressed_hot = record.compressed_hot.detach().to("cpu", non_blocking=True)
            if record.residual_warm is not None:
                record.residual_warm = record.residual_warm.detach().to("cpu", non_blocking=True)
        self._ram_used_bytes += record.size_warm_bytes
        if count_spill:
            self._spilled_blocks += 1

    def _rebalance_capacity(self) -> None:
        while self._gpu_used_bytes > self._gpu_capacity_bytes:
            demote_id = self._pick_hot_demotion_candidate()
            if demote_id is None:
                break
            record = self.records[demote_id]
            self._gpu_used_bytes -= record.size_hot_bytes
            self._move_record_to_warm(demote_id, count_spill=True)

        while self._ram_used_bytes > self._ram_capacity_bytes:
            evict_id = self._pick_warm_eviction_candidate()
            if evict_id is None:
                break
            self._evict_block(evict_id)

    def _pick_hot_demotion_candidate(self) -> int | None:
        candidate_id: int | None = None
        candidate_key: tuple[float, int] | None = None
        for block_id, record in self.records.items():
            if record.tier != TierKind.VRAM_HOT:
                continue
            key = (record.importance, record.last_access_tick)
            if candidate_key is None or key < candidate_key:
                candidate_id = block_id
                candidate_key = key
        return candidate_id

    def _pick_warm_eviction_candidate(self) -> int | None:
        candidate_id: int | None = None
        candidate_key: tuple[float, int] | None = None
        for block_id, record in self.records.items():
            if record.tier != TierKind.RAM_WARM:
                continue
            key = (record.importance, record.last_access_tick)
            if candidate_key is None or key < candidate_key:
                candidate_id = block_id
                candidate_key = key
        return candidate_id

    def _evict_block(self, block_id: int) -> None:
        record = self.records.pop(block_id)
        if record.tier == TierKind.VRAM_HOT:
            self._gpu_used_bytes = max(0, self._gpu_used_bytes - record.size_hot_bytes)
        else:
            self._ram_used_bytes = max(0, self._ram_used_bytes - record.size_warm_bytes)
            if record.spill_key is not None and self.spill_manager is not None:
                self.spill_manager.discard(record.spill_key)
        self.handles.pop(block_id, None)
        self._evicted_blocks += 1
