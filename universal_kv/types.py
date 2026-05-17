from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import struct


class TierKind(str, Enum):
    VRAM_HOT = "vram_hot"
    RAM_WARM = "ram_warm"


@dataclass(frozen=True)
class UniversalKVBlockHeader:
    """Compact metadata header for a universal compressed KV block."""

    block_size: int
    bit_width: int
    rotor_id: int
    origin_model_tag: int
    turbo_residual_flag: bool
    scale: float

    _PACK_FMT = "<BBBB?f"
    _PACK_SIZE = struct.calcsize(_PACK_FMT)

    def to_bytes(self) -> bytes:
        return struct.pack(
            self._PACK_FMT,
            self.block_size,
            self.bit_width,
            self.rotor_id,
            self.origin_model_tag,
            self.turbo_residual_flag,
            self.scale,
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "UniversalKVBlockHeader":
        if len(raw) != cls._PACK_SIZE:
            raise ValueError(
                f"Invalid header length {len(raw)}; expected {cls._PACK_SIZE}"
            )
        bsz, bw, rid, model_tag, residual, scale = struct.unpack(cls._PACK_FMT, raw)
        return cls(
            block_size=bsz,
            bit_width=bw,
            rotor_id=rid,
            origin_model_tag=model_tag,
            turbo_residual_flag=residual,
            scale=scale,
        )
