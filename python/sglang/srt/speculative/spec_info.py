from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.ngram_worker import NGRAMWorker


class SpeculativeAlgorithm(Enum):
    """Enumeration of speculative decoding algorithms."""

    EAGLE = auto()
    EAGLE3 = auto()
    P_EAGLE = auto()
    STANDALONE = auto()
    NGRAM = auto()
    P_CASCADE = auto()
    MEDUSA = auto()
    SAGUARO = auto()  # wrapper, not standalone — wraps any other algorithm
    CHIMERA = auto()  # CHIMERA-SD: fused P-EAGLE + Hydra + DyTC + n-gram + SSD
    SELF_SPEC = auto()  # Self-Speculative Decoding via early-exit / hidden reuse
    PHANTOM_SD = auto()  # NGRAM with CPU-threaded tree pre-building
    TQ5_X = auto()  # TurboQuant 5 eXtended: HSA zero-copy ghost-draft (AMD gfx103x)
    NONE = auto()

    @classmethod
    def from_string(cls, name: Optional[str]) -> SpeculativeAlgorithm:
        if name is None:
            return cls.NONE
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown speculative algorithm name: {name}")

    def is_none(self) -> bool:
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self) -> bool:
        # NOTE: EAGLE3 and P_EAGLE are variants of EAGLE
        return self in {
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.P_EAGLE,
        }

    def is_eagle3(self) -> bool:
        return self in {
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.P_EAGLE,
        }

    def is_p_eagle(self) -> bool:
        return self == SpeculativeAlgorithm.P_EAGLE

    def is_standalone(self) -> bool:
        return self == SpeculativeAlgorithm.STANDALONE

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def is_p_cascade(self) -> bool:
        return self == SpeculativeAlgorithm.P_CASCADE

    def is_medusa(self) -> bool:
        return self == SpeculativeAlgorithm.MEDUSA

    def is_chimera(self) -> bool:
        return self == SpeculativeAlgorithm.CHIMERA

    def is_self_spec(self) -> bool:
        return self == SpeculativeAlgorithm.SELF_SPEC

    def is_phantom_sd(self) -> bool:
        return self == SpeculativeAlgorithm.PHANTOM_SD

    def is_tq5x(self) -> bool:
        return self == SpeculativeAlgorithm.TQ5_X

    def supports_spec_v2(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def needs_draft_model(self) -> bool:
        """Whether the algorithm requires --speculative-draft-model-path."""
        return self in {
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.P_EAGLE,
            SpeculativeAlgorithm.STANDALONE,
            SpeculativeAlgorithm.P_CASCADE,
            SpeculativeAlgorithm.CHIMERA,
        }

    def create_worker(
        self, server_args: ServerArgs
    ) -> Optional[Union[Type[BaseSpecWorker], Type[TpModelWorker], Type[NGRAMWorker]]]:
        assert (
            not self.is_none()
        ), "Cannot create worker for NONE speculative algorithm."

        enable_overlap = not server_args.disable_overlap_schedule

        if self.is_p_cascade():
            if enable_overlap:
                raise ValueError(
                    "P_CASCADE does not support overlap scheduling yet. "
                    "Use --disable-overlap-schedule."
                )
            from sglang.srt.speculative.p_cascade_worker import PCascadeWorker

            return PCascadeWorker

        elif self.is_eagle() and server_args.enable_multi_layer_eagle:
            # FIXME: migrate to EagleWorker
            if enable_overlap:
                from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                    MultiLayerEagleWorkerV2,
                )

                return MultiLayerEagleWorkerV2

            from sglang.srt.speculative.multi_layer_eagle_worker import (
                MultiLayerEagleWorker,
            )

            return MultiLayerEagleWorker

        elif self.is_eagle():
            if enable_overlap:
                from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

                return EAGLEWorkerV2

            from sglang.srt.speculative.eagle_worker import EAGLEWorker

            return EAGLEWorker
        elif self.is_standalone():
            if enable_overlap:
                from sglang.srt.speculative.standalone_worker_v2 import (
                    StandaloneWorkerV2,
                )

                return StandaloneWorkerV2

            from sglang.srt.speculative.standalone_worker import StandaloneWorker

            return StandaloneWorker
        elif self.is_ngram():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support overlap worker creation."
                )

            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        elif self.is_medusa():
            if enable_overlap:
                raise ValueError(
                    "MEDUSA does not support overlap scheduling yet. "
                    "Use --disable-overlap-schedule."
                )
            from sglang.srt.speculative.medusa_worker import MedusaWorker

            return MedusaWorker

        elif self.is_chimera():
            if enable_overlap:
                raise ValueError(
                    "CHIMERA does not support overlap scheduling yet. "
                    "Use --disable-overlap-schedule."
                )
            from sglang.srt.speculative.chimera_worker import ChimeraWorker

            return ChimeraWorker

        elif self.is_self_spec():
            if enable_overlap:
                raise ValueError(
                    "SELF_SPEC does not support overlap scheduling yet. "
                    "Use --disable-overlap-schedule."
                )
            from sglang.srt.speculative.ssd_worker import SSDWorker

            return SSDWorker

        elif self.is_phantom_sd():
            if enable_overlap:
                raise ValueError(
                    "PHANTOM_SD does not support overlap scheduling yet. "
                    "Use --disable-overlap-schedule."
                )
            from sglang.srt.speculative.phantom_tree_worker import PhantomTreeWorker

            return PhantomTreeWorker

        elif self.is_tq5x():
            if enable_overlap:
                raise ValueError(
                    "TQ5_X does not support overlap scheduling yet. "
                    "Use --disable-overlap-schedule."
                )
            from sglang.srt.speculative.tq5x_worker import TQ5XWorker

            return TQ5XWorker

        raise ValueError("Unreachable code path in create_worker.")


class SpecInputType(IntEnum):
    # NOTE: introduce this to distinguish the SpecInput types of multiple algorithms when asserting in attention backends.
    # If all algorithms can share the same datastrucutre of draft_input and verify_input, consider simplify it
    EAGLE_DRAFT = auto()
    EAGLE_VERIFY = auto()
    NGRAM_VERIFY = auto()
    MEDUSA_VERIFY = auto()  # reuses NGRAM tree infrastructure


class SpecInput(ABC):
    def __init__(self, spec_input_type: SpecInputType):
        self.spec_input_type = spec_input_type

    def is_draft_input(self) -> bool:
        # FIXME: remove this function which is only used for assertion
        # or use another variable name like `draft_input` to substitute `spec_info`
        return self.spec_input_type == SpecInputType.EAGLE_DRAFT

    def is_verify_input(self) -> bool:
        return self.spec_input_type in {
            SpecInputType.EAGLE_VERIFY,
            SpecInputType.NGRAM_VERIFY,
            SpecInputType.MEDUSA_VERIFY,
        }

    @abstractmethod
    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        pass

    def get_spec_adjusted_global_num_tokens(
        self, forward_batch: ModelWorkerBatch
    ) -> Tuple[List[int], List[int]]:
        c1, c2 = self.get_spec_adjust_token_coefficient()
        global_num_tokens = [x * c1 for x in forward_batch.global_num_tokens]
        global_num_tokens_for_logprob = [
            x * c2 for x in forward_batch.global_num_tokens_for_logprob
        ]
        return global_num_tokens, global_num_tokens_for_logprob
