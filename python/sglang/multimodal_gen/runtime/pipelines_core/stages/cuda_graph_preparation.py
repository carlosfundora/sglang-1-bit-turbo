from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )


class CudaGraphPreparationStage(PipelineStage):
    def __init__(self, denoising_stage: "DenoisingStage") -> None:
        super().__init__()
        self.denoising_stage = denoising_stage

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not (
            getattr(server_args, "enable_cuda_graph", False)
            or getattr(server_args, "enable_piecewise_cuda_graph", False)
        ):
            return batch

        if batch.is_warmup:
            return batch

        if batch.extra is None:
            batch.extra = {}

        prepared_vars = self.denoising_stage._prepare_denoising_loop(batch, server_args)
        batch.extra["_denoising_prepared_vars"] = prepared_vars

        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        timesteps_cpu = timesteps.cpu()

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            self.denoising_stage._maybe_prepare_cuda_graph(
                batch=batch,
                server_args=server_args,
                timesteps=timesteps,
                timesteps_cpu=timesteps_cpu,
                target_dtype=target_dtype,
                image_kwargs=prepared_vars["image_kwargs"],
                pos_cond_kwargs=prepared_vars["pos_cond_kwargs"],
                neg_cond_kwargs=prepared_vars["neg_cond_kwargs"],
                latents=prepared_vars["latents"],
                boundary_timestep=prepared_vars["boundary_timestep"],
                seq_len=prepared_vars["seq_len"],
                reserved_frames_mask=prepared_vars["reserved_frames_mask"],
                guidance=prepared_vars["guidance"],
            )

        return batch
