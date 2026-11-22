"""Launch the inference server."""

import asyncio
import logging
import os
import sys
import warnings

import torch  # must be imported before sgl_kernel to set up lib paths on ROCm

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

logger = logging.getLogger(__name__)

suppress_noisy_warnings()


def register_atom_backend():
    """Ensure ATOM hybrid backend is present in the runtime registry."""
    from sglang.srt.layers.attention import attention_registry

    if "atom_hybrid" in attention_registry.ATTENTION_BACKENDS:
        return

    from sglang.srt.layers.attention.hybrid_atom_backend import create_hybrid_atom_backend

    attention_registry.ATTENTION_BACKENDS["atom_hybrid"] = create_hybrid_atom_backend
    logger.info("Registered atom_hybrid backend at launch time.")


def setup_atom_backend_with_flags(server_args):
    """Apply ATOM launch flags to attention backend routing."""
    atom_backends = {
        server_args.attention_backend,
        server_args.decode_attention_backend,
        server_args.prefill_attention_backend,
    }
    if not any(backend in {"atom", "atom_hybrid"} for backend in atom_backends):
        return server_args

    register_atom_backend()

    model_impl = getattr(server_args, "model_impl", "auto")
    if model_impl not in {"auto", "sglang"}:
        logger.info(
            "Skipping ATOM backend auto-routing for model_impl=%s to preserve runtime behavior.",
            model_impl,
        )
        return server_args

    get_model_config = getattr(server_args, "get_model_config", None)
    if callable(get_model_config):
        try:
            model_config = get_model_config()
            if getattr(model_config, "is_audio_understandable_model", False) or getattr(
                model_config, "is_audio_model", False
            ):
                logger.info(
                    "Skipping ATOM backend auto-routing for audio-capable model to keep transcription endpoints stable."
                )
                return server_args
        except Exception as exc:
            logger.debug("Unable to infer audio model capability: %s", exc)

    if server_args.atom_wave32 and server_args.decode_attention_backend is None:
        server_args.decode_attention_backend = "atom"

    if server_args.prefill_attention_backend is None:
        if server_args.atom_fallback_to_triton:
            server_args.prefill_attention_backend = "triton"
        elif server_args.attention_backend in {"atom", "atom_hybrid"}:
            server_args.prefill_attention_backend = "atom"

    if server_args.atom_wave32 and server_args.attention_backend == "atom":
        server_args.attention_backend = "atom_hybrid"

    return server_args


def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    server_args = prepare_server_args(sys.argv[1:])
    server_args = setup_atom_backend_with_flags(server_args)

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
