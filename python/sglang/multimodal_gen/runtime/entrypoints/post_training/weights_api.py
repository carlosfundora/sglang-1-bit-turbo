"""Weight update API for the diffusion engine."""

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct import (
    GetWeightsChecksumReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightFromDiskReqInput,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

router = APIRouter()

logger = init_logger(__name__)


@router.post("/update_weights_from_disk")
async def update_weights_from_disk(request: Request):
    """Update model weights from disk inplace without restarting the server."""
    body = await request.json()
    model_path = body.get("model_path")
    if not model_path:
        return ORJSONResponse(
            {"success": False, "message": "model_path is required"},
            status_code=400,
        )

    req = UpdateWeightFromDiskReqInput(
        model_path=model_path,
        flush_cache=body.get("flush_cache", True),
        target_modules=body.get("target_modules"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse(
            {"success": False, "message": str(e)},
            status_code=500,
        )

    result = response.output
    success = result.get("success", False)
    message = result.get("message", "Unknown status")
    return ORJSONResponse(
        {"success": success, "message": message},
        status_code=200 if success else 400,
    )


@router.post("/get_weights_checksum")
async def get_weights_checksum(request: Request):
    """Return SHA-256 checksum of each requested module's weights."""
    body = await request.json()
    req = GetWeightsChecksumReqInput(
        module_names=body.get("module_names"),
    )

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse({"error": str(e)}, status_code=500)

    return ORJSONResponse(response.output, status_code=200)


async def _handle_memory_occupation_request(request: Request, req_class: type):
    """Handle memory sleep/wake requests forwarded to scheduler."""
    req = req_class()

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        logger.exception("scheduler_client.forward failed for %s", req_class.__name__)
        return ORJSONResponse({"success": False, "message": str(e)}, status_code=500)

    if response is None:
        logger.error("scheduler returned None response for %s", req_class.__name__)
        return ORJSONResponse(
            {"success": False, "message": "Empty response object from scheduler."},
            status_code=500,
        )

    out = response.output

    if out is None:
        return ORJSONResponse(
            {"success": False, "message": "Empty response from scheduler."},
            status_code=500,
        )

    if (
        "detail" in out
        and isinstance(out["detail"], dict)
        and "success" in out["detail"]
    ):
        payload = out["detail"]
    else:
        logger.error("missing success in scheduler output detail: %r", out)
        return ORJSONResponse(
            {
                "success": False,
                "message": "Missing 'success' field in scheduler response.",
            },
            status_code=500,
        )

    success = bool(payload["success"])
    return ORJSONResponse(payload, status_code=200 if success else 400)


@router.post("/release_memory_occupation")
async def release_memory_occupation(request: Request):
    """Release GPU memory occupation (sleep the engine)."""
    return await _handle_memory_occupation_request(
        request, ReleaseMemoryOccupationReqInput
    )


@router.post("/resume_memory_occupation")
async def resume_memory_occupation(request: Request):
    """Resume GPU memory occupation (wake the engine)."""
    return await _handle_memory_occupation_request(
        request, ResumeMemoryOccupationReqInput
    )
