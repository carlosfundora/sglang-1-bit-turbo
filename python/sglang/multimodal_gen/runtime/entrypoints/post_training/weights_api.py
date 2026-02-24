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

router = APIRouter()


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
    """Handle sleep/wake requests. No tags. Return 400 on success=False."""
    # Body may be empty; do not crash on request.json()
    try:
        body = await request.json()
        if body is None:
            body = {}
        if not isinstance(body, dict):
            return ORJSONResponse(
                {"success": False, "message": "Request body must be a JSON object."},
                status_code=400,
            )
    except Exception:
        body = {}

    req = req_class()

    try:
        response = await async_scheduler_client.forward(req)
    except Exception as e:
        return ORJSONResponse({"success": False, "message": str(e)}, status_code=500)

    out = getattr(response, "output", None)

    # 1) dict output: prefer explicit success; fall back to heuristic / compat
    if isinstance(out, dict):
        if "success" in out:
            success = bool(out["success"])
        else:
            # Backward compat: treat known "ok" patterns as success
            # (recommended: remove this after worker always returns success)
            success = True
            out["success"] = True

        return ORJSONResponse(out, status_code=200 if success else 400)

    # 2) non-dict output: treat None as failure
    if out is None:
        return ORJSONResponse(
            {"success": False, "message": "Empty response from scheduler."},
            status_code=500,
        )

    return ORJSONResponse({"success": True, "output": out}, status_code=200)


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