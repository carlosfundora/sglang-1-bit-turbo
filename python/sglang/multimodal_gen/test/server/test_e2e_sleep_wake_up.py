"""
End-to-end test for GPU memory sleep/wake and in-place weight update workflow
in a running SGLang multimodal server.

Author:

Kun Lin, https://github.com/klhhhhh
Chenyang Zhao, https://github.com/zhaochenyang20
Menyang Liu, https://github.com/dreamyang-liu
shuwen, https://github.com/alphabetc1
Mook, https://github.com/Godmook

This test validates both functional correctness of /release_memory_occupation
and /resume_memory_occupation:

1. Launch a SGLang server process without offloading DiT and text encoder.
   This roughly takes 56GB on H200.

2. Trigger GPU memory release via the `/release_memory_occupation` endpoint
   and verify that GPU memory usage decreases.

   TODO (chenyang): still found some memory usage that can not be released:

   https://github.com/sgl-project/sglang/issues/19441

3. Trigger GPU memory resume via the `/resume_memory_occupation` endpoint
   and verify that GPU memory usage increases accordingly.

4. Perform an in-place model weight update using the
   `/update_weights_from_disk` endpoint without restarting the server.

    This is inherent with the usage in RL. SGLang Diffusion Server is slept
    while training and resumed to do the next rollout with new weights.
"""

import os
import subprocess
import time
from typing import Optional

import httpx
import pytest

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.utils import launch_server_cmd, terminate_process, wait_for_http_ready

logger = init_logger(__name__)

_MODEL_ID = "Qwen/Qwen-Image"


def _read_tail(p: subprocess.Popen, max_lines: int = 300) -> str:
    if p.stdout is None:
        logger.warning("[TAIL] Process stdout is None; cannot read log tail")
        return ""

    lines = []
    start = time.time()
    while time.time() - start < 2.0:
        line = p.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines[-max_lines:])


def _post_json(
    base_url: str, path: str, payload: dict, timeout_s: float = 300.0
) -> httpx.Response:
    request_start_time_s = time.time()
    response = httpx.post(f"{base_url}{path}", json=payload, timeout=timeout_s)
    request_elapsed_s = time.time() - request_start_time_s
    logger.info(
        f"[HTTP] POST {path} status={response.status_code} time={request_elapsed_s:.2f}s"
    )
    return response


def _do_generate(base_url: str) -> None:
    base_payload = {
        "prompt": "a cute panda",
        "width": 256,
        "height": 256,
        "num_inference_steps": 2,
    }

    payload = dict(base_payload)
    payload["response_format"] = "b64_json"

    logger.info(
        "[STEP 6] generate: POST /v1/images/generations (try response_format=b64_json)"
    )
    r = _post_json(base_url, "/v1/images/generations", payload, timeout_s=900.0)
    logger.info(f"[STEP 6] generate: status={r.status_code} body_head={r.text[:800]}")

    if r.status_code == 200:
        logger.info("[STEP 6] generate: success (200)")
        return
    elif r.status_code == 400 and "requires cloud storage" in r.text:
        logger.warning(
            "[STEP 6] generate: got 400 due to missing cloud storage; "
            "treating as known non-fatal response_format/url behavior."
        )
        return
    else:
        raise AssertionError(f"generate failed: {r.status_code} {r.text}")


def _query_gpu_mem_used_mib(gpu_index: int = 0) -> Optional[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used = int(out.splitlines()[0].strip())
        return used
    except Exception as e:
        logger.warning(
            f"[GPU] Failed to query nvidia-smi memory.used: {type(e).__name__}: {e}"
        )
        return None


def _require_gpu_mem_query(gpu_index: int = 0) -> int:
    mem = _query_gpu_mem_used_mib(gpu_index)
    assert mem is not None, (
        "nvidia-smi memory query is unavailable; cannot enforce GPU memory assertions. "
        "Make sure the CI runner has nvidia-smi accessible."
    )
    return mem


def _assert_mem_changed(
    label: str,
    before: int,
    after: int,
    min_delta_mib: int,
    *,
    expect_decrease: bool,
) -> None:
    if expect_decrease:
        delta = before - after
        logger.info(
            f"[MEM] {label}: before={before} MiB after={after} MiB delta={delta} MiB (expect decrease)"
        )
        assert delta >= min_delta_mib, (
            f"GPU memory did not decrease enough for '{label}': before-after={delta} MiB < {min_delta_mib} MiB "
            f"(before={before} MiB, after={after} MiB)"
        )
    else:
        delta = after - before
        logger.info(
            f"[MEM] {label}: before={before} MiB after={after} MiB delta={delta} MiB (expect increase)"
        )
        assert delta >= min_delta_mib, (
            f"GPU memory did not increase enough for '{label}': after-before={delta} MiB < {min_delta_mib} MiB "
            f"(before={before} MiB, after={after} MiB)"
        )


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_sleep_wake_refit_generate_e2e():
    cmd = (
        "sglang serve "
        f"--model-path {_MODEL_ID} "
        "--num-gpus 1 "
        "--dit-cpu-offload false "
        "--text-encoder-cpu-offload false"
    )
    process, port = launch_server_cmd(cmd, host="127.0.0.1")
    base_url = f"http://127.0.0.1:{port}"
    logger.info(f"Test start: model={_MODEL_ID} port={port} base_url={base_url}")

    try:
        # launch
        wait_for_http_ready(f"{base_url}/health", timeout=900, process=process)

        # Baseline GPU memory
        mem_before_sleep = _require_gpu_mem_query(0)
        logger.info(f"[STEP 1] baseline: GPU mem = {mem_before_sleep} MiB")

        # sleep
        logger.info("[STEP 2] sleep: POST /release_memory_occupation")
        r = _post_json(
            base_url, "/release_memory_occupation", payload={}, timeout_s=180.0
        )
        assert r.status_code == 200, f"sleep failed: {r.status_code} {r.text}"
        out = r.json()
        logger.info(f"[STEP 2] sleep: response={out}")
        assert out.get("success", True) is True, f"sleep response: {out}"
        if "sleeping" in out:
            assert out["sleeping"] is True, f"sleep response: {out}"

        mem_after_sleep = _require_gpu_mem_query(0)
        min_sleep_delta = int(
            os.environ.get("SGLANG_MMGEN_SLEEP_MEM_DELTA_MIB", "1024")
        )
        _assert_mem_changed(
            "sleep (baseline -> after sleep)",
            mem_before_sleep,
            mem_after_sleep,
            min_sleep_delta,
            expect_decrease=True,
        )

        # wake
        logger.info("[STEP 4] wake: POST /resume_memory_occupation")
        r = _post_json(
            base_url, "/resume_memory_occupation", payload={}, timeout_s=300.0
        )
        assert r.status_code == 200, f"wake failed: {r.status_code} {r.text}"
        out = r.json()
        logger.info(f"[STEP 4] wake: response={out}")
        assert out.get("success", True) is True, f"wake response: {out}"
        if "sleeping" in out:
            assert out["sleeping"] is False, f"wake response: {out}"

        mem_after_wake = _require_gpu_mem_query(0)
        min_wake_delta = int(os.environ.get("SGLANG_MMGEN_WAKE_MEM_DELTA_MIB", "1024"))
        _assert_mem_changed(
            "wake (after sleep -> after wake)",
            mem_after_sleep,
            mem_after_wake,
            min_wake_delta,
            expect_decrease=False,
        )

        # refit/update_weights_from_disk using SAME model snapshot path
        logger.info(
            "[STEP 5] refit: resolving local snapshot path via maybe_download_model()"
        )
        refit_start_time = time.time()
        model_snapshot_path = maybe_download_model(_MODEL_ID)
        logger.info(
            f"[STEP 5] refit: snapshot_path={model_snapshot_path} (took {time.time() - refit_start_time:.2f}s)"
        )
        logger.info("[STEP 5] refit: POST /update_weights_from_disk")
        r = _post_json(
            base_url,
            "/update_weights_from_disk",
            payload={"model_path": model_snapshot_path, "flush_cache": True},
            timeout_s=900.0,
        )
        assert (
            r.status_code == 200
        ), f"update_weights_from_disk failed: {r.status_code} {r.text}"
        out = r.json()
        logger.info(f"[STEP 5] refit: response={out}")
        assert out.get("success") is True, f"update_weights_from_disk response: {out}"

        # generate
        _do_generate(base_url)

        logger.info("Test finished: SUCCESS")

    except Exception as e:
        tail = _read_tail(process)
        raise AssertionError(f"{e}\n\n---- server log tail ----\n{tail}") from e
    finally:
        terminate_process(process)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
