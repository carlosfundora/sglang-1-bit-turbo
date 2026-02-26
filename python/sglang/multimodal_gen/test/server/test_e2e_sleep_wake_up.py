import os
import signal
import socket
import subprocess
import time
from typing import Optional

import httpx
import pytest

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_MODEL_ID = "Qwen/Qwen-Image"


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_http_ready(base_url: str, timeout_s: float = 180.0) -> None:
    logger.info(
        f"[STEP 1] Waiting for server ready: GET {base_url}/health (timeout={timeout_s}s)"
    )
    deadline = time.time() + timeout_s
    last_err = None
    last_status = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=5.0)
            last_status = r.status_code
            if r.status_code == 200:
                logger.info("[STEP 1] Server is ready (health=200)")
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(
        f"Server not ready after {timeout_s}s. last_status={last_status} last_err={last_err}"
    )


def _launch_server(model_path: str, port: int) -> subprocess.Popen:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--num-gpus",
        "1",
        "--dit-cpu-offload",
        "false",
        "--text-encoder-cpu-offload",
        "false",
    ]
    logger.info(f"[STEP 0] Launching server: {' '.join(cmd)}")
    env = os.environ.copy()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    return p


def _terminate_proc(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return

    logger.info("[CLEANUP] Terminating server process (SIGINT -> KILL if needed)")
    try:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=20)
        logger.info("[CLEANUP] Server process terminated gracefully via SIGINT")
        return
    except Exception as e:
        logger.warning(f"[CLEANUP] SIGINT termination failed: {type(e).__name__}: {e}")

    try:
        p.kill()
        logger.info("[CLEANUP] Server process killed via SIGKILL")
    except Exception as e:
        logger.error(f"[CLEANUP] SIGKILL failed: {type(e).__name__}: {e}")


def _read_tail(p: subprocess.Popen, max_lines: int = 300) -> str:
    if p.stdout is None:
        logger.warning("[TAIL] Process stdout is None; cannot read log tail")
        return ""

    try:
        lines = []
        start = time.time()
        while time.time() - start < 2.0:
            line = p.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        logger.warning(
            f"[TAIL] Failed to read server log tail: {type(e).__name__}: {e}"
        )
        return ""


def _post_json(
    base_url: str, path: str, payload: dict, timeout_s: float = 300.0
) -> httpx.Response:
    t0 = time.time()
    r = httpx.post(f"{base_url}{path}", json=payload, timeout=timeout_s)
    dt = time.time() - t0
    logger.info(f"[HTTP] POST {path} status={r.status_code} time={dt:.2f}s")
    return r


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

    if r.status_code in (200, 201):
        logger.info("[STEP 6] generate: success (200/201)")
        return

    if r.status_code == 400 and "requires cloud storage" in r.text:
        logger.warning(
            "[STEP 6] generate: got 400 due to missing cloud storage; "
            "treating as known non-fatal response_format/url behavior."
        )
        return

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
) -> None:
    delta = abs(after - before)
    logger.info(
        f"[MEM] {label}: before={before} MiB after={after} MiB |delta|={delta} MiB"
    )
    assert delta >= min_delta_mib, (
        f"GPU memory change too small for '{label}': |after-before|={delta} MiB < {min_delta_mib} MiB "
        f"(before={before} MiB, after={after} MiB)"
    )


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_sleep_wake_refit_generate_e2e():
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    logger.info(f"Test start: model={_MODEL_ID} port={port} base_url={base_url}")

    p = _launch_server(model_path=_MODEL_ID, port=port)

    try:
        # 1) launch
        _wait_http_ready(base_url, timeout_s=900.0)

        # Baseline GPU memory
        mem0 = _require_gpu_mem_query(0)
        logger.info(f"[STEP 1] baseline: GPU mem = {mem0} MiB")

        # 2) sleep
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

        mem1 = _require_gpu_mem_query(0)
        min_sleep_delta = int(
            os.environ.get("SGLANG_MMGEN_SLEEP_MEM_DELTA_MIB", "1024")
        )
        _assert_mem_changed(
            "sleep (baseline -> after sleep)", mem0, mem1, min_sleep_delta
        )

        # 4) wake
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

        mem2 = _require_gpu_mem_query(0)
        min_wake_delta = int(os.environ.get("SGLANG_MMGEN_WAKE_MEM_DELTA_MIB", "1024"))
        _assert_mem_changed(
            "wake (after sleep -> after wake)", mem1, mem2, min_wake_delta
        )

        # 5) refit/update_weights_from_disk using SAME model snapshot path
        logger.info(
            "[STEP 5] refit: resolving local snapshot path via maybe_download_model()"
        )
        t0 = time.time()
        model_snapshot_path = maybe_download_model(_MODEL_ID)
        logger.info(
            f"[STEP 5] refit: snapshot_path={model_snapshot_path} (took {time.time() - t0:.2f}s)"
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

        # 6) generate
        _do_generate(base_url)

        logger.info("Test finished: SUCCESS")

    except Exception as e:
        tail = _read_tail(p)
        raise AssertionError(f"{e}\n\n---- server log tail ----\n{tail}") from e
    finally:
        _terminate_proc(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
