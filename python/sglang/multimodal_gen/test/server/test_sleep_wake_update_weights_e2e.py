import os
import signal
import socket
import subprocess
import time
from typing import Optional

import httpx
import pytest

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model


# Single-model fast path (per user request).
_MODEL_ID = "Qwen/Qwen-Image"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log(msg: str) -> None:
    # Keep logs visible under `pytest -s`
    print(f"[{_now()}][E2E] {msg}", flush=True)


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_http_ready(base_url: str, timeout_s: float = 180.0) -> None:
    _log(f"[STEP 1] Waiting for server ready: GET {base_url}/health (timeout={timeout_s}s)")
    deadline = time.time() + timeout_s
    last_err = None
    last_status = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=5.0)
            last_status = r.status_code
            if r.status_code == 200:
                _log("[STEP 1] Server is ready (health=200)")
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(
        f"Server not ready after {timeout_s}s. last_status={last_status} last_err={last_err}"
    )


def _launch_server(model_path: str, port: int) -> subprocess.Popen:
    """
    Launch diffusion server via `sglang serve`.

    NOTE:
    - CLI requires `--model-path`.
    - We intentionally disable cpu offload to match reviewer requirement.
    """
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        model_path,  # HF repo id is accepted here in your setup
        "--port",
        str(port),
        "--num-gpus",
        "1",
        "--dit-cpu-offload",
        "false",
        "--text-encoder-cpu-offload",
        "false",
    ]

    _log(f"[STEP 0] Launching server: {' '.join(cmd)}")
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
    _log("[CLEANUP] Terminating server process (SIGINT -> KILL if needed)")
    try:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=20)
        return
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass


def _read_tail(p: subprocess.Popen, max_lines: int = 300) -> str:
    if p.stdout is None:
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
    except Exception:
        return ""


def _post_json(base_url: str, path: str, payload: dict, timeout_s: float = 300.0) -> httpx.Response:
    url = f"{base_url}{path}"
    t0 = time.time()
    r = httpx.post(url, json=payload, timeout=timeout_s)
    dt = time.time() - t0
    _log(f"[HTTP] POST {path} status={r.status_code} time={dt:.2f}s")
    return r


def _do_generate(base_url: str) -> None:
    """
    Trigger a minimal images generation request.

    If your server supports these knobs, they can significantly speed up:
    - num_inference_steps / steps
    - guidance_scale
    - size
    If unsupported, server may ignore them or return 400 (then we'll see it).
    """
    payload = {
        "prompt": "a simple photo of a cat",
        "n": 1,
        "size": "256x256",
        # best-effort speed knobs (may be ignored depending on schema)
        "num_inference_steps": 1,
        "guidance_scale": 1.0,
    }
    _log("[STEP 6] generate: POST /v1/images/generations")
    r = _post_json(base_url, "/v1/images/generations", payload, timeout_s=900.0)
    assert r.status_code in (200, 201), f"generate failed: {r.status_code} {r.text}"
    _log("[STEP 6] generate: success")


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
    except Exception:
        return None


def _wait_for_mem_drop(
    mem_before: int,
    drop_mib: int,
    timeout_s: float = 60.0,
    poll_s: float = 2.0,
) -> Optional[int]:
    deadline = time.time() + timeout_s
    target = max(0, mem_before - drop_mib)
    last = None
    _log(
        f"[STEP 3] Waiting for GPU mem drop >= {drop_mib} MiB "
        f"(before={mem_before} MiB, target<= {target} MiB, timeout={timeout_s}s)"
    )
    while time.time() < deadline:
        last = _query_gpu_mem_used_mib(0)
        if last is None:
            _log("[STEP 3] nvidia-smi unavailable; skip mem assertion")
            return None
        if last <= target:
            _log(f"[STEP 3] GPU mem drop observed: after={last} MiB (target met)")
            return last
        time.sleep(poll_s)
    _log(f"[STEP 3] GPU mem drop NOT observed in time: last={last} MiB")
    return last


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_sleep_wake_refit_generate_e2e():
    """
    Fast single-model regression with step logs:
      0) launch
      1) wait health
      2) sleep
      3) verify GPU memory decreases (best-effort)
      4) wake
      5) refit/update_weights_from_disk using SAME model snapshot path
      6) generate
    """
    port = _find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    _log(f"Test start: model={_MODEL_ID} port={port} base_url={base_url}")

    # STEP 0: launch
    p = _launch_server(model_path=_MODEL_ID, port=port)

    try:
        # STEP 1: wait ready
        _wait_http_ready(base_url, timeout_s=900.0)

        # STEP 2: sleep
        mem_before_sleep = _query_gpu_mem_used_mib(0)
        _log(f"[STEP 2] sleep: GPU mem before sleep = {mem_before_sleep} MiB")
        _log("[STEP 2] sleep: POST /release_memory_occupation")
        r = _post_json(base_url, "/release_memory_occupation", payload={}, timeout_s=180.0)
        assert r.status_code == 200, f"sleep failed: {r.status_code} {r.text}"
        out = r.json()
        _log(f"[STEP 2] sleep: response={out}")
        assert out.get("success", True) is True, f"sleep response: {out}"
        if "sleeping" in out:
            assert out["sleeping"] is True, f"sleep response: {out}"
        _log("[STEP 2] sleep: success")

        # STEP 3: GPU mem check (best-effort)
        if mem_before_sleep is not None:
            target_drop_mib = int(os.environ.get("SGLANG_MMGEN_SLEEP_MEM_DROP_MIB", "256"))
            mem_after_sleep = _wait_for_mem_drop(
                mem_before_sleep,
                drop_mib=target_drop_mib,
                timeout_s=60.0,
                poll_s=2.0,
            )
            if mem_after_sleep is not None:
                assert mem_after_sleep <= max(0, mem_before_sleep - target_drop_mib), (
                    f"GPU memory did not drop enough after sleep: "
                    f"before={mem_before_sleep} MiB, after={mem_after_sleep} MiB, "
                    f"threshold_drop={target_drop_mib} MiB"
                )
        else:
            _log("[STEP 3] GPU mem check skipped (nvidia-smi unavailable)")

        # STEP 4: wake
        _log("[STEP 4] wake: POST /resume_memory_occupation")
        r = _post_json(base_url, "/resume_memory_occupation", payload={}, timeout_s=300.0)
        assert r.status_code == 200, f"wake failed: {r.status_code} {r.text}"
        out = r.json()
        _log(f"[STEP 4] wake: response={out}")
        assert out.get("success", True) is True, f"wake response: {out}"
        if "sleeping" in out:
            assert out["sleeping"] is False, f"wake response: {out}"
        mem_after_wake = _query_gpu_mem_used_mib(0)
        _log(f"[STEP 4] wake: GPU mem after wake = {mem_after_wake} MiB")
        _log("[STEP 4] wake: success")

        # STEP 5: refit/update_weights_from_disk (same model snapshot path)
        _log("[STEP 5] refit: resolving local snapshot path via maybe_download_model()")
        t0 = time.time()
        model_snapshot_path = maybe_download_model(_MODEL_ID)
        _log(f"[STEP 5] refit: snapshot_path={model_snapshot_path} (took {time.time() - t0:.2f}s)")
        _log("[STEP 5] refit: POST /update_weights_from_disk")
        r = _post_json(
            base_url,
            "/update_weights_from_disk",
            payload={"model_path": model_snapshot_path, "flush_cache": True},
            timeout_s=900.0,
        )
        assert r.status_code == 200, f"update_weights_from_disk failed: {r.status_code} {r.text}"
        out = r.json()
        _log(f"[STEP 5] refit: response={out}")
        assert out.get("success") is True, f"update_weights_from_disk response: {out}"
        _log("[STEP 5] refit: success")

        # STEP 6: generate
        _do_generate(base_url)

        _log("Test finished: SUCCESS")

    except Exception as e:
        tail = _read_tail(p)
        raise AssertionError(f"{e}\n\n---- server log tail ----\n{tail}") from e
    finally:
        _terminate_proc(p)