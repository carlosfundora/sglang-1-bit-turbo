# RDNA2 (gfx1030/1031) serving guide — verified config

The one place for "how to actually run this on an RX 6700 XT class card." All numbers
measured 2026-06-14 on an RX 6700 XT (gfx1031 reported as gfx1030 via HSA override),
ROCm 7.2, torch 2.13.0a0+hip7.2, Qwen2.5-0.5B f16. Background + per-finding detail:
`RDNA2_E2E_FINDINGS.md`; the fix backlog: `RDNA2_HARDENING_SUGGESTIONS.md`.

## TL;DR known-good launch
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 -m sglang.launch_server \
  --model-path <model> \
  --attention-backend triton \      # or universal_broker; NOT raw aiter / wave
  --dtype float16 \                 # bf16 auto-overridden anyway
  --mem-fraction-static 0.6         # (auto-lowered to 0.45 for rq*/tq* KV)
```
Most of this is auto-applied by the gfx1031 defaults (`hardware_backend/rocm/gfx1031_defaults.py`)
when an RDNA2 GPU is detected — the flags above are the explicit, portable form.

## Attention backends (decode tok/s, Qwen2.5-0.5B f16, d512)
| backend | tok/s | use it? |
|---|---|---|
| `universal_broker` (our unified KV-broker) | ~60 | ✅ best |
| `triton` (rdna2 two-stage split-KV) | 57.5 | ✅ default |
| `aiter` | (auto-routes to triton on RDNA) | ✅ safe — resolves to triton (no CK on RDNA) |
| `atom` | ~51–54 | ✅ ok (triton fallback) |
| `torch_native` | 36 | ⚠️ slowest working; the old "prefer torch_native" advice is STALE |
| `rdna2_hip` (our native HIP flash-decode) | 52 (d512) | 🧪 experimental; degrades at long ctx until inter-block split-K lands |
| `wave` | — | 🟥 needs `wave_lang` (absent) + matrix cores RDNA2 lacks — don't use |
| llama.cpp (reference, same model) | ~181 | (different engine; framework overhead is sglang's ceiling, not attention) |

## Settings that matter
- **dtype:** `float16`. bf16 GEMM crashes on gfx1030 (missing `fdot2.bf16.bf16`) — auto-overridden to fp16.
- **cuda-graph:** **off by default on RDNA2** (auto-disabled in `server_args`). The serving `cuda_graph_runner`
  decode path still hits `hipErrorIllegalAddress` (RDNA2_HARDENING_SUGGESTIONS P0.1). Force-enable to test
  with `SGLANG_RDNA2_FORCE_CUDA_GRAPH=1` — gfxGRAPH (auto-enabled, `GFXGRAPH_GUARD=1`) makes it a graceful
  diagnosable path, not a SIGSEGV.
- **mem-fraction:** 0.6 general; **auto-lowered to 0.45 for rq*/tq*/iso/planar KV** (codec working buffers
  OOM otherwise on 12 GB).
- **KV-quant:** `--kv-cache-dtype tq3` works (~21 t/s, 4.92x compression — needs GPU-QR rotation fix, done);
  `rq3` ~1.4 t/s (CPU codec). Both are **capacity-bound, not latency-bound** until the GPU KV-codec lands
  (P1.5). Use fp16 KV for latency.
- **spec-decode:** `--speculative-algorithm NGRAM` is the RDNA2 fast path — ~99 t/s vs ~15 baseline.
- **gfxGRAPH:** installed system-wide; auto-enabled on RDNA2 (`GFXGRAPH=1`, `GFXGRAPH_GUARD=1`).

## Prohibited / not used
- **ollama:** PROHIBITED for model loading unless the user explicitly asks. We use llama.cpp / sglang /
  ATOM-RS. The service is disabled (on-demand, hard 2 GB cap); removed from litellm + consul.

## Known ROCm landmines (defended)
- **CPU LAPACK absent:** any `torch.linalg.{qr,svd,eig,solve,...}` on a CPU tensor raises (geqrf). Route via
  `sglang.srt.hardware_backend.rocm.linalg.linalg_on_device`. Done for the turboquant rotation; the
  `multimodal_gen` diffusion schedulers (`scheduling_*unipc*`, `helios_denoising` eigh) still have raw CPU
  `torch.linalg.solve`/`eigh` — fix with the same helper IF multimodal gen is ever run on gfx1030 (not the
  LLM serving path).
- **`pkill -f launch_server`** matches the calling shell — kill by `--port`/PID.
- **memory pressure:** heavy concurrent model loads can thrash the box; the ENCOM adaptive memory manager +
  oomd@80% + overcommit=0 defend it (see memory `system-memory-management`).
