# RDNA2 (gfx1030) end-to-end test findings — fix sprint backlog

Live e2e test of sglang-1-bit-turbo on **AMD RX 6700 XT (gfx1030, Wave32)**, 2026-06-14.
Env: torch 2.13.0a0+hip7.2, `sgl_kernel` rebuilt against installed torch, `HSA_OVERRIDE_GFX_VERSION=10.3.0`.
Model: Qwen2.5-0.5B-Instruct. Each row is an independent finding; group the fixes into one sprint.

## Severity legend
🟥 CRITICAL (blocks serving) · 🟧 HIGH (severe perf/feature loss) · 🟨 MED · ⬜ INFO

---

## 🟥 1. triton attention backend crashed at init — `UnboundLocalError: torch`
- `layers/attention/triton_backend.py:107`: a redundant local `import torch` inside
  `TritonAttnBackend.__init__` scoped `torch` as a function-local, so the earlier
  `torch.compiler.disable(...)` (line ~135) raised `UnboundLocalError`. **No server started on the
  default backend.**
- **Status: FIXED during test** (removed the redundant local import; module imports torch at L9).
- Sprint: add a smoke test that instantiates each attention backend; lint against local re-imports
  shadowing module globals.

## 🟥 2. cuda-graph runtime decode SIGSEGVs (capture OK, replay crashes)
- Capture succeeds (`Capture cuda graph end, 1.95s`) but the first decode batch crashes:
  `Subprocess scheduler_0 crashed with exit code -11` (SIGSEGV) in `run_batch`/`event_loop_overlap`.
- **Root cause (via gfxGRAPH, see §8): non-contiguous tensors captured into the graph are replayed
  at a stale address.** `rs_gfxgraph_toolbox::analyze_shape_for_graph_capture` flags strided
  KV/`retrive_*` tensors as `graph_capture_copy_required=true`.
- Impact: forces `--disable-cuda-graph`, which costs ~3–6× decode throughput (see §3).
- Sprint: extend sglang's existing manual RDNA2 contiguity workaround (already applied to ngram
  `retrive_*`) to the **decode-attention** capture path; gate in CI with the gfxGRAPH analyzer.

## 🟧 3. Low baseline decode throughput (cuda-graph disabled)
- Qwen2.5-0.5B baseline ~15–16 tok/s (triton, cuda-graph OFF). Expected 50–100+ for a 0.5B.
- Direct consequence of §2. Fixing cuda-graph replay is the main lever. NOTE: spec-decode (§4)
  recovers throughput **without** cuda-graph, so it's the sanctioned RDNA2 fast path today.

## 🟧 4. RotorQuant `rq3` KV-cache: 4.92× compression but ~1.4 tok/s (≈10× slowdown)
- `--kv-cache-dtype rq3` → "RotorQuant planar 3-bit", 52 B/token K vs 256 FP16 (4.92×), KV pool
  0.58 GB vs 2.05 GB, **correct output**. Functionally good.
- BUT decode collapses to ~1.4 tok/s: the codec compress/decompress is on the per-token critical
  path and runs **CPU-side on gfx1030** (the aiter/HIP KV-codec kernel targets gfx950, not gfx1030;
  matches the documented "gfx1030 segfaults on float32 GEMM / uint8 bitwise unpack → CPU fallback").
- Sprint: wire the gfx1030 GPU KV-codec path (we have `rotorquant-kv-hip` + `rs_kv_codec_bridge`),
  or restrict rq*/tq* KV-quant to capacity-bound (not latency-bound) use and document it.

## ✅ 5. NGRAM speculative decoding — WORKS, ~6.5× speedup (headline)
- `--speculative-algorithm NGRAM` (draft=8, steps=4, BFS): stable, no crashes; prose ~99 tok/s vs
  ~15 baseline. Uses the rebuilt native HIP `tree_speculative_sampling_target_only`.
- TODO-verify: confirm greedy losslessness (same prompt @ temp=0 == non-spec output; lossless by
  construction but not yet A/B-checked live).
- ⬜ ngram auto-disables overlap scheduler + mixed chunked prefill (logged; expected).

## ✅ 6. Tool calling — WORKS (qwen25, non-streaming)
- `get_weather({"city":"Paris"})`, finish_reason=tool_calls.
- 🟨 Reach limitation: our Rust `rs_tool_parser` accelerates `BaseFormatDetector.parse_streaming_increment`;
  qwen25/pythonic/mistral **override** it, so `SGLANG_RUST_TOOL_PARSER=1` is a no-op for them. Only
  base-method detectors (llama3, deepseek, …) use the rust path. Document the supported set.

## 🟨 7. VRAM not released cleanly across crashed servers
- An rq3 launch OOM'd ("0 bytes free") because it overlapped the just-crashed cuda-graph server's
  VRAM. Crashed scheduler children should release GPU memory promptly; add teardown/retry headroom.

## ⬜ 8. gfxGRAPH diagnostic deployed (root-cause tool for §2)
- `rs_gfxgraph_toolbox` `examples/rdna2_graph_diag.rs`: row-contiguous decode tensors are
  graph-safe; strided KV/`retrive_*` tensors flag `graph_capture_copy_required=true`; RDNA2 launch
  shapes bs[1,2,4,8] all VALID (Wave32, no matrix instr). Use `analyze_shape_for_graph_capture` as a
  pre-capture CI gate.

## ⬜ 9. Misc RDNA2 notes
- bf16 auto-overridden to fp16 ("bf16 GEMM crashes on gfx1030: missing fdot2.bf16.bf16") — handled.
- Harness gotcha (our tooling): `pkill -f launch_server` matches the test shell itself; kill by
  PID/port.

---

## Suggested single-sprint plan
1. (🟥) Fix cuda-graph replay contiguity (§2) → unlocks §3 baseline throughput. Highest ROI.
2. (🟧) GPU KV-codec decompress for gfx1030 (§4) → makes rq*/tq* usable at latency.
3. (🟥, done) Keep the triton_backend import fix (§1) + add the backend-init smoke test.
4. (🟨) Tool-parser reach: either accelerate the overriding detectors or document supported set (§6).
5. (🟨) Crash-time VRAM release (§7). (✅) NGRAM losslessness A/B check (§5).

---

## gfxGRAPH deployment result (Tier 1) — and the true cuda-graph root cause

Deployed gfxGRAPH v0.3.4 Tier 1 into sglang (GFXGRAPH=1 via sitecustomize so the spawned scheduler
gets the `torch.cuda.CUDAGraph → BridgedCUDAGraph` monkey-patch). Workers logged
"Enabling gfxGRAPH ... Native bridge loaded ... enabled successfully".

Outcome (high value):
- gfxGRAPH **converted the opaque hard SIGSEGV (§2) into a graceful eager fallback + a precise,
  fixable Python error.** It safely refused the unsafe capture on ROCm/HIP 7.2.26015
  ("disabled by default ... set GFXGRAPH_ENABLE_UNSAFE_GRAPH_CAPTURE") rather than crashing.
- The eager-replay then surfaced the REAL blocker, independent of cuda-graph:
  `triton_ops/rdna2/decode_attention.py:734  assert max_kv_splits == attn_logits.shape[2]`
  -> `AttributeError: 'NoneType' object has no attribute 'shape'` (attn_logits is None in the
  cuda_graph_runner run_once path). The RDNA2 decode-attention op is not wired for the graph-runner
  code path — this is almost certainly what manifested as the §2 SIGSEGV too.

### Tier 1 vs Tier 2 — recommendation
- **Tier 1 (deploy now):** zero-build safety net + diagnostics. Turns RDNA2 graph segfaults into
  graceful eager fallback with clear errors. Keep it on. BUT on ROCm 7.2 it defaults capture OFF, so
  it gives NO throughput win by itself (eager ≈ --disable-cuda-graph).
- **For the perf win you need BOTH, not Tier 1 alone:** (1) fix the sglang RDNA2 decode-attention
  `attn_logits is None` bug (decode_attention.py:734) so the model forward works in the graph-runner
  path; THEN (2) enable real HIP-graph capture via GFXGRAPH_ENABLE_UNSAFE_GRAPH_CAPTURE and/or the
  Tier 2 native bridge. Order matters: (1) before (2), else capture replays a broken forward.
