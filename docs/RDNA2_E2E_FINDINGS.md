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

## Step-1 fix applied + next layer
- FIXED: `attn_logits is None` in forward_decode (allocate split-KV scratch on the fly). Commit
  above. The eager-replay decode no longer AttributeErrors.
- NEXT (🟥): the decode path then hits `CUDA error: illegal memory access` (hipErrorIllegalAddress)
  under the cuda_graph_runner run_once buffers on RDNA2 — a deeper GPU memory bug (stale
  kv_indices/req_to_token mapping or non-contiguous buffer, consistent with the gfxGRAPH contiguity
  signal). Cascading failures here indicate the fork's custom RDNA2 cuda-graph integration is
  unfinished. Two strategic options below.

### Strategic option A — rebase onto current upstream sglang
- Fork = ~11,375 commits (upstream base) + ~226 custom commits (RDNA2/atom/rotorquant/ngram/etc).
  Currently on `forwardport/rdna2-20260519` (a forward-port already ~1 month behind upstream).
- Pro: upstream ROCm support is maturing fast; rebasing trims the custom-maintenance burden and may
  bring cleaner attention/graph code. The team already does forward-ports (it's the strategy).
- Con: the cuda-graph bugs are in OUR CUSTOM code (decode_attention.py is a fork RDNA2 op; the
  cuda_graph_runner RDNA2 path is ours) — a rebase will NOT auto-fix them and would likely DROP our
  custom RDNA2 decode-attn for upstream's (which targets CDNA/MI300; gfx1030 RDNA2 is niche and may
  not work). Re-applying 226 commits onto a fast base is conflict-heavy (we just resolved 5 committed
  conflicts from the last forward-port).
- Verdict: worth a DELIBERATE divergence audit as its own task (fetch upstream, compute merge-base,
  triage which custom commits are superseded by upstream ROCm work vs still needed), NOT a quick fix
  for the cuda-graph bug. Sanctioned RDNA2 fast path meanwhile = spec-decode (99 t/s, works) with
  cuda-graph OFF.

### Strategic option B — own a HIP flash-decode attention kernel (rust+hip)  ✅ BUILT 2026-06-14
- The decode-attention is currently sglang's custom RDNA2 *Triton* op; we did NOT own a HIP version.
  A 3-agent search confirmed nothing reusable exists anywhere (every asset is CPU/stub/Triton or FFI
  to external AOTriton/AITER — the thing crashing). So we built one.
- **DONE:** `rs_rdna2_kernels::flash_decode` (`projects/rust/crates/rs_rdna2_kernels/hip/flash_decode.cpp`)
  + library copy `build/kernels/flash-decode-hip` (standalone `.hsaco`). One Wave32 warp/query, QK dot
  via `__shfl_xor`, online softmax + output accumulator in registers — no LDS, no block barriers,
  contiguous reads → deterministic and graph-capture-safe (the property §2 needs). f32 Q, int8 KV +
  per-page-per-head descale, GQA, causal; head_dim multiple of 32, ≤256. Wired as the real `SDPAKernel`
  (`Rdna2FlashDecodeKernel`) in `rs_universal_kv_broker_core` behind the `rocm` feature, replacing the
  zeroed `RustWave32Kernel` / `-38` AOTriton stub. Verified exact vs CPU ref on RX 6700 XT (rust commit
  72e4c19).
- The int8-KV + descale path also gives §4 (rq3/tq3 KV-quant) a **GPU decode path** (was CPU-fallback
  at ~1.4 t/s).
- **REMAINING (next sprint):** wire this kernel into the sglang serving loop as a selectable attention
  backend (replace the Triton `triton_ops/rdna2/decode_attention.py` decode path), so §2/§3 cuda-graph
  capture can succeed on the deterministic kernel. The kernel + Rust binding + broker `SDPAKernel` are
  done; the remaining work is the sglang-side backend glue (fp16 path / KV-layout marshalling) +
  re-enabling capture.

## Attention-backend + KV-quant sweep (measured 2026-06-14, RX 6700 XT gfx1030)
Qwen2.5-0.5B **f16**, batch 1, input-len 512, output-len 64, `--disable-cuda-graph`, decode tok/s:

| backend (`--attention-backend`) | decode tok/s | status |
|---|---|---|
| `universal_broker` (our unified KV-broker) | **~54–60** | ✅ best sglang backend (≈/> triton) |
| `triton` (rdna2 two-stage) | 57.5 | ✅ |
| `atom` (Hybrid AITER/Triton) | ~51–54 | ✅ |
| `torch_native` (Torch SDPA) | 36.0 | ✅ (slowest working) |
| `aiter` (AOTriton/AITER) | ~6 (erratic 6–59) | ⚠️ AITER not tuned for RDNA2 (targets MI3xx); very slow/unreliable |
| `wave` (Wave DSL) | — | 🟥 FAILS: `ModuleNotFoundError: No module named 'wave_lang'` (dep not installed) |
| `radix` | n/a | RadixAttention is the shared prefix-cache layer, not a standalone decode kernel |

**KV-quant:** `--kv-cache-dtype tq3` (TurboQuant) 🟥 FAILS at pool init —
`RuntimeError: Calling torch.geqrf on a CPU tensor requires compiling PyTorch with LAPACK`
(TurboQuant's rotation does a CPU QR; the installed ROCm PyTorch has no LAPACK). Fix = move the
geqrf to GPU (rocSOLVER) or build torch with LAPACK, or precompute the rotation. rq3 had the separate
~1.4 t/s CPU-codec issue (§4).

**Cross-engine (same f16 model, d512):** llama.cpp HIP fattn-vec **181 tok/s** ≫ best sglang
(~60, universal_broker/triton) ≫ torch_native 36. Confirms the native-HIP stack advantage and
motivates the HIP decode-attention backend (rs_rdna2_kernels::flash_decode) for sglang. **Lesson
(bench, don't trust docs):** the prior "Triton ~7x slower than torch_native" doc claim was FALSE
(Triton 1.6x faster); corrected.
