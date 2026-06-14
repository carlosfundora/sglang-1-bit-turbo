# RDNA2 (gfx1030) stack — code-health / hardening / docs suggestions

Prioritized, actionable backlog distilled from the 2026-06-14 work + measurements. P0 = correctness/
crash, P1 = perf/usability, P2 = polish. Each item: what, why, where.

## P0 — correctness & crash-safety
1. **cuda_graph_runner decode illegal-access (the real cuda-graph unblock).** Even with gfxGRAPH
   installed, the serving decode forward inside `cuda_graph_runner`'s static buffers hits
   `hipErrorIllegalAddress` (localized to `cuda_graph_runner.py:1282` replay → eager forward; see
   RDNA2_E2E_FINDINGS §"gfxGRAPH installed"). Fix: make every captured decode buffer
   (kv_indices / req_to_token / split-KV scratch) a stable, correctly-sized static tensor; use
   `GFXGRAPH_GUARD=3` + `compute_sanitizer_cmd` to pin the exact OOB tensor. Until fixed, RDNA2
   default stays cuda-graph OFF (already hardened in `server_args`).
2. **Backend-init smoke test (CI).** A test that instantiates every registered attention backend on
   a tiny config and asserts no import/init crash would have caught the triton `UnboundLocalError`
   (§1) and the aiter-on-RDNA slow path. Add to the test suite; gate the GPU part on `is_hip()`.
3. **LAPACK-free guard for linalg on ROCm.** TurboQuant's `geqrf` crash (fixed) is a class: ROCm
   PyTorch ships no CPU LAPACK. Add a tiny helper `linalg_on_device(fn, tensor)` that routes
   `qr/svd/eig/cholesky` to GPU when CUDA is available, and grep the codebase for other CPU
   `torch.linalg.*` on the hot path.

## P1 — performance & usability
4. **rdna2_hip inter-block split-K.** The native HIP decode backend (`rdna2_hip`) degrades at long
   context because it only splits KV across ≤4 warps in one block. Add inter-block split-K
   (`grid.y = parallel_blocks(seq_k)` + a combine pass, like llama.cpp fattn-vec) to make it
   competitive at depth. (Same TODO in `build/kernels/flash-decode-hip`.)
5. **GPU KV-codec for rq3/tq3.** rq3 ≈1.4 t/s and tq3 ≈21 t/s because the codec runs CPU-side on
   gfx1030. Wire the `rotorquant-kv-hip` / `rs_kv_codec_bridge` GPU path, or document rq*/tq* as
   capacity-bound (not latency-bound) and default off for latency-critical serving.
6. **tq3 memory budget.** tq3 OOMs at the default mem-fraction on 12 GB cards (codec working
   buffers). Either size the codec scratch from `mem_fraction_static`, or auto-lower mem-fraction to
   ~0.45 when a rq*/tq* KV dtype is selected on RDNA2.
7. **gfxGRAPH auto-enable on RDNA2.** Now that gfxgraph is system-installed, have the RDNA2 init call
   `gfxgraph.enable()` (in the scheduler process) so any force-enabled cuda-graph path is the
   graceful BridgedCUDAGraph (precise errors) rather than a raw SIGSEGV. Pair with `GFXGRAPH_GUARD=1`
   default on RDNA2 for free capture-safety.

## P2 — polish & docs
8. **Teardown traceback noise.** Every run exits with `torch/library.py:623 qualname.split("::") →
   too many values to unpack` (a torch atexit op-cache finalizer bug in this build). Harmless but
   pollutes logs/CI. Patch the torch build's `_clear_torch_ops_cache` to tolerate qualnames with
   extra `::`, or filter it in the launcher.
9. **One RDNA2 serving guide.** Consolidate the verified config into a single doc: backends
   (`triton` or `universal_broker` — best, ~60 t/s; not `aiter` raw, not `wave`), `--dtype float16`
   (bf16 auto-overridden), cuda-graph off (until P0.1), `--mem-fraction-static 0.45` for tq3, gfxgraph
   installed. Today this is spread across `rdna2_support.md` + `RDNA2_E2E_FINDINGS.md`.
10. **Tool-parser reach.** `rs_tool_parser` only accelerates base-method detectors; qwen25/pythonic/
    mistral override `parse_streaming_increment`. Either accelerate the overriding detectors or
    document the supported set (RDNA2_E2E_FINDINGS §6).
11. **Kill-by-port helper.** `pkill -f launch_server` matches the caller; ship a `scripts/kill_server.sh`
    that kills by `--port` to avoid the recurring self-kill footgun.

## Cross-engine hygiene (done this cycle, keep doing)
- Branch pruning: removed disposable agent-bot branches (bolt/copilot/rusty/jules/triangulator/
  auralis/fusion/cherry/turbo) on sglang + rust; kept upstream mirrors + canonical/active. The ~360
  sglang upstream-mirror branches are intentionally left (reproducible from upstream; not ours).
- Canonical-copy + cross-pollination: every kernel/crate keeps a copy in `projects/rust` +
  `build/kernels` with an ADOPTED.md entry; learnings back-propagated (split-K, wave64-unavailable).
