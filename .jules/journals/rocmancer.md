
## 2024-05-24 - RDNA2 Triton Kernel and Memory Defaults Optimization
**Learning:**
Triton attention kernels were universally applying CDNA-specific optimizations (`waves_per_eu=1`, `matrix_instr_nonkdim=16`, `kpack=2`) on all HIP platforms. For RDNA2 (gfx1030) GPUs, which use Wave32 execution and lack matrix cores, these configurations are highly suboptimal and sometimes lead to crashes or severe throughput degradation. Additionally, default RDNA2 memory constraints (`mem_fraction_static`, `chunked_prefill_size`) weren't successfully overriding base defaults when missing in `server_args.py`.

**Action:**
1. Dynamically check `torch.cuda.get_device_properties(0).gcnArchName` in `extend_attention`, `double_sparsity_attention`, `prefill_attention`, and `rocm_mla_decode_rope` before kernel launches.
2. If `gfx103` is detected, remove CDNA-specific `extra_kargs` and tune `num_warps` (typically 2 for Wave32) and `waves_per_eu` appropriately.
3. Update `server_args.py` to properly propagate `mem_fraction_static` and `chunked_prefill_size` from `gfx1031_defaults` when initializing an RDNA2 server.

## 2024-05-24 - CI Docs Advisory Failure Note
**Learning:**
SGLang's CI requires a `CHANGELOG.md` update for PRs that modify code files. When the `CHANGELOG.md` is updated, the changes must include a `Documentation` or `Docs` section outlining modified documentation, or explicitly stating `- None required`. The recent `docs-policy` check failed because no `CHANGELOG.md` was updated in the PR.

**Action:**
Update the `CHANGELOG.md` file under the current unreleased version with a note about the AMD RDNA2 optimizations and the required `- None required` documentation block.

## 2024-05-24 - CI Dependency Failure
**Learning:**
The CI job `stage-a-test-cpu` failed with `ModuleNotFoundError: No module named 'tabulate'`. This appears to be a missing dependency in the CI test runner (`test/run_suite.py`) rather than a direct consequence of the RDNA2 Triton kernel optimizations I introduced. The script `test/run_suite.py` likely expects `tabulate` to be installed but it was missing from the `uv pip install -e "python[dev]"` environment resolving phase.

**Action:**
Add `tabulate` to the development dependencies list (either `requirements-test.txt` or within `pyproject.toml`'s dev dependencies section depending on project structure) to fix the `stage-a-test-cpu` job.

## 2024-05-24 - CI Docs Advisory Failure Update
**Learning:**
I updated `CHANGELOG.md` correctly but there were multiple undocumented file modifications (`test/srt/test_p_eagle_wiring.py` and `python/pyproject.toml`) that the docs policy script considers missing from the changelog. The CI script failed because we didn't specify `- None required`. Wait, actually I did add `- None required`, but I used `- None required` without a period, or it was in the wrong section, or it detected new journal files created by my prompt logic (`.jules/reports/research/*`) that were not tracked. The exact error is:
`- Changed documentation file not listed in CHANGELOG.md: .jules/journals/triangulator-forge.md`
`- Changed documentation file not listed in CHANGELOG.md: .jules/reports/research/repo-triangulation-20260427-024844.md`
`- Changed documentation file not listed in CHANGELOG.md: .jules/reports/research/repo-triangulation-20260427-120000.md`

**Action:**
Update `CHANGELOG.md` to properly document the new `.jules/reports/*` files, since any markdown file modification triggers the policy checker.

## 2024-05-24 - Docs Policy CI Refinement
**Learning:**
I previously encountered issues with the docs-policy CI job failing because it detected markdown file modifications inside `.jules/`. I thought I had solved this by resetting the `.jules/` directory from `git`. However, the CI output still reports `- Changed documentation file not listed in CHANGELOG.md: .jules/journals/rocmancer.md.`
This happened because `git reset HEAD .jules/` removes the files from the index (unstages them), but they might still be untracked or the commit itself still contains the previously committed versions of `.jules` files if I only did `git commit --amend` without actually dropping them from the commit tree.

**Action:**
I need to definitively purge the `.jules/` files from the *branch history* (or the last commit) so they don't show up in the `origin/main...HEAD` diff. I will use `git rm --cached -r .jules/` and amend the commit, ensuring they are ignored.

## 2026-05-17 - Universal Broker Capacity Tiering and Spill Hardening
**What changed:**
1. Upgraded `UniversalKVBroker` to enforce explicit GPU and RAM budgets with active hot-to-warm demotion and warm-tier eviction.
2. Added tier metrics (`gpu_used_bytes`, `ram_used_bytes`, capacities, hot/warm block counts, spilled/evicted block counters) so runtime can reason about broker pressure.
3. Added spill payload support for structured broker records (`compressed_hot` + optional residual) and explicit spill-key discard handling.
4. Tightened broker materialization with model/layer ownership validation to avoid cross-model misuse.

**Why this matters on gfx1030:**
Consumer RDNA2 cards are capacity-constrained, so broker metadata without budget control quickly turns into hidden VRAM pressure. Deterministic demotion/eviction behavior and metrics are mandatory prerequisites before wiring deeper Rust/TurboQuant kernels.

**Remaining risk:**
Compression/decompression is still scaffold-level in Python tensors; true throughput/latency wins require Rust/HIP fused paths and direct integration with quantized KV kernels.

## 2026-05-25 - ROCm Segfault Root Cause Analysis: gfx1031/gfx1030 Arch Identity Conflict

### Problem
SGLang server crashes during request-time prefill; plain Python segfaults on first `torch.zeros(device='cuda')` even after `torch.cuda.init()` succeeds. Reproduces across all interpreters and with `python -S`.

### Definitive Root Cause: Three-Way Arch Identity Conflict

The system has three parties that must agree on GPU architecture, and they don't:

| Actor | Reports/Targets |
|---|---|
| Physical GPU (rocm-smi) | gfx1031 (RX 6700 XT, device 0x73df) |
| HSA_OVERRIDE_GFX_VERSION=10.3.0 | Forces HSA to report gfx1030 |
| PyTorch binary (torch.cuda.get_arch_list()) | ['gfx1031'] only |

### Why It Crashes: Fill Kernel Dispatch Failure

- `torch.empty(device='cuda')` → works: HIP malloc, no kernel needed
- `torch.zeros(device='cuda')` → SEGFAULT: requires PyTorch's bundled fill kernel
- Raw `hipMemset()` → works: lives in HIP runtime, not PyTorch user kernels
- `torch.cuda.init()` → works: calls hipInit + creates device context, no fill dispatch

When PyTorch dispatches `fill_` / `zero_()`:
1. Looks up pre-compiled kernel for current arch (HSA says gfx1030)
2. Finds NO gfx1030 kernel in its binary (only compiled gfx1031 kernels exist)
3. Kernel lookup → null/invalid pointer → segfault on first dispatch

### The Double-Bind
- **Without override**: PyTorch works, rocBLAS aborts ("Cannot read TensileLibrary.dat: Illegal seek for GPU arch: gfx1031"). No gfx1031 Tensile library exists.
- **With override (10.3.0=gfx1030)**: rocBLAS GEMM works, PyTorch fill kernel crashes.

rocBLAS available: gfx1030, gfx1101, gfx1102, gfx1100, gfx1150, gfx1151, gfx1200, gfx1201, gfx908, gfx90a, gfx942, gfx950. No gfx1031. Override is mandatory.

### Why Server Gets Past Weight Load But Crashes at Prefill

Weight loading: `model_runner.py:1599` uses `torch.empty(shape, device=...)` → safe, no fill kernel.
KV buffer: actual K/V tensors allocated with `torch.empty` → safe.
Prefill: attention backends initialize indptr / offset / mask tensors with `torch.zeros(device='cuda')` on first request (e.g., `flashmla_backend.py:561 kv_indptr`, `extend_attention.py:202`, `triton_ops/rdna2/extend_attention.py:157`) → crash.

CPU `torch.zeros` (no device arg) in memory_pool.py for req_to_token management: safe.

### Fix Hierarchy

**Fix 1 (Proper, permanent)**: Reinstall PyTorch with gfx1030 kernels bundled. The current nightly `2.13.0a0+git53bbebe` only has gfx1031. Use official ROCm 6.x/7.x wheel or rebuild with `PYTORCH_ROCM_ARCH=gfx1030`. Keep `HSA_OVERRIDE_GFX_VERSION=10.3.0` for rocBLAS.

**Fix 2 (Immediate workaround)**: Monkey-patch `torch.zeros` / `fill_` to use `hipMemset` via ctypes for cuda tensors. Unblocks development while proper PyTorch is sourced. See `universal_kv/hip_zero_patch.py` (to be created).

**Fix 3 (Investigational)**: Test if `HSA_OVERRIDE_GFX_VERSION=10.3.0` + `ROC_USE_FGS_KERNARG=0` + `HSA_ENABLE_SDMA=0` changes kernel dispatch. Unlikely to help since the issue is pre-compiled binary arch mismatch, not runtime behavior.

### Validation Commands
```bash
# Confirms root cause:
python -u -c "import torch; torch.cuda.init(); torch.empty(4,device='cuda')"  # works
python -u -c "import torch; torch.cuda.init(); torch.zeros(4,device='cuda')"  # segfaults
HSA_OVERRIDE_GFX_VERSION='' python -u -c "import torch; torch.zeros(4,device='cuda')"  # works but GEMM aborts
```

### Next Actions
1. Source a PyTorch ROCm wheel with gfx1030 kernels (pip install torch --index-url rocm7.x)
2. OR build local PyTorch with PYTORCH_ROCM_ARCH=gfx1030
3. Deploy `hip_zero_patch.py` as immediate unblock for OpenCoder-8B testing
4. After PyTorch fix: validate full prefill path with `torch.zeros` on CUDA

## 2026-05-25 - ai/.venv Bridge Audit and Cleanup

### Problem
`/home/local/ai/.venv` was resolving packages from a mixed set of locations:
- system ROCm stack from `/usr/local/lib/python3.12/dist-packages`
- `sglang` from stale `/mnt/ai/forks/sglang-1-bit-turbo`
- `sentencepiece` from user site `/home/local/.local/lib/python3.12/site-packages`
- multiple dead bridge entries in `engine-bridge.pth`

### Findings
- `pyvenv.cfg` intentionally sets `include-system-site-packages = true`, so the system ROCm stack is part of the design for this venv.
- The manual bridge file `/home/local/ai/.venv/lib/python3.12/site-packages/engine-bridge.pth` still referenced several missing venvs and an old SGLang fork path.
- The active ATOM-RS working tree is `/home/local/ai/build/wip/ATOM-RS/sglang-1-bit-turbo`, and it is distinct from `/mnt/ai/forks/sglang-1-bit-turbo`.

### Action
Rewrote `engine-bridge.pth` to keep only the active ATOM-RS SGLang paths:
- `/home/local/ai/build/wip/ATOM-RS/sglang-1-bit-turbo/python`
- `/home/local/ai/build/wip/ATOM-RS/sglang-1-bit-turbo/sgl-kernel/python`

### Remaining cleanup
- `sentencepiece` is still resolving from user site and should be installed into `/home/local/ai/.venv` directly so the venv no longer depends on `/home/local/.local`.
