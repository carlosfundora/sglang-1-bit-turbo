
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
## 2026-04-27 - Optimization of Hot-Loop Hardware Checks and Attention Kernels
**Learning:** Dynamic calls to `torch.cuda.get_device_properties(0)` inside attention hot loops introduce significant CPU overhead. Additionally, the `waves_per_eu` triton kernel argument for RDNA2 (gfx1030) was historically kept at 1 or 4, but setting it to 2 provides a more balanced optimization target on this hardware according to docs and testing. RDNA2 doesn't have matrix instructions, so `matrix_instr_nonkdim` and `kpack` can be omitted entirely instead of passing unsupported params to triton on RDNA2.
**Action:** Replaced dynamic hardware checks in `extend_attention.py`, `decode_attention.py`, `double_sparsity_attention.py`, and `rocm_mla_decode_rope.py` with module-level cached flags using `is_rdna2()` from the arch detection util. Set `waves_per_eu` to 2 and avoided passing CDNA-specific kwargs like `matrix_instr_nonkdim` and `kpack` for RDNA2 targets to improve kernel launch efficiency and stability.
