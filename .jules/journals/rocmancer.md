
## 2024-05-24 - RDNA2 Triton Kernel and Memory Defaults Optimization
**Learning:**
Triton attention kernels were universally applying CDNA-specific optimizations (`waves_per_eu=1`, `matrix_instr_nonkdim=16`, `kpack=2`) on all HIP platforms. For RDNA2 (gfx1030) GPUs, which use Wave32 execution and lack matrix cores, these configurations are highly suboptimal and sometimes lead to crashes or severe throughput degradation. Additionally, default RDNA2 memory constraints (`mem_fraction_static`, `chunked_prefill_size`) weren't successfully overriding base defaults when missing in `server_args.py`.

**Action:**
1. Dynamically check `torch.cuda.get_device_properties(0).gcnArchName` in `extend_attention`, `double_sparsity_attention`, `prefill_attention`, and `rocm_mla_decode_rope` before kernel launches.
2. If `gfx103` is detected, remove CDNA-specific `extra_kargs` and tune `num_warps` (typically 2 for Wave32) and `waves_per_eu` appropriately.
3. Update `server_args.py` to properly propagate `mem_fraction_static` and `chunked_prefill_size` from `gfx1031_defaults` when initializing an RDNA2 server.
