# sglang-1-bit-turbo — Engineering Notes

**Project**: [sglang-1-bit-turbo](https://github.com/carlosfundora/sglang-1-bit-turbo)
**Target**: PrismML Bonsai 1-bit GGUF + EAGLE3 speculative decoding on ROCm gfx1030
**Source**: Consolidated from THOTH research (`/home/local/Projects/THOTH/`) and active development
**Hardware**: AMD Radeon RX 6700 XT (gfx1030/gfx1031), 12GB VRAM, ROCm 7.2

---

## Proven Launch Configuration (THOTH Docker)

```bash
# ENV
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE=1

python -m sglang.launch_server \
  --model-path /mnt/ai/models/registry/PrismML/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /mnt/ai/models/registry/PrismML/Bonsai-1.7B-EAGLE3 \
  --speculative-draft-load-format safetensors \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --attention-backend triton \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-cuda-graph-padding \
  --disable-flashinfer-autotune \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.35 \
  --max-running-requests 1 \
  --schedule-policy fcfs \
  --host 0.0.0.0 --port 30000
```

**Key conservative settings**:
- `--max-running-requests 1` — single request avoids batching shape bugs
- `--disable-cuda-graph` — ROCm CUDA graph capture has issues
- `--disable-overlap-schedule` — simpler scheduling, more stable
- `--mem-fraction-static 0.35` — conservative VRAM, avoid OOM
- `--speculative-eagle-topk 1` — minimal draft branching
- `SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE=1` — avoids embed sharing crash

---

## Critical Fixes Applied (commit e8da8e785)

### 1. Rotary Embedding CPU Index Select
**Problem**: GPU `index_select` on cos_sin_cache produces wrong results on ROCm
**Fix**: `_index_cos_sin_cache()` — CPU-based index_select for HIP
**File**: `layers/rotary_embedding/base.py`

### 2. Triton Backend Prefix Sum
**Problem**: GPU cumsum for KV indptr unreliable on ROCm
**Fix**: `_fill_prefix_sum_buffer()` — optional CPU cumsum path
**File**: `layers/attention/triton_backend.py`

### 3. EAGLE Draft Position Recompute
**Problem**: `forward_batch.positions` stale/incorrect in EAGLE draft extend on HIP
**Fix**: Recompute positions from `extend_prefix_lens_cpu + extend_seq_lens_cpu`
**File**: `speculative/eagle_worker.py` (line ~971)

### 4. EAGLE Capture CPU Softmax
**Problem**: GPU softmax in `capture_for_decode` unreliable on HIP
**Fix**: CPU-based softmax + topk for draft token selection
**File**: `speculative/eagle_worker.py` (line ~1041)

### 5. TQ Memory Pool Split Storage
**Problem**: `indexSelectSmallIndex` faults during packed KV buffer compression on HIP
**Fix**: Split TQ storage into separate packed + norms buffers on HIP
**File**: `mem_cache/memory_pool.py`

### 6. GGUF Type Remapping (Q1_0 collision)
**Problem**: PrismML type IDs 40/41 collide with TurboQuant NVFP4/TQ3_0
**Fix**: Remap 40→42 (Q1_0), 41→43 (Q1_0_G128) in GGUF reader
**File**: `utils/gguf_compat.py`

### 7. CPU Bridge for Q1 Matmul
**Problem**: No MMQ kernel for Q1, MMVQ limited to batch ≤ 8
**Fix**: CPU dequantize + matmul as fallback (slow but correct)
**Now superseded**: GPU dequantize + GPU matmul (our fix, 2ms/layer)

### 8. aiter Backend Auto-Selection
**Problem**: HIP auto-selected `aiter` even when not installed
**Fix**: Fallback to `triton` unless `SGLANG_USE_AITER` explicitly set
**File**: `server_args.py`

---

## Q1 Matmul Dispatch Architecture

```
fused_mul_mat_gguf(x, qweight, qtype)
  ├── batch ≤ 8:  GPU MMVQ kernel (ggml_mul_mat_vec_a8)  ~0.2ms/layer
  ├── batch > 8:  GPU dequantize + GPU matmul             ~2ms/layer
  └── CPU forced: CPU dequantize + CPU matmul              ~slow (fallback)
```

**Q1 dequantization**: `bit ? +scale : -scale` (trivially cheap)
- GPU kernel: `dequantize_q1_0_g128()` in `dequantize.cuh`
- CPU Python: `_PrismQ10G128Compat.dequantize()` in `gguf_compat.py`
- Both verified working 2026-04-05

---

## GPU Dequant+Matmul Fix (Our Key Contribution)

The original THOTH approach used a CPU bridge (dequant on CPU → matmul on CPU → transfer back to GPU) for batch > 8 Q1 matmuls. This was slow and fragile.

**Our fix** in `gguf.py::fused_mul_mat_gguf()`:
- Uses `ggml_dequantize()` GPU kernel to decompress Q1 → fp16 **on GPU**
- Then standard `torch.matmul()` **on GPU** — no CPU round-trip
- Mirrors llama.cpp's cuBLAS fallback architecture for batch > MMVQ_MAX_BATCH_SIZE

**Performance**:
- GPU MMVQ (batch ≤ 8): ~0.2ms/layer — used for single-token decode
- GPU dequant+matmul (batch > 8): ~2ms/layer — used for EAGLE extends
- CPU bridge (old): ~10-50ms/layer — eliminated

**Env escape hatch**: `SGLANG_PRISM_Q1_CPU_FALLBACK=1` forces the old CPU bridge if needed.

---

## Argparse Prefix Ambiguity (Critical Gotcha)

sglang uses Python argparse, which supports prefix matching. When `--speculative-draft-*` args are present, shorter args like `--attention-backend` or `--log-level` can be ambiguous.

**Solutions**:
- Use `=` syntax: `--attention-backend=triton` instead of `--attention-backend triton`
- Use full arg names: `--speculative-draft-model-path` not `--speculative-draft`
- Use env vars where possible: `SGLANG_LOG_LEVEL=info` instead of `--log-level info`

---

## Bonsai Speed Reference (llama.cpp GPU, NOT sglang)

| Model | Prompt (pp512) | Generation (tg128) | VRAM |
|-------|---------------|-------------------|------|
| Bonsai-1.7B Q1_0_G128 | 2096.87 t/s | 75.60 t/s | ~0.5 GB |
| Bonsai-4B Q1_0_G128 | 856.64 t/s | 120.66 t/s | ~1.0 GB |
| Bonsai-8B Q1_0_G128 | 453.90 t/s | 91.56 t/s | ~1.5 GB |

**Bonsai-4B is the sweet spot**: 121 t/s gen, best VRAM/perf ratio

### sglang-1-bit-turbo Benchmarks (ROCm gfx1030, RX 6700 XT)

| Config | Gen Speed | Accept Rate | Notes |
|--------|-----------|-------------|-------|
| Bonsai-1.7B + EAGLE3 (topk=4, steps=3) | ~9 t/s | 8% | Greedy verify only, spec sampling not ported |
| Bonsai-1.7B + EAGLE3 (with spec kernel) | TBD | TBD | Pending HIP spec sampling kernel |

**sglang vs llama.cpp gap sources**:
- Python/PyTorch dispatch overhead
- Triton attention (not optimized for gfx1030)
- No CUDA graphs on ROCm
- Per-call buffer allocation
- sglang official: "gguf quantization is not fully optimized yet"

---

## Known Unresolved Issues

### Batch Extend Shape Bug (ACTIVE)
- Shape mismatch during EAGLE target extend with batch > 8
- Manifests as `RuntimeError: shape '[N, -1, 128]' is invalid for input of size M`
- Occurs in attention reshape AFTER matmul (not in Q1 dispatch itself)
- THOTH status: "Not yet resolved as of 2026-04-04"
- **Workaround**: `--max-running-requests 1`, conservative settings

### ROCm Version Sensitivity
- ROCm 5.7: HSA_OVERRIDE works
- ROCm 6.2: HSA_OVERRIDE broken (invalid device function)
- ROCm 7.2: HSA_OVERRIDE works again
- **Action**: Pin ROCm version, regression test after upgrade

### TurboQuant + Hybrid RNN
- TQ3_0 V-cache corrupts output on LFM2 (hybrid RNN/Transformer)
- Works fine on standard Transformer (Qwen3, LLaMA)

---

## Bonsai Model Facts
- Architecture: `qwen3` (NOT bitnet)
- Loaded via: `Qwen3ForCausalLM`
- Q1_0_G128: 128 elements/block, 18 bytes (2B fp16 scale + 16B sign bits)
- Ternary dequant: `bit==1 ? +d : -d`
- Type IDs: 42 (Q1_0), 43 (Q1_0_G128) — remapped from PrismML 40/41

---

## FasterDecoding/Medusa Audit Findings (2026-04-08)

Comprehensive audit of the [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) reference
implementation (`/home/local/ai/repos/training/Medusa/`). Key techniques adopted into sglang-1-bit-turbo:

### Adopted: Typical Acceptance (Entropy-Adaptive Candidate Generation)

**Source:** `medusa/model/utils.py` lines 227-256, `get_typical_one_token()`

Instead of greedy argmax for draft candidate selection, uses entropy-adaptive thresholding:
```
probs = softmax(logits)
entropy = -Σ(probs × log(probs + 1e-5))
threshold = min(posterior_threshold, exp(-entropy) × posterior_alpha)
```
- Low-entropy (confident model) → strict threshold → fewer but higher-quality candidates
- High-entropy (uncertain model) → looser threshold → more diversity, better exploration
- Default: `posterior_threshold=0.09`, `posterior_alpha=0.3` (α ≈ √τ empirically)
- **Placement:** In candidate generation (`medusa_model.predict_tokens(typical=True)`), NOT post-verify.
  Post-verify rejection breaks cache/index coherency — verify() mutates request state in-place.

### Adopted: Cached Tree Buffers

**Source:** `medusa/model/medusa_model.py` — buffer caching pattern

FasterDecoding caches `medusa_buffers` (attn mask, tree indices, position IDs, retrieve indices) and
reuses across iterations unless `medusa_choices` changes. We now pre-compute and cache:
- `_cached_mask_per_req_flat`: flattened single-request mask [T*T]
- `_cached_mask_tiled_gpu`: pre-tiled GPU tensor [max_bs * T * T]
- `_cached_mask_np_3d`: reshaped view [max_bs, T, T]

This eliminates per-call `np.tile()` + `torch.from_numpy()` conversion overhead.

### Adopted: Non-Blocking KV Copy + Stream Sync

**Source:** `medusa/model/kv_cache.py` — `dst.copy_(tgt, non_blocking=True)`

All buffer copies use `non_blocking=True` for DMA overlap. Added explicit
`torch.cuda.current_stream().synchronize()` before verify to prevent race conditions
where `reconstruct_indices_from_tree_mask()` reads stale buffer data.

### Adopted: PALTROW Mixed CPU/GPU Placement

**Novel extension** — not in FasterDecoding. PALTROW heads run on CPU with pinned memory.
Auto-detection from tiered config + OOM auto-fallback. Integrated with AMD SAM/ReBAR detection
for optimal pinned DMA bandwidth (~14 GB/s).

### Noted: Cumprod Prefix Validation

**Source:** `medusa/model/utils.py` line 464

```python
candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
```
Ensures accepted tokens form contiguous prefix (can't skip tokens). Already handled by
SGLang's `NgramVerifyInput.verify()` — no additional implementation needed.

### Noted: KV Cache Pre-Allocation (current_length on CPU)

**Source:** `medusa/model/kv_cache.py` lines 94-113

Their KV cache pre-allocates one contiguous tensor for all layers and keeps `current_length`
on CPU for zero-sync access. SGLang's RadixCache handles this differently (token-level pooling),
so this pattern doesn't directly apply. But the CPU-length-tracking insight is relevant for
any auxiliary KV management we add.

### Noted: TOPK=10 Hardcoded

FasterDecoding uses `TOPK=10` candidates per head position. Their comment says "10 is sufficient."
Our `--medusa-topk` defaults to 1 (greedy), but the infrastructure supports higher values.

### NOT Adopted: Batch > 1 Verification

FasterDecoding explicitly asserts `batch_size == 1` in `medusa_generate()`. Batched tree verification
would require significant rework of their approach. SGLang's `NgramVerifyInput` already handles
batched verification, so this is not a limitation for us.

### Key Reference Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `medusa/model/utils.py` (593 lines) | Tree buffers, typical acceptance, KV compaction | 32-125 (tree), 227-256 (typical) |
| `medusa/model/medusa_model.py` (409 lines) | Generation loop, candidate flow | 197-340 (medusa_generate) |
| `medusa/model/kv_cache.py` (113 lines) | Pre-allocated KV, non_blocking copy | 38-50 (copy), 94-113 (init) |
| `medusa/model/medusa_choices.py` | Pre-computed tree structures | mc_sim_7b_63 (63 candidates) |
| `train_medusa_heads.py` (438 lines) | Chunked CE loss, head-specific shifts | 138-171 (loss) |

---

## AMD SAM / ReBAR / GTT Integration Notes

### Hardware Detection
- SAM active when PCIe BAR ≥ VRAM (read from `/sys/class/drm/card*/device/mem_info_vis_vram_total`)
- GTT pool from `/sys/class/drm/card*/device/mem_info_gtt_total` (typically ~half system RAM)
- Driver detection: symlink at `device/driver` contains "amdgpu"
- Auto-detect runs once at MedusaWorker init, cached in `_sam_info` dict

### Bandwidth Characteristics (RX 6700 XT, PCIe 4.0 x16)
- Pinned DMA: ~14 GB/s (with SAM active)
- Pageable copy: ~8 GB/s
- GTT access from GPU: ~13.8 GB/s (measured)
- Hidden state D2H for PALTROW: ~4 KB per batch element (negligible at any bandwidth)

### CLI Args
- `--sam-enabled auto|true|false` — auto-detect or force
- `--gtt-pool-mb N` — override detected GTT pool size
- `--paltrow-pin-memory` / `--no-paltrow-pin-memory` — control pinned allocation
