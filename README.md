<div align="center"><img src="assets/sgl-1-bit-turbo-eagle.png" alt="SGLang 1-Bit Turbo" width="600"></img></div>




[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/carlosfundora/sglang-1-bit-turbo/blob/main/LICENSE)

---

## A fork of [SGLang](https://github.com/sgl-project/sglang) optimized for TurboQuant and EAGLE3 speculative decoding on AMD ROCm gfx1030x GPUs, with support for PrismML's 1-bit quantized models.

SGLang 1-Bit Turbo EAGLE extends upstream SGLang with a stack of features designed to run aggressively quantized models at practical speeds on AMD RDNA2/RDNA3 hardware (RX 6000/7000 series):

### 🧊 TurboQuant KV Cache Compression
Extreme KV cache quantization modes that dramatically reduce VRAM usage, enabling larger context and bigger models on memory-constrained GPUs.

| Mode | Bits per element | VRAM savings vs FP16 |
|------|-----------------|---------------------|
| `--kv-cache-dtype tq4` | 4-bit | ~75% |
| `--kv-cache-dtype tq3` | 3-bit | ~81% |
| `--kv-cache-dtype tq2` | 2-bit | ~87% |

### 🦅 EAGLE3 + Medusa + 9 Speculative Algorithms on ROCm
Full speculative decoding suite ported to AMD GPUs:
- **9 algorithms**: EAGLE3, P_EAGLE, NGRAM (27.8 t/s ✅), MEDUSA, P_CASCADE, CHIMERA, SAGUARO, TQ5_X, STANDALONE
- **NGRAM**: Stable at **27.8 t/s** (1.6× baseline) — zero extra compute, 80/80 stress test
- **MEDUSA**: 2–7 parallel MLP draft heads with tree verification + DraftPreFilter adaptive pre-rejection
- **Typical Acceptance**: Entropy-adaptive candidate generation from FasterDecoding/Medusa — low-entropy (confident) distributions use strict thresholds, high-entropy (uncertain) ones explore more broadly
- **DraftPreFilter**: Novel 3-layer pre-rejection filter with adaptive self-tuning thresholds
- **HIP C++ probabilistic sampling kernel** — self-contained port, no flashinfer dependencies
- **Triton kernel fallback** — device-agnostic `@triton.jit` for any GPU
- **Pure PyTorch fallback** — universal last-resort using only tensor ops
- **Three-tier automatic fallback**: HIP C++ → Triton → PyTorch, detected at import time

### 🧠 PALTROW Mixed CPU/GPU Head Placement
**P**inned-memory **A**uxiliary **L**atency-**T**olerant **R**elocated **O**n-CPU **W**orkers.
Medusa screen/bloom/filter heads run on CPU with pinned host memory, freeing GPU VRAM for KV cache and precision draft heads:
- **Auto-detection**: Screen/bloom heads identified from tiered config or keyword matching
- **OOM auto-fallback**: Heads that fail GPU allocation are automatically promoted to PALTROW
- **AMD SAM/ReBAR integration**: Auto-detects Smart Access Memory for optimal DMA bandwidth (~14 GB/s pinned)
- **GTT pool awareness**: Reads AMD GTT (Graphics Translation Table) pool from sysfs for memory planning
- **Cached tree buffers**: Pre-computed tree attention masks avoid per-call numpy→torch conversion

### 📦 PrismML Q1_0_G128 1-Bit GGUF Model Support
Native serving of [Bonsai](https://huggingface.co/PrismML) Q1_0_G128 1-bit GGUF models through sglang's weight loading pipeline with GPU dequantization kernels (dp4a), bridging the GGUF quantization ecosystem with sglang's high-performance runtime.

### 🔧 Pre-Built ROCm sgl_kernel
Ships a pre-compiled `sgl_kernel` binary for ROCm gfx1030 (RDNA2), so you can skip the build step entirely. Includes all speculative decoding kernels:
- `verify_tree_greedy`
- `build_tree_kernel_efficient`
- `tree_speculative_sampling_target_only` (HIP port)
- All standard sglang ops (activation, MoE, rotary, allreduce, etc.)

---

## Quick Start (ROCm)

### Prerequisites
- AMD GPU with ROCm support (tested on gfx1030 / RX 6900 XT)
- ROCm 6.x with PyTorch nightly (`torch` with ROCm backend)
- Python 3.12+

### Install
```bash
git clone https://github.com/carlosfundora/sglang-1-bit-turbo.git
cd sglang-1-bit-turbo

# Install the runtime
pip install -e "python[all]"

# Install the pre-built kernel (ROCm gfx1030)
pip install -e sgl-kernel
```

### Serve a PrismML 1-Bit Model with EAGLE3
```bash
python -m sglang.launch_server \
  --model-path /path/to/Bonsai-1.7B.gguf \
  --attention-backend triton \
  --kv-cache-dtype tq4 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /path/to/Bonsai-1.7B-EAGLE3/weights \
  --mem-fraction-static 0.28 \
  --port 30400
```

### Environment Variables (ROCm)
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0    # Required for gfx1030
export PYTORCH_ROCM_ARCH=gfx1030
export SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE=1  # Required for GGUF + EAGLE3
```

---

## Building sgl_kernel from Source (ROCm)

If you need to rebuild the kernel (e.g., for a different GPU target):

```bash
cd sgl-kernel
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export AMDGPU_TARGET=gfx1030
python setup_rocm.py build_ext --inplace
```

Verify:
```bash
python -c "from sgl_kernel import common_ops; print('tree_speculative_sampling_target_only' in dir(common_ops))"
```

---

## Architecture: Speculative Sampling Fallback Chain

On CUDA, sglang uses a flashinfer-backed C++ kernel for probabilistic tree speculative sampling. On ROCm, flashinfer is unavailable. This fork implements a three-tier fallback so EAGLE speculative decoding works everywhere:

```
┌─────────────────────────────────────────────────┐
│  HIP C++ Kernel (speculative_sampling.hip)      │ ← Fastest, gfx1030
│  Self-contained port, no flashinfer dependency  │
├─────────────────────────────────────────────────┤
│  Triton Kernel (speculative_sampling_triton.py) │ ← Fast, any GPU
│  @triton.jit, device-agnostic                   │
├─────────────────────────────────────────────────┤
│  PyTorch Fallback (speculative_sampling_pytorch)│ ← Universal
│  Pure tensor ops, works on any backend          │
└─────────────────────────────────────────────────┘
```

Detection is automatic — the best available backend is selected at import time.

---

## Fork Features

| Feature | Details |
|---------|---------|
| **9 Speculative Decoding Algorithms** | EAGLE3, P_EAGLE, NGRAM, MEDUSA, P_CASCADE, CHIMERA, SAGUARO, TQ5_X, STANDALONE |
| **NGRAM Speculation — 27.8 t/s stable** | N-gram trie speculation with BFS/PROB tree search, 1.6× over baseline, 80/80 stress test |
| **MEDUSA Multi-Head Decoding** | 2–7 parallel MLP draft heads with tree verification, DraftPreFilter integration |
| **Typical Acceptance (entropy-adaptive)** | Entropy-aware candidate generation: `threshold = min(τ, exp(-H) × α)` — from FasterDecoding/Medusa |
| **PALTROW CPU/GPU Head Placement** | Screen/bloom heads on CPU with pinned memory, precision heads on GPU — OOM auto-fallback |
| **AMD SAM/ReBAR Auto-Detection** | Auto-detects Smart Access Memory, PCIe BAR size, GTT pool — optimizes DMA for PALTROW |
| **Cached Tree Buffers** | Pre-computed tree attention masks cached on GPU, avoiding per-call numpy→torch overhead |
| **DraftPreFilter (novel)** | 3-layer pre-rejection: L0 n-gram surprisal, L1 screen inversion, L2 head agreement — adaptive self-tuning |
| **TQ5_X Ghost-Draft** | HSA zero-copy ghost-draft speculative decoding for AMD gfx103x |
| **EAGLE3 on ROCm** | Full probabilistic tree sampling via 3-tier fallback (HIP C++ → Triton → PyTorch) |
| **Self-contained HIP C++ sampling kernel** | Port of `tree_speculative_sampling_target_only` — no flashinfer dependency |
| **`top_k` / `top_p` renorm fallbacks** | PyTorch implementations with cached capability probe and kth-pivot tie-correct top-k |
| **TurboQuant KV cache** | TQ4 (4-bit), TQ3 (3-bit), TQ2 (2-bit) — up to 87% VRAM savings vs FP16 |
| **PrismML 1-bit GGUF model serving** | Native [Bonsai](https://huggingface.co/PrismML) Q1_0_G128 1-bit models via GPU dequant |
| **Pre-built ROCm `sgl_kernel`** | Compiled `.so` for gfx1030 (RDNA2) with all speculative decoding ops |
| **Radix Cache** | Prefix-sharing token reuse for faster multi-turn and batch inference |
| **Triton Kernels** | Device-agnostic attention, dequant, and tree ops with gfx1030 fallbacks |
| **Consumer AMD GPU focus** | Optimized for RX 6000/7000 series (12–16 GB VRAM) |

---

## Tested Configurations

| GPU | Model | Algorithm | Draft Model | KV Cache | Throughput | Status |
|-----|-------|-----------|-------------|----------|------------|--------|
| RX 6700 XT (12GB) | Bonsai-1.7B (Q1_0_G128 GGUF) | NGRAM | — | fp16 | **27.8 t/s** | ✅ Stable (80/80 stress) |
| RX 6700 XT (12GB) | Bonsai-1.7B (unpacked fp16) | MEDUSA (3-head) | contrastive-3head | fp16 | **9.5 t/s** | ✅ Coherent output |
| RX 6700 XT (12GB) | Bonsai-1.7B (Q1_0_G128 GGUF) | Baseline (no spec) | — | fp16 | **17.2 t/s** | ✅ Baseline |
| RX 6700 XT (12GB) | Bonsai-1.7B (unpacked fp16) | EAGLE3 | Bonsai-EAGLE3 | TQ4 | — | 🔴 0% acceptance (untrained model) |

---

## Upstream SGLang

This fork is based on [SGLang](https://github.com/sgl-project/sglang), a high-performance serving framework for large language models by [LMSYS](https://lmsys.org/about/). For general SGLang documentation, features, and community:

- [SGLang Documentation](https://docs.sglang.io/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Slack](https://slack.sglang.io/)

---

## Speculative Decoding Suite — CLI Reference

### Supported Algorithms

| Algorithm | Flag | Description | Needs Draft Model |
|-----------|------|-------------|-------------------|
| `EAGLE3` | `--speculative-algorithm EAGLE3` | 3-layer feature extraction + 1-layer decoder | ✅ |
| `P_EAGLE` | `--speculative-algorithm P_EAGLE` | Parallel EAGLE3 via mask_hidden | ✅ |
| `NGRAM` | `--speculative-algorithm NGRAM` | Statistical trie-based, zero extra compute | ❌ |
| `P_CASCADE` | `--speculative-algorithm P_CASCADE` | Adaptive DyTC router: L1=EAGLE, L2=reduced, L3=ngram | ✅ |
| `MEDUSA` | `--speculative-algorithm MEDUSA` | 2–7 parallel MLP draft heads + tree verify | Medusa heads (`--medusa-model-path`) |
| `CHIMERA` | `--speculative-algorithm CHIMERA` | Fused P-EAGLE + Hydra + DyTC + SSD (experimental) | ✅ |
| `TQ5_X` | `--speculative-algorithm TQ5_X` | TurboQuant 5 eXtended: HSA zero-copy ghost-draft (AMD gfx103x) | ❌ |
| `SAGUARO` | `--ssd-enable` (wraps any algorithm) | LRU draft caching, async pre-generation wrapper | ❌ |
| `STANDALONE` | `--speculative-algorithm STANDALONE` | Independent draft model (no shared weights) | ✅ |

### Common Flags

```
--speculative-draft-model-path PATH    Path to EAGLE3/P_EAGLE draft model weights
--speculative-eagle-topk K             Top-k candidates per draft step (default: auto)
--speculative-num-steps N              Max draft steps per round (default: auto)
--speculative-num-draft-tokens N       Max total draft tokens (default: auto)
--disable-overlap-schedule             Required for P_CASCADE, MEDUSA, CHIMERA
```

### Algorithm-Specific Flags

```
# NGRAM
--speculative-ngram-max-trie-depth 4
--speculative-ngram-match-type BFS     # or PROB

# MEDUSA
--medusa-model-path PATH               Path to trained Medusa head weights
--medusa-num-heads 3                   Number of parallel draft heads (2-7)
--medusa-topk 1                        Top-k per head
--medusa-typical-acceptance            Enable entropy-adaptive candidate generation
--medusa-posterior-threshold 0.09      Fixed ceiling for typical acceptance
--medusa-posterior-alpha 0.3           Entropy scaling (≈ √threshold)
--medusa-tree-structure linear         Tree topology: linear | mc_sim_63 | auto
--medusa-paltrow-heads auto            PALTROW CPU offload: auto | none | 0,1,2

# AMD SAM / ReBAR / PALTROW hardware
--sam-enabled auto                     SAM status: auto | true | false
--gtt-pool-mb 33600                    GTT pool size override (MB, auto-detected)
--paltrow-pin-memory                   Use pinned memory for CPU heads (default: on)
--no-paltrow-pin-memory                Disable pinned memory

# TQ5_X (AMD gfx103x)
--speculative-algorithm TQ5_X         HSA zero-copy ghost-draft

# SAGUARO (wraps any algorithm)
--ssd-enable                           Enable LRU draft caching

# CHIMERA (experimental)
--chimera-num-steps 6
--chimera-ssd-enable
--chimera-level 1|2|3                  Force cascade level (omit for dynamic)
```

### DraftPreFilter (Novel Architecture)

DraftPreFilter is a 3-layer pre-rejection filter that drops low-quality draft tokens **before** expensive GPU verification, improving throughput on weak heads:

| Layer | Name | Mechanism | When Active |
|-------|------|-----------|-------------|
| L0 | N-gram surprisal | CPU trie lookup, drops tokens with surprisal > τ₀ | Always |
| L1 | Screen inversion | INT8 screen head, inverted logic: high confidence = noise | When screen head present |
| L2 | Head agreement | Tracks unanimity/divergence across all heads | Always |

**Adaptive self-tuning:** Each layer has an independent `AdaptiveThresholdController` with EMA precision tracking:
- Warmup (30 steps) → Dial-in → Tighten (precision>85%) → Loosen (precision<75%) → Backoff (precision<40%) → Recovery (60-step cooldown)

**Environment overrides:**
```bash
SGLANG_PREFILTER=0                     # Disable prefilter entirely
SGLANG_PREFILTER=1                     # Enable (default when Medusa heads loaded)
SGLANG_PREFILTER_NGRAM_THRESHOLD=8.0   # L0 surprisal threshold
SGLANG_PREFILTER_SCREEN_THRESHOLD=0.5  # L1 confidence threshold
```

### Quick-Start Examples

```bash
# NGRAM — zero-compute baseline (27.8 t/s verified)
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_ROCM_ARCH=gfx1030 \
python -m sglang.launch_server \
  --model-path Bonsai-1.7B.gguf \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens 5 \
  --attention-backend torch_native --disable-cuda-graph \
  --dtype float16 --tp 1 --port 30000 --trust-remote-code

# MEDUSA — multi-head parallel draft
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_ROCM_ARCH=gfx1030 \
python -m sglang.launch_server \
  --model-path /path/to/Bonsai-1.7B-unpacked/ \
  --speculative-algorithm MEDUSA \
  --medusa-model-path /path/to/Bonsai-1.7B-Medusa/contrastive-3head/ \
  --medusa-num-heads 3 --speculative-num-draft-tokens 3 \
  --mem-fraction-static 0.50 --attention-backend torch_native \
  --disable-cuda-graph --dtype float16 --port 30000 --trust-remote-code

# MEDUSA with Typical Acceptance + PALTROW CPU offload (7-head tiered)
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_ROCM_ARCH=gfx1030 \
python -m sglang.launch_server \
  --model-path /path/to/Bonsai-1.7B-unpacked/ \
  --speculative-algorithm MEDUSA \
  --medusa-model-path /path/to/Bonsai-1.7B-Medusa/tiered-7head-fp16/ \
  --medusa-num-heads 7 --speculative-num-draft-tokens 6 \
  --medusa-typical-acceptance \
  --medusa-posterior-threshold 0.09 --medusa-posterior-alpha 0.3 \
  --medusa-paltrow-heads auto \
  --mem-fraction-static 0.40 --kv-cache-dtype tq4 \
  --attention-backend torch_native --disable-cuda-graph \
  --dtype float16 --port 30000 --trust-remote-code

# EAGLE3 with TurboQuant KV cache
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_ROCM_ARCH=gfx1030 \
python -m sglang.launch_server \
  --model-path Bonsai-4B.gguf \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path Bonsai-4B-EAGLE3/ \
  --speculative-eagle-topk 10 --speculative-num-steps 6 \
  --kv-cache-dtype tq4 --tp 1 --port 30000 --trust-remote-code

# P_CASCADE (adaptive routing, best throughput)
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_ROCM_ARCH=gfx1030 \
python -m sglang.launch_server \
  --model-path Bonsai-4B.gguf \
  --speculative-algorithm P_CASCADE \
  --speculative-draft-model-path Bonsai-4B-EAGLE3/ \
  --kv-cache-dtype tq4 --tp 1 --port 30000 --trust-remote-code

# TQ5_X — HSA zero-copy ghost-draft (AMD only)
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_ROCM_ARCH=gfx1030 \
python -m sglang.launch_server \
  --model-path Bonsai-1.7B.gguf \
  --speculative-algorithm TQ5_X \
  --disable-overlap-schedule \
  --attention-backend torch_native --disable-cuda-graph \
  --dtype float16 --port 30000 --trust-remote-code
```

### PALTROW Head Placement (CPU/GPU Mixed Inference)

PALTROW (**P**inned-memory **A**uxiliary **L**atency-**T**olerant **R**elocated **O**n-CPU **W**orkers) offloads designated Medusa heads to CPU, freeing GPU VRAM for KV cache and precision draft heads.

**How it works:**
- Screen/bloom/filter heads are latency-tolerant pre-rejection filters — they don't need GPU speed
- Hidden state copy to CPU is tiny (~4 KB per batch element)
- With AMD SAM/ReBAR, pinned DMA runs at ~14 GB/s — more than enough
- Heads that OOM on GPU are automatically promoted to PALTROW with a warning

**Auto-detection sources (in order):**
1. `tiered_architecture.screen_heads` in medusa_config.json
2. `tiered_architecture.bloom_heads` in medusa_config.json
3. Keyword matching in `head_offsets`: "screen", "bloom", "filter", "volatility", "negative"
4. OOM auto-fallback during GPU loading

**CLI:**
```bash
--medusa-paltrow-heads auto     # Auto-detect from config (default)
--medusa-paltrow-heads none     # Force all heads to GPU
--medusa-paltrow-heads 0,1,2    # Specify exact head indices for CPU
```

### Typical Acceptance (Entropy-Adaptive Candidate Generation)

From [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa). Instead of greedy argmax, Medusa heads generate candidates using entropy-aware filtering:

```
probs = softmax(logits)
entropy = -Σ(probs × log(probs + ε))
threshold = min(posterior_threshold, exp(-entropy) × posterior_alpha)
mask out tokens where prob < threshold
sample from remaining distribution
```

**Key insight:** Low-entropy (confident) distributions → strict threshold → fewer but higher-quality candidates. High-entropy (uncertain) distributions → looser threshold → more diversity.

**Default parameters:** `posterior_threshold=0.09`, `posterior_alpha=0.3` (α ≈ √τ is the empirical sweet spot).

### AMD SAM / ReBAR Hardware Integration

Auto-detects AMD Smart Access Memory status by reading PCIe BAR size from sysfs:
- **SAM active:** BAR ≥ VRAM (full GPU memory visible to CPU via PCIe)
- **GTT pool:** System RAM accessible by GPU (typically ~half system RAM)
- **Bandwidth:** ~14 GB/s pinned, ~8 GB/s pageable when SAM is active

### Environment Variables (AMD ROCm)

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE=1
```

## Acknowledgments
This fork builds on the work of:
- [SGLang / LMSYS](https://github.com/sgl-project/sglang) — the upstream inference engine
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — sampling kernels adapted for the HIP port
- [EAGLE](https://github.com/SafeAILab/EAGLE) — speculative decoding algorithm
- [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) — entropy-adaptive typical acceptance, tree buffer architecture
- [PrismML / Bonsai](https://huggingface.co/PrismML) — 1-bit GGUF model ecosystem
- [vLLM](https://github.com/vllm-project/vllm) — reference for Triton-based rejection sampling patterns

## License
Apache 2.0 — same as upstream SGLang.
