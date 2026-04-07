<div align="center">
<img src="assets/sgl-1-bit-turbo-eagle.png" alt="SGLang 1-Bit Turbo" width="600"></img>




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

### 🦅 EAGLE3 Speculative Decoding on ROCm
Full EAGLE3 speculative decoding support ported to AMD GPUs, including:
- **HIP C++ probabilistic sampling kernel** — self-contained port of the CUDA `tree_speculative_sampling_target_only` kernel, compiled without flashinfer dependencies
- **Triton kernel fallback** — device-agnostic `@triton.jit` implementation for systems where the HIP kernel can't compile
- **Pure PyTorch fallback** — universal last-resort implementation using only tensor ops
- **Three-tier automatic fallback**: HIP C++ → Triton → PyTorch, detected at import time

### 📦 PrismML 1-Bit GGUF Model Support
Native serving of [Bonsai](https://huggingface.co/PrismML) 1-bit GGUF models through sglang's weight loading pipeline, bridging the GGUF quantization ecosystem with sglang's high-performance runtime.

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
| **EAGLE3 speculative decoding on ROCm** | Full probabilistic tree sampling via three-tier fallback (HIP C++ → Triton → PyTorch) |
| **Self-contained HIP C++ sampling kernel** | Port of `tree_speculative_sampling_target_only` with no flashinfer dependency |
| **`top_k` / `top_p` renorm fallbacks** | PyTorch implementations with cached capability probe and kth-pivot tie-correct top-k |
| **TurboQuant KV cache** | TQ4 (4-bit), TQ3 (3-bit), TQ2 (2-bit) — up to 87% VRAM savings vs FP16 |
| **PrismML 1-bit GGUF model serving** | Tested with [Bonsai](https://huggingface.co/PrismML) 1-bit GGUF models (IQ1_S, IQ1_M) |
| **Pre-built ROCm `sgl_kernel`** | Ships a compiled `.so` for gfx1030 (RDNA2) with all speculative decoding ops included |
| **Consumer AMD GPU focus** | Optimized for RX 6000/7000 series (12–16 GB VRAM) |

---

## Tested Configurations

| GPU | Model | Draft Model | KV Cache | Status |
|-----|-------|-------------|----------|--------|
| RX 6900 XT (12GB) | Bonsai-1.7B (1-bit GGUF) | Bonsai-1.7B-EAGLE3 (FP16) | TQ4 | ✅ Working |

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
| `MEDUSA` | `--speculative-algorithm MEDUSA` | 2-6 parallel MLP draft heads | ❌ (needs `--medusa-model-path`) |
| `CHIMERA` | `--speculative-algorithm CHIMERA` | Fused P-EAGLE + Hydra + DyTC + SSD (experimental) | ✅ |

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
--medusa-num-heads 5                   Number of parallel draft heads
--medusa-topk 1                        Top-k per head

# SAGUARO (wraps any algorithm)
--ssd-enable                           Enable LRU draft caching

# CHIMERA (experimental)
--chimera-num-steps 6
--chimera-ssd-enable
--chimera-level 1|2|3                  Force cascade level (omit for dynamic)
```

### Quick-Start Examples

```bash
# EAGLE3 with TurboQuant KV cache
python -m sglang.launch_server \
  --model-path Bonsai-4B.gguf \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path Bonsai-4B-EAGLE3/ \
  --speculative-eagle-topk 10 --speculative-num-steps 6 \
  --kv-cache-dtype tq4 --tp 1 --port 30000 --trust-remote-code

# P_CASCADE (adaptive routing, best throughput)
python -m sglang.launch_server \
  --model-path Bonsai-4B.gguf \
  --speculative-algorithm P_CASCADE \
  --speculative-draft-model-path Bonsai-4B-EAGLE3/ \
  --kv-cache-dtype tq4 --tp 1 --port 30000 --trust-remote-code

# NGRAM (zero-compute baseline)
python -m sglang.launch_server \
  --model-path Bonsai-4B.gguf \
  --speculative-algorithm NGRAM \
  --tp 1 --port 30000 --trust-remote-code
```

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
- [PrismML / Bonsai](https://huggingface.co/PrismML) — 1-bit GGUF model ecosystem
- [vLLM](https://github.com/vllm-project/vllm) — reference for Triton-based rejection sampling patterns

## License
Apache 2.0 — same as upstream SGLang.
