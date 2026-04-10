<div align="center">
<img src="assets/sgl-1-bit-turbo-eagle.png" alt="SGLang 1-Bit Turbo" width="600"></img>
</div>

[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/carlosfundora/sglang-1-bit-turbo/blob/main/LICENSE)
---

## SGLang 1-Bit Turbo EAGLE

A fork of [SGLang](https://github.com/sgl-project/sglang) optimized for TurboQuant and EAGLE3 speculative decoding on AMD ROCm gfx1030x GPUs, with support for PrismML's 1-bit quantized models. It extends upstream SGLang with a stack of features designed to run aggressively quantized models at practical speeds on AMD RDNA2/RDNA3 hardware (RX 6000/7000 series).

---

## Key Features

### 🧊 KV Cache Compression (TurboQuant & RotorQuant)
Extreme KV cache quantization modes that dramatically reduce VRAM usage, enabling larger context and bigger models on memory-constrained GPUs.

| Method | Mode | Bits | VRAM Savings / Details | Speed | Quality |
|--------|------|------|------------------------|-------|---------|
| **TurboQuant** | `--kv-cache-dtype tq4` | 4-bit | ~75% VRAM savings vs FP16 | Fast | Good |
| **TurboQuant** | `--kv-cache-dtype tq3` | 3-bit | ~81% VRAM savings vs FP16 | Fast | Fair |
| **TurboQuant** | `--kv-cache-dtype tq2` | 2-bit | ~87% VRAM savings vs FP16 | Fast | Basic |
| **PlanarQuant** | `--kv-cache-dtype rq4_planar` | 4-bit | 2D Givens (64× fewer FMAs than TQ) | ⚡ Fastest | Better |
| **PlanarQuant** | `--kv-cache-dtype rq3_planar` | 3-bit | 2D Givens (HIP/ROCm compatible) | ⚡ Fastest | Good |
| **IsoQuant** | `--kv-cache-dtype rq4_iso` | 4-bit | 4D Quaternion (PyTorch fallback) | Fast | Best overall |
| **IsoQuant** | `--kv-cache-dtype rq3_iso` | 3-bit | 4D Quaternion | Fast | Best at 3-bit |

*Note: RotorQuant methods are data-oblivious (no calibration needed).*

### 🦅 Speculative Decoding Algorithms
A full suite of 9 speculative decoding algorithms ported to AMD GPUs.

- **EAGLE3** (`EAGLE3` / `P_EAGLE`): 3-layer feature extraction + 1-layer decoder.
- **MEDUSA** (`MEDUSA`): 2–7 parallel MLP draft heads with tree verification. Features **Typical Acceptance** (entropy-adaptive candidate generation) and **DraftPreFilter** (a novel 3-layer pre-rejection filter with adaptive self-tuning thresholds).
- **NGRAM** (`NGRAM`): Stable at **27.8 t/s** (1.6× baseline) using zero extra compute. Statistical trie-based.
- **P_CASCADE** (`P_CASCADE`): Adaptive DyTC router (L1=EAGLE, L2=reduced, L3=ngram).
- **CHIMERA** (`CHIMERA`): Fused P-EAGLE + Hydra + DyTC + SSD (experimental).
- **TQ5_X** (`TQ5_X`): TurboQuant 5 eXtended: HSA zero-copy ghost-draft (AMD gfx103x).
- **SAGUARO** (`--ssd-enable`): LRU draft caching, async pre-generation wrapper.
- **STANDALONE** (`STANDALONE`): Independent draft model (no shared weights).

#### Three-Tier Speculative Sampling Fallback Chain
On CUDA, sglang uses a flashinfer-backed C++ kernel. On ROCm, we implement a three-tier fallback so EAGLE speculative decoding works everywhere:
1. **HIP C++ Kernel** (`speculative_sampling.hip`): Fastest, gfx1030. Self-contained port, no flashinfer dependency.
2. **Triton Kernel** (`speculative_sampling_triton.py`): Fast, any GPU. `@triton.jit`, device-agnostic.
3. **PyTorch Fallback** (`speculative_sampling_pytorch`): Universal. Pure tensor ops, works on any backend.

Detection is automatic — the best available backend is selected at import time.

### 🧠 PALTROW Mixed CPU/GPU Head Placement
**P**inned-memory **A**uxiliary **L**atency-**T**olerant **R**elocated **O**n-CPU **W**orkers offloads designated Medusa heads (like screen/bloom/filter) to the CPU.
- Frees GPU VRAM for KV cache and precision draft heads.
- **Auto-detection**: Heads are identified from the tiered config, keyword matching, or OOM auto-fallback.
- **AMD SAM/ReBAR integration**: Auto-detects Smart Access Memory for optimal DMA bandwidth (~14 GB/s pinned) and reads the GTT pool size.

### 📦 PrismML Q1_0_G128 1-Bit GGUF Model Support
Native serving of [Bonsai](https://huggingface.co/PrismML) Q1_0_G128 1-bit GGUF models through sglang's weight loading pipeline with GPU dequantization kernels (dp4a).

### 🔧 Pre-Built ROCm `sgl_kernel`
Ships a pre-compiled `sgl_kernel` binary for ROCm gfx1030 (RDNA2), so you can skip the build step entirely. Includes all speculative decoding kernels and standard sglang ops.

---

## Installation & Environment

### Prerequisites
- AMD GPU with ROCm support (tested on gfx1030 / RX 6900 XT / RX 6700 XT)
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

### Environment Variables (AMD ROCm)
For the most optimal setup on RDNA2 (gfx1030) and when using GGUF + EAGLE3:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE=1
```

---

## Usage & CLI Reference

### Common Flags
```text
--speculative-algorithm ALGO           (EAGLE3, P_EAGLE, NGRAM, MEDUSA, P_CASCADE, CHIMERA, TQ5_X, STANDALONE)
--speculative-draft-model-path PATH    Path to EAGLE3/P_EAGLE draft model weights
--speculative-eagle-topk K             Top-k candidates per draft step (default: auto)
--speculative-num-steps N              Max draft steps per round (default: auto)
--speculative-num-draft-tokens N       Max total draft tokens (default: auto)
--disable-overlap-schedule             Required for P_CASCADE, MEDUSA, CHIMERA
--ssd-enable                           Enable SAGUARO LRU draft caching
```

<details>
<summary><b>Algorithm-Specific & Hardware Flags</b></summary>

```text
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

# CHIMERA (experimental)
--chimera-num-steps 6
--chimera-ssd-enable
--chimera-level 1|2|3                  Force cascade level (omit for dynamic)
```
</details>

<details>
<summary><b>DraftPreFilter Environment Overrides</b></summary>

```bash
SGLANG_PREFILTER=0                     # Disable prefilter entirely
SGLANG_PREFILTER=1                     # Enable (default when Medusa heads loaded)
SGLANG_PREFILTER_NGRAM_THRESHOLD=8.0   # L0 surprisal threshold
SGLANG_PREFILTER_SCREEN_THRESHOLD=0.5  # L1 confidence threshold
```
</details>

### Quick-Start Examples

<details open>
<summary><b>Serve a PrismML 1-Bit Model with EAGLE3</b></summary>

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
</details>

<details>
<summary><b>NGRAM Baseline (Zero Extra Compute)</b></summary>

```bash
python -m sglang.launch_server \
  --model-path Bonsai-1.7B.gguf \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens 5 \
  --attention-backend torch_native --disable-cuda-graph \
  --dtype float16 --tp 1 --port 30000 --trust-remote-code
```
</details>

<details>
<summary><b>MEDUSA Multi-Head Parallel Draft</b></summary>

```bash
python -m sglang.launch_server \
  --model-path /path/to/Bonsai-1.7B-unpacked/ \
  --speculative-algorithm MEDUSA \
  --medusa-model-path /path/to/Bonsai-1.7B-Medusa/contrastive-3head/ \
  --medusa-num-heads 3 --speculative-num-draft-tokens 3 \
  --mem-fraction-static 0.50 --attention-backend torch_native \
  --disable-cuda-graph --dtype float16 --port 30000 --trust-remote-code
```
</details>

<details>
<summary><b>MEDUSA with Typical Acceptance & PALTROW (7-head tiered)</b></summary>

```bash
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
```
</details>

<details>
<summary><b>P_CASCADE (Adaptive Routing, Best Throughput)</b></summary>

```bash
python -m sglang.launch_server \
  --model-path Bonsai-4B.gguf \
  --speculative-algorithm P_CASCADE \
  --speculative-draft-model-path Bonsai-4B-EAGLE3/ \
  --kv-cache-dtype tq4 --tp 1 --port 30000 --trust-remote-code
```
</details>

<details>
<summary><b>TQ5_X HSA Zero-Copy Ghost-Draft (AMD Only)</b></summary>

```bash
python -m sglang.launch_server \
  --model-path Bonsai-1.7B.gguf \
  --speculative-algorithm TQ5_X \
  --disable-overlap-schedule \
  --attention-backend torch_native --disable-cuda-graph \
  --dtype float16 --port 30000 --trust-remote-code
```
</details>

---

## Tested Configurations

| GPU | Model | Algorithm | Draft Model | KV Cache | Throughput | Status |
|-----|-------|-----------|-------------|----------|------------|--------|
| RX 6700 XT (12GB) | Bonsai-1.7B (Q1_0_G128 GGUF) | NGRAM | — | fp16 | **27.8 t/s** | ✅ Stable (80/80 stress) |
| RX 6700 XT (12GB) | Bonsai-1.7B (unpacked fp16) | MEDUSA (3-head) | contrastive-3head | fp16 | **9.5 t/s** | ✅ Coherent output |
| RX 6700 XT (12GB) | Bonsai-1.7B (Q1_0_G128 GGUF) | Baseline (no spec) | — | fp16 | **17.2 t/s** | ✅ Baseline |
| RX 6700 XT (12GB) | Bonsai-1.7B (unpacked fp16) | EAGLE3 | Bonsai-EAGLE3 | TQ4 | — | 🔴 0% acceptance (untrained model) |

---

## Building `sgl_kernel` from Source (ROCm)

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

## Upstream SGLang

This fork is based on [SGLang](https://github.com/sgl-project/sglang), a high-performance serving framework for large language models by [LMSYS](https://lmsys.org/about/). For general SGLang documentation, features, and community:

- [SGLang Documentation](https://docs.sglang.io/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SGLang Slack](https://slack.sglang.io/)

---

## Acknowledgments
This fork builds on the work of:
- [SGLang / LMSYS](https://github.com/sgl-project/sglang) — the upstream inference engine
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — sampling kernels adapted for the HIP port
- [EAGLE](https://github.com/SafeAILab/EAGLE) — speculative decoding algorithm
- [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) — entropy-adaptive typical acceptance, tree buffer architecture
- [PrismML / Bonsai](https://huggingface.co/PrismML) — 1-bit GGUF model ecosystem
- [vLLM](https://github.com/vllm-project/vllm) — reference for Triton-based rejection sampling patterns
- [RotorQuant](https://github.com/scrya-com/rotorquant) — advanced geometric-rotation-based KV cache quantization

## License
Apache 2.0 — same as upstream SGLang.
