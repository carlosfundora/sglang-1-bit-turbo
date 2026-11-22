# TurboQuant + EAGLE3/P-EAGLE Speculative Decoding Guide

**Target hardware**: AMD RDNA3 (gfx1030/gfx1031, e.g. RX 6700 XT 12 GB)
**Target models**: PrismML Bonsai family (Qwen3-based, PrismML custom-quantised (GGML type 41) via GGUF)

## Quick Start

### EAGLE3 (sequential draft, proven working)

```bash
cd /path/to/sglang-1-bit-turbo

export PYTHONPATH=python:sgl-kernel/python
export HSA_OVERRIDE_GFX_VERSION=10.3.0

python3 -m sglang.launch_server \
  --model-path /path/to/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /path/to/Bonsai-1.7B-EAGLE3 \
  --speculative-num-steps 5 \
  --quantization turboquant \
  --load-format gguf \
  --attention-backend triton \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-overlap-schedule \
  --tp 1 --port 30000 \
  --trust-remote-code
```

### P-EAGLE (parallel draft, single forward for all K tokens)

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/Bonsai-1.7B-gguf/Bonsai-1.7B.gguf \
  --speculative-algorithm P_EAGLE \
  --speculative-draft-model-path /path/to/Bonsai-1.7B-P-EAGLE-local-smoke/epoch_0_step_500 \
  --speculative-num-steps 7 \
  --quantization turboquant \
  --load-format gguf \
  --attention-backend triton \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-overlap-schedule \
  --tp 1 --port 30000 \
  --trust-remote-code
```

## Required Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONPATH` | `python:sgl-kernel/python` | Ensure fork modules are on path |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | ROCm compatibility for gfx1030/gfx1031 |
| `SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE` | `1` | *Optional* — force draft to use its own embeddings (auto-detected for GGUF targets) |
| `SGLANG_USE_PYTORCH_TREE_OPS` | `1` | *Optional* — force PyTorch fallback for spec-decode tree ops on ROCm |
| `AMD_SERIALIZE_KERNEL` | `3` | *Optional* — serialise kernel dispatch (helps gfx1031 stability) |

## Required Server Flags

| Flag | Why |
|------|-----|
| `--load-format gguf` | Load PrismML custom-quantised (GGML type 41) target model |
| `--quantization turboquant` | Enable TurboQuant KV cache compression (tq4) |
| `--attention-backend triton` | Use Triton attention (FlashAttention ROCm kernels crash on gfx1030) |
| `--disable-cuda-graph` | CUDA graphs are not compatible with ROCm gfx1030 |
| `--disable-piecewise-cuda-graph` | Same |
| `--disable-overlap-schedule` | Overlap scheduling races with HIP non-blocking transfers |

## Model Compatibility Matrix

| Target Model | Draft Model | Algorithm | Aux Layers | Status |
|-------------|-------------|-----------|------------|--------|
| Bonsai-1.7B (PrismML type-41) | Bonsai-1.7B-EAGLE3 | EAGLE3 | [1, 14, 27] | ✅ Working |
| Bonsai-1.7B (PrismML type-41) | Bonsai-1.7B-P-EAGLE-local-smoke/step_500 | P_EAGLE | [1, 14, 27] | ✅ Ready |
| Bonsai-4B (PrismML type-41) | Bonsai-4B-EAGLE3 | EAGLE3 | [1, 18, 35] | ⚠️ Needs retrained head |
| Bonsai-8B (PrismML type-41) | — | — | — | 🔜 Planned |

## Key Bug Fixes in This Fork

### 1. Qwen3 Aux Hidden State Capture (qwen3.py)

**Problem**: The upstream `Qwen2Model.forward` method (which Qwen3 inherited) had no
EAGLE3 aux hidden state capture. The Llama-style `layers_to_capture = [val+1 for val in layer_ids]`
applied a +1 offset that is wrong for Qwen3's post-layer capture timing.

**Fix**: Added standalone `Qwen3Model.forward` with post-layer capture at the exact
`eagle_aux_hidden_state_layer_ids` indices (no +1 offset). The captured activation is
`hidden_states + residual` to reconstruct the true post-layer output from SGLang's
fused-residual scheme.

### 2. fc Projection Condition (llama_eagle3.py)

**Problem**: The fc layer was applied based on `self.fc.in_features != self.fc.out_features`
(always true for EAGLE3). During draft decode steps, the hidden state is already in
draft space (2560-dim) and must NOT be re-projected. Zero-padding 2560→7680 then
projecting through fc produced garbage → 0% acceptance.

**Fix**: Changed condition to `hidden_states.shape[-1] != self.fc.out_features` so fc
is applied only when hidden states are in target aux space (7680-dim) and skipped when
they're already in draft space (2560-dim).

### 3. PrismML type-41 Hidden State Scale Mismatch (llama_eagle3.py)

**Problem**: EAGLE3 heads are trained on full-precision (fp16/bf16) auxiliary hidden
states with per-layer norm ~150–250. PrismML type-41 dequantised hidden states have norms
~3000–4000 per layer (23–30× larger). The fc layer receives out-of-distribution
inputs → output explodes → midlayer produces near-uniform logits → 0% acceptance.

**Fix**: Applied 1/25 rescale + clamp(±100) before fc when projecting target aux hidden
states. This is a temporary band-aid; the proper fix is retraining the EAGLE3 head on
quantised hidden states.

### 4. GGUF Embed Sharing Skip (eagle_worker.py)

**Problem**: PrismML type-41 dequantised `embed_tokens` have ~70× smaller norm than the fp16
embeddings the EAGLE3 head was trained with. Sharing GGUF target embeddings into the
draft model caused near-zero input to the midlayer.

**Fix**: Auto-detect `quantization == "gguf"` and skip embed sharing, keeping the draft
model's own full-precision safetensors `embed_tokens`.

### 5. ROCm Triton tl.sum Workaround (eagle_info.py)

**Problem**: The `create_extend_after_decode_spec_info` Triton kernel uses `tl.sum()`
reductions that produce incorrect offsets on gfx1030. This corrupts the `positions`
tensor with uninitialized GPU memory values (e.g. -9223372034707292160), which then
crash the rotary embedding lookup.

**Fix**: Added pure-PyTorch fallback `_create_extend_after_decode_pytorch` that runs on
HIP targets. The Triton kernel is preserved for CUDA.

## P-EAGLE Architecture Notes

P-EAGLE extends EAGLE3 with parallel multi-token prediction (MTP). Instead of K
sequential drafter forwards, it produces all K speculative tokens in a single forward
pass using a learnable `mask_hidden` parameter and a `mask_token_id`.

**Key model parameters** (in `llama_eagle3.py` `LlamaModel`):
- `mask_hidden: nn.Parameter([1, 1, fc_in_features])` — learnable hidden state for mask positions
- `mask_token_id: int` — vocabulary ID used for mask token embedding (typically `pad_token_id`)
- `parallel_drafting: bool` — read from `config.json`

**Input construction** (`prepare_p_eagle_inputs`):
- Position 0: real token embedding + real fused aux hidden states → fc → midlayer
- Positions 1..K-1: `embed_tokens(mask_token_id)` + `mask_hidden` → fc → midlayer
- All K positions run through a single forward pass

**Config requirements** (in draft model `config.json`):
```json
{
  "parallel_drafting": true,
  "mask_token_id": 151643,
  "speculative_algorithm": "P_EAGLE"
}
```

## Troubleshooting

### 0% acceptance rate
1. Check aux hidden state norms: if >1000 per layer, the scale factor may need tuning
2. Verify draft model path matches the target (same family, same vocab size)
3. Check that `eagle_aux_hidden_state_layer_ids` in draft config matches the target's layer count
4. If using GGUF target: ensure embed sharing is skipped (check server log for "skipping embed share")

### Server crash on startup
- `OpenAIServingResponses` import error: may need `pip install openai>=1.0` or check transformers version
- OOM: reduce `--speculative-num-steps` or use a smaller draft model
- Position corruption (huge negative ints in logs): ROCm Triton bug — ensure `is_hip()` fallback is active

### Low throughput
- Ensure `--attention-backend triton` is set (FlashAttention ROCm is slow on gfx1030)
- Disable cuda graphs (`--disable-cuda-graph`)
- Check GPU utilisation with `rocm-smi`
