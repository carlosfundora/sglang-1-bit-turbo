## 2024-05-18 - Lift dynamic Python imports from forward_hip inner loops

**Learning:** SGLang used inline dynamic Python imports within `forward_hip` across multiple layer operators (`layernorm.py`, `activation.py`, `rotary_embedding/base.py`). On RDNA2/gfx1030 systems this can introduce measurable latency overhead in the hot inference loop.
**Action:** Modified `_check_rdna2_*` init functions to load and cache the kernels (`rdna2_rms_norm`, `rdna2_fused_add_rms_norm`, `rdna2_silu_and_mul`, `rdna2_gelu_and_mul`, `rdna2_ops`) at the module level. `forward_hip` now directly invokes these cached function references.
**Validation:** Since the local test suite has significant unresolved dependencies (`numpy`, `pytest`, `mooncake`, `gguf`, `vllm`, etc.), local validation is restricted to import testing using mocked/simplified import scripts, which pass cleanly. Will rely on pre-commit and CI for broader validation.
