# Comparative Repository Audit: vLLM vs Text Generation Inference vs LMDeploy

## 1. Executive Technical Summary
After performing a deep comparative audit of vLLM, Text Generation Inference (TGI), and LMDeploy, the evidence demonstrates varying approaches to the modern inference stack. vLLM operates as a monolithic event loop with deep framework entanglement, providing incredible algorithmic innovation (like PagedAttention) but suffering from operational brittleness due to scheduling logic being tightly coupled with KV cache physical management. TGI provides a robust service-oriented architecture, strictly bounding its Rust routing layer from its Python execution layer via gRPC, resulting in unmatched operational stability at the cost of integration rigidity. LMDeploy uses a structured C++ (Turbomind) and Python hybrid approach but heavily couples its engine with specific model implementations, optimizing for rapid prototyping.

For `sglang-1-bit-turbo`, TGI’s strict bounded architecture should be heavily referenced for system design (using high-speed IPC rather than gRPC to maintain performance), while vLLM and LMDeploy should be aggressively mined for algorithmic extractions (particularly PagedAttention and speculative decoding) without adopting their monolithic control planes.

## 2. Repository Targets & Assumptions
- **Repo A: vLLM** (https://github.com/vllm-project/vllm) - Assumed as the standard high-throughput, async-monolith Python engine.
- **Repo B: Text Generation Inference (TGI)** (https://github.com/huggingface/text-generation-inference) - Assumed as the production-grade, service-oriented architecture (Rust/Python).
- **Repo C: LMDeploy** (https://github.com/InternLM/lmdeploy) - Assumed as the highly optimized but coupled C++/Python engine focusing on specific topologies.

*Note: Targets inferred from `.jules/journals/triangulator-forge.md` and industry context for 1-bit turbo inference development.*

## 3. Per-Repo Deep Audit

### 3.1 Repo A: vLLM
- **Core Architecture & Logic Flow:** Async-await Python monolith. The `AsyncLLMEngine` handles the entire flow from request intake to token generation within a single asyncio event loop.
- **Functional Decomposition & "The Heart":** The hot path lives in `vllm/engine/async_llm_engine.py` and `vllm/core/scheduler.py`. The execution logic is heavily smeared across the scheduler and KV cache manager (`vllm/core/kv_cache_manager.py`), violating separation of concerns.
- **Dependency & Health Audit:** Heavy Python dependency tree (PyTorch, Ray, xFormers, Triton). High commit frequency but significant maintainer concentration risk. Fast-paced innovation but frequent breaking changes.
- **Complexity Score:** 8/10. High complexity due to asyncio entanglement and deeply nested abstractions in the scheduler.

### 3.2 Repo B: Text Generation Inference
- **Core Architecture & Logic Flow:** Service-oriented. A Rust router (`router/src/server.rs`) handles connections, batching, and token streaming. It communicates via gRPC to a Python backend (`server/text_generation_server/`) which handles PyTorch/CUDA execution.
- **Functional Decomposition & "The Heart":** The hot path crosses the Rust/Python boundary. The router batches requests (`router/src/infer/mod.rs`) and sends them to the execution server. This strict boundary provides incredible stability but limits dynamic payload passing.
- **Dependency & Health Audit:** Controlled dependency tree. Rust `Cargo.toml` is clean, Python dependencies are tightly version-locked. Backed by HuggingFace. Maintenance mode announced, making it a stable reference rather than a moving target.
- **Complexity Score:** 6/10. Moderate complexity. The architecture is clear, but crossing the language boundary adds cognitive load.

### 3.3 Repo C: LMDeploy
- **Core Architecture & Logic Flow:** Modular monolith with dual engines (PyTorch and Turbomind/C++). The PyTorch engine (`lmdeploy/pytorch/engine/`) uses an actor-model-like approach with a dedicated `model_agent`.
- **Functional Decomposition & "The Heart":** The hot path is deeply integrated into specific model runner classes and block managers (`lmdeploy/pytorch/paging/block_manager/`). It has strong abstraction for paging but ties execution tightly to specific backends.
- **Dependency & Health Audit:** Moderate dependency tree. Heavy reliance on custom kernels (`lmdeploy/pytorch/kernels/`). High activity, but geared heavily toward specific hardware (Nvidia/Ascend/Maca) with varying support quality.
- **Complexity Score:** 7/10. Complexity stems from supporting multiple backend paradigms (Turbomind vs PyTorch) within the same repository.

## 4. Feature Parity Table

| Feature | vLLM | TGI | LMDeploy |
| :--- | :--- | :--- | :--- |
| **Plugin/Module System** | Poor (Monolithic) | Poor (Opinionated) | Moderate (Backend specific) |
| **Schema Validation** | Pydantic (Strong) | Rust Structs (Strong) | Pydantic (Moderate) |
| **CLI Support** | Strong | Strong | Strong |
| **API Surface** | FastAPI (OpenAI compat) | WebSockets/HTTP (Custom + OpenAI) | FastAPI (OpenAI compat) |
| **Streaming** | Async Generators | gRPC streams | Async Generators |
| **Job Orchestration** | Ray/Asyncio | Rust Router | Asyncio/Multiprocessing |
| **Test Harness Depth** | Deep but brittle | Excellent integration tests | Moderate |

## 5. Comparative Trade-off Matrix

| Aspect | vLLM | TGI | LMDeploy |
| :--- | :--- | :--- | :--- |
| **Architectural Clarity** | Low (Entangled) | High (Bounded) | Moderate |
| **Extensibility** | Low (Invasive changes needed) | Low (gRPC schema rigid) | Moderate |
| **Maintainability** | Moderate | High | Moderate |
| **Performance Potential** | High (Algorithmic) | High (Pipeline) | Very High (C++ core) |
| **Dependency Risk** | High | Low | Moderate |
| **Migration Risk** | Severe (Deep integration) | Moderate (Adapter possible) | High |

## 6. Integration Opportunity Mapping

1. **Service-Oriented Router Boundary (from TGI)**
   - **Location:** `router/src/server.rs`
   - **Value:** Prevents event-loop stalls in the API layer by isolating batching from execution.
   - **Difficulty:** High
   - **Recommendation:** Adapt the concept to `sglang-1-bit-turbo` using ZeroMQ or shared memory instead of gRPC to avoid serialization bottlenecks while maintaining the architectural seam.

2. **Abstract Block Management (from LMDeploy)**
   - **Location:** `lmdeploy/pytorch/paging/block_manager/base_block_manager.py`
   - **Value:** Cleaner separation of physical memory allocation from logical token scheduling compared to vLLM.
   - **Difficulty:** Medium
   - **Recommendation:** Adopt directly as inspiration for the `Scheduler` / `KVCacheManager` split required for RotorQuant integration on AMD.

3. **PagedAttention Kernels (from vLLM)**
   - **Location:** `vllm/v1/attention/backends/`
   - **Value:** Industry standard memory-efficient attention.
   - **Difficulty:** Low
   - **Recommendation:** Extract as pure utility functions, completely decoupled from vLLM's `Scheduler`.

## 7. Adoption Plan

For `sglang-1-bit-turbo`, we will not adopt any of these repositories directly. Instead, we will construct a hybrid architecture:
1. **Routing Layer:** Implement a lightweight, dedicated batching router inspired by TGI, communicating with the execution core via zero-copy IPC (shm/ZMQ).
2. **Execution Core:** Use an SGLang-derived execution loop but strictly separate the `Scheduler` (logical token IDs) from the `KVCacheManager` (physical ROCm blocks), fixing the architectural deception found in vLLM/LMDeploy.
3. **Algorithms:** Extract isolated kernels (PagedAttention, Speculative Decoding) from vLLM and adapt them for AMD gfx1030 using wave32 tuning.

## 8. Concrete Work Items

1. **[TICKET-1] Decouple Scheduler from KV Cache Manager**
   - **Purpose:** Create an abstraction boundary so the scheduler only handles logical blocks, enabling custom RotorQuant hardware logic.
   - **Affected Area:** `sglang/srt/managers/`
   - **Dependency Order:** 1
   - **Risk Level:** High
   - **Acceptance Criteria:** Scheduler outputs abstract integer IDs; KVCacheManager handles all ROCm allocations.

2. **[TICKET-2] Implement Zero-Copy IPC Router Boundary**
   - **Purpose:** Prevent API event loop stalls during heavy inference by moving batching to a separate process communicating via ZMQ.
   - **Affected Area:** `sglang/srt/server/`
   - **Dependency Order:** 2
   - **Risk Level:** Medium
   - **Acceptance Criteria:** API layer runs independently of the execution engine, passing batches via shared memory.

3. **[TICKET-3] Extract and Tune PagedAttention for AMD RDNA2**
   - **Purpose:** Port vLLM's PagedAttention kernel, strictly applying `waves_per_eu=2` and stripping CDNA-specific arguments.
   - **Affected Area:** `sgl-kernel/`
   - **Dependency Order:** 1
   - **Risk Level:** Low
   - **Acceptance Criteria:** Kernel executes cleanly on gfx1030 without `matrix_instr_nonkdim` arguments.

**First PR:** Implement the `Scheduler` / `KVCacheManager` interface abstraction (TICKET-1).
**Second PR:** Extract and optimize the PagedAttention kernel (TICKET-3).
**What not to do:** Do not import asyncio execution logic into the hot path; do not couple physical memory structs to the scheduling algorithm.

## 9. Final Recommendation

**Best Internal Fork Candidate:** SGLang (our current base).
**Best to Mine for Ideas:** vLLM (Algorithms), TGI (Architecture).
**Best Avoided:** Direct adoption of any, due to deep framework entanglement (vLLM) or rigid IPC constraints (TGI). We must build the TGI architectural shape using vLLM's algorithmic ideas, optimized for ROCm.

## 10. Horizon Scanning

1. **The Rising Star: SGLang (Main)** - Chosen as our fork base due to its unique RadixAttention capability, but requires significant ROCm optimization.
2. **The Legacy Standard: llama.cpp** - Unmatched hardware portability (CPU/Edge), but lacks the extreme throughput optimizations needed for data-center scale 1-bit serving.
3. **The Niche Specialist: MLC-LLM** - Excellent for TVM-based compilation across diverse edge devices, but too generalized for our specific AMD ROCm / RotorQuant hot-path requirements.

## 11. Appendix: Evidence Notes

- `vllm/engine/async_llm_engine.py` heavily relies on Python `asyncio`, leading to event loop bottlenecks under high load.
- `text-generation-inference/router/src/server.rs` clearly demonstrates the strict bounded context approach.
- LMDeploy's architecture separates paging logic but remains heavily bound to specific model runner implementations.

## 12. Scoring Matrix

### vLLM
- **Architectural Clarity (4/10):** Entangled. The scheduling layer deeply understands physical memory blocks.
- **Maintainability (5/10):** Fast-moving but brittle due to heavy asyncio and monolithic design.
- **Extensibility (4/10):** Difficult to add novel execution logic (like custom quantization) without rewriting core loops.
- **Performance Potential (9/10):** Industry-leading algorithmic implementations (PagedAttention).
- **Dependency Risk (5/10):** Heavy Python ML stack with frequent breaking changes.
- **Migration Flexibility (3/10):** Severe lock-in; wrapping it is hard due to its stateful engine.
- **DX / Onboarding (7/10):** Great for rapid prototyping, complex to modify internally.
- **Test Trustworthiness (6/10):** Extensive tests, but frequently flaky due to async concurrency.
- **Operational Maturity (6/10):** Widely used, but known to suffer from event loop stalls under high concurrent load.
- **Integration Readiness (7/10):** Easy to drop in via FastAPI, but hard to embed as a library in complex architectures.
- **Licensing Suitability (9/10):** Apache 2.0. Clean for adoption.

### Text Generation Inference (TGI)
- **Architectural Clarity (9/10):** Excellent bounded context between routing and execution.
- **Maintainability (8/10):** Highly stable Rust core, explicit boundaries limit cascading failures.
- **Extensibility (3/10):** gRPC schema is highly rigid, making dynamic payload passing (e.g. spec decode trees) difficult.
- **Performance Potential (8/10):** Highly optimized pipeline, though gRPC serialization introduces minor latency.
- **Dependency Risk (8/10):** Controlled and version-locked.
- **Migration Flexibility (7/10):** Service-oriented nature means clients are decoupled from the engine.
- **DX / Onboarding (5/10):** Crossing the Rust/Python boundary is challenging for Python-centric ML engineers.
- **Test Trustworthiness (9/10):** Robust Rust tests and integration suites.
- **Operational Maturity (9/10):** Built for HuggingFace production; extremely resilient.
- **Integration Readiness (8/10):** Excellent as a standalone service, poor for library embedding.
- **Licensing Suitability (2/10):** HFOIL (Hugging Face Optimized Inference License). Extremely restrictive for commercial use, preventing direct adoption.

### LMDeploy
- **Architectural Clarity (6/10):** Modular, but dual engines (C++/Python) complicate the mental model.
- **Maintainability (6/10):** Strong paging abstraction, but hardware-specific kernels are heavily fragmented.
- **Extensibility (5/10):** Better than vLLM due to base classes, but still tied to specific runner implementations.
- **Performance Potential (9/10):** Turbomind C++ engine is exceptionally fast for supported models.
- **Dependency Risk (6/10):** Heavy reliance on fragmented custom kernels for various hardware backends.
- **Migration Flexibility (5/10):** Moderate lock-in.
- **DX / Onboarding (6/10):** Good docs, but the C++ core limits Python developer velocity.
- **Test Trustworthiness (6/10):** Moderate, heavily tied to specific hardware availability.
- **Operational Maturity (7/10):** Solid, heavily optimized for specific data center topologies.
- **Integration Readiness (6/10):** Good via APIs, complex internally.
- **Licensing Suitability (9/10):** Apache 2.0.
