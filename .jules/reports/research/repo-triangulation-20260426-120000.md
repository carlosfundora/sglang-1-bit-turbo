# Comparative Repository Audit: vLLM vs Text-Generation-Inference (TGI) vs LMDeploy

## 1. Executive Technical Summary
This report analyzes three leading open-source LLM inference engines to determine the optimal architecture for integration into the SGLang 1-Bit Turbo project. The goal is to identify which engine provides the best balance of maintainability, performance, and extensibility, specifically focusing on how their design patterns can inform our handling of extreme KV cache compression (RotorQuant/TurboQuant) and speculative decoding (EAGLE3).

## 2. Repository Targets & Assumptions
- **Target A: vLLM** (https://github.com/vllm-project/vllm) - The most popular framework, known for PagedAttention.
- **Target B: Text-Generation-Inference (TGI)** (https://github.com/huggingface/text-generation-inference) - HuggingFace's production-grade router/server architecture.
- **Target C: LMDeploy** (https://github.com/InternLM/lmdeploy) - A high-performance inference engine focused on extreme quantization and C++ core efficiency.
*Assumption:* The evaluation prioritizes architectures that can robustly support asynchronous request handling while cleanly isolating hardware-specific optimization layers (e.g., ROCm Triton kernels).

## 3. Per-Repo Deep Audit

### 3.1 Repo A: vLLM
#### A. Core Architecture & Logic Flow
- **Architecture:** Monolithic async event loop.
- **Entrypoints:** `vllm/engine/async_llm_engine.py`.
- **Logic Flow:** API requests enter an asyncio queue, are processed by a Python scheduler, and dispatched to worker processes executing PyTorch code.
- **Concurrency:** asyncio + multiprocessing workers.
- **Classification:** Framework-bound application.

#### B. Functional Decomposition & “The Heart”
- **Hot Path:** `vllm/worker/model_runner.py` -> `vllm/attention/backends/`.
- **Complexity Score:** 7/10. High coupling between scheduling and attention backends makes adding new hardware targets difficult.
- **Value Prop:** PagedAttention abstraction.

#### C. Dependency & Health Audit
- **Dependencies:** Heavy PyTorch reliance, numerous hardware-specific branches (CUDA, ROCm, Neuron).
- **Health:** Extremely active, but PR review cycles are long due to architectural complexity and regression risks.

#### D. Developer Experience & Integration
- **DX:** High onboarding friction due to nested abstractions.
- **Integration:** Invasive; extending requires modifying core schedulers.

#### E. Lock-in & Migration Risk
- **Risk:** High. The custom PagedAttention block manager is tightly woven into the engine.
- **What is hardest to replace:** Custom block manager abstractions.
- **What could be isolated:** Top-level request routing interface.
- **What to wrap immediately:** Attention backend dispatch layer.

### 3.2 Repo B: Text-Generation-Inference (TGI)
#### A. Core Architecture & Logic Flow
- **Architecture:** Service-oriented (Rust Router + Python Server).
- **Entrypoints:** `router/src/main.rs`, `server/text_generation_server/server.py`.
- **Logic Flow:** Rust handles HTTP/gRPC, tokenization, and request batching. Python server only executes forward passes.
- **Concurrency:** Rust Tokio (async) -> gRPC -> Python blocking workers.
- **Classification:** Modular monolith.

#### B. Functional Decomposition & “The Heart”
- **Hot Path:** Rust `Batcher` -> Python `Model.forward`.
- **Complexity Score:** 4/10. Very clean separation of concerns.
- **Value Prop:** Operational stability via strict Rust/Python boundary.

#### C. Dependency & Health Audit
- **Dependencies:** Clean separation. Cargo for routing, Poetry for server.
- **Health:** Stable, enterprise-backed. Slower to adopt bleeding-edge research.

#### D. Developer Experience & Integration
- **DX:** Excellent if you respect the boundary; difficult if you need cross-boundary features.
- **Integration:** Clean wrapper-friendly design.

#### E. Lock-in & Migration Risk
- **Risk:** Moderate. The gRPC protocol is a soft lock-in, but the Python server is relatively stateless.
- **What is hardest to replace:** Rust Router batching mechanism.
- **What could be isolated:** The Python worker entrypoint.
- **What to wrap immediately:** gRPC communication layer.

### 3.3 Repo C: LMDeploy
#### A. Core Architecture & Logic Flow
- **Architecture:** C++ Core with Python bindings.
- **Entrypoints:** `lmdeploy/turbomind/`.
- **Logic Flow:** Python API calls into a highly optimized C++ backend (TurboMind).
- **Concurrency:** C++ threading model.
- **Classification:** Library core with CLI/API shell.

#### B. Functional Decomposition & “The Heart”
- **Hot Path:** TurboMind C++ kernels.
- **Complexity Score:** 8/10. High barrier to entry for Python developers due to C++ complexity.
- **Value Prop:** Extreme low-level performance and KV cache quantization support.

#### C. Dependency & Health Audit
- **Dependencies:** Lean Python layer, heavy C++ build system (CMake).
- **Health:** Active, focused on specific hardware (NVIDIA/Ascend).

#### D. Developer Experience & Integration
- **DX:** Difficult for non-C++ developers.
- **Integration:** Brittle if modifying core kernels; clean if just using Python API.

#### E. Lock-in & Migration Risk
- **Risk:** Severe. Modifying behavior requires deep C++ expertise.
- **What is hardest to replace:** TurboMind inference engine.
- **What could be isolated:** Python API client.
- **What to wrap immediately:** C++ kernel launch mechanisms.

## 4. Feature Parity Table
| Feature | vLLM | TGI | LMDeploy |
| :--- | :--- | :--- | :--- |
| **PagedAttention** | Yes | Yes (FlashInfer) | Yes (TurboMind) |
| **Speculative Decoding** | Yes | Basic | Yes |
| **KV Quantization** | FP8 | FP8/Int8 | W4A16/KV4 |
| **Rust Router** | No | Yes | No |
| **Hardware Agnosticism** | High | Medium | Low |
| **Schema Validation** | High (Pydantic) | Medium | Low |
| **Config Layering** | High | Medium | Low |
| **CLI Support** | Yes | Yes | Yes |

## 5. Comparative Trade-off Matrix
| Metric | vLLM | TGI | LMDeploy | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Clarity** | 5 | 9 | 6 | TGI's router/server boundary is much cleaner than vLLM's entangled asyncio loop. |
| **Extensibility** | 6 | 8 | 4 | TGI's gRPC boundary makes replacing the model engine easy. LMDeploy's C++ core is hard to extend. |
| **Maintainability** | 5 | 8 | 5 | vLLM's fast movement causes regressions. TGI is exceptionally stable. |
| **Performance Potential**| 8 | 7 | 9 | LMDeploy's raw C++ core yields the highest peak performance. |
| **Dependency Risk** | 6 | 4 | 8 | vLLM has heavy PyTorch/CUDA dependencies. TGI separates concerns. LMDeploy has huge CMake baggage. |
| **Migration Flexibility**| 4 | 7 | 2 | vLLM locks you into their block manager. TGI uses standard gRPC. LMDeploy is a black-box C++ engine. |
| **DX / Onboarding** | 5 | 7 | 3 | TGI's separation makes it easier to onboard to just the Python side. |
| **Test Trustworthiness** | 7 | 8 | 6 | TGI has solid e2e tests. vLLM tests are often flaky due to async complexity. |
| **Operational Maturity**| 7 | 9 | 5 | TGI is built specifically for Hugging Face production. |
| **Integration Readiness**| 6 | 8 | 4 | TGI's gRPC server is ready for orchestration. |
| **Licensing Suitability**| 9 | 8 | 9 | Apache 2.0 all around, though TGI had some historical license changes (now back to Apache 2.0). |

## 6. Integration Opportunity Mapping

**Opportunity 1:** TGI's Rust/Python gRPC boundary.
- **Where:** `router/` and `server/`.
- **Why:** Isolates our complex ROCm kernels from the asyncio web server.
- **Difficulty:** High.
- **Recommendation:** Adapt to our architecture.

**Opportunity 2:** LMDeploy's KV Cache Quantization Interface.
- **Where:** TurboMind engine configurations.
- **Why:** Informs how to cleanly expose RotorQuant/TurboQuant parameters.
- **Difficulty:** Low.
- **Recommendation:** Adopt directly (interface design only).

**Opportunity 3:** vLLM's Abstract PagedAttention Scheduler.
- **Where:** `vllm/core/scheduler.py`
- **Why:** Robust handling of dynamic batching and token eviction.
- **Difficulty:** Medium.
- **Recommendation:** Use as inspiration only.

### Categorizations
- **Fast Wins:** LMDeploy's config parameter style for exposing RQ/TQ bit-widths cleanly.
- **Strategic Extractions:** TGI's service-boundary concept (splitting our `async_llm_engine.py` into a distinct Router and Worker).
- **Attractive Traps:** vLLM's worker multiprocessing implementation (brittle on ROCm with NCCL).

## 7. Adoption Plan
**Target Architecture:** Adopt TGI's bounded service architecture conceptually. We will maintain our Python core but introduce a strict internal abstraction boundary between request routing/batching and the execution engine (where ROCm/PrismML logic lives).
**Seams:** Create a new `sglang/core/router.py` that handles all incoming requests and batching, completely decoupled from `sglang/worker/`.
**Rollout Order:**
1. Implement the generic `Router` interface with standard Pydantic models.
2. Build an in-memory adapter that passes requests from the Router to our existing Worker.
3. Decouple the Worker to accept pure batch configurations, removing its dependency on the web server state.
**Fallback Strategy:** Revert the internal adapter boundary and continue using the current monolith if latency overhead is >5%.
**Wrap vs Fork:**
- Wrap: vLLM's PagedAttention scheduling logic behind a clean interface.
- Fork: None. We build our own ROCm-specific worker implementation.
- Steal the idea: TGI's architectural boundary without using their exact Rust code.

## 8. Concrete Work Items

- **Ticket 1: Decouple Request Routing from Model Execution**
  - **Purpose:** Extract the routing logic from `AsyncEngine` into a standalone `Router` class to mirror TGI's architecture.
  - **Affected Area:** `sglang/srt/server.py`, `sglang/srt/managers/`
  - **Dependency Order:** 1
  - **Risk Level:** High (touches core entrypoints)
  - **Acceptance Criteria:** A `Router` class exists that accepts requests and outputs standardized `Batch` objects without importing PyTorch or worker logic.

- **Ticket 2: Standardize Worker Input Protocol**
  - **Purpose:** Define a strict protocol (Pydantic models) for communication between the `Router` and `Worker`.
  - **Affected Area:** `sglang/srt/managers/schedule_batch.py`
  - **Dependency Order:** 2
  - **Risk Level:** Medium
  - **Acceptance Criteria:** `Worker` methods only accept the new `Batch` objects; all tests pass.

- **Ticket 3: Isolate KV Cache Quantization Parameters**
  - **Purpose:** Create a configuration layer inspired by LMDeploy to neatly manage RotorQuant/TurboQuant parameters without polluting the worker.
  - **Affected Area:** `sglang/srt/managers/kv_cache_manager.py`
  - **Dependency Order:** 3
  - **Risk Level:** Low
  - **Acceptance Criteria:** RQ/TQ parameters are cleanly passed via a single `QuantConfig` object.

### Recommended PR Strategy
- **Suggested First PR:** Implement the strict `Batch` Pydantic models (Ticket 2) to establish the safe, high-leverage data boundary before changing any execution logic.
- **Suggested Second PR:** Extract the `Router` class (Ticket 1) and hook it up to the existing worker via the new `Batch` objects.
- **What NOT to do:** Do not import the entire vLLM `Scheduler` class directly; it brings too much async baggage. Build our own lean scheduler around the PagedAttention concept.

## 9. Final Recommendation
**Best For:**
- **vLLM:** Best Reference Architecture (for PagedAttention logic).
- **TGI:** Best Internal Fork Candidate (for architectural boundaries).
- **LMDeploy:** Best to Learn From, Not Adopt (due to C++ lock-in).
**Verdict:** We should structurally adapt TGI's separation of concerns while porting vLLM's PagedAttention logic, heavily customized for our ROCm/RotorQuant environment.

## 10. Horizon Scanning
1. **The Rising Star:** SGLang (Upstream) - Excellent for dynamic batching and structured generation (RadixAttention). Not deep-dived because it is our upstream base.
2. **The Legacy Standard:** DeepSpeed-MII - Good historical reference for multi-GPU communication. Not deep-dived due to architectural stagnation compared to vLLM/TGI.
3. **The Niche Specialist:** ExLlamaV2 - Unmatched for local, highly quantized (GPTQ/EXL2) single-GPU inference. Not deep-dived because it does not support distributed serving or dynamic batching required for our enterprise targets.

## 11. Appendix: Evidence Notes
- vLLM evaluation based on `vllm/engine/async_llm_engine.py` and `vllm/worker/model_runner.py`.
- TGI evaluation based on architectural separation in `router/` and `server/` directories.
- LMDeploy evaluation based on the dominance of C++ source files in the `lmdeploy/turbomind/` directory.
