# Comparative Repository Audit: vLLM vs TGI vs LMDeploy

## 1. Executive Technical Summary
This report analyzes three leading high-throughput inference engines: **vLLM**, **Text Generation Inference (TGI)**, and **LMDeploy**. The goal is to evaluate their internal architectures, maintainability, performance potential, and integration suitability for our system (`sglang-1-bit-turbo` on AMD ROCm).

**Key Findings:**
- **vLLM** provides the most advanced algorithmic research (PagedAttention, chunked prefill) but suffers from high framework entanglement and a monolithic async architecture.
- **TGI** excels in operational maturity with a strict service-oriented boundary (Rust router / gRPC / Python execution core), offering excellent stability but high opinionated lock-in.
- **LMDeploy** offers a lean, highly optimized C++ core (TurboMind) for deployment, but lacks the broader ecosystem compatibility of vLLM or the service-oriented boundaries of TGI.

**Strategic Recommendation:** Do not adopt any engine monolithically. Instead, adopt TGI's bounded service architecture (strict Router/Server IPC separation) as a reference model, while selectively extracting vLLM's memory management algorithms (PagedAttention) and adapting them to our existing system.

## 2. Repository Targets & Assumptions
The three target repositories are inferred from the `triangulator-forge.md` journal and context clues regarding inference engines, KV cache coupling, and cross-engine architecture.
- **Repo A:** [vLLM](https://github.com/vllm-project/vllm)
- **Repo B:** [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
- **Repo C:** [LMDeploy](https://github.com/InternLM/lmdeploy)

**Assumptions:** The primary integration target is a high-throughput, latency-sensitive serving engine for 1-bit quantized models running on AMD ROCm (gfx1030). We prioritize clean adapter boundaries and abstraction over rapid prototyping.

## 3. Per-Repo Deep Audit

### 4. Repo A: vLLM
- **Core Architecture & Logic Flow:**
  - **Entrypoint:** `vllm/engine/async_llm_engine.py` orchestrates the event loop.
  - **Data Flow:** Requests enter the `AsyncLLMEngine`, are queued, and a monolithic scheduler (`vllm/core/scheduler.py`) determines batching. The `ModelRunner` executes the forward pass, coupled with `CacheEngine`.
  - **Type:** Framework shell/monolithic async application.
  - **Concurrency Model:** asyncio event loop with worker processes.
- **Functional Decomposition & "The Heart":**
  - **Hot Path:** The scheduling loop in `vllm/core/scheduler.py` and the attention kernels (`vllm/attention/ops/`).
  - **Critique:** The scheduler is tightly coupled to the hardware-specific KV cache block manager, making it difficult to inject custom 1-bit quantization schemes without modifying the core loop.
  - **Complexity Score:** 7/10 (High framework entanglement, deeply nested abstractions).
  - **Unique Value Proposition:** PagedAttention implementation.
- **Dependency & Health Audit:**
  - Heavy reliance on PyTorch, Ray (for distributed), and Triton/CUDA (`requirements.txt`).
  - Deep and fragile dependency tree, tightly coupled to specific PyTorch and CUDA versions.
  - Excellent contributor spread and pulse, but at the cost of rapid architectural churn.
- **Developer Experience & Integration:**
  - Boilerplate: High, requires setting up complex `EngineArgs` and `AsyncLLMEngine`.
  - Tests: Good unit coverage, but heavily rely on mocked CUDA environments. Tests decorate rather than protect the core hot path from structural changes.
  - Integration Surface: SDK entrypoints exist but are brittle to internal refactors.
- **Lock-in & Migration Risk:**
  - **Question:** How difficult would it be to migrate away in 5 years?
  - **Analysis:** High framework lock-in. Migrating away would require rewriting the entire serving layer.
  - **Rating:** High.
  - **Action:** Isolate behind a custom gRPC/REST adapter now.

### 5. Repo B: Text Generation Inference (TGI)
- **Core Architecture & Logic Flow:**
  - **Entrypoint:** `router/src/main.rs` (Rust API gateway) and `server/text_generation_server/server.py` (Python gRPC server).
  - **Data Flow:** The Rust router handles HTTP, validation, and batching queues. It communicates via gRPC to the Python backend, which executes the forward pass.
  - **Type:** Service-oriented architecture.
  - **Concurrency Model:** Rust async/tokio for routing, separate process for Python execution.
- **Functional Decomposition & "The Heart":**
  - **Hot Path:** The Rust continuous batcher (`router/src/batcher.rs`) and the Python model forward pass (`server/text_generation_server/models/`).
  - **Critique:** The rigid gRPC boundary isolates the scheduler from the execution engine perfectly. However, the custom model implementations require explicit integration per architecture.
  - **Complexity Score:** 5/10 (Clean boundaries, but polyglot complexity).
  - **Unique Value Proposition:** Production-grade operational stability.
- **Dependency & Health Audit:**
  - Dependencies: Rust (`Cargo.toml`), Python (`pyproject.toml`), gRPC.
  - Flat and controlled dependency tree in Rust, moderately layered in Python.
  - Strong corporate backing (HuggingFace) with steady release cadence.
- **Developer Experience & Integration:**
  - Boilerplate: Low for usage, very high for extending.
  - Tests: Robust E2E tests, strong CI gates. Tests protect the hot path.
  - Integration Surface: Excellent stable HTTP interfaces, but invasive internal extension.
- **Lock-in & Migration Risk:**
  - **Question:** How difficult would it be to migrate away in 5 years?
  - **Analysis:** Moderate. The API is standard OpenAI-compatible, but extending it internally locks you into HuggingFace's custom modeling conventions.
  - **Rating:** Moderate.
  - **Action:** Sandboxing the execution core is immediate priority.

### 6. Repo C: LMDeploy
- **Core Architecture & Logic Flow:**
  - **Entrypoint:** `lmdeploy/serve/async_engine.py` or C++ TurboMind API.
  - **Data Flow:** Python API wraps a highly optimized C++ engine (TurboMind).
  - **Type:** CLI-first with a wrapper around a compiled C++ core.
  - **Concurrency Model:** Threading in C++, asyncio in Python wrapper.
- **Functional Decomposition & "The Heart":**
  - **Hot Path:** The C++ scheduling and execution kernels in `src/turbomind/`.
  - **Critique:** Extremely fast, but extending the C++ core with custom 1-bit quantization for AMD ROCm requires significant low-level engineering.
  - **Complexity Score:** 8/10 (High barrier to entry for custom modifications).
  - **Unique Value Proposition:** TurboMind C++ inference engine.
- **Dependency & Health Audit:**
  - Python/C++ split. Lean dependency profile for deployment, but complex build chain.
  - Risk of maintainer concentration.
- **Developer Experience & Integration:**
  - Boilerplate: Medium. Setup friction is high due to compilation.
  - Tests: Integration tests focus on correctness against upstream models.
  - Integration Surface: Wrapper-friendly but invasive if modifying C++ core.
- **Lock-in & Migration Risk:**
  - **Question:** How difficult would it be to migrate away in 5 years?
  - **Analysis:** Severe if deeply integrated with TurboMind APIs.
  - **Rating:** Severe.
  - **Action:** Avoid deep TurboMind integration; use only standard API endpoints.

## 7. Feature Parity Table
| Feature | vLLM | TGI | LMDeploy |
| :--- | :--- | :--- | :--- |
| **Plugin System** | Weak (Monolithic) | Weak (Hardcoded models) | Weak |
| **Architecture** | Async Event Loop | Rust Router / Python gRPC | Python / C++ Engine |
| **Scheduling** | Python (Coupled) | Rust (Isolated) | C++ / Python |
| **Observability** | Prometheus, OTEL | Prometheus, Tracing (Rust) | Basic Logging |
| **Hardware Coupling**| High (CUDA/ROCm via Torch) | High (Custom Kernels) | Very High (TurboMind) |

## 8. Comparative Trade-off Matrix
| Metric | vLLM Score | TGI Score | LMDeploy Score | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Architectural Clarity** | 4/10 | 9/10 | 6/10 | vLLM is entangled; TGI has strict boundaries; LMDeploy is split Python/C++. |
| **Maintainability** | 5/10 | 8/10 | 6/10 | vLLM's rapid churn makes it hard; TGI's strictness enforces quality. |
| **Extensibility** | 6/10 | 5/10 | 4/10 | vLLM allows Python hacking; TGI requires polyglot knowledge; LMDeploy requires C++. |
| **Performance Potential**| 9/10 | 8/10 | 9/10 | vLLM and LMDeploy lead in sheer algorithmic optimization. |
| **Dependency Risk** | 7/10 | 4/10 | 5/10 | vLLM has deep Pytorch/Ray lock-in; TGI is self-contained. |
| **Migration Flexibility**| 3/10 | 7/10 | 4/10 | TGI's router architecture makes swapping execution easier. |
| **DX / Onboarding** | 8/10 | 6/10 | 7/10 | vLLM is easy to pip install; TGI requires Rust toolchains. |
| **Test Trustworthiness** | 7/10 | 8/10 | 6/10 | TGI has the strongest CI and boundary testing. |
| **Operational Maturity** | 5/10 | 9/10 | 6/10 | TGI is built for robust production ops; vLLM is research-first. |
| **Integration Readiness**| 5/10 | 8/10 | 6/10 | TGI's strict gRPC makes wrapping easy but deep integration hard. |
| **Licensing Suitability**| 10/10 | 10/10 | 10/10 | All are Apache 2.0. |

## 9. Integration Opportunity Mapping
- **Opportunity 1: Strict IPC Boundary (TGI)**
  - **Area:** `router/src/server.rs` -> `server/text_generation_server/server.py`
  - **Value:** Separates the batching loop from the PyTorch execution environment.
  - **Difficulty:** High
  - **Recommendation:** Adapt to our architecture. Do not use gRPC due to overhead; explore shared-memory or ZMQ boundaries.
- **Opportunity 2: PagedAttention Abstraction (vLLM)**
  - **Area:** `vllm/core/block_manager.py`
  - **Value:** Industry-standard KV cache memory management.
  - **Difficulty:** Medium
  - **Recommendation:** Extract and decouple from the scheduler.

**Identified Patterns:**
- **Fast Wins:** Implement a clean `KVBlockManager` interface to abstract memory allocation from scheduling (1 week).
- **Strategic Extractions:** Porting pure PagedAttention logic from vLLM into standalone utility functions without pulling in the event loop.
- **Attractive Traps:** Attempting to adopt TGI's exact gRPC boundary for 1-bit models; the serialization overhead will destroy performance gains.

## 10. Adoption Plan
1. **Target Architecture:** Adopt a TGI-style bounded service architecture using shared memory IPC instead of gRPC. Implement a clean interface between the `Scheduler` and the `ExecutionEngine`.
2. **Isolation Layers:** Build an abstract `KVBlockManager` interface that the `Scheduler` talks to, completely hiding the physical memory allocation.
3. **Execution:** Extract the pure mathematical algorithms for PagedAttention from vLLM, wrap them in pure functions, and inject them into our execution layer.
4. **Rollout Order:** Phase 1: Refactor Scheduler. Phase 2: Implement IPC boundary. Phase 3: Integrate custom ROCm kernels.
5. **Fallback Strategy:** If IPC overhead is too high, revert to a modular monolith like vLLM but maintain strict class interfaces.

## 11. Concrete Work Items
- **Ticket 1: Abstract KV Cache Manager Boundary**
  - **Purpose:** Decouple scheduling from physical memory allocation.
  - **Affected Area:** `sglang.srt.managers`
  - **Dependency Order:** 1
  - **Risk Level:** High
  - **Acceptance Criteria:** Scheduler operates exclusively on abstract block IDs; physical allocation is handled by a backend-specific manager.
- **Ticket 2: Extract PagedAttention Utilities**
  - **Purpose:** Port pure PagedAttention logic without vLLM's event loop.
  - **Affected Area:** `sglang.srt.layers`
  - **Dependency Order:** 2
  - **Risk Level:** Medium
  - **Acceptance Criteria:** PagedAttention can be executed as a pure function passing only tensors and block tables.
- **Ticket 3: Implement Zero-Copy IPC Interface**
  - **Purpose:** Connect router to execution engine without serialization overhead.
  - **Affected Area:** `sglang.srt.server`
  - **Dependency Order:** 3
  - **Risk Level:** High
  - **Acceptance Criteria:** IPC benchmark shows <100us overhead compared to in-process calls.

**Suggested PR Strategy:**
- **First PR:** Create the abstract `KVBlockManager` interface and update the scheduler to use it (safest high-leverage foundation).
- **Second PR:** Extract pure PagedAttention functions from vLLM into a new standalone utility module.
- **What NOT to do:** Do not import vLLM's `AsyncLLMEngine` wholesale; doing so imports their monolithic scheduling assumptions.

## 12. Final Recommendation
- **Best for Experimental High-Upside:** vLLM
- **Best Reference Architecture:** TGI
- **Best to Mine for Ideas:** LMDeploy (TurboMind memory patterns)

**Verdict:** Do not adopt any repository directly. **Selectively adapt** TGI's strict router/execution boundary concept, while **extracting** vLLM's memory management algorithms.

## 13. Horizon Scanning
1. **The Rising Star:** `SGLang` (Original upstream). Highly optimized for structured generation, relevant for our RadixAttention focus.
2. **The Legacy Standard:** `FasterTransformer` (NVIDIA). Replaced by TensorRT-LLM, but excellent reference for pure C++ attention.
3. **The Niche Specialist:** `MLC-LLM`. Best for edge/device deployment using TVM, valuable for cross-platform compilation insights.

## 14. Appendix: Evidence Notes
- **vLLM Entanglement:** Verified via `vllm/engine/async_llm_engine.py` where the scheduling step and model execution are tightly bound in the `step_async` method.
- **TGI Boundary:** Verified via TGI's architectural split between the Rust `router` workspace and Python `server` component.
- **KV Cache Deception:** As noted in `triangulator-forge.md`, across engines the block IDs managed by the scheduler often bleed into the hardware-specific tensors managed by the engine.
