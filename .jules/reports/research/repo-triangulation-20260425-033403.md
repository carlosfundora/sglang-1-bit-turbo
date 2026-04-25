# Comparative Repository Audit: vLLM vs Text-Generation-Inference vs LMDeploy

## 1. Executive Technical Summary

This report evaluates three dominant open-source LLM inference engines—**vLLM**, **Text-Generation-Inference (TGI)**, and **LMDeploy**—to determine their architectural rigor, integration viability, and value as reference architectures for our AMD ROCm/PrismML-focused system.

The core finding is a classic dichotomy between extensibility and operational stability.
- **vLLM** presents a massive surface area for integration but is architecturally compromised by an entangled, monolithic event loop that mixes API serving with low-level execution state.
- **TGI** enforces a strict gRPC service boundary between its Rust-based router and Python execution core, delivering superior stability at the cost of opinionated ecosystem lock-in (Hugging Face).
- **LMDeploy** (via TurboMind) provides exceptional hardware-specific C++ optimization but presents a steep learning curve and brittle integration surface outside its core supported pathways.

**Recommendation:** We should adopt **TGI's architectural shape** (a bounded, headless execution engine decoupled from a fast batching router) while **mining vLLM** for algorithmic concepts (PagedAttention, scheduling heuristics). We must avoid inheriting vLLM's dependency bloat and monolithic concurrency model.

## 2. Repository Targets & Assumptions

The following repositories were audited based on their utility as high-throughput, memory-efficient inference backends compatible with modern GPUs:

- **Repo A:** `vLLM` (`vllm-project/vllm`) - Python-first monolith emphasizing ecosystem integration.
- **Repo B:** `Text-Generation-Inference` (`huggingface/text-generation-inference`) - Service-oriented application separating HTTP/routing (Rust) from execution (Python/C++).
- **Repo C:** `LMDeploy` (`InternLM/lmdeploy`) - C++ core (TurboMind) with a Python wrapper focused on raw hardware speed.

**Assumption:** The analysis prioritizes architectural clarity and maintainability for long-term integration over short-term feature checklists. Popularity is treated as a risk vector (dependency churn) rather than an inherent strength.

## 3. Per-Repo Deep Audit

### 3.1 Repo A: vLLM

**A. Core Architecture & Logic Flow**
- **Entrypoints:** `vllm/entrypoints/cli/main.py`, `vllm/engine/async_llm_engine.py`
- **Data Flow:** HTTP Request -> Asyncio Event Loop -> `LLMEngine` -> `Scheduler` -> `Worker` -> GPU execution.
- **Orchestration:** Monolithic `asyncio` loop running the web server, batching queue, and model execution coordination in the same process.
- **Concurrency Model:** Async/await tightly coupled with synchronous GPU calls, requiring delicate thread-pool management and multiprocessing for tensor parallelism.
- **Architectural Shape:** Monolith with a thick plugin layer.

**B. Functional Decomposition & "The Heart"**
- **Hot Path:** `vllm.core.scheduler.Scheduler._schedule()` and `vllm.worker.model_runner.ModelRunner.execute_model()`.
- **The Heart:** The `BlockSpaceManager` (memory allocation) and the `Scheduler` (continuous batching logic).
- **Analysis:** The hot path is smeared. The `AsyncLLMEngine` is overloaded, handling API schemas, tokenization, and queue management simultaneously.
- **Unique Value Proposition:** The most comprehensive and community-tested implementation of PagedAttention.
- **Complexity Score:** 8/10. Abstraction depth is moderate, but coupling between the web layer and execution layer is high.

**C. Dependency & Health Audit**
- **Manifests:** `pyproject.toml`, `requirements*.txt`.
- **Dependencies:** Deep, fragile tree. Heavy reliance on `torch`, `xformers`, `triton`, `fastapi`, `pydantic`, and numerous hardware-specific packages.
- **Health:** Extremely active (dozens of daily PRs), which introduces significant commit churn and regression risk.
- **License Constraint:** Apache 2.0 (Permissive).

**D. Developer Experience & Integration**
- **Boilerplate:** High. Requires spinning up an entire async event loop or overriding internal signal handlers to embed.
- **API Style:** Over-abstracted. Heavy use of Pydantic models for internal state transitions.
- **Tests:** Broad integration tests and CI gates, but internal logic often lacks granular, mocked unit tests, making the hot path brittle to refactoring.
- **Integration Surface:** Invasive. Expects to own the Python process.

**E. Lock-in & Migration Risk**
- **Migration Risk:** High
- **Hardest to Replace:** The internal PagedAttention block manager and bespoke scheduling loop.
- **Isolation Strategy:** Must be sandboxed behind an HTTP/gRPC adapter immediately. Do not import `vllm` directly into business logic.

### 3.2 Repo B: Text-Generation-Inference (TGI)

**A. Core Architecture & Logic Flow**
- **Entrypoints:** Rust HTTP Server (`router/src/main.rs`), Python gRPC Server (`server/text_generation_server/cli.py`).
- **Data Flow:** Request -> Rust Router (Queue/Batching) -> gRPC -> Python Server -> Execution.
- **Orchestration:** Rust-based actor model (router) managing a pool of headless Python execution nodes via gRPC.
- **Concurrency Model:** Multi-threaded async Rust (Tokio) for I/O; synchronous Python processing behind the gRPC boundary.
- **Architectural Shape:** Service-oriented, bounded contexts.

**B. Functional Decomposition & "The Heart"**
- **Hot Path:** `router/src/batcher.rs` (Rust) and `server/text_generation_server/models/model.py` (Python).
- **The Heart:** The Rust continuous batcher. It abstracts the complexity of dynamic batching away from the GIL.
- **Analysis:** Cleanly isolated. The Python code does not know about HTTP requests or token queues; it only understands batched forward passes.
- **Unique Value Proposition:** Strict service boundary preventing HTTP/I/O issues from interfering with GPU execution state.
- **Complexity Score:** 5/10. Dual-language overhead, but exceptional modularity and internal documentation.

**C. Dependency & Health Audit**
- **Manifests:** `Cargo.toml`, `pyproject.toml`.
- **Dependencies:** Controlled Python tree, heavy Rust crate ecosystem. Strong reliance on `transformers` and Hugging Face Hub.
- **Health:** Steady, tightly managed by core maintainers. Less feature churn than vLLM, resulting in higher stability.
- **License Constraint:** Apache 2.0, but deeply entangled with Hugging Face's commercial ecosystem.

**D. Developer Experience & Integration**
- **Boilerplate:** Medium. The dual-language structure requires compiling Rust alongside Python setup.
- **API Style:** Composable and declarative at the Rust boundary; imperative Python core.
- **Tests:** Excellent Rust unit tests protecting the batching hot path. Python tests rely heavily on fixtures.
- **Integration Surface:** Clean transport abstraction via gRPC and stable REST endpoints.

**E. Lock-in & Migration Risk**
- **Migration Risk:** Moderate
- **Hardest to Replace:** The deep reliance on the Hugging Face ecosystem (Safetensors, tokenizers, hub downloads).
- **Isolation Strategy:** Wrap the model loading interface to decouple it from Hugging Face specific APIs.

### 3.3 Repo C: LMDeploy

**A. Core Architecture & Logic Flow**
- **Entrypoints:** `lmdeploy/cli/serve.py` -> C++ TurboMind Engine.
- **Data Flow:** Python Request -> PyBind11 -> C++ TurboMind Engine -> C++ Custom Kernels.
- **Orchestration:** Python acts merely as a thin shell over a C++ execution engine.
- **Concurrency Model:** Threaded C++ execution; Python simply waits on futures.
- **Architectural Shape:** CLI-first wrapper around a heavy C++ shared library.

**B. Functional Decomposition & "The Heart"**
- **Hot Path:** `src/turbomind/models/llama/LlamaBatch.cc`.
- **The Heart:** The custom C++ CUDA kernels and the TurboMind scheduler written entirely in C++.
- **Analysis:** Highly isolated from Python but deeply coupled to specific NVIDIA/CUDA paradigms.
- **Unique Value Proposition:** Maximum raw throughput on specific hardware configurations by bypassing Python overhead entirely.
- **Complexity Score:** 9/10. Extremely dense C++ abstraction; difficult for non-specialists to debug or extend.

**C. Dependency & Health Audit**
- **Manifests:** `CMakeLists.txt`, `requirements.txt`.
- **Dependencies:** Lean on Python dependencies, heavy on system-level CUDA/compiler toolchains.
- **Health:** Active, driven primarily by a single corporate sponsor (InternLM).
- **License Constraint:** Apache 2.0.

**D. Developer Experience & Integration**
- **Boilerplate:** High for C++ development, low for Python inference.
- **API Style:** Imperative. Python layer is a simple wrapper passing pointers.
- **Tests:** End-to-end benchmark heavy. Few unit tests protecting granular internal logic.
- **Integration Surface:** Minimal. Primarily integrates via raw shared libraries (`.so`) or pybind11.

**E. Lock-in & Migration Risk**
- **Migration Risk:** Severe
- **Hardest to Replace:** The TurboMind C++ core execution engine and custom weight formats.
- **Isolation Strategy:** Treat as a completely opaque black box; do not attempt to extend the internal C++ layer without deep vendor commitment.

## 4. Feature Parity Table

| Feature | vLLM | Text-Generation-Inference | LMDeploy |
| :--- | :--- | :--- | :--- |
| **Plugin/Module System** | Custom models/backends via registry | Opinionated, fixed models | Fixed C++ architecture |
| **Schema Validation** | FastAPI / Pydantic | Rust Serde | Basic Python |
| **Config Layering** | Messy CLI/Env mix | Clean CLI -> Env variables | CLI centric |
| **API Surface** | OpenAI Compatible (Native) | Custom / OpenAI compat | Custom |
| **Streaming** | Async generator over HTTP | Server-Sent Events (gRPC streaming) | Python generator |
| **Continuous Batching** | Python Scheduler (Complex state) | Rust Batcher (Clean boundary) | C++ Scheduler |
| **Observability** | Prometheus, OTEL metrics | Deep OTEL, Tracing (Rust) | Basic logging |
| **Test Harness Depth** | Broad, flaky E2E tests | Rigorous Rust unit tests | Benchmark-focused |

## 5. Comparative Trade-off Matrix

| Metric | vLLM | TGI | LMDeploy |
| :--- | :--- | :--- | :--- |
| **Architectural Clarity** | Poor (Entangled monolith) | Excellent (Strict boundaries) | Good (C++ core) |
| **Extensibility** | High (Python ecosystem) | Low (Opinionated) | Low (C++ barrier) |
| **Maintainability** | Low (High churn/coupling) | High (Strict CI/bounds) | Medium |
| **Performance Potential** | Very High | High | Very High |
| **Hot-Path Efficiency** | Smeared by Python overhead | Clean Python, fast Rust I/O | C++ Native |
| **Dependency Risk** | High | Medium | Low |
| **Lock-in Risk** | Moderate (Internal APIs) | High (HF Hub dependence) | Low |
| **Integration Difficulty**| Invasive (Takes over process) | Clean (Network boundary) | Wrapper-friendly |
| **DX / Onboarding** | Easy to use, hard to hack | Medium (Rust learning curve) | Hard (C++ heavy) |
| **Test Trustworthiness**| Moderate | High | Moderate |
| **Documentation Substance**| API Reference heavy | Operational guide heavy | Sparse internals |
| **Community Health** | Massive, chaotic | Concentrated | Niche |
| **Licensing Suitability**| Excellent | Good | Excellent |
| **Long-term Fit** | Unstable | Bounded | Rigid |

## 6. Integration Opportunity Mapping

### Opportunity 1: Bounded Rust Router Concept (from TGI)
- **Location:** `text-generation-inference/router/src/batcher.rs`
- **Why it is valuable:** Moving HTTP parsing, queueing, and continuous batching out of the Python process eliminates GIL contention and makes the execution engine purely stateless and testable.
- **Estimated Integration Difficulty:** High
- **Recommendation:** **Adapt to our architecture**. Build a lightweight IPC router in Go or Rust that feeds our existing execution core.

### Opportunity 2: PagedAttention Block Manager (from vLLM)
- **Location:** `vllm/core/block_manager.py`
- **Why it is valuable:** A highly tuned, hardware-agnostic conceptual model for virtualizing KV cache memory.
- **Estimated Integration Difficulty:** Medium
- **Recommendation:** **Mine for ideas / Extract**. Isolate the logical block allocation algorithms and implement them as pure functions.

### Opportunity 3: Speculative Decoding State Trees
- **Location:** `vllm/model_executor/layers/spec_decode/`
- **Why it is valuable:** Complex tree-verification logic required for EAGLE3/speculative models.
- **Estimated Integration Difficulty:** High
- **Recommendation:** **Avoid importing**. vLLM's implementation is deeply tied to their specific worker state. We must build this natively.

**Opportunity Classification:**
- **Fast Wins:** Extracting vLLM's configuration validation utilities.
- **Strategic Extractions:** Implementing TGI's gRPC/IPC execution boundary.
- **Attractive Traps:** Trying to embed `vllm.AsyncLLMEngine` as a library module. It will hijack signal handlers and event loops, causing instability.

## 7. Adoption Plan

**Recommended Target Architecture:** We will construct a **Bounded Execution Architecture**, adopting TGI's conceptual shape but retaining our highly optimized Rocm/PrismML execution core.

**Seams & Isolation Layers:**
1.  **Transport Boundary:** Create an IPC or gRPC protocol defining a strict boundary: `Router -> batched_requests -> Engine`.
2.  **State Boundary:** The Execution Engine must hold *zero* queue state. It only processes the physical batch it is handed.

**Rollout Order:**
1.  Define the internal protobuf/IPC contract.
2.  Wrap our existing ROCm execution code in a headless server adhering to the contract.
3.  Implement the router layer to handle HTTP and queueing.

**Fallback Strategy:** If the router overhead is too high, revert to an in-process `multiprocessing.Queue` implementation, preserving the API boundary but removing the network hop.

## 8. Concrete Work Items

**Ticket 1: Define Engine IPC Protocol**
- **Purpose:** Define a Protobuf schema for communication between a router and the execution engine.
- **Affected Area:** `src/ipc/`, `src/router/`
- **Dependency Order:** 1
- **Risk Level:** High
- **Acceptance Criteria:** A compiling `.proto` file and basic client/server stub code generating valid messages.

**Ticket 2: Extract Bounded Block Manager**
- **Purpose:** Build a stateless KV cache block manager inspired by vLLM, decoupled from any execution engine.
- **Affected Area:** `src/engine/memory/`
- **Dependency Order:** 2
- **Risk Level:** Medium
- **Acceptance Criteria:** A pure Python module that handles block allocation math with 100% unit test coverage.

**Ticket 3: Implement Headless Execution Server**
- **Purpose:** Wrap the ROCm/PrismML execution core in a headless service that consumes the IPC protocol.
- **Affected Area:** `src/engine/`
- **Dependency Order:** 3
- **Risk Level:** Medium
- **Acceptance Criteria:** The engine successfully processes a mock IPC request and returns output without relying on a web framework.

**Suggested PR Sequence:**
- **PR 1:** Safest high-leverage foundation: Implement the IPC schema (Ticket 1).
- **PR 2:** Extract real feature: The Bounded Block Manager (Ticket 2).

**What NOT to do:** Do not add `fastapi` or `asyncio` to the core execution loop. Do not pass HTTP Request objects into model code.

## 9. Final Recommendation

- **Best for Production Stability:** TGI
- **Best for Rapid Prototyping:** vLLM
- **Best Lightweight / Embedded:** LMDeploy
- **Best Internal Fork Candidate:** None
- **Best Reference Architecture:** TGI

**Final Verdict:** **TGI** is the best repository to use as a **reference architecture** to model our system boundaries. **vLLM** is best to **mine for algorithmic ideas** (PagedAttention logic). We should **avoid** adopting any of them directly, as doing so would compromise our specialized ROCm/EAGLE3 optimizations with generalized framework bloat.

## 10. Horizon Scanning

**1. The Rising Star: SGLang (upstream)**
- **Why selected:** It is the upstream for our current fork, featuring novel RadixAttention for prompt caching.
- **Category:** High-Performance Framework.
- **Why it matters:** RadixAttention drastically reduces TTFT for multi-turn conversations.
- **Why skipped:** We are already a specialized fork; we needed external perspective.

**2. The Legacy Standard: llama.cpp**
- **Why selected:** The absolute standard for pure C/C++ minimal dependency inference.
- **Category:** Lightweight / Embedded.
- **Why it matters:** Proves that complex orchestration is not strictly necessary for high performance.
- **Why skipped:** Lacks the enterprise serving features (continuous batching, gRPC) needed for our target scale.

**3. The Niche Specialist: TensorRT-LLM**
- **Why selected:** NVIDIA's highly optimized, compiled graph engine.
- **Category:** Vendor-Specific Compiler.
- **Why it matters:** Represents the ceiling of potential performance via static graph compilation.
- **Why skipped:** Fundamentally incompatible with our AMD ROCm hardware target.

## 11. Appendix: Evidence Notes

- `vLLM` code inspection confirms `AsyncLLMEngine` mixes Fastapi context and execution state, creating deep coupling.
- `TGI` code inspection shows `router/src/` handles all HTTP logic, while `server/` handles purely execution.
- Dependency trees generated via `pipdeptree` and `cargo tree` confirm vLLM's deep transitive dependencies compared to TGI's controlled environment.

================================================================
SCORING
================================================================

**vLLM**
- Architectural Clarity: 3/10 (Monolithic async event loop tangled with execution state).
- Maintainability: 4/10 (High churn, complex dependencies).
- Extensibility: 9/10 (Modular plugin design).
- Performance Potential: 9/10 (Excellent custom kernels).
- Dependency Risk: 2/10 (Deeply fragile Python tree).
- Migration Flexibility: 3/10 (High lock-in to its internal abstractions).
- DX / Onboarding: 6/10 (Easy external API, terrible internal debugging).
- Test Trustworthiness: 5/10 (Flaky, slow E2E tests).
- Operational Maturity: 7/10 (Widely deployed but operationally brittle).
- Integration Readiness: 4/10 (Invasive to embed).
- Licensing Suitability: 9/10 (Apache 2.0).

**TGI**
- Architectural Clarity: 9/10 (Strict gRPC boundary separating concerns).
- Maintainability: 8/10 (Strictly enforced CI, lower churn).
- Extensibility: 4/10 (Opinionated, hard to extend).
- Performance Potential: 8/10 (Optimized Rust router).
- Dependency Risk: 6/10 (Controlled, but heavy HF reliance).
- Migration Flexibility: 6/10 (Moderate lock-in).
- DX / Onboarding: 7/10 (Clear bounds, but dual-language).
- Test Trustworthiness: 8/10 (Excellent Rust unit tests).
- Operational Maturity: 9/10 (Designed for production).
- Integration Readiness: 8/10 (Clean network API).
- Licensing Suitability: 7/10 (Apache 2.0, tied to HF tooling).

**LMDeploy**
- Architectural Clarity: 6/10 (Standard C++ library pattern).
- Maintainability: 5/10 (Complex C++ codebase).
- Extensibility: 4/10 (Rigid hardware optimizations).
- Performance Potential: 9/10 (TurboMind is extremely fast).
- Dependency Risk: 8/10 (Lean Python, heavy CUDA).
- Migration Flexibility: 5/10 (Tied to specific formats).
- DX / Onboarding: 4/10 (High C++ barrier).
- Test Trustworthiness: 6/10 (Benchmark focused).
- Operational Maturity: 7/10 (Solid, less widely adopted).
- Integration Readiness: 6/10 (Usable via PyBind).
- Licensing Suitability: 9/10 (Apache 2.0).
