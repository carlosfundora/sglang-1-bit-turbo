# Comparative Repository Audit: vLLM vs TGI vs LMDeploy

## 1. Executive Technical Summary

This report provides a decision-grade analysis of three prominent open-source inference engines for large language models: **vLLM**, **Text Generation Inference (TGI)**, and **LMDeploy**. The goal is to evaluate their architecture, maintainability, dependency health, developer ergonomics, and lock-in risk to extract concrete integration opportunities for the `sglang-1-bit-turbo` project, specifically focusing on our core directive of AMD ROCm compatibility, RotorQuant KV cache compression, and high-throughput speculative decoding.

**Key Findings:**
*   **vLLM** offers the most ubiquitous ecosystem integration and pioneered the critical `PagedAttention` algorithm, but its monolithic, heavily entangled async architecture creates operational brittleness and makes core engine modification difficult. It is the best repository to *mine for algorithmic ideas* (specifically PagedAttention and continuous batching) but perilous to adopt as a wholesale framework due to framework lock-in.
*   **TGI (Text Generation Inference)** presents the strongest, most resilient internal architecture via strict service-oriented boundaries (Rust router communicating with a Python execution server via gRPC). This isolation prevents the event loop from stalling during heavy inference. However, its opinionated ecosystem and restrictive licensing create severe lock-in risk. It is the best *reference architecture* for building a robust API/batching layer.
*   **LMDeploy** excels in high-performance GPU scheduling and optimized quantization kernels (particularly for AWQ and TurboMind). While less globally popular than vLLM, its execution engine is often faster and leaner, though its custom C++ backend creates a steep learning curve. It is an *experimental high-upside* repository to study for extreme scheduling efficiency, though its KV cache and scheduler coupling presents challenges for custom quantization schemes.

**Final Recommendation:** Do not adopt any repository wholesale. Instead, adapt our current `sglang` fork to emulate TGI’s strict router/execution boundary (via IPC) to resolve async event loop stalls, while extracting vLLM's `PagedAttention` logic into pure, framework-agnostic utilities to support our custom RotorQuant integrations.

---

## 2. Repository Targets & Assumptions

*   **Repo A:** `vLLM` (`vllm-project/vllm`) - The current community standard for LLM serving, known for PagedAttention and continuous batching. Selected as the baseline for algorithmic innovation and ecosystem compatibility.
*   **Repo B:** `Text Generation Inference` (`huggingface/text-generation-inference`) - Hugging Face's production serving solution. Selected for its distinct Rust/Python service-oriented architecture and focus on production stability.
*   **Repo C:** `LMDeploy` (`InternLM/lmdeploy`) - A high-performance inference engine from InternLM. Selected for its reputation for raw throughput and optimized custom backends (TurboMind).

*Assumption:* These three represent the primary architectural paradigms in the current LLM serving landscape (Monolithic Async, Service-Oriented Polyglot, and Optimized C++ Engine). The analysis focuses on their suitability as inspiration for our ROCm/RotorQuant optimized fork.

---

## 3. Per-Repo Deep Audit

### Repo A: vLLM (`vllm-project/vllm`)

**A. Core Architecture & Logic Flow**
vLLM operates primarily as a monolithic asynchronous Python application. The primary entrypoint is typically the `AsyncLLMEngine` (in `vllm/engine/async_llm_engine.py`), which wraps the core `LLMEngine`. The orchestrator (the `Scheduler` in `vllm/core/scheduler.py`) manages a continuous batching queue and allocates block-level KV cache memory using the `BlockSpaceManager` (in `vllm/core/block_manager.py`). The actual execution engine is a mix of PyTorch model definitions and custom CUDA/HIP kernels (e.g., `vllm/attention/ops/paged_attn.py`). The concurrency model relies heavily on Python's `asyncio` event loop mixed with threading for background tasks. It is fundamentally a framework-bound application.

**B. Functional Decomposition & “The Heart”**
The hot path begins in `AsyncLLMEngine.generate`, passes to `LLMEngine.step`, where the `Scheduler` determines which sequences to schedule. The `Worker` (in `vllm/worker/worker.py`) then executes the forward pass, invoking the model and the `PagedAttention` kernels. A recurring architectural deception here is the tight coupling between the `Scheduler` and the `BlockSpaceManager`; the scheduler has deep knowledge of physical blocks, making it difficult to inject custom hardware quantization (like our RotorQuant).
*   **Unique Value Prop:** `PagedAttention` and continuous batching algorithms.
*   **Complexity Score:** 7/10. Abstraction depth is moderate, but framework entanglement (heavy async usage intertwined with synchronous CUDA calls) and coupling (Scheduler + KV Cache) elevate complexity.

**C. Dependency & Health Audit**
Dependencies (`requirements.txt`, `pyproject.toml`) are heavy, anchoring deeply into `torch`, `xformers` (conditionally), and `triton`. The tree is moderately layered but prone to dependency hell due to strict PyTorch version requirements and CUDA/ROCm environment sensitivity. The project pulse is extremely high (frequent commits, massive PR activity), but maintainer concentration risk exists around core kernel development. License is Apache 2.0 (friendly).

### Repo B: Text Generation Inference (`huggingface/text-generation-inference`)

**A. Core Architecture & Logic Flow**
TGI uses a strict service-oriented architecture. The entrypoint is a Rust-based web server and router (`router/src/main.rs`). This router handles HTTP requests, continuous batching logic, and tokenization. It communicates via gRPC to a Python-based execution server (`server/text_generation_server/server.py`). The execution server manages the GPU model weights and executes the forward passes. The concurrency model is a robust actor model/event loop in Rust, safely isolated from the synchronous execution pipeline in Python.

**B. Functional Decomposition & “The Heart”**
The hot path triggers at the Rust router, which batches requests and sends a gRPC `Prefill` or `Decode` command to the Python server. The Python server (e.g., `server/text_generation_server/models/flash_causal_lm.py`) executes the FlashAttention or custom kernel forward pass. The true heart is the strict gRPC boundary (`proto/generate.proto`) that separates scheduling from execution. This prevents the Python GIL and long GPU operations from stalling the API server.
*   **Unique Value Prop:** Strict Router/Server isolation preventing event loop stalls.
*   **Complexity Score:** 8/10. The polyglot nature (Rust + Python) and gRPC overhead introduce significant complexity, though the boundaries are explicitly clear.

**C. Dependency & Health Audit**
Dependencies are split: Cargo (`router/Cargo.toml`) for Rust (axum, tonic, tokenizers) and Poetry/pip (`server/pyproject.toml`) for Python (torch, transformers). The Rust tree is relatively flat and controlled; the Python tree is framework-heavy. Project pulse is steady but highly controlled by Hugging Face. The license is a restrictive proprietary/open-core hybrid (Hugging Face License), strictly limiting commercial embedding or competitive use.

### Repo C: LMDeploy (`InternLM/lmdeploy`)

**A. Core Architecture & Logic Flow**
LMDeploy offers a dual architecture: a lightweight Python PyTorch engine and a highly optimized C++ backend named `TurboMind`. The primary entrypoint for high performance is the TurboMind pipeline (`lmdeploy/turbomind/turbomind.py` wrapping C++ extensions). Orchestration and scheduling are handled in C++ for maximum throughput, utilizing custom memory managers and persistent threads. It is fundamentally a layered app with a CLI/library front-end and a heavy compiled core.

**B. Functional Decomposition & “The Heart”**
The hot path resides in the C++ `TurboMind` engine (`src/turbomind/models/llama/LlamaV2.cc` or similar). It features extremely aggressive continuous batching and custom quantization implementations (AWQ, W4A16). Unlike vLLM, the scheduling logic is pushed down into the compiled layer, reducing Python overhead but making it incredibly opaque to modify. Like vLLM, the KV cache management is tightly coupled to the scheduler.
*   **Unique Value Prop:** The `TurboMind` C++ inference engine for raw, sustained throughput.
*   **Complexity Score:** 9/10. The heavy reliance on custom C++ kernels and complex bindings makes it very difficult to extend without deep low-level expertise.

**C. Dependency & Health Audit**
Dependencies are complex, requiring careful compilation of the `TurboMind` backend via CMake, alongside standard Python ML dependencies (`requirements.txt`). The dependency tree is deep and fragile regarding build environments. Project pulse is healthy within its specific community, though issue resolution can be slower for niche hardware. License is Apache 2.0.

---

## 4. Feature Parity Table

| Feature | vLLM | TGI | LMDeploy |
| :--- | :--- | :--- | :--- |
| **Plugin/Module System** | Weak (Monolithic) | Weak (Monolithic) | Moderate (Dual Engines) |
| **CLI Support** | Yes (Entrypoint) | Yes (Entrypoint) | Yes (Entrypoint) |
| **API Surface** | OpenAI Compatible | Custom REST / gRPC | OpenAI Compatible |
| **Streaming** | Yes (Async Generator) | Yes (Server-Sent Events) | Yes |
| **Batching** | Continuous (Python) | Continuous (Rust Router) | Continuous (C++) |
| **KV Cache Mgmt** | PagedAttention | PagedAttention / FlashInfer | TurboMind Custom |
| **Auth/Security** | Basic API Key | Basic API Key / TLS | Basic API Key |
| **Logging/Observability**| Prometheus Metrics | OpenTelemetry / Prometheus | Basic Logging |
| **Test Harness Depth** | Strong (E2E, Unit) | Strong (E2E, Unit) | Moderate |

---

## 5. Comparative Trade-off Matrix

| Dimension | vLLM | TGI | LMDeploy |
| :--- | :--- | :--- | :--- |
| **Architectural Clarity** | Moderate. Monolithic, heavily entangled async/sync logic. | High. Strict gRPC boundary between router and execution. | Low (Python) / Moderate (C++). High barrier to entry. |
| **Extensibility** | Low. Modifying the scheduler requires deep internal refactors. | Low. Opinionated ecosystem and restrictive license. | Low. Requires C++ kernel development. |
| **Maintainability** | Moderate. Fast-moving target, frequent internal API breaks. | High. Rust router provides extreme stability. | Moderate. Complex build system. |
| **Performance Potential**| High. Excellent continuous batching and kernel optimization. | High. Zero-overhead routing, optimized execution. | Very High. TurboMind engine pushes hardware limits. |
| **Hot-Path Efficiency** | Moderate. Python event loop can become a bottleneck. | High. Rust router never blocks. | Very High. Pure C++ execution path. |
| **Dependency Risk** | High. Tied closely to PyTorch versions and specific CUDA/ROCm stacks. | Moderate. Controlled environments, but HF ecosystem lock-in. | High. Complex C++ build dependencies. |
| **Lock-in Risk** | High. Internal APIs are highly volatile and framework-bound. | Severe. Restrictive license limits commercial adoption. | Moderate. TurboMind is a black box, but APIs are standard. |
| **Integration Difficulty**| Wrapper-friendly, but invasive to modify internals. | Fork-required due to architecture and license. | Wrapper-friendly for inference, invasive for custom kernels. |
| **DX/Onboarding** | Excellent. Easy `pip install vllm`. | Moderate. Requires Docker or complex local setup (Rust+Python). | Moderate. Good documentation, but custom builds are hard. |
| **Test Trustworthiness** | High. Extensive regression suites. | High. Rigorous CI for production stability. | Moderate. Focused on benchmark performance over edge cases. |
| **Documentation Substance** | High. Detailed API and operational reference. | Moderate. Good deployment docs, weak internal code docs. | Low. Fast moving, often relies on translated Chinese docs. |
| **Community Health** | High. Massive corporate and open source backing. | Moderate. Strong backing from Hugging Face, but highly opinionated. | Moderate. Strong in Asia, growing globally. |
| **Licensing Suitability** | High. Apache 2.0. | Low. Restrictive Hugging Face License limits commercial forks. | High. Apache 2.0. |
| **Long-term Ownership Fit** | Moderate. We can fork it, but upstream churn will cause merge conflicts. | Low. License makes it non-viable for our project. | Low. C++ backend is too difficult for our Python-heavy team to maintain. |

---

## 6. Integration Opportunity Mapping

### Opportunity 1: Service-Oriented Isolation Boundary (Inspired by TGI)
*   **Feature:** Strict separation of the API/Batching router from the GPU Execution engine via IPC (Inter-Process Communication).
*   **Location:** Modeled after TGI's `router/` and `server/` gRPC boundary.
*   **Value:** Prevents the primary Python `asyncio` event loop from stalling during heavy synchronous ROCm kernel execution, a critical stability issue in current architectures.
*   **Difficulty:** High (Strategic Extraction)
*   **Recommendation:** **Adapt to our architecture.** We should not adopt Rust, but we must implement a strict multi-process boundary (e.g., via ZeroMQ or customized multiprocessing) in `sglang` separating the HTTP/Scheduler loop from the PyTorch worker processes.

### Opportunity 2: Pure PagedAttention Utilities (Inspired by vLLM)
*   **Feature:** The core block-mapping logic of PagedAttention, stripped of the `Scheduler` framework.
*   **Location:** Abstracted from `vllm/core/block_manager.py` and `vllm/attention/ops/paged_attn.py`.
*   **Value:** Allows us to manage memory efficiently while integrating our custom RotorQuant `rq3_iso` and `rq4_planar` data types without fighting vLLM's internal framework.
*   **Difficulty:** Medium
*   **Recommendation:** **Adopt selectively.** Extract the logical mapping functions into pure utility classes that our custom ROCm/Triton attention backends can consume.

### Opportunity 3: Decoupled KV Cache Manager (Avoiding vLLM/LMDeploy Traps)
*   **Feature:** A strict abstraction boundary between the `Scheduler` and the `KVCacheManager`.
*   **Location:** To be built in `sglang/srt/managers/`.
*   **Value:** Solves the recurring architectural deception where the scheduler knows about physical memory types. The scheduler should only handle logical token counts and abstract block IDs.
*   **Difficulty:** Medium
*   **Recommendation:** **Implement as net-new.** Use the failure of vLLM and LMDeploy to easily support new quant schemes as the exact reason to build this boundary.

---

## 7. Adoption Plan

**Target Architecture for `sglang-1-bit-turbo`:**
We will maintain the core `sglang` foundation but pivot the internal architecture toward a decoupled, multi-process model.

1.  **Isolation Layer (The TGI Influence):** Introduce a strict multiprocessing boundary between the API/Router process and the GPU execution Worker(s).
2.  **Scheduler Decoupling (The Anti-vLLM Pattern):** Refactor the `Scheduler` to output logical block allocations, delegating physical allocation and data type management strictly to a hardware-specific `KVCacheManager`.
3.  **Kernel Integration:** Adapt our Triton-based RotorQuant kernels (PlanarQuant and IsoQuant) behind this new `KVCacheManager` boundary, ensuring the `Scheduler` is completely unaware of the 3-bit or 4-bit underlying representations.

**Rollout Order:**
1.  Refactor Scheduler/KV Cache boundary (Zero logic changes, just interface segregation).
2.  Implement IPC boundary between API and Worker.
3.  Integrate RotorQuant kernels behind the new KV Manager.

**Fallback Strategy:**
If the IPC boundary introduces too much latency, fallback to threaded execution but maintain the strict interface separation to preserve code health.

---

## 8. Concrete Work Items

*   **Ticket 1: Decouple Scheduler and BlockManager Interfaces**
    *   *Purpose:* Break the tight coupling where the scheduler knows physical memory layouts.
    *   *Affected Area:* `sglang.srt.managers.scheduler`
    *   *Dependency Order:* 1
    *   *Risk Level:* Medium
    *   *Acceptance Criteria:* `Scheduler` operates exclusively on integer block IDs; `KVCacheManager` handles all PyTorch tensor allocations. All tests pass.

*   **Ticket 2: Extract PagedAttention Logical Mapping**
    *   *Purpose:* Isolate the block mapping logic from vLLM into pure functions for reuse.
    *   *Affected Area:* `sglang.srt.utils.paged_attention_utils` (New)
    *   *Dependency Order:* 2
    *   *Risk Level:* Low
    *   *Acceptance Criteria:* Provide pure functions that map sequence IDs to block tables without requiring framework context.

*   **Ticket 3: Implement Multiprocessing API/Worker Boundary**
    *   *Purpose:* Prevent event loop stalls during ROCm kernel execution.
    *   *Affected Area:* `sglang.srt.server`, `sglang.srt.managers.io_struct`
    *   *Dependency Order:* 3
    *   *Risk Level:* High
    *   *Acceptance Criteria:* API requests are passed to the worker via a dedicated queue/IPC mechanism; the main `asyncio` loop remains responsive during heavy load.

**First PR:**
Focus entirely on Ticket 1. It creates the safest high-leverage foundation for all future memory optimizations without changing runtime behavior.

**Second PR:**
Focus on Ticket 2 (Extract PagedAttention Logical Mapping). This extracts real, usable algorithmic logic from the vLLM research into pure utilities that our RotorQuant kernels can consume immediately.

**What NOT to do:**
*   Do NOT import vLLM's `AsyncLLMEngine` wholesale. It will entangle our codebase and break our ROCm optimizations.
*   Do NOT attempt to write a Rust router (TGI style) yet. The polyglot overhead is not justified for our current team size.

---

## 9. Final Recommendation

**Best For:**
*   **vLLM:** Best Reference Architecture for PagedAttention Algorithms.
*   **TGI:** Best Reference Architecture for Operational Stability and Component Isolation.
*   **LMDeploy:** Experimental High-Upside for Raw Kernel Optimization.

**Verdict:**
Do not adopt any of the three repositories directly. `vLLM` is too monolithic and brittle; `TGI` is too legally restrictive and polyglot-heavy; `LMDeploy` is too deeply coupled to C++. We must continue building upon our `sglang` fork, but we must urgently **mine TGI for its architectural boundaries (router vs worker)** and **mine vLLM for its PagedAttention algorithms**, while explicitly avoiding the tight scheduler-to-hardware coupling seen in both vLLM and LMDeploy.

---

## 10. Horizon Scanning

1.  **The Rising Star: SGLang (upstream)**
    *   *Category:* High-performance structured generation.
    *   *Why it matters:* Utilizes RadixAttention for massive caching improvements in multi-turn prompts. It is our upstream base, but we must watch their core architectural changes closely to avoid painful merge conflicts while we implement our IPC boundaries.

2.  **The Legacy Standard: Hugging Face Transformers (`generate()`)**
    *   *Category:* Baseline Reference.
    *   *Why it matters:* It defines the standard API and tokenizer behaviors. It was not deep-dived because it is not an inference engine, but a synchronous library; however, all engines must ultimately match its mathematical output.

3.  **The Niche Specialist: MLC LLM**
    *   *Category:* Universal Edge Deployment.
    *   *Why it matters:* Uses Apache TVM to compile models to highly optimized native code for any backend (iOS, WebGPU, ROCm). Not deep-dived because its compilation-heavy AOT architecture conflicts with our requirement for dynamic continuous batching and Python-level speculative decoding control.

---

## 11. Appendix: Evidence Notes

*   **Scoring (1-10):**
    *   **vLLM:**
        *   Architectural Clarity: 5
        *   Maintainability: 6
        *   Extensibility: 4
        *   Performance Potential: 8
        *   Dependency Risk: 7
        *   Migration Flexibility: 4
        *   DX / Onboarding: 9
        *   Test Trustworthiness: 8
        *   Operational Maturity: 7
        *   Integration Readiness: 8
        *   Licensing Suitability: 10
    *   **TGI:**
        *   Architectural Clarity: 9
        *   Maintainability: 8
        *   Extensibility: 3
        *   Performance Potential: 8
        *   Dependency Risk: 5
        *   Migration Flexibility: 2
        *   DX / Onboarding: 6
        *   Test Trustworthiness: 9
        *   Operational Maturity: 9
        *   Integration Readiness: 5
        *   Licensing Suitability: 1
    *   **LMDeploy:**
        *   Architectural Clarity: 6
        *   Maintainability: 5
        *   Extensibility: 3
        *   Performance Potential: 9
        *   Dependency Risk: 8
        *   Migration Flexibility: 5
        *   DX / Onboarding: 6
        *   Test Trustworthiness: 6
        *   Operational Maturity: 7
        *   Integration Readiness: 6
        *   Licensing Suitability: 10
    *   *Note: Detailed justifications for these scores are integrated into the Comparative Trade-off Matrix (Section 5) and the Executive Summary (Section 1).*
*   **Journal Context:** This report explicitly addresses the findings in `.jules/journals/triangulator-forge.md` regarding the "recurring architectural deception" of tight KV cache/scheduler coupling and the "Service-Oriented Integration vs Framework Entanglement" trade-off.
