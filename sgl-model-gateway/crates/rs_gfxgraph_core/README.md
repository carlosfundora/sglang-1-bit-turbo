# rs_gfxgraph_core

Pure-Rust shared contracts for the **gfxGRAPH** ecosystem. Defines lightweight, high-performance DTOs, schemas, and enumerations without native bindings or external execution frameworks.

> Vendored engine copy: this crate intentionally excludes PyO3, policy mesh enforcement, embedded policy keys, and mandatory logly sidecars.
>
> Source lineage: extracted from the original gfxGRAPH workspace at `/home/local/ai/build/wip/gfxGRAPH`. This replaces direct engine linkage to the legacy PyO3 `rs_gfxgraph` and `rs_gfxgraph_stats` crates for contract-only validation.

---

## Architectural Role in the gfxGRAPH Family

```text
       ┌────────────────────────┐
       │   rs_gfxgraph_core     │◄─────── Pure-Rust contracts (zero-cost, fast compile)
       └───────────┬────────────┘
                   │
       ┌───────────┼───────────────┐
       ▼           ▼               ▼
┌──────────────┐ ┌───────────────┐ ┌──────────────────┐
│rs_gfxgraph   │ │rs_gfxgraph    │ │rs_gfxgraph_logly │
│   _pyo3      │ │   _logly      │ │ (Deep obs crate) │
│(PyO3 bindings│ │(Observability)│ └──────────────────┘
└──────────────┘ └───────────────┘
```

### Federation Crate Naming Convention

| Suffix | Role | Example |
|---|---|---|
| `_core` | Pure Rust logic, contracts | `rs_gfxgraph_core` |
| `_pyo3` | Python bindings (PyO3) | `rs_gfxgraph_pyo3` |
| `_logly` | Deep observability (5-level) | `rs_gfxgraph_logly` |
| `_node` | Node.js bindings (napi) | Reserved |

### Architectural Analysis

1. **`rs_gfxgraph_core` (This Crate — Core Contract Layer)**:
   - **Characteristics**: Extremely lightweight, pure Rust, minimal external dependencies (only `serde`), and near-zero compile time.
   - **Role**: Defines the core schema models (`GfxGraphNodeSpec`), telemetry storage contracts (`GfxGraphStatsSample`), and routing enums (`GfxGraphAdapterKind`). A **pure contract layer**.
   - **Modularity Rationale**: Keeping this crate pure ensures that downstream Rust systems (database interfaces, CLI parsers, metadata pipelines) can serialize, deserialize, and reference these types without compiling heavy FFI or deep-learning runtimes.

2. **`rs_gfxgraph_pyo3` (Native PyO3 Execution Crate)**:
   - **Characteristics**: Coupled tightly to the Python interpreter via `PyO3`.
   - **Role**: Contains the high-performance conditional graph runner and bucket router for deep-learning inference workloads.
   - **Why Separate**: Merging with core would destroy the lightweight contract nature, introducing complex PyO3 and native library linkage.

3. **`rs_gfxgraph_logly` (Deep Observability)**:
   - **Characteristics**: Self-sufficient 5-level observability with pluggable sinks.
   - **Role**: Collects live execution statistics, provides benchmark infrastructure, and integrates with `rs_logly_logger` as a drop-in.
   - **Why Separate**: Decoupled to keep execution telemetry completely separate from pure schema contracts.

---

## Features

- **Schema Contracts**: `GfxGraphNodeSpec` — graph node registry specifications.
- **Observability Models**: `GfxGraphStatsSample` — bucket performance telemetry.
- **Unified Error Handling**: `GfxGraphError` with engine-local stderr fallback.

## Error Reporting

```rust
use rs_gfxgraph_core::error::{GfxGraphError, report_error};

let err = GfxGraphError::InvalidNode {
    name: "flash_attention_decode".to_string(),
    reason: "impl_path does not exist".to_string(),
};
report_error(&err, "GFX_NODE_VALIDATION");
```

---

Last Updated: 2026-05-20
