---
name: hipvs-ann
description: Selects, builds, benchmarks, and validates hipVS ANN indexes and query paths for ROCm-DS workloads.
tools:
  - bash
  - python
  - git
inputs:
  - dataset_shape
  - vector_dim
  - metric
  - recall_target
  - latency_target
  - gpu_facts
outputs:
  - index_recommendation
  - benchmark_summary
  - validation_notes
tags:
  - rocm
  - rocm-ds
  - hipvs
  - ann
  - vector-search
---

# Purpose

Choose the correct hipVS path for vector search workloads and validate that the selected ANN method is appropriate for the data, constraints, and hardware.

# When to use

- Build or tune ANN/vector-search workflows on ROCm-DS
- Compare brute force, HNSW, IVF-Flat, IVF-PQ, CAGRA, or NN-Descent
- Validate query latency, throughput, and recall tradeoffs
- Design a GPU-native retrieval subsystem

# When not to use

- The workload is ordinary tabular ETL with no vector retrieval requirement
- The target GPU is outside supported or realistically testable configurations and no experimental mode is explicitly allowed
- There is no benchmark dataset or success criterion

# Required inputs

- Repository path
- Dataset shape and vector dimensionality
- Distance metric
- Recall / latency / throughput targets
- GPU and ROCm environment facts
- Whether source builds are allowed

# Procedure

1. Check compatibility assumptions first.
2. Classify the workload:
   - exact k-NN
   - high-recall ANN
   - memory-constrained ANN
   - high-throughput batch query
3. Pick candidate index families.
4. Build the minimal reproducible benchmark.
5. Compare build time, memory use, query latency, and recall.
6. Recommend the least-complex path that satisfies the target.
7. Document unsupported assumptions and fallback options.

# Deliverables

- Recommended index type and rationale
- Build and runtime notes
- Benchmark summary
- Validation evidence
- Risks and fallback path
