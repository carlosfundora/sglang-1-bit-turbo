---
name: router
description: Classifies and routes tasks to the appropriate ROCm-DS component (hipDF, hipVS, hipGRAPH, hipRAFT, hipMM) based on workload requirements.
tools:
  - bash
  - python
inputs:
  - task_description
  - source_files
outputs:
  - selected_component
  - routing_rationale
tags:
  - rocm
  - rocm-ds
  - routing
---

# Purpose

Act as the traffic cop to decide which ROCm-DS component applies to a given data science task.

# When to use

- When a new ROCm-DS task is received and the specific component is unknown.
- To map a generic request (e.g., "port pandas", "vector search") to a specific subsystem.

# Procedure

1. Read the task description and inspect source files if available.
2. Match keywords and operations to ROCm-DS components:
   - Tabular ETL / pandas-like ops -> `hipdf-pandas-port`
   - Vector search / embeddings / ANN -> `hipvs-ann`
   - Community detection / PageRank / graph ops -> `hipgraph-analytics`
   - Reusable kernels / clustering / low-level math -> `hipraft-primitives`
   - Memory management / OOM issues -> `hipmm-memory-ops`
3. Document the routing decision and rationale.
