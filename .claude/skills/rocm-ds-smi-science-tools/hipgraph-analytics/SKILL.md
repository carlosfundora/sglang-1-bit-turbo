---
name: hipgraph-analytics
description: Leverages GPU acceleration to process and analyze complex graph structures using hipGRAPH. Note: hipGRAPH is early access.
tools:
  - bash
  - python
inputs:
  - graph_data
  - algorithm_target
outputs:
  - analytics_report
  - computed_metrics
tags:
  - rocm
  - rocm-ds
  - hipgraph
  - graph-analytics
---

# Purpose

Utilize hipGRAPH for high-performance graph processing and analysis, while acknowledging its early-access status.

# When to use

- When a data science task involves complex network or graph structures.
- To execute diverse graph algorithms such as centrality, traversal, similarity, sampling, and labeling.
- When graph outputs need to integrate seamlessly with hipDF DataFrames across the ROCm-DS ecosystem.

# When not to use

- **Production Environments:** The hipGRAPH libraries are currently in an **early access state**. Running production workloads with these libraries is explicitly not recommended by AMD.
- If the workload is purely tabular (use `hipdf-pandas-port`) or purely vector search (use `hipvs-ann`).

# Procedure

1. Verify that the requested workload is a non-production, experimental, or benchmark task.
2. Formulate the graph using hipGRAPH APIs, loading data (potentially from hipDF).
3. Execute the target algorithm (e.g., centrality, traversal, PageRank, BFS, node2vec).
4. Analyze the results or pipe them back into hipDF.
5. Report on the execution, highlighting any early-access instability or API quirks.
