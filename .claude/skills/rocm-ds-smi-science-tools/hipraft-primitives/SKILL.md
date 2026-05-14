---
name: hipraft-primitives
description: Utilizes hipRAFT for foundational, reusable GPU-accelerated primitives like clustering, dimensionality reduction, and statistical operations.
tools:
  - bash
  - python
inputs:
  - mathematical_target
outputs:
  - computed_results
  - primitive_report
tags:
  - rocm
  - rocm-ds
  - hipraft
  - math
---

# Purpose

Deploy hipRAFT as the computational backbone for higher-level data science and AI applications.

# When to use

- When you need a foundational layer of reusable GPU-accelerated primitives.
- For tasks involving core machine learning math: clustering, dimensionality reduction, and statistical operations.
- When building new algorithms that require low-level computational building blocks on AMD GPUs.

# Procedure

1. Identify the core mathematical operations required for the workload.
2. Select the appropriate hipRAFT primitive (e.g., clustering algorithms, solvers, linear algebra ops).
3. Integrate the primitive into the broader pipeline (often serving as the backbone for custom tools or hipVS integrations).
4. Verify correctness against standard CPU implementations.
