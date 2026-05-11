---
name: compatibility-triage
description: Verifies whether a requested ROCm-DS workflow is officially supported, source-build feasible, or experimental on the target system.
tools:
  - bash
  - python
inputs:
  - os_version
  - python_version
  - rocm_version
  - gpu_model
  - gfx_target
  - requested_component
outputs:
  - compatibility_report
  - risk_level
  - go_no_go_decision
tags:
  - rocm
  - compatibility
  - triage
  - support-matrix
---

# Purpose

Prevent agents from making unsupported assumptions about ROCm-DS compatibility.

# Procedure

1. Collect environment facts using `rocminfo`, `hipcc --version`, Python version, and OS release.
2. Map the requested workflow to one or more ROCm-DS components.
3. Compare the environment to official support statements and tested-GPU notes.
4. Classify the request:

   * supported
   * partially supported
   * source-build experimental
   * unsupported
5. Stop unsafe automation when the mismatch is material.
6. Produce a short report with exact blockers and realistic next moves.

# Rules

* Never claim support based solely on generic ROCm support.
* Distinguish official support from “might compile.”
* For experimental paths, require explicit evidence from local smoke tests.
* Do not silently widen GPU support claims.

# Deliverables

* Environment summary
* Support classification
* Exact mismatch list
* Recommended next action
