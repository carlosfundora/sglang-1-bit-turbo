---
name: install-build
description: Handles installation and source compilation of ROCm-DS components via conda, PyPI, or source builds.
tools:
  - bash
  - python
  - git
inputs:
  - component
  - environment_facts
  - build_flags
outputs:
  - installation_status
  - build_logs
tags:
  - rocm
  - rocm-ds
  - build
  - install
---

# Purpose

Execute end-user installations (via conda/PyPI) or from-source builds for ROCm-DS components, handling architecture-specific flags.

# When to use

- When a component is missing and needs to be installed.
- When an early-access component (like hipGRAPH) requires a source build.
- To set up a reproducible environment.

# Procedure

1. Determine if an end-user package (conda/pip) is available for the component.
2. If available, construct and execute the installation command.
3. If source-build is required, clone the repository, apply necessary architecture flags (e.g., `--gpu-arch`), and compile.
4. Run basic smoke tests to verify the installation.
5. Record the build log and installation status.
