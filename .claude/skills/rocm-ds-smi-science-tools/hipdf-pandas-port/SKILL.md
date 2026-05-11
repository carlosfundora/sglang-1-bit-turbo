---
name: hipdf-pandas-port
description: Migrates pandas-like workflows to hipDF and cudf.pandas style acceleration, auditing for unsupported features.
tools:
  - bash
  - python
inputs:
  - source_scripts
outputs:
  - porting_report
  - migrated_scripts
tags:
  - rocm
  - rocm-ds
  - hipdf
  - pandas
---

# Purpose

Accelerate tabular workflows by porting pandas code to hipDF.

# When to use

- When a task involves slow pandas ETL, joins, or aggregations.
- To implement GPU DataFrames for performance gains.

# Procedure

1. Audit existing pandas usage in the provided scripts.
2. Identify unsupported features in hipDF compared to pandas.
3. Rewrite compatible sections using hipDF or cudf.pandas style acceleration.
4. Run parity tests to ensure the output matches the original pandas logic.
5. Report on performance gains and any remaining pandas fallback operations.
