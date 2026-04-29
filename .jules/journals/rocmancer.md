
## 2024-05-24 - RDNA2 Triton Kernel and Memory Defaults Optimization
**Learning:**
Triton attention kernels were universally applying CDNA-specific optimizations (`waves_per_eu=1`, `matrix_instr_nonkdim=16`, `kpack=2`) on all HIP platforms. For RDNA2 (gfx1030) GPUs, which use Wave32 execution and lack matrix cores, these configurations are highly suboptimal and sometimes lead to crashes or severe throughput degradation. Additionally, default RDNA2 memory constraints (`mem_fraction_static`, `chunked_prefill_size`) weren't successfully overriding base defaults when missing in `server_args.py`.

**Action:**
1. Dynamically check `torch.cuda.get_device_properties(0).gcnArchName` in `extend_attention`, `double_sparsity_attention`, `prefill_attention`, and `rocm_mla_decode_rope` before kernel launches.
2. If `gfx103` is detected, remove CDNA-specific `extra_kargs` and tune `num_warps` (typically 2 for Wave32) and `waves_per_eu` appropriately.
3. Update `server_args.py` to properly propagate `mem_fraction_static` and `chunked_prefill_size` from `gfx1031_defaults` when initializing an RDNA2 server.

## 2024-05-24 - CI Docs Advisory Failure Note
**Learning:**
SGLang's CI requires a `CHANGELOG.md` update for PRs that modify code files. When the `CHANGELOG.md` is updated, the changes must include a `Documentation` or `Docs` section outlining modified documentation, or explicitly stating `- None required`. The recent `docs-policy` check failed because no `CHANGELOG.md` was updated in the PR.

**Action:**
Update the `CHANGELOG.md` file under the current unreleased version with a note about the AMD RDNA2 optimizations and the required `- None required` documentation block.

## 2024-05-24 - CI Dependency Failure
**Learning:**
The CI job `stage-a-test-cpu` failed with `ModuleNotFoundError: No module named 'tabulate'`. This appears to be a missing dependency in the CI test runner (`test/run_suite.py`) rather than a direct consequence of the RDNA2 Triton kernel optimizations I introduced. The script `test/run_suite.py` likely expects `tabulate` to be installed but it was missing from the `uv pip install -e "python[dev]"` environment resolving phase.

**Action:**
Add `tabulate` to the development dependencies list (either `requirements-test.txt` or within `pyproject.toml`'s dev dependencies section depending on project structure) to fix the `stage-a-test-cpu` job.

## 2024-05-24 - CI Docs Advisory Failure Update
**Learning:**
I updated `CHANGELOG.md` correctly but there were multiple undocumented file modifications (`test/srt/test_p_eagle_wiring.py` and `python/pyproject.toml`) that the docs policy script considers missing from the changelog. The CI script failed because we didn't specify `- None required`. Wait, actually I did add `- None required`, but I used `- None required` without a period, or it was in the wrong section, or it detected new journal files created by my prompt logic (`.jules/reports/research/*`) that were not tracked. The exact error is:
`- Changed documentation file not listed in CHANGELOG.md: .jules/journals/triangulator-forge.md`
`- Changed documentation file not listed in CHANGELOG.md: .jules/reports/research/repo-triangulation-20260427-024844.md`
`- Changed documentation file not listed in CHANGELOG.md: .jules/reports/research/repo-triangulation-20260427-120000.md`

**Action:**
Update `CHANGELOG.md` to properly document the new `.jules/reports/*` files, since any markdown file modification triggers the policy checker.

## 2024-05-24 - Docs Policy CI Refinement
**Learning:**
I previously encountered issues with the docs-policy CI job failing because it detected markdown file modifications inside `.jules/`. I thought I had solved this by resetting the `.jules/` directory from `git`. However, the CI output still reports `- Changed documentation file not listed in CHANGELOG.md: .jules/journals/rocmancer.md.`
This happened because `git reset HEAD .jules/` removes the files from the index (unstages them), but they might still be untracked or the commit itself still contains the previously committed versions of `.jules` files if I only did `git commit --amend` without actually dropping them from the commit tree.

**Action:**
I need to definitively purge the `.jules/` files from the *branch history* (or the last commit) so they don't show up in the `origin/main...HEAD` diff. I will use `git rm --cached -r .jules/` and amend the commit, ensuring they are ignored.
