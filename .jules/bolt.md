## 2024-05-14 - Caching Hardware Capability Checks
**Learning:** Calling `torch.cuda.get_device_properties(0).gcnArchName` inside hot inference loops (like Triton kernel dispatch in attention functions) is a massive performance bottleneck due to synchronization and Python overhead.
**Action:** Always lift hardware checks (like `is_rdna2()`) to the module level or initialization phase and cache them instead of checking per-request during decoding or extending attention.

## 2024-05-14 - Communicator Deepcopy Overhead
**Learning:** Using `copy.deepcopy` inside async message passing loops (like `_Communicator.watching_call` for `get_loads`) introduces massive CPU overhead for simple objects. `deepcopy` on large lists of metrics or objects blocks the event loop unnecessarily when structural immutability (like replacing the list reference via `list()`) provides the identical decoupling with over 100x speedup.
**Action:** When saving state off a shared mutable list (like `self._result_values`) in internal queuing/watching routines, prefer shallow copies like `list(self._result_values)` if you only need reference isolation, rather than full `deepcopy`.

## 2024-05-14 - Batch Merging Invariants and Early Returns
**Learning:** In complex batch managers like `ScheduleBatch`, an early return from `filter_batch` (e.g. when `keep_indices` is empty) can leave residual, stale tensor data even if the request list (`self.reqs`) is cleared. If `merge_batch` checks only `other.is_empty()` but NOT `self.is_empty()`, the merge will append `other`'s valid tensors onto `self`'s residual tensors, causing massive size mismatches and corruption in subsequent steps.
**Action:** Always guard state-mutating batch merge operations (like `ScheduleBatch.merge_batch` and `SamplingBatchInfo.merge_batch`) with an explicit `is_empty()` check on `self` to prevent appending valid incoming data to residual/stale internal states.
## 2024-05-15 - Triton Kernel Prefix Sum Optimization
**Learning:** Using an O(N) loop to compute a prefix sum per thread (e.g., `for i in range(pid): cumsum_start += tl.load(extend_lens + i)`) inside a Triton kernel creates an O(N²) anti-pattern that becomes painfully slow at large batch sizes.
**Action:** Optimize this by precomputing the prefix sum on the Python side using PyTorch (`cumsum_extend_lens = extend_lens.cumsum(dim=0, dtype=torch.int64)`) and passing it to the kernel for an O(1) lookup per thread (`cumsum_start = tl.load(cumsum_extend_lens + pid - 1)`).
## 2026-05-21 - [Replace copy.deepcopy with list() in hot loop]
**Learning:** In highly asynchronous or batched contexts (like tokenizer management or IO handling), `copy.deepcopy()` is incredibly slow for copying simple primitive structures like lists of integers. It adds huge per-request processing time due to Python's internal tracking mechanisms for circular references.
**Action:** Use shallow copy methods like `list(x)`, `x.copy()`, or `x[:]` when dealing with primitive 1D list structures rather than blindly utilizing deepcopy.
## 2026-05-21 - [Deepcopy in filter_batch]
**Learning:** The `SamplingBatchInfo.filter_batch()` in speculative decoding components called `copy.deepcopy()` to maintain references. This needlessly invoked Pytorch's expensive tensor deepcopy allocations for elements that are fully replaced within `filter_batch` anyways.
**Action:** Implemented a robust `clone()` method utilizing `dataclasses.replace` for shallow tensor copying and precise `.copy()`/`deepcopy` instructions for specific mutating inner list/dict configurations, avoiding CUDA synchronizations and CPU-bottlenecks.
## 2026-05-22 - [Avoid copy.deepcopy in SamplingBatchInfo]
**Learning:** `copy.deepcopy(sampling_info)` in `python/sglang/srt/speculative/eagle_info.py` and `python/sglang/srt/speculative/ngram_info.py` introduces significant CPU overhead in the hot loop of speculative decoding because `SamplingBatchInfo` contains many primitive lists and tensors. Deepcopy takes excessive time verifying cycle references across PyTorch structures.
**Action:** Replaced `copy.deepcopy` with a bespoke `.clone()` method on `SamplingBatchInfo` that leverages `dataclasses.replace` along with shallow copying mutable sub-structures (e.g., `list()`, `dict()`) and recursively cloning `BatchedPenalizerOrchestrator`. This provides massive speedups for speculative decode inference while maintaining full safety.

## 2026-05-23 - [Shadowing Methods with Unoptimized Implementations]
**Learning:** In Python codebases with complex merge histories, duplicate method definitions within the same class can silently degrade performance. Because Python evaluates class bodies sequentially, a less-optimized duplicate method at the bottom of a file will silently override a highly optimized version declared earlier. In `SamplingBatchInfo`, an optimized `.clone()` method was overridden by a legacy method using slow `copy.deepcopy()`.
**Action:** Always verify that optimized methods are not accidentally shadowed by duplicate declarations lower down in the same class file.
