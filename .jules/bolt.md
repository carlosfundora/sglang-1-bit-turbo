## 2024-05-14 - Caching Hardware Capability Checks
**Learning:** Calling `torch.cuda.get_device_properties(0).gcnArchName` inside hot inference loops (like Triton kernel dispatch in attention functions) is a massive performance bottleneck due to synchronization and Python overhead.
**Action:** Always lift hardware checks (like `is_rdna2()`) to the module level or initialization phase and cache them instead of checking per-request during decoding or extending attention.

## 2024-05-14 - Communicator Deepcopy Overhead
**Learning:** Using `copy.deepcopy` inside async message passing loops (like `_Communicator.watching_call` for `get_loads`) introduces massive CPU overhead for simple objects. `deepcopy` on large lists of metrics or objects blocks the event loop unnecessarily when structural immutability (like replacing the list reference via `list()`) provides the identical decoupling with over 100x speedup.
**Action:** When saving state off a shared mutable list (like `self._result_values`) in internal queuing/watching routines, prefer shallow copies like `list(self._result_values)` if you only need reference isolation, rather than full `deepcopy`.

## 2024-05-14 - Batch Merging Invariants and Early Returns
**Learning:** In complex batch managers like `ScheduleBatch`, an early return from `filter_batch` (e.g. when `keep_indices` is empty) can leave residual, stale tensor data even if the request list (`self.reqs`) is cleared. If `merge_batch` checks only `other.is_empty()` but NOT `self.is_empty()`, the merge will append `other`'s valid tensors onto `self`'s residual tensors, causing massive size mismatches and corruption in subsequent steps.
**Action:** Always guard state-mutating batch merge operations (like `ScheduleBatch.merge_batch` and `SamplingBatchInfo.merge_batch`) with an explicit `is_empty()` check on `self` to prevent appending valid incoming data to residual/stale internal states.
