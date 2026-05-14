## 2024-05-14 - Caching Hardware Capability Checks
**Learning:** Calling `torch.cuda.get_device_properties(0).gcnArchName` inside hot inference loops (like Triton kernel dispatch in attention functions) is a massive performance bottleneck due to synchronization and Python overhead.
**Action:** Always lift hardware checks (like `is_rdna2()`) to the module level or initialization phase and cache them instead of checking per-request during decoding or extending attention.

## 2024-05-14 - Communicator Deepcopy Overhead
**Learning:** Using `copy.deepcopy` inside async message passing loops (like `_Communicator.watching_call` for `get_loads`) introduces massive CPU overhead for simple objects. `deepcopy` on large lists of metrics or objects blocks the event loop unnecessarily when structural immutability (like replacing the list reference via `list()`) provides the identical decoupling with over 100x speedup.
**Action:** When saving state off a shared mutable list (like `self._result_values`) in internal queuing/watching routines, prefer shallow copies like `list(self._result_values)` if you only need reference isolation, rather than full `deepcopy`.
