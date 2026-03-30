# Abort Request Cleanup (`rid_to_state` leak in `_handle_abort_req`)

## Current Issue

`_handle_abort_req` in `tokenizer_manager.py` does not perform `del rid_to_state` or LoRA release. The design table at the bottom of the file (line 2494-2505) specifies that validation/waiting-queue aborts should have `del in _handle_abort_req`, but the implementation defers cleanup to `_wait_one_response`, which only runs it when `status_code` matches a hardcoded set (`SERVICE_UNAVAILABLE`, `INTERNAL_SERVER_ERROR`). When `status_code` is `None` or a custom value, cleanup is skipped entirely -- causing a slow `rid_to_state` memory leak and LoRA resource leak.

## Key Files

- `python/sglang/srt/managers/tokenizer_manager.py` -- abort consumer (`_handle_abort_req`, `_wait_one_response`, `_handle_batch_token_id_output`)
- `python/sglang/srt/managers/scheduler.py` -- abort producer (3 abort methods + scheduler-initiated aborts)
- `python/sglang/srt/managers/schedule_batch.py` -- `FINISH_ABORT` class, `set_finish_with_abort()`
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` -- forward timeout abort
- `python/sglang/srt/managers/io_struct.py` -- `AbortReq` dataclass

## Background

### Scheduler abort methods

The scheduler has 3 abort methods that determine how the abort signal reaches `TokenizerManager`:

- **Method 1** (`scheduler.py` line 2760): Pop from waiting queue, send `AbortReq(rid=...)` back. Request never did a forward pass. Reaches TM via `_handle_abort_req` (Path B).
- **Method 2** (`scheduler.py` line 2785): `set_finish_with_abort()` -- sets `origin_input_ids=[0]`, `to_finish=FINISH_ABORT(msg, BAD_REQUEST)`. Runs a single-token prefill, then `stream_output`. Reaches TM via `_handle_batch_token_id_output` (Path A).
- **Method 3** (`scheduler.py` line 2844): `req.to_finish = FINISH_ABORT()`. Request is in the running batch, runs one more decode forward, then `stream_output`. Reaches TM via `_handle_batch_token_id_output` (Path A).

Additionally, the scheduler **proactively** sends `AbortReq(finished_reason={...})` for: priority disabled (line 1663, `SERVICE_UNAVAILABLE`), queue full/preempt (line 1711, env var default 503), queue timeout (line 1737, env var default 503). These reach TM via `_handle_abort_req` (Path B).

### Two data paths and cleanup ownership

`AbortReq` is bidirectional: TM→Scheduler means "please abort", Scheduler→TM means "already aborted, clean up".

- **Path A** (`stream_output` → `_handle_batch_token_id_output`, Method 2+3): Cleanup is done at line 1609 (`del rid_to_state` + LoRA release) **before** `_wait_one_response` picks up the output. `status_code=None` fall-through in `_wait_one_response` is harmless.
- **Path B** (`AbortReq` → `_handle_abort_req`, Method 1 + scheduler-initiated): **No cleanup is performed.** `_handle_abort_req` just sets `state.finished=True`, constructs output, and calls `state.event.set()`. Cleanup depends entirely on `_wait_one_response`'s hardcoded status code checks.

## Principles

- Cleanup (`del rid_to_state` + LoRA release) must happen at the **data arrival point** -- `_handle_batch_token_id_output` for Path A, `_handle_abort_req` for Path B. This makes the two paths symmetric.
- `_wait_one_response` should only decide **response behavior** (raise error vs yield), never perform resource cleanup.
- Moving cleanup to `_handle_abort_req` also fixes the **client disconnect + waiting queue** case, where `_wait_one_response` generator has already exited via `raise ValueError` and nobody consumes the abort output.
- The `status_code` check in `_wait_one_response` should use `status_code is not None` catch-all instead of a hardcoded tuple, so custom env var values (e.g., 429) are never missed.

## Affected Scenarios

### Category A: Via `stream_output` -- no leak

| Source | Method | `status_code` |
|---|---|---|
| User abort running request | Method 3: `to_finish = FINISH_ABORT()` | `None` |
| Session cleanup | Method 3: `to_finish = FINISH_ABORT()` | `None` |
| Grammar `accept_token` failure | `abort_request` → Method 3 | `None` |
| Forward timeout | `to_finish = FINISH_ABORT(msg, SERVICE_UNAVAILABLE)` | `SERVICE_UNAVAILABLE` |
| Grammar/input validation | Method 2: `set_finish_with_abort` | `BAD_REQUEST` |
| MM receiver error | `prepare_abort()` | `BAD_REQUEST` / `INTERNAL_SERVER_ERROR` |

### Category B: Via `_handle_abort_req` -- leak when `status_code` mismatches

| Source | `status_code` | Result |
|---|---|---|
| Priority disabled | `SERVICE_UNAVAILABLE` | OK (matches) |
| Queue full / preempt | env var (default 503) | **Leak if custom non-503/500 value** |
| Queue timeout | env var (default 503) | **Leak if custom non-503/500 value** |
| User abort waiting queue (`/abort_request` API) | None (missing) | **Leak** |
| Client disconnect + waiting queue | None (missing) | **Leak + generator already exited** |

## TODOs

1. **Add cleanup to `_handle_abort_req`** (`tokenizer_manager.py` line 2156): Insert `del self.rid_to_state[recv_obj.rid]` and `asyncio.create_task(self.lora_registry.release(...))` immediately after `state.finished_time = time.time()`, before constructing the output. This aligns with the design table and makes Path B symmetric with Path A. Safety: subsequent `abort_request(rid)` calls will early-return at the `if rid not in self.rid_to_state: return` guard; the existing `del` in `_wait_one_response` becomes a no-op.

2. **Simplify `_wait_one_response` abort handling** (`tokenizer_manager.py` line 1153-1188): Remove the `del self.rid_to_state` and `lora_registry.release()` calls (now redundant). Replace the hardcoded `(SERVICE_UNAVAILABLE, INTERNAL_SERVER_ERROR)` tuple with `status_code is not None` catch-all. Keep `BAD_REQUEST` as a separate branch (raises `ValueError` not `HTTPException`). `status_code=None` (user-initiated abort) should yield normally without raising.

## Appendix: All Abort Source Reference

### `FINISH_ABORT` creation sites (Path A)

| Location | `status_code` |
|---|---|
| `schedule_batch.py` `set_finish_with_abort()` | `BAD_REQUEST` |
| `schedule_batch.py` `Req.__init__` error | `BAD_REQUEST` |
| `scheduler_output_processor_mixin.py` forward timeout | `SERVICE_UNAVAILABLE` |
| `scheduler.py` abort running request (Method 3) | `None` |
| `session_controller.py` cleanup/abort | `None` |
| `disaggregation/utils.py` `prepare_abort()` | `BAD_REQUEST` or `INTERNAL_SERVER_ERROR` |

### `AbortReq` sent from scheduler → tokenizer_manager (Path B)

| Location | Has `finished_reason`? | `status_code` |
|---|---|---|
| `scheduler.py` line 2767 (Method 1: waiting queue pop) | No | None |
| `scheduler.py` line 1663 (priority disabled) | Yes | `SERVICE_UNAVAILABLE` |
| `scheduler.py` line 1711 (queue full / preempt) | Yes | env var default 503 |
| `scheduler.py` line 1737 (queue timeout) | Yes | env var default 503 |
| `scheduler.py` line 2827 (disagg retract) | No | None |
