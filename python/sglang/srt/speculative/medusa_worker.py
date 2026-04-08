"""Medusa multi-head speculative decoding worker.

Generates K draft tokens in parallel using lightweight MLP heads on the target
model's last hidden state.  No autoregressive loop — all drafts from one pass.
Verification reuses the NgramVerifyInput tree infrastructure (same tree attention
as EAGLE3/NGRAM).

Includes DraftPreFilter integration: layered pre-rejection of draft tokens
before verification (n-gram surprisal → screen head inversion → head agreement).

Worker pattern: coordinator (like NGRAMWorker / PCascadeWorker).
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class _MedusaCaptureCarrier:
    """Minimal spec_info stand-in that requests hidden-state capture.

    The scheduler may call filter_batch / merge_batch on spec_info.  This
    carrier implements them as no-ops so it doesn't break the scheduler loop.
    It is set only during the forward call and cleared immediately after.
    """

    capture_hidden_mode = CaptureHiddenMode.LAST

    def filter_batch(self, **kwargs):
        pass

    def merge_batch(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Medusa tree choices (predefined candidate combinations from the paper)
# ---------------------------------------------------------------------------

# Simple linear chain: head-0 → head-1 → … → head-K
# Each entry is a "path" — len(path) = depth, path[-1] = top-k index
def _make_linear_choices(n_heads: int) -> List[List[int]]:
    return [[0] * (i + 1) for i in range(n_heads)]


# Standard Medusa mc_sim_7b_63 tree (63 candidates, 5 heads)
MC_SIM_63: List[List[int]] = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2],
    [3], [0, 0, 1], [1, 1], [0, 3], [4], [0, 0, 2], [1, 2],
    [2, 0], [0, 0, 3], [0, 0, 0, 0], [1, 0, 0], [0, 4],
    [1, 3], [2, 1], [3, 0], [0, 0, 4], [0, 0, 0, 1],
    [0, 1, 0], [1, 0, 1], [0, 0, 0, 2], [4, 0], [2, 2],
    [0, 1, 1], [3, 1], [1, 1, 0], [0, 2, 0], [1, 4],
    [0, 0, 0, 3], [2, 0, 0], [0, 3, 0], [0, 1, 2], [1, 0, 2],
    [0, 0, 1, 0], [3, 2], [2, 3], [4, 1], [1, 2, 0],
    [0, 0, 0, 4], [0, 2, 1], [1, 0, 0, 0], [0, 0, 0, 0, 0],
    [1, 1, 1], [0, 0, 2, 0], [0, 0, 1, 1], [2, 0, 1],
    [0, 1, 0, 0], [0, 4, 0], [3, 0, 0], [0, 1, 3],
    [1, 3, 0], [0, 0, 3, 0], [2, 1, 0], [4, 0, 0],
    [1, 0, 0, 1], [0, 3, 1],
]


def _build_linear_tree_mask(tree_size: int) -> np.ndarray:
    """Lower-triangular mask for a linear draft chain (current + K drafts).

    The NgramVerifyInput infrastructure expects the first token in the tree
    to be the current (uncommitted) token from output_ids[-1].  Subsequent
    tokens are the actual draft predictions.  This matches the C++ ngram
    corpus behaviour (result.cpp:fillResult prepends ``last_token``).

    Returns:
        (tree_size, tree_size) bool numpy array — lower-triangular.
    """
    return np.tril(np.ones((tree_size, tree_size), dtype=bool))


# ---------------------------------------------------------------------------
# MedusaWorker
# ---------------------------------------------------------------------------


class MedusaWorker:
    """Coordinator that runs Medusa heads + NgramVerifyInput tree verification.

    The worker follows the same interface the scheduler expects (model_runner,
    model_config, forward_batch_generation, clear_cache_pool, …).

    Hidden-state capture:
        Medusa requires the target model's last hidden state.  After each
        extend or verification forward, we attempt to read it from
        ``model_runner.last_hidden_states``.  If unavailable, Medusa falls
        back to target-only decoding (no drafting) with a warning.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.target_worker = target_worker
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        # Medusa-specific config
        self.num_heads = getattr(server_args, "medusa_num_heads", 5)
        self.medusa_topk = getattr(server_args, "medusa_topk", 1)
        draft_tokens = server_args.speculative_num_draft_tokens or 8

        # Typical acceptance (entropy-adaptive candidate generation)
        self.typical_acceptance = getattr(server_args, "medusa_typical_acceptance", False)
        self.posterior_threshold = getattr(server_args, "medusa_posterior_threshold", 0.09)
        self.posterior_alpha = getattr(server_args, "medusa_posterior_alpha", 0.3)
        self.tree_structure = getattr(server_args, "medusa_tree_structure", "linear")
        self.no_step0 = getattr(server_args, "medusa_no_step0", False)

        # AMD SAM / ReBAR / PALTROW hardware detection
        self._sam_info = self._detect_sam_rebar(server_args)

        # Load the model
        model_path = getattr(server_args, "medusa_model_path", None)
        if model_path is None:
            raise ValueError(
                "Medusa requires --medusa-model-path pointing to trained heads."
            )

        from sglang.srt.speculative.medusa_model import MedusaModel

        # Parse PALTROW head indices from server args or auto-detect from config
        paltrow_arg = getattr(server_args, "medusa_paltrow_heads", None)
        paltrow_head_indices = None  # None = auto-detect from tiered config
        if paltrow_arg is not None:
            if paltrow_arg == "none":
                paltrow_head_indices = []  # force all GPU
            elif paltrow_arg == "auto":
                paltrow_head_indices = None  # auto-detect screen+bloom heads
            else:
                paltrow_head_indices = [int(x) for x in paltrow_arg.split(",") if x.strip()]

        self.medusa_model = MedusaModel.from_pretrained(
            model_path,
            device=self.device,
            dtype=target_worker.model_runner.model_config.dtype,
            paltrow_head_indices=paltrow_head_indices,
        )
        logger.info(
            "MedusaWorker: loaded %d heads from %s", self.medusa_model.num_heads, model_path
        )

        # Detect tiered architecture: build head reorder map that skips screen
        # heads and orders remaining heads by their prediction offset (t+1, t+2, ...)
        self._draft_head_indices = self._build_draft_head_order(model_path)

        # --medusa-no-step0: skip the t+1 head (it echoes with stale hidden).
        # Head t+2 becomes draft[0], t+3 becomes draft[1], etc.
        if self.no_step0:
            if len(self._draft_head_indices) > 1:
                self._draft_head_indices = self._draft_head_indices[1:]
                logger.info(
                    "MedusaWorker: no-step0 mode — shifted heads, "
                    "skipped t+1 echo head, using %s",
                    self._draft_head_indices,
                )
            else:
                logger.warning(
                    "MedusaWorker: no-step0 requires ≥2 non-screen heads; "
                    "falling back to step0 mode"
                )
                self.no_step0 = False

        num_draft_heads = len(self._draft_head_indices)
        self.num_draft_tokens = min(draft_tokens, num_draft_heads)  # K = pure drafts

        # tree_token_num = K + 1: the NgramVerifyInput tree must have the
        # current (uncommitted) token at position 0, followed by K drafts.
        # This matches the invariant from the C++ ngram corpus (result.cpp:15).
        self.draft_token_num = self.num_draft_tokens + 1  # tree size for verify infra
        logger.info(
            "MedusaWorker: draft head order = %s (skipped screen), K=%d, tree_size=%d",
            self._draft_head_indices[:self.num_draft_tokens],
            self.num_draft_tokens,
            self.draft_token_num,
        )

        # Use linear chain tree (current + K drafts)
        self.tree_mask_np = _build_linear_tree_mask(self.draft_token_num)

        # Pre-allocated tensors + cached tree buffers
        self.max_batch_size = target_worker.max_running_requests
        self._init_preallocated_tensors()
        self._init_cached_tree_buffers()

        # Hidden-state cache (populated after each target forward)
        self._cached_hidden: Optional[torch.Tensor] = None
        self._hidden_available = False
        self._last_accept_lens: Optional[torch.Tensor] = None

        # DraftPreFilter: layered pre-rejection (n-gram → screen → agreement)
        self.prefilter = self._init_prefilter(model_path)

        logger.info(
            "MedusaWorker ready: %d heads, %d draft tokens (tree size %d), "
            "tree=%s, typical=%s (τ=%.3f α=%.3f), prefilter=%s, SAM=%s, step0=%s",
            self.num_heads,
            self.num_draft_tokens,
            self.draft_token_num,
            self.tree_structure,
            self.typical_acceptance,
            self.posterior_threshold,
            self.posterior_alpha,
            "ON" if self.prefilter else "OFF",
            "ON" if self._sam_info.get("enabled") else "OFF",
            "OFF (shifted heads)" if self.no_step0 else "ON",
        )

    def _build_draft_head_order(self, model_path: str) -> list:
        """Build ordered list of head indices for drafting.

        For tiered models: skip screen heads, order remaining by prediction
        offset (t+1 first, then t+2, ...).  Pick the best head per offset
        (precision > easy when both cover the same offset).

        For flat models: identity order [0, 1, ..., num_heads-1].
        """
        config_path = os.path.join(model_path, "medusa_config.json")
        if not os.path.exists(config_path):
            return list(range(self.num_heads))

        with open(config_path) as f:
            cfg = json.load(f)

        tiered = cfg.get("tiered_architecture")
        if not tiered:
            return list(range(self.num_heads))

        screen_set = set(tiered.get("screen_heads", []))
        offsets = cfg.get("head_offsets", {})

        def _parse_offset(desc: str) -> int:
            if "screen" in desc.lower():
                return -1
            import re
            m = re.search(r"t\+(\d+)", desc)
            return int(m.group(1)) if m else 0

        candidates = []
        for idx in range(self.num_heads):
            if idx in screen_set:
                continue
            desc = offsets.get(str(idx), "")
            off = _parse_offset(desc)
            if off < 0:
                continue
            is_precision = "precision" in desc.lower()
            candidates.append((idx, off, 0 if is_precision else 1))

        candidates.sort(key=lambda x: (x[1], x[2]))

        seen_offsets = set()
        result = []
        for idx, off, _prio in candidates:
            if off in seen_offsets:
                continue
            seen_offsets.add(off)
            result.append(idx)

        if not result:
            logger.warning("No non-screen heads found, falling back to identity order")
            return list(range(self.num_heads))

        return result

    # ---- AMD SAM / ReBAR auto-detection ----

    @staticmethod
    def _detect_sam_rebar(server_args: ServerArgs) -> dict:
        """Detect AMD Smart Access Memory / Resizable BAR hardware status.

        Checks PCIe BAR size vs VRAM and GTT pool size from sysfs.
        SAM is active when BAR >= VRAM (full GPU memory visible to CPU).

        Returns:
            dict with keys: enabled, bar_mb, vram_mb, gtt_mb, bandwidth_note
        """
        info = {"enabled": False, "bar_mb": 0, "vram_mb": 0, "gtt_mb": 0, "bandwidth_note": ""}

        # Check explicit override
        sam_arg = getattr(server_args, "sam_enabled", None)
        if sam_arg is not None:
            if isinstance(sam_arg, str):
                if sam_arg.lower() == "false":
                    info["enabled"] = False
                    logger.info("SAM/ReBAR: disabled by --sam-enabled=false")
                    return info
                elif sam_arg.lower() == "true":
                    info["enabled"] = True
                    logger.info("SAM/ReBAR: force-enabled by --sam-enabled=true")
                    return info

        # Auto-detect from sysfs (AMD GPUs)
        try:
            import glob as globmod
            drm_dirs = sorted(globmod.glob("/sys/class/drm/card*/device"))
            for drm_dir in drm_dirs:
                # Check if this is an AMD GPU (amdgpu driver)
                driver_link = os.path.join(drm_dir, "driver")
                if os.path.islink(driver_link) and "amdgpu" in os.readlink(driver_link):
                    # Read VRAM size
                    vram_path = os.path.join(drm_dir, "mem_info_vram_total")
                    if os.path.exists(vram_path):
                        with open(vram_path) as f:
                            info["vram_mb"] = int(f.read().strip()) // (1024 * 1024)

                    # Read visible VRAM (BAR size) — SAM = visible == total
                    vis_path = os.path.join(drm_dir, "mem_info_vis_vram_total")
                    if os.path.exists(vis_path):
                        with open(vis_path) as f:
                            info["bar_mb"] = int(f.read().strip()) // (1024 * 1024)

                    # Read GTT pool size
                    gtt_path = os.path.join(drm_dir, "mem_info_gtt_total")
                    if os.path.exists(gtt_path):
                        with open(gtt_path) as f:
                            info["gtt_mb"] = int(f.read().strip()) // (1024 * 1024)

                    # Override GTT from CLI if provided
                    cli_gtt = getattr(server_args, "gtt_pool_mb", None)
                    if cli_gtt is not None:
                        info["gtt_mb"] = cli_gtt

                    # SAM is active when BAR >= VRAM
                    if info["bar_mb"] > 0 and info["vram_mb"] > 0:
                        info["enabled"] = info["bar_mb"] >= info["vram_mb"]

                    if info["enabled"]:
                        info["bandwidth_note"] = (
                            "PCIe DMA ~14 GB/s pinned, ~8 GB/s pageable"
                        )
                        logger.info(
                            "SAM/ReBAR ACTIVE: BAR=%d MB, VRAM=%d MB, GTT=%d MB. "
                            "PALTROW heads will use pinned DMA for optimal bandwidth.",
                            info["bar_mb"], info["vram_mb"], info["gtt_mb"],
                        )
                    else:
                        logger.info(
                            "SAM/ReBAR inactive: BAR=%d MB < VRAM=%d MB. "
                            "PALTROW heads use standard PCIe copies.",
                            info["bar_mb"], info["vram_mb"],
                        )
                    break  # Use first AMD GPU found
        except Exception as e:
            logger.debug("SAM/ReBAR detection failed: %s", e)

        return info

    # ---- Cached tree buffers (avoid per-call recomputation) ----

    def _init_cached_tree_buffers(self):
        """Pre-compute and cache tree mask buffers for the maximum batch size.

        Avoids repeated numpy→torch conversion and tiling in _forward_medusa().
        Buffers are sliced to actual batch size at call time.

        Cached:
          _cached_mask_per_req_flat: [T*T] numpy, flattened single-request mask
          _cached_mask_tiled_gpu: [max_bs * T * T] torch.bool on GPU, pre-tiled
          _cached_mask_np_3d: [max_bs, T, T] numpy, for attention mask construction
        """
        T = self.draft_token_num
        max_bs = self.max_batch_size

        # Flatten single-request mask
        self._cached_mask_per_req_flat = self.tree_mask_np.flatten()  # [T*T]

        # Pre-tile for max batch and convert to GPU tensor once
        mask_tiled_np = np.tile(self._cached_mask_per_req_flat, max_bs)  # [max_bs * T * T]
        self._cached_mask_tiled_gpu = torch.from_numpy(mask_tiled_np).to(
            device=self.device, dtype=torch.bool
        )

        # 3D view for attention mask construction
        self._cached_mask_np_3d = mask_tiled_np.reshape(max_bs, T, T)

        logger.debug(
            "Cached tree buffers: T=%d, max_bs=%d, GPU mask=%.1f KB",
            T, max_bs, self._cached_mask_tiled_gpu.numel() / 1024,
        )

    def _init_prefilter(self, model_path: str):
        """Initialize DraftPreFilter if tiered config or env var enables it."""
        from sglang.srt.speculative.draft_prefilter import DraftPreFilter

        # Check for tiered architecture config
        config_path = os.path.join(model_path, "medusa_config.json")
        screen_head_idx = None
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            tiered = cfg.get("tiered_architecture", {})
            screen_heads = tiered.get("screen_heads", [])
            if screen_heads:
                screen_head_idx = screen_heads[0]
                logger.info("DraftPreFilter: screen head at index %d", screen_head_idx)

        # Enable prefilter if screen head exists or env var set
        enable = screen_head_idx is not None or os.environ.get(
            "SGLANG_MEDUSA_PREFILTER", ""
        ) == "1"

        if not enable:
            return None

        # Load n-gram trie if available alongside the medusa model
        ngram_trie = None
        trie_path = os.path.join(model_path, "ngram_trie.pkl")
        if os.path.exists(trie_path):
            import pickle
            with open(trie_path, "rb") as f:
                ngram_trie = pickle.load(f)
            logger.info("DraftPreFilter: loaded n-gram trie from %s", trie_path)

        return DraftPreFilter(
            ngram_trie=ngram_trie,
            screen_head_idx=screen_head_idx,
            surprisal_threshold=float(os.environ.get(
                "SGLANG_PREFILTER_SURPRISAL", "8.0"
            )),
            screen_confidence_threshold=float(os.environ.get(
                "SGLANG_PREFILTER_SCREEN_CONF", "0.3"
            )),
            collect_telemetry=os.environ.get(
                "SGLANG_PREFILTER_TELEMETRY", "1"
            ) == "1",
        )

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )
        self._draft_tokens_buf = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self._positions_buf = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self._tree_mask_buf = torch.empty(
            (max_total_mask,), dtype=torch.bool, device=self.device
        )
        self._retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self._retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self._retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )

    # ---- Proxy attributes for scheduler ----

    @property
    def model_runner(self):
        return self.target_worker.model_runner

    @property
    def model_config(self):
        return self.target_worker.model_runner.model_config

    @property
    def max_running_requests(self):
        return self.target_worker.max_running_requests

    def get_memory_pool(self):
        return self.target_worker.get_memory_pool()

    def clear_cache_pool(self):
        self._cached_hidden = None
        self._hidden_available = False

    def get_prefilter_stats(self) -> Optional[dict]:
        """Get DraftPreFilter statistics (for monitoring/API)."""
        if self.prefilter is not None:
            return self.prefilter.get_stats()
        return None

    def export_prefilter_telemetry(self) -> Optional[list]:
        """Export head agreement data for contrastive fine-tuning."""
        if self.prefilter is not None:
            return self.prefilter.export_agreement_data()
        return None

    def get_sam_info(self) -> dict:
        """Get AMD SAM/ReBAR hardware info (for monitoring/API)."""
        return dict(self._sam_info)

    def get_medusa_config(self) -> dict:
        """Get full Medusa configuration (for monitoring/API)."""
        return {
            "num_heads": self.num_heads,
            "num_draft_tokens": self.num_draft_tokens,
            "tree_structure": self.tree_structure,
            "typical_acceptance": self.typical_acceptance,
            "posterior_threshold": self.posterior_threshold,
            "posterior_alpha": self.posterior_alpha,
            "topk": self.medusa_topk,
            "paltrow_heads": sorted(self.medusa_model._cpu_head_indices)
            if hasattr(self.medusa_model, "_cpu_head_indices")
            else [],
            "gpu_heads": sorted(self.medusa_model._gpu_head_indices)
            if hasattr(self.medusa_model, "_gpu_head_indices")
            else list(range(self.num_heads)),
            "sam": self._sam_info,
            "prefilter": "ON" if self.prefilter else "OFF",
        }

    # ---- Batch preparation ----

    def _prepare_batch_for_decode(self, batch: ScheduleBatch):
        """Prepare a ScheduleBatch for a standard 1-token decode.

        When speculative decoding is active, schedule_batch.prepare_for_decode()
        returns early (it defers to the speculative worker).  This method does
        the work that was skipped: set input_ids from last output, allocate one
        KV cache slot per request, and bump seq_lens.
        """
        bs = batch.batch_size()
        if bs == 0:
            return

        # Clear any stale spec_info (e.g. NgramVerifyInput from warmup)
        # so it doesn't leak positions into ForwardBatch.init_new.
        batch.spec_info = None

        # Build input_ids from the per-request output_ids (authoritative).
        # batch.output_ids may be stale after tree verify (contains all accepted
        # tokens, not one per request) and filter_batch doesn't trim it when
        # no requests are removed.
        last_tokens = [req.output_ids[-1] for req in batch.reqs]
        batch.input_ids = torch.tensor(
            last_tokens, dtype=torch.int32, device=batch.seq_lens.device
        )
        batch.output_ids = None

        # Allocate 1 KV cache slot per request
        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        # Bump per-request bookkeeping
        for req in batch.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        # Bump aggregate seq_lens
        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        batch.orig_seq_lens.add_(1)
        batch.seq_lens_sum += bs

    # ---- Main dispatch ----

    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        # On extend, delegate to target (with hidden capture enabled)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            self._enable_hidden_capture(batch)
            result = self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )
            self._disable_hidden_capture(batch)
            self._capture_hidden_from_result(result)
            return result

        bs = batch.batch_size()
        if bs == 0 or batch.forward_mode.is_idle():
            return self.target_worker.forward_batch_generation(
                batch.get_model_worker_batch()
            )

        # Try to generate Medusa drafts
        if self._hidden_available and self._cached_hidden is not None:
            try:
                return self._forward_medusa(batch)
            except Exception as e:
                logger.warning("Medusa draft failed (%s), falling back to target-only", e)
                if batch.forward_mode != ForwardMode.DECODE:
                    # _forward_medusa partially modified the batch (e.g., switched
                    # to TARGET_VERIFY and allocated KV).  Recovery is unsafe — 
                    # re-raise so the scheduler can handle it.
                    raise

        # Fallback: target-only decode (with hidden capture enabled).
        # prepare_for_decode() was skipped by the scheduler for spec algorithms,
        # so we must do the standard decode prep here.
        self._prepare_batch_for_decode(batch)
        self._enable_hidden_capture(batch)
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )
        self._disable_hidden_capture(batch)
        self._capture_hidden_from_result(result)

        # Spec v1 scheduler won't auto-append output_ids for spec algorithms,
        # so we do it here (same as non-spec path in process_batch_result_decode).
        if result.next_token_ids is not None:
            next_ids = result.next_token_ids.tolist()
            for req, tid in zip(batch.reqs, next_ids):
                req.output_ids.append(tid)
                req.check_finished()

        return result

    # ---- Medusa draft + verify ----

    def _forward_medusa(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Generate Medusa drafts and verify via NgramVerifyInput.

        The NgramVerifyInput tree requires the current (uncommitted) token at
        position 0, followed by K draft predictions.  This matches the C++
        ngram corpus invariant (result.cpp:fillResult prepends last_token).

        **Step 0 Decode (Off-by-one fix):**
        After verify, _cached_hidden = h(last_accepted_tree_position).  The bonus
        token (sampled from logits at that position) has NEVER been through the
        model, so Medusa heads working from the stale hidden always echo the
        current token at Head 0.

        We fix this by decoding the current token through the target model at
        the start of each round, then rolling back ALL batch state so the verify
        tree sees the original seq_lens.  The Step 0 KV slot is freed — only the
        captured hidden state h(current) survives.
        """
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        bs = batch.batch_size()
        K = self.num_draft_tokens       # pure draft count from Medusa heads
        T = self.draft_token_num         # tree size = K + 1 (current + drafts)

        # Determine if we can skip Step 0 this round.
        # no_step0 mode uses the stale hidden from verify, but the very first
        # round after prefill may not have cached hidden yet — fall back to Step 0.
        skip_step0 = self.no_step0 and self._hidden_available

        # ---- Step 0: Decode current token to get fresh h(current) ----
        # The bonus token from the previous verify has NEVER been through the
        # model, so _cached_hidden is stale.  We run ONE target forward of the
        # current token, capture h(current), then FULLY roll back batch state.
        # The only surviving side-effect is self._cached_hidden = h(current).
        #
        # When --medusa-no-step0 is enabled, we skip this entirely and use the
        # stale hidden from verify.  Head 0 will echo (useless) but Heads 1..K
        # shift down: Head 1 predicts +1 from current, Head 2 predicts +2, etc.
        # This eliminates 1 extra forward pass per round (~2× speed improvement).
        #
        # All mutations are wrapped in try/finally so an exception (OOM, NCCL,
        # etc.) cannot leave batch state corrupted.
        if not skip_step0:
            seq_lens_save = batch.seq_lens.clone()
            seq_lens_cpu_save = batch.seq_lens_cpu.clone()
            orig_seq_lens_save = batch.orig_seq_lens.clone()
            seq_lens_sum_save = batch.seq_lens_sum
            output_ids_save = batch.output_ids
            input_ids_save = batch.input_ids
            spec_info_save = batch.spec_info
            out_cache_loc_save = batch.out_cache_loc
            req_state_save = [
                (r.decode_batch_idx, r.kv_committed_len, r.kv_allocated_len)
                for r in batch.reqs
            ]
            # Save the req_to_token_pool slot that alloc_for_decode will overwrite
            rtp_positions = seq_lens_save.to(torch.int64)
            rtp_old_values = batch.req_to_token_pool.req_to_token[
                batch.req_pool_indices, rtp_positions
            ].clone()

            try:
                self._prepare_batch_for_decode(batch)
                self._enable_hidden_capture(batch)
                step0_mwb = batch.get_model_worker_batch()
                step0_result = self.target_worker.forward_batch_generation(step0_mwb)
                self._disable_hidden_capture(batch)
                self._capture_hidden_from_result(step0_result)
            finally:
                # Unconditional rollback — runs even on exception
                if batch.out_cache_loc is not None:
                    batch.token_to_kv_pool_allocator.free(batch.out_cache_loc)
                # Restore the req_to_token_pool entry overwritten by alloc_for_decode
                batch.req_to_token_pool.req_to_token[
                    batch.req_pool_indices, rtp_positions
                ] = rtp_old_values
                batch.seq_lens.copy_(seq_lens_save)
                batch.seq_lens_cpu.copy_(seq_lens_cpu_save)
                batch.orig_seq_lens.copy_(orig_seq_lens_save)
                batch.seq_lens_sum = seq_lens_sum_save
                batch.output_ids = output_ids_save
                batch.input_ids = input_ids_save
                batch.spec_info = spec_info_save
                batch.out_cache_loc = out_cache_loc_save
                for r, (dbi, kcl, kal) in zip(batch.reqs, req_state_save):
                    r.decode_batch_idx = dbi
                    r.kv_committed_len = kcl
                    r.kv_allocated_len = kal

        # Step 1: Run Medusa heads on cached hidden states (with logits for prefilter)
        hidden = self._cached_hidden
        if hidden.shape[0] < bs:
            hidden = hidden[:bs]
        elif hidden.shape[0] > bs:
            hidden = hidden[:bs]

        with torch.no_grad():
            if self.prefilter is not None:
                all_tokens, head_logits = self.medusa_model.predict_with_logits(
                    hidden,
                    typical=self.typical_acceptance,
                    posterior_threshold=self.posterior_threshold,
                    posterior_alpha=self.posterior_alpha,
                )
            else:
                all_tokens = self.medusa_model.predict_tokens(
                    hidden,
                    typical=self.typical_acceptance,
                    posterior_threshold=self.posterior_threshold,
                    posterior_alpha=self.posterior_alpha,
                )
                head_logits = None

            # Reorder heads: skip screen, use offset-sorted draft heads
            reorder = self._draft_head_indices[:K]
            draft_tokens = all_tokens[:, reorder]  # [bs, K]

        # Step 1.5: DraftPreFilter — drop unlikely candidates before verification
        if self.prefilter is not None and head_logits is not None:
            context_ids = None
            if self.prefilter.ngram_trie is not None:
                context_ids = []
                for req in batch.reqs:
                    ctx = list(req.origin_input_ids) + list(req.output_ids)
                    context_ids.append(ctx[-16:])

            screen_logits = None
            if self.prefilter.screen_head_idx is not None:
                sidx = self.prefilter.screen_head_idx
                if sidx < len(head_logits):
                    screen_logits = head_logits[sidx][:bs]

            filtered, keep_mask, telem = self.prefilter.filter_drafts(
                draft_tokens, [head_logits[i][:bs] for i in reorder],
                context_ids=context_ids,
                screen_logits=screen_logits,
            )

            # Replace dropped tokens with a safe fallback (repeat last accepted token)
            if not keep_mask.all():
                for b in range(bs):
                    last_tok = batch.reqs[b].output_ids[-1] if batch.reqs[b].output_ids else 0
                    for k in range(K):
                        if not keep_mask[b, k]:
                            draft_tokens[b, k] = last_tok

            self.prefilter.log_periodic_stats(interval=50)

        # Step 2: Build tree token array — prepend current token, then K drafts.
        # This matches the NGRAM invariant: tree[0] = current_token.
        # batch.input_ids was set by Step 0's _prepare_batch_for_decode and
        # still holds the current token per request.
        tree_tokens_list = []
        current_tokens = torch.tensor(
            [req.output_ids[-1] for req in batch.reqs],
            dtype=torch.int32, device="cuda",
        )
        for i in range(bs):
            tree_tokens_list.append(current_tokens[i:i+1])     # current token
            tree_tokens_list.append(draft_tokens[i])            # K draft predictions
        flat_tree = torch.cat(tree_tokens_list).contiguous()    # [bs * T]

        # Tree mask: use pre-cached GPU tensor (avoid numpy→torch per call)
        # _cached_mask_tiled_gpu is pre-tiled for max_batch_size; slice to actual bs
        cached_mask_slice = self._cached_mask_tiled_gpu[: bs * T * T]

        # Copy to pre-allocated GPU tensors with non-blocking DMA
        self._draft_tokens_buf[: bs * T].copy_(flat_tree, non_blocking=True)
        self._tree_mask_buf[: bs * T * T].copy_(cached_mask_slice, non_blocking=True)

        # Stream sync: ensure non-blocking copies complete before verify reads them.
        # Without this, reconstruct_indices may read stale buffer data.
        torch.cuda.current_stream().synchronize()

        # Step 3: Reconstruct tree indices
        positions = self._positions_buf[: bs * T]
        retrive_index = self._retrieve_indexes[:bs, :T]
        retrive_next_token = self._retrive_next_token[:bs, :T]
        retrive_next_sibling = self._retrive_next_sibling[:bs, :T]
        tree_mask_gpu = self._tree_mask_buf[: bs * T * T]

        try:
            from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

            reconstruct_indices_from_tree_mask(
                tree_mask_gpu,
                batch.seq_lens,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                bs,
                T,
            )
        except Exception as e:
            logger.warning("Medusa reconstruct_indices failed (%s), falling back", e)
            return self._fallback_target_only(batch)

        # Step 4: Build full attention mask (prefix + tree) using cached 3D mask
        # seq_lens is restored to pre-Step-0 state; prefix = everything before
        # the current token (tree[0]), which is seq_len - 1.
        tree_mask_list = []
        mask_np_3d = self._cached_mask_np_3d[:bs]  # use pre-cached, avoid reshape per call
        for i, req in enumerate(batch.reqs):
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            prefix_mask = torch.ones((T, seq_len - 1), device="cuda")
            tree_part = torch.from_numpy(mask_np_3d[i]).cuda()
            full_mask = torch.cat((prefix_mask, tree_part), dim=1).to(torch.bool)
            tree_mask_list.append(full_mask.flatten())
        full_tree_mask = torch.cat(tree_mask_list, dim=0)

        # Step 5: Set verify mode and run target
        original_algo = batch.spec_algorithm
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM  # reuse ngram verify path
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = NgramVerifyInput(
            self._draft_tokens_buf[: bs * T],
            full_tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            T,
        )
        # Capture hidden states during verify so we can feed Medusa heads next round.
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.spec_info.prepare_for_verify(batch, self.page_size)

        model_worker_batch = batch.get_model_worker_batch()

        if model_worker_batch.forward_mode.is_target_verify():
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True
            )
            logits_output = batch_result.logits_output
            can_run_cuda_graph = batch_result.can_run_cuda_graph

            verify_input: NgramVerifyInput = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted = verify_input.verify(
                batch, logits_output, self.page_size, None
            )
            accept_lens = verify_input.accept_length
            batch.forward_mode = ForwardMode.DECODE
            batch.spec_algorithm = original_algo

            # Lightweight periodic acceptance logging
            if not hasattr(self, "_acc_count"):
                self._acc_count = 0
                self._acc_sum = 0.0
            if accept_lens is not None:
                self._acc_count += 1
                mean_al = accept_lens.float().mean().item() if isinstance(accept_lens, torch.Tensor) else sum(accept_lens)/max(len(accept_lens),1)
                self._acc_sum += mean_al
                if self._acc_count % 10 == 0:
                    logger.info("MEDUSA accept_rate: last=%.2f avg=%.2f (K=%d, %d rounds)", mean_al, self._acc_sum / self._acc_count, K, self._acc_count)

            # Advance decode_batch_idx by accepted tokens + bonus.
            # This counter drives SWA eviction scheduling; without it, SWA
            # windows grow unbounded on the Medusa path.
            if accept_lens is not None:
                accept_lens_list = (
                    accept_lens.tolist()
                    if isinstance(accept_lens, torch.Tensor)
                    else accept_lens
                )
                for i, req in enumerate(batch.reqs):
                    if i < len(accept_lens_list):
                        req.decode_batch_idx += int(accept_lens_list[i]) + 1

            # Feed verify results back to prefilter for adaptive threshold tuning.
            # accept_length counts accepted nodes in the TREE (including the always-
            # accepted current token at position 0).  For prefilter feedback we only
            # care about the K draft positions (positions 1..K), so subtract 1.
            if self.prefilter is not None and accept_lens is not None:
                try:
                    accepted_mask = torch.zeros(bs, K, dtype=torch.bool, device="cuda")
                    accept_lens_list = (
                        accept_lens.tolist()
                        if isinstance(accept_lens, torch.Tensor)
                        else accept_lens
                    )
                    for b_idx, alen in enumerate(accept_lens_list):
                        if b_idx < bs:
                            # alen includes current token at pos 0; draft acceptance = alen - 1
                            n_draft_accepted = min(max(int(alen) - 1, 0), K)
                            accepted_mask[b_idx, :n_draft_accepted] = True
                    self.prefilter.record_verify_results(accepted_mask)
                except Exception as e:
                    logger.debug("Prefilter feedback failed: %s", e)

            # Capture hidden state from verify for next round's Medusa heads.
            # Note: this captures h(last_accepted), which is stale by one position.
            # The Step 0 decode at the start of _forward_medusa refreshes it.
            self._last_accept_lens = accept_lens
            self._capture_hidden_from_verify(batch_result, logits_output)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=num_accepted,
                can_run_cuda_graph=can_run_cuda_graph,
                accept_lens=accept_lens,
            )

        # Unexpected mode — fallback
        batch.spec_algorithm = original_algo
        return self._fallback_target_only(batch)

    # ---- Hidden state capture ----

    def _enable_hidden_capture(self, batch: ScheduleBatch):
        """Ensure the target forward will capture last hidden states for Medusa.

        ALWAYS replace spec_info — stale objects (e.g. NgramVerifyInput left
        by warmup or a previous scheduler cycle) carry a .positions attribute
        that overrides the correct decode positions in ForwardBatch.init_new.
        """
        batch.spec_info = _MedusaCaptureCarrier()

    def _disable_hidden_capture(self, batch: ScheduleBatch):
        """Clear spec_info after forward so the scheduler doesn't trip over it."""
        batch.spec_info = None

    def _capture_hidden_from_result(self, result: GenerationBatchResult):
        """Extract hidden states from logits_output (same path as EAGLE)."""
        try:
            lo = result.logits_output
            if lo is not None:
                h = getattr(lo, "hidden_states", None)
                if h is not None and isinstance(h, torch.Tensor) and h.numel() > 0:
                    self._cached_hidden = h.detach()
                    self._hidden_available = True
                    return
        except Exception:
            pass
        self._hidden_available = False

    def _capture_hidden_from_verify(
        self, batch_result: GenerationBatchResult, verified_logits: LogitsProcessorOutput
    ):
        """After verify, hidden_states in verified_logits are filtered to accepted
        positions (flattened across all requests).  Extract the last accepted
        hidden per request using cumulative accept_length offsets."""
        try:
            h = getattr(verified_logits, "hidden_states", None)
            if h is not None and isinstance(h, torch.Tensor) and h.numel() > 0:
                # h shape: [sum(accept_length_i + 1), hidden] — flattened
                if hasattr(self, '_last_accept_lens') and self._last_accept_lens is not None:
                    al = self._last_accept_lens  # [bs] tensor
                    offsets = torch.cumsum(al + 1, dim=0) - 1  # last pos per req
                    offsets = offsets.clamp(max=h.shape[0] - 1)
                    self._cached_hidden = h[offsets].detach()
                else:
                    self._cached_hidden = h[-1:].detach()
                self._hidden_available = True
                return
        except Exception:
            pass
        # Fallback: try batch_result (pre-verify, unfiltered)
        self._capture_hidden_from_result(batch_result)

    # ---- Fallback ----

    def _fallback_target_only(self, batch: ScheduleBatch) -> GenerationBatchResult:
        batch.forward_mode = ForwardMode.DECODE
        self._prepare_batch_for_decode(batch)
        self._enable_hidden_capture(batch)
        result = self.target_worker.forward_batch_generation(
            batch.get_model_worker_batch()
        )
        self._disable_hidden_capture(batch)
        self._capture_hidden_from_result(result)
        if result.next_token_ids is not None:
            next_ids = result.next_token_ids.tolist()
            for req, tid in zip(batch.reqs, next_ids):
                req.output_ids.append(tid)
                req.check_finished()
        return result
