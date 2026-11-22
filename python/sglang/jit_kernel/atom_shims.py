"""
ATOM-RS — atom.* namespace installer.

Mounts ATOM-RS's own real implementations from sglang.jit_kernel.atom_utils
into the sys.modules atom.* namespace so attention_ops/*.py can import from
atom.utils, atom.utils.forward_context, etc.

Rules:
  - NEVER touch aiter.* (real installed package).
  - NEVER return fake/mock data — all implementations must be real.
  - atom.utils.* → sglang.jit_kernel.atom_utils.* (real code).
  - atom.model_ops / atom.plugin / atom.config → protocol classes only.
"""

import sys
import types

# Track the stub AttentionBackend so _promote_real_backends_module() can
# retroactively fix subclasses that were created before the real module loaded.
_stub_AttentionBackend = None


# ---------------------------------------------------------------------------
# Internal helper: register a module under a given dotted name and also wire
# it into every parent package already in sys.modules.
# ---------------------------------------------------------------------------
def _register(name: str, mod: types.ModuleType) -> None:
    sys.modules[name] = mod
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent_name, attr = parts
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr, mod)


# ---------------------------------------------------------------------------
# 1. Mount real atom_utils/ implementations as atom.utils.*
# ---------------------------------------------------------------------------
def _mount_atom_utils() -> None:
    """
    Mount sglang.jit_kernel.atom_utils (and submodules) as atom.utils.*.

    Uses importlib to load each module *once* and then aliases it.  This
    ensures forward_context.py, block_convert.py, tbo/ubatch_splitting.py,
    etc. are the live objects — not re-parsed stubs.
    """
    import importlib

    # Map of target atom.* name → source sglang.* name
    module_map = [
        ("atom.utils",
         "sglang.jit_kernel.atom_utils"),
        ("atom.utils.forward_context",
         "sglang.jit_kernel.atom_utils.forward_context"),
        ("atom.utils.block_convert",
         "sglang.jit_kernel.atom_utils.block_convert"),
        ("atom.utils.tbo",
         "sglang.jit_kernel.atom_utils.tbo"),
        ("atom.utils.tbo.ubatch_splitting",
         "sglang.jit_kernel.atom_utils.tbo.ubatch_splitting"),
    ]

    for atom_name, src_name in module_map:
        try:
            mod = importlib.import_module(src_name)
            _register(atom_name, mod)
        except Exception as exc:
            print(f"[WARN] atom_shims: could not mount {atom_name} "
                  f"from {src_name}: {exc}")


# ---------------------------------------------------------------------------
# 2. atom base + atom.config (protocol / config stubs — not mock data)
# ---------------------------------------------------------------------------
def _install_atom_protocol_shims() -> None:
    atom_mod = sys.modules.get("atom") or types.ModuleType("atom")

    # atom.config — lightweight config types used only as type annotations
    atom_config = types.ModuleType("atom.config")

    class KVCacheTensor:
        """Protocol type for KV cache tensors."""
        pass

    class Config:
        pass

    class ParallelConfig:
        tp_size: int = 1
        pp_size: int = 1

    def get_current_atom_config():
        return None

    atom_config.KVCacheTensor = KVCacheTensor
    atom_config.Config = Config
    atom_config.ParallelConfig = ParallelConfig
    atom_config.get_current_atom_config = get_current_atom_config

    # atom.kv_transfer — optional, safe to be absent
    kv_transfer = types.ModuleType("atom.kv_transfer")
    disaggregation = types.ModuleType("atom.kv_transfer.disaggregation")
    kv_transfer.disaggregation = disaggregation

    _register("atom.config", atom_config)
    _register("atom.kv_transfer", kv_transfer)
    _register("atom.kv_transfer.disaggregation", disaggregation)

    atom_mod.config = atom_config
    atom_mod.kv_transfer = kv_transfer
    _register("atom", atom_mod)


# ---------------------------------------------------------------------------
# 3. atom.plugin.* — decorator shims that leave classes unchanged
# ---------------------------------------------------------------------------
def _install_plugin_shims() -> None:
    def passthrough(*args, **kwargs):
        """@passthrough and @passthrough(...) both return the class unchanged."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _inner(obj):
            return obj
        return _inner

    atom_plugin = types.ModuleType("atom.plugin")
    plugin_attention = types.ModuleType("atom.plugin.attention")
    plugin_attention.AiterAttentionMetadataBuilderDecoratorForPluginMode = passthrough
    plugin_attention.AiterBackendDecoratorForPluginMode = passthrough
    plugin_attention.AiterMLAAttentionMetadataBuilderDecoratorForPluginMode = passthrough

    plugin_prepare = types.ModuleType("atom.plugin.prepare")
    plugin_prepare.is_plugin_mode = lambda: False

    plugin_sglang = types.ModuleType("atom.plugin.sglang")
    attn_backend = types.ModuleType("atom.plugin.sglang.attention_backend")
    radix_attn = types.ModuleType("atom.plugin.sglang.attention_backend.radix_attention")

    class RadixAttention:
        pass

    radix_attn.RadixAttention = RadixAttention
    attn_backend.radix_attention = radix_attn
    plugin_sglang.attention_backend = attn_backend
    atom_plugin.attention = plugin_attention
    atom_plugin.prepare = plugin_prepare
    atom_plugin.sglang = plugin_sglang

    _register("atom.plugin", atom_plugin)
    _register("atom.plugin.attention", plugin_attention)
    _register("atom.plugin.prepare", plugin_prepare)
    _register("atom.plugin.sglang", plugin_sglang)
    _register("atom.plugin.sglang.attention_backend", attn_backend)
    _register("atom.plugin.sglang.attention_backend.radix_attention", radix_attn)


# ---------------------------------------------------------------------------
# 4. atom.model_ops.* — protocol base classes for attention implementations
# ---------------------------------------------------------------------------
def _install_model_ops_shims() -> None:
    global _stub_AttentionBackend
    import torch
    from abc import ABC

    # ---- base classes -------------------------------------------------------
    # IMPORTANT: Must use ABC so metaclass matches the real AttentionBackend(ABC)
    # in sglang.jit_kernel.attention_ops.backends. Python requires the same
    # metaclass when reassigning __bases__ in _promote_real_backends_module().
    class AttentionBackend(ABC):
        accept_output_buffer: bool = False
        @staticmethod
        def get_name() -> str: return "base"
        @staticmethod
        def get_builder_cls(): return None
        @staticmethod
        def get_impl_cls(): return None

    _stub_AttentionBackend = AttentionBackend
    class AttentionMetadataBuilder:
        pass

    class CommonAttentionBuilder:
        pass

    # ---- attention model ops ------------------------------------------------
    class PagedAttention:
        pass

    class PagedAttentionImpl:
        pass

    class MLAAttention:
        pass

    class GatedDeltaNet:
        pass

    # ---- forward context (re-use real one from atom_utils) ------------------
    _fwd_ctx_mod = sys.modules.get("atom.utils.forward_context")

    def get_forward_context():
        if _fwd_ctx_mod and hasattr(_fwd_ctx_mod, "get_forward_context"):
            return _fwd_ctx_mod.get_forward_context()
        return None

    # ---- v4 kernels (no-op until Rust kernels are wired) --------------------
    # These are called only if GatedDeltaNet / DeepSeek V4 path is active.
    # They raise NotImplementedError so failures are obvious, not silent.
    def _not_implemented(*a, **kw):
        raise NotImplementedError(
            "atom.model_ops.v4_kernels: Rust V4 kernels not yet wired into ATOM-RS. "
            "Use standard paged attention path.")

    # ---- model_engine.scheduler ---------------------------------------------
    class ScheduledBatch:
        def __init__(self, batch_size: int = 1):
            self.batch_size = batch_size

    class KernelScheduler:
        def schedule(self, kernel_name, batch):
            return True

    # ---- assemble modules ---------------------------------------------------
    model_ops = types.ModuleType("atom.model_ops")
    model_ops.get_forward_context = get_forward_context

    attentions = types.ModuleType("atom.model_ops.attentions")
    backends_mod = types.ModuleType("atom.model_ops.attentions.backends")
    backends_mod.AttentionBackend = AttentionBackend
    backends_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
    backends_mod.CommonAttentionBuilder = CommonAttentionBuilder
    attentions.backends = backends_mod

    attn_mha = types.ModuleType("atom.model_ops.attention_mha")
    attn_mha.PagedAttention = PagedAttention
    attn_mha.PagedAttentionImpl = PagedAttentionImpl

    paged_attn = types.ModuleType("atom.model_ops.paged_attention")
    paged_attn.PagedAttention = PagedAttention

    attn_mla = types.ModuleType("atom.model_ops.attention_mla")
    attn_mla.MLAAttention = MLAAttention
    attn_mla._MLA_MIN_HEADS = 8
    attn_mla.MLAModules = dict

    attn_gdn = types.ModuleType("atom.model_ops.attention_gdn")
    attn_gdn.GatedDeltaNet = GatedDeltaNet

    v4_kernels = types.ModuleType("atom.model_ops.v4_kernels")
    for fn_name in (
        "write_v4_paged_decode_indices",
        "write_v4_paged_prefill_indices",
        "gd_paged_attention_v4_fwd",
        "gd_paged_attention_v4_fwd_bwd",
    ):
        setattr(v4_kernels, fn_name, _not_implemented)

    model_engine = types.ModuleType("atom.model_engine")
    scheduler_mod = types.ModuleType("atom.model_engine.scheduler")
    scheduler_mod.ScheduledBatch = ScheduledBatch
    scheduler_mod.KernelScheduler = KernelScheduler
    model_engine.scheduler = scheduler_mod

    model_ops.attentions = attentions
    model_ops.attention_mha = attn_mha
    model_ops.paged_attention = paged_attn
    model_ops.attention_mla = attn_mla
    model_ops.attention_gdn = attn_gdn
    model_ops.v4_kernels = v4_kernels

    _register("atom.model_ops", model_ops)
    _register("atom.model_ops.attentions", attentions)
    _register("atom.model_ops.attentions.backends", backends_mod)
    _register("atom.model_ops.attention_mha", attn_mha)
    _register("atom.model_ops.paged_attention", paged_attn)
    _register("atom.model_ops.attention_mla", attn_mla)
    _register("atom.model_ops.attention_gdn", attn_gdn)
    _register("atom.model_ops.v4_kernels", v4_kernels)
    _register("atom.model_engine", model_engine)
    _register("atom.model_engine.scheduler", scheduler_mod)


# ---------------------------------------------------------------------------
# 5. Final: ensure plugin decorators on atom.plugin.attention are still the
#    passthrough versions after all modules are wired.
# ---------------------------------------------------------------------------
def _install_decorators() -> None:
    # Already installed by _install_plugin_shims; this is a no-op safety pass.
    plugin_attention = sys.modules.get("atom.plugin.attention")
    if plugin_attention is None:
        return

    def passthrough(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def _inner(obj):
            return obj
        return _inner

    for name in (
        "AiterAttentionMetadataBuilderDecoratorForPluginMode",
        "AiterBackendDecoratorForPluginMode",
        "AiterMLAAttentionMetadataBuilderDecoratorForPluginMode",
    ):
        setattr(plugin_attention, name, passthrough)


# ---------------------------------------------------------------------------
# Auto-install on import
# ---------------------------------------------------------------------------
def install_all_shims() -> None:
    """
    Installation order is critical:
    1. atom.config + atom.model_engine.scheduler (needed by tbo and backends.py)
    2. atom.plugin shims (decorators)
    3. atom.model_ops stubs (MLAModules, etc.) — needed by backends.py imports
    4. atom.utils.* real mounts (CpuGpuBuffer etc.)
    5. PROMOTE real sglang.jit_kernel.attention_ops.backends → atom.model_ops.attentions.backends
       BEFORE any attention backend file is imported, so all backends inherit
       from the same AttentionBackend class object
    6. Re-run decorators
    """
    _install_atom_protocol_shims()
    _install_plugin_shims()
    _install_model_ops_shims()   # must precede mount_atom_utils so MLAModules etc. exist
    _mount_atom_utils()          # atom.utils.* real implementations
    _promote_real_backends_module()  # MUST run before any attention_ops/*.py is imported
    _install_decorators()
    print("[INFO] ✅ atom.* shims installed (real implementations)")


def _promote_real_backends_module() -> None:
    """
    Replace the stub atom.model_ops.attentions.backends with the real
    sglang.jit_kernel.attention_ops.backends module so that all backends
    that inherit AttentionBackend use the SAME class object.

    Because importing sglang.jit_kernel.attention_ops.backends triggers the
    package __init__.py (which auto-imports ALL backends), those backend classes
    are created before we can swap the module.  After the swap, we walk all
    __subclasses__ of the stub AttentionBackend and rebase them onto the real one.

    This must run after atom.utils.* is mounted (backends.py imports from it).
    """
    import importlib
    try:
        real_backends = importlib.import_module(
            "sglang.jit_kernel.attention_ops.backends"
        )
        _register("atom.model_ops.attentions.backends", real_backends)
        # Also patch the parent module reference
        attentions = sys.modules.get("atom.model_ops.attentions")
        if attentions is not None:
            attentions.backends = real_backends

        # --- Rebase stub subclasses onto the real AttentionBackend -----------
        # When sglang.jit_kernel.attention_ops was imported above (via its
        # __init__.py), all backends (deepseek_v4_attn, aiter_attention, etc.)
        # were loaded and inherited from _stub_AttentionBackend.  Python allows
        # mutating __bases__, so we swap the stub for the real class now.
        real_AB = getattr(real_backends, "AttentionBackend", None)
        stub_AB = _stub_AttentionBackend
        if real_AB is not None and stub_AB is not None and real_AB is not stub_AB:
            def _rebase(cls):
                new_bases = tuple(
                    real_AB if b is stub_AB else b for b in cls.__bases__
                )
                if new_bases != cls.__bases__:
                    try:
                        cls.__bases__ = new_bases
                    except TypeError:
                        pass  # built-in types can't be rebased; skip
                for sub in cls.__subclasses__():
                    _rebase(sub)
            _rebase(stub_AB)

    except Exception as exc:
        print(f"[WARN] atom_shims: could not promote real backends module: {exc}")


install_all_shims()

__all__ = ["install_all_shims"]
