"""
ATOM-RS Rust adapter — thin pass-through.

atom_shims.py already mounts sglang.jit_kernel.atom_utils.* as atom.utils.*.
This module is kept for compatibility; its install_atom_utils_shim() call is
now a no-op because atom_shims.py handles everything.

If rs_atom_utils_pyo3 (or other PyO3 crates) become available in the future
they can be wired in here to replace the Python implementations.
"""

import sys
import importlib
from typing import Any, Dict, List


def _try_import_rust_binding(module_name: str, items: List[str]) -> Dict[str, Any]:
    """Attempt to import a PyO3 Rust binding module."""
    try:
        mod = importlib.import_module(module_name)
        return {item: getattr(mod, item) for item in items if hasattr(mod, item)}
    except ImportError:
        return {}


def install_atom_utils_shim() -> None:
    """
    No-op: atom_shims.py already mounts atom.utils.* from the real
    sglang.jit_kernel.atom_utils package.  This function is kept so that
    any code that imports and calls it does not break.
    """
    pass


__all__ = ["install_atom_utils_shim"]
