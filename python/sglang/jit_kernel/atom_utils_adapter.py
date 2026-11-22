"""
Adapter to redirect atom.utils imports to local jit_kernel.atom_utils.
This allows ported kernels to import from atom.utils while using the ported utilities.
"""

import sys
import types

def install_utils_shim():
    """Install atom.utils shim that points to jit_kernel.atom_utils."""
    
    # Import actual utilities
    from sglang.jit_kernel import atom_utils
    
    # Create shim module
    shim = types.ModuleType("atom.utils")
    
    # Copy all exports from actual utils to shim
    for name in dir(atom_utils):
        if not name.startswith('_'):
            setattr(shim, name, getattr(atom_utils, name))
    
    sys.modules["atom.utils"] = shim
    
    # Also install submodules
    for name in dir(atom_utils):
        if not name.startswith('_'):
            attr = getattr(atom_utils, name)
            if isinstance(attr, types.ModuleType):
                sys.modules[f"atom.utils.{name}"] = attr

# Auto-install
install_utils_shim()
