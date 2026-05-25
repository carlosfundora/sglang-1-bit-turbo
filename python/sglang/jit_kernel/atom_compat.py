"""
gfxATOM compatibility layer for ATOM-RS.

Maps gfxATOM concepts to ATOM-RS equivalents to enable kernel reuse.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScheduledBatch:
    """Compatibility stub for gfxATOM ScheduledBatch.
    
    Maps to ATOM-RS batch structures as needed.
    """
    pass


# Model operations namespace
class ModelOps:
    """Compatibility namespace for model operations."""
    pass


# Create dummy module structure
class AtomModule:
    model_engine = type('obj', (object,), {'scheduler': ScheduledBatch})()
    model_ops = ModelOps()


# Export for import compatibility
atom = AtomModule()


__all__ = ['atom', 'ScheduledBatch']
