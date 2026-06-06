"""Layer 1: physical open-quantum-system models.

Importing this package registers the available models with
:class:`ModelRegistry`.
"""

from __future__ import annotations

from .base import AbstractOQSModel
from .registry import ModelRegistry
from .spin_boson import SpinBosonBathParams, SpinBosonModel

ModelRegistry.register("spin_boson", SpinBosonModel)

__all__ = [
    "AbstractOQSModel",
    "ModelRegistry",
    "SpinBosonModel",
    "SpinBosonBathParams",
]
