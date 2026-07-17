"""Layer 1: physical open-quantum-system models.

Importing this package registers the available models with
:class:`ModelRegistry`.
"""

from __future__ import annotations

from .base import AbstractOQSModel, validate_channel
from .gaudin import (
    COUPLING_PROFILES,
    GaudinBathParams,
    GaudinModel,
    coupling_profile,
    exponential_couplings,
    linear_couplings,
    ou_couplings,
    random_couplings,
    uniform_couplings,
)
from .registry import ModelRegistry
from .spin_boson import SpinBosonBathParams, SpinBosonModel

ModelRegistry.register("spin_boson", SpinBosonModel)
ModelRegistry.register("gaudin", GaudinModel)

__all__ = [
    "AbstractOQSModel",
    "validate_channel",
    "ModelRegistry",
    "SpinBosonModel",
    "SpinBosonBathParams",
    "GaudinModel",
    "GaudinBathParams",
    "COUPLING_PROFILES",
    "coupling_profile",
    "linear_couplings",
    "uniform_couplings",
    "exponential_couplings",
    "random_couplings",
    "ou_couplings",
]
