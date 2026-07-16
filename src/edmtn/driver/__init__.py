"""Layer 7: orchestration.

``EDMSolver`` wires the model, engines, and observable extraction into a single
``solve`` call, auto-selecting the pipeline from the model's ``bath_type``.
"""

from __future__ import annotations

from .auto_config import (
    SolverConfig,
    available_pipelines,
    build_pipeline,
    register_pipeline,
    resolve_config_for_model,
    resolve_expansion_order,
)
from .solver import EDMSolver, SolverResult, solve

__all__ = [
    "EDMSolver",
    "SolverConfig",
    "SolverResult",
    "solve",
    "build_pipeline",
    "register_pipeline",
    "available_pipelines",
    "resolve_expansion_order",
    "resolve_config_for_model",
]
