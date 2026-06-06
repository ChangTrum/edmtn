"""Model registry (Layer 1).

Maps model names to factories so callers can construct a model by name without
importing the concrete class::

    model = ModelRegistry.create("spin_boson", J0=0.5, omega_c=5.0, mu=1.0)
"""

from __future__ import annotations

from typing import Callable

from .base import AbstractOQSModel


class ModelRegistry:
    """Name -> factory registry for open-quantum-system models."""

    _registry: dict[str, Callable[..., AbstractOQSModel]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[..., AbstractOQSModel],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a model ``factory`` (usually the class) under ``name``."""
        if not overwrite and name in cls._registry:
            raise KeyError(f"model {name!r} already registered")
        cls._registry[name] = factory

    @classmethod
    def create(cls, name: str, **kwargs) -> AbstractOQSModel:
        """Instantiate the model registered under ``name``."""
        try:
            factory = cls._registry[name]
        except KeyError:
            raise KeyError(
                f"unknown model {name!r}; available: {cls.available()}"
            ) from None
        return factory(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return the sorted list of registered model names."""
        return sorted(cls._registry)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry
