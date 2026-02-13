"""Factor registry for name-based discovery and instantiation.

Factors register themselves with :func:`register_factor` so that YAML
configs and CLI scripts can reference them by short name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant.factors.base import Factor

_FACTOR_REGISTRY: dict[str, type[Factor]] = {}


def register_factor(name: str) -> Callable[[type[Factor]], type[Factor]]:
    """Class decorator that registers a factor under *name*.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"momentum_12m"``).

    Examples
    --------
    >>> @register_factor("my_factor")
    ... class MyFactor(Factor): ...
    """

    def decorator(cls: type[Factor]) -> type[Factor]:
        _FACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_factor(name: str) -> type[Factor]:
    """Retrieve a registered factor class by *name*.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _FACTOR_REGISTRY:
        available = ", ".join(sorted(_FACTOR_REGISTRY)) or "(none)"
        raise KeyError(
            f"Factor {name!r} not found. Available: {available}"
        )
    return _FACTOR_REGISTRY[name]


def list_factors() -> list[str]:
    """Return sorted list of all registered factor names."""
    return sorted(_FACTOR_REGISTRY)
