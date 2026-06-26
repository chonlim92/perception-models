"""Decorator-based registry for models, datasets, losses, metrics, and transforms.

This module provides a generic ``Registry`` class that supports:

- Registering callables (classes or functions) under a unique name.
- Optional category/group annotations for organizational purposes.
- Strict duplicate detection with clear error messages.
- Type checking to ensure registered items match an expected base type.
- Convenient retrieval and listing helpers.

Pre-built global registries are available as module-level singletons:

    >>> from common.registry import MODELS
    >>> @MODELS.register("resnet50")
    ... class ResNet50(nn.Module): ...

Usage
-----
Creating a custom registry::

    from common.registry import Registry
    MY_REGISTRY = Registry("my_things")

    @MY_REGISTRY.register("thing_a", category="group1")
    class ThingA: ...

    cls = MY_REGISTRY.get("thing_a")
    all_names = MY_REGISTRY.list()
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, overload

__all__ = [
    "Registry",
    "MODELS",
    "DATASETS",
    "LOSSES",
    "METRICS",
    "TRANSFORMS",
]

T = TypeVar("T", bound=Callable[..., Any])


class RegistryError(Exception):
    """Base exception for registry-related errors."""


class DuplicateRegistrationError(RegistryError):
    """Raised when attempting to register a name that already exists."""


class NotRegisteredError(RegistryError):
    """Raised when attempting to retrieve a name that has not been registered."""


class RegistryTypeError(RegistryError):
    """Raised when a registered item does not satisfy the expected type constraint."""


class Registry:
    """A named registry that maps string keys to callables (classes or functions).

    Parameters
    ----------
    name : str
        Human-readable name for this registry (used in error messages).
    base_type : type | None
        If provided, all registered items must be subclasses of this type.
        Only applies when the registered item is a class (not a plain function).

    Examples
    --------
    >>> reg = Registry("optimizers")
    >>> @reg.register("sgd")
    ... class SGD: ...
    >>> reg.get("sgd") is SGD
    True
    >>> reg.list()
    ['sgd']
    """

    def __init__(self, name: str, *, base_type: type | None = None) -> None:
        self._name = name
        self._base_type = base_type
        self._registry: dict[str, Callable[..., Any]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the human-readable name of this registry."""
        return self._name

    @property
    def base_type(self) -> type | None:
        """Return the base type constraint, or ``None`` if unconstrained."""
        return self._base_type

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @overload
    def register(self, name: str, *, category: str | None = ...) -> Callable[[T], T]: ...

    @overload
    def register(self, name: str, obj: T, *, category: str | None = ...) -> T: ...

    def register(
        self,
        name: str,
        obj: T | None = None,
        *,
        category: str | None = None,
    ) -> T | Callable[[T], T]:
        """Register a callable under *name*.

        Can be used as a decorator (with or without parentheses) or called
        directly with the object to register.

        Parameters
        ----------
        name : str
            Unique identifier within this registry.
        obj : callable, optional
            The item to register.  If omitted, a decorator is returned.
        category : str | None
            Optional category/group label stored as metadata.

        Returns
        -------
        The original callable (unchanged) when used as a decorator, enabling
        transparent stacking with other decorators.

        Raises
        ------
        DuplicateRegistrationError
            If *name* is already registered.
        RegistryTypeError
            If *obj* is a class and does not satisfy *base_type*.
        TypeError
            If *name* is not a string or *obj* is not callable.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"[{self._name}] Registry key must be a string, got {type(name).__name__!r}."
            )
        if not name:
            raise ValueError(f"[{self._name}] Registry key must be a non-empty string.")

        def _do_register(item: T) -> T:
            if not callable(item):
                raise TypeError(
                    f"[{self._name}] Only callable objects can be registered, "
                    f"got {type(item).__name__!r} for key {name!r}."
                )

            if name in self._registry:
                existing = self._registry[name]
                raise DuplicateRegistrationError(
                    f"[{self._name}] Key {name!r} is already registered to "
                    f"{_qualified_name(existing)}. Cannot register "
                    f"{_qualified_name(item)} under the same key."
                )

            # Type checking: if base_type is set and item is a class, verify inheritance.
            if self._base_type is not None and isinstance(item, type):
                if not issubclass(item, self._base_type):
                    raise RegistryTypeError(
                        f"[{self._name}] {_qualified_name(item)} is not a subclass of "
                        f"{_qualified_name(self._base_type)}. All registered classes must "
                        f"inherit from {self._base_type.__name__}."
                    )

            self._registry[name] = item
            self._metadata[name] = {
                "category": category,
                "qualname": _qualified_name(item),
            }
            return item

        # Support both `@registry.register("x")` and `registry.register("x", obj)`.
        if obj is not None:
            return _do_register(obj)
        return _do_register

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> Callable[..., Any]:
        """Retrieve the callable registered under *name*.

        Parameters
        ----------
        name : str
            The key to look up.

        Returns
        -------
        The registered callable.

        Raises
        ------
        NotRegisteredError
            If *name* has not been registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise NotRegisteredError(
                f"[{self._name}] {name!r} is not registered. "
                f"Available entries: {available}."
            )
        return self._registry[name]

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Return metadata dict for a registered entry.

        Raises
        ------
        NotRegisteredError
            If *name* has not been registered.
        """
        if name not in self._metadata:
            raise NotRegisteredError(
                f"[{self._name}] {name!r} is not registered."
            )
        return dict(self._metadata[name])

    # ------------------------------------------------------------------
    # Listing / Querying
    # ------------------------------------------------------------------

    def list(self, *, category: str | None = None) -> list[str]:
        """Return a sorted list of all registered names.

        Parameters
        ----------
        category : str | None
            If provided, only return names whose category matches.

        Returns
        -------
        Sorted list of registered keys.
        """
        if category is None:
            return sorted(self._registry.keys())
        return sorted(
            name
            for name, meta in self._metadata.items()
            if meta.get("category") == category
        )

    def categories(self) -> list[str]:
        """Return a sorted list of all unique non-None categories."""
        cats: set[str] = set()
        for meta in self._metadata.values():
            cat = meta.get("category")
            if cat is not None:
                cats.add(cat)
        return sorted(cats)

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._registry

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        """Iterate over registered names in sorted order."""
        return iter(sorted(self._registry.keys()))

    def __repr__(self) -> str:
        return (
            f"Registry(name={self._name!r}, entries={len(self._registry)}, "
            f"base_type={_qualified_name(self._base_type) if self._base_type else None})"
        )

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries. Primarily useful for testing."""
        self._registry.clear()
        self._metadata.clear()

    def unregister(self, name: str) -> None:
        """Remove a single entry by name.

        Raises
        ------
        NotRegisteredError
            If *name* is not currently registered.
        """
        if name not in self._registry:
            raise NotRegisteredError(
                f"[{self._name}] Cannot unregister {name!r} because it is not registered."
            )
        del self._registry[name]
        del self._metadata[name]


# ---------------------------------------------------------------------------
# Module-level singleton registries
# ---------------------------------------------------------------------------

MODELS: Registry = Registry("MODELS")
"""Global registry for perception model architectures."""

DATASETS: Registry = Registry("DATASETS")
"""Global registry for dataset classes."""

LOSSES: Registry = Registry("LOSSES")
"""Global registry for loss functions."""

METRICS: Registry = Registry("METRICS")
"""Global registry for evaluation metrics."""

TRANSFORMS: Registry = Registry("TRANSFORMS")
"""Global registry for data transforms / augmentations."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _qualified_name(obj: Any) -> str:
    """Return the best available qualified name for *obj*."""
    if obj is None:
        return "None"
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", repr(obj))
    if module and module != "__main__":
        return f"{module}.{qualname}"
    return str(qualname)
