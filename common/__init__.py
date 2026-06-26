"""Common utilities for the perception-models project.

This package provides shared infrastructure used across the perception
model training and evaluation pipeline:

- **registry** -- Decorator-based registries for models, datasets, losses,
  metrics, and transforms.
- **datasets** -- Dataset loading and preprocessing utilities.
- **metrics** -- Evaluation metric implementations.
- **transforms** -- Data augmentation and transformation pipelines.
- **visualization** -- Plotting and visual debugging helpers.
"""

from common.registry import (
    DATASETS,
    LOSSES,
    METRICS,
    MODELS,
    TRANSFORMS,
    Registry,
)

__all__ = [
    # Registry infrastructure
    "Registry",
    "MODELS",
    "DATASETS",
    "LOSSES",
    "METRICS",
    "TRANSFORMS",
]
