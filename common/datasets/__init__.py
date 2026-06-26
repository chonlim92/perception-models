"""Dataset registry and factory for perception model training.

This module provides:
- Automatic registration of dataset classes via the global ``DATASETS`` registry.
- A ``build_dataset()`` factory function that instantiates datasets by name.
- Convenient imports for all supported dataset loaders.

Supported datasets:
- ``"nuscenes"`` : nuScenes multi-modal perception dataset
- ``"kitti"`` : KITTI 3D object detection dataset

Usage
-----
::

    from common.datasets import build_dataset

    # Build by name with keyword arguments
    train_ds = build_dataset("nuscenes", dataroot="/data/nuscenes", split="train")
    val_ds = build_dataset("kitti", dataroot="/data/kitti", split="val")

    # Or import and use directly
    from common.datasets import NuScenesDataset, KITTIDataset
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from common.registry import DATASETS

# Import dataset modules to trigger registration via @DATASETS.register(...)
from common.datasets.nuscenes_dataset import NuScenesDataset
from common.datasets.kitti_dataset import KITTIDataset

__all__ = [
    "build_dataset",
    "NuScenesDataset",
    "KITTIDataset",
    "DATASETS",
]


def build_dataset(name: str, **kwargs: Any) -> Any:
    """Factory function to instantiate a dataset by registered name.

    Looks up the dataset class in the global ``DATASETS`` registry and
    instantiates it with the provided keyword arguments.

    Parameters
    ----------
    name : str
        Registered name of the dataset (e.g., ``"nuscenes"``, ``"kitti"``).
    **kwargs
        Keyword arguments passed to the dataset constructor.

    Returns
    -------
    Dataset instance
        An instantiated dataset (typically a ``torch.utils.data.Dataset``).

    Raises
    ------
    common.registry.NotRegisteredError
        If ``name`` is not registered in the DATASETS registry.
    TypeError
        If required constructor arguments are missing from ``kwargs``.

    Examples
    --------
    >>> from common.datasets import build_dataset
    >>> ds = build_dataset("kitti", dataroot="/data/kitti", split="train")
    >>> len(ds)
    3712
    """
    dataset_cls = DATASETS.get(name)
    return dataset_cls(**kwargs)


def list_datasets() -> list[str]:
    """Return a sorted list of all registered dataset names.

    Returns
    -------
    list of str
        Available dataset names that can be passed to ``build_dataset()``.

    Examples
    --------
    >>> from common.datasets import list_datasets
    >>> list_datasets()
    ['kitti', 'nuscenes']
    """
    return DATASETS.list()
