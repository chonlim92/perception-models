"""
Inference pipeline for RadarPillarNet 3D object detection from radar point clouds.

Provides single-frame and batch inference modes with full pre/post-processing:
    1. Multi-sweep accumulation (ego-motion compensated)
    2. Clutter filtering
    3. Pillarization
    4. Model forward pass with NMS
    5. Output formatting (nuScenes JSON or CSV)

Usage:
    python inference.py --checkpoint model.pth --input sweeps/ --output results.json

    # Batch mode with custom thresholds
    python inference.py --checkpoint model.pth --input sweeps/ --output results.csv \
        --batch_size 4 --score_threshold 0.3 --nms_threshold 0.3 --output_format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .model import RadarPillarNet
from .pillar_encoder import create_pillars
from .radar_preprocessing import (
    RadarClutterFilter,
    RadarSweep,
    accumulate_sweeps,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUSCENES_CLASSES: List[str] = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]

DEFAULT_POINT_RANGE: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
DEFAULT_PILLAR_SIZE: List[float] = [0.4, 0.4, 8.0]
DEFAULT_MAX_POINTS_PER_PILLAR: int = 20
DEFAULT_MAX_PILLARS: int = 12000
DEFAULT_NUM_SWEEPS: int = 6
DEFAULT_WARMUP_ITERATIONS: int = 5


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_radar_sweeps_from_directory(
    sweep_dir: Path,
    num_sweeps: int = DEFAULT_NUM_SWEEPS,
) -> Tuple[RadarSweep, List[RadarSweep]]:
    """Load radar sweep data from a directory of .npy or .bin files.

    Expects files named with timestamps or sequential indices. Each file contains
    an (N, 6) array of [x, y, z, RCS, vr_compensated, vr_raw]. An optional
    metadata JSON alongside each sweep provides timestamp and ego_pose.

    Args:
        sweep_dir: Directory containing sweep files (.npy or .bin).
        num_sweeps: Total number of sweeps to load (current + history).

    Returns:
        Tuple of (current_sweep, history_sweeps) suitable for accumulate_sweeps().

    Raises:
        FileNotFoundError: If sweep_dir does not exist or contains no valid files.
        ValueError: If sweep files have unexpected shape.
    """
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    # Collect sweep files sorted by name (most recent last by convention)
    extensions = (".npy", ".bin")
    sweep_files = sorted(
        [f for f in sweep_dir.iterdir() if f.suffix in extensions],
        key=lambda p: p.stem,
    )

    if not sweep_files:
        raise FileNotFoundError(
            f"No .npy or .bin files found in {sweep_dir}"
        )

    # Limit to num_sweeps most recent files
    sweep_files = sweep_files[-num_sweeps:]

    sweeps: List[RadarSweep] = []
    for sweep_file in sweep_files:
        # Load points
        if sweep_file.suffix == ".npy":
            points = np.load(str(sweep_file)).astype(np.float32)
        else:
            # Binary format: float32, 6 columns
            points = np.fromfile(str(sweep_file), dtype=np.float32).reshape(-1, 6)

        if points.ndim != 2 or points.shape[1] < 6:
            raise ValueError(
                f"Expected (N, 6) array in {sweep_file}, got shape {points.shape}"
            )
        points = points[:, :6]

        # Load metadata if available
        meta_file = sweep_file.with_suffix(".json")
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            timestamp = float(meta.get("timestamp", 0.0))
            ego_pose = np.array(meta["ego_pose"], dtype=np.float64).reshape(4, 4)
        else:
            # Fallback: use file index as pseudo-timestamp, identity pose
            timestamp = float(sweep_files.index(sweep_file)) * 0.05  # 50ms intervals
            ego_pose = np.eye(4, dtype=np.float64)

        sweeps.append(RadarSweep(points=points, timestamp=timestamp, ego_pose=ego_pose))

    # The last sweep is current, the rest are history (reverse order: most recent first)
    current_sweep = sweeps[-1]
    history_sweeps = list(reversed(sweeps[:-1]))

    return current_sweep, history_sweeps


def load_single_point_cloud(file_path: Path) -> np.ndarray:
    """Load a single pre-accumulated point cloud file.

    Supports .npy and .bin formats. Expected shape: (N, 7) with columns
    [x, y, z, RCS, vr_compensated, vr_raw, dt].

    Args:
        file_path: Path to the point cloud file.

    Returns:
        (N, 7) float32 array of accumulated radar points.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file has unexpected shape.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    if file_path.suffix == ".npy":
        points = np.load(str(file_path)).astype(np.float32)
    else:
        points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 7)

    if points.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {points.shape}")

    if points.shape[1] == 6:
        # No time delta column; append zeros (single-sweep assumption)
        points = np.column_stack(
            [points, np.zeros(points.shape[0], dtype=np.float32)]
        )
    elif points.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns in {file_path}, got {points.shape[1]}"
        )

    return points[:, :7]


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────


class InferencePreprocessor:
    """Pre-processing pipeline for RadarPillarNet inference.

    Handles multi-sweep accumulation, clutter filtering, and pillarization.
    All configuration parameters match the training pipeline defaults.
    """

    def __init__(
        self,
        point_range: List[float] = DEFAULT_POINT_RANGE,
        pillar_size: List[float] = DEFAULT_PILLAR_SIZE,
        max_points_per_pillar: int = DEFAULT_MAX_POINTS_PER_PILLAR,
        max_pillars: int = DEFAULT_MAX_PILLARS,
        num_sweeps: int = DEFAULT_NUM_SWEEPS,
        enable_clutter_filter: bool = True,
    ) -> None:
        """Initialize the inference preprocessor.

        Args:
            point_range: [x_min, y_min, z_min, x_max, y_max, z_max] detection range.
            pillar_size: [dx, dy, dz] size of each pillar in meters.
            max_points_per_pillar: Maximum number of points per pillar.
            max_pillars: Maximum number of non-empty pillars to keep.
            num_sweeps: Number of radar sweeps to accumulate.
            enable_clutter_filter: Whether to apply radar clutter filtering.
        """
        self.point_range = point_range
        self.pillar_size = pillar_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        self.num_sweeps = num_sweeps
        self.enable_clutter_filter = enable_clutter_filter

        if enable_clutter_filter:
            self.clutter_filter = RadarClutterFilter()

    def preprocess_sweeps(
        self,
        current_sweep: RadarSweep,
        history_sweeps: List[RadarSweep],
    ) -> Dict[str, np.ndarray]:
        """Full preprocessing from raw sweeps to model-ready pillar tensors.

        Steps:
            1. Multi-sweep accumulation with ego-motion compensation
            2. Optional clutter filtering
            3. Feature selection: [x, y, z, RCS, vr_compensated, dt]
            4. Pillarization via create_pillars()

        Args:
            current_sweep: The current (most recent) radar sweep.
            history_sweeps: Historical sweeps ordered most-recent first.

        Returns:
            Dictionary with keys:
                'pillars': (max_pillars, max_points_per_pillar, 9) float32
                'pillar_indices': (max_pillars, 3) int32
                'num_points_per_pillar': (max_pillars,) int32
        """
        # Step 1: Multi-sweep accumulation
        accumulated = accumulate_sweeps(
            current_sweep=current_sweep,
            history_sweeps=history_sweeps,
            num_sweeps=self.num_sweeps,
        )  # (M, 7): [x, y, z, RCS, vr_compensated, vr_raw, dt]

        return self.preprocess_points(accumulated)

    def preprocess_points(
        self,
        points: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Preprocess an already-accumulated point cloud into pillars.

        Args:
            points: (N, 7) array [x, y, z, RCS, vr_compensated, vr_raw, dt].

        Returns:
            Dictionary with pillar tensors ready for model input.
        """
        # Step 2: Clutter filtering (operates on accumulated points)
        if self.enable_clutter_filter:
            points = self.clutter_filter.filter(points)

        # Step 3: Select model input features
        # From [x, y, z, RCS, vr_compensated, vr_raw, dt] -> [x, y, z, RCS, vr_compensated, dt]
        # Indices: 0, 1, 2, 3, 4, 6
        model_points = points[:, [0, 1, 2, 3, 4, 6]]  # (N, 6)

        # create_pillars expects (N, 7) but only uses first 6 columns as raw features.
        # Pad with a dummy column to satisfy the interface.
        model_points_padded = np.column_stack(
            [model_points, np.zeros(model_points.shape[0], dtype=np.float32)]
        )  # (N, 7)

        # Step 4: Pillarization
        pillars, pillar_indices, num_points_per_pillar = create_pillars(
            points=model_points_padded,
            point_range=self.point_range,
            pillar_size=self.pillar_size,
            max_points_per_pillar=self.max_points_per_pillar,
            max_pillars=self.max_pillars,
        )

        return {
            "pillars": pillars,
            "pillar_indices": pillar_indices,
            "num_points_per_pillar": num_points_per_pillar,
        }

    def collate_batch(
        self,
        samples: List[Dict[str, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """Collate multiple preprocessed samples into a batched tensor dict.

        Args:
            samples: List of dictionaries from preprocess_sweeps or preprocess_points.

        Returns:
            Batched dictionary with torch tensors on CPU:
                'pillars': (B, max_pillars, max_points_per_pillar, 9)
                'pillar_indices': (B, max_pillars, 3)
                'num_points_per_pillar': (B, max_pillars)
        """
        batch_pillars = np.stack([s["pillars"] for s in samples], axis=0)
        batch_indices = np.stack([s["pillar_indices"] for s in samples], axis=0)
        batch_num_points = np.stack(
            [s["num_points_per_pillar"] for s in samples], axis=0
        )

        return {
            "pillars": torch.from_numpy(batch_pillars),
            "pillar_indices": torch.from_numpy(batch_indices),
            "num_points_per_pillar": torch.from_numpy(batch_num_points),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────


class InferencePostprocessor:
    """Post-processing for RadarPillarNet detections.

    Applies per-class score filtering and formats outputs for downstream use.
    NMS is already handled within the model's predict() method; this class
    provides additional score thresholding and output formatting.
    """

    def __init__(
        self,
        score_threshold: float = 0.1,
        class_names: List[str] = NUSCENES_CLASSES,
        per_class_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize postprocessor.

        Args:
            score_threshold: Global minimum detection score (default 0.1).
            class_names: Ordered list of class names matching model output labels.
            per_class_thresholds: Optional per-class score thresholds. If provided,
                overrides global threshold for specified classes. Example:
                {'pedestrian': 0.3, 'traffic_cone': 0.4}
        """
        self.score_threshold = score_threshold
        self.class_names = class_names
        self.per_class_thresholds = per_class_thresholds or {}

    def filter_detections(
        self,
        detections: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """Apply per-class score filtering to raw model detections.

        Args:
            detections: Dictionary from model.predict() containing:
                'boxes': (K, 7) tensor [x, y, z, w, l, h, yaw]
                'scores': (K,) tensor
                'labels': (K,) tensor (0-indexed class IDs)
                'velocities': (K, 2) tensor [vx, vy]

        Returns:
            Filtered detections as numpy arrays:
                'boxes': (M, 7) array
                'scores': (M,) array
                'labels': (M,) array
                'velocities': (M, 2) array
                'class_names': list of M class name strings
        """
        boxes = detections["boxes"].cpu().numpy()
        scores = detections["scores"].cpu().numpy()
        labels = detections["labels"].cpu().numpy()
        velocities = detections["velocities"].cpu().numpy()

        # Build per-detection threshold mask
        keep_mask = np.zeros(len(scores), dtype=bool)
        for i in range(len(scores)):
            class_idx = int(labels[i])
            class_name = (
                self.class_names[class_idx]
                if class_idx < len(self.class_names)
                else f"class_{class_idx}"
            )
            threshold = self.per_class_thresholds.get(
                class_name, self.score_threshold
            )
            if scores[i] >= threshold:
                keep_mask[i] = True

        # Apply mask
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = labels[keep_mask]
        velocities = velocities[keep_mask]

        # Map labels to class names
        class_name_list = []
        for label in labels:
            idx = int(label)
            if idx < len(self.class_names):
                class_name_list.append(self.class_names[idx])
            else:
                class_name_list.append(f"class_{idx}")

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "velocities": velocities,
            "class_names": class_name_list,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Output formatting
# ──────────────────────────────────────────────────────────────────────────────


def format_nuscenes_submission(
    detections_per_sample: List[Dict[str, Any]],
    sample_tokens: List[str],
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Format detections into nuScenes detection submission JSON format.

    Follows the nuScenes detection challenge submission format:
    https://www.nuscenes.org/object-detection

    Args:
        detections_per_sample: List of filtered detection dicts (one per sample).
        sample_tokens: nuScenes sample tokens corresponding to each detection set.
        metadata: Optional submission metadata (use_camera, use_lidar, use_radar, etc.).

    Returns:
        Dictionary in nuScenes submission format, ready for json.dump().
    """
    if metadata is None:
        metadata = {
            "use_camera": False,
            "use_lidar": False,
            "use_radar": True,
            "use_map": False,
            "use_external": False,
        }

    results: Dict[str, List[Dict[str, Any]]] = {}

    for sample_token, dets in zip(sample_tokens, detections_per_sample):
        sample_results: List[Dict[str, Any]] = []

        boxes = dets["boxes"]
        scores = dets["scores"]
        velocities = dets["velocities"]
        class_names = dets["class_names"]

        for i in range(len(scores)):
            x, y, z, w, l, h, yaw = boxes[i]

            # nuScenes expects quaternion for rotation (yaw around z-axis)
            # quaternion = [w, x, y, z] where rotation is around z
            qw = float(np.cos(yaw / 2.0))
            qx = 0.0
            qy = 0.0
            qz = float(np.sin(yaw / 2.0))

            detection_entry = {
                "sample_token": sample_token,
                "translation": [float(x), float(y), float(z)],
                "size": [float(w), float(l), float(h)],
                "rotation": [qw, qx, qy, qz],
                "velocity": [float(velocities[i, 0]), float(velocities[i, 1])],
                "detection_name": class_names[i],
                "detection_score": float(scores[i]),
                "attribute_name": _get_default_attribute(class_names[i]),
            }
            sample_results.append(detection_entry)

        results[sample_token] = sample_results

    return {"meta": metadata, "results": results}


def _get_default_attribute(class_name: str) -> str:
    """Get default nuScenes attribute for a detection class.

    Args:
        class_name: Detection class name.

    Returns:
        Default attribute string for the class.
    """
    # nuScenes requires attributes for certain classes
    vehicle_classes = {"car", "truck", "bus", "trailer", "construction_vehicle"}
    cycle_classes = {"motorcycle", "bicycle"}

    if class_name in vehicle_classes:
        return "vehicle.moving"
    elif class_name in cycle_classes:
        return "cycle.with_rider"
    elif class_name == "pedestrian":
        return "pedestrian.moving"
    else:
        return ""


def format_csv_output(
    detections: Dict[str, np.ndarray],
    sample_id: str = "",
) -> List[Dict[str, str]]:
    """Format detections as flat CSV rows.

    Args:
        detections: Filtered detection dict with boxes, scores, labels, velocities.
        sample_id: Optional identifier for the input sample.

    Returns:
        List of dictionaries suitable for csv.DictWriter.
    """
    rows: List[Dict[str, str]] = []

    boxes = detections["boxes"]
    scores = detections["scores"]
    velocities = detections["velocities"]
    class_names = detections["class_names"]

    for i in range(len(scores)):
        x, y, z, w, l, h, yaw = boxes[i]
        row = {
            "sample_id": sample_id,
            "class_name": class_names[i],
            "score": f"{scores[i]:.4f}",
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "z": f"{z:.3f}",
            "width": f"{w:.3f}",
            "length": f"{l:.3f}",
            "height": f"{h:.3f}",
            "yaw": f"{yaw:.4f}",
            "vx": f"{velocities[i, 0]:.3f}",
            "vy": f"{velocities[i, 1]:.3f}",
        }
        rows.append(row)

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main inference engine
# ──────────────────────────────────────────────────────────────────────────────


class RadarPillarNetInference:
    """End-to-end inference engine for RadarPillarNet.

    Manages model loading, pre/post-processing, and timing measurement.
    Supports both single-frame and batched inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        score_threshold: float = 0.1,
        nms_threshold: float = 0.2,
        class_names: List[str] = NUSCENES_CLASSES,
        per_class_thresholds: Optional[Dict[str, float]] = None,
        point_range: List[float] = DEFAULT_POINT_RANGE,
        pillar_size: List[float] = DEFAULT_PILLAR_SIZE,
        max_points_per_pillar: int = DEFAULT_MAX_POINTS_PER_PILLAR,
        max_pillars: int = DEFAULT_MAX_PILLARS,
        num_sweeps: int = DEFAULT_NUM_SWEEPS,
        enable_clutter_filter: bool = True,
    ) -> None:
        """Initialize the inference engine.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file).
            device: Compute device ('cuda', 'cuda:0', 'cpu').
            score_threshold: Global minimum detection score.
            nms_threshold: IoU threshold for NMS (overrides model default).
            class_names: Ordered list of class names.
            per_class_thresholds: Optional per-class score thresholds.
            point_range: Detection range [x_min, y_min, z_min, x_max, y_max, z_max].
            pillar_size: Pillar dimensions [dx, dy, dz].
            max_points_per_pillar: Maximum points per pillar.
            max_pillars: Maximum non-empty pillars.
            num_sweeps: Number of sweeps to accumulate.
            enable_clutter_filter: Whether to apply clutter filtering.
        """
        self.device = torch.device(device)
        self.class_names = class_names
        self.nms_threshold = nms_threshold

        # Load model
        self.model = self._load_model(checkpoint_path, nms_threshold, score_threshold)
        self.model.to(self.device)
        self.model.eval()

        # Initialize preprocessor
        self.preprocessor = InferencePreprocessor(
            point_range=point_range,
            pillar_size=pillar_size,
            max_points_per_pillar=max_points_per_pillar,
            max_pillars=max_pillars,
            num_sweeps=num_sweeps,
            enable_clutter_filter=enable_clutter_filter,
        )

        # Initialize postprocessor
        self.postprocessor = InferencePostprocessor(
            score_threshold=score_threshold,
            class_names=class_names,
            per_class_thresholds=per_class_thresholds,
        )

        logger.info(
            "RadarPillarNet inference engine initialized on %s", self.device
        )

    def _load_model(
        self,
        checkpoint_path: str,
        nms_threshold: float,
        score_threshold: float,
    ) -> RadarPillarNet:
        """Load model from checkpoint.

        Supports checkpoints saved as:
            - Full state dict: {'model_state_dict': ..., 'config': ...}
            - Plain state dict (raw OrderedDict)

        Args:
            checkpoint_path: Path to the .pth checkpoint file.
            nms_threshold: NMS IoU threshold to set on the loaded model.
            score_threshold: Score threshold to set on the loaded model.

        Returns:
            Loaded and configured RadarPillarNet model.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            RuntimeError: If checkpoint cannot be loaded or is incompatible.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(
            str(ckpt_path), map_location=self.device, weights_only=False
        )

        # Determine checkpoint format
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("config", {})
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            config = checkpoint.get("config", {})
        else:
            # Assume raw state dict
            state_dict = checkpoint
            config = {}

        # Build model with config from checkpoint or defaults
        model = RadarPillarNet(
            in_channels=config.get("in_channels", 9),
            pillar_feat_channels=config.get("pillar_feat_channels", 64),
            x_range=tuple(config.get("x_range", (-51.2, 51.2))),
            y_range=tuple(config.get("y_range", (-51.2, 51.2))),
            z_range=tuple(config.get("z_range", (-5.0, 3.0))),
            pillar_size=tuple(config.get("pillar_size", (0.4, 0.4, 8.0))),
            max_points_per_pillar=config.get("max_points_per_pillar", 20),
            max_pillars=config.get("max_pillars", 12000),
            num_classes=config.get("num_classes", len(self.class_names)),
            nms_threshold=nms_threshold,
            score_threshold=score_threshold,
        )

        # Load weights
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model weights loaded successfully")

        return model

    @torch.no_grad()
    def infer_single(
        self,
        current_sweep: RadarSweep,
        history_sweeps: Optional[List[RadarSweep]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference on a single frame (multi-sweep input).

        Args:
            current_sweep: The current radar sweep.
            history_sweeps: Optional list of historical sweeps.

        Returns:
            Filtered detection dict with keys: boxes, scores, labels, velocities,
            class_names.
        """
        if history_sweeps is None:
            history_sweeps = []

        # Preprocess
        sample = self.preprocessor.preprocess_sweeps(current_sweep, history_sweeps)
        batch = self.preprocessor.collate_batch([sample])

        # Move to device
        batch_device = {
            k: v.to(self.device) for k, v in batch.items()
        }

        # Forward pass
        predictions = self.model.predict(batch_device)

        # Post-process first (and only) batch element
        detections = self.postprocessor.filter_detections(predictions[0])

        return detections

    @torch.no_grad()
    def infer_from_points(
        self,
        points: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Run inference on a pre-accumulated point cloud.

        Args:
            points: (N, 7) array [x, y, z, RCS, vr_compensated, vr_raw, dt].

        Returns:
            Filtered detection dict.
        """
        sample = self.preprocessor.preprocess_points(points)
        batch = self.preprocessor.collate_batch([sample])

        batch_device = {k: v.to(self.device) for k, v in batch.items()}

        predictions = self.model.predict(batch_device)
        detections = self.postprocessor.filter_detections(predictions[0])

        return detections

    @torch.no_grad()
    def infer_batch(
        self,
        point_clouds: List[np.ndarray],
        batch_size: int = 4,
    ) -> List[Dict[str, np.ndarray]]:
        """Run batched inference on multiple pre-accumulated point clouds.

        Processes point clouds in mini-batches for GPU efficiency.

        Args:
            point_clouds: List of (N_i, 7) arrays, each a pre-accumulated point cloud.
            batch_size: Number of samples per forward pass.

        Returns:
            List of filtered detection dicts, one per input point cloud.
        """
        all_detections: List[Dict[str, np.ndarray]] = []
        num_samples = len(point_clouds)

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_points = point_clouds[batch_start:batch_end]

            # Preprocess each sample in the batch
            samples = [
                self.preprocessor.preprocess_points(pc) for pc in batch_points
            ]
            batch = self.preprocessor.collate_batch(samples)

            # Move to device
            batch_device = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            predictions = self.model.predict(batch_device)

            # Post-process each sample in the batch
            for pred in predictions:
                detections = self.postprocessor.filter_detections(pred)
                all_detections.append(detections)

        return all_detections

    def benchmark(
        self,
        points: np.ndarray,
        num_iterations: int = 100,
        warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    ) -> Dict[str, float]:
        """Measure inference timing with warmup iterations.

        Runs the full pipeline (preprocess + forward + postprocess) multiple times
        and reports timing statistics.

        Args:
            points: (N, 7) sample point cloud for benchmarking.
            num_iterations: Number of timed iterations.
            warmup_iterations: Number of untimed warmup iterations for GPU caching.

        Returns:
            Dictionary with timing statistics:
                'preprocess_ms': Average preprocessing time in milliseconds.
                'inference_ms': Average model inference time in milliseconds.
                'postprocess_ms': Average post-processing time in milliseconds.
                'total_ms': Average total pipeline time in milliseconds.
                'fps': Frames per second based on total time.
        """
        logger.info(
            "Running benchmark: %d warmup + %d timed iterations",
            warmup_iterations,
            num_iterations,
        )

        # Warmup
        for _ in range(warmup_iterations):
            _ = self.infer_from_points(points)

        # Synchronize GPU before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        preprocess_times: List[float] = []
        inference_times: List[float] = []
        postprocess_times: List[float] = []

        for _ in range(num_iterations):
            # Preprocess timing
            t0 = time.perf_counter()
            sample = self.preprocessor.preprocess_points(points)
            batch = self.preprocessor.collate_batch([sample])
            batch_device = {k: v.to(self.device) for k, v in batch.items()}
            t1 = time.perf_counter()

            # Inference timing
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            predictions = self.model.predict(batch_device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t2 = time.perf_counter()

            # Postprocess timing
            _ = self.postprocessor.filter_detections(predictions[0])
            t3 = time.perf_counter()

            preprocess_times.append((t1 - t0) * 1000.0)
            inference_times.append((t2 - t1) * 1000.0)
            postprocess_times.append((t3 - t2) * 1000.0)

        avg_preprocess = float(np.mean(preprocess_times))
        avg_inference = float(np.mean(inference_times))
        avg_postprocess = float(np.mean(postprocess_times))
        avg_total = avg_preprocess + avg_inference + avg_postprocess
        fps = 1000.0 / avg_total if avg_total > 0 else 0.0

        results = {
            "preprocess_ms": avg_preprocess,
            "inference_ms": avg_inference,
            "postprocess_ms": avg_postprocess,
            "total_ms": avg_total,
            "fps": fps,
        }

        logger.info(
            "Benchmark results: preprocess=%.2fms, inference=%.2fms, "
            "postprocess=%.2fms, total=%.2fms, FPS=%.1f",
            avg_preprocess,
            avg_inference,
            avg_postprocess,
            avg_total,
            fps,
        )

        return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def _discover_input_samples(input_path: Path) -> Tuple[List[Path], bool]:
    """Discover input samples from the given path.

    Args:
        input_path: Path to a single file or directory of samples.

    Returns:
        Tuple of (list of sample paths, is_sweep_directories) where
        is_sweep_directories indicates whether each path is a directory
        of sweeps (True) or a single accumulated point cloud file (False).

    Raises:
        FileNotFoundError: If input_path does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        return [input_path], False

    # Directory: check if it contains subdirectories (multi-sample sweep dirs)
    # or point cloud files directly
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    files = [
        f for f in input_path.iterdir() if f.suffix in (".npy", ".bin")
    ]

    if subdirs and not files:
        # Each subdir is a sweep directory for one sample
        return sorted(subdirs), True
    elif files:
        # Directory of pre-accumulated point cloud files
        return sorted(files), False
    else:
        raise FileNotFoundError(
            f"No valid input files or subdirectories found in {input_path}"
        )


def _write_nuscenes_output(
    all_detections: List[Dict[str, Any]],
    sample_paths: List[Path],
    output_path: Path,
) -> None:
    """Write detections in nuScenes submission JSON format.

    Args:
        all_detections: List of filtered detection dicts.
        sample_paths: Paths used as sample token proxies.
        output_path: Output JSON file path.
    """
    sample_tokens = [p.stem for p in sample_paths]
    submission = format_nuscenes_submission(all_detections, sample_tokens)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    logger.info("nuScenes submission written to %s", output_path)


def _write_csv_output(
    all_detections: List[Dict[str, Any]],
    sample_paths: List[Path],
    output_path: Path,
) -> None:
    """Write detections in CSV format.

    Args:
        all_detections: List of filtered detection dicts.
        sample_paths: Paths used for sample IDs.
        output_path: Output CSV file path.
    """
    fieldnames = [
        "sample_id",
        "class_name",
        "score",
        "x",
        "y",
        "z",
        "width",
        "length",
        "height",
        "yaw",
        "vx",
        "vy",
    ]

    all_rows: List[Dict[str, str]] = []
    for dets, path in zip(all_detections, sample_paths):
        rows = format_csv_output(dets, sample_id=path.stem)
        all_rows.extend(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("CSV output written to %s (%d detections)", output_path, len(all_rows))


def main() -> None:
    """Main entry point for command-line inference."""
    parser = argparse.ArgumentParser(
        description="RadarPillarNet inference: 3D object detection from radar point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Input path. Can be: (1) a single .npy/.bin point cloud file, "
            "(2) a directory of .npy/.bin files (batch mode), or "
            "(3) a directory of subdirectories each containing multi-sweep files."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.json for nuScenes format, .csv for CSV format).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="Minimum detection score threshold.",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.2,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["nuscenes", "csv"],
        default=None,
        help=(
            "Output format. If not specified, inferred from output file extension "
            "(.json -> nuscenes, .csv -> csv)."
        ),
    )
    parser.add_argument(
        "--num_sweeps",
        type=int,
        default=DEFAULT_NUM_SWEEPS,
        help="Number of radar sweeps to accumulate.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run timing benchmark and print FPS statistics.",
    )
    parser.add_argument(
        "--benchmark_iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarking.",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
        help="Number of warmup iterations before benchmarking.",
    )
    parser.add_argument(
        "--no_clutter_filter",
        action="store_true",
        help="Disable radar clutter filtering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Determine output format
    output_path = Path(args.output)
    if args.output_format is not None:
        output_format = args.output_format
    elif output_path.suffix == ".json":
        output_format = "nuscenes"
    elif output_path.suffix == ".csv":
        output_format = "csv"
    else:
        output_format = "nuscenes"
        logger.warning(
            "Could not infer output format from extension '%s', defaulting to nuScenes JSON",
            output_path.suffix,
        )

    # Initialize inference engine
    engine = RadarPillarNetInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        num_sweeps=args.num_sweeps,
        enable_clutter_filter=not args.no_clutter_filter,
    )

    # Discover input samples
    input_path = Path(args.input)
    sample_paths, is_sweep_dirs = _discover_input_samples(input_path)
    logger.info("Found %d input sample(s)", len(sample_paths))

    # Run benchmark if requested
    if args.benchmark:
        if is_sweep_dirs:
            current_sweep, history_sweeps = load_radar_sweeps_from_directory(
                sample_paths[0], num_sweeps=args.num_sweeps
            )
            test_points = accumulate_sweeps(
                current_sweep, history_sweeps, num_sweeps=args.num_sweeps
            )
        else:
            test_points = load_single_point_cloud(sample_paths[0])

        timing = engine.benchmark(
            test_points,
            num_iterations=args.benchmark_iterations,
            warmup_iterations=args.warmup_iterations,
        )
        print("\n=== Benchmark Results ===")
        print(f"  Preprocessing: {timing['preprocess_ms']:.2f} ms")
        print(f"  Inference:     {timing['inference_ms']:.2f} ms")
        print(f"  Postprocessing:{timing['postprocess_ms']:.2f} ms")
        print(f"  Total:         {timing['total_ms']:.2f} ms")
        print(f"  FPS:           {timing['fps']:.1f}")
        print()

    # Run inference
    all_detections: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    if is_sweep_dirs:
        # Each sample is a directory of sweep files
        for sample_dir in sample_paths:
            current_sweep, history_sweeps = load_radar_sweeps_from_directory(
                sample_dir, num_sweeps=args.num_sweeps
            )
            detections = engine.infer_single(current_sweep, history_sweeps)
            all_detections.append(detections)
            logger.debug(
                "Sample %s: %d detections", sample_dir.stem, len(detections["scores"])
            )
    else:
        # Pre-accumulated point cloud files
        point_clouds = [load_single_point_cloud(p) for p in sample_paths]

        if args.batch_size > 1 and len(point_clouds) > 1:
            all_detections = engine.infer_batch(
                point_clouds, batch_size=args.batch_size
            )
        else:
            for pc in point_clouds:
                detections = engine.infer_from_points(pc)
                all_detections.append(detections)

    total_elapsed = time.perf_counter() - total_start
    total_detections = sum(len(d["scores"]) for d in all_detections)

    logger.info(
        "Inference complete: %d samples, %d total detections in %.2fs",
        len(sample_paths),
        total_detections,
        total_elapsed,
    )

    # Write output
    if output_format == "nuscenes":
        _write_nuscenes_output(all_detections, sample_paths, output_path)
    else:
        _write_csv_output(all_detections, sample_paths, output_path)

    print(
        f"Done. Processed {len(sample_paths)} sample(s), "
        f"{total_detections} detection(s) -> {output_path}"
    )


if __name__ == "__main__":
    main()
