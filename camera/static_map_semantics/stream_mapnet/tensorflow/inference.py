"""
StreamMapNet TensorFlow 2 Inference Script.

Handles loading a trained StreamMapNet model and running inference on input
sequences with temporal state propagation, benchmarking, export, and
visualization capabilities.

Usage:
    python inference.py infer --model-path /path/to/model --input-path /path/to/data
    python inference.py benchmark --model-path /path/to/model --batch-size 4
    python inference.py export --model-path /path/to/model --output-dir /path/to/export
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_BEV_HEIGHT = 200
DEFAULT_BEV_WIDTH = 100
DEFAULT_BEV_CHANNELS = 64
DEFAULT_NUM_CLASSES = 3  # lane, crossing, boundary
DEFAULT_NUM_POINTS = 20  # points per polyline
DEFAULT_MAX_INSTANCES = 50
DEFAULT_SEQUENCE_LENGTH = 8
DEFAULT_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_BENCHMARK_ITERATIONS = 100

CLASS_NAMES = {0: "lane", 1: "crossing", 2: "boundary"}
CLASS_COLORS = {
    0: (0, 0, 255),    # lane = blue (BGR for cv2 compat, RGB for PIL)
    1: (0, 255, 0),    # crossing = green
    2: (255, 0, 0),    # boundary = red
}


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def detect_model_format(model_path: str) -> str:
    """Auto-detect whether the path is a SavedModel or a checkpoint.

    Returns:
        'savedmodel' if path contains saved_model.pb or is a SavedModel dir.
        'checkpoint' if path is a checkpoint prefix or directory with .index files.
    """
    path = Path(model_path)

    # SavedModel detection
    if path.is_dir():
        if (path / "saved_model.pb").exists():
            return "savedmodel"
        # Check for checkpoint files
        if (path / "checkpoint").exists():
            return "checkpoint"
        # Look for .index files
        index_files = list(path.glob("*.index"))
        if index_files:
            return "checkpoint"

    # If path points to a specific file
    if path.suffix == ".pb" and path.name == "saved_model.pb":
        return "savedmodel"
    if path.with_suffix(".index").exists():
        return "checkpoint"

    # Check parent directory for SavedModel structure
    if path.parent.is_dir() and (path.parent / "saved_model.pb").exists():
        return "savedmodel"

    raise ValueError(
        f"Cannot determine model format for path: {model_path}. "
        "Expected a SavedModel directory or checkpoint prefix."
    )


def build_stream_mapnet_model(
    bev_height: int = DEFAULT_BEV_HEIGHT,
    bev_width: int = DEFAULT_BEV_WIDTH,
    bev_channels: int = DEFAULT_BEV_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    num_points: int = DEFAULT_NUM_POINTS,
    max_instances: int = DEFAULT_MAX_INSTANCES,
) -> tf.keras.Model:
    """Reconstruct the StreamMapNet architecture for checkpoint loading.

    This builds a simplified version of the model suitable for inference.
    The model takes camera images and temporal BEV state as input, producing
    map element predictions.
    """
    # Input layers
    camera_input = tf.keras.Input(
        shape=(6, 224, 480, 3), name="camera_images"
    )  # 6 surround cameras
    temporal_bev_input = tf.keras.Input(
        shape=(bev_height, bev_width, bev_channels), name="temporal_bev_state"
    )
    ego_motion_input = tf.keras.Input(
        shape=(4, 4), name="ego_motion"
    )  # 4x4 transformation matrix

    # Image backbone (simplified EfficientNet-style)
    # Process each camera view
    cam_features_list = []
    shared_backbone = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
    ], name="image_backbone")

    # Process cameras (unrolled for tracing compatibility)
    for i in range(6):
        cam_i = camera_input[:, i, :, :, :]
        feat_i = shared_backbone(cam_i)
        cam_features_list.append(feat_i)

    cam_features = tf.stack(cam_features_list, axis=1)  # (B, 6, 256)
    cam_features_flat = tf.keras.layers.Reshape((6 * 256,))(cam_features)

    # BEV projection
    bev_projection = tf.keras.layers.Dense(
        bev_height * bev_width * bev_channels // 4, activation="relu"
    )(cam_features_flat)
    bev_projection = tf.keras.layers.Reshape(
        (bev_height // 2, bev_width // 2, bev_channels)
    )(bev_projection)
    bev_current = tf.keras.layers.Conv2DTranspose(
        bev_channels, 3, strides=2, padding="same", activation="relu"
    )(bev_projection)

    # Temporal fusion with ego-motion compensation
    ego_flat = tf.keras.layers.Flatten()(ego_motion_input)
    ego_scale = tf.keras.layers.Dense(bev_channels, activation="sigmoid")(ego_flat)
    ego_scale = tf.keras.layers.Reshape((1, 1, bev_channels))(ego_scale)
    temporal_warped = temporal_bev_input * ego_scale

    # Fuse current BEV with temporal state
    fused_bev = tf.keras.layers.Concatenate(axis=-1)(
        [bev_current, temporal_warped]
    )
    fused_bev = tf.keras.layers.Conv2D(
        bev_channels, 3, padding="same", activation="relu"
    )(fused_bev)
    fused_bev = tf.keras.layers.BatchNormalization()(fused_bev)
    updated_bev_state = tf.keras.layers.Conv2D(
        bev_channels, 3, padding="same", activation="relu", name="updated_bev_state"
    )(fused_bev)

    # Map element decoder head
    decoder_pool = tf.keras.layers.GlobalAveragePooling2D()(updated_bev_state)

    # Instance predictions
    instance_logits = tf.keras.layers.Dense(512, activation="relu")(decoder_pool)
    instance_logits = tf.keras.layers.Dense(256, activation="relu")(instance_logits)

    # Class predictions: (batch, max_instances, num_classes)
    class_logits = tf.keras.layers.Dense(
        max_instances * num_classes, name="class_logits_raw"
    )(instance_logits)
    class_logits = tf.keras.layers.Reshape(
        (max_instances, num_classes), name="class_logits"
    )(class_logits)

    # Point predictions: (batch, max_instances, num_points, 2)
    point_preds = tf.keras.layers.Dense(
        max_instances * num_points * 2, name="point_preds_raw"
    )(instance_logits)
    point_preds = tf.keras.layers.Reshape(
        (max_instances, num_points, 2), name="point_predictions"
    )(point_preds)

    # Confidence scores: (batch, max_instances)
    confidence_logits = tf.keras.layers.Dense(
        max_instances, name="confidence_logits"
    )(instance_logits)

    model = tf.keras.Model(
        inputs=[camera_input, temporal_bev_input, ego_motion_input],
        outputs={
            "class_logits": class_logits,
            "point_predictions": point_preds,
            "confidence_logits": confidence_logits,
            "updated_bev_state": updated_bev_state,
        },
        name="StreamMapNet",
    )
    return model


def load_model(
    model_path: str,
    bev_height: int = DEFAULT_BEV_HEIGHT,
    bev_width: int = DEFAULT_BEV_WIDTH,
    bev_channels: int = DEFAULT_BEV_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    num_points: int = DEFAULT_NUM_POINTS,
    max_instances: int = DEFAULT_MAX_INSTANCES,
) -> Any:
    """Load model from checkpoint or SavedModel format.

    Args:
        model_path: Path to model checkpoint or SavedModel directory.
        bev_height: BEV grid height.
        bev_width: BEV grid width.
        bev_channels: Number of BEV feature channels.
        num_classes: Number of map element classes.
        num_points: Points per polyline prediction.
        max_instances: Maximum number of predicted instances.

    Returns:
        Loaded model (keras Model or SavedModel concrete function).
    """
    fmt = detect_model_format(model_path)
    logger.info("Detected model format: %s at %s", fmt, model_path)

    if fmt == "savedmodel":
        model = tf.saved_model.load(model_path)
        logger.info("Loaded SavedModel from %s", model_path)
        return model

    # Checkpoint format: reconstruct model then restore weights
    model = build_stream_mapnet_model(
        bev_height=bev_height,
        bev_width=bev_width,
        bev_channels=bev_channels,
        num_classes=num_classes,
        num_points=num_points,
        max_instances=max_instances,
    )

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = model_path

    # If path is a directory, find the latest checkpoint
    if Path(model_path).is_dir():
        latest = tf.train.latest_checkpoint(model_path)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint found in directory: {model_path}"
            )
        checkpoint_path = latest

    status = checkpoint.restore(checkpoint_path)
    # Allow partial restore (some optimizer variables may be missing)
    status.expect_partial()
    logger.info("Restored checkpoint from %s", checkpoint_path)

    return model


# ---------------------------------------------------------------------------
# Post-Processing Pipeline
# ---------------------------------------------------------------------------

def apply_softmax(logits: np.ndarray) -> np.ndarray:
    """Apply softmax to class logits."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def filter_by_confidence(
    class_probs: np.ndarray,
    confidence_scores: np.ndarray,
    point_predictions: np.ndarray,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter predictions below confidence threshold.

    Args:
        class_probs: (num_instances, num_classes) class probabilities.
        confidence_scores: (num_instances,) confidence scores.
        point_predictions: (num_instances, num_points, 2) polyline points.
        threshold: Minimum confidence to keep.

    Returns:
        Filtered (class_ids, class_scores, confidence_scores, points).
    """
    # Combine class probability with instance confidence
    class_ids = np.argmax(class_probs, axis=-1)
    class_scores = np.max(class_probs, axis=-1)
    combined_scores = class_scores * confidence_scores

    mask = combined_scores >= threshold
    return (
        class_ids[mask],
        combined_scores[mask],
        confidence_scores[mask],
        point_predictions[mask],
    )


def polyline_simplification(
    points: np.ndarray, epsilon: float = 0.5
) -> np.ndarray:
    """Simplify polyline using Ramer-Douglas-Peucker algorithm.

    Args:
        points: (num_points, 2) array of polyline vertices.
        epsilon: Maximum perpendicular distance for simplification.

    Returns:
        Simplified polyline points.
    """
    if len(points) <= 2:
        return points

    # Find point with maximum distance from line between first and last
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-8:
        return points[[0]]

    line_unit = line_vec / line_len
    # Perpendicular distances
    vecs = points - start
    projections = np.dot(vecs, line_unit)
    closest_points = start + np.outer(projections, line_unit)
    distances = np.linalg.norm(points - closest_points, axis=-1)

    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]

    if max_dist > epsilon:
        # Recurse on both halves
        left = polyline_simplification(points[: max_idx + 1], epsilon)
        right = polyline_simplification(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return points[[0, -1]]


def nms_deduplication(
    class_ids: np.ndarray,
    scores: np.ndarray,
    points: np.ndarray,
    distance_threshold: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NMS-like deduplication based on polyline proximity.

    For each pair of predictions of the same class, if their average point
    distance is below threshold, suppress the lower-scoring one.

    Args:
        class_ids: (N,) predicted class IDs.
        scores: (N,) combined scores.
        points: (N, num_points, 2) polyline points.
        distance_threshold: Max average distance to consider duplicates.

    Returns:
        Deduplicated (class_ids, scores, points).
    """
    if len(class_ids) == 0:
        return class_ids, scores, points

    # Sort by score descending
    order = np.argsort(-scores)
    keep = []

    suppressed = set()
    for i in range(len(order)):
        idx_i = order[i]
        if idx_i in suppressed:
            continue
        keep.append(idx_i)

        for j in range(i + 1, len(order)):
            idx_j = order[j]
            if idx_j in suppressed:
                continue
            # Only compare same class
            if class_ids[idx_i] != class_ids[idx_j]:
                continue
            # Compute average point distance
            avg_dist = np.mean(
                np.linalg.norm(points[idx_i] - points[idx_j], axis=-1)
            )
            if avg_dist < distance_threshold:
                suppressed.add(idx_j)

    keep = np.array(keep)
    return class_ids[keep], scores[keep], points[keep]


def post_process(
    outputs: Dict[str, np.ndarray],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    simplification_epsilon: float = 0.5,
    nms_distance_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Complete post-processing pipeline.

    Args:
        outputs: Raw model outputs dict with keys:
            - class_logits: (batch, max_instances, num_classes)
            - confidence_logits: (batch, max_instances)
            - point_predictions: (batch, max_instances, num_points, 2)
        confidence_threshold: Minimum score to retain predictions.
        simplification_epsilon: Polyline simplification tolerance.
        nms_distance_threshold: NMS deduplication distance.

    Returns:
        List of per-batch results, each containing:
            - class_ids: predicted class for each instance
            - scores: confidence scores
            - polylines: simplified polyline points
    """
    class_logits = outputs["class_logits"]
    confidence_logits = outputs["confidence_logits"]
    point_predictions = outputs["point_predictions"]

    batch_size = class_logits.shape[0]
    results = []

    for b in range(batch_size):
        # Apply softmax to class logits
        class_probs = apply_softmax(class_logits[b])

        # Sigmoid for confidence
        confidence_scores = 1.0 / (1.0 + np.exp(-confidence_logits[b]))

        # Filter by confidence
        class_ids, scores, _, points = filter_by_confidence(
            class_probs, confidence_scores, point_predictions[b],
            threshold=confidence_threshold,
        )

        # NMS deduplication
        class_ids, scores, points = nms_deduplication(
            class_ids, scores, points,
            distance_threshold=nms_distance_threshold,
        )

        # Polyline simplification
        simplified_polylines = []
        for i in range(len(points)):
            simplified = polyline_simplification(
                points[i], epsilon=simplification_epsilon
            )
            simplified_polylines.append(simplified)

        results.append({
            "class_ids": class_ids,
            "scores": scores,
            "polylines": simplified_polylines,
            "num_predictions": len(class_ids),
        })

    return results


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class StreamMapNetInference:
    """Inference engine for StreamMapNet with temporal state management."""

    def __init__(
        self,
        model: Any,
        bev_height: int = DEFAULT_BEV_HEIGHT,
        bev_width: int = DEFAULT_BEV_WIDTH,
        bev_channels: int = DEFAULT_BEV_CHANNELS,
        num_classes: int = DEFAULT_NUM_CLASSES,
        num_points: int = DEFAULT_NUM_POINTS,
        max_instances: int = DEFAULT_MAX_INSTANCES,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        batch_size: int = 1,
    ):
        self.model = model
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.bev_channels = bev_channels
        self.num_classes = num_classes
        self.num_points = num_points
        self.max_instances = max_instances
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        # Temporal BEV state
        self._bev_state: Optional[np.ndarray] = None
        self._is_savedmodel = not isinstance(model, tf.keras.Model)

        # Trace the inference function for performance
        self._traced_infer = tf.function(
            self._infer_step,
            input_signature=[
                tf.TensorSpec(
                    [None, 6, 224, 480, 3], dtype=tf.float32, name="camera_images"
                ),
                tf.TensorSpec(
                    [None, bev_height, bev_width, bev_channels],
                    dtype=tf.float32,
                    name="temporal_bev_state",
                ),
                tf.TensorSpec(
                    [None, 4, 4], dtype=tf.float32, name="ego_motion"
                ),
            ],
        )

    def reset_state(self):
        """Reset temporal BEV state for a new sequence."""
        self._bev_state = np.zeros(
            (self.batch_size, self.bev_height, self.bev_width, self.bev_channels),
            dtype=np.float32,
        )
        logger.info("Temporal BEV state reset.")

    def _infer_step(
        self,
        camera_images: tf.Tensor,
        temporal_bev_state: tf.Tensor,
        ego_motion: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Single inference step (traced with tf.function)."""
        if self._is_savedmodel:
            # SavedModel inference via signatures
            if hasattr(self.model, "signatures"):
                infer_fn = self.model.signatures["serving_default"]
                outputs = infer_fn(
                    camera_images=camera_images,
                    temporal_bev_state=temporal_bev_state,
                    ego_motion=ego_motion,
                )
            else:
                outputs = self.model(
                    camera_images=camera_images,
                    temporal_bev_state=temporal_bev_state,
                    ego_motion=ego_motion,
                )
        else:
            outputs = self.model(
                [camera_images, temporal_bev_state, ego_motion],
                training=False,
            )
        return outputs

    def infer_frame(
        self,
        camera_images: np.ndarray,
        ego_motion: np.ndarray,
    ) -> Dict[str, Any]:
        """Run inference on a single frame, maintaining temporal state.

        Args:
            camera_images: (batch, 6, 224, 480, 3) surround camera images.
            ego_motion: (batch, 4, 4) ego-motion transformation matrix.

        Returns:
            Post-processed predictions dict.
        """
        if self._bev_state is None:
            self.reset_state()

        # Ensure correct batch dimension
        if camera_images.shape[0] != self.batch_size:
            self.batch_size = camera_images.shape[0]
            self.reset_state()

        # Convert to tensors
        cam_tensor = tf.constant(camera_images, dtype=tf.float32)
        bev_tensor = tf.constant(self._bev_state, dtype=tf.float32)
        ego_tensor = tf.constant(ego_motion, dtype=tf.float32)

        # Run traced inference
        outputs = self._traced_infer(cam_tensor, bev_tensor, ego_tensor)

        # Extract numpy outputs
        output_np = {}
        for key, val in outputs.items():
            if isinstance(val, tf.Tensor):
                output_np[key] = val.numpy()
            else:
                output_np[key] = np.array(val)

        # Update temporal BEV state
        self._bev_state = output_np["updated_bev_state"]

        # Post-process
        results = post_process(
            output_np,
            confidence_threshold=self.confidence_threshold,
        )

        return results

    def infer_sequence(
        self,
        frames: List[Dict[str, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Run inference on a sequence of frames.

        Args:
            frames: List of frame dicts, each containing:
                - camera_images: (batch, 6, 224, 480, 3)
                - ego_motion: (batch, 4, 4)

        Returns:
            List of per-frame post-processed results.
        """
        self.reset_state()
        all_results = []

        for idx, frame in enumerate(frames):
            logger.debug("Processing frame %d/%d", idx + 1, len(frames))
            result = self.infer_frame(
                camera_images=frame["camera_images"],
                ego_motion=frame["ego_motion"],
            )
            all_results.append(result)

        return all_results


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_frame_data(frame_path: str) -> Dict[str, np.ndarray]:
    """Load a single frame's data from disk.

    Supports .npz files with keys: camera_images, ego_motion.
    """
    path = Path(frame_path)
    if path.suffix == ".npz":
        data = np.load(str(path))
        return {
            "camera_images": data["camera_images"],
            "ego_motion": data["ego_motion"],
        }
    elif path.suffix == ".npy":
        # Assume single camera images array; use identity ego motion
        images = np.load(str(path))
        batch_size = images.shape[0] if images.ndim == 5 else 1
        if images.ndim == 4:
            images = images[np.newaxis, ...]
        ego_motion = np.tile(
            np.eye(4, dtype=np.float32)[np.newaxis, ...], (batch_size, 1, 1)
        )
        return {"camera_images": images, "ego_motion": ego_motion}
    else:
        raise ValueError(f"Unsupported frame format: {path.suffix}")


def load_sequence_from_directory(
    data_dir: str,
    sequence_length: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """Load a sequence of frames from a directory.

    Args:
        data_dir: Directory containing frame files (sorted by name).
        sequence_length: Maximum number of frames to load.

    Returns:
        List of frame data dicts.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find frame files
    frame_files = sorted(
        list(data_path.glob("*.npz")) + list(data_path.glob("*.npy"))
    )

    if not frame_files:
        raise FileNotFoundError(f"No .npz or .npy files found in {data_dir}")

    if sequence_length is not None:
        frame_files = frame_files[:sequence_length]

    frames = []
    for f in frame_files:
        frames.append(load_frame_data(str(f)))
        logger.debug("Loaded frame: %s", f.name)

    logger.info("Loaded %d frames from %s", len(frames), data_dir)
    return frames


def generate_dummy_data(
    batch_size: int = 1,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Generate dummy data for testing without real data.

    Args:
        batch_size: Batch size for generated data.
        sequence_length: Number of frames to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of frame data dicts with random values.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for i in range(sequence_length):
        camera_images = rng.random(
            (batch_size, 6, 224, 480, 3), dtype=np.float32
        )
        # Generate plausible ego motion (small rotation + translation)
        ego_motion = np.tile(
            np.eye(4, dtype=np.float32)[np.newaxis, ...], (batch_size, 1, 1)
        )
        # Add small translation
        ego_motion[:, 0, 3] = rng.uniform(-0.5, 0.5, size=batch_size)
        ego_motion[:, 1, 3] = rng.uniform(-0.1, 0.1, size=batch_size)

        frames.append({
            "camera_images": camera_images,
            "ego_motion": ego_motion,
        })

    logger.info(
        "Generated %d dummy frames (batch_size=%d)", sequence_length, batch_size
    )
    return frames


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Get current GPU memory usage if available.

    Returns:
        Dict with memory stats in MB, or None if no GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None

    try:
        memory_info = tf.config.experimental.get_memory_info("GPU:0")
        return {
            "current_mb": memory_info["current"] / (1024 * 1024),
            "peak_mb": memory_info["peak"] / (1024 * 1024),
        }
    except Exception:
        # Fallback: try nvidia-smi based approach
        try:
            mem_usage = tf.config.experimental.get_memory_usage("GPU:0")
            return {"current_mb": mem_usage / (1024 * 1024), "peak_mb": -1.0}
        except Exception:
            return None


def run_benchmark(
    engine: StreamMapNetInference,
    batch_size: int = 1,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    benchmark_iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> Dict[str, Any]:
    """Run throughput and latency benchmark.

    Args:
        engine: Inference engine instance.
        batch_size: Batch size for benchmarking.
        warmup_iterations: Number of warmup iterations.
        benchmark_iterations: Number of timed iterations.
        sequence_length: Frames per sequence for temporal testing.

    Returns:
        Dict with benchmark results.
    """
    logger.info(
        "Starting benchmark: batch_size=%d, warmup=%d, iterations=%d",
        batch_size, warmup_iterations, benchmark_iterations,
    )

    # Generate dummy data
    dummy_frames = generate_dummy_data(
        batch_size=batch_size, sequence_length=1
    )
    dummy_frame = dummy_frames[0]

    # Warmup
    logger.info("Running %d warmup iterations...", warmup_iterations)
    engine.reset_state()
    for i in range(warmup_iterations):
        engine.infer_frame(
            camera_images=dummy_frame["camera_images"],
            ego_motion=dummy_frame["ego_motion"],
        )
    logger.info("Warmup complete.")

    # Record GPU memory before
    mem_before = get_gpu_memory_usage()

    # Timed runs
    latencies = []
    engine.reset_state()
    logger.info("Running %d timed iterations...", benchmark_iterations)

    for i in range(benchmark_iterations):
        start_time = time.perf_counter()
        engine.infer_frame(
            camera_images=dummy_frame["camera_images"],
            ego_motion=dummy_frame["ego_motion"],
        )
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # ms

        # Reset state periodically to simulate sequence boundaries
        if (i + 1) % sequence_length == 0:
            engine.reset_state()

    # Record GPU memory after
    mem_after = get_gpu_memory_usage()

    # Compute statistics
    latencies_arr = np.array(latencies)
    total_time_s = np.sum(latencies_arr) / 1000.0
    fps = benchmark_iterations * batch_size / total_time_s

    results = {
        "batch_size": batch_size,
        "num_iterations": benchmark_iterations,
        "total_time_seconds": float(total_time_s),
        "fps": float(fps),
        "latency_ms": {
            "mean": float(np.mean(latencies_arr)),
            "std": float(np.std(latencies_arr)),
            "min": float(np.min(latencies_arr)),
            "max": float(np.max(latencies_arr)),
            "p50": float(np.percentile(latencies_arr, 50)),
            "p95": float(np.percentile(latencies_arr, 95)),
            "p99": float(np.percentile(latencies_arr, 99)),
        },
        "gpu_memory": {
            "before": mem_before,
            "after": mem_after,
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Batch size:       {batch_size}")
    print(f"  Iterations:       {benchmark_iterations}")
    print(f"  Total time:       {total_time_s:.2f} s")
    print(f"  Throughput (FPS): {fps:.2f}")
    print(f"  Latency (mean):   {results['latency_ms']['mean']:.2f} ms")
    print(f"  Latency (p50):    {results['latency_ms']['p50']:.2f} ms")
    print(f"  Latency (p95):    {results['latency_ms']['p95']:.2f} ms")
    print(f"  Latency (p99):    {results['latency_ms']['p99']:.2f} ms")
    if mem_after is not None:
        print(f"  GPU Memory (cur): {mem_after['current_mb']:.1f} MB")
        print(f"  GPU Memory (peak):{mem_after['peak_mb']:.1f} MB")
    else:
        print("  GPU Memory:       N/A (no GPU detected)")
    print("=" * 60 + "\n")

    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_savedmodel(
    model: tf.keras.Model,
    output_dir: str,
    batch_size: Optional[int] = None,
    bev_height: int = DEFAULT_BEV_HEIGHT,
    bev_width: int = DEFAULT_BEV_WIDTH,
    bev_channels: int = DEFAULT_BEV_CHANNELS,
) -> str:
    """Export model to SavedModel format with concrete functions.

    Args:
        model: Keras model to export.
        output_dir: Directory to save the exported model.
        batch_size: Fixed batch size for signatures (None = dynamic).
        bev_height: BEV grid height.
        bev_width: BEV grid width.
        bev_channels: BEV feature channels.

    Returns:
        Path to exported SavedModel.
    """
    export_path = Path(output_dir) / "savedmodel"
    export_path.mkdir(parents=True, exist_ok=True)

    # Define input signatures
    b = batch_size
    input_signature = [
        tf.TensorSpec([b, 6, 224, 480, 3], tf.float32, name="camera_images"),
        tf.TensorSpec(
            [b, bev_height, bev_width, bev_channels],
            tf.float32,
            name="temporal_bev_state",
        ),
        tf.TensorSpec([b, 4, 4], tf.float32, name="ego_motion"),
    ]

    # Create concrete function
    @tf.function(input_signature=input_signature)
    def serve(camera_images, temporal_bev_state, ego_motion):
        outputs = model([camera_images, temporal_bev_state, ego_motion], training=False)
        return outputs

    # Save with signatures
    tf.saved_model.save(
        model,
        str(export_path),
        signatures={"serving_default": serve},
    )

    logger.info("Exported SavedModel to %s", export_path)
    return str(export_path)


def export_to_tflite(
    model: tf.keras.Model,
    output_dir: str,
    quantize: str = "none",
    batch_size: int = 1,
    bev_height: int = DEFAULT_BEV_HEIGHT,
    bev_width: int = DEFAULT_BEV_WIDTH,
    bev_channels: int = DEFAULT_BEV_CHANNELS,
) -> str:
    """Export model to TFLite format with optional quantization.

    Args:
        model: Keras model to export.
        output_dir: Directory to save TFLite model.
        quantize: Quantization mode: 'none', 'dynamic', 'float16', 'int8'.
        batch_size: Fixed batch size for TFLite.
        bev_height: BEV grid height.
        bev_width: BEV grid width.
        bev_channels: BEV feature channels.

    Returns:
        Path to exported TFLite model.
    """
    export_path = Path(output_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    # Create concrete function with fixed batch size
    @tf.function(input_signature=[
        tf.TensorSpec([batch_size, 6, 224, 480, 3], tf.float32),
        tf.TensorSpec([batch_size, bev_height, bev_width, bev_channels], tf.float32),
        tf.TensorSpec([batch_size, 4, 4], tf.float32),
    ])
    def concrete_func(camera_images, temporal_bev_state, ego_motion):
        return model([camera_images, temporal_bev_state, ego_motion], training=False)

    # Get concrete function
    concrete = concrete_func.get_concrete_function()

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])

    # Apply quantization
    if quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for full integer quantization
        def representative_dataset():
            rng = np.random.default_rng(0)
            for _ in range(100):
                yield [
                    rng.random(
                        (batch_size, 6, 224, 480, 3), dtype=np.float32
                    ),
                    rng.random(
                        (batch_size, bev_height, bev_width, bev_channels),
                        dtype=np.float32,
                    ),
                    np.eye(4, dtype=np.float32)[np.newaxis, ...].repeat(
                        batch_size, axis=0
                    ),
                ]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Determine output filename
    suffix = f"_q{quantize}" if quantize != "none" else ""
    tflite_path = export_path / f"stream_mapnet{suffix}.tflite"

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    logger.info(
        "Exported TFLite model (%s quantization) to %s",
        quantize, tflite_path,
    )
    return str(tflite_path)


def verify_exported_model(
    original_model: tf.keras.Model,
    exported_path: str,
    batch_size: int = 1,
    bev_height: int = DEFAULT_BEV_HEIGHT,
    bev_width: int = DEFAULT_BEV_WIDTH,
    bev_channels: int = DEFAULT_BEV_CHANNELS,
    atol: float = 1e-5,
) -> bool:
    """Verify that exported model produces same outputs as original.

    Args:
        original_model: Original Keras model.
        exported_path: Path to exported SavedModel.
        batch_size: Batch size for verification.
        bev_height: BEV grid height.
        bev_width: BEV grid width.
        bev_channels: BEV feature channels.
        atol: Absolute tolerance for output comparison.

    Returns:
        True if outputs match within tolerance.
    """
    logger.info("Verifying exported model at %s", exported_path)

    # Generate test input
    rng = np.random.default_rng(123)
    test_cameras = rng.random(
        (batch_size, 6, 224, 480, 3), dtype=np.float32
    )
    test_bev = rng.random(
        (batch_size, bev_height, bev_width, bev_channels), dtype=np.float32
    )
    test_ego = np.tile(
        np.eye(4, dtype=np.float32)[np.newaxis, ...], (batch_size, 1, 1)
    )

    # Original model output
    orig_outputs = original_model(
        [
            tf.constant(test_cameras),
            tf.constant(test_bev),
            tf.constant(test_ego),
        ],
        training=False,
    )

    # Exported model output
    exported_model = tf.saved_model.load(exported_path)
    if hasattr(exported_model, "signatures"):
        serve_fn = exported_model.signatures["serving_default"]
        export_outputs = serve_fn(
            camera_images=tf.constant(test_cameras),
            temporal_bev_state=tf.constant(test_bev),
            ego_motion=tf.constant(test_ego),
        )
    else:
        export_outputs = exported_model(
            camera_images=tf.constant(test_cameras),
            temporal_bev_state=tf.constant(test_bev),
            ego_motion=tf.constant(test_ego),
        )

    # Compare outputs
    all_match = True
    for key in orig_outputs:
        orig_val = orig_outputs[key].numpy()
        # SavedModel may use different output key naming
        export_key = key
        if key not in export_outputs:
            # Try common SavedModel output naming patterns
            possible_keys = [
                k for k in export_outputs.keys()
                if key.replace("_", "") in k.replace("_", "")
            ]
            if possible_keys:
                export_key = possible_keys[0]
            else:
                logger.warning("Key %s not found in exported outputs", key)
                all_match = False
                continue

        export_val = export_outputs[export_key].numpy()
        if not np.allclose(orig_val, export_val, atol=atol):
            max_diff = np.max(np.abs(orig_val - export_val))
            logger.warning(
                "Output mismatch for key '%s': max_diff=%.6e (atol=%.6e)",
                key, max_diff, atol,
            )
            all_match = False
        else:
            logger.info("Output '%s' matches (atol=%.6e)", key, atol)

    if all_match:
        logger.info("Verification PASSED: all outputs match within tolerance.")
    else:
        logger.warning("Verification FAILED: some outputs differ.")

    return all_match


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_bev_predictions(
    results: Dict[str, Any],
    canvas_height: int = 800,
    canvas_width: int = 400,
    bev_range_x: Tuple[float, float] = (-30.0, 30.0),
    bev_range_y: Tuple[float, float] = (-15.0, 15.0),
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Draw predicted map elements on a BEV canvas.

    Args:
        results: Post-processed predictions for one batch item.
        canvas_height: Output image height in pixels.
        canvas_width: Output image width in pixels.
        bev_range_x: BEV x-axis range in meters (forward).
        bev_range_y: BEV y-axis range in meters (lateral).
        output_path: Optional path to save the visualization image.

    Returns:
        (canvas_height, canvas_width, 3) uint8 RGB image.
    """
    # Create dark canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # Dark gray background

    # Draw grid lines
    for i in range(0, canvas_height, canvas_height // 10):
        canvas[i, :] = (50, 50, 50)
    for j in range(0, canvas_width, canvas_width // 10):
        canvas[:, j] = (50, 50, 50)

    # Draw ego vehicle marker at center
    cx, cy = canvas_width // 2, canvas_height // 2
    # Simple cross marker
    size = 8
    canvas[cy - size:cy + size, cx - 1:cx + 1] = (255, 255, 255)
    canvas[cy - 1:cy + 1, cx - size:cx + size] = (255, 255, 255)

    def world_to_pixel(x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        px = int(
            (y - bev_range_y[0])
            / (bev_range_y[1] - bev_range_y[0])
            * canvas_width
        )
        py = int(
            (1.0 - (x - bev_range_x[0]) / (bev_range_x[1] - bev_range_x[0]))
            * canvas_height
        )
        return px, py

    class_ids = results.get("class_ids", np.array([]))
    scores = results.get("scores", np.array([]))
    polylines = results.get("polylines", [])

    for i in range(len(class_ids)):
        class_id = int(class_ids[i])
        score = float(scores[i])
        points = polylines[i]

        # Get color for this class (RGB)
        color = CLASS_COLORS.get(class_id, (200, 200, 200))
        class_name = CLASS_NAMES.get(class_id, f"cls{class_id}")

        # Draw polyline
        for j in range(len(points) - 1):
            x0, y0 = points[j]
            x1, y1 = points[j + 1]

            px0, py0 = world_to_pixel(x0, y0)
            px1, py1 = world_to_pixel(x1, y1)

            # Bresenham-like line drawing
            _draw_line(canvas, px0, py0, px1, py1, color, thickness=2)

        # Draw confidence text (simple bitmap-style)
        if len(points) > 0:
            mid_pt = points[len(points) // 2]
            px, py = world_to_pixel(mid_pt[0], mid_pt[1])
            _draw_text_simple(
                canvas, px, py - 10,
                f"{class_name}:{score:.2f}", color,
            )

    # Draw legend
    legend_y = 20
    for cls_id, cls_name in CLASS_NAMES.items():
        color = CLASS_COLORS[cls_id]
        canvas[legend_y:legend_y + 10, 10:30] = color
        _draw_text_simple(canvas, 35, legend_y, cls_name, (200, 200, 200))
        legend_y += 20

    if output_path is not None:
        _save_image(canvas, output_path)
        logger.info("Saved visualization to %s", output_path)

    return canvas


def _draw_line(
    canvas: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw a line on canvas using Bresenham's algorithm."""
    h, w = canvas.shape[:2]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        # Draw point with thickness
        for tx in range(-thickness // 2, thickness // 2 + 1):
            for ty in range(-thickness // 2, thickness // 2 + 1):
                px, py = x0 + tx, y0 + ty
                if 0 <= px < w and 0 <= py < h:
                    canvas[py, px] = color

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_text_simple(
    canvas: np.ndarray,
    x: int, y: int,
    text: str,
    color: Tuple[int, int, int],
) -> None:
    """Draw simple text on canvas (pixel font, no external deps)."""
    # Minimal 3x5 pixel font for digits, letters, and common chars
    FONT = {
        "0": ["111", "101", "101", "101", "111"],
        "1": ["010", "110", "010", "010", "111"],
        "2": ["111", "001", "111", "100", "111"],
        "3": ["111", "001", "111", "001", "111"],
        "4": ["101", "101", "111", "001", "001"],
        "5": ["111", "100", "111", "001", "111"],
        "6": ["111", "100", "111", "101", "111"],
        "7": ["111", "001", "001", "001", "001"],
        "8": ["111", "101", "111", "101", "111"],
        "9": ["111", "101", "111", "001", "111"],
        ".": ["000", "000", "000", "000", "010"],
        ":": ["000", "010", "000", "010", "000"],
        " ": ["000", "000", "000", "000", "000"],
        "a": ["000", "111", "011", "101", "111"],
        "b": ["100", "100", "111", "101", "111"],
        "c": ["000", "111", "100", "100", "111"],
        "d": ["001", "001", "111", "101", "111"],
        "e": ["111", "101", "111", "100", "111"],
        "f": ["011", "100", "111", "100", "100"],
        "g": ["111", "101", "111", "001", "111"],
        "h": ["100", "100", "111", "101", "101"],
        "i": ["010", "000", "010", "010", "010"],
        "j": ["001", "000", "001", "001", "111"],
        "k": ["101", "101", "110", "101", "101"],
        "l": ["110", "010", "010", "010", "111"],
        "m": ["000", "000", "111", "111", "101"],
        "n": ["000", "000", "111", "101", "101"],
        "o": ["000", "111", "101", "101", "111"],
        "p": ["111", "101", "111", "100", "100"],
        "q": ["111", "101", "111", "001", "001"],
        "r": ["000", "111", "100", "100", "100"],
        "s": ["011", "100", "010", "001", "110"],
        "t": ["010", "111", "010", "010", "011"],
        "u": ["000", "101", "101", "101", "111"],
        "v": ["000", "101", "101", "101", "010"],
        "w": ["000", "101", "111", "111", "101"],
        "x": ["000", "101", "010", "010", "101"],
        "y": ["101", "101", "111", "001", "111"],
        "z": ["111", "001", "010", "100", "111"],
    }

    h, w = canvas.shape[:2]
    cursor_x = x

    for ch in text.lower():
        glyph = FONT.get(ch)
        if glyph is None:
            cursor_x += 4
            continue
        for row_idx, row in enumerate(glyph):
            for col_idx, pixel in enumerate(row):
                if pixel == "1":
                    px = cursor_x + col_idx
                    py = y + row_idx
                    if 0 <= px < w and 0 <= py < h:
                        canvas[py, px] = color
        cursor_x += 4  # char width + spacing


def _save_image(canvas: np.ndarray, path: str) -> None:
    """Save RGB numpy array as image. Uses PIL if available, else raw PPM."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
        img = Image.fromarray(canvas.astype(np.uint8), mode="RGB")
        img.save(str(output_path))
    except ImportError:
        # Fallback: save as PPM (no dependencies)
        ppm_path = output_path.with_suffix(".ppm")
        h, w, _ = canvas.shape
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(canvas.astype(np.uint8).tobytes())
        logger.info(
            "PIL not available; saved as PPM: %s", ppm_path
        )


# ---------------------------------------------------------------------------
# Subcommand Handlers
# ---------------------------------------------------------------------------

def cmd_infer(args: argparse.Namespace) -> None:
    """Handle the 'infer' subcommand."""
    # Load model
    logger.info("Loading model from %s", args.model_path)
    model = load_model(
        args.model_path,
        bev_height=DEFAULT_BEV_HEIGHT,
        bev_width=DEFAULT_BEV_WIDTH,
        bev_channels=DEFAULT_BEV_CHANNELS,
    )

    # Create inference engine
    engine = StreamMapNetInference(
        model=model,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
    )

    # Load data
    if args.input_path and Path(args.input_path).exists():
        if Path(args.input_path).is_dir():
            frames = load_sequence_from_directory(
                args.input_path, sequence_length=args.sequence_length
            )
        else:
            # Single file or file list
            if args.input_path.endswith(".txt"):
                with open(args.input_path, "r") as f:
                    file_list = [line.strip() for line in f if line.strip()]
                frames = [load_frame_data(fp) for fp in file_list]
            else:
                frames = [load_frame_data(args.input_path)]
    else:
        logger.warning(
            "Input path not found or not specified. Using dummy data."
        )
        frames = generate_dummy_data(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )

    # Run inference
    results = engine.infer_sequence(frames)

    # Output results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(results):
        # Save per-frame results as JSON
        frame_result = {
            "frame_index": idx,
            "num_predictions": int(result[0]["num_predictions"])
            if result else 0,
            "predictions": [],
        }

        if result:
            batch_result = result[0]  # First batch item
            for i in range(batch_result["num_predictions"]):
                pred = {
                    "class_id": int(batch_result["class_ids"][i]),
                    "class_name": CLASS_NAMES.get(
                        int(batch_result["class_ids"][i]), "unknown"
                    ),
                    "score": float(batch_result["scores"][i]),
                    "polyline": batch_result["polylines"][i].tolist(),
                }
                frame_result["predictions"].append(pred)

        result_path = output_dir / f"frame_{idx:04d}.json"
        with open(result_path, "w") as f:
            json.dump(frame_result, f, indent=2)

        # Visualization
        if args.visualize and result:
            vis_path = output_dir / f"frame_{idx:04d}_vis.png"
            draw_bev_predictions(
                batch_result,
                output_path=str(vis_path),
            )

    logger.info(
        "Inference complete. %d frames processed. Results saved to %s",
        len(results), output_dir,
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Handle the 'benchmark' subcommand."""
    # Load or build model
    if args.model_path and Path(args.model_path).exists():
        logger.info("Loading model from %s", args.model_path)
        model = load_model(args.model_path)
    else:
        logger.info("No model path provided; building fresh model for benchmark.")
        model = build_stream_mapnet_model()

    # Create inference engine
    engine = StreamMapNetInference(
        model=model,
        batch_size=args.batch_size,
    )

    # Run benchmark
    results = run_benchmark(
        engine,
        batch_size=args.batch_size,
        warmup_iterations=args.warmup_iterations,
        benchmark_iterations=args.benchmark_iterations,
        sequence_length=args.sequence_length,
    )

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Benchmark results saved to %s", results_path)


def cmd_export(args: argparse.Namespace) -> None:
    """Handle the 'export' subcommand."""
    # Load model (must be a checkpoint for export)
    if args.model_path and Path(args.model_path).exists():
        logger.info("Loading model from %s", args.model_path)
        model = load_model(args.model_path)
    else:
        logger.info("No model path; building fresh model for export demo.")
        model = build_stream_mapnet_model()

    if not isinstance(model, tf.keras.Model):
        logger.error(
            "Export requires a Keras model (checkpoint format). "
            "Cannot re-export from SavedModel."
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to SavedModel
    savedmodel_path = export_to_savedmodel(
        model, str(output_dir), batch_size=args.batch_size
    )
    logger.info("SavedModel exported to: %s", savedmodel_path)

    # Verify exported model
    verified = verify_exported_model(
        model, savedmodel_path, batch_size=args.batch_size or 1
    )
    if not verified:
        logger.warning("Export verification failed! Check model outputs.")

    # Optionally export to TFLite
    if args.tflite:
        tflite_path = export_to_tflite(
            model,
            str(output_dir / "tflite"),
            quantize=args.quantize,
            batch_size=args.batch_size or 1,
        )
        logger.info("TFLite model exported to: %s", tflite_path)

    logger.info("Export complete. All artifacts in: %s", output_dir)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="StreamMapNet TensorFlow 2 Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on a sequence directory
  python inference.py infer --model-path ./checkpoints --input-path ./data/seq001

  # Benchmark throughput
  python inference.py benchmark --model-path ./checkpoints --batch-size 4

  # Export to SavedModel and TFLite
  python inference.py export --model-path ./checkpoints --output-dir ./exported --tflite
        """,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # --- Infer subcommand ---
    infer_parser = subparsers.add_parser(
        "infer", help="Run inference on input data"
    )
    infer_parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model checkpoint or SavedModel directory",
    )
    infer_parser.add_argument(
        "--input-path", type=str, default=None,
        help="Input data path (directory, file, or file list .txt)",
    )
    infer_parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory for results (default: ./output)",
    )
    infer_parser.add_argument(
        "--confidence-threshold", type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )
    infer_parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size (default: 1)",
    )
    infer_parser.add_argument(
        "--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
        help=f"Max sequence length (default: {DEFAULT_SEQUENCE_LENGTH})",
    )
    infer_parser.add_argument(
        "--visualize", action="store_true",
        help="Generate BEV visualizations",
    )

    # --- Benchmark subcommand ---
    bench_parser = subparsers.add_parser(
        "benchmark", help="Benchmark inference throughput and latency"
    )
    bench_parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model (optional; builds fresh model if not given)",
    )
    bench_parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for benchmarking (default: 1)",
    )
    bench_parser.add_argument(
        "--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS,
        help=f"Warmup iterations (default: {DEFAULT_WARMUP_ITERATIONS})",
    )
    bench_parser.add_argument(
        "--benchmark-iterations", type=int,
        default=DEFAULT_BENCHMARK_ITERATIONS,
        help=f"Timed iterations (default: {DEFAULT_BENCHMARK_ITERATIONS})",
    )
    bench_parser.add_argument(
        "--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
        help=f"Sequence length for state reset (default: {DEFAULT_SEQUENCE_LENGTH})",
    )
    bench_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save benchmark results JSON",
    )

    # --- Export subcommand ---
    export_parser = subparsers.add_parser(
        "export", help="Export model to SavedModel / TFLite"
    )
    export_parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model checkpoint (optional; builds fresh if not given)",
    )
    export_parser.add_argument(
        "--output-dir", type=str, default="./exported",
        help="Output directory for exported models (default: ./exported)",
    )
    export_parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Fixed batch size for export (None = dynamic)",
    )
    export_parser.add_argument(
        "--tflite", action="store_true",
        help="Also export to TFLite format",
    )
    export_parser.add_argument(
        "--quantize", type=str, default="none",
        choices=["none", "dynamic", "float16", "int8"],
        help="TFLite quantization mode (default: none)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to subcommand handler
    if args.command == "infer":
        cmd_infer(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()
        sys.exit(1)
