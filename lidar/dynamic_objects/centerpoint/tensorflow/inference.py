"""
CenterPoint TF2 Inference with Online Tracking.

Runs CenterPoint 3D object detection on LiDAR point cloud frames using a
TensorFlow SavedModel and performs online multi-object tracking via greedy
center-distance matching.

Point cloud format: .bin files with float32 columns [x, y, z, intensity, time_lag].
Output: per-frame JSON with detections including track IDs.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOXEL_SIZE = [0.075, 0.075, 0.2]
POINT_CLOUD_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
BEV_RESOLUTION = (180, 180)

NUSCENES_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

NUM_CLASSES = len(NUSCENES_CLASSES)


# ---------------------------------------------------------------------------
# Point Cloud Utilities
# ---------------------------------------------------------------------------


def load_point_cloud(bin_path: str) -> np.ndarray:
    """Load a point cloud from a .bin file.

    Each point has 5 float32 values: x, y, z, intensity, time_lag.
    Points outside the configured range are removed.

    Args:
        bin_path: Path to the .bin point cloud file.

    Returns:
        Numpy array of shape (N, 5).
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)

    # Filter points outside the detection range
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_CLOUD_RANGE
    mask = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    return points[mask]


# ---------------------------------------------------------------------------
# Circle NMS (center-distance based)
# ---------------------------------------------------------------------------


def circle_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Circle NMS using 2D center distance instead of IoU.

    Two detections suppress each other if their 2D center distance is less
    than the threshold (in meters) AND they share the same class label.

    Args:
        boxes: (N, 7) array [x, y, z, w, l, h, yaw].
        scores: (N,) confidence scores.
        labels: (N,) integer class labels.
        threshold: Distance threshold in meters.

    Returns:
        Array of indices to keep.
    """
    if len(scores) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep = []

    suppressed = np.zeros(len(scores), dtype=bool)

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)

        # Suppress all lower-scored detections of the same class within distance
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue
            if labels[idx] != labels[jdx]:
                continue
            dx = boxes[idx, 0] - boxes[jdx, 0]
            dy = boxes[idx, 1] - boxes[jdx, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < threshold:
                suppressed[jdx] = True

    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# CenterPoint Tracker
# ---------------------------------------------------------------------------


class CenterPointTracker:
    """Online multi-object tracker using greedy center-distance matching.

    Tracks are represented by their predicted 2D position (x, y) propagated
    forward using the estimated velocity. New detections are associated to
    existing tracks using a greedy assignment on the L2 distance matrix.

    Attributes:
        max_age: Number of frames a track can survive without a match.
        distance_threshold: Maximum L2 distance (meters) for a valid match.
    """

    def __init__(
        self,
        max_age: int = 3,
        distance_threshold: float = 4.0,
    ) -> None:
        self.max_age = max_age
        self.distance_threshold = distance_threshold
        self._next_id = 1
        self._tracks: List[Dict] = []

    def reset(self) -> None:
        """Reset tracker state."""
        self._next_id = 1
        self._tracks = []

    def update(
        self,
        detections: List[Dict],
    ) -> List[Dict]:
        """Update tracks with new detections for a single frame.

        Args:
            detections: List of detection dicts with keys:
                - box: [x, y, z, w, l, h, yaw]
                - score: float
                - class_name: str
                - velocity: [vx, vy]

        Returns:
            Same detection list with 'track_id' assigned to each entry.
        """
        # Predict existing track positions using velocity
        for track in self._tracks:
            vx, vy = track["velocity"]
            track["predicted_x"] = track["x"] + vx
            track["predicted_y"] = track["y"] + vy

        num_tracks = len(self._tracks)
        num_dets = len(detections)

        if num_tracks == 0 and num_dets == 0:
            return detections

        # Assign track_ids via greedy matching
        matched_track_indices = set()
        matched_det_indices = set()
        det_track_ids = [None] * num_dets

        if num_tracks > 0 and num_dets > 0:
            # Build L2 distance matrix: (num_tracks, num_dets)
            track_positions = np.array(
                [[t["predicted_x"], t["predicted_y"]] for t in self._tracks]
            )
            det_positions = np.array(
                [[d["box"][0], d["box"][1]] for d in detections]
            )

            # Compute pairwise L2 distances
            diff = track_positions[:, None, :] - det_positions[None, :, :]
            dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

            # Greedy matching: pick the smallest distance first
            flat_indices = np.argsort(dist_matrix, axis=None)
            for flat_idx in flat_indices:
                t_idx = int(flat_idx // num_dets)
                d_idx = int(flat_idx % num_dets)

                if t_idx in matched_track_indices or d_idx in matched_det_indices:
                    continue

                if dist_matrix[t_idx, d_idx] > self.distance_threshold:
                    break  # All remaining distances exceed threshold

                # Match found
                matched_track_indices.add(t_idx)
                matched_det_indices.add(d_idx)
                det_track_ids[d_idx] = self._tracks[t_idx]["track_id"]

                # Update track state
                self._tracks[t_idx]["x"] = detections[d_idx]["box"][0]
                self._tracks[t_idx]["y"] = detections[d_idx]["box"][1]
                self._tracks[t_idx]["velocity"] = detections[d_idx]["velocity"]
                self._tracks[t_idx]["age"] = 0

        # Create new tracks for unmatched detections
        for d_idx in range(num_dets):
            if d_idx not in matched_det_indices:
                new_id = self._next_id
                self._next_id += 1
                det_track_ids[d_idx] = new_id
                self._tracks.append(
                    {
                        "track_id": new_id,
                        "x": detections[d_idx]["box"][0],
                        "y": detections[d_idx]["box"][1],
                        "velocity": detections[d_idx]["velocity"],
                        "age": 0,
                    }
                )

        # Age unmatched tracks and remove stale ones
        surviving_tracks = []
        for t_idx, track in enumerate(self._tracks):
            if t_idx in matched_track_indices:
                surviving_tracks.append(track)
            elif t_idx < num_tracks:
                # Existing track that was not matched this frame
                track["age"] += 1
                if track["age"] < self.max_age:
                    surviving_tracks.append(track)
            else:
                # Newly created track (always survives)
                surviving_tracks.append(track)

        self._tracks = surviving_tracks

        # Assign track_ids back to detections
        for i, det in enumerate(detections):
            det["track_id"] = det_track_ids[i]

        return detections


# ---------------------------------------------------------------------------
# CenterPoint Inference Engine
# ---------------------------------------------------------------------------


class CenterPointInference:
    """TF2 inference wrapper for CenterPoint 3D detection.

    Loads a TensorFlow SavedModel and runs inference on point cloud data,
    followed by circle NMS post-processing and online tracking.

    Args:
        model_path: Path to the TF SavedModel directory.
        score_threshold: Minimum detection confidence score.
        nms_threshold: Circle NMS distance threshold (meters).
        max_age: Tracker max age (frames without match before deletion).
        distance_threshold: Tracker matching distance threshold (meters).
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.1,
        nms_threshold: float = 0.5,
        max_age: int = 3,
        distance_threshold: float = 4.0,
    ) -> None:
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

        # Load SavedModel
        print(f"[INFO] Loading SavedModel from: {model_path}")
        load_start = time.perf_counter()
        self.model = tf.saved_model.load(model_path)
        load_elapsed = time.perf_counter() - load_start
        print(f"[INFO] Model loaded in {load_elapsed:.2f}s")

        # Get the default serving signature
        if hasattr(self.model, "signatures"):
            self.infer_fn = self.model.signatures["serving_default"]
        else:
            self.infer_fn = self.model.__call__

        # Initialize tracker
        self.tracker = CenterPointTracker(
            max_age=max_age,
            distance_threshold=distance_threshold,
        )

    def preprocess(self, points: np.ndarray) -> tf.Tensor:
        """Preprocess raw point cloud for model input.

        Converts the point cloud array to a float32 TensorFlow tensor.
        The model expects a batched input of shape (1, N, 5).

        Args:
            points: Numpy array of shape (N, 5).

        Returns:
            TensorFlow tensor of shape (1, N, 5).
        """
        points_tensor = tf.constant(points, dtype=tf.float32)
        points_tensor = tf.expand_dims(points_tensor, axis=0)  # Add batch dim
        return points_tensor

    def decode_output(
        self,
        output: Dict[str, tf.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw model output into boxes, scores, labels, and velocities.

        Expected output keys from the SavedModel:
            - 'boxes': (B, M, 7) - [x, y, z, w, l, h, yaw]
            - 'scores': (B, M)
            - 'labels': (B, M) - integer class indices
            - 'velocities': (B, M, 2) - [vx, vy]

        Args:
            output: Dictionary of model output tensors.

        Returns:
            Tuple of (boxes, scores, labels, velocities) as numpy arrays,
            each with batch dimension removed.
        """
        # Handle both dict-like and attribute-like access patterns
        if isinstance(output, dict):
            boxes = output["boxes"].numpy()
            scores = output["scores"].numpy()
            labels = output["labels"].numpy()
            velocities = output["velocities"].numpy()
        else:
            boxes = output["boxes"].numpy()
            scores = output["scores"].numpy()
            labels = output["labels"].numpy()
            velocities = output["velocities"].numpy()

        # Remove batch dimension (assume batch size = 1)
        boxes = boxes[0]
        scores = scores[0]
        labels = labels[0]
        velocities = velocities[0]

        return boxes, scores, labels, velocities

    def postprocess(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        velocities: np.ndarray,
    ) -> List[Dict]:
        """Apply score filtering and circle NMS.

        Args:
            boxes: (M, 7) detection boxes.
            scores: (M,) confidence scores.
            labels: (M,) class label indices.
            velocities: (M, 2) velocity estimates.

        Returns:
            List of detection dicts after filtering and NMS.
        """
        # Score threshold filter
        mask = scores >= self.score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        velocities = velocities[mask]

        if len(scores) == 0:
            return []

        # Circle NMS
        keep_indices = circle_nms(boxes, scores, labels, self.nms_threshold)

        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        velocities = velocities[keep_indices]

        # Build detection list
        detections = []
        for i in range(len(scores)):
            label_idx = int(labels[i])
            class_name = (
                NUSCENES_CLASSES[label_idx]
                if 0 <= label_idx < NUM_CLASSES
                else "unknown"
            )
            det = {
                "box": boxes[i].tolist(),
                "score": float(scores[i]),
                "class_name": class_name,
                "velocity": velocities[i].tolist(),
            }
            detections.append(det)

        return detections

    def run_frame(self, bin_path: str) -> Tuple[List[Dict], float]:
        """Run inference on a single point cloud frame.

        Args:
            bin_path: Path to the .bin point cloud file.

        Returns:
            Tuple of (detections with track_id, inference_time_seconds).
        """
        # Load and preprocess
        points = load_point_cloud(bin_path)
        input_tensor = self.preprocess(points)

        # Run model
        t_start = time.perf_counter()
        output = self.infer_fn(input_tensor)
        t_end = time.perf_counter()
        inference_time = t_end - t_start

        # Decode and postprocess
        boxes, scores, labels, velocities = self.decode_output(output)
        detections = self.postprocess(boxes, scores, labels, velocities)

        # Online tracking
        detections = self.tracker.update(detections)

        return detections, inference_time

    def run_sequence(
        self,
        bin_paths: List[str],
        output_dir: str,
    ) -> Dict:
        """Run inference and tracking on a sequence of frames.

        Args:
            bin_paths: Ordered list of .bin file paths.
            output_dir: Directory to save per-frame JSON results.

        Returns:
            Dictionary with timing statistics.
        """
        os.makedirs(output_dir, exist_ok=True)

        self.tracker.reset()

        total_time = 0.0
        frame_times = []
        total_detections = 0

        print(f"[INFO] Processing {len(bin_paths)} frames...")
        print(f"[INFO] Output directory: {output_dir}")
        print("-" * 60)

        for frame_idx, bin_path in enumerate(bin_paths):
            detections, inference_time = self.run_frame(bin_path)
            frame_times.append(inference_time)
            total_time += inference_time
            total_detections += len(detections)

            # Build output structure
            frame_result = {
                "frame_id": frame_idx,
                "file": os.path.basename(bin_path),
                "num_detections": len(detections),
                "inference_time_ms": round(inference_time * 1000, 2),
                "detections": detections,
            }

            # Save per-frame JSON
            output_filename = f"frame_{frame_idx:06d}.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "w") as f:
                json.dump(frame_result, f, indent=2)

            # Progress logging
            fps_instant = 1.0 / inference_time if inference_time > 0 else 0.0
            print(
                f"  Frame {frame_idx:4d} | "
                f"{os.path.basename(bin_path):40s} | "
                f"Dets: {len(detections):3d} | "
                f"Time: {inference_time * 1000:6.1f} ms | "
                f"FPS: {fps_instant:5.1f}"
            )

        # Compute statistics
        frame_times_np = np.array(frame_times)
        num_frames = len(bin_paths)
        avg_time = total_time / num_frames if num_frames > 0 else 0.0
        avg_fps = num_frames / total_time if total_time > 0 else 0.0
        median_time = float(np.median(frame_times_np)) if num_frames > 0 else 0.0
        min_time = float(np.min(frame_times_np)) if num_frames > 0 else 0.0
        max_time = float(np.max(frame_times_np)) if num_frames > 0 else 0.0

        stats = {
            "num_frames": num_frames,
            "total_detections": total_detections,
            "total_time_s": round(total_time, 3),
            "avg_time_per_frame_ms": round(avg_time * 1000, 2),
            "median_time_per_frame_ms": round(median_time * 1000, 2),
            "min_time_per_frame_ms": round(min_time * 1000, 2),
            "max_time_per_frame_ms": round(max_time * 1000, 2),
            "avg_fps": round(avg_fps, 2),
        }

        # Print summary
        print("-" * 60)
        print("[INFO] Inference Summary:")
        print(f"  Total frames:        {num_frames}")
        print(f"  Total detections:    {total_detections}")
        print(f"  Total time:          {total_time:.3f} s")
        print(f"  Avg time/frame:      {avg_time * 1000:.2f} ms")
        print(f"  Median time/frame:   {median_time * 1000:.2f} ms")
        print(f"  Min time/frame:      {min_time * 1000:.2f} ms")
        print(f"  Max time/frame:      {max_time * 1000:.2f} ms")
        print(f"  Avg FPS:             {avg_fps:.2f}")

        # Save summary statistics
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] Statistics saved to: {stats_path}")

        return stats


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CenterPoint TF2 Inference with Online Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the TF SavedModel directory.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help=(
            "Path to input data. Can be a single .bin file or a directory "
            "containing .bin files (processed in sorted order)."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to save per-frame JSON output files.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Minimum confidence score to keep a detection.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="Circle NMS distance threshold in meters.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=3,
        help="Tracker: max frames without match before track deletion.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=4.0,
        help="Tracker: maximum L2 distance (meters) for a valid match.",
    )
    return parser.parse_args()


def collect_bin_files(input_path: str) -> List[str]:
    """Collect .bin file paths from the given input path.

    If input_path is a single .bin file, returns a list with that file.
    If it is a directory, returns all .bin files within it sorted by name.

    Args:
        input_path: Path to a .bin file or directory of .bin files.

    Returns:
        Sorted list of absolute .bin file paths.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If no .bin files are found.
    """
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if os.path.isfile(input_path):
        if not input_path.endswith(".bin"):
            raise ValueError(f"Input file is not a .bin file: {input_path}")
        return [input_path]

    # Directory: collect all .bin files
    bin_files = sorted(
        [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".bin")
        ]
    )

    if not bin_files:
        raise ValueError(f"No .bin files found in directory: {input_path}")

    return bin_files


def main() -> None:
    """Main entry point for CenterPoint inference."""
    args = parse_args()

    print("=" * 60)
    print("CenterPoint TF2 Inference with Online Tracking")
    print("=" * 60)
    print(f"  Model path:         {args.model_path}")
    print(f"  Input path:         {args.input_path}")
    print(f"  Output path:        {args.output_path}")
    print(f"  Score threshold:    {args.score_threshold}")
    print(f"  NMS threshold:      {args.nms_threshold} m")
    print(f"  Tracker max age:    {args.max_age}")
    print(f"  Tracker dist thres: {args.distance_threshold} m")
    print(f"  Voxel size:         {VOXEL_SIZE}")
    print(f"  Point cloud range:  {POINT_CLOUD_RANGE}")
    print(f"  BEV resolution:     {BEV_RESOLUTION}")
    print(f"  Classes ({NUM_CLASSES}):      {NUSCENES_CLASSES}")
    print("=" * 60)

    # Collect input files
    bin_files = collect_bin_files(args.input_path)
    print(f"[INFO] Found {len(bin_files)} .bin file(s)")

    # Initialize inference engine
    engine = CenterPointInference(
        model_path=args.model_path,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        max_age=args.max_age,
        distance_threshold=args.distance_threshold,
    )

    # Run inference on the sequence
    stats = engine.run_sequence(bin_files, args.output_path)

    print("=" * 60)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
