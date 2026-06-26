"""
MapTR TensorFlow 2 Inference and Visualization Script.

Performs inference using the MapTR model for vectorized HD map construction
from multi-camera surround-view images, and provides comprehensive
visualization utilities for BEV maps and camera views.

Map classes:
    0: ped_crossing (blue)
    1: divider (orange)
    2: boundary (green)

BEV range: x=[-30, 30]m, y=[-15, 15]m (60m x 30m)
Input: 6 cameras (FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT)
Image size: 480 x 800
"""

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .model import MapTRModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAMERA_NAMES = [
    "FRONT",
    "FRONT_RIGHT",
    "FRONT_LEFT",
    "BACK",
    "BACK_LEFT",
    "BACK_RIGHT",
]

MAP_CLASSES = {
    0: "ped_crossing",
    1: "divider",
    2: "boundary",
}

CLASS_COLORS = {
    0: (0.12, 0.47, 0.71),   # blue
    1: (1.00, 0.50, 0.05),   # orange
    2: (0.17, 0.63, 0.17),   # green
}

NUM_QUERIES = 50
NUM_POINTS_PER_POLYLINE = 20
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 800
BEV_X_RANGE = (-30.0, 30.0)
BEV_Y_RANGE = (-15.0, 15.0)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# MapTRPredictor
# ---------------------------------------------------------------------------


class MapTRPredictor:
    """Inference wrapper for the MapTR model.

    Supports loading from either a TensorFlow checkpoint or a SavedModel
    directory. Provides preprocessing, prediction, and postprocessing methods.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        saved_model_path: Optional[str] = None,
    ) -> None:
        """Initialize the predictor by loading the model weights.

        Args:
            checkpoint_path: Path to a tf.train.Checkpoint directory or prefix.
                If provided, a MapTRModel is instantiated and weights are
                restored from the checkpoint.
            saved_model_path: Path to a SavedModel directory exported via
                tf.saved_model.save. If provided, the model is loaded directly
                using tf.saved_model.load.

        Raises:
            ValueError: If neither checkpoint_path nor saved_model_path is
                provided.
        """
        if checkpoint_path is None and saved_model_path is None:
            raise ValueError(
                "Either checkpoint_path or saved_model_path must be provided."
            )

        if saved_model_path is not None:
            self.model = tf.saved_model.load(saved_model_path)
            self._use_saved_model = True
        else:
            self.model = MapTRModel(
                num_classes=len(MAP_CLASSES),
                num_queries=NUM_QUERIES,
                num_points=NUM_POINTS_PER_POLYLINE,
            )
            checkpoint = tf.train.Checkpoint(model=self.model)
            status = checkpoint.restore(
                tf.train.latest_checkpoint(checkpoint_path)
                if tf.io.gfile.isdir(checkpoint_path)
                else checkpoint_path
            )
            status.expect_partial()
            self._use_saved_model = False

    def preprocess_images(
        self, images: np.ndarray
    ) -> tf.Tensor:
        """Resize and normalize camera images for model input.

        Args:
            images: NumPy array of shape (num_cameras, H, W, 3) with uint8
                values in [0, 255] or float32 in [0, 1].

        Returns:
            A tf.Tensor of shape (1, num_cameras, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
            normalized with ImageNet statistics, channels-last format.
        """
        processed = []
        for i in range(images.shape[0]):
            img = tf.cast(images[i], tf.float32)
            # Ensure [0, 1] range
            if img.numpy().max() > 1.0:
                img = img / 255.0
            # Resize to target dimensions
            img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
            # Normalize with ImageNet stats
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            processed.append(img)

        # Stack cameras and add batch dimension: (1, 6, H, W, 3)
        stacked = tf.stack(processed, axis=0)
        batched = tf.expand_dims(stacked, axis=0)
        return batched

    def predict(
        self,
        images: tf.Tensor,
        intrinsics: tf.Tensor,
        extrinsics: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Run forward inference through the model.

        Args:
            images: Preprocessed image tensor of shape
                (batch, num_cameras, H, W, 3).
            intrinsics: Camera intrinsic matrices of shape
                (batch, num_cameras, 3, 3).
            extrinsics: Camera extrinsic matrices (camera-to-ego) of shape
                (batch, num_cameras, 4, 4).

        Returns:
            Dictionary containing raw model predictions:
                - "scores": (batch, num_queries, num_classes + 1) classification
                  logits including background class.
                - "points": (batch, num_queries, num_points, 2) normalized point
                  coordinates in [0, 1].
        """
        if self._use_saved_model:
            predictions = self.model.signatures["serving_default"](
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
        else:
            predictions = self.model(
                images, intrinsics, extrinsics, training=False
            )
        return predictions

    def postprocess(
        self,
        predictions: Dict[str, tf.Tensor],
        confidence_threshold: float = 0.4,
    ) -> List[Dict[str, object]]:
        """Filter predictions by confidence and denormalize coordinates.

        Converts normalized point coordinates from [0, 1] to real-world BEV
        coordinates within the configured range.

        Args:
            predictions: Raw model output dictionary with "scores" and "points"
                tensors.
            confidence_threshold: Minimum confidence score to retain a
                prediction. Defaults to 0.4.

        Returns:
            List of dictionaries (one per batch element), each containing:
                - "polylines": list of np.ndarray of shape (num_points, 2) with
                  real-world coordinates.
                - "labels": list of int class indices.
                - "scores": list of float confidence scores.
        """
        scores_logits = predictions["scores"].numpy()  # (B, Q, C+1)
        points_norm = predictions["points"].numpy()    # (B, Q, P, 2)

        batch_size = scores_logits.shape[0]
        results = []

        for b in range(batch_size):
            # Apply softmax to get class probabilities
            scores_exp = np.exp(
                scores_logits[b] - scores_logits[b].max(axis=-1, keepdims=True)
            )
            scores_prob = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

            # Exclude background class (last index)
            fg_scores = scores_prob[:, :-1]  # (Q, num_classes)
            max_scores = fg_scores.max(axis=-1)  # (Q,)
            max_labels = fg_scores.argmax(axis=-1)  # (Q,)

            # Filter by confidence threshold
            valid_mask = max_scores >= confidence_threshold
            valid_indices = np.where(valid_mask)[0]

            polylines = []
            labels = []
            confs = []

            for idx in valid_indices:
                pts = points_norm[b, idx]  # (P, 2)
                # Denormalize: x from [0,1] -> [-30, 30], y from [0,1] -> [-15, 15]
                pts_real = np.empty_like(pts)
                pts_real[:, 0] = (
                    pts[:, 0] * (BEV_X_RANGE[1] - BEV_X_RANGE[0])
                    + BEV_X_RANGE[0]
                )
                pts_real[:, 1] = (
                    pts[:, 1] * (BEV_Y_RANGE[1] - BEV_Y_RANGE[0])
                    + BEV_Y_RANGE[0]
                )
                polylines.append(pts_real)
                labels.append(int(max_labels[idx]))
                confs.append(float(max_scores[idx]))

            results.append({
                "polylines": polylines,
                "labels": labels,
                "scores": confs,
            })

        return results


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------


def visualize_bev(
    predictions: Dict[str, object],
    ground_truth: Optional[Dict[str, object]] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Visualize predicted map elements in BEV (top-down) view.

    Args:
        predictions: Postprocessed predictions dict with "polylines", "labels",
            and "scores" keys.
        ground_truth: Optional ground truth dict with the same structure.
            If provided, GT polylines are drawn as dashed lines.
        save_path: Optional file path to save the figure.
        ax: Optional matplotlib Axes to draw on. If None, a new figure is
            created.

    Returns:
        The matplotlib Figure object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Draw grid
    ax.set_xlim(BEV_X_RANGE)
    ax.set_ylim(BEV_Y_RANGE)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.set_title("BEV Map Predictions")

    # Ego vehicle marker at center
    ego_marker = plt.Polygon(
        [[-1.0, -0.5], [1.0, 0.0], [-1.0, 0.5]],
        closed=True,
        facecolor="red",
        edgecolor="black",
        linewidth=1.0,
        zorder=10,
    )
    ax.add_patch(ego_marker)

    # Plot ground truth (dashed) if available
    if ground_truth is not None:
        for polyline, label in zip(
            ground_truth["polylines"], ground_truth["labels"]
        ):
            color = CLASS_COLORS.get(label, (0.5, 0.5, 0.5))
            ax.plot(
                polyline[:, 0],
                polyline[:, 1],
                color=color,
                linestyle="--",
                linewidth=2.0,
                alpha=0.6,
            )

    # Plot predictions (solid)
    for polyline, label in zip(predictions["polylines"], predictions["labels"]):
        color = CLASS_COLORS.get(label, (0.5, 0.5, 0.5))
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            color=color,
            linestyle="-",
            linewidth=2.0,
            alpha=0.9,
        )

    # Legend
    legend_patches = []
    for cls_id, cls_name in MAP_CLASSES.items():
        legend_patches.append(
            mpatches.Patch(color=CLASS_COLORS[cls_id], label=cls_name)
        )
    if ground_truth is not None:
        legend_patches.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="gray",
                linestyle="--",
                label="ground truth",
            )
        )
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_camera_grid(
    images: np.ndarray,
    predictions: Optional[Dict[str, object]] = None,
    intrinsics: Optional[np.ndarray] = None,
    extrinsics: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    axes: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Display the 6 surround-view camera images in a 2x3 grid.

    Args:
        images: Array of shape (6, H, W, 3) with uint8 values [0, 255] or
            float values [0, 1].
        predictions: Optional postprocessed predictions dict. If provided along
            with intrinsics and extrinsics, map elements are projected onto the
            camera images.
        intrinsics: Camera intrinsic matrices of shape (6, 3, 3). Required for
            projection overlay.
        extrinsics: Camera extrinsic matrices of shape (6, 4, 4). Required for
            projection overlay.
        save_path: Optional file path to save the figure.
        axes: Optional 2x3 array of matplotlib Axes to draw on.

    Returns:
        The matplotlib Figure object.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    else:
        fig = axes.flat[0].get_figure()

    for idx, (ax, cam_name) in enumerate(zip(axes.flat, CAMERA_NAMES)):
        img = images[idx]
        # Convert to displayable range
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        ax.imshow(img)
        ax.set_title(cam_name, fontsize=10, fontweight="bold")
        ax.axis("off")

        # Optionally overlay projected map elements
        if (
            predictions is not None
            and intrinsics is not None
            and extrinsics is not None
        ):
            _overlay_projections(
                ax, predictions, intrinsics[idx], extrinsics[idx], img.shape
            )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _overlay_projections(
    ax: plt.Axes,
    predictions: Dict[str, object],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    img_shape: Tuple[int, int, int],
) -> None:
    """Project BEV polylines onto a single camera image and overlay.

    Args:
        ax: Matplotlib axes for the camera image.
        predictions: Postprocessed predictions dict.
        intrinsic: 3x3 camera intrinsic matrix.
        extrinsic: 4x4 camera extrinsic matrix (ego-to-camera transform).
    """
    h, w = img_shape[0], img_shape[1]

    # Compute inverse extrinsic (ego-to-camera)
    try:
        ego_to_cam = np.linalg.inv(extrinsic)
    except np.linalg.LinAlgError:
        return

    for polyline, label in zip(predictions["polylines"], predictions["labels"]):
        color = CLASS_COLORS.get(label, (0.5, 0.5, 0.5))

        # Convert BEV points to 3D (z=0 ground plane in ego frame)
        pts_3d = np.zeros((polyline.shape[0], 4), dtype=np.float64)
        pts_3d[:, 0] = polyline[:, 0]
        pts_3d[:, 1] = polyline[:, 1]
        pts_3d[:, 2] = 0.0
        pts_3d[:, 3] = 1.0

        # Transform to camera frame
        pts_cam = (ego_to_cam @ pts_3d.T).T[:, :3]  # (P, 3)

        # Filter points behind the camera
        valid = pts_cam[:, 2] > 0.1
        if not np.any(valid):
            continue

        # Project to image plane
        pts_img = (intrinsic @ pts_cam[valid].T).T  # (N, 3)
        pts_img = pts_img[:, :2] / pts_img[:, 2:3]

        # Filter points outside image bounds
        in_bounds = (
            (pts_img[:, 0] >= 0)
            & (pts_img[:, 0] < w)
            & (pts_img[:, 1] >= 0)
            & (pts_img[:, 1] < h)
        )
        pts_img = pts_img[in_bounds]

        if len(pts_img) >= 2:
            ax.plot(
                pts_img[:, 0],
                pts_img[:, 1],
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )


def visualize_sample(
    images: np.ndarray,
    predictions: Dict[str, object],
    ground_truth: Optional[Dict[str, object]] = None,
    intrinsics: Optional[np.ndarray] = None,
    extrinsics: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Combined visualization: camera grid on top, BEV map on bottom.

    Args:
        images: Array of shape (6, H, W, 3).
        predictions: Postprocessed predictions dict.
        ground_truth: Optional ground truth dict.
        intrinsics: Optional camera intrinsics of shape (6, 3, 3).
        extrinsics: Optional camera extrinsics of shape (6, 4, 4).
        save_path: Optional file path to save the combined figure.

    Returns:
        The matplotlib Figure object.
    """
    fig = plt.figure(figsize=(16, 14))

    # Top: 2x3 camera grid (rows 0-1)
    cam_axes = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(3, 3, row * 3 + col + 1)
            cam_axes.append(ax)
    cam_axes = np.array(cam_axes).reshape(2, 3)

    for idx, (ax, cam_name) in enumerate(zip(cam_axes.flat, CAMERA_NAMES)):
        img = images[idx]
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        ax.imshow(img)
        ax.set_title(cam_name, fontsize=10, fontweight="bold")
        ax.axis("off")

        if (
            predictions is not None
            and intrinsics is not None
            and extrinsics is not None
        ):
            _overlay_projections(
                ax, predictions, intrinsics[idx], extrinsics[idx], img.shape
            )

    # Bottom: BEV visualization (row 2, spanning all columns)
    bev_ax = fig.add_subplot(3, 1, 3)
    visualize_bev(predictions, ground_truth=ground_truth, ax=bev_ax)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------


def load_sample(sample_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single sample's images and calibration from disk.

    Expected directory structure:
        sample_dir/
            images/
                FRONT.jpg (or .png)
                FRONT_RIGHT.jpg
                FRONT_LEFT.jpg
                BACK.jpg
                BACK_LEFT.jpg
                BACK_RIGHT.jpg
            calibration/
                intrinsics.npy   # shape (6, 3, 3)
                extrinsics.npy   # shape (6, 4, 4)

    Args:
        sample_dir: Path to the sample directory.

    Returns:
        Tuple of (images, intrinsics, extrinsics) where images is
        (6, H, W, 3) uint8, intrinsics is (6, 3, 3), extrinsics is (6, 4, 4).

    Raises:
        FileNotFoundError: If expected files are missing.
    """
    images_dir = os.path.join(sample_dir, "images")
    calib_dir = os.path.join(sample_dir, "calibration")

    images = []
    for cam_name in CAMERA_NAMES:
        # Try common image extensions
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(images_dir, cam_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(
                f"Could not find image for camera {cam_name} in {images_dir}"
            )
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        images.append(img.numpy())

    images = np.stack(images, axis=0)  # (6, H, W, 3)

    intrinsics_path = os.path.join(calib_dir, "intrinsics.npy")
    extrinsics_path = os.path.join(calib_dir, "extrinsics.npy")

    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics not found at {intrinsics_path}")
    if not os.path.exists(extrinsics_path):
        raise FileNotFoundError(f"Extrinsics not found at {extrinsics_path}")

    intrinsics = np.load(intrinsics_path).astype(np.float32)  # (6, 3, 3)
    extrinsics = np.load(extrinsics_path).astype(np.float32)  # (6, 4, 4)

    return images, intrinsics, extrinsics


def find_samples(input_dir: str) -> List[str]:
    """Discover sample directories within an input directory.

    A valid sample directory contains an 'images' subdirectory.

    Args:
        input_dir: Root directory to search for samples.

    Returns:
        Sorted list of valid sample directory paths.
    """
    sample_dirs = []
    for entry in sorted(os.listdir(input_dir)):
        candidate = os.path.join(input_dir, entry)
        if os.path.isdir(candidate) and os.path.isdir(
            os.path.join(candidate, "images")
        ):
            sample_dirs.append(candidate)
    return sample_dirs


# ---------------------------------------------------------------------------
# CLI and Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "MapTR TensorFlow 2 inference: run vectorized HD map prediction "
            "from multi-camera surround-view images and visualize results."
        )
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to tf.train.Checkpoint directory or prefix.",
    )
    model_group.add_argument(
        "--saved_model_path",
        type=str,
        default=None,
        help="Path to a SavedModel directory.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing sample data (images + calibration).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.4,
        help="Minimum confidence score to retain predictions (default: 0.4).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Maximum number of samples to process (default: 10).",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save raw predictions to .npz files in the output directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualizations interactively with plt.show().",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point: load model, process samples, visualize, and save."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading MapTR model...")
    predictor = MapTRPredictor(
        checkpoint_path=args.checkpoint_path,
        saved_model_path=args.saved_model_path,
    )
    print("Model loaded successfully.")

    # Find samples
    sample_dirs = find_samples(args.input_dir)
    if not sample_dirs:
        print(f"No valid samples found in {args.input_dir}")
        return

    num_to_process = min(args.num_samples, len(sample_dirs))
    sample_dirs = sample_dirs[:num_to_process]
    print(f"Found {len(sample_dirs)} sample(s) to process.")

    # Process samples
    inference_times = []

    for sample_idx, sample_dir in enumerate(sample_dirs):
        sample_name = os.path.basename(sample_dir)
        print(f"\nProcessing sample {sample_idx + 1}/{num_to_process}: "
              f"{sample_name}")

        # Load data
        try:
            images, intrinsics, extrinsics = load_sample(sample_dir)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Preprocess
        images_tensor = predictor.preprocess_images(images)
        intrinsics_tensor = tf.constant(
            intrinsics[np.newaxis], dtype=tf.float32
        )
        extrinsics_tensor = tf.constant(
            extrinsics[np.newaxis], dtype=tf.float32
        )

        # Inference
        t_start = time.perf_counter()
        raw_predictions = predictor.predict(
            images_tensor, intrinsics_tensor, extrinsics_tensor
        )
        t_end = time.perf_counter()
        inference_time = t_end - t_start
        inference_times.append(inference_time)
        print(f"  Inference time: {inference_time:.3f}s")

        # Postprocess
        results = predictor.postprocess(
            raw_predictions, confidence_threshold=args.confidence_threshold
        )
        result = results[0]  # single batch element
        num_detections = len(result["polylines"])
        print(f"  Detections: {num_detections} polylines above threshold "
              f"{args.confidence_threshold}")

        # Load ground truth if available
        gt_path = os.path.join(sample_dir, "ground_truth.npz")
        ground_truth = None
        if os.path.exists(gt_path):
            gt_data = np.load(gt_path, allow_pickle=True)
            ground_truth = {
                "polylines": list(gt_data["polylines"]),
                "labels": list(gt_data["labels"]),
            }

        # Visualize
        vis_path = os.path.join(
            args.output_dir, f"{sample_name}_combined.png"
        )
        visualize_sample(
            images=images,
            predictions=result,
            ground_truth=ground_truth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            save_path=vis_path,
        )
        print(f"  Saved visualization: {vis_path}")

        # Save BEV separately
        bev_path = os.path.join(args.output_dir, f"{sample_name}_bev.png")
        fig_bev = visualize_bev(
            result, ground_truth=ground_truth, save_path=bev_path
        )
        plt.close(fig_bev)

        # Save camera grid separately
        cam_path = os.path.join(args.output_dir, f"{sample_name}_cameras.png")
        fig_cam = visualize_camera_grid(
            images=images,
            predictions=result,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            save_path=cam_path,
        )
        plt.close(fig_cam)

        # Save raw predictions if requested
        if args.save_predictions:
            pred_path = os.path.join(
                args.output_dir, f"{sample_name}_predictions.npz"
            )
            np.savez(
                pred_path,
                scores=raw_predictions["scores"].numpy(),
                points=raw_predictions["points"].numpy(),
                polylines=np.array(result["polylines"], dtype=object),
                labels=np.array(result["labels"]),
                confidences=np.array(result["scores"]),
            )
            print(f"  Saved predictions: {pred_path}")

        if args.show:
            plt.show()

        plt.close("all")

    # Summary
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"  Samples processed: {len(inference_times)}")
    if inference_times:
        avg_time = np.mean(inference_times)
        print(f"  Average inference time: {avg_time:.3f}s")
        print(f"  Min inference time: {min(inference_times):.3f}s")
        print(f"  Max inference time: {max(inference_times):.3f}s")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
