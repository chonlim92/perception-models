#!/usr/bin/env python3
"""
StreamMapNet TensorFlow 2 Evaluation Script.

Evaluates a trained StreamMapNet model on a validation set, computing standard
HD map prediction metrics including Chamfer Distance, Average Precision at
multiple thresholds, per-class metrics, and Frechet distance.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


# =============================================================================
# Metric Computation Functions
# =============================================================================


def chamfer_distance(
    pred_points: np.ndarray, gt_points: np.ndarray
) -> float:
    """
    Compute symmetric Chamfer Distance between two point sets.

    For each predicted point, find the nearest GT point and vice versa.
    Returns the average of both directions.

    Args:
        pred_points: Array of shape (N, 2) with predicted polyline points.
        gt_points: Array of shape (M, 2) with ground truth polyline points.

    Returns:
        Symmetric Chamfer distance (scalar).
    """
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return float("inf")

    # pred_points: (N, 2), gt_points: (M, 2)
    # Compute pairwise squared distances: (N, M)
    diff = pred_points[:, np.newaxis, :] - gt_points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)

    # For each pred point, find nearest GT point
    min_pred_to_gt = np.min(dist_sq, axis=1)  # (N,)
    # For each GT point, find nearest pred point
    min_gt_to_pred = np.min(dist_sq, axis=0)  # (M,)

    # Symmetric Chamfer distance (using L2, not squared)
    cd = 0.5 * (np.mean(np.sqrt(min_pred_to_gt)) + np.mean(np.sqrt(min_gt_to_pred)))
    return float(cd)


def directed_chamfer_distance(
    source: np.ndarray, target: np.ndarray
) -> float:
    """
    Compute directed Chamfer Distance from source to target.

    Args:
        source: Array of shape (N, 2).
        target: Array of shape (M, 2).

    Returns:
        Mean nearest-neighbor distance from source to target.
    """
    if source.shape[0] == 0 or target.shape[0] == 0:
        return float("inf")

    diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    min_dists = np.sqrt(np.min(dist_sq, axis=1))
    return float(np.mean(min_dists))


def frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute the discrete Frechet distance between two polylines.

    Uses dynamic programming to compute the exact discrete Frechet distance.

    Args:
        P: Array of shape (N, 2) representing the first polyline.
        Q: Array of shape (M, 2) representing the second polyline.

    Returns:
        Discrete Frechet distance (scalar).
    """
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return float("inf")

    n = P.shape[0]
    m = Q.shape[0]

    # Compute distance matrix
    dist_matrix = np.sqrt(
        np.sum((P[:, np.newaxis, :] - Q[np.newaxis, :, :]) ** 2, axis=-1)
    )

    # DP table for Frechet distance
    dp = np.full((n, m), -1.0)

    def _frechet_rec(i: int, j: int) -> float:
        """Recursive computation with memoization."""
        if dp[i, j] > -0.5:
            return dp[i, j]
        if i == 0 and j == 0:
            dp[i, j] = dist_matrix[0, 0]
        elif i == 0:
            dp[i, j] = max(_frechet_rec(0, j - 1), dist_matrix[0, j])
        elif j == 0:
            dp[i, j] = max(_frechet_rec(i - 1, 0), dist_matrix[i, 0])
        else:
            dp[i, j] = max(
                min(
                    _frechet_rec(i - 1, j),
                    _frechet_rec(i - 1, j - 1),
                    _frechet_rec(i, j - 1),
                ),
                dist_matrix[i, j],
            )
        return dp[i, j]

    # Use iterative approach for larger sequences to avoid recursion limits
    if n * m > 10000:
        return _frechet_distance_iterative(dist_matrix, n, m)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), n * m + 100))
    return float(_frechet_rec(n - 1, m - 1))


def _frechet_distance_iterative(
    dist_matrix: np.ndarray, n: int, m: int
) -> float:
    """Iterative computation of discrete Frechet distance for large inputs."""
    dp = np.zeros((n, m), dtype=np.float64)
    dp[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], dist_matrix[0, j])

    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(
                min(dp[i - 1, j], dp[i - 1, j - 1], dp[i, j - 1]),
                dist_matrix[i, j],
            )

    return float(dp[n - 1, m - 1])


def compute_average_precision(
    predictions: List[Dict],
    ground_truths: List[Dict],
    threshold: float,
    class_name: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Average Precision for a single class at a given threshold.

    A prediction matches a ground truth if the Chamfer distance between them
    is below the threshold. Each GT element can only be matched once.

    Args:
        predictions: List of dicts with keys 'points' (Nx2 array),
                     'confidence' (float), 'class' (str).
        ground_truths: List of dicts with keys 'points' (Mx2 array),
                       'class' (str).
        threshold: Distance threshold for matching.
        class_name: The class to evaluate.

    Returns:
        Tuple of (AP, precision_array, recall_array).
    """
    # Filter by class
    class_preds = [p for p in predictions if p["class"] == class_name]
    class_gts = [g for g in ground_truths if g["class"] == class_name]

    n_gt = len(class_gts)
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    # Sort predictions by confidence (descending)
    class_preds = sorted(class_preds, key=lambda x: x["confidence"], reverse=True)

    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))
    matched_gt = set()

    for pred_idx, pred in enumerate(class_preds):
        best_dist = float("inf")
        best_gt_idx = -1

        for gt_idx, gt in enumerate(class_gts):
            if gt_idx in matched_gt:
                continue
            dist = chamfer_distance(pred["points"], gt["points"])
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_dist < threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1.0
            matched_gt.add(best_gt_idx)
        else:
            fp[pred_idx] = 1.0

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using all-point interpolation (PASCAL VOC style)
    ap = _compute_ap_from_pr(precision, recall)

    return ap, precision, recall


def _compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute AP from precision-recall curve using all-point interpolation.

    Uses the PASCAL VOC 2010+ method: for each recall level, precision is the
    maximum precision at any recall >= that level.
    """
    if len(precision) == 0:
        return 0.0

    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing (from right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    recall_change = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum area under PR curve
    ap = np.sum((mrec[recall_change + 1] - mrec[recall_change]) * mpre[recall_change + 1])
    return float(ap)


def compute_all_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    thresholds: List[float],
    compute_frechet: bool = True,
) -> Dict:
    """
    Compute all evaluation metrics across classes and thresholds.

    Args:
        predictions: List of prediction dicts.
        ground_truths: List of ground truth dicts.
        class_names: List of class names to evaluate.
        thresholds: List of distance thresholds for AP computation.
        compute_frechet: Whether to compute Frechet distance metrics.

    Returns:
        Dictionary with all metrics organized by class and threshold.
    """
    results = {
        "per_class": {},
        "thresholds": thresholds,
        "class_names": class_names,
    }

    all_aps = []

    for class_name in class_names:
        results["per_class"][class_name] = {"ap_per_threshold": {}, "mean_ap": 0.0}
        class_aps = []

        for threshold in thresholds:
            ap, precision, recall = compute_average_precision(
                predictions, ground_truths, threshold, class_name
            )
            results["per_class"][class_name]["ap_per_threshold"][f"{threshold:.1f}"] = {
                "ap": ap,
                "n_predictions": len([p for p in predictions if p["class"] == class_name]),
                "n_ground_truths": len([g for g in ground_truths if g["class"] == class_name]),
            }
            class_aps.append(ap)
            all_aps.append(ap)

        results["per_class"][class_name]["mean_ap"] = float(np.mean(class_aps)) if class_aps else 0.0

        # Compute Frechet distances for this class
        if compute_frechet:
            class_preds = [p for p in predictions if p["class"] == class_name]
            class_gts = [g for g in ground_truths if g["class"] == class_name]
            frechet_dists = _compute_class_frechet(class_preds, class_gts)
            results["per_class"][class_name]["frechet"] = frechet_dists

    results["mAP"] = float(np.mean(all_aps)) if all_aps else 0.0

    # Compute overall Chamfer distance statistics
    all_chamfer = []
    for pred in predictions:
        best_cd = float("inf")
        for gt in ground_truths:
            if gt["class"] == pred["class"]:
                cd = chamfer_distance(pred["points"], gt["points"])
                best_cd = min(best_cd, cd)
        if best_cd < float("inf"):
            all_chamfer.append(best_cd)

    if all_chamfer:
        results["chamfer_stats"] = {
            "mean": float(np.mean(all_chamfer)),
            "median": float(np.median(all_chamfer)),
            "std": float(np.std(all_chamfer)),
            "min": float(np.min(all_chamfer)),
            "max": float(np.max(all_chamfer)),
        }
    else:
        results["chamfer_stats"] = {
            "mean": float("inf"),
            "median": float("inf"),
            "std": 0.0,
            "min": float("inf"),
            "max": float("inf"),
        }

    return results


def _compute_class_frechet(
    predictions: List[Dict], ground_truths: List[Dict]
) -> Dict:
    """Compute Frechet distance statistics for matched predictions/GT pairs."""
    if not predictions or not ground_truths:
        return {"mean": float("inf"), "median": float("inf"), "std": 0.0}

    # Match each prediction to its nearest GT by Chamfer distance
    frechet_dists = []
    for pred in predictions:
        best_gt = None
        best_cd = float("inf")
        for gt in ground_truths:
            cd = chamfer_distance(pred["points"], gt["points"])
            if cd < best_cd:
                best_cd = cd
                best_gt = gt
        if best_gt is not None:
            fd = frechet_distance(pred["points"], best_gt["points"])
            frechet_dists.append(fd)

    if frechet_dists:
        return {
            "mean": float(np.mean(frechet_dists)),
            "median": float(np.median(frechet_dists)),
            "std": float(np.std(frechet_dists)),
        }
    return {"mean": float("inf"), "median": float("inf"), "std": 0.0}


# =============================================================================
# StreamMapNet Model Definition (Minimal for evaluation)
# =============================================================================


class BEVEncoder(tf.keras.layers.Layer):
    """Bird's Eye View feature encoder."""

    def __init__(self, feature_dim: int = 256, bev_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.bev_size = bev_size
        self.conv1 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(feature_dim, 3, padding="same", activation="relu")
        self.pool = tf.keras.layers.MaxPool2D(2)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        return x


class TemporalFusion(tf.keras.layers.Layer):
    """Temporal state propagation module for BEV features."""

    def __init__(self, feature_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.gru_cell = tf.keras.layers.GRUCell(feature_dim)
        self.proj = tf.keras.layers.Dense(feature_dim)

    def call(self, current_bev, prev_state, training=False):
        """
        Fuse current BEV features with previous temporal state.

        Args:
            current_bev: Current BEV features (B, H, W, C).
            prev_state: Previous hidden state (B, H*W, C) or None.

        Returns:
            fused_features: Fused BEV features (B, H, W, C).
            new_state: Updated hidden state.
        """
        batch_size = tf.shape(current_bev)[0]
        h = tf.shape(current_bev)[1]
        w = tf.shape(current_bev)[2]
        c = current_bev.shape[-1] or self.feature_dim

        # Flatten spatial dims
        flat_bev = tf.reshape(current_bev, [batch_size, h * w, c])

        if prev_state is None:
            prev_state = tf.zeros_like(flat_bev)

        # Apply GRU cell across spatial locations
        flat_bev_2d = tf.reshape(flat_bev, [batch_size * h * w, c])
        prev_state_2d = tf.reshape(prev_state, [batch_size * h * w, c])

        output, new_state_list = self.gru_cell(flat_bev_2d, [prev_state_2d])
        new_state = tf.reshape(new_state_list[0], [batch_size, h * w, c])

        fused = tf.reshape(output, [batch_size, h, w, c])
        return fused, new_state


class MapDecoder(tf.keras.layers.Layer):
    """Decodes BEV features into polyline predictions with confidence scores."""

    def __init__(
        self,
        num_classes: int = 3,
        num_points_per_element: int = 20,
        max_elements: int = 50,
        feature_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_points_per_element = num_points_per_element
        self.max_elements = max_elements
        self.feature_dim = feature_dim

        self.flatten = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")

        # Per-class prediction heads
        self.class_heads = []
        for _ in range(num_classes):
            head = {
                "points": tf.keras.layers.Dense(
                    max_elements * num_points_per_element * 2
                ),
                "confidence": tf.keras.layers.Dense(max_elements, activation="sigmoid"),
                "existence": tf.keras.layers.Dense(max_elements, activation="sigmoid"),
            }
            self.class_heads.append(head)

    def build(self, input_shape):
        # Ensure sub-layers are built
        super().build(input_shape)

    def call(self, bev_features, training=False):
        """
        Decode BEV features into map element predictions.

        Args:
            bev_features: BEV feature map (B, H, W, C).

        Returns:
            Dict with keys per class containing points, confidence, existence.
        """
        x = self.flatten(bev_features)
        x = self.dense1(x)
        x = self.dense2(x)

        outputs = {}
        for cls_idx, head in enumerate(self.class_heads):
            points_flat = head["points"](x)
            confidence = head["confidence"](x)
            existence = head["existence"](x)

            batch_size = tf.shape(x)[0]
            points = tf.reshape(
                points_flat,
                [batch_size, self.max_elements, self.num_points_per_element, 2],
            )

            outputs[cls_idx] = {
                "points": points,
                "confidence": confidence,
                "existence": existence,
            }

        return outputs


class StreamMapNet(tf.keras.Model):
    """
    StreamMapNet: Temporal HD Map prediction model.

    Processes multi-camera images and maintains a temporal BEV state
    to predict vectorized HD map elements (polylines).
    """

    def __init__(
        self,
        num_classes: int = 3,
        feature_dim: int = 256,
        bev_size: int = 100,
        num_points_per_element: int = 20,
        max_elements: int = 50,
        input_image_shape: Tuple[int, int, int] = (224, 480, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.bev_size = bev_size
        self.num_points_per_element = num_points_per_element
        self.max_elements = max_elements
        self.input_image_shape = input_image_shape

        # Image backbone (simplified for evaluation)
        self.backbone = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=None,
            input_shape=input_image_shape,
            pooling=None,
        )
        self.neck = tf.keras.layers.Conv2D(feature_dim, 1, padding="same")

        # BEV encoder
        self.bev_encoder = BEVEncoder(feature_dim=feature_dim, bev_size=bev_size)

        # Temporal fusion
        self.temporal_fusion = TemporalFusion(feature_dim=feature_dim)

        # Map decoder
        self.map_decoder = MapDecoder(
            num_classes=num_classes,
            num_points_per_element=num_points_per_element,
            max_elements=max_elements,
            feature_dim=feature_dim,
        )

    def call(self, inputs, prev_state=None, training=False):
        """
        Forward pass.

        Args:
            inputs: Camera images tensor (B, H, W, C).
            prev_state: Previous temporal state or None.

        Returns:
            predictions: Dict of predictions per class.
            new_state: Updated temporal state.
        """
        # Extract image features
        features = self.backbone(inputs, training=training)
        features = self.neck(features)

        # Encode into BEV space
        bev_features = self.bev_encoder(features, training=training)

        # Temporal fusion
        fused_features, new_state = self.temporal_fusion(
            bev_features, prev_state, training=training
        )

        # Decode map elements
        predictions = self.map_decoder(fused_features, training=training)

        return predictions, new_state

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "feature_dim": self.feature_dim,
                "bev_size": self.bev_size,
                "num_points_per_element": self.num_points_per_element,
                "max_elements": self.max_elements,
                "input_image_shape": self.input_image_shape,
            }
        )
        return config


# =============================================================================
# Synthetic Data Generator for Testing
# =============================================================================


CLASS_NAMES = ["lane_divider", "pedestrian_crossing", "road_boundary"]


def generate_random_polyline(
    num_points: int = 20,
    x_range: Tuple[float, float] = (-30.0, 30.0),
    y_range: Tuple[float, float] = (-15.0, 15.0),
    smoothness: float = 2.0,
) -> np.ndarray:
    """
    Generate a smooth random polyline.

    Args:
        num_points: Number of points in the polyline.
        x_range: Range of x coordinates.
        y_range: Range of y coordinates.
        smoothness: Controls curve smoothness (higher = smoother).

    Returns:
        Array of shape (num_points, 2).
    """
    # Generate control points
    n_control = max(3, num_points // 4)
    t_control = np.linspace(0, 1, n_control)
    t_fine = np.linspace(0, 1, num_points)

    # Random control points
    x_control = np.random.uniform(x_range[0], x_range[1], n_control)
    y_control = np.random.uniform(y_range[0], y_range[1], n_control)

    # Sort x to make a reasonable polyline (left to right)
    x_control = np.sort(x_control)

    # Interpolate
    x_fine = np.interp(t_fine, t_control, x_control)
    y_fine = np.interp(t_fine, t_control, y_control)

    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter1d

    x_fine = gaussian_filter1d(x_fine, sigma=smoothness)
    y_fine = gaussian_filter1d(y_fine, sigma=smoothness)

    return np.stack([x_fine, y_fine], axis=-1)


def generate_synthetic_scene(
    num_elements_per_class: Tuple[int, int] = (2, 8),
    num_points: int = 20,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate a synthetic scene with ground truth and predictions.

    Creates GT elements and corresponding predictions with noise added.

    Args:
        num_elements_per_class: (min, max) number of elements per class.
        num_points: Number of points per polyline.

    Returns:
        Tuple of (predictions, ground_truths).
    """
    ground_truths = []
    predictions = []

    for class_name in CLASS_NAMES:
        n_gt = np.random.randint(num_elements_per_class[0], num_elements_per_class[1] + 1)

        class_gts = []
        for _ in range(n_gt):
            polyline = generate_random_polyline(num_points=num_points)
            class_gts.append({"points": polyline, "class": class_name})

        ground_truths.extend(class_gts)

        # Generate predictions: some match GT (with noise), some are false positives
        n_true_pos = np.random.randint(0, n_gt + 1)
        n_false_pos = np.random.randint(0, 3)

        # True positive predictions (noisy versions of GT)
        matched_indices = np.random.choice(n_gt, size=min(n_true_pos, n_gt), replace=False)
        for idx in matched_indices:
            noise_level = np.random.uniform(0.1, 1.5)
            noisy_points = class_gts[idx]["points"] + np.random.randn(num_points, 2) * noise_level
            confidence = np.random.uniform(0.3, 0.99)
            predictions.append(
                {
                    "points": noisy_points,
                    "class": class_name,
                    "confidence": confidence,
                }
            )

        # False positive predictions
        for _ in range(n_false_pos):
            fp_polyline = generate_random_polyline(num_points=num_points)
            confidence = np.random.uniform(0.1, 0.6)
            predictions.append(
                {
                    "points": fp_polyline,
                    "class": class_name,
                    "confidence": confidence,
                }
            )

    return predictions, ground_truths


def generate_synthetic_dataset(
    num_scenes: int = 20,
    num_frames_per_scene: int = 10,
    image_shape: Tuple[int, int, int] = (224, 480, 3),
    num_points: int = 20,
) -> List[Dict]:
    """
    Generate a synthetic validation dataset with scene boundaries.

    Args:
        num_scenes: Number of scenes.
        num_frames_per_scene: Frames per scene.
        image_shape: Shape of input images.
        num_points: Points per polyline.

    Returns:
        List of sample dicts with keys: image, predictions, ground_truths,
        scene_id, frame_id, is_first_frame.
    """
    dataset = []
    for scene_id in range(num_scenes):
        for frame_id in range(num_frames_per_scene):
            # Generate synthetic image
            image = np.random.rand(*image_shape).astype(np.float32)

            predictions, ground_truths = generate_synthetic_scene(
                num_points=num_points
            )

            dataset.append(
                {
                    "image": image,
                    "predictions": predictions,
                    "ground_truths": ground_truths,
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "is_first_frame": frame_id == 0,
                }
            )

    return dataset


def create_tf_dataset(
    samples: List[Dict], batch_size: int = 4
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from the synthetic samples (images only).

    For model inference, we only need images batched. Ground truth and
    predictions are handled separately.

    Args:
        samples: List of sample dicts.
        batch_size: Batch size.

    Returns:
        tf.data.Dataset of image batches.
    """
    images = np.array([s["image"] for s in samples])
    scene_ids = np.array([s["scene_id"] for s in samples])
    frame_ids = np.array([s["frame_id"] for s in samples])

    ds = tf.data.Dataset.from_tensor_slices(
        {"image": images, "scene_id": scene_ids, "frame_id": frame_ids}
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# Model Loading Utilities
# =============================================================================


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: Optional[Dict] = None,
) -> StreamMapNet:
    """
    Load a StreamMapNet model from a TensorFlow checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory or prefix.
        model_config: Model configuration dict. If None, uses defaults.

    Returns:
        Loaded StreamMapNet model.
    """
    if model_config is None:
        model_config = {
            "num_classes": 3,
            "feature_dim": 256,
            "bev_size": 100,
            "num_points_per_element": 20,
            "max_elements": 50,
            "input_image_shape": (224, 480, 3),
        }

    model = StreamMapNet(**model_config)

    # Build model by running a dummy input
    dummy_input = tf.zeros([1] + list(model_config["input_image_shape"]))
    model(dummy_input, prev_state=None, training=False)

    # Restore checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    try:
        status.expect_partial()
        print(f"[INFO] Checkpoint loaded from: {checkpoint_path}")
    except Exception as e:
        print(f"[WARN] Partial checkpoint restore: {e}")

    return model


def load_model_from_saved_model(saved_model_path: str) -> tf.keras.Model:
    """
    Load a StreamMapNet model from a SavedModel directory.

    Args:
        saved_model_path: Path to SavedModel directory.

    Returns:
        Loaded model.
    """
    model = tf.saved_model.load(saved_model_path)
    print(f"[INFO] SavedModel loaded from: {saved_model_path}")
    return model


# =============================================================================
# Evaluation Pipeline
# =============================================================================


def decode_model_output(
    raw_output: Dict,
    class_names: List[str],
    confidence_threshold: float = 0.3,
    existence_threshold: float = 0.5,
) -> List[Dict]:
    """
    Decode raw model output into a list of prediction dicts.

    Args:
        raw_output: Dict with integer keys (class indices) containing
                    'points', 'confidence', 'existence' tensors.
        class_names: List of class names.
        confidence_threshold: Minimum confidence to keep a prediction.
        existence_threshold: Minimum existence probability.

    Returns:
        List of prediction dicts with 'points', 'confidence', 'class'.
    """
    predictions = []

    for cls_idx in range(len(class_names)):
        if cls_idx not in raw_output:
            continue

        cls_output = raw_output[cls_idx]
        points = cls_output["points"].numpy()  # (B, max_elements, num_points, 2)
        confidence = cls_output["confidence"].numpy()  # (B, max_elements)
        existence = cls_output["existence"].numpy()  # (B, max_elements)

        batch_size = points.shape[0]

        for b in range(batch_size):
            for elem_idx in range(points.shape[1]):
                conf = confidence[b, elem_idx]
                exist = existence[b, elem_idx]

                if conf >= confidence_threshold and exist >= existence_threshold:
                    predictions.append(
                        {
                            "points": points[b, elem_idx],
                            "confidence": float(conf),
                            "class": class_names[cls_idx],
                            "batch_idx": b,
                        }
                    )

    return predictions


def evaluate_with_model(
    model: StreamMapNet,
    dataset: List[Dict],
    class_names: List[str],
    thresholds: List[float],
    batch_size: int = 4,
    confidence_threshold: float = 0.3,
    compute_frechet: bool = True,
) -> Dict:
    """
    Run full evaluation using a trained model with temporal state propagation.

    Processes sequences maintaining temporal BEV state, resetting at scene
    boundaries.

    Args:
        model: Trained StreamMapNet model.
        dataset: List of sample dicts.
        class_names: List of class names.
        thresholds: AP thresholds.
        batch_size: Batch size for inference.
        confidence_threshold: Confidence threshold for predictions.
        compute_frechet: Whether to compute Frechet distance.

    Returns:
        Metrics dict.
    """
    all_predictions = []
    all_ground_truths = []

    # Group by scene for temporal processing
    scenes = {}
    for sample in dataset:
        sid = sample["scene_id"]
        if sid not in scenes:
            scenes[sid] = []
        scenes[sid].append(sample)

    # Sort frames within each scene
    for sid in scenes:
        scenes[sid].sort(key=lambda x: x["frame_id"])

    print(f"[INFO] Evaluating {len(scenes)} scenes with temporal state propagation...")

    for scene_idx, (scene_id, frames) in enumerate(scenes.items()):
        temporal_state = None

        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            images = np.array([f["image"] for f in batch_frames])
            images_tensor = tf.constant(images, dtype=tf.float32)

            # Reset state at scene boundary (first frame)
            if i == 0:
                temporal_state = None

            # Model inference
            raw_output, temporal_state = model(
                images_tensor, prev_state=temporal_state, training=False
            )

            # Decode predictions
            batch_preds = decode_model_output(
                raw_output, class_names, confidence_threshold=confidence_threshold
            )

            # Assign predictions to correct frames
            for pred in batch_preds:
                batch_idx = pred.pop("batch_idx")
                if batch_idx < len(batch_frames):
                    all_predictions.append(pred)

            # Collect ground truths
            for frame in batch_frames:
                all_ground_truths.extend(frame["ground_truths"])

        if (scene_idx + 1) % 5 == 0:
            print(f"  Processed {scene_idx + 1}/{len(scenes)} scenes")

    print(f"[INFO] Total predictions: {len(all_predictions)}")
    print(f"[INFO] Total ground truths: {len(all_ground_truths)}")

    # Compute metrics
    metrics = compute_all_metrics(
        all_predictions, all_ground_truths, class_names, thresholds, compute_frechet
    )

    return metrics


def evaluate_synthetic(
    class_names: List[str],
    thresholds: List[float],
    num_scenes: int = 10,
    num_frames_per_scene: int = 5,
    compute_frechet: bool = True,
) -> Dict:
    """
    Run evaluation on synthetic data without a trained model.

    Uses synthetic predictions and ground truths directly for testing
    the evaluation pipeline.

    Args:
        class_names: List of class names.
        thresholds: AP thresholds.
        num_scenes: Number of synthetic scenes.
        num_frames_per_scene: Frames per scene.
        compute_frechet: Whether to compute Frechet distance.

    Returns:
        Metrics dict.
    """
    print(f"[INFO] Generating synthetic dataset: {num_scenes} scenes x {num_frames_per_scene} frames")
    dataset = generate_synthetic_dataset(
        num_scenes=num_scenes, num_frames_per_scene=num_frames_per_scene
    )

    all_predictions = []
    all_ground_truths = []
    for sample in dataset:
        all_predictions.extend(sample["predictions"])
        all_ground_truths.extend(sample["ground_truths"])

    print(f"[INFO] Total predictions: {len(all_predictions)}")
    print(f"[INFO] Total ground truths: {len(all_ground_truths)}")

    metrics = compute_all_metrics(
        all_predictions, all_ground_truths, class_names, thresholds, compute_frechet
    )

    return metrics


# =============================================================================
# Visualization
# =============================================================================


def visualize_predictions(
    predictions: List[Dict],
    ground_truths: List[Dict],
    class_names: List[str],
    output_path: Optional[str] = None,
    max_elements: int = 50,
):
    """
    Visualize predictions vs ground truth in BEV.

    Args:
        predictions: List of prediction dicts.
        ground_truths: List of ground truth dicts.
        class_names: Class names.
        output_path: Path to save figure. If None, displays interactively.
        max_elements: Maximum number of elements to plot per class.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not available, skipping visualization.")
        return

    colors_pred = {"lane_divider": "blue", "pedestrian_crossing": "green", "road_boundary": "red"}
    colors_gt = {"lane_divider": "cyan", "pedestrian_crossing": "lime", "road_boundary": "salmon"}

    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 6))
    if len(class_names) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, class_names):
        ax.set_title(f"{class_name}", fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Plot ground truth
        class_gts = [g for g in ground_truths if g["class"] == class_name][:max_elements]
        for gt in class_gts:
            points = gt["points"]
            ax.plot(
                points[:, 0], points[:, 1],
                color=colors_gt.get(class_name, "gray"),
                linewidth=2, alpha=0.7, linestyle="--",
            )

        # Plot predictions
        class_preds = [p for p in predictions if p["class"] == class_name][:max_elements]
        for pred in class_preds:
            points = pred["points"]
            ax.plot(
                points[:, 0], points[:, 1],
                color=colors_pred.get(class_name, "darkblue"),
                linewidth=1.5, alpha=0.8,
            )

        # Legend
        handles = [
            mpatches.Patch(color=colors_gt.get(class_name, "gray"), label="Ground Truth"),
            mpatches.Patch(color=colors_pred.get(class_name, "darkblue"), label="Prediction"),
        ]
        ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# Results Printing and Saving
# =============================================================================


def print_results(metrics: Dict):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 80)
    print("STREAMMAPNET EVALUATION RESULTS")
    print("=" * 80)

    class_names = metrics.get("class_names", [])
    thresholds = metrics.get("thresholds", [])

    # Header
    threshold_strs = [f"AP@{t:.1f}m" for t in thresholds]
    header = f"{'Class':<25}" + "".join(f"{s:>12}" for s in threshold_strs) + f"{'Mean AP':>12}"
    print(header)
    print("-" * 80)

    # Per-class results
    for class_name in class_names:
        class_metrics = metrics["per_class"][class_name]
        row = f"{class_name:<25}"
        for t in thresholds:
            ap = class_metrics["ap_per_threshold"][f"{t:.1f}"]["ap"]
            row += f"{ap * 100:>11.1f}%"
        row += f"{class_metrics['mean_ap'] * 100:>11.1f}%"
        print(row)

    # Overall mAP
    print("-" * 80)
    print(f"{'Overall mAP':<25}" + " " * (12 * len(thresholds)) + f"{metrics['mAP'] * 100:>11.1f}%")

    # Chamfer distance stats
    if "chamfer_stats" in metrics:
        print("\n" + "-" * 40)
        print("Chamfer Distance Statistics:")
        stats = metrics["chamfer_stats"]
        print(f"  Mean:   {stats['mean']:.4f} m")
        print(f"  Median: {stats['median']:.4f} m")
        print(f"  Std:    {stats['std']:.4f} m")
        print(f"  Min:    {stats['min']:.4f} m")
        print(f"  Max:    {stats['max']:.4f} m")

    # Frechet distance stats
    has_frechet = any(
        "frechet" in metrics["per_class"][c]
        for c in class_names
    )
    if has_frechet:
        print("\n" + "-" * 40)
        print("Frechet Distance (per class):")
        for class_name in class_names:
            if "frechet" in metrics["per_class"][class_name]:
                fd = metrics["per_class"][class_name]["frechet"]
                print(f"  {class_name:<25} mean={fd['mean']:.4f}  median={fd['median']:.4f}  std={fd['std']:.4f}")

    print("=" * 80 + "\n")


def save_results_to_json(metrics: Dict, output_path: str):
    """
    Save evaluation results to a JSON file.

    Args:
        metrics: Metrics dict.
        output_path: Output JSON file path.
    """
    # Convert numpy types to Python types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    serializable = _convert(metrics)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[INFO] Results saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="StreamMapNet TensorFlow 2 Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with synthetic data (no model needed):
  python evaluate.py --synthetic --num-scenes 20

  # Evaluate with checkpoint:
  python evaluate.py --checkpoint-path ./checkpoints/stream_mapnet

  # Evaluate with SavedModel:
  python evaluate.py --saved-model-path ./saved_model

  # Evaluate and save results:
  python evaluate.py --synthetic --output-json results.json --visualize
        """,
    )

    # Model loading
    model_group = parser.add_argument_group("Model Loading")
    model_group.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Path to TF checkpoint directory.",
    )
    model_group.add_argument(
        "--saved-model-path", type=str, default=None,
        help="Path to SavedModel directory.",
    )
    model_group.add_argument(
        "--model-config", type=str, default=None,
        help="Path to model config JSON file.",
    )

    # Data
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-path", type=str, default=None,
        help="Path to validation data directory.",
    )
    data_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing the evaluation pipeline.",
    )
    data_group.add_argument(
        "--num-scenes", type=int, default=10,
        help="Number of synthetic scenes (only with --synthetic).",
    )
    data_group.add_argument(
        "--num-frames-per-scene", type=int, default=5,
        help="Frames per scene (only with --synthetic).",
    )
    data_group.add_argument(
        "--scene-ids", type=int, nargs="*", default=None,
        help="Specific scene IDs to evaluate.",
    )

    # Evaluation parameters
    eval_group = parser.add_argument_group("Evaluation Parameters")
    eval_group.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for inference.",
    )
    eval_group.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.5, 1.0, 1.5],
        help="AP distance thresholds in meters.",
    )
    eval_group.add_argument(
        "--confidence-threshold", type=float, default=0.3,
        help="Minimum confidence threshold for predictions.",
    )
    eval_group.add_argument(
        "--no-frechet", action="store_true",
        help="Skip Frechet distance computation (faster).",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save results as JSON.",
    )
    output_group.add_argument(
        "--visualize", action="store_true",
        help="Visualize predictions vs ground truth.",
    )
    output_group.add_argument(
        "--vis-output", type=str, default=None,
        help="Path to save visualization (if not set, displays interactively).",
    )

    return parser.parse_args()


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Configuration
    class_names = CLASS_NAMES
    thresholds = args.thresholds
    compute_frechet = not args.no_frechet

    print("[INFO] StreamMapNet Evaluation")
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Thresholds: {thresholds}")
    print(f"[INFO] Compute Frechet: {compute_frechet}")

    # Load model config
    model_config = None
    if args.model_config:
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
        print(f"[INFO] Model config loaded from: {args.model_config}")

    # Determine evaluation mode
    if args.synthetic:
        # Synthetic evaluation (no model needed)
        print("[INFO] Running synthetic evaluation (no model)...")
        metrics = evaluate_synthetic(
            class_names=class_names,
            thresholds=thresholds,
            num_scenes=args.num_scenes,
            num_frames_per_scene=args.num_frames_per_scene,
            compute_frechet=compute_frechet,
        )

        # For visualization with synthetic data
        if args.visualize:
            dataset = generate_synthetic_dataset(
                num_scenes=min(2, args.num_scenes),
                num_frames_per_scene=args.num_frames_per_scene,
            )
            vis_preds = []
            vis_gts = []
            for sample in dataset[:5]:
                vis_preds.extend(sample["predictions"])
                vis_gts.extend(sample["ground_truths"])
            visualize_predictions(
                vis_preds, vis_gts, class_names, output_path=args.vis_output
            )

    elif args.checkpoint_path or args.saved_model_path:
        # Load model
        if args.checkpoint_path:
            model = load_model_from_checkpoint(args.checkpoint_path, model_config)
        else:
            model = load_model_from_saved_model(args.saved_model_path)

        # Load or generate dataset
        if args.data_path:
            print(f"[INFO] Loading validation data from: {args.data_path}")
            # In production, this would load real data from disk
            # For now, fall back to synthetic if path doesn't have expected structure
            if os.path.exists(args.data_path):
                # Attempt to load data from directory structure
                # Expected: data_path/scene_XXX/frame_YYY.npz
                dataset = _load_dataset_from_path(args.data_path, args.scene_ids)
            else:
                print(f"[WARN] Data path not found, using synthetic data")
                dataset = generate_synthetic_dataset(
                    num_scenes=args.num_scenes,
                    num_frames_per_scene=args.num_frames_per_scene,
                )
        else:
            print("[INFO] No data path specified, using synthetic data for model eval")
            dataset = generate_synthetic_dataset(
                num_scenes=args.num_scenes,
                num_frames_per_scene=args.num_frames_per_scene,
            )

        # Filter scenes if specified
        if args.scene_ids is not None:
            dataset = [s for s in dataset if s["scene_id"] in args.scene_ids]
            print(f"[INFO] Filtered to {len(dataset)} samples from scenes: {args.scene_ids}")

        # Run evaluation with model
        metrics = evaluate_with_model(
            model=model,
            dataset=dataset,
            class_names=class_names,
            thresholds=thresholds,
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            compute_frechet=compute_frechet,
        )

        # Visualization
        if args.visualize:
            # Collect some predictions and GTs for visualization
            vis_preds = []
            vis_gts = []
            for sample in dataset[:5]:
                vis_gts.extend(sample["ground_truths"])

            # Run inference on a few samples for visualization
            sample_images = np.array([s["image"] for s in dataset[:4]])
            raw_output, _ = model(
                tf.constant(sample_images, dtype=tf.float32),
                prev_state=None,
                training=False,
            )
            vis_preds = decode_model_output(
                raw_output, class_names, confidence_threshold=args.confidence_threshold
            )
            for p in vis_preds:
                p.pop("batch_idx", None)

            visualize_predictions(
                vis_preds, vis_gts, class_names, output_path=args.vis_output
            )
    else:
        print("[ERROR] Must specify one of: --synthetic, --checkpoint-path, or --saved-model-path")
        sys.exit(1)

    # Print results
    print_results(metrics)

    # Save results
    if args.output_json:
        save_results_to_json(metrics, args.output_json)

    return metrics


def _load_dataset_from_path(
    data_path: str, scene_ids: Optional[List[int]] = None
) -> List[Dict]:
    """
    Load validation dataset from disk.

    Expected directory structure:
        data_path/
            scene_000/
                frame_000.npz  (contains: image, gt_points, gt_classes)
                frame_001.npz
                ...
            scene_001/
                ...

    Args:
        data_path: Root directory of validation data.
        scene_ids: Optional list of scene IDs to load.

    Returns:
        List of sample dicts.
    """
    dataset = []
    scene_dirs = sorted(
        [d for d in os.listdir(data_path) if d.startswith("scene_")]
    )

    for scene_dir in scene_dirs:
        scene_id = int(scene_dir.split("_")[1])
        if scene_ids is not None and scene_id not in scene_ids:
            continue

        scene_path = os.path.join(data_path, scene_dir)
        frame_files = sorted(
            [f for f in os.listdir(scene_path) if f.endswith(".npz")]
        )

        for frame_idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(scene_path, frame_file)
            try:
                data = np.load(frame_path, allow_pickle=True)
                image = data["image"].astype(np.float32)

                # Reconstruct ground truths from stored arrays
                gt_points_list = data["gt_points"]  # List of (N, 2) arrays
                gt_classes = data["gt_classes"]  # List of class name strings

                ground_truths = []
                for pts, cls in zip(gt_points_list, gt_classes):
                    ground_truths.append({"points": pts, "class": str(cls)})

                dataset.append(
                    {
                        "image": image,
                        "predictions": [],  # Will be filled by model
                        "ground_truths": ground_truths,
                        "scene_id": scene_id,
                        "frame_id": frame_idx,
                        "is_first_frame": frame_idx == 0,
                    }
                )
            except Exception as e:
                print(f"[WARN] Failed to load {frame_path}: {e}")
                continue

    print(f"[INFO] Loaded {len(dataset)} samples from {data_path}")
    return dataset


if __name__ == "__main__":
    main()
