"""
HDMapNet TensorFlow 2 Evaluation Script.

Evaluates a trained HDMapNet model on validation data, computing:
  - Per-class IoU and mean IoU (semantic segmentation quality)
  - Chamfer distance between predicted and ground-truth vectorized map elements
  - Per-class precision and recall at distance threshold 0.5

BEV grid: 200x200 covering 60m x 30m (resolution: 0.3m per pixel)
Semantic classes: lane dividers (0), road boundaries (1), pedestrian crossings (2)

Usage:
    python evaluate.py \
        --checkpoint_dir ./checkpoints \
        --data_dir ./val_data \
        --output_file ./results.json \
        --batch_size 1 \
        --view_transform lss
"""

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from scipy import ndimage


# =============================================================================
# Constants
# =============================================================================

BEV_HEIGHT = 200
BEV_WIDTH = 200
NUM_CLASSES = 3
INSTANCE_EMB_DIM = 16
DIRECTION_DIM = 2
NUM_CAMERAS = 6
IMG_HEIGHT = 128
IMG_WIDTH = 352

CLASS_NAMES = ["lane_dividers", "road_boundaries", "ped_crossings"]

# BEV spatial parameters
BEV_X_RANGE = 60.0  # meters (total lateral extent)
BEV_Y_RANGE = 30.0  # meters (total longitudinal extent)
RESOLUTION = BEV_X_RANGE / BEV_WIDTH  # 0.3 m/pixel

# Thresholds
SEMANTIC_THRESHOLD = 0.5
CHAMFER_DISTANCE_THRESHOLD = 0.5  # pixels for precision/recall
EMBEDDING_DISTANCE_THRESHOLD = 1.5
MIN_INSTANCE_PIXELS = 10


# =============================================================================
# Model Loading
# =============================================================================

class ImageEncoder(tf.keras.layers.Layer):
    """EfficientNet-B0 based image encoder for each camera view."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        )
        self.neck = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 1, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

    def call(self, images, training=False):
        batch_size = tf.shape(images)[0]
        imgs_flat = tf.reshape(images, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
        features = self.backbone(imgs_flat, training=training)
        features = self.neck(features, training=training)
        fh, fw = features.shape[1], features.shape[2]
        features = tf.reshape(features, (batch_size, NUM_CAMERAS, fh, fw, 256))
        return features


class IPMViewTransform(tf.keras.layers.Layer):
    """Inverse Perspective Mapping view transform."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(BEV_HEIGHT * BEV_WIDTH),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((BEV_HEIGHT, BEV_WIDTH, 1)),
        ])
        self.combine = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

    def call(self, features, training=False):
        pooled = tf.reduce_mean(features, axis=[2, 3])
        bev_per_cam = []
        for i in range(NUM_CAMERAS):
            cam_feat = pooled[:, i, :]
            bev_map = self.fc(cam_feat, training=training)
            bev_per_cam.append(bev_map)
        bev_concat = tf.concat(bev_per_cam, axis=-1)
        bev = self.combine(bev_concat, training=training)
        return bev


class LSSViewTransform(tf.keras.layers.Layer):
    """Lift-Splat-Shoot style view transform."""

    def __init__(self, num_depth_bins=41, **kwargs):
        super().__init__(**kwargs)
        self.num_depth_bins = num_depth_bins
        self.depth_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(num_depth_bins, 1, padding="same"),
        ])
        self.bev_pool = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.reduce = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(BEV_HEIGHT * BEV_WIDTH * 4),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((BEV_HEIGHT, BEV_WIDTH, 4)),
        ])
        self.final_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])

    def call(self, features, training=False):
        cam_bevs = []
        for i in range(NUM_CAMERAS):
            cam_feat = features[:, i, :, :, :]
            depth_logits = self.depth_net(cam_feat, training=training)
            depth_probs = tf.nn.softmax(depth_logits, axis=-1)
            lifted = tf.expand_dims(depth_probs, -1) * tf.expand_dims(cam_feat, 3)
            lifted = tf.reduce_sum(lifted, axis=3)
            cam_bevs.append(lifted)
        combined = tf.concat(cam_bevs, axis=-1)
        pooled = self.bev_pool(combined, training=training)
        bev = self.reduce(pooled, training=training)
        bev = self.final_conv(bev, training=training)
        return bev


class BEVDecoder(tf.keras.layers.Layer):
    """Decodes BEV features into semantic, instance, and direction heads."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
        ])
        self.semantic_head = tf.keras.layers.Conv2D(
            NUM_CLASSES, 1, padding="same", name="semantic_logits"
        )
        self.instance_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(INSTANCE_EMB_DIM, 1, padding="same"),
        ], name="instance_embedding")
        self.direction_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(DIRECTION_DIM, 1, padding="same"),
        ], name="direction")

    def call(self, bev_features, training=False):
        shared = self.shared_conv(bev_features, training=training)
        semantic_logits = self.semantic_head(shared)
        instance_emb = self.instance_head(shared, training=training)
        direction = self.direction_head(shared, training=training)
        return semantic_logits, instance_emb, direction


def build_hdmapnet_model(view_transform: str = "lss") -> tf.keras.Model:
    """Build the full HDMapNet model for checkpoint restoration."""
    images_input = tf.keras.Input(
        shape=(NUM_CAMERAS, IMG_HEIGHT, IMG_WIDTH, 3), name="images"
    )

    encoder = ImageEncoder(name="image_encoder")
    if view_transform == "ipm":
        view_tf = IPMViewTransform(name="view_transform")
    else:
        view_tf = LSSViewTransform(name="view_transform")
    decoder = BEVDecoder(name="bev_decoder")

    features = encoder(images_input)
    bev_features = view_tf(features)
    semantic_logits, instance_emb, direction = decoder(bev_features)

    model = tf.keras.Model(
        inputs=images_input,
        outputs={
            "semantic_logits": semantic_logits,
            "instance_embedding": instance_emb,
            "direction": direction,
        },
        name="HDMapNet",
    )
    return model


def load_model(checkpoint_dir: str, view_transform: str = "lss") -> tf.keras.Model:
    """Load a trained HDMapNet model from a checkpoint directory.

    Supports SavedModel, .keras files, .h5 files, and TF checkpoints.

    Args:
        checkpoint_dir: path to the checkpoint directory
        view_transform: 'ipm' or 'lss' (needed for checkpoint-based restoration)

    Returns:
        Loaded tf.keras.Model ready for inference
    """
    # Try SavedModel format
    if os.path.exists(os.path.join(checkpoint_dir, "saved_model.pb")):
        print(f"[INFO] Loading SavedModel from: {checkpoint_dir}")
        model = tf.saved_model.load(checkpoint_dir)
        return model

    saved_model_subdir = os.path.join(checkpoint_dir, "saved_model")
    if os.path.isdir(saved_model_subdir) and os.path.exists(
        os.path.join(saved_model_subdir, "saved_model.pb")
    ):
        print(f"[INFO] Loading SavedModel from: {saved_model_subdir}")
        model = tf.saved_model.load(saved_model_subdir)
        return model

    # Try .keras file
    if os.path.isdir(checkpoint_dir):
        for fname in os.listdir(checkpoint_dir):
            if fname.endswith(".keras"):
                keras_path = os.path.join(checkpoint_dir, fname)
                print(f"[INFO] Loading Keras model from: {keras_path}")
                model = tf.keras.models.load_model(keras_path)
                return model

    # Try .h5 file
    for ext in [".h5", ".hdf5"]:
        candidate = os.path.join(checkpoint_dir, f"hdmapnet{ext}")
        if os.path.exists(candidate):
            print(f"[INFO] Loading model from: {candidate}")
            model = tf.keras.models.load_model(candidate)
            return model

    # Try TF checkpoint
    checkpoint_prefix = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_prefix is not None:
        print(f"[INFO] Found checkpoint: {checkpoint_prefix}")
        model = build_hdmapnet_model(view_transform=view_transform)
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(checkpoint_prefix)
        status.expect_partial()
        print("[INFO] Checkpoint restored (expect_partial).")
        return model

    raise FileNotFoundError(
        f"Could not find a valid model in '{checkpoint_dir}'. "
        "Expected a SavedModel, .keras, .h5, or TF checkpoint."
    )


# =============================================================================
# Data Pipeline
# =============================================================================

def load_npz_sample(file_path: str) -> Dict[str, np.ndarray]:
    """Load a single validation .npz file.

    Expected keys:
        - images: [6, 128, 352, 3] float32
        - extrinsics: [6, 4, 4] float32
        - intrinsics: [6, 3, 3] float32
        - semantic_masks: [200, 200, 3] float32 (binary ground truth)
        - instance_masks: [200, 200] int32
        - direction_masks: [200, 200, 2] float32

    Optionally (for Chamfer distance):
        - gt_polylines: list of arrays, each shape [N, 2], per class

    Returns:
        Dictionary with numpy arrays.
    """
    data = np.load(file_path, allow_pickle=True)
    sample = {}

    sample["images"] = data["images"].astype(np.float32)
    sample["extrinsics"] = data["extrinsics"].astype(np.float32)
    sample["intrinsics"] = data["intrinsics"].astype(np.float32)
    sample["semantic_masks"] = data["semantic_masks"].astype(np.float32)
    sample["instance_masks"] = data["instance_masks"].astype(np.int32)
    sample["direction_masks"] = data["direction_masks"].astype(np.float32)

    # Optional ground-truth polylines for Chamfer distance
    if "gt_polylines" in data:
        sample["gt_polylines"] = data["gt_polylines"]

    return sample


def build_eval_dataset(data_dir: str, batch_size: int) -> tf.data.Dataset:
    """Build a tf.data.Dataset pipeline for evaluation.

    Args:
        data_dir: directory containing .npz validation files
        batch_size: evaluation batch size

    Returns:
        tf.data.Dataset yielding (images, extrinsics, intrinsics,
                                   semantic_masks, instance_masks, direction_masks)
    """
    file_pattern = os.path.join(data_dir, "*.npz")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        raise ValueError(f"No .npz files found in {data_dir}")

    print(f"[INFO] Found {len(file_list)} validation samples in: {data_dir}")

    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    def _load_npz(path_bytes):
        path_str = path_bytes.numpy().decode("utf-8")
        data = np.load(path_str)
        images = data["images"].astype(np.float32)
        extrinsics = data["extrinsics"].astype(np.float32)
        intrinsics = data["intrinsics"].astype(np.float32)
        semantic_masks = data["semantic_masks"].astype(np.float32)
        instance_masks = data["instance_masks"].astype(np.int32)
        direction_masks = data["direction_masks"].astype(np.float32)
        return images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks

    def parse_npz_file(file_path):
        images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks = tf.py_function(
            _load_npz,
            [file_path],
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.float32],
        )
        images.set_shape([NUM_CAMERAS, IMG_HEIGHT, IMG_WIDTH, 3])
        extrinsics.set_shape([NUM_CAMERAS, 4, 4])
        intrinsics.set_shape([NUM_CAMERAS, 3, 3])
        semantic_masks.set_shape([BEV_HEIGHT, BEV_WIDTH, NUM_CLASSES])
        instance_masks.set_shape([BEV_HEIGHT, BEV_WIDTH])
        direction_masks.set_shape([BEV_HEIGHT, BEV_WIDTH, 2])
        return images, extrinsics, intrinsics, semantic_masks, instance_masks, direction_masks

    dataset = dataset.map(parse_npz_file, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, file_list


# =============================================================================
# Post-Processing
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def threshold_predictions(logits: np.ndarray, threshold: float = SEMANTIC_THRESHOLD) -> np.ndarray:
    """Apply sigmoid and threshold to get binary masks.

    Args:
        logits: [H, W, C] raw model logits (pre-sigmoid)
        threshold: decision threshold

    Returns:
        binary masks [H, W, C] uint8
    """
    probs = sigmoid(logits)
    return (probs >= threshold).astype(np.uint8)


def extract_boundary_points(binary_mask: np.ndarray) -> np.ndarray:
    """Extract boundary/contour points from a binary mask.

    Uses morphological erosion to find boundary pixels. Returns the
    (row, col) coordinates of all boundary pixels.

    Args:
        binary_mask: [H, W] binary mask (uint8 or bool)

    Returns:
        points: [N, 2] array of (row, col) boundary pixel coordinates.
                Returns empty array shape [0, 2] if no boundary found.
    """
    mask = binary_mask.astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # Erode the mask by 1 pixel
    struct = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=1).astype(np.uint8)

    # Boundary = original - eroded
    boundary = mask - eroded
    boundary_points = np.argwhere(boundary > 0).astype(np.float64)

    if boundary_points.shape[0] == 0:
        # If erosion removed everything (very thin structure), use all mask pixels
        boundary_points = np.argwhere(mask > 0).astype(np.float64)

    return boundary_points


def extract_polylines_from_mask(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Extract connected components and their boundary polylines from a binary mask.

    Each connected component's boundary is extracted as a separate polyline.

    Args:
        binary_mask: [H, W] binary mask

    Returns:
        List of polyline arrays, each shape [N_i, 2] (row, col coordinates).
    """
    mask = binary_mask.astype(np.uint8)
    if mask.sum() == 0:
        return []

    labeled, num_components = ndimage.label(mask)
    polylines = []

    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled == comp_id).astype(np.uint8)
        if comp_mask.sum() < MIN_INSTANCE_PIXELS:
            continue
        boundary_pts = extract_boundary_points(comp_mask)
        if boundary_pts.shape[0] > 0:
            polylines.append(boundary_pts)

    return polylines


def cluster_instances(
    semantic_masks: np.ndarray,
    instance_embedding: np.ndarray,
) -> np.ndarray:
    """Cluster instance embeddings using connected components + embedding similarity.

    For each semantic class, find connected components, then merge components
    whose mean embeddings are within EMBEDDING_DISTANCE_THRESHOLD.

    Args:
        semantic_masks: [H, W, NUM_CLASSES] binary uint8
        instance_embedding: [H, W, INSTANCE_EMB_DIM] float32

    Returns:
        instance_map: [H, W] int32 with unique instance IDs (0 = background)
    """
    h, w = semantic_masks.shape[:2]
    instance_map = np.zeros((h, w), dtype=np.int32)
    current_id = 1

    for cls_idx in range(NUM_CLASSES):
        class_mask = semantic_masks[:, :, cls_idx]
        if class_mask.sum() == 0:
            continue

        labeled, num_components = ndimage.label(class_mask)

        component_embeddings = {}
        component_pixels = {}
        for comp_id in range(1, num_components + 1):
            comp_mask = labeled == comp_id
            pixel_count = comp_mask.sum()
            if pixel_count < MIN_INSTANCE_PIXELS:
                continue
            emb_mean = instance_embedding[comp_mask].mean(axis=0)
            component_embeddings[comp_id] = emb_mean
            component_pixels[comp_id] = comp_mask

        comp_ids = list(component_embeddings.keys())
        merged = {}
        cluster_counter = 0

        for i, cid in enumerate(comp_ids):
            if cid in merged:
                continue
            cluster_counter += 1
            merged[cid] = cluster_counter
            emb_i = component_embeddings[cid]

            for j in range(i + 1, len(comp_ids)):
                cid_j = comp_ids[j]
                if cid_j in merged:
                    continue
                emb_j = component_embeddings[cid_j]
                dist = np.linalg.norm(emb_i - emb_j)
                if dist < EMBEDDING_DISTANCE_THRESHOLD:
                    merged[cid_j] = merged[cid]

        for cid, cluster_id in merged.items():
            instance_map[component_pixels[cid]] = current_id + cluster_id - 1

        if cluster_counter > 0:
            current_id += cluster_counter

    return instance_map


# =============================================================================
# Metric Computation
# =============================================================================

class IoUAccumulator:
    """Accumulates true positives, false positives, false negatives across batches
    for per-class IoU computation."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.tp = np.zeros(num_classes, dtype=np.int64)
        self.fp = np.zeros(num_classes, dtype=np.int64)
        self.fn = np.zeros(num_classes, dtype=np.int64)

    def update(self, pred_masks: np.ndarray, gt_masks: np.ndarray):
        """Update counts with a batch of predictions and ground truth.

        Args:
            pred_masks: [B, H, W, C] binary predictions (uint8)
            gt_masks: [B, H, W, C] binary ground truth (float32 or uint8)
        """
        pred = pred_masks.astype(np.bool_)
        gt = gt_masks.astype(np.bool_)

        for cls_idx in range(self.num_classes):
            p = pred[..., cls_idx]
            g = gt[..., cls_idx]
            self.tp[cls_idx] += np.logical_and(p, g).sum()
            self.fp[cls_idx] += np.logical_and(p, np.logical_not(g)).sum()
            self.fn[cls_idx] += np.logical_and(np.logical_not(p), g).sum()

    def compute_iou(self) -> Tuple[np.ndarray, float]:
        """Compute per-class IoU and mean IoU.

        Returns:
            per_class_iou: [num_classes] array
            mean_iou: scalar float
        """
        denominator = self.tp + self.fp + self.fn
        per_class_iou = np.where(
            denominator > 0,
            self.tp.astype(np.float64) / denominator.astype(np.float64),
            0.0,
        )
        # Mean IoU over classes that have at least one GT pixel
        valid_classes = (self.tp + self.fn) > 0
        if valid_classes.sum() > 0:
            mean_iou = per_class_iou[valid_classes].mean()
        else:
            mean_iou = 0.0
        return per_class_iou, float(mean_iou)


class PrecisionRecallAccumulator:
    """Accumulates precision and recall at a distance threshold for vectorized
    map element evaluation."""

    def __init__(self, num_classes: int, distance_threshold: float = CHAMFER_DISTANCE_THRESHOLD):
        self.num_classes = num_classes
        self.distance_threshold = distance_threshold
        # For precision: fraction of predicted points within threshold of GT
        self.total_pred_points = np.zeros(num_classes, dtype=np.int64)
        self.matched_pred_points = np.zeros(num_classes, dtype=np.int64)
        # For recall: fraction of GT points within threshold of prediction
        self.total_gt_points = np.zeros(num_classes, dtype=np.int64)
        self.matched_gt_points = np.zeros(num_classes, dtype=np.int64)

    def update(self, pred_points: np.ndarray, gt_points: np.ndarray, class_idx: int):
        """Update precision/recall counts for one class in one sample.

        Args:
            pred_points: [M, 2] predicted boundary points
            gt_points: [N, 2] ground truth boundary points
            class_idx: which semantic class
        """
        num_pred = pred_points.shape[0]
        num_gt = gt_points.shape[0]

        if num_pred == 0 and num_gt == 0:
            return

        self.total_pred_points[class_idx] += num_pred
        self.total_gt_points[class_idx] += num_gt

        if num_pred == 0 or num_gt == 0:
            # No matches possible
            return

        # Compute pairwise distances using chunked computation to save memory
        # For large point sets, process in chunks
        chunk_size = 5000

        # Precision: for each pred point, find min distance to any GT point
        for start in range(0, num_pred, chunk_size):
            end = min(start + chunk_size, num_pred)
            pred_chunk = pred_points[start:end]  # [chunk, 2]
            # Compute distances from pred_chunk to all gt_points
            # pred_chunk[:, None, :] - gt_points[None, :, :] => [chunk, N, 2]
            diffs = pred_chunk[:, np.newaxis, :] - gt_points[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))  # [chunk, N]
            min_dists = dists.min(axis=1)  # [chunk]
            self.matched_pred_points[class_idx] += (min_dists <= self.distance_threshold).sum()

        # Recall: for each GT point, find min distance to any pred point
        for start in range(0, num_gt, chunk_size):
            end = min(start + chunk_size, num_gt)
            gt_chunk = gt_points[start:end]  # [chunk, 2]
            diffs = gt_chunk[:, np.newaxis, :] - pred_points[np.newaxis, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))  # [chunk, M]
            min_dists = dists.min(axis=1)  # [chunk]
            self.matched_gt_points[class_idx] += (min_dists <= self.distance_threshold).sum()

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-class precision and recall.

        Returns:
            precision: [num_classes] array
            recall: [num_classes] array
        """
        precision = np.where(
            self.total_pred_points > 0,
            self.matched_pred_points.astype(np.float64) / self.total_pred_points.astype(np.float64),
            0.0,
        )
        recall = np.where(
            self.total_gt_points > 0,
            self.matched_gt_points.astype(np.float64) / self.total_gt_points.astype(np.float64),
            0.0,
        )
        return precision, recall


def compute_chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
) -> float:
    """Compute symmetric Chamfer distance between two point sets.

    Chamfer distance = (1/|P|) * sum_{p in P} min_{q in Q} ||p - q||
                     + (1/|Q|) * sum_{q in Q} min_{p in P} ||q - p||

    Args:
        pred_points: [M, 2] predicted points
        gt_points: [N, 2] ground truth points

    Returns:
        Chamfer distance (scalar). Returns 0.0 if both sets are empty,
        or a large penalty value if only one set is empty.
    """
    num_pred = pred_points.shape[0]
    num_gt = gt_points.shape[0]

    if num_pred == 0 and num_gt == 0:
        return 0.0
    if num_pred == 0 or num_gt == 0:
        # Penalty: max possible distance on the BEV grid (diagonal)
        return float(np.sqrt(BEV_HEIGHT ** 2 + BEV_WIDTH ** 2))

    chunk_size = 5000

    # Forward direction: pred -> gt
    forward_sum = 0.0
    for start in range(0, num_pred, chunk_size):
        end = min(start + chunk_size, num_pred)
        pred_chunk = pred_points[start:end]
        diffs = pred_chunk[:, np.newaxis, :] - gt_points[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        forward_sum += dists.min(axis=1).sum()

    # Backward direction: gt -> pred
    backward_sum = 0.0
    for start in range(0, num_gt, chunk_size):
        end = min(start + chunk_size, num_gt)
        gt_chunk = gt_points[start:end]
        diffs = gt_chunk[:, np.newaxis, :] - pred_points[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        backward_sum += dists.min(axis=1).sum()

    chamfer = (forward_sum / num_pred) + (backward_sum / num_gt)
    return float(chamfer)


class ChamferDistanceAccumulator:
    """Accumulates Chamfer distances across samples for per-class averaging."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.distances = [[] for _ in range(num_classes)]

    def update(self, pred_points: np.ndarray, gt_points: np.ndarray, class_idx: int):
        """Add a Chamfer distance measurement for one class in one sample.

        Args:
            pred_points: [M, 2] predicted boundary points
            gt_points: [N, 2] ground truth boundary points
            class_idx: semantic class index
        """
        cd = compute_chamfer_distance(pred_points, gt_points)
        self.distances[class_idx].append(cd)

    def compute(self) -> Tuple[np.ndarray, float]:
        """Compute per-class mean Chamfer distance and overall mean.

        Returns:
            per_class_chamfer: [num_classes] array of mean Chamfer distances
            mean_chamfer: overall mean across classes with data
        """
        per_class_chamfer = np.zeros(self.num_classes, dtype=np.float64)
        valid_count = 0

        for cls_idx in range(self.num_classes):
            if len(self.distances[cls_idx]) > 0:
                per_class_chamfer[cls_idx] = np.mean(self.distances[cls_idx])
                valid_count += 1
            else:
                per_class_chamfer[cls_idx] = 0.0

        if valid_count > 0:
            mean_chamfer = per_class_chamfer[per_class_chamfer > 0].mean() if (per_class_chamfer > 0).any() else 0.0
        else:
            mean_chamfer = 0.0

        return per_class_chamfer, float(mean_chamfer)


# =============================================================================
# Model Inference
# =============================================================================

def run_model_inference(
    model,
    images: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run forward pass through the model.

    Handles both Keras models and SavedModel objects.

    Args:
        model: loaded model (Keras or SavedModel)
        images: [B, 6, 128, 352, 3] float32

    Returns:
        Dict with 'semantic_logits', 'instance_embedding', 'direction'
    """
    images_tensor = tf.constant(images, dtype=tf.float32)

    if isinstance(model, tf.keras.Model):
        outputs = model(images_tensor, training=False)
    else:
        if hasattr(model, "__call__"):
            outputs = model(images_tensor)
        elif hasattr(model, "signatures"):
            serve_fn = model.signatures.get(
                "serving_default", list(model.signatures.values())[0]
            )
            outputs = serve_fn(images=images_tensor)
        else:
            raise RuntimeError("Cannot determine how to call the loaded model.")

    # Normalize output keys to a consistent format
    if isinstance(outputs, dict):
        result = {}
        for key in ["semantic_logits", "semantic", "output_0"]:
            if key in outputs:
                val = outputs[key]
                result["semantic_logits"] = val.numpy() if hasattr(val, "numpy") else np.array(val)
                break
        for key in ["instance_embedding", "instance", "output_1"]:
            if key in outputs:
                val = outputs[key]
                result["instance_embedding"] = val.numpy() if hasattr(val, "numpy") else np.array(val)
                break
        for key in ["direction", "output_2"]:
            if key in outputs:
                val = outputs[key]
                result["direction"] = val.numpy() if hasattr(val, "numpy") else np.array(val)
                break
        return result
    elif isinstance(outputs, (list, tuple)):
        return {
            "semantic_logits": outputs[0].numpy(),
            "instance_embedding": outputs[1].numpy(),
            "direction": outputs[2].numpy(),
        }
    else:
        raise RuntimeError(f"Unexpected model output type: {type(outputs)}")


# =============================================================================
# Evaluation Loop
# =============================================================================

def evaluate(args: argparse.Namespace) -> Dict:
    """Main evaluation function.

    Loads model, iterates over validation data, computes all metrics, and
    returns a results dictionary.

    Args:
        args: parsed command-line arguments

    Returns:
        Dictionary with all evaluation results
    """
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("HDMapNet TensorFlow 2 Evaluation")
    print("=" * 70)
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Data directory:       {args.data_dir}")
    print(f"  Output file:          {args.output_file}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  View transform:       {args.view_transform}")
    print("=" * 70)

    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"[INFO] Using {len(gpus)} GPU(s) with memory growth enabled.")
    else:
        print("[INFO] No GPUs detected. Running on CPU.")

    # -------------------------------------------------------------------------
    # Load Model
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading model...")
    model = load_model(args.checkpoint_dir, view_transform=args.view_transform)
    print("  Model loaded successfully.")

    # -------------------------------------------------------------------------
    # Build Dataset
    # -------------------------------------------------------------------------
    print("\n[2/4] Loading validation dataset...")
    dataset, file_list = build_eval_dataset(args.data_dir, args.batch_size)
    num_samples = len(file_list)
    print(f"  Total validation samples: {num_samples}")

    # -------------------------------------------------------------------------
    # Initialize Metric Accumulators
    # -------------------------------------------------------------------------
    iou_accum = IoUAccumulator(NUM_CLASSES)
    chamfer_accum = ChamferDistanceAccumulator(NUM_CLASSES)
    pr_accum = PrecisionRecallAccumulator(NUM_CLASSES, distance_threshold=CHAMFER_DISTANCE_THRESHOLD)

    # -------------------------------------------------------------------------
    # Evaluation Loop
    # -------------------------------------------------------------------------
    print("\n[3/4] Running evaluation...")
    eval_start_time = time.time()
    samples_processed = 0
    batch_count = 0

    for batch_data in dataset:
        images, extrinsics, intrinsics, semantic_masks_gt, instance_masks_gt, direction_masks_gt = batch_data

        # Convert to numpy for metric computation
        images_np = images.numpy()
        semantic_gt_np = semantic_masks_gt.numpy()  # [B, 200, 200, 3]

        # Run model inference
        outputs = run_model_inference(model, images_np)
        semantic_logits = outputs["semantic_logits"]  # [B, 200, 200, 3]
        instance_embedding = outputs["instance_embedding"]  # [B, 200, 200, 16]

        batch_size_actual = semantic_logits.shape[0]

        # Apply sigmoid and threshold to get binary predictions
        semantic_probs = sigmoid(semantic_logits)
        semantic_pred_binary = (semantic_probs >= SEMANTIC_THRESHOLD).astype(np.uint8)

        # --- IoU ---
        iou_accum.update(semantic_pred_binary, semantic_gt_np)

        # --- Per-sample Chamfer distance and Precision/Recall ---
        for b in range(batch_size_actual):
            pred_masks_b = semantic_pred_binary[b]  # [200, 200, 3]
            gt_masks_b = (semantic_gt_np[b] >= SEMANTIC_THRESHOLD).astype(np.uint8)  # [200, 200, 3]

            for cls_idx in range(NUM_CLASSES):
                pred_mask_cls = pred_masks_b[:, :, cls_idx]
                gt_mask_cls = gt_masks_b[:, :, cls_idx]

                # Extract boundary points from predicted mask
                pred_boundary = extract_boundary_points(pred_mask_cls)
                # Extract boundary points from GT mask
                gt_boundary = extract_boundary_points(gt_mask_cls)

                # Chamfer distance
                chamfer_accum.update(pred_boundary, gt_boundary, cls_idx)

                # Precision and recall at threshold
                pr_accum.update(pred_boundary, gt_boundary, cls_idx)

        samples_processed += batch_size_actual
        batch_count += 1

        # Progress reporting
        if batch_count % 10 == 0 or samples_processed >= num_samples:
            elapsed = time.time() - eval_start_time
            rate = samples_processed / max(elapsed, 1e-6)
            print(
                f"  Processed {samples_processed}/{num_samples} samples "
                f"({elapsed:.1f}s, {rate:.1f} samples/s)"
            )

    eval_duration = time.time() - eval_start_time

    # -------------------------------------------------------------------------
    # Compute Final Metrics
    # -------------------------------------------------------------------------
    print("\n[4/4] Computing final metrics...")

    per_class_iou, mean_iou = iou_accum.compute_iou()
    per_class_chamfer, mean_chamfer = chamfer_accum.compute()
    precision, recall = pr_accum.compute()

    # Convert Chamfer distance from pixels to meters
    per_class_chamfer_m = per_class_chamfer * RESOLUTION
    mean_chamfer_m = mean_chamfer * RESOLUTION

    # -------------------------------------------------------------------------
    # Print Results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nSamples evaluated: {samples_processed}")
    print(f"Evaluation time:   {eval_duration:.1f}s ({samples_processed / max(eval_duration, 1e-6):.1f} samples/s)")

    print(f"\n--- IoU (Intersection over Union) ---")
    print(f"{'Class':<25} {'IoU':>10}")
    print("-" * 37)
    for cls_idx in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[cls_idx]:<25} {per_class_iou[cls_idx]:>10.4f}")
    print("-" * 37)
    print(f"{'Mean IoU':<25} {mean_iou:>10.4f}")

    print(f"\n--- Chamfer Distance ---")
    print(f"{'Class':<25} {'CD (px)':>10} {'CD (m)':>10}")
    print("-" * 47)
    for cls_idx in range(NUM_CLASSES):
        print(
            f"{CLASS_NAMES[cls_idx]:<25} "
            f"{per_class_chamfer[cls_idx]:>10.4f} "
            f"{per_class_chamfer_m[cls_idx]:>10.4f}"
        )
    print("-" * 47)
    print(f"{'Mean':<25} {mean_chamfer:>10.4f} {mean_chamfer_m:>10.4f}")

    print(f"\n--- Precision / Recall @ threshold={CHAMFER_DISTANCE_THRESHOLD} px ---")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 57)
    f1_scores = np.zeros(NUM_CLASSES, dtype=np.float64)
    for cls_idx in range(NUM_CLASSES):
        p = precision[cls_idx]
        r = recall[cls_idx]
        f1 = 2.0 * p * r / max(p + r, 1e-8)
        f1_scores[cls_idx] = f1
        print(
            f"{CLASS_NAMES[cls_idx]:<25} "
            f"{p:>10.4f} "
            f"{r:>10.4f} "
            f"{f1:>10.4f}"
        )
    mean_precision = precision.mean()
    mean_recall = recall.mean()
    mean_f1 = f1_scores.mean()
    print("-" * 57)
    print(f"{'Mean':<25} {mean_precision:>10.4f} {mean_recall:>10.4f} {mean_f1:>10.4f}")

    print("\n" + "=" * 70)

    # -------------------------------------------------------------------------
    # Assemble Results Dictionary
    # -------------------------------------------------------------------------
    results = {
        "metadata": {
            "checkpoint_dir": args.checkpoint_dir,
            "data_dir": args.data_dir,
            "view_transform": args.view_transform,
            "batch_size": args.batch_size,
            "num_samples": samples_processed,
            "evaluation_time_seconds": round(eval_duration, 2),
            "bev_grid_size": [BEV_HEIGHT, BEV_WIDTH],
            "bev_coverage_meters": [BEV_X_RANGE, BEV_Y_RANGE],
            "resolution_m_per_pixel": RESOLUTION,
            "semantic_threshold": SEMANTIC_THRESHOLD,
            "chamfer_threshold_pixels": CHAMFER_DISTANCE_THRESHOLD,
            "class_names": CLASS_NAMES,
        },
        "iou": {
            "per_class": {
                CLASS_NAMES[i]: round(float(per_class_iou[i]), 6) for i in range(NUM_CLASSES)
            },
            "mean_iou": round(float(mean_iou), 6),
            "tp_per_class": {
                CLASS_NAMES[i]: int(iou_accum.tp[i]) for i in range(NUM_CLASSES)
            },
            "fp_per_class": {
                CLASS_NAMES[i]: int(iou_accum.fp[i]) for i in range(NUM_CLASSES)
            },
            "fn_per_class": {
                CLASS_NAMES[i]: int(iou_accum.fn[i]) for i in range(NUM_CLASSES)
            },
        },
        "chamfer_distance": {
            "per_class_pixels": {
                CLASS_NAMES[i]: round(float(per_class_chamfer[i]), 6) for i in range(NUM_CLASSES)
            },
            "per_class_meters": {
                CLASS_NAMES[i]: round(float(per_class_chamfer_m[i]), 6) for i in range(NUM_CLASSES)
            },
            "mean_pixels": round(float(mean_chamfer), 6),
            "mean_meters": round(float(mean_chamfer_m), 6),
        },
        "precision_recall": {
            "threshold_pixels": CHAMFER_DISTANCE_THRESHOLD,
            "precision_per_class": {
                CLASS_NAMES[i]: round(float(precision[i]), 6) for i in range(NUM_CLASSES)
            },
            "recall_per_class": {
                CLASS_NAMES[i]: round(float(recall[i]), 6) for i in range(NUM_CLASSES)
            },
            "f1_per_class": {
                CLASS_NAMES[i]: round(float(f1_scores[i]), 6) for i in range(NUM_CLASSES)
            },
            "mean_precision": round(float(mean_precision), 6),
            "mean_recall": round(float(mean_recall), 6),
            "mean_f1": round(float(mean_f1), 6),
        },
    }

    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")

    return results


# =============================================================================
# Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained HDMapNet model on validation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py --checkpoint_dir ./checkpoints --data_dir ./val --output_file results.json
    python evaluate.py --checkpoint_dir ./ckpt --data_dir ./val --batch_size 4 --view_transform ipm
        """,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory (SavedModel, .keras, .h5, or TF checkpoint).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to validation data directory containing preprocessed .npz files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_results.json",
        help="Path to save detailed results as JSON (default: ./evaluation_results.json).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Evaluation batch size (default: 1).",
    )
    parser.add_argument(
        "--view_transform",
        type=str,
        default="lss",
        choices=["ipm", "lss"],
        help="View transformation method: 'ipm' or 'lss' (default: 'lss').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate(args)
