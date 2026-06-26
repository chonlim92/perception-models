"""
RangeNet++ Evaluation Script for SemanticKITTI (TensorFlow 2)

Loads a trained checkpoint, runs inference on validation sequences,
computes per-class IoU and mean IoU (mIoU), with optional KNN post-processing.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from scipy.spatial import KDTree

# ==============================================================================
# SemanticKITTI Configuration
# ==============================================================================

# 20 training classes (index 0 = unlabeled/ignored in mIoU)
CLASS_NAMES = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

NUM_CLASSES = 20

# SemanticKITTI label remapping: raw label ID -> training ID (0-19)
# Based on the official semantic-kitti-api configuration
LABEL_REMAP = {
    0: 0,       # unlabeled
    1: 0,       # outlier -> unlabeled
    10: 1,      # car
    11: 2,      # bicycle
    13: 5,      # bus -> other-vehicle
    15: 3,      # motorcycle
    16: 5,      # on-rails -> other-vehicle
    18: 4,      # truck
    20: 5,      # other-vehicle
    30: 6,      # person
    31: 7,      # bicyclist
    32: 8,      # motorcyclist
    40: 9,      # road
    44: 10,     # parking
    48: 11,     # sidewalk
    49: 12,     # other-ground
    50: 13,     # building
    51: 14,     # fence
    52: 0,      # other-structure -> unlabeled
    60: 9,      # lane-marking -> road
    70: 15,     # vegetation
    71: 16,     # trunk
    72: 17,     # terrain
    80: 18,     # pole
    81: 19,     # traffic-sign
    99: 0,      # other-object -> unlabeled
    252: 1,     # moving-car -> car
    253: 7,     # moving-bicyclist -> bicyclist
    254: 6,     # moving-person -> person
    255: 8,     # moving-motorcyclist -> motorcyclist
    256: 5,     # moving-on-rails -> other-vehicle
    257: 5,     # moving-bus -> other-vehicle
    258: 4,     # moving-truck -> truck
    259: 5,     # moving-other-vehicle -> other-vehicle
}

# Build a numpy lookup table for fast remapping (max raw label = 259)
REMAP_LUT = np.zeros(260, dtype=np.int32)
for raw_id, train_id in LABEL_REMAP.items():
    REMAP_LUT[raw_id] = train_id

# Spherical projection parameters
PROJ_H = 64
PROJ_W = 2048
FOV_UP = 2.0        # degrees
FOV_DOWN = -24.8    # degrees


# ==============================================================================
# Spherical Projection
# ==============================================================================

def spherical_projection(points, remission):
    """
    Project a 3D point cloud into a 2D range image using spherical projection.

    Args:
        points: (N, 3) numpy array of XYZ coordinates
        remission: (N,) numpy array of remission/intensity values

    Returns:
        proj_range: (H, W) range image
        proj_xyz: (H, W, 3) projected XYZ coordinates
        proj_remission: (H, W) projected remission values
        proj_idx: (H, W) index of the point that projects to each pixel (-1 if empty)
        proj_mask: (H, W) binary mask (1 where a point projects, 0 otherwise)
        point_proj_x: (N,) x-coordinate in the projection for each point
        point_proj_y: (N,) y-coordinate in the projection for each point
    """
    fov_up_rad = FOV_UP * np.pi / 180.0
    fov_down_rad = FOV_DOWN * np.pi / 180.0
    fov_total = abs(fov_up_rad) + abs(fov_down_rad)

    # Compute spherical coordinates
    depth = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    depth_safe = np.maximum(depth, 1e-8)

    pitch = np.arcsin(points[:, 2] / depth_safe)
    yaw = np.arctan2(points[:, 1], points[:, 0])

    # Normalize to image coordinates
    proj_x = 0.5 * (1.0 - yaw / np.pi)  # [0, 1]
    proj_y = 1.0 - (pitch - fov_down_rad) / fov_total  # [0, 1]

    # Scale to image size
    proj_x = proj_x * PROJ_W - 0.5
    proj_y = proj_y * PROJ_H - 0.5

    # Round and clamp
    proj_x = np.clip(np.round(proj_x).astype(np.int32), 0, PROJ_W - 1)
    proj_y = np.clip(np.round(proj_y).astype(np.int32), 0, PROJ_H - 1)

    # Store per-point projection coordinates
    point_proj_x = proj_x.copy()
    point_proj_y = proj_y.copy()

    # Order points by decreasing depth so closer points overwrite farther ones
    order = np.argsort(depth)[::-1]

    # Initialize projection arrays
    proj_range = np.full((PROJ_H, PROJ_W), -1.0, dtype=np.float32)
    proj_xyz = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.float32)
    proj_remission = np.zeros((PROJ_H, PROJ_W), dtype=np.float32)
    proj_idx = np.full((PROJ_H, PROJ_W), -1, dtype=np.int32)
    proj_mask = np.zeros((PROJ_H, PROJ_W), dtype=np.float32)

    # Fill projection
    for idx in order:
        px = proj_x[idx]
        py = proj_y[idx]
        proj_range[py, px] = depth[idx]
        proj_xyz[py, px] = points[idx]
        proj_remission[py, px] = remission[idx]
        proj_idx[py, px] = idx
        proj_mask[py, px] = 1.0

    return proj_range, proj_xyz, proj_remission, proj_idx, proj_mask, point_proj_x, point_proj_y


# ==============================================================================
# Data Loading (SemanticKITTI Format)
# ==============================================================================

def load_point_cloud(bin_path):
    """
    Load a point cloud from a .bin file (SemanticKITTI format).

    Args:
        bin_path: path to the .bin file

    Returns:
        points: (N, 3) XYZ coordinates
        remission: (N,) remission/intensity values
    """
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]
    remission = scan[:, 3]
    return points, remission


def load_labels(label_path):
    """
    Load labels from a .label file (SemanticKITTI format).

    Args:
        label_path: path to the .label file

    Returns:
        sem_labels: (N,) semantic label for each point (remapped to training IDs)
    """
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    # Lower 16 bits are the semantic label, upper 16 bits are instance ID
    sem_labels = (raw_labels & 0xFFFF).astype(np.int32)
    # Remap to training IDs
    # Handle labels outside our LUT range
    sem_labels = np.clip(sem_labels, 0, len(REMAP_LUT) - 1)
    sem_labels = REMAP_LUT[sem_labels]
    return sem_labels


def get_scan_paths(data_dir, sequence):
    """
    Get sorted paths to all scans and labels in a sequence.

    Args:
        data_dir: root directory of SemanticKITTI dataset
        sequence: sequence number (e.g., "08")

    Returns:
        scan_paths: sorted list of .bin file paths
        label_paths: sorted list of .label file paths
    """
    seq_str = str(sequence).zfill(2)
    scan_dir = os.path.join(data_dir, "sequences", seq_str, "velodyne")
    label_dir = os.path.join(data_dir, "sequences", seq_str, "labels")

    if not os.path.isdir(scan_dir):
        raise FileNotFoundError(f"Scan directory not found: {scan_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    scan_files = sorted([f for f in os.listdir(scan_dir) if f.endswith(".bin")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".label")])

    if len(scan_files) != len(label_files):
        raise ValueError(
            f"Mismatch: {len(scan_files)} scans vs {len(label_files)} labels in sequence {seq_str}"
        )

    scan_paths = [os.path.join(scan_dir, f) for f in scan_files]
    label_paths = [os.path.join(label_dir, f) for f in label_files]

    return scan_paths, label_paths


# ==============================================================================
# RangeNet++ Model Definition
# ==============================================================================

class RangeNetPP(tf.keras.Model):
    """
    RangeNet++ encoder-decoder architecture for LiDAR semantic segmentation.

    Input: 5-channel range image (range, x, y, z, remission) of shape (H, W, 5)
    Output: per-pixel class logits of shape (H, W, NUM_CLASSES)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super(RangeNetPP, self).__init__()
        self.num_classes = num_classes

        # Encoder (DarkNet53-inspired backbone)
        self.enc1 = self._encoder_block(64, 2)
        self.enc2 = self._encoder_block(128, 2)
        self.enc3 = self._encoder_block(256, 2)
        self.enc4 = self._encoder_block(512, 2)
        self.enc5 = self._encoder_block(1024, 2)

        # Initial convolution
        self.input_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
        ])

        # Decoder with skip connections
        self.dec5 = self._decoder_block(512)
        self.dec4 = self._decoder_block(256)
        self.dec3 = self._decoder_block(128)
        self.dec2 = self._decoder_block(64)
        self.dec1 = self._decoder_block(32)

        # Final classification head
        self.head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Dropout(0.01),
            tf.keras.layers.Conv2D(num_classes, 1, padding="same"),
        ])

    def _encoder_block(self, filters, stride):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
        ])

    def _decoder_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
        ])

    def call(self, inputs, training=False):
        # Input: (batch, H, W, 5)
        x = self.input_conv(inputs, training=training)

        # Encoder path
        e1 = self.enc1(x, training=training)       # (batch, H/2, W/2, 64)
        e2 = self.enc2(e1, training=training)      # (batch, H/4, W/4, 128)
        e3 = self.enc3(e2, training=training)      # (batch, H/8, W/8, 256)
        e4 = self.enc4(e3, training=training)      # (batch, H/16, W/16, 512)
        e5 = self.enc5(e4, training=training)      # (batch, H/32, W/32, 1024)

        # Decoder path with skip connections
        d5 = self.dec5(e5, training=training) + e4
        d4 = self.dec4(d5, training=training) + e3
        d3 = self.dec3(d4, training=training) + e2
        d2 = self.dec2(d3, training=training) + e1
        d1 = self.dec1(d2, training=training) + x

        # Classification
        logits = self.head(d1, training=training)
        return logits


# ==============================================================================
# KNN Post-Processing
# ==============================================================================

def knn_post_processing(points, predictions, k=5):
    """
    Apply KNN-based post-processing to smooth predictions using majority voting.

    For each predicted point, find K nearest neighbors in the point cloud
    and use majority voting among neighbors to smooth predictions.

    Args:
        points: (N, 3) numpy array of XYZ coordinates
        predictions: (N,) numpy array of predicted class labels
        k: number of nearest neighbors

    Returns:
        smoothed_predictions: (N,) numpy array of smoothed class labels
    """
    print(f"    Building KDTree for {len(points)} points...")
    tree = KDTree(points)

    print(f"    Querying {k} nearest neighbors...")
    distances, indices = tree.query(points, k=k)

    # Majority voting
    neighbor_labels = predictions[indices]  # (N, K)
    smoothed_predictions = np.zeros(len(predictions), dtype=np.int32)

    for i in range(len(predictions)):
        labels = neighbor_labels[i]
        # Count occurrences of each label among neighbors
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        smoothed_predictions[i] = np.argmax(counts)

    return smoothed_predictions


# ==============================================================================
# IoU Computation
# ==============================================================================

class IoUMetric:
    """Accumulates confusion matrix and computes per-class IoU and mIoU."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions, labels):
        """
        Update confusion matrix with a batch of predictions and labels.

        Args:
            predictions: (N,) predicted class labels
            labels: (N,) ground truth class labels
        """
        # Only evaluate on valid labels (ignore class 0 = unlabeled for mIoU)
        mask = (labels >= 0) & (labels < self.num_classes)
        predictions = predictions[mask]
        labels = labels[mask]

        for pred, gt in zip(predictions, labels):
            self.confusion_matrix[gt, pred] += 1

    def update_fast(self, predictions, labels):
        """
        Vectorized confusion matrix update (much faster than element-wise).

        Args:
            predictions: (N,) predicted class labels
            labels: (N,) ground truth class labels
        """
        mask = (labels >= 0) & (labels < self.num_classes)
        predictions = predictions[mask]
        labels = labels[mask]

        # Use np.add.at for fast accumulation
        indices = labels * self.num_classes + predictions
        np.add.at(self.confusion_matrix.ravel(), indices, 1)

    def compute_iou(self):
        """
        Compute per-class IoU and mean IoU.

        Returns:
            per_class_iou: (num_classes,) IoU for each class
            miou: mean IoU (excluding class 0 = unlabeled)
        """
        per_class_iou = np.zeros(self.num_classes, dtype=np.float64)

        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp

            denominator = tp + fp + fn
            if denominator > 0:
                per_class_iou[c] = tp / denominator
            else:
                per_class_iou[c] = 0.0

        # mIoU excludes class 0 (unlabeled)
        valid_classes = per_class_iou[1:]  # classes 1-19
        miou = np.mean(valid_classes)

        return per_class_iou, miou

    def reset(self):
        """Reset the confusion matrix."""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )


# ==============================================================================
# Inference Pipeline
# ==============================================================================

def prepare_input(points, remission):
    """
    Prepare a 5-channel range image input tensor from point cloud data.

    Args:
        points: (N, 3) XYZ coordinates
        remission: (N,) remission/intensity

    Returns:
        input_tensor: (1, H, W, 5) TensorFlow tensor
        proj_idx: (H, W) mapping from pixel to point index
        proj_mask: (H, W) valid pixel mask
    """
    proj_range, proj_xyz, proj_remission, proj_idx, proj_mask, _, _ = spherical_projection(
        points, remission
    )

    # Stack into 5-channel image: [range, x, y, z, remission]
    range_image = np.stack(
        [proj_range, proj_xyz[:, :, 0], proj_xyz[:, :, 1], proj_xyz[:, :, 2], proj_remission],
        axis=-1,
    )  # (H, W, 5)

    # Replace invalid pixels (range=-1) with zeros
    invalid_mask = proj_range < 0
    range_image[invalid_mask] = 0.0

    # Normalize range channel
    max_range = np.max(proj_range[~invalid_mask]) if np.any(~invalid_mask) else 1.0
    range_image[:, :, 0] = range_image[:, :, 0] / max(max_range, 1e-8)

    # Add batch dimension
    input_tensor = tf.constant(range_image[np.newaxis, ...], dtype=tf.float32)

    return input_tensor, proj_idx, proj_mask


def unproject_predictions(proj_predictions, proj_idx, num_points):
    """
    Unproject 2D predictions back to 3D point cloud.

    Args:
        proj_predictions: (H, W) per-pixel class predictions
        proj_idx: (H, W) index mapping from pixel to point
        num_points: total number of points in the scan

    Returns:
        point_predictions: (N,) per-point class predictions
    """
    point_predictions = np.zeros(num_points, dtype=np.int32)

    valid = proj_idx >= 0
    valid_indices = proj_idx[valid]
    valid_preds = proj_predictions[valid]

    point_predictions[valid_indices] = valid_preds

    return point_predictions


def evaluate(args):
    """
    Main evaluation loop.

    Args:
        args: parsed command-line arguments
    """
    print("=" * 70)
    print("RangeNet++ Evaluation on SemanticKITTI")
    print("=" * 70)
    print(f"  Checkpoint:    {args.checkpoint_path}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Sequence:      {args.sequence}")
    print(f"  KNN enabled:   {args.use_knn}")
    if args.use_knn:
        print(f"  KNN K:         {args.knn_k}")
    print(f"  Batch size:    {args.batch_size}")
    print("=" * 70)

    # ---- Load model and checkpoint ----
    print("\n[1/3] Loading model and checkpoint...")
    model = RangeNetPP(num_classes=NUM_CLASSES)

    # Build model with a dummy input
    dummy_input = tf.zeros((1, PROJ_H, PROJ_W, 5), dtype=tf.float32)
    _ = model(dummy_input, training=False)

    # Load weights from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(args.checkpoint_path)
    # Allow partial restores if needed (e.g., optimizer state not present)
    status.expect_partial()
    print(f"  Checkpoint loaded: {args.checkpoint_path}")
    print(f"  Model parameters: {model.count_params():,}")

    # ---- Load data paths ----
    print("\n[2/3] Loading validation data...")
    scan_paths, label_paths = get_scan_paths(args.data_dir, args.sequence)
    num_scans = len(scan_paths)
    print(f"  Found {num_scans} scans in sequence {args.sequence}")

    # ---- Run inference ----
    print("\n[3/3] Running inference...")
    iou_metric = IoUMetric(NUM_CLASSES)

    total_time = 0.0
    total_points = 0

    for i, (scan_path, label_path) in enumerate(zip(scan_paths, label_paths)):
        # Load point cloud and labels
        points, remission = load_point_cloud(scan_path)
        gt_labels = load_labels(label_path)
        num_points = len(points)
        total_points += num_points

        # Prepare input
        input_tensor, proj_idx, proj_mask = prepare_input(points, remission)

        # Run inference
        t_start = time.time()
        logits = model(input_tensor, training=False)
        proj_predictions = tf.argmax(logits[0], axis=-1).numpy().astype(np.int32)
        t_inference = time.time() - t_start

        # Unproject predictions from 2D range image to 3D point cloud
        point_predictions = unproject_predictions(proj_predictions, proj_idx, num_points)

        # Optional KNN post-processing
        if args.use_knn:
            t_knn_start = time.time()
            point_predictions = knn_post_processing(points, point_predictions, k=args.knn_k)
            t_knn = time.time() - t_knn_start
            total_time += t_inference + t_knn
        else:
            total_time += t_inference

        # Update IoU metric
        iou_metric.update_fast(point_predictions, gt_labels)

        # Progress reporting
        if (i + 1) % 100 == 0 or (i + 1) == num_scans:
            elapsed = total_time
            fps = (i + 1) / max(elapsed, 1e-8)
            print(
                f"  Processed {i + 1}/{num_scans} scans | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Speed: {fps:.1f} scans/s | "
                f"Points: {total_points:,}"
            )

    # ---- Compute and print results ----
    per_class_iou, miou = iou_metric.compute_iou()

    print("\n")
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Class ID':<10} {'Class Name':<18} {'IoU (%)':<10}")
    print("-" * 40)
    for c in range(NUM_CLASSES):
        iou_pct = per_class_iou[c] * 100.0
        marker = " *" if c == 0 else ""
        print(f"{c:<10} {CLASS_NAMES[c]:<18} {iou_pct:>7.2f}{marker}")
    print("-" * 40)
    print(f"{'':10} {'mIoU (1-19)':<18} {miou * 100.0:>7.2f}")
    print("=" * 70)
    print(f"\n  * Class 0 (unlabeled) is excluded from mIoU computation.")
    print(f"\n  Total scans:       {num_scans}")
    print(f"  Total points:      {total_points:,}")
    print(f"  Total time:        {total_time:.2f} s")
    print(f"  Avg time/scan:     {total_time / max(num_scans, 1) * 1000:.1f} ms")
    print(f"  KNN enabled:       {args.use_knn}")
    if args.use_knn:
        print(f"  KNN K:             {args.knn_k}")
    print(f"\n  Final mIoU: {miou * 100.0:.2f}%")
    print("=" * 70)

    return miou


# ==============================================================================
# Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RangeNet++ on SemanticKITTI validation set."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained TensorFlow checkpoint (e.g., checkpoints/ckpt-50).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the SemanticKITTI dataset (containing sequences/).",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="08",
        help="Sequence number to evaluate on (default: 08, the validation split).",
    )
    parser.add_argument(
        "--use_knn",
        action="store_true",
        default=False,
        help="Enable KNN post-processing for prediction smoothing.",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="Number of nearest neighbors for KNN post-processing (default: 5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0). Set to -1 for CPU.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure GPU
    if args.gpu >= 0:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[args.gpu], "GPU")
                tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
                print(f"Using GPU: {gpus[args.gpu].name}")
            except (IndexError, RuntimeError) as e:
                print(f"GPU configuration error: {e}. Falling back to CPU.")
                tf.config.set_visible_devices([], "GPU")
        else:
            print("No GPUs found. Running on CPU.")
    else:
        tf.config.set_visible_devices([], "GPU")
        print("Running on CPU (--gpu=-1).")

    miou = evaluate(args)
    return miou


if __name__ == "__main__":
    main()
