#!/usr/bin/env python3
"""
RangeNet++ Training Script — TensorFlow 2
Complete training pipeline for SemanticKITTI point cloud semantic segmentation
using spherical projection to range images.
"""

import os
import argparse
import math
import numpy as np
import tensorflow as tf
from pathlib import Path

# ==============================================================================
# SemanticKITTI Configuration
# ==============================================================================

# Raw label -> training ID remapping (20 valid classes, 0-19)
# Labels not in this map are mapped to 255 (ignored)
LABEL_REMAP = {
    0: 0,        # "unlabeled" -> mapped to 0 but ignored in loss via ignore_index
    1: 0,        # "outlier" -> ignored
    10: 0,       # "car"
    11: 1,       # "bicycle"
    13: 5,       # "bus"
    15: 2,       # "motorcycle"
    16: 5,       # "on-rails" -> bus
    18: 3,       # "truck"
    20: 4,       # "other-vehicle"
    30: 6,       # "person"
    31: 7,       # "bicyclist"
    32: 8,       # "motorcyclist"
    40: 9,       # "road"
    44: 10,      # "parking"
    48: 11,      # "sidewalk"
    49: 12,      # "other-ground"
    50: 13,      # "building"
    51: 14,      # "fence"
    52: 0,       # "other-structure" -> unlabeled
    60: 9,       # "lane-marking" -> road
    70: 15,      # "vegetation"
    71: 16,      # "trunk"
    72: 17,      # "terrain"
    80: 18,      # "pole"
    81: 19,      # "traffic-sign"
    99: 0,       # "other-object" -> unlabeled
    252: 0,      # "moving-car" -> car
    253: 7,      # "moving-bicyclist" -> bicyclist
    254: 6,      # "moving-person" -> person
    255: 8,      # "moving-motorcyclist" -> motorcyclist
    256: 5,      # "moving-bus" -> bus
    257: 5,      # "moving-on-rails" -> bus
    258: 4,      # "moving-truck" -> other-vehicle
    259: 4,      # "moving-other-vehicle" -> other-vehicle
}

# Build a full lookup table for fast remapping (max label value in SemanticKITTI)
MAX_LABEL_VALUE = 260
REMAP_LUT = np.full(MAX_LABEL_VALUE, 0, dtype=np.int32)
for raw_label, train_id in LABEL_REMAP.items():
    if raw_label < MAX_LABEL_VALUE:
        REMAP_LUT[raw_label] = train_id

NUM_CLASSES = 20
IGNORE_INDEX = 0  # class 0 is "unlabeled/ignored" in training loss

# Class names for the 20 training classes
CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "bus", "person", "bicyclist", "motorcyclist", "road",
    "parking", "sidewalk", "other-ground", "building", "fence",
    "vegetation", "trunk", "terrain", "pole", "traffic-sign"
]

# Class weights for imbalanced SemanticKITTI data (inverse log frequency)
# Precomputed from training set statistics
CLASS_WEIGHTS = np.array([
    2.30,   # car
    34.12,  # bicycle
    26.43,  # motorcycle
    6.08,   # truck
    9.54,   # other-vehicle
    7.89,   # bus
    29.75,  # person
    24.56,  # bicyclist
    42.13,  # motorcyclist
    1.00,   # road
    4.67,   # parking
    2.12,   # sidewalk
    8.34,   # other-ground
    1.45,   # building
    5.78,   # fence
    1.23,   # vegetation
    7.92,   # trunk
    1.87,   # terrain
    11.34,  # pole
    18.67,  # traffic-sign
], dtype=np.float32)

# Spherical projection parameters
PROJ_H = 64
PROJ_W = 2048
FOV_UP = 2.0  # degrees
FOV_DOWN = -24.8  # degrees
FOV_HORIZ = 360.0  # degrees

# SemanticKITTI sequence splits
TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQUENCES = ["08"]


# ==============================================================================
# Spherical Projection
# ==============================================================================

def spherical_projection_np(points):
    """
    Project 3D point cloud to 2D range image using spherical projection.

    Args:
        points: numpy array (N, 4) with [x, y, z, intensity]

    Returns:
        proj_image: (H, W, 5) range image [range, x, y, z, intensity]
        proj_labels: (H, W) label image (initialized to IGNORE_INDEX)
        proj_mask: (H, W) boolean mask of valid pixels
        proj_idx: (H, W) index into original point array
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    # Compute range
    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Compute yaw and pitch
    yaw = np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / (depth + 1e-8), -1.0, 1.0))

    # Convert FOV to radians
    fov_up_rad = FOV_UP * np.pi / 180.0
    fov_down_rad = FOV_DOWN * np.pi / 180.0
    fov_total = fov_up_rad - fov_down_rad

    # Map to pixel coordinates
    # u: horizontal (0 to W-1), v: vertical (0 to H-1)
    u = 0.5 * (1.0 - yaw / np.pi) * PROJ_W
    v = (1.0 - (pitch - fov_down_rad) / fov_total) * PROJ_H

    # Clamp to image bounds
    u = np.clip(np.floor(u).astype(np.int32), 0, PROJ_W - 1)
    v = np.clip(np.floor(v).astype(np.int32), 0, PROJ_H - 1)

    # Order by depth (far to near so closer points overwrite)
    order = np.argsort(-depth)
    x = x[order]
    y = y[order]
    z = z[order]
    depth = depth[order]
    intensity = intensity[order]
    u = u[order]
    v = v[order]

    # Create projection arrays
    proj_image = np.zeros((PROJ_H, PROJ_W, 5), dtype=np.float32)
    proj_mask = np.zeros((PROJ_H, PROJ_W), dtype=np.bool_)
    proj_idx = np.full((PROJ_H, PROJ_W), -1, dtype=np.int32)

    # Fill projection
    proj_image[v, u, 0] = depth
    proj_image[v, u, 1] = x
    proj_image[v, u, 2] = y
    proj_image[v, u, 3] = z
    proj_image[v, u, 4] = intensity
    proj_mask[v, u] = True
    proj_idx[v, u] = order

    return proj_image, proj_mask, proj_idx


def project_labels_np(labels, proj_idx):
    """
    Project point labels to range image using projection indices.

    Args:
        labels: (N,) int32 training labels
        proj_idx: (H, W) indices into original point array

    Returns:
        proj_labels: (H, W) projected label image
    """
    proj_labels = np.zeros((PROJ_H, PROJ_W), dtype=np.int32)
    valid_mask = proj_idx >= 0
    proj_labels[valid_mask] = labels[proj_idx[valid_mask]]
    return proj_labels


# ==============================================================================
# Data Pipeline
# ==============================================================================

def get_scan_paths(data_dir, sequences):
    """Get sorted lists of .bin and .label file paths for given sequences."""
    scan_paths = []
    label_paths = []

    for seq in sequences:
        scan_dir = Path(data_dir) / "sequences" / seq / "velodyne"
        label_dir = Path(data_dir) / "sequences" / seq / "labels"

        if not scan_dir.exists():
            print(f"Warning: {scan_dir} does not exist, skipping.")
            continue

        scans = sorted(scan_dir.glob("*.bin"))
        for scan_path in scans:
            frame_id = scan_path.stem
            label_path = label_dir / f"{frame_id}.label"
            if label_path.exists():
                scan_paths.append(str(scan_path))
                label_paths.append(str(label_path))

    return scan_paths, label_paths


def load_and_project(scan_path_bytes, label_path_bytes):
    """
    TF py_function wrapper: load .bin and .label, remap labels, project to range image.
    """
    scan_path = scan_path_bytes.numpy().decode("utf-8")
    label_path = label_path_bytes.numpy().decode("utf-8")

    # Load point cloud (N x 4: x, y, z, intensity)
    points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)

    # Load labels (N x 1: lower 16 bits = semantic label, upper 16 bits = instance)
    raw_labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
    semantic_labels = (raw_labels & 0xFFFF).astype(np.int32)

    # Remap to training IDs
    # Clip to valid LUT range, anything beyond is mapped to 0
    semantic_labels = np.clip(semantic_labels, 0, MAX_LABEL_VALUE - 1)
    train_labels = REMAP_LUT[semantic_labels]

    # Spherical projection
    proj_image, proj_mask, proj_idx = spherical_projection_np(points)

    # Project labels
    proj_labels = project_labels_np(train_labels, proj_idx)

    # Normalize range image channels
    # Channel 0: range (normalize by max range)
    max_range = proj_image[:, :, 0].max()
    if max_range > 0:
        proj_image[:, :, 0] /= max_range

    # Channel 4: intensity (already 0-1 typically, clip just in case)
    proj_image[:, :, 4] = np.clip(proj_image[:, :, 4], 0.0, 1.0)

    # Channels 1-3: xyz (normalize by range for each point)
    range_vals = proj_image[:, :, 0:1] * max_range
    range_vals = np.where(range_vals > 0, range_vals, 1.0)
    proj_image[:, :, 1:4] /= range_vals

    return proj_image.astype(np.float32), proj_labels.astype(np.int32)


def create_dataset(data_dir, sequences, batch_size, shuffle=True, num_parallel=8):
    """
    Create a tf.data.Dataset pipeline for SemanticKITTI range images.
    """
    scan_paths, label_paths = get_scan_paths(data_dir, sequences)

    if len(scan_paths) == 0:
        raise ValueError(f"No scan/label pairs found for sequences {sequences} in {data_dir}")

    print(f"  Found {len(scan_paths)} scan/label pairs")

    dataset = tf.data.Dataset.from_tensor_slices((scan_paths, label_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(scan_paths), 10000))

    def tf_load_and_project(scan_path, label_path):
        image, labels = tf.py_function(
            func=load_and_project,
            inp=[scan_path, label_path],
            Tout=(tf.float32, tf.int32)
        )
        image.set_shape([PROJ_H, PROJ_W, 5])
        labels.set_shape([PROJ_H, PROJ_W])
        return image, labels

    dataset = dataset.map(tf_load_and_project, num_parallel_calls=num_parallel)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(scan_paths)


# ==============================================================================
# Loss Functions
# ==============================================================================

def weighted_cross_entropy_loss(logits, labels, class_weights, num_classes=NUM_CLASSES):
    """
    Compute weighted cross-entropy loss, ignoring class 0 (unlabeled).

    Args:
        logits: (B, H, W, num_classes) raw predictions
        labels: (B, H, W) integer labels [0, num_classes)
        class_weights: (num_classes,) per-class weights

    Returns:
        Scalar loss value
    """
    # Flatten spatial dimensions
    batch_size = tf.shape(logits)[0]
    logits_flat = tf.reshape(logits, [-1, num_classes])
    labels_flat = tf.reshape(labels, [-1])

    # Create ignore mask (class 0 is unlabeled/ignored)
    valid_mask = tf.cast(labels_flat > 0, tf.float32)

    # Compute per-pixel cross entropy
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels_flat, logits=logits_flat
    )

    # Apply class weights
    weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    pixel_weights = tf.gather(weights_tensor, labels_flat)

    # Apply both class weights and ignore mask
    weighted_loss = ce_loss * pixel_weights * valid_mask

    # Average over valid pixels
    num_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    loss = tf.reduce_sum(weighted_loss) / num_valid

    return loss


def lovasz_grad(gt_sorted):
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors.

    Args:
        gt_sorted: sorted ground truth (descending by errors)

    Returns:
        Gradient tensor
    """
    p = tf.cast(tf.size(gt_sorted), tf.float32)
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1.0 - gt_sorted)
    jaccard = 1.0 - intersection / union

    # Compute gradient as difference of consecutive jaccard values
    jaccard_shifted = tf.concat([[0.0], jaccard[:-1]], axis=0)
    grad = jaccard - jaccard_shifted
    return grad


def lovasz_softmax_flat(probas, labels, num_classes=NUM_CLASSES):
    """
    Multi-class Lovasz-Softmax loss (flat version).

    Args:
        probas: (P, C) class probabilities for each pixel
        labels: (P,) integer ground truth labels

    Returns:
        Scalar Lovasz-Softmax loss
    """
    losses = []

    for c in range(1, num_classes):  # Skip class 0 (ignored)
        # Foreground for class c
        fg = tf.cast(tf.equal(labels, c), tf.float32)

        # If this class is not present and not predicted, skip
        if tf.reduce_sum(fg) == 0:
            continue

        # Errors: 1 - probability of correct class for foreground,
        #         probability of this class for background
        fg_prob = probas[:, c]
        errors = tf.abs(fg - fg_prob)

        # Sort by descending errors
        perm = tf.argsort(errors, direction="DESCENDING")
        errors_sorted = tf.gather(errors, perm)
        fg_sorted = tf.gather(fg, perm)

        # Compute Lovasz gradient
        grad = lovasz_grad(fg_sorted)

        # Loss for this class
        loss_c = tf.reduce_sum(errors_sorted * tf.stop_gradient(grad))
        losses.append(loss_c)

    if len(losses) == 0:
        return tf.constant(0.0)

    return tf.reduce_mean(tf.stack(losses))


def lovasz_softmax_loss(logits, labels, num_classes=NUM_CLASSES):
    """
    Compute Lovasz-Softmax loss over a batch.

    Args:
        logits: (B, H, W, num_classes) raw predictions
        labels: (B, H, W) integer labels

    Returns:
        Scalar loss value
    """
    batch_size = tf.shape(logits)[0]
    probas = tf.nn.softmax(logits, axis=-1)

    # Flatten spatial dimensions per sample and compute loss per sample
    losses = []
    for b in tf.range(batch_size):
        prob_b = tf.reshape(probas[b], [-1, num_classes])
        label_b = tf.reshape(labels[b], [-1])

        # Filter out ignored pixels (class 0)
        valid_mask = label_b > 0
        prob_valid = tf.boolean_mask(prob_b, valid_mask)
        label_valid = tf.boolean_mask(label_b, valid_mask)

        if tf.size(label_valid) > 0:
            loss_b = lovasz_softmax_flat(prob_valid, label_valid, num_classes)
            losses.append(loss_b)

    if len(losses) == 0:
        return tf.constant(0.0)

    return tf.reduce_mean(tf.stack(losses))


@tf.function
def lovasz_softmax_loss_tf(logits, labels, num_classes=NUM_CLASSES):
    """
    TF-graph-compatible Lovasz-Softmax loss.
    Processes each class and computes the Lovasz extension.

    Args:
        logits: (B, H, W, num_classes) raw predictions
        labels: (B, H, W) integer labels

    Returns:
        Scalar loss value
    """
    probas = tf.nn.softmax(logits, axis=-1)

    # Flatten everything
    probas_flat = tf.reshape(probas, [-1, num_classes])
    labels_flat = tf.reshape(labels, [-1])

    # Filter valid pixels
    valid_mask = labels_flat > 0
    probas_valid = tf.boolean_mask(probas_flat, valid_mask)
    labels_valid = tf.boolean_mask(labels_flat, valid_mask)

    n_valid = tf.shape(labels_valid)[0]

    # Handle empty case
    if n_valid == 0:
        return tf.constant(0.0, dtype=tf.float32)

    class_losses = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    idx = 0

    for c in tf.range(1, num_classes):
        fg = tf.cast(tf.equal(labels_valid, c), tf.float32)
        n_fg = tf.reduce_sum(fg)

        if n_fg > 0:
            fg_prob = probas_valid[:, c]
            errors = tf.abs(fg - fg_prob)

            # Sort descending
            perm = tf.argsort(errors, direction="DESCENDING")
            errors_sorted = tf.gather(errors, perm)
            fg_sorted = tf.gather(fg, perm)

            # Lovasz gradient
            p = tf.cast(tf.size(fg_sorted), tf.float32)
            gts = tf.reduce_sum(fg_sorted)
            intersection = gts - tf.cumsum(fg_sorted)
            union = gts + tf.cumsum(1.0 - fg_sorted)
            jaccard = 1.0 - intersection / union
            jaccard_shifted = tf.concat([[0.0], jaccard[:-1]], axis=0)
            grad = jaccard - jaccard_shifted

            loss_c = tf.reduce_sum(errors_sorted * tf.stop_gradient(grad))
            class_losses = class_losses.write(idx, loss_c)
            idx += 1

    if idx == 0:
        return tf.constant(0.0, dtype=tf.float32)

    return tf.reduce_mean(class_losses.stack())


def combined_loss(logits, labels, class_weights, lovasz_weight=1.0, ce_weight=1.0):
    """
    Combined weighted cross-entropy + Lovasz-Softmax loss.

    Args:
        logits: (B, H, W, num_classes) raw model predictions
        labels: (B, H, W) ground truth labels
        class_weights: per-class weights for CE loss
        lovasz_weight: weight for Lovasz loss component
        ce_weight: weight for CE loss component

    Returns:
        total_loss, ce_loss, lovasz_loss
    """
    ce = weighted_cross_entropy_loss(logits, labels, class_weights)
    lovasz = lovasz_softmax_loss_tf(logits, labels)
    total = ce_weight * ce + lovasz_weight * lovasz
    return total, ce, lovasz


# ==============================================================================
# Learning Rate Schedules
# ==============================================================================

class PolynomialDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Polynomial decay learning rate with linear warmup."""

    def __init__(self, initial_lr, end_lr, decay_steps, warmup_steps, power=0.9):
        super().__init__()
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.power = power

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.initial_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Polynomial decay
        decay_step = tf.minimum(step - warmup_steps, decay_steps)
        decay_step = tf.maximum(decay_step, 0.0)
        decay_factor = (1.0 - decay_step / decay_steps) ** self.power
        decay_lr = (self.initial_lr - self.end_lr) * decay_factor + self.end_lr

        # Select based on current step
        lr = tf.where(step < warmup_steps, warmup_lr, decay_lr)
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "end_lr": self.end_lr,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
        }


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay learning rate with linear warmup."""

    def __init__(self, initial_lr, end_lr, decay_steps, warmup_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.initial_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay
        decay_step = tf.minimum(step - warmup_steps, decay_steps)
        decay_step = tf.maximum(decay_step, 0.0)
        cosine_factor = 0.5 * (1.0 + tf.cos(math.pi * decay_step / decay_steps))
        decay_lr = (self.initial_lr - self.end_lr) * cosine_factor + self.end_lr

        lr = tf.where(step < warmup_steps, warmup_lr, decay_lr)
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "end_lr": self.end_lr,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
        }


# ==============================================================================
# RangeNet++ Model (Encoder-Decoder with skip connections)
# ==============================================================================

def conv_bn_relu(x, filters, kernel_size=3, strides=1, dilation_rate=1, name=None):
    """Convolution + BatchNorm + LeakyReLU block."""
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding="same",
        dilation_rate=dilation_rate, use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_conv" if name else None
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = tf.keras.layers.LeakyReLU(0.1, name=f"{name}_relu" if name else None)(x)
    return x


def residual_block(x, filters, strides=1, dilation_rate=1, name=None):
    """Residual block with two conv-bn-relu layers."""
    shortcut = x

    out = conv_bn_relu(x, filters, 3, strides=strides, dilation_rate=dilation_rate,
                       name=f"{name}_conv1" if name else None)
    out = conv_bn_relu(out, filters, 3, strides=1, dilation_rate=dilation_rate,
                       name=f"{name}_conv2" if name else None)

    # Adjust shortcut if dimensions change
    if strides > 1 or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=strides, padding="same", use_bias=False,
            kernel_initializer="he_normal", name=f"{name}_shortcut_conv" if name else None
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name=f"{name}_shortcut_bn" if name else None
        )(shortcut)

    out = tf.keras.layers.Add(name=f"{name}_add" if name else None)([out, shortcut])
    out = tf.keras.layers.LeakyReLU(0.1, name=f"{name}_out_relu" if name else None)(out)
    return out


def build_rangenet_pp(input_shape=(PROJ_H, PROJ_W, 5), num_classes=NUM_CLASSES,
                      dropout_rate=0.2):
    """
    Build RangeNet++ encoder-decoder architecture.

    Encoder: ResNet-like blocks with progressive downsampling.
    Decoder: Transpose convolutions with skip connections from encoder.

    Args:
        input_shape: (H, W, C) input range image shape
        num_classes: number of output classes
        dropout_rate: spatial dropout rate

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name="range_image")

    # Encoder
    # Stage 1: 64xW -> 64xW
    e1 = conv_bn_relu(inputs, 64, 3, strides=1, name="enc1_entry")
    e1 = residual_block(e1, 64, name="enc1_res1")
    e1 = residual_block(e1, 64, name="enc1_res2")

    # Stage 2: 64xW -> 32x(W/2)
    e2 = residual_block(e1, 128, strides=2, name="enc2_res1")
    e2 = residual_block(e2, 128, name="enc2_res2")

    # Stage 3: 32x(W/2) -> 16x(W/4)
    e3 = residual_block(e2, 256, strides=2, name="enc3_res1")
    e3 = residual_block(e3, 256, name="enc3_res2")

    # Stage 4: 16x(W/4) -> 8x(W/8)
    e4 = residual_block(e3, 512, strides=2, name="enc4_res1")
    e4 = residual_block(e4, 512, name="enc4_res2")

    # Stage 5: 8x(W/8) -> 4x(W/16) (bottleneck)
    e5 = residual_block(e4, 512, strides=2, name="enc5_res1")
    e5 = residual_block(e5, 512, name="enc5_res2")
    e5 = tf.keras.layers.SpatialDropout2D(dropout_rate, name="enc5_dropout")(e5)

    # Decoder
    # Stage 4: 4x(W/16) -> 8x(W/8)
    d4 = tf.keras.layers.Conv2DTranspose(
        512, 3, strides=2, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="dec4_upconv"
    )(e5)
    d4 = tf.keras.layers.BatchNormalization(name="dec4_bn")(d4)
    d4 = tf.keras.layers.LeakyReLU(0.1, name="dec4_relu")(d4)
    d4 = tf.keras.layers.Concatenate(name="dec4_concat")([d4, e4])
    d4 = residual_block(d4, 512, name="dec4_res")
    d4 = tf.keras.layers.SpatialDropout2D(dropout_rate, name="dec4_dropout")(d4)

    # Stage 3: 8x(W/8) -> 16x(W/4)
    d3 = tf.keras.layers.Conv2DTranspose(
        256, 3, strides=2, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="dec3_upconv"
    )(d4)
    d3 = tf.keras.layers.BatchNormalization(name="dec3_bn")(d3)
    d3 = tf.keras.layers.LeakyReLU(0.1, name="dec3_relu")(d3)
    d3 = tf.keras.layers.Concatenate(name="dec3_concat")([d3, e3])
    d3 = residual_block(d3, 256, name="dec3_res")
    d3 = tf.keras.layers.SpatialDropout2D(dropout_rate, name="dec3_dropout")(d3)

    # Stage 2: 16x(W/4) -> 32x(W/2)
    d2 = tf.keras.layers.Conv2DTranspose(
        128, 3, strides=2, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="dec2_upconv"
    )(d3)
    d2 = tf.keras.layers.BatchNormalization(name="dec2_bn")(d2)
    d2 = tf.keras.layers.LeakyReLU(0.1, name="dec2_relu")(d2)
    d2 = tf.keras.layers.Concatenate(name="dec2_concat")([d2, e2])
    d2 = residual_block(d2, 128, name="dec2_res")

    # Stage 1: 32x(W/2) -> 64xW
    d1 = tf.keras.layers.Conv2DTranspose(
        64, 3, strides=2, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="dec1_upconv"
    )(d2)
    d1 = tf.keras.layers.BatchNormalization(name="dec1_bn")(d1)
    d1 = tf.keras.layers.LeakyReLU(0.1, name="dec1_relu")(d1)
    d1 = tf.keras.layers.Concatenate(name="dec1_concat")([d1, e1])
    d1 = residual_block(d1, 64, name="dec1_res")

    # Classification head
    logits = tf.keras.layers.Conv2D(
        num_classes, 1, padding="same",
        kernel_initializer="he_normal", name="logits"
    )(d1)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="RangeNetPP")
    return model


# ==============================================================================
# Metrics
# ==============================================================================

class MeanIoU:
    """Compute mean Intersection over Union across classes, ignoring class 0."""

    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions, labels):
        """
        Update confusion matrix with batch predictions.

        Args:
            predictions: (B, H, W) predicted class IDs
            labels: (B, H, W) ground truth class IDs
        """
        preds_np = predictions.numpy() if isinstance(predictions, tf.Tensor) else predictions
        labels_np = labels.numpy() if isinstance(labels, tf.Tensor) else labels

        # Flatten
        preds_flat = preds_np.reshape(-1)
        labels_flat = labels_np.reshape(-1)

        # Only consider valid pixels (label > 0)
        valid = labels_flat > 0
        preds_flat = preds_flat[valid]
        labels_flat = labels_flat[valid]

        # Update confusion matrix
        for i in range(len(preds_flat)):
            if 0 <= preds_flat[i] < self.num_classes and 0 <= labels_flat[i] < self.num_classes:
                self.confusion_matrix[labels_flat[i], preds_flat[i]] += 1

    def update_batch(self, predictions, labels):
        """Vectorized confusion matrix update."""
        preds_np = predictions.numpy() if isinstance(predictions, tf.Tensor) else predictions
        labels_np = labels.numpy() if isinstance(labels, tf.Tensor) else labels

        preds_flat = preds_np.reshape(-1).astype(np.int64)
        labels_flat = labels_np.reshape(-1).astype(np.int64)

        valid = labels_flat > 0
        preds_flat = preds_flat[valid]
        labels_flat = labels_flat[valid]

        # Vectorized bincount-based confusion matrix
        mask = (preds_flat >= 0) & (preds_flat < self.num_classes) & \
               (labels_flat >= 0) & (labels_flat < self.num_classes)
        preds_flat = preds_flat[mask]
        labels_flat = labels_flat[mask]

        indices = labels_flat * self.num_classes + preds_flat
        cm_flat = np.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += cm_flat.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Compute per-class IoU and mean IoU (excluding class 0).

        Returns:
            miou: mean IoU across valid classes (1-19)
            per_class_iou: dict mapping class name to IoU
        """
        per_class_iou = {}
        ious = []

        for c in range(1, self.num_classes):  # Skip class 0
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp

            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = float("nan")

            per_class_iou[CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"] = iou
            if not np.isnan(iou):
                ious.append(iou)

        miou = np.mean(ious) if len(ious) > 0 else 0.0
        return miou, per_class_iou


# ==============================================================================
# Training Loop
# ==============================================================================

def train(args):
    """Main training function with multi-GPU support."""

    print("=" * 80)
    print("RangeNet++ TensorFlow 2 Training")
    print("=" * 80)

    # Setup distribution strategy
    if args.gpus > 1:
        devices = [f"/gpu:{i}" for i in range(args.gpus)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)
        print(f"Using MirroredStrategy with {args.gpus} GPUs")
    elif args.gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        print("Using single GPU")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        print("Using CPU")

    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset, num_train = create_dataset(
        args.data_dir, TRAIN_SEQUENCES, global_batch_size,
        shuffle=True, num_parallel=args.num_workers
    )
    print(f"\nCreating validation dataset...")
    val_dataset, num_val = create_dataset(
        args.data_dir, VAL_SEQUENCES, global_batch_size,
        shuffle=False, num_parallel=args.num_workers
    )

    steps_per_epoch = num_train // global_batch_size
    val_steps = num_val // global_batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"\nTraining samples: {num_train}")
    print(f"Validation samples: {num_val}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Distribute datasets
    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # Build model and optimizer within strategy scope
    with strategy.scope():
        model = build_rangenet_pp(
            input_shape=(PROJ_H, PROJ_W, 5),
            num_classes=NUM_CLASSES,
            dropout_rate=args.dropout
        )
        model.summary()

        # Learning rate schedule
        if args.lr_schedule == "polynomial":
            lr_schedule = PolynomialDecayWithWarmup(
                initial_lr=args.learning_rate,
                end_lr=args.min_lr,
                decay_steps=total_steps - warmup_steps,
                warmup_steps=warmup_steps,
                power=args.poly_power
            )
        elif args.lr_schedule == "cosine":
            lr_schedule = CosineDecayWithWarmup(
                initial_lr=args.learning_rate,
                end_lr=args.min_lr,
                decay_steps=total_steps - warmup_steps,
                warmup_steps=warmup_steps
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {args.lr_schedule}")

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        # Checkpoint
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_dir, max_to_keep=args.max_checkpoints
        )

        # Restore if exists
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint)
            print(f"\nRestored from checkpoint: {ckpt_manager.latest_checkpoint}")

    # TensorBoard
    tb_log_dir = os.path.join(args.log_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(tb_log_dir, "train")
    )
    val_summary_writer = tf.summary.create_file_writer(
        os.path.join(tb_log_dir, "val")
    )

    # Class weights tensor
    class_weights_tf = tf.constant(CLASS_WEIGHTS, dtype=tf.float32)

    # Metrics
    train_miou_metric = MeanIoU(NUM_CLASSES)
    val_miou_metric = MeanIoU(NUM_CLASSES)

    # Define training step
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            total_loss, ce_loss, lov_loss = combined_loss(
                logits, labels, class_weights_tf,
                lovasz_weight=args.lovasz_weight,
                ce_weight=args.ce_weight
            )
            # Scale loss for distributed training
            scaled_loss = total_loss / strategy.num_replicas_in_sync

        gradients = tape.gradient(scaled_loss, model.trainable_variables)

        # Gradient clipping
        if args.grad_clip > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, args.grad_clip)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return total_loss, ce_loss, lov_loss, predictions

    @tf.function
    def val_step(images, labels):
        logits = model(images, training=False)
        total_loss, ce_loss, lov_loss = combined_loss(
            logits, labels, class_weights_tf,
            lovasz_weight=args.lovasz_weight,
            ce_weight=args.ce_weight
        )
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return total_loss, ce_loss, lov_loss, predictions

    # Distributed step wrappers
    @tf.function
    def distributed_train_step(images, labels):
        per_replica_results = strategy.run(train_step, args=(images, labels))
        total_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None
        )
        ce_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None
        )
        lov_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_results[2], axis=None
        )
        return total_loss, ce_loss, lov_loss, per_replica_results[3]

    @tf.function
    def distributed_val_step(images, labels):
        per_replica_results = strategy.run(val_step, args=(images, labels))
        total_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None
        )
        ce_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None
        )
        lov_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_results[2], axis=None
        )
        return total_loss, ce_loss, lov_loss, per_replica_results[3]

    # Training loop
    global_step = 0
    best_miou = 0.0

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Training
        train_miou_metric.reset()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_lov_loss = 0.0
        num_batches = 0

        for step, (images, labels) in enumerate(dist_train_dataset):
            total_loss, ce_loss, lov_loss, predictions = distributed_train_step(
                images, labels
            )

            epoch_loss += total_loss.numpy()
            epoch_ce_loss += ce_loss.numpy()
            epoch_lov_loss += lov_loss.numpy()
            num_batches += 1
            global_step += 1

            # Update mIoU (subsample for speed)
            if step % args.miou_every == 0:
                if isinstance(predictions, tf.distribute.DistributedValues):
                    for pred, lab in zip(
                        strategy.experimental_local_results(predictions),
                        strategy.experimental_local_results(labels)
                    ):
                        train_miou_metric.update_batch(pred, lab)
                else:
                    train_miou_metric.update_batch(predictions, labels)

            # Log to TensorBoard
            if step % args.log_every == 0:
                current_lr = optimizer.learning_rate(optimizer.iterations).numpy()
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss/total", total_loss, step=global_step)
                    tf.summary.scalar("loss/cross_entropy", ce_loss, step=global_step)
                    tf.summary.scalar("loss/lovasz", lov_loss, step=global_step)
                    tf.summary.scalar("learning_rate", current_lr, step=global_step)

                print(
                    f"  Step {step}/{steps_per_epoch} | "
                    f"Loss: {total_loss.numpy():.4f} "
                    f"(CE: {ce_loss.numpy():.4f}, Lov: {lov_loss.numpy():.4f}) | "
                    f"LR: {current_lr:.6f}"
                )

        # Epoch training metrics
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_ce = epoch_ce_loss / max(num_batches, 1)
        avg_train_lov = epoch_lov_loss / max(num_batches, 1)
        train_miou, train_per_class = train_miou_metric.compute()

        with train_summary_writer.as_default():
            tf.summary.scalar("epoch/loss", avg_train_loss, step=epoch)
            tf.summary.scalar("epoch/miou", train_miou, step=epoch)

        print(f"\n  Train Loss: {avg_train_loss:.4f} | Train mIoU: {train_miou:.4f}")

        # Validation
        print("  Running validation...")
        val_miou_metric.reset()
        val_loss_total = 0.0
        val_ce_total = 0.0
        val_lov_total = 0.0
        val_batches = 0

        for images, labels in dist_val_dataset:
            total_loss, ce_loss, lov_loss, predictions = distributed_val_step(
                images, labels
            )

            val_loss_total += total_loss.numpy()
            val_ce_total += ce_loss.numpy()
            val_lov_total += lov_loss.numpy()
            val_batches += 1

            if isinstance(predictions, tf.distribute.DistributedValues):
                for pred, lab in zip(
                    strategy.experimental_local_results(predictions),
                    strategy.experimental_local_results(labels)
                ):
                    val_miou_metric.update_batch(pred, lab)
            else:
                val_miou_metric.update_batch(predictions, labels)

        avg_val_loss = val_loss_total / max(val_batches, 1)
        avg_val_ce = val_ce_total / max(val_batches, 1)
        avg_val_lov = val_lov_total / max(val_batches, 1)
        val_miou, val_per_class = val_miou_metric.compute()

        with val_summary_writer.as_default():
            tf.summary.scalar("epoch/loss", avg_val_loss, step=epoch)
            tf.summary.scalar("epoch/loss_ce", avg_val_ce, step=epoch)
            tf.summary.scalar("epoch/loss_lovasz", avg_val_lov, step=epoch)
            tf.summary.scalar("epoch/miou", val_miou, step=epoch)
            for cls_name, cls_iou in val_per_class.items():
                if not np.isnan(cls_iou):
                    tf.summary.scalar(f"iou/{cls_name}", cls_iou, step=epoch)

        print(f"  Val Loss: {avg_val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        print(f"  Per-class IoU:")
        for cls_name, cls_iou in val_per_class.items():
            if not np.isnan(cls_iou):
                print(f"    {cls_name:20s}: {cls_iou:.4f}")

        # Save checkpoint
        ckpt_path = ckpt_manager.save()
        print(f"\n  Checkpoint saved: {ckpt_path}")

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_model_dir = os.path.join(args.log_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_weights(os.path.join(best_model_dir, "model_weights"))
            print(f"  New best mIoU: {best_miou:.4f} — model saved!")

    # Final summary
    print("\n" + "=" * 80)
    print(f"Training complete!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"TensorBoard logs in: {tb_log_dir}")
    print("=" * 80)


# ==============================================================================
# Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RangeNet++ Training on SemanticKITTI (TensorFlow 2)"
    )

    # Data
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to SemanticKITTI dataset root (contains sequences/ folder)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/rangenet_pp",
        help="Directory for checkpoints and TensorBoard logs"
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel data loading workers")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate at end of schedule")
    parser.add_argument("--lr_schedule", type=str, default="polynomial",
                        choices=["polynomial", "cosine"],
                        help="Learning rate schedule type")
    parser.add_argument("--warmup_ratio", type=float, default=0.01,
                        help="Fraction of total steps for warmup")
    parser.add_argument("--poly_power", type=float, default=0.9,
                        help="Power for polynomial decay schedule")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="Gradient clipping norm (0 to disable)")

    # Loss
    parser.add_argument("--ce_weight", type=float, default=1.0,
                        help="Weight for cross-entropy loss component")
    parser.add_argument("--lovasz_weight", type=float, default=1.5,
                        help="Weight for Lovasz-Softmax loss component")

    # Model
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Spatial dropout rate in the model")

    # Hardware
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use (0 for CPU)")

    # Logging
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log training metrics every N steps")
    parser.add_argument("--miou_every", type=int, default=10,
                        help="Update mIoU metric every N steps (for speed)")
    parser.add_argument("--max_checkpoints", type=int, default=5,
                        help="Maximum number of checkpoints to keep")

    # Mixed precision
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training (float16)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Print configuration
    print("\nConfiguration:")
    print("-" * 50)
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print("-" * 50)

    # Enable mixed precision if requested
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("\nMixed precision (float16) enabled")

    # GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\nFound {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    else:
        print("\nNo GPUs found, running on CPU")

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
