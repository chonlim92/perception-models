"""
HDMapNet TensorFlow 2 Inference Script

Runs inference on a saved/checkpointed HDMapNet model, produces BEV semantic
segmentation, instance segmentation, and direction predictions, then visualizes
and saves the results.

BEV grid: 200x200 covering 60m x 30m
Semantic classes: lane dividers (0), road boundaries (1), pedestrian crossings (2)
Instance embedding: 16-dimensional
Direction: 2D unit vector per pixel
Camera images: 6 cameras, each 128x352

Usage:
    python inference.py \
        --checkpoint_dir ./checkpoints \
        --input_file ./sample.npz \
        --output_dir ./output \
        --view_transform lss \
        --show
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
from scipy import ndimage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEV_HEIGHT = 200
BEV_WIDTH = 200
NUM_CLASSES = 3
INSTANCE_EMB_DIM = 16
DIRECTION_DIM = 2
NUM_CAMERAS = 6
IMG_HEIGHT = 128
IMG_WIDTH = 352

CLASS_NAMES = ["lane_dividers", "road_boundaries", "ped_crossings"]
CLASS_COLORS = np.array([
    [1.0, 0.0, 0.0],   # red - lane dividers
    [0.0, 0.0, 1.0],   # blue - road boundaries
    [0.0, 1.0, 0.0],   # green - pedestrian crossings
], dtype=np.float32)

SEMANTIC_THRESHOLD = 0.5
EMBEDDING_DISTANCE_THRESHOLD = 1.5
MIN_INSTANCE_PIXELS = 10


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str) -> tf.keras.Model:
    """Load a saved HDMapNet model from checkpoint directory.

    Supports both SavedModel format and checkpoint-based loading.
    """
    saved_model_path = os.path.join(checkpoint_dir, "saved_model.pb")
    if os.path.exists(os.path.join(checkpoint_dir, "saved_model.pb")) or \
       os.path.exists(os.path.join(checkpoint_dir, "saved_model")):
        print(f"[INFO] Loading SavedModel from: {checkpoint_dir}")
        model = tf.saved_model.load(checkpoint_dir)
        return model

    # Try loading as a Keras model directory
    keras_model_path = checkpoint_dir
    if os.path.isdir(keras_model_path):
        # Check for keras_metadata or .keras file
        keras_file = None
        for f in os.listdir(keras_model_path):
            if f.endswith(".keras") or f == "keras_metadata.pb":
                keras_file = os.path.join(keras_model_path, f)
                break

        if keras_file and keras_file.endswith(".keras"):
            print(f"[INFO] Loading Keras model from: {keras_file}")
            model = tf.keras.models.load_model(keras_file)
            return model

    # Try loading from checkpoint
    checkpoint_prefix = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_prefix is not None:
        print(f"[INFO] Found checkpoint: {checkpoint_prefix}")
        model = build_hdmapnet_model()
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(checkpoint_prefix)
        status.expect_partial()
        print("[INFO] Checkpoint restored (expect_partial).")
        return model

    # Last resort: try to load as a single .h5 or .keras file
    for ext in [".h5", ".keras"]:
        candidate = os.path.join(checkpoint_dir, f"hdmapnet{ext}")
        if os.path.exists(candidate):
            print(f"[INFO] Loading model from: {candidate}")
            model = tf.keras.models.load_model(candidate)
            return model

    raise FileNotFoundError(
        f"Could not find a valid model in '{checkpoint_dir}'. "
        "Expected SavedModel, .keras, .h5, or a TF checkpoint."
    )


# ---------------------------------------------------------------------------
# Minimal HDMapNet Model Definition (for checkpoint restoration)
# ---------------------------------------------------------------------------

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
        # images: (B, NUM_CAMERAS, H, W, 3)
        batch_size = tf.shape(images)[0]
        # Reshape to process all cameras together
        imgs_flat = tf.reshape(images, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
        features = self.backbone(imgs_flat, training=training)
        features = self.neck(features, training=training)
        # Reshape back: (B, NUM_CAMERAS, fh, fw, C)
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
        # features: (B, NUM_CAMERAS, fh, fw, C)
        batch_size = tf.shape(features)[0]
        # Global average pool per camera, then project to BEV
        pooled = tf.reduce_mean(features, axis=[2, 3])  # (B, NUM_CAMERAS, C)
        bev_per_cam = []
        for i in range(NUM_CAMERAS):
            cam_feat = pooled[:, i, :]  # (B, C)
            bev_map = self.fc(cam_feat, training=training)  # (B, H, W, 1)
            bev_per_cam.append(bev_map)
        bev_concat = tf.concat(bev_per_cam, axis=-1)  # (B, H, W, NUM_CAMERAS)
        bev = self.combine(bev_concat, training=training)  # (B, H, W, 256)
        return bev


class LSSViewTransform(tf.keras.layers.Layer):
    """Lift-Splat-Shoot style view transform."""

    def __init__(self, num_depth_bins: int = 41, **kwargs):
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
        # features: (B, NUM_CAMERAS, fh, fw, C)
        batch_size = tf.shape(features)[0]
        # Process each camera
        cam_bevs = []
        for i in range(NUM_CAMERAS):
            cam_feat = features[:, i, :, :, :]  # (B, fh, fw, C)
            depth_logits = self.depth_net(cam_feat, training=training)
            depth_probs = tf.nn.softmax(depth_logits, axis=-1)
            # Outer product of depth and features, then reduce
            lifted = tf.expand_dims(depth_probs, -1) * tf.expand_dims(cam_feat, 3)
            # (B, fh, fw, D, C) -> reduce depth
            lifted = tf.reduce_sum(lifted, axis=3)  # (B, fh, fw, C)
            cam_bevs.append(lifted)

        # Concatenate camera BEVs and pool to standard BEV size
        combined = tf.concat(cam_bevs, axis=-1)  # (B, fh, fw, C*6)
        pooled = self.bev_pool(combined, training=training)
        bev = self.reduce(pooled, training=training)  # (B, H, W, 4)
        bev = self.final_conv(bev, training=training)  # (B, H, W, 256)
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
        # Semantic head: NUM_CLASSES channels (logits)
        self.semantic_head = tf.keras.layers.Conv2D(
            NUM_CLASSES, 1, padding="same", name="semantic_logits"
        )
        # Instance embedding head: INSTANCE_EMB_DIM channels
        self.instance_head = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(INSTANCE_EMB_DIM, 1, padding="same"),
        ], name="instance_embedding")
        # Direction head: 2 channels
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    images: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run model inference on input images.

    Args:
        model: loaded TF model (Keras or SavedModel)
        images: numpy array of shape (B, 6, 128, 352, 3), float32, range [0,1]

    Returns:
        Dictionary with keys: semantic_logits, instance_embedding, direction
    """
    images_tensor = tf.constant(images, dtype=tf.float32)

    # Handle both SavedModel and Keras model inference
    if isinstance(model, tf.keras.Model):
        outputs = model(images_tensor, training=False)
    else:
        # SavedModel: try __call__ or serve signature
        if hasattr(model, "__call__"):
            outputs = model(images_tensor)
        elif hasattr(model, "signatures"):
            serve_fn = model.signatures.get(
                "serving_default", list(model.signatures.values())[0]
            )
            outputs = serve_fn(images=images_tensor)
        else:
            raise RuntimeError("Cannot determine how to call the loaded model.")

    # Normalize output keys
    if isinstance(outputs, dict):
        result = {}
        for key in ["semantic_logits", "semantic", "output_0"]:
            if key in outputs:
                result["semantic_logits"] = outputs[key].numpy()
                break
        for key in ["instance_embedding", "instance", "output_1"]:
            if key in outputs:
                result["instance_embedding"] = outputs[key].numpy()
                break
        for key in ["direction", "output_2"]:
            if key in outputs:
                result["direction"] = outputs[key].numpy()
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


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------

def postprocess_semantic(logits: np.ndarray) -> np.ndarray:
    """Apply sigmoid and threshold to semantic logits.

    Args:
        logits: (B, H, W, NUM_CLASSES)

    Returns:
        Binary masks: (B, H, W, NUM_CLASSES), dtype uint8
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    masks = (probs >= SEMANTIC_THRESHOLD).astype(np.uint8)
    return masks


def postprocess_direction(direction: np.ndarray) -> np.ndarray:
    """Normalize direction vectors to unit length.

    Args:
        direction: (B, H, W, 2)

    Returns:
        Normalized direction vectors: (B, H, W, 2)
    """
    magnitude = np.linalg.norm(direction, axis=-1, keepdims=True)
    magnitude = np.clip(magnitude, a_min=1e-6, a_max=None)
    normalized = direction / magnitude
    return normalized


def cluster_instances(
    semantic_masks: np.ndarray,
    instance_embedding: np.ndarray,
) -> np.ndarray:
    """Cluster instance embeddings using connected components + embedding similarity.

    For each semantic class, find connected components, then merge components
    whose mean embeddings are within EMBEDDING_DISTANCE_THRESHOLD.

    Args:
        semantic_masks: (H, W, NUM_CLASSES), binary uint8
        instance_embedding: (H, W, INSTANCE_EMB_DIM), float32

    Returns:
        instance_map: (H, W), int32 with unique instance IDs (0 = background)
    """
    h, w = semantic_masks.shape[:2]
    instance_map = np.zeros((h, w), dtype=np.int32)
    current_id = 1

    for cls_idx in range(NUM_CLASSES):
        class_mask = semantic_masks[:, :, cls_idx]
        if class_mask.sum() == 0:
            continue

        # Find connected components
        labeled, num_components = ndimage.label(class_mask)

        # Compute mean embedding per component
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

        # Merge components with similar embeddings (simple greedy clustering)
        comp_ids = list(component_embeddings.keys())
        merged = {}  # comp_id -> cluster_id
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

        # Assign instance IDs
        for cid, cluster_id in merged.items():
            instance_map[component_pixels[cid]] = current_id + cluster_id - 1

        if cluster_counter > 0:
            current_id += cluster_counter

    return instance_map


def postprocess(
    raw_outputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Full post-processing pipeline for a batch of predictions.

    Args:
        raw_outputs: dict with semantic_logits, instance_embedding, direction

    Returns:
        Dictionary with processed results for each sample in batch.
    """
    semantic_logits = raw_outputs["semantic_logits"]
    instance_embedding = raw_outputs["instance_embedding"]
    direction_raw = raw_outputs["direction"]

    batch_size = semantic_logits.shape[0]

    # Semantic
    semantic_masks = postprocess_semantic(semantic_logits)

    # Direction
    direction_norm = postprocess_direction(direction_raw)

    # Instance clustering (per sample)
    instance_maps = []
    for b in range(batch_size):
        inst_map = cluster_instances(
            semantic_masks[b], instance_embedding[b]
        )
        instance_maps.append(inst_map)
    instance_maps = np.stack(instance_maps, axis=0)

    return {
        "semantic_logits": semantic_logits,
        "semantic_masks": semantic_masks,
        "instance_embedding": instance_embedding,
        "instance_maps": instance_maps,
        "direction": direction_norm,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_instance_colors(instance_map: np.ndarray) -> np.ndarray:
    """Generate unique colors for each instance using HSV spacing.

    Args:
        instance_map: (H, W), int32

    Returns:
        color_image: (H, W, 3), float32 RGB
    """
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]  # exclude background
    num_instances = len(unique_ids)

    color_image = np.zeros((*instance_map.shape, 3), dtype=np.float32)
    for idx, inst_id in enumerate(unique_ids):
        hue = idx / max(num_instances, 1)
        rgb = hsv_to_rgb([hue, 0.9, 0.9])
        color_image[instance_map == inst_id] = rgb

    return color_image


def visualize_semantic(semantic_masks: np.ndarray) -> np.ndarray:
    """Create color-coded semantic BEV map.

    Args:
        semantic_masks: (H, W, NUM_CLASSES), binary

    Returns:
        color_image: (H, W, 3), float32 RGB
    """
    h, w = semantic_masks.shape[:2]
    color_image = np.zeros((h, w, 3), dtype=np.float32)
    for cls_idx in range(NUM_CLASSES):
        mask = semantic_masks[:, :, cls_idx].astype(bool)
        color_image[mask] = CLASS_COLORS[cls_idx]
    return color_image


def plot_cameras(images: np.ndarray, ax_grid: List) -> None:
    """Plot 6 camera images in a 2x3 grid.

    Args:
        images: (6, 128, 352, 3), float32 [0,1]
        ax_grid: list of 6 matplotlib Axes
    """
    cam_names = ["front_left", "front", "front_right",
                 "back_left", "back", "back_right"]
    for idx, ax in enumerate(ax_grid):
        img = np.clip(images[idx], 0.0, 1.0)
        ax.imshow(img)
        ax.set_title(cam_names[idx], fontsize=8)
        ax.axis("off")


def plot_direction_field(
    ax,
    direction: np.ndarray,
    semantic_masks: np.ndarray,
    step: int = 8,
) -> None:
    """Plot direction vectors as quiver overlaid on semantic map.

    Args:
        ax: matplotlib Axes
        direction: (H, W, 2), normalized direction vectors
        semantic_masks: (H, W, NUM_CLASSES), binary
        step: downsample step for quiver arrows
    """
    # Background: semantic color map
    semantic_color = visualize_semantic(semantic_masks)
    ax.imshow(semantic_color, origin="lower", extent=[0, BEV_WIDTH, 0, BEV_HEIGHT])

    # Create mask of any semantic class for arrow plotting
    any_mask = semantic_masks.max(axis=-1) > 0

    # Subsample grid
    y_coords, x_coords = np.mgrid[0:BEV_HEIGHT:step, 0:BEV_WIDTH:step]
    dx = direction[::step, ::step, 0]
    dy = direction[::step, ::step, 1]
    mask_sub = any_mask[::step, ::step]

    # Only plot arrows where there is a semantic prediction
    x_plot = x_coords[mask_sub]
    y_plot = y_coords[mask_sub]
    dx_plot = dx[mask_sub]
    dy_plot = dy[mask_sub]

    if len(x_plot) > 0:
        ax.quiver(
            x_plot, y_plot, dx_plot, dy_plot,
            color="yellow", scale=30, width=0.003, headwidth=3,
            alpha=0.8,
        )
    ax.set_title("Direction Field", fontsize=10)
    ax.set_xlim(0, BEV_WIDTH)
    ax.set_ylim(0, BEV_HEIGHT)
    ax.set_aspect("equal")


def create_composite_visualization(
    images: np.ndarray,
    processed: Dict[str, np.ndarray],
    sample_idx: int = 0,
) -> plt.Figure:
    """Create a composite figure with all visualization outputs.

    Layout:
        Row 1: 6 camera images (2x3 grid within a subplot area)
        Row 2: Semantic BEV | Instance BEV | Direction Field

    Args:
        images: (B, 6, 128, 352, 3)
        processed: post-processed outputs dict
        sample_idx: batch index to visualize

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(18, 14))

    # -- Row 1: Camera images (top half) --
    gs_top = fig.add_gridspec(2, 3, left=0.05, right=0.95, top=0.95, bottom=0.55,
                              wspace=0.05, hspace=0.1)
    cam_axes = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs_top[row, col])
            cam_axes.append(ax)
    plot_cameras(images[sample_idx], cam_axes)

    # -- Row 2: BEV outputs (bottom half) --
    gs_bot = fig.add_gridspec(1, 3, left=0.05, right=0.95, top=0.48, bottom=0.02,
                              wspace=0.15)

    # Semantic segmentation
    ax_sem = fig.add_subplot(gs_bot[0, 0])
    semantic_color = visualize_semantic(processed["semantic_masks"][sample_idx])
    ax_sem.imshow(semantic_color, origin="lower", extent=[0, BEV_WIDTH, 0, BEV_HEIGHT])
    ax_sem.set_title("Semantic Segmentation (BEV)", fontsize=10)
    ax_sem.set_xlabel("x (pixels)")
    ax_sem.set_ylabel("y (pixels)")
    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    ax_sem.legend(handles=patches, loc="upper right", fontsize=7)

    # Instance segmentation
    ax_inst = fig.add_subplot(gs_bot[0, 1])
    instance_color = generate_instance_colors(processed["instance_maps"][sample_idx])
    ax_inst.imshow(instance_color, origin="lower", extent=[0, BEV_WIDTH, 0, BEV_HEIGHT])
    ax_inst.set_title("Instance Segmentation (BEV)", fontsize=10)
    ax_inst.set_xlabel("x (pixels)")

    # Direction field
    ax_dir = fig.add_subplot(gs_bot[0, 2])
    plot_direction_field(
        ax_dir,
        processed["direction"][sample_idx],
        processed["semantic_masks"][sample_idx],
    )

    fig.suptitle("HDMapNet Inference Results", fontsize=14, fontweight="bold", y=0.99)
    return fig


def save_individual_visualizations(
    images: np.ndarray,
    processed: Dict[str, np.ndarray],
    output_dir: str,
    sample_idx: int = 0,
) -> List[str]:
    """Save individual visualization images as PNGs.

    Args:
        images: (B, 6, 128, 352, 3)
        processed: post-processed outputs dict
        output_dir: directory to save images
        sample_idx: batch index to visualize

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # 1. Camera images grid
    fig_cam, axes = plt.subplots(2, 3, figsize=(12, 5))
    plot_cameras(images[sample_idx], axes.flatten().tolist())
    fig_cam.suptitle("Input Camera Images", fontsize=12)
    fig_cam.tight_layout()
    cam_path = os.path.join(output_dir, "camera_images.png")
    fig_cam.savefig(cam_path, dpi=150, bbox_inches="tight")
    plt.close(fig_cam)
    saved_paths.append(cam_path)
    print(f"  Saved: {cam_path}")

    # 2. Semantic segmentation
    fig_sem, ax_sem = plt.subplots(1, 1, figsize=(8, 8))
    semantic_color = visualize_semantic(processed["semantic_masks"][sample_idx])
    ax_sem.imshow(semantic_color, origin="lower", extent=[-30, 30, -15, 15])
    ax_sem.set_title("Semantic Segmentation (BEV)")
    ax_sem.set_xlabel("Lateral (m)")
    ax_sem.set_ylabel("Longitudinal (m)")
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    ax_sem.legend(handles=patches, loc="upper right")
    fig_sem.tight_layout()
    sem_path = os.path.join(output_dir, "semantic_bev.png")
    fig_sem.savefig(sem_path, dpi=150, bbox_inches="tight")
    plt.close(fig_sem)
    saved_paths.append(sem_path)
    print(f"  Saved: {sem_path}")

    # 3. Instance segmentation
    fig_inst, ax_inst = plt.subplots(1, 1, figsize=(8, 8))
    instance_color = generate_instance_colors(processed["instance_maps"][sample_idx])
    ax_inst.imshow(instance_color, origin="lower", extent=[-30, 30, -15, 15])
    ax_inst.set_title("Instance Segmentation (BEV)")
    ax_inst.set_xlabel("Lateral (m)")
    ax_inst.set_ylabel("Longitudinal (m)")
    num_instances = len(np.unique(processed["instance_maps"][sample_idx])) - 1
    ax_inst.text(
        0.02, 0.98, f"Instances: {num_instances}",
        transform=ax_inst.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig_inst.tight_layout()
    inst_path = os.path.join(output_dir, "instance_bev.png")
    fig_inst.savefig(inst_path, dpi=150, bbox_inches="tight")
    plt.close(fig_inst)
    saved_paths.append(inst_path)
    print(f"  Saved: {inst_path}")

    # 4. Direction field
    fig_dir, ax_dir = plt.subplots(1, 1, figsize=(8, 8))
    plot_direction_field(
        ax_dir,
        processed["direction"][sample_idx],
        processed["semantic_masks"][sample_idx],
    )
    ax_dir.set_xlabel("x (pixels)")
    ax_dir.set_ylabel("y (pixels)")
    fig_dir.tight_layout()
    dir_path = os.path.join(output_dir, "direction_field.png")
    fig_dir.savefig(dir_path, dpi=150, bbox_inches="tight")
    plt.close(fig_dir)
    saved_paths.append(dir_path)
    print(f"  Saved: {dir_path}")

    # 5. Composite view
    fig_comp = create_composite_visualization(images, processed, sample_idx)
    comp_path = os.path.join(output_dir, "composite_view.png")
    fig_comp.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig_comp)
    saved_paths.append(comp_path)
    print(f"  Saved: {comp_path}")

    return saved_paths


def save_raw_predictions(
    raw_outputs: Dict[str, np.ndarray],
    processed: Dict[str, np.ndarray],
    output_dir: str,
) -> str:
    """Save raw and processed predictions as .npz file.

    Args:
        raw_outputs: raw model outputs
        processed: post-processed results
        output_dir: directory to save

    Returns:
        Path to saved .npz file
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "predictions.npz")
    np.savez_compressed(
        npz_path,
        semantic_logits=raw_outputs["semantic_logits"],
        instance_embedding=raw_outputs["instance_embedding"],
        direction_raw=raw_outputs["direction"],
        semantic_masks=processed["semantic_masks"],
        instance_maps=processed["instance_maps"],
        direction_normalized=processed["direction"],
    )
    print(f"  Saved raw predictions: {npz_path}")
    return npz_path


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_input(input_file: str) -> Dict[str, np.ndarray]:
    """Load input data from .npz file.

    Expected keys in the .npz file:
        - images: (B, 6, 128, 352, 3) or (6, 128, 352, 3) float32 [0,1]
        Optionally:
        - intrinsics: camera intrinsic matrices
        - extrinsics: camera extrinsic matrices

    Args:
        input_file: path to .npz file

    Returns:
        Dictionary with at least 'images' key
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    data = np.load(input_file, allow_pickle=True)
    result = {}

    # Load images
    if "images" in data:
        images = data["images"].astype(np.float32)
    elif "imgs" in data:
        images = data["imgs"].astype(np.float32)
    elif "camera_images" in data:
        images = data["camera_images"].astype(np.float32)
    else:
        raise KeyError(
            f"Input file must contain 'images', 'imgs', or 'camera_images'. "
            f"Found keys: {list(data.keys())}"
        )

    # Normalize to [0, 1] if needed
    if images.max() > 1.5:
        images = images / 255.0

    # Add batch dimension if single sample
    if images.ndim == 4:
        # (6, H, W, 3) -> (1, 6, H, W, 3)
        images = np.expand_dims(images, axis=0)

    # Validate shape
    assert images.ndim == 5, f"Expected 5D images, got shape {images.shape}"
    assert images.shape[1] == NUM_CAMERAS, (
        f"Expected {NUM_CAMERAS} cameras, got {images.shape[1]}"
    )

    result["images"] = images

    # Optionally load calibration data
    for key in ["intrinsics", "extrinsics", "lidar2cam", "cam2ego"]:
        if key in data:
            result[key] = data[key]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HDMapNet TensorFlow 2 Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inference.py --checkpoint_dir ./model --input_file sample.npz --output_dir ./results
    python inference.py --checkpoint_dir ./model --input_file sample.npz --show
    python inference.py --checkpoint_dir ./model --input_file sample.npz --view_transform ipm
        """,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory (SavedModel, .keras, .h5, or TF checkpoint)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input .npz sample file containing camera images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hdmapnet_output",
        help="Directory to save visualization outputs (default: ./hdmapnet_output)",
    )
    parser.add_argument(
        "--view_transform",
        type=str,
        choices=["ipm", "lss"],
        default="lss",
        help="View transform type: 'ipm' (Inverse Perspective Mapping) or 'lss' (Lift-Splat-Shoot). Default: lss",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (requires display/GUI)",
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=0,
        help="Index of sample in batch to visualize (default: 0)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index to use (default: auto-select)",
    )
    return parser.parse_args()


def configure_gpu(gpu_index: Optional[int] = None) -> None:
    """Configure GPU memory growth and device selection."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[INFO] No GPUs detected. Running on CPU.")
        return

    try:
        if gpu_index is not None and gpu_index < len(gpus):
            tf.config.set_visible_devices(gpus[gpu_index], "GPU")
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            print(f"[INFO] Using GPU {gpu_index}: {gpus[gpu_index].name}")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Using {len(gpus)} GPU(s) with memory growth enabled.")
    except RuntimeError as e:
        print(f"[WARN] GPU configuration error: {e}")


def main() -> None:
    """Main entry point for HDMapNet inference."""
    args = parse_args()

    # Configure GPU
    configure_gpu(args.gpu)

    print("=" * 60)
    print("HDMapNet TensorFlow 2 Inference")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint_dir}")
    print(f"  Input file: {args.input_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  View transform: {args.view_transform}")
    print(f"  Show plots: {args.show}")
    print("=" * 60)

    # Step 1: Load model
    print("\n[1/5] Loading model...")
    model = load_model(args.checkpoint_dir)
    print("  Model loaded successfully.")

    # Step 2: Load input data
    print("\n[2/5] Loading input data...")
    input_data = load_input(args.input_file)
    images = input_data["images"]
    print(f"  Input shape: {images.shape}")
    print(f"  Batch size: {images.shape[0]}")
    print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")

    # Step 3: Run inference
    print("\n[3/5] Running inference...")
    raw_outputs = run_inference(model, images)
    print(f"  Semantic logits shape: {raw_outputs['semantic_logits'].shape}")
    print(f"  Instance embedding shape: {raw_outputs['instance_embedding'].shape}")
    print(f"  Direction shape: {raw_outputs['direction'].shape}")

    # Step 4: Post-processing
    print("\n[4/5] Post-processing predictions...")
    processed = postprocess(raw_outputs)
    sample_idx = min(args.batch_idx, images.shape[0] - 1)
    num_instances = len(np.unique(processed["instance_maps"][sample_idx])) - 1
    active_classes = []
    for cls_idx in range(NUM_CLASSES):
        pixel_count = processed["semantic_masks"][sample_idx, :, :, cls_idx].sum()
        if pixel_count > 0:
            active_classes.append(f"{CLASS_NAMES[cls_idx]}({pixel_count}px)")
    print(f"  Active semantic classes: {', '.join(active_classes) if active_classes else 'none'}")
    print(f"  Detected instances: {num_instances}")

    # Step 5: Save and visualize
    print("\n[5/5] Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save raw predictions
    save_raw_predictions(raw_outputs, processed, args.output_dir)

    # Save visualizations
    print("\n  Generating visualizations...")
    saved_files = save_individual_visualizations(
        images, processed, args.output_dir, sample_idx
    )

    print(f"\n{'=' * 60}")
    print(f"Inference complete. {len(saved_files)} files saved to: {args.output_dir}")
    print("=" * 60)

    # Show interactive plots if requested
    if args.show:
        print("\n[INFO] Displaying interactive plots (close window to exit)...")
        fig = create_composite_visualization(images, processed, sample_idx)
        plt.show()


if __name__ == "__main__":
    main()
