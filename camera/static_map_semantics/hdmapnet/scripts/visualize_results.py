#!/usr/bin/env python3
"""
Visualization script for HDMapNet prediction results.

Generates multiple visualization types from prediction and ground truth .npz files:
- Semantic map overlay on BEV grid
- 6-camera image grid
- Instance segmentation visualization
- Direction field quiver plot
- Vectorized polylines on BEV
- Composite figure combining all views
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection

# BEV grid dimensions
BEV_HEIGHT = 200
BEV_WIDTH = 200

# Camera image dimensions
CAM_HEIGHT = 128
CAM_WIDTH = 352

# Number of semantic classes
NUM_CLASSES = 3

# Semantic class colors (BGR for OpenCV, RGB for matplotlib)
SEMANTIC_COLORS_RGB = {
    0: (1.0, 0.0, 0.0),    # Lane dividers - Red
    1: (0.0, 0.0, 1.0),    # Road boundaries - Blue
    2: (0.0, 1.0, 0.0),    # Pedestrian crossings - Green
}

SEMANTIC_COLORS_UINT8 = {
    0: (255, 0, 0),      # Lane dividers - Red
    1: (0, 0, 255),      # Road boundaries - Blue
    2: (0, 255, 0),      # Pedestrian crossings - Green
}

CLASS_NAMES = {
    0: "Lane Dividers",
    1: "Road Boundaries",
    2: "Pedestrian Crossings",
}

# Camera order for visualization
CAMERA_ORDER = [
    "FRONT_LEFT", "FRONT", "FRONT_RIGHT",
    "BACK_LEFT", "BACK", "BACK_RIGHT",
]


def create_semantic_rgb(semantic_mask):
    """
    Convert a 3-channel binary semantic mask to an RGB color image.

    Args:
        semantic_mask: numpy array of shape (3, H, W) or (H, W, 3) with binary values.
                       Channel 0 = lane dividers, 1 = road boundaries, 2 = ped crossings.

    Returns:
        RGB image as numpy array of shape (H, W, 3) with float values in [0, 1].
    """
    if semantic_mask.ndim == 3 and semantic_mask.shape[0] == NUM_CLASSES:
        # Shape is (C, H, W), transpose to (H, W, C)
        mask = semantic_mask.transpose(1, 2, 0)
    elif semantic_mask.ndim == 3 and semantic_mask.shape[2] == NUM_CLASSES:
        mask = semantic_mask
    elif semantic_mask.ndim == 2:
        # Single channel with class indices
        h, w = semantic_mask.shape
        mask = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
        for c in range(NUM_CLASSES):
            mask[:, :, c] = (semantic_mask == c).astype(np.float32)
    else:
        raise ValueError(f"Unexpected semantic mask shape: {semantic_mask.shape}")

    h, w = mask.shape[0], mask.shape[1]
    rgb = np.ones((h, w, 3), dtype=np.float32) * 0.2  # Dark gray background

    for class_idx in range(NUM_CLASSES):
        class_mask = mask[:, :, class_idx] > 0.5
        color = SEMANTIC_COLORS_RGB[class_idx]
        for ch in range(3):
            rgb[:, :, ch] = np.where(class_mask, color[ch], rgb[:, :, ch])

    return rgb


def create_instance_rgb(instance_mask):
    """
    Convert an instance segmentation mask to a colored RGB image.

    Each unique instance ID gets a distinct color from the tab20 colormap.

    Args:
        instance_mask: numpy array of shape (H, W) with integer instance IDs.
                       Background is assumed to be 0.

    Returns:
        RGB image as numpy array of shape (H, W, 3) with float values in [0, 1].
    """
    h, w = instance_mask.shape[:2]
    if instance_mask.ndim == 3:
        instance_mask = instance_mask[0] if instance_mask.shape[0] < instance_mask.shape[-1] else instance_mask[:, :, 0]
        h, w = instance_mask.shape

    rgb = np.zeros((h, w, 3), dtype=np.float32)

    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]  # Remove background

    if len(unique_ids) == 0:
        return rgb

    cmap = plt.cm.get_cmap("tab20", max(len(unique_ids), 20))

    for idx, inst_id in enumerate(unique_ids):
        color = cmap(idx % 20)[:3]
        inst_pixels = instance_mask == inst_id
        for ch in range(3):
            rgb[:, :, ch] = np.where(inst_pixels, color[ch], rgb[:, :, ch])

    return rgb


def extract_polylines(binary_mask, min_contour_length=10):
    """
    Extract contours/polylines from a binary mask using OpenCV.

    Args:
        binary_mask: numpy array of shape (H, W) with binary values (0 or 1/255).
        min_contour_length: minimum number of points for a contour to be kept.

    Returns:
        List of polylines, each polyline is a numpy array of shape (N, 2) with (x, y) coords.
    """
    # Ensure uint8 format for OpenCV
    if binary_mask.dtype != np.uint8:
        mask_uint8 = (binary_mask > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = binary_mask.copy()
        if mask_uint8.max() == 1:
            mask_uint8 = mask_uint8 * 255

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    polylines = []
    for contour in contours:
        if len(contour) >= min_contour_length:
            # Reshape from (N, 1, 2) to (N, 2)
            polyline = contour.reshape(-1, 2).astype(np.float32)
            polylines.append(polyline)

    return polylines


def draw_direction_field(semantic_mask, direction_field, ax, subsample=4):
    """
    Create a quiver plot of direction vectors on the given axes.

    Only draws arrows where the semantic mask is active. Arrows are colored
    by their direction angle.

    Args:
        semantic_mask: numpy array of shape (3, H, W) or (H, W) — combined binary mask
                       indicating where direction arrows should be drawn.
        direction_field: numpy array of shape (2, H, W) with dx, dy components.
        ax: matplotlib axes to draw on.
        subsample: factor to subsample the grid for visibility.
    """
    # Get combined mask (any semantic class active)
    if semantic_mask.ndim == 3:
        if semantic_mask.shape[0] == NUM_CLASSES:
            combined_mask = semantic_mask.max(axis=0) > 0.5
        else:
            combined_mask = semantic_mask.max(axis=-1) > 0.5
    else:
        combined_mask = semantic_mask > 0.5

    # Direction field: (2, H, W)
    if direction_field.ndim == 3 and direction_field.shape[0] == 2:
        dx = direction_field[0]
        dy = direction_field[1]
    elif direction_field.ndim == 3 and direction_field.shape[2] == 2:
        dx = direction_field[:, :, 0]
        dy = direction_field[:, :, 1]
    else:
        raise ValueError(f"Unexpected direction field shape: {direction_field.shape}")

    h, w = combined_mask.shape

    # Create subsampled grid
    y_coords = np.arange(0, h, subsample)
    x_coords = np.arange(0, w, subsample)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Subsample all arrays
    mask_sub = combined_mask[::subsample, ::subsample]
    dx_sub = dx[::subsample, ::subsample]
    dy_sub = dy[::subsample, ::subsample]

    # Ensure shapes match
    min_h = min(mask_sub.shape[0], dx_sub.shape[0], xx.shape[0])
    min_w = min(mask_sub.shape[1], dx_sub.shape[1], xx.shape[1])
    mask_sub = mask_sub[:min_h, :min_w]
    dx_sub = dx_sub[:min_h, :min_w]
    dy_sub = dy_sub[:min_h, :min_w]
    xx = xx[:min_h, :min_w]
    yy = yy[:min_h, :min_w]

    # Filter to active mask locations
    active = mask_sub.flatten()
    x_active = xx.flatten()[active]
    y_active = yy.flatten()[active]
    dx_active = dx_sub.flatten()[active]
    dy_active = dy_sub.flatten()[active]

    if len(x_active) == 0:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title("Direction Field (no active pixels)")
        return

    # Compute angles for coloring
    angles = np.arctan2(dy_active, dx_active)
    # Normalize angles to [0, 1] for colormap
    angles_norm = (angles + np.pi) / (2 * np.pi)

    ax.quiver(
        x_active, y_active, dx_active, -dy_active,
        angles_norm,
        cmap="hsv",
        scale=30,
        width=0.003,
        headwidth=3,
        headlength=4,
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title("Direction Field")
    ax.set_facecolor("black")


def visualize_semantic_bev(pred_semantic, gt_semantic, output_path, dpi=150, show=False):
    """
    Visualize semantic map overlay on BEV grid.

    Shows prediction and ground truth side by side with color-coded classes:
    red=lane dividers, blue=road boundaries, green=pedestrian crossings.

    Args:
        pred_semantic: prediction semantic mask, shape (3, H, W).
        gt_semantic: ground truth semantic mask, shape (3, H, W).
        output_path: path to save the figure.
        dpi: output DPI.
        show: whether to call plt.show().
    """
    pred_rgb = create_semantic_rgb(pred_semantic)
    gt_rgb = create_semantic_rgb(gt_semantic)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(pred_rgb)
    axes[0].set_title("Prediction - Semantic BEV", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth - Semantic BEV", fontsize=12)
    axes[1].axis("off")

    # Add legend
    legend_elements = []
    for cls_idx, cls_name in CLASS_NAMES.items():
        color = SEMANTIC_COLORS_RGB[cls_idx]
        legend_elements.append(
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color,
                       markersize=10, label=cls_name)
        )
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved semantic BEV visualization to: {output_path}")


def visualize_camera_images(camera_images, output_path, dpi=150, show=False):
    """
    Visualize all 6 camera images in a 2x3 grid.

    Row 1: FRONT_LEFT, FRONT, FRONT_RIGHT
    Row 2: BACK_LEFT, BACK, BACK_RIGHT

    Args:
        camera_images: dict mapping camera name to image array (H, W, 3),
                       OR numpy array of shape (6, H, W, 3) or (6, 3, H, W).
        output_path: path to save the figure.
        dpi: output DPI.
        show: whether to call plt.show().
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    for idx, cam_name in enumerate(CAMERA_ORDER):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        if isinstance(camera_images, dict):
            img = camera_images.get(cam_name, np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8))
        elif isinstance(camera_images, np.ndarray):
            if camera_images.ndim == 4:
                img = camera_images[idx]
                # If channels-first (3, H, W), convert to (H, W, 3)
                if img.shape[0] == 3 and img.ndim == 3:
                    img = img.transpose(1, 2, 0)
            else:
                img = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        else:
            img = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

        # Normalize to [0, 1] if needed
        if img.dtype in [np.float32, np.float64]:
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        ax.imshow(img)
        ax.set_title(cam_name, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Surround Camera Views", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved camera images visualization to: {output_path}")


def visualize_instance_segmentation(pred_instance, gt_instance, output_path, dpi=150, show=False):
    """
    Visualize instance segmentation with unique colors per instance.

    Args:
        pred_instance: prediction instance mask, shape (H, W).
        gt_instance: ground truth instance mask, shape (H, W).
        output_path: path to save the figure.
        dpi: output DPI.
        show: whether to call plt.show().
    """
    pred_rgb = create_instance_rgb(pred_instance)
    gt_rgb = create_instance_rgb(gt_instance)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(pred_rgb)
    axes[0].set_title("Prediction - Instance Segmentation", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth - Instance Segmentation", fontsize=12)
    axes[1].axis("off")

    # Add instance count info
    pred_count = len(np.unique(pred_instance)) - (1 if 0 in pred_instance else 0)
    gt_count = len(np.unique(gt_instance)) - (1 if 0 in gt_instance else 0)
    fig.text(0.25, 0.02, f"Instances: {pred_count}", ha="center", fontsize=10)
    fig.text(0.75, 0.02, f"Instances: {gt_count}", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved instance segmentation visualization to: {output_path}")


def visualize_direction_field(pred_semantic, pred_direction, gt_semantic, gt_direction,
                              output_path, dpi=150, show=False):
    """
    Visualize direction fields as quiver plots.

    Args:
        pred_semantic: prediction semantic mask, shape (3, H, W).
        pred_direction: prediction direction field, shape (2, H, W).
        gt_semantic: ground truth semantic mask, shape (3, H, W).
        gt_direction: ground truth direction field, shape (2, H, W).
        output_path: path to save the figure.
        dpi: output DPI.
        show: whether to call plt.show().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    draw_direction_field(pred_semantic, pred_direction, axes[0], subsample=4)
    axes[0].set_title("Prediction - Direction Field", fontsize=12)

    draw_direction_field(gt_semantic, gt_direction, axes[1], subsample=4)
    axes[1].set_title("Ground Truth - Direction Field", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved direction field visualization to: {output_path}")


def visualize_polylines(pred_semantic, gt_semantic, output_path, dpi=150, show=False):
    """
    Visualize vectorized polylines extracted from semantic predictions on BEV.

    Extracts contours from each semantic class and draws as colored polylines.

    Args:
        pred_semantic: prediction semantic mask, shape (3, H, W).
        gt_semantic: ground truth semantic mask, shape (3, H, W).
        output_path: path to save the figure.
        dpi: output DPI.
        show: whether to call plt.show().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, semantic, title in [
        (axes[0], pred_semantic, "Prediction - Polylines"),
        (axes[1], gt_semantic, "Ground Truth - Polylines"),
    ]:
        ax.set_facecolor("black")
        ax.set_xlim(0, BEV_WIDTH)
        ax.set_ylim(BEV_HEIGHT, 0)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)

        # Handle shape
        if semantic.ndim == 3 and semantic.shape[0] == NUM_CLASSES:
            masks = semantic
        elif semantic.ndim == 3 and semantic.shape[2] == NUM_CLASSES:
            masks = semantic.transpose(2, 0, 1)
        else:
            masks = semantic[np.newaxis, :, :]

        for class_idx in range(min(NUM_CLASSES, masks.shape[0])):
            binary_mask = masks[class_idx]
            polylines = extract_polylines(binary_mask, min_contour_length=5)
            color = SEMANTIC_COLORS_RGB[class_idx]

            for polyline in polylines:
                # polyline shape: (N, 2) with (x, y)
                ax.plot(polyline[:, 0], polyline[:, 1],
                        color=color, linewidth=1.5, alpha=0.8)

    # Add legend
    legend_elements = []
    for cls_idx, cls_name in CLASS_NAMES.items():
        color = SEMANTIC_COLORS_RGB[cls_idx]
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=2, label=cls_name)
        )
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved polylines visualization to: {output_path}")


def visualize_composite(camera_images, pred_semantic, pred_instance, pred_direction,
                        output_path, dpi=150, show=False):
    """
    Create a composite figure with camera images on top and BEV predictions on bottom.

    Top row: 6 camera images (2x3 arranged as single row of 6 small images).
    Bottom row: semantic, instance, and direction visualizations side by side.

    Args:
        camera_images: camera images array or dict.
        pred_semantic: prediction semantic mask, shape (3, H, W).
        pred_instance: prediction instance mask, shape (H, W).
        pred_direction: prediction direction field, shape (2, H, W).
        output_path: path to save the figure.
        dpi: output DPI.
        show: whether to call plt.show().
    """
    fig = plt.figure(figsize=(18, 12))

    # Top section: 6 camera images in a 2x3 sub-grid
    # Using gridspec for flexible layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.1,
                          top=0.95, bottom=0.52)

    for idx, cam_name in enumerate(CAMERA_ORDER):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        if isinstance(camera_images, dict):
            img = camera_images.get(cam_name, np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8))
        elif isinstance(camera_images, np.ndarray) and camera_images.ndim == 4:
            img = camera_images[idx]
            if img.shape[0] == 3 and img.ndim == 3:
                img = img.transpose(1, 2, 0)
        else:
            img = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

        if img.dtype in [np.float32, np.float64]:
            if img.max() > 1.0:
                img = img / 255.0
            img = np.clip(img, 0, 1)
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        ax.imshow(img)
        ax.set_title(cam_name, fontsize=9)
        ax.axis("off")

    # Bottom section: 3 BEV visualizations
    gs_bottom = fig.add_gridspec(1, 3, hspace=0.1, wspace=0.15,
                                 top=0.45, bottom=0.05)

    # Semantic BEV
    ax_sem = fig.add_subplot(gs_bottom[0, 0])
    sem_rgb = create_semantic_rgb(pred_semantic)
    ax_sem.imshow(sem_rgb)
    ax_sem.set_title("Semantic BEV", fontsize=11)
    ax_sem.axis("off")

    # Instance BEV
    ax_inst = fig.add_subplot(gs_bottom[0, 1])
    inst_rgb = create_instance_rgb(pred_instance)
    ax_inst.imshow(inst_rgb)
    ax_inst.set_title("Instance BEV", fontsize=11)
    ax_inst.axis("off")

    # Direction Field BEV
    ax_dir = fig.add_subplot(gs_bottom[0, 2])
    draw_direction_field(pred_semantic, pred_direction, ax_dir, subsample=5)
    ax_dir.set_title("Direction Field", fontsize=11)

    plt.suptitle("HDMapNet - Composite Visualization", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"  Saved composite visualization to: {output_path}")


def load_npz_data(file_path):
    """
    Load data from a .npz file and return as a dict-like object.

    Expected keys in prediction/ground truth files:
    - 'semantic': shape (3, 200, 200) - binary masks per class
    - 'instance': shape (200, 200) - integer instance IDs
    - 'direction': shape (2, 200, 200) - dx, dy direction vectors
    - 'cameras': shape (6, 3, 128, 352) or (6, 128, 352, 3) - camera images

    Args:
        file_path: path to the .npz file.

    Returns:
        dict with loaded arrays.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    result = {}

    # Load semantic mask
    for key in ["semantic", "semantic_mask", "seg", "segmentation"]:
        if key in data:
            result["semantic"] = data[key]
            break

    # Load instance mask
    for key in ["instance", "instance_mask", "inst", "instance_seg"]:
        if key in data:
            result["instance"] = data[key]
            break

    # Load direction field
    for key in ["direction", "direction_field", "dir", "directions"]:
        if key in data:
            result["direction"] = data[key]
            break

    # Load camera images
    for key in ["cameras", "camera_images", "imgs", "images", "camera"]:
        if key in data:
            result["cameras"] = data[key]
            break

    return result


def ensure_shape(arr, expected_leading_dim, name="array"):
    """
    Ensure array has the expected shape conventions.

    Args:
        arr: input numpy array.
        expected_leading_dim: expected first dimension size.
        name: name for error messages.

    Returns:
        Array reshaped/transposed if needed.
    """
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] == expected_leading_dim:
        return arr
    if arr.ndim == 3 and arr.shape[2] == expected_leading_dim:
        return arr.transpose(2, 0, 1)
    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Visualize HDMapNet prediction results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_results.py --prediction_file pred.npz --ground_truth_file gt.npz --output_dir ./vis/
  python visualize_results.py --prediction_file pred.npz --ground_truth_file gt.npz --show
  python visualize_results.py --prediction_file pred.npz --ground_truth_file gt.npz --output_dir ./vis/ --dpi 300
        """,
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        required=True,
        help="Path to prediction .npz file.",
    )
    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to ground truth .npz file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualization_output",
        help="Directory to save visualizations (default: ./visualization_output).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualizations interactively with plt.show().",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for saved figures (default: 150).",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("HDMapNet Results Visualization")
    print("=" * 60)
    print(f"  Prediction file:   {args.prediction_file}")
    print(f"  Ground truth file: {args.ground_truth_file}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  DPI:               {args.dpi}")
    print(f"  Interactive show:  {args.show}")
    print("=" * 60)

    # Load data
    print("\nLoading prediction data...")
    pred_data = load_npz_data(args.prediction_file)
    print(f"  Loaded keys: {list(pred_data.keys())}")

    print("Loading ground truth data...")
    gt_data = load_npz_data(args.ground_truth_file)
    print(f"  Loaded keys: {list(gt_data.keys())}")

    # Validate required data
    pred_semantic = pred_data.get("semantic")
    gt_semantic = gt_data.get("semantic")
    pred_instance = pred_data.get("instance")
    gt_instance = gt_data.get("instance")
    pred_direction = pred_data.get("direction")
    gt_direction = gt_data.get("direction")
    camera_images = pred_data.get("cameras", gt_data.get("cameras"))

    # Ensure correct shapes
    if pred_semantic is not None:
        pred_semantic = ensure_shape(pred_semantic, NUM_CLASSES, "pred_semantic")
    if gt_semantic is not None:
        gt_semantic = ensure_shape(gt_semantic, NUM_CLASSES, "gt_semantic")
    if pred_direction is not None:
        pred_direction = ensure_shape(pred_direction, 2, "pred_direction")
    if gt_direction is not None:
        gt_direction = ensure_shape(gt_direction, 2, "gt_direction")

    print("\nGenerating visualizations...")

    # (a) Semantic BEV overlay
    if pred_semantic is not None and gt_semantic is not None:
        print("\n[1/6] Semantic BEV overlay...")
        output_path = os.path.join(args.output_dir, "semantic_bev.png")
        visualize_semantic_bev(pred_semantic, gt_semantic, output_path,
                               dpi=args.dpi, show=args.show)
    else:
        print("\n[1/6] Skipping semantic BEV (data not found).")

    # (b) Camera images grid
    if camera_images is not None:
        print("\n[2/6] Camera images grid...")
        output_path = os.path.join(args.output_dir, "camera_images.png")
        visualize_camera_images(camera_images, output_path,
                                dpi=args.dpi, show=args.show)
    else:
        print("\n[2/6] Skipping camera images (data not found).")

    # (c) Instance segmentation
    if pred_instance is not None and gt_instance is not None:
        print("\n[3/6] Instance segmentation...")
        # Handle potential extra dimensions
        if pred_instance.ndim == 3:
            pred_instance_2d = pred_instance[0] if pred_instance.shape[0] == 1 else pred_instance[:, :, 0]
        else:
            pred_instance_2d = pred_instance
        if gt_instance.ndim == 3:
            gt_instance_2d = gt_instance[0] if gt_instance.shape[0] == 1 else gt_instance[:, :, 0]
        else:
            gt_instance_2d = gt_instance

        output_path = os.path.join(args.output_dir, "instance_segmentation.png")
        visualize_instance_segmentation(pred_instance_2d, gt_instance_2d, output_path,
                                        dpi=args.dpi, show=args.show)
    else:
        print("\n[3/6] Skipping instance segmentation (data not found).")

    # (d) Direction field
    if pred_semantic is not None and pred_direction is not None:
        print("\n[4/6] Direction field...")
        gt_dir = gt_direction if gt_direction is not None else np.zeros_like(pred_direction)
        gt_sem = gt_semantic if gt_semantic is not None else pred_semantic
        output_path = os.path.join(args.output_dir, "direction_field.png")
        visualize_direction_field(pred_semantic, pred_direction, gt_sem, gt_dir,
                                  output_path, dpi=args.dpi, show=args.show)
    else:
        print("\n[4/6] Skipping direction field (data not found).")

    # (e) Vectorized polylines
    if pred_semantic is not None and gt_semantic is not None:
        print("\n[5/6] Vectorized polylines...")
        output_path = os.path.join(args.output_dir, "polylines.png")
        visualize_polylines(pred_semantic, gt_semantic, output_path,
                            dpi=args.dpi, show=args.show)
    else:
        print("\n[5/6] Skipping polylines (data not found).")

    # (f) Composite figure
    if pred_semantic is not None and pred_instance is not None and pred_direction is not None:
        print("\n[6/6] Composite figure...")
        cam_imgs = camera_images if camera_images is not None else np.zeros(
            (6, 3, CAM_HEIGHT, CAM_WIDTH), dtype=np.uint8
        )
        # Ensure instance is 2D for composite
        if pred_instance.ndim == 3:
            pi_2d = pred_instance[0] if pred_instance.shape[0] == 1 else pred_instance[:, :, 0]
        else:
            pi_2d = pred_instance

        output_path = os.path.join(args.output_dir, "composite.png")
        visualize_composite(cam_imgs, pred_semantic, pi_2d, pred_direction,
                            output_path, dpi=args.dpi, show=args.show)
    else:
        print("\n[6/6] Skipping composite (insufficient data).")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Output saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
