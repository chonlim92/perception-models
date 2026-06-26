#!/usr/bin/env python3
"""
Visualization script for RangeNet++ semantic segmentation results.

Loads a point cloud (.bin) and predicted labels (.label or .npy),
optionally loads ground truth labels for comparison, and produces:
  - 3D scatter plots (bird's eye view and perspective) colored by semantic class
  - Range image visualizations (H=64, W=2048) colored by predicted class
  - Side-by-side GT vs prediction comparisons
  - Open3D interactive visualization (if available)

All output figures are saved as PNG to the specified output directory.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# SemanticKITTI color map and class definitions
# ──────────────────────────────────────────────────────────────────────────────

SEMANTIC_KITTI_LABELS = {
    0: "unlabeled",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}

SEMANTIC_KITTI_COLORS_RGB = {
    0: (0, 0, 0),          # unlabeled
    1: (245, 150, 100),    # car
    2: (245, 230, 100),    # bicycle
    3: (150, 60, 30),      # motorcycle
    4: (180, 30, 80),      # truck
    5: (255, 0, 0),        # other-vehicle
    6: (30, 30, 255),      # person
    7: (200, 40, 255),     # bicyclist
    8: (90, 30, 150),      # motorcyclist
    9: (255, 0, 255),      # road
    10: (255, 150, 255),   # parking
    11: (75, 0, 75),       # sidewalk
    12: (75, 0, 175),      # other-ground
    13: (0, 200, 255),     # building
    14: (50, 120, 255),    # fence
    15: (0, 175, 0),       # vegetation
    16: (0, 60, 135),      # trunk
    17: (80, 240, 150),    # terrain
    18: (150, 240, 255),   # pole
    19: (0, 0, 255),       # traffic-sign
}

# Normalized [0,1] color map for matplotlib
SEMANTIC_KITTI_COLORS_NORM = {
    k: (r / 255.0, g / 255.0, b / 255.0)
    for k, (r, g, b) in SEMANTIC_KITTI_COLORS_RGB.items()
}

# SemanticKITTI original label ID to learning-mapped class ID
# This maps the original .label IDs (e.g., 10=car, 40=road) to sequential IDs
SEMANTICKITTI_ID_TO_CLASS = {
    0: 0,    # unlabeled
    1: 0,    # outlier -> unlabeled
    10: 1,   # car
    11: 2,   # bicycle
    13: 5,   # bus -> other-vehicle
    15: 3,   # motorcycle
    16: 5,   # on-rails -> other-vehicle
    18: 4,   # truck
    20: 5,   # other-vehicle
    30: 6,   # person
    31: 7,   # bicyclist
    32: 8,   # motorcyclist
    40: 9,   # road
    44: 10,  # parking
    48: 11,  # sidewalk
    49: 12,  # other-ground
    50: 13,  # building
    51: 14,  # fence
    52: 0,   # other-structure -> unlabeled
    60: 9,   # lane-marking -> road
    70: 15,  # vegetation
    71: 16,  # trunk
    72: 17,  # terrain
    80: 18,  # pole
    81: 19,  # traffic-sign
    99: 0,   # other-object -> unlabeled
    252: 1,  # moving-car -> car
    253: 7,  # moving-bicyclist -> bicyclist
    254: 6,  # moving-person -> person
    255: 8,  # moving-motorcyclist -> motorcyclist
    256: 5,  # moving-on-rails -> other-vehicle
    257: 5,  # moving-bus -> other-vehicle
    258: 4,  # moving-truck -> truck
    259: 5,  # moving-other-vehicle -> other-vehicle
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_point_cloud(scan_path):
    """Load a point cloud from a .bin file (float32, Nx4: x, y, z, intensity)."""
    if not os.path.isfile(scan_path):
        raise FileNotFoundError(f"Point cloud file not found: {scan_path}")
    points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_labels(label_path):
    """
    Load semantic labels from a .label file (SemanticKITTI format) or .npy file.

    For .label files: uint32 where lower 16 bits = semantic label, upper 16 bits = instance id.
    For .npy files: assumed to contain class indices directly (mapped 0..19).
    """
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    ext = os.path.splitext(label_path)[1].lower()

    if ext == ".label":
        raw_labels = np.fromfile(label_path, dtype=np.uint32)
        semantic_ids = raw_labels & 0xFFFF  # lower 16 bits
        # Map original SemanticKITTI IDs to sequential class IDs
        mapped = np.zeros_like(semantic_ids, dtype=np.int32)
        for orig_id, class_id in SEMANTICKITTI_ID_TO_CLASS.items():
            mapped[semantic_ids == orig_id] = class_id
        # Any unmapped ID gets class 0 (unlabeled)
        return mapped
    elif ext == ".npy":
        labels = np.load(label_path).astype(np.int32)
        # Clamp to valid range
        labels = np.clip(labels, 0, 19)
        return labels
    else:
        raise ValueError(f"Unsupported label format: {ext}. Use .label or .npy")


def labels_to_colors(labels):
    """Convert an array of class labels (0..19) to Nx3 float colors in [0,1]."""
    colors = np.zeros((len(labels), 3), dtype=np.float64)
    for class_id, rgb_norm in SEMANTIC_KITTI_COLORS_NORM.items():
        mask = labels == class_id
        colors[mask] = rgb_norm
    return colors


# ──────────────────────────────────────────────────────────────────────────────
# Range image generation
# ──────────────────────────────────────────────────────────────────────────────

def create_range_image(points, labels, height=64, width=2048):
    """
    Project a point cloud into a spherical range image and color by semantic label.

    Args:
        points: Nx4 array (x, y, z, intensity)
        labels: N array of class indices (0..19)
        height: vertical resolution (default 64 for Velodyne HDL-64E)
        width: horizontal resolution (default 2048)

    Returns:
        range_img: HxWx3 uint8 image colored by semantic class
        depth_img: HxW float array of range values (for reference)
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute spherical coordinates
    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Avoid division by zero
    depth_safe = np.maximum(depth, 1e-8)

    # Elevation angle (pitch)
    pitch = np.arcsin(z / depth_safe)
    # Azimuth angle (yaw)
    yaw = np.arctan2(y, x)

    # Velodyne HDL-64E vertical FOV: approximately +3 to -25 degrees
    fov_up = 3.0 / 180.0 * np.pi
    fov_down = -25.0 / 180.0 * np.pi
    fov_total = fov_up - fov_down

    # Normalize to image coordinates
    u = 0.5 * (1.0 - yaw / np.pi) * width  # horizontal: [0, width)
    v = (1.0 - (pitch - fov_down) / fov_total) * height  # vertical: [0, height)

    # Clamp to valid pixel indices
    u = np.clip(np.floor(u).astype(np.int32), 0, width - 1)
    v = np.clip(np.floor(v).astype(np.int32), 0, height - 1)

    # Initialize output images
    range_img = np.zeros((height, width, 3), dtype=np.uint8)
    depth_img = np.full((height, width), -1.0, dtype=np.float64)

    # Sort by depth (farthest first) so closer points overwrite farther ones
    order = np.argsort(-depth)
    u_sorted = u[order]
    v_sorted = v[order]
    labels_sorted = labels[order]
    depth_sorted = depth[order]

    for i in range(len(order)):
        vi = v_sorted[i]
        ui = u_sorted[i]
        lbl = labels_sorted[i]
        range_img[vi, ui] = SEMANTIC_KITTI_COLORS_RGB.get(lbl, (0, 0, 0))
        depth_img[vi, ui] = depth_sorted[i]

    return range_img, depth_img


def create_range_image_fast(points, labels, height=64, width=2048):
    """
    Vectorized range image projection (faster than the loop version).
    Uses scatter with last-write-wins for closest points.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    depth_safe = np.maximum(depth, 1e-8)

    pitch = np.arcsin(z / depth_safe)
    yaw = np.arctan2(y, x)

    fov_up = 3.0 / 180.0 * np.pi
    fov_down = -25.0 / 180.0 * np.pi
    fov_total = fov_up - fov_down

    u = (0.5 * (1.0 - yaw / np.pi) * width).astype(np.int32)
    v = ((1.0 - (pitch - fov_down) / fov_total) * height).astype(np.int32)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)

    # Sort by depth descending so that closer points are written last (overwrite)
    order = np.argsort(-depth)

    # Build color lookup table (20 classes x 3 channels)
    color_lut = np.zeros((20, 3), dtype=np.uint8)
    for cid, rgb in SEMANTIC_KITTI_COLORS_RGB.items():
        if cid < 20:
            color_lut[cid] = rgb

    # Ensure labels are in valid range
    safe_labels = np.clip(labels, 0, 19)

    # Assign colors via advanced indexing (last write wins = closest point)
    range_img = np.zeros((height, width, 3), dtype=np.uint8)
    depth_img = np.full((height, width), -1.0, dtype=np.float64)

    v_sorted = v[order]
    u_sorted = u[order]
    lbl_sorted = safe_labels[order]
    depth_sorted = depth[order]

    range_img[v_sorted, u_sorted] = color_lut[lbl_sorted]
    depth_img[v_sorted, u_sorted] = depth_sorted

    return range_img, depth_img


# ──────────────────────────────────────────────────────────────────────────────
# Matplotlib visualization
# ──────────────────────────────────────────────────────────────────────────────

def create_legend_patches():
    """Create legend patches for all semantic classes."""
    patches = []
    for class_id in sorted(SEMANTIC_KITTI_LABELS.keys()):
        color = SEMANTIC_KITTI_COLORS_NORM[class_id]
        label = SEMANTIC_KITTI_LABELS[class_id]
        patches.append(mpatches.Patch(color=color, label=f"{class_id}: {label}"))
    return patches


def plot_bev_scatter(points, colors, title, ax, subsample_factor=4):
    """Plot bird's eye view (top-down XY) scatter on given axes."""
    pts = points[::subsample_factor]
    clr = colors[::subsample_factor]

    ax.scatter(pts[:, 0], pts[:, 1], c=clr, s=0.3, edgecolors="none", alpha=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_facecolor("black")


def plot_perspective_scatter(points, colors, title, ax, subsample_factor=4):
    """Plot 3D perspective scatter on given 3D axes."""
    pts = points[::subsample_factor]
    clr = colors[::subsample_factor]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=clr, s=0.2, edgecolors="none", alpha=0.7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    # Set reasonable view angle
    ax.view_init(elev=30, azim=-60)
    ax.set_facecolor("black")


def visualize_3d_scatter(points, pred_colors, gt_colors, output_dir, subsample_factor=4):
    """
    Generate and save 3D scatter plots for predictions (and GT if available).
    Produces both bird's eye view and perspective view.
    """
    # ── Bird's Eye View ──
    if gt_colors is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        plot_bev_scatter(points, gt_colors, "Ground Truth (BEV)", axes[0], subsample_factor)
        plot_bev_scatter(points, pred_colors, "Prediction (BEV)", axes[1], subsample_factor)
        fig.suptitle("RangeNet++ Results: Bird's Eye View Comparison", fontsize=14, fontweight="bold")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        plot_bev_scatter(points, pred_colors, "Prediction (BEV)", ax, subsample_factor)
        fig.suptitle("RangeNet++ Prediction: Bird's Eye View", fontsize=14, fontweight="bold")

    # Add legend
    patches = create_legend_patches()
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=7,
               framealpha=0.9, fancybox=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    bev_path = os.path.join(output_dir, "bev_scatter.png")
    fig.savefig(bev_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {bev_path}")

    # ── Perspective View (3D) ──
    if gt_colors is not None:
        fig = plt.figure(figsize=(20, 9))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        plot_perspective_scatter(points, gt_colors, "Ground Truth (3D)", ax1, subsample_factor)
        plot_perspective_scatter(points, pred_colors, "Prediction (3D)", ax2, subsample_factor)
        fig.suptitle("RangeNet++ Results: 3D Perspective Comparison", fontsize=14, fontweight="bold")
    else:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        plot_perspective_scatter(points, pred_colors, "Prediction (3D)", ax, subsample_factor)
        fig.suptitle("RangeNet++ Prediction: 3D Perspective View", fontsize=14, fontweight="bold")

    patches = create_legend_patches()
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=7,
               framealpha=0.9, fancybox=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    persp_path = os.path.join(output_dir, "perspective_scatter.png")
    fig.savefig(persp_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {persp_path}")


def visualize_range_images(points, pred_labels, gt_labels, output_dir):
    """
    Generate and save range image visualizations colored by semantic class.
    Side-by-side GT vs prediction if GT is available.
    """
    print("  Generating range images (this may take a moment)...")

    pred_range_img, pred_depth = create_range_image_fast(points, pred_labels)

    if gt_labels is not None:
        gt_range_img, _ = create_range_image_fast(points, gt_labels)

        fig, axes = plt.subplots(2, 1, figsize=(24, 6))
        axes[0].imshow(gt_range_img)
        axes[0].set_title("Ground Truth Range Image (H=64, W=2048)", fontsize=11)
        axes[0].set_xlabel("Azimuth (pixels)")
        axes[0].set_ylabel("Elevation (pixels)")
        axes[0].set_aspect("auto")

        axes[1].imshow(pred_range_img)
        axes[1].set_title("Prediction Range Image (H=64, W=2048)", fontsize=11)
        axes[1].set_xlabel("Azimuth (pixels)")
        axes[1].set_ylabel("Elevation (pixels)")
        axes[1].set_aspect("auto")

        fig.suptitle("RangeNet++ Range Image Comparison", fontsize=13, fontweight="bold")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(24, 4))
        ax.imshow(pred_range_img)
        ax.set_title("Prediction Range Image (H=64, W=2048)", fontsize=11)
        ax.set_xlabel("Azimuth (pixels)")
        ax.set_ylabel("Elevation (pixels)")
        ax.set_aspect("auto")
        fig.suptitle("RangeNet++ Prediction Range Image", fontsize=13, fontweight="bold")

    # Add legend below the image(s)
    patches = create_legend_patches()
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=7,
               framealpha=0.9, fancybox=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    range_path = os.path.join(output_dir, "range_image.png")
    fig.savefig(range_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {range_path}")

    # Also save the depth image for reference
    fig, ax = plt.subplots(1, 1, figsize=(24, 4))
    depth_display = pred_depth.copy()
    depth_display[depth_display < 0] = 0
    im = ax.imshow(depth_display, cmap="viridis", aspect="auto")
    ax.set_title("Range (Depth) Image", fontsize=11)
    ax.set_xlabel("Azimuth (pixels)")
    ax.set_ylabel("Elevation (pixels)")
    plt.colorbar(im, ax=ax, label="Depth (m)", fraction=0.02, pad=0.02)
    plt.tight_layout()
    depth_path = os.path.join(output_dir, "range_depth.png")
    fig.savefig(depth_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {depth_path}")


def save_class_legend(output_dir):
    """Save a standalone class name legend as a PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.axis("off")
    ax.set_title("SemanticKITTI Class Legend", fontsize=12, fontweight="bold", pad=20)

    patches = create_legend_patches()
    legend = ax.legend(handles=patches, loc="center", ncol=1, fontsize=10,
                       framealpha=1.0, fancybox=True, edgecolor="gray")
    legend.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    legend_path = os.path.join(output_dir, "class_legend.png")
    fig.savefig(legend_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {legend_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Open3D visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_open3d(points, pred_colors, gt_colors=None):
    """
    Interactive 3D visualization using Open3D.
    Shows prediction point cloud; if GT is available, shows both side by side.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("  [WARNING] Open3D not available. Skipping interactive 3D visualization.")
        print("           Install with: pip install open3d")
        return

    print("  Launching Open3D visualization (close the window to continue)...")

    # Create prediction point cloud
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)

    if gt_colors is not None:
        # Show GT shifted to the left for comparison
        pcd_gt = o3d.geometry.PointCloud()
        gt_points = points[:, :3].copy()
        # Shift GT cloud to the left by the extent of the point cloud
        x_range = points[:, 0].max() - points[:, 0].min()
        gt_points[:, 0] -= (x_range + 10.0)
        pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
        pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)

        # Add labels as text (approximation: use coordinate frame)
        print("  Showing: Left = Ground Truth, Right = Prediction")
        o3d.visualization.draw_geometries(
            [pcd_gt, pcd_pred],
            window_name="RangeNet++ Results: GT (left) vs Prediction (right)",
            width=1600,
            height=900,
            point_show_normal=False,
        )
    else:
        o3d.visualization.draw_geometries(
            [pcd_pred],
            window_name="RangeNet++ Prediction",
            width=1200,
            height=800,
            point_show_normal=False,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Statistics and metrics
# ──────────────────────────────────────────────────────────────────────────────

def print_label_statistics(labels, name="Labels"):
    """Print distribution of semantic classes in the label array."""
    print(f"\n  {name} distribution:")
    print(f"  {'Class ID':<10} {'Class Name':<16} {'Count':<10} {'Percentage':<10}")
    print(f"  {'-'*46}")
    total = len(labels)
    for class_id in sorted(SEMANTIC_KITTI_LABELS.keys()):
        count = np.sum(labels == class_id)
        if count > 0:
            pct = 100.0 * count / total
            name_str = SEMANTIC_KITTI_LABELS[class_id]
            print(f"  {class_id:<10} {name_str:<16} {count:<10} {pct:.2f}%")
    print(f"  {'Total':<10} {'':<16} {total:<10}")


def compute_accuracy(pred_labels, gt_labels):
    """Compute overall and per-class accuracy if both labels are available."""
    if gt_labels is None:
        return

    overall_acc = np.mean(pred_labels == gt_labels) * 100.0
    print(f"\n  Overall Accuracy: {overall_acc:.2f}%")
    print(f"\n  {'Class ID':<10} {'Class Name':<16} {'Accuracy':<12} {'IoU':<10}")
    print(f"  {'-'*48}")

    ious = []
    for class_id in sorted(SEMANTIC_KITTI_LABELS.keys()):
        gt_mask = gt_labels == class_id
        pred_mask = pred_labels == class_id
        intersection = np.sum(gt_mask & pred_mask)
        union = np.sum(gt_mask | pred_mask)

        if union > 0:
            iou = intersection / union * 100.0
            acc = intersection / np.sum(gt_mask) * 100.0 if np.sum(gt_mask) > 0 else 0.0
            ious.append(iou)
            name_str = SEMANTIC_KITTI_LABELS[class_id]
            print(f"  {class_id:<10} {name_str:<16} {acc:.2f}%{'':>4} {iou:.2f}%")

    if ious:
        mean_iou = np.mean(ious)
        print(f"\n  Mean IoU: {mean_iou:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize RangeNet++ semantic segmentation results on LiDAR point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize predictions only
  python visualize_results.py --scan_path scan.bin --pred_path pred.label --output_dir ./vis

  # Compare predictions with ground truth
  python visualize_results.py --scan_path scan.bin --pred_path pred.npy --gt_path gt.label --output_dir ./vis

  # With Open3D interactive visualization
  python visualize_results.py --scan_path scan.bin --pred_path pred.label --output_dir ./vis --use_open3d

  # Adjust subsampling for faster rendering
  python visualize_results.py --scan_path scan.bin --pred_path pred.label --output_dir ./vis --subsample_factor 8
        """,
    )
    parser.add_argument(
        "--scan_path", type=str, required=True,
        help="Path to the point cloud .bin file (float32, Nx4: x, y, z, intensity)"
    )
    parser.add_argument(
        "--pred_path", type=str, required=True,
        help="Path to the predicted labels file (.label or .npy)"
    )
    parser.add_argument(
        "--gt_path", type=str, default=None,
        help="Path to the ground truth labels file (.label or .npy) [optional]"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./visualization_output",
        help="Directory to save output PNG figures (default: ./visualization_output)"
    )
    parser.add_argument(
        "--use_open3d", action="store_true",
        help="Launch interactive Open3D visualization (requires open3d package)"
    )
    parser.add_argument(
        "--subsample_factor", type=int, default=4,
        help="Subsample factor for matplotlib scatter plots (default: 4, i.e., every 4th point)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  RangeNet++ Results Visualization")
    print("=" * 60)

    # ── Load data ──
    print(f"\n[1/5] Loading point cloud: {args.scan_path}")
    points = load_point_cloud(args.scan_path)
    print(f"  Loaded {points.shape[0]} points (shape: {points.shape})")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
    print(f"  Intensity range: [{points[:, 3].min():.4f}, {points[:, 3].max():.4f}]")

    print(f"\n[2/5] Loading predicted labels: {args.pred_path}")
    pred_labels = load_labels(args.pred_path)
    print(f"  Loaded {len(pred_labels)} labels")
    if len(pred_labels) != points.shape[0]:
        print(f"  [WARNING] Label count ({len(pred_labels)}) != point count ({points.shape[0]})")
        min_len = min(len(pred_labels), points.shape[0])
        pred_labels = pred_labels[:min_len]
        points = points[:min_len]
        print(f"  Truncated to {min_len} points/labels")
    print_label_statistics(pred_labels, "Prediction")

    gt_labels = None
    gt_colors = None
    if args.gt_path is not None:
        print(f"\n[2b] Loading ground truth labels: {args.gt_path}")
        gt_labels = load_labels(args.gt_path)
        print(f"  Loaded {len(gt_labels)} GT labels")
        if len(gt_labels) != points.shape[0]:
            print(f"  [WARNING] GT label count ({len(gt_labels)}) != point count ({points.shape[0]})")
            min_len = min(len(gt_labels), points.shape[0])
            gt_labels = gt_labels[:min_len]
            if min_len < points.shape[0]:
                pred_labels = pred_labels[:min_len]
                points = points[:min_len]
            print(f"  Truncated to {min_len}")
        print_label_statistics(gt_labels, "Ground Truth")
        compute_accuracy(pred_labels, gt_labels)
        gt_colors = labels_to_colors(gt_labels)

    # Convert labels to colors
    pred_colors = labels_to_colors(pred_labels)

    # ── Create output directory ──
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n  Output directory: {os.path.abspath(args.output_dir)}")

    # ── Generate visualizations ──
    print(f"\n[3/5] Generating 3D scatter plots (subsample factor: {args.subsample_factor})...")
    visualize_3d_scatter(points, pred_colors, gt_colors, args.output_dir, args.subsample_factor)

    print(f"\n[4/5] Generating range image visualizations...")
    visualize_range_images(points, pred_labels, gt_labels, args.output_dir)

    # Save standalone legend
    save_class_legend(args.output_dir)

    # ── Open3D visualization ──
    if args.use_open3d:
        print(f"\n[5/5] Open3D interactive visualization...")
        visualize_open3d(points, pred_colors, gt_colors)
    else:
        print(f"\n[5/5] Open3D visualization skipped (use --use_open3d to enable)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Visualization complete!")
    print(f"  Output saved to: {os.path.abspath(args.output_dir)}")
    print("  Generated files:")
    for fname in sorted(os.listdir(args.output_dir)):
        if fname.endswith(".png"):
            fpath = os.path.join(args.output_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024.0
            print(f"    - {fname} ({size_kb:.1f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
