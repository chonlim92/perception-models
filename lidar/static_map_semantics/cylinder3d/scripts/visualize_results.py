#!/usr/bin/env python3
"""
visualize_results.py - Visualize Cylinder3D semantic segmentation results.

This script:
  - Loads .bin point clouds and .label predictions
  - Assigns colors based on SemanticKITTI semantic class color map
  - Visualizes using Open3D (with matplotlib fallback)
  - Supports side-by-side GT vs prediction comparison
  - Supports saving to .ply files
  - Supports batch visualization over multiple scans

Usage:
  # Single scan visualization
  python visualize_results.py --scan path/to/scan.bin --prediction path/to/pred.label

  # Side-by-side GT vs prediction
  python visualize_results.py --scan scan.bin --prediction pred.label --ground_truth gt.label

  # Batch mode
  python visualize_results.py --scan_dir sequences/08/velodyne/ --pred_dir predictions/08/

  # Save to PLY
  python visualize_results.py --scan scan.bin --prediction pred.label --save_ply output.ply
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# SemanticKITTI Color Map (19 classes + unlabeled)
# =============================================================================
# BGR colors from SemanticKITTI API, converted to RGB [0-255]
SEMANTICKITTI_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),          # unlabeled - black
    1: (100, 150, 245),    # car - blue
    2: (100, 230, 245),    # bicycle - cyan
    3: (30, 60, 150),      # motorcycle - dark blue
    4: (80, 30, 180),      # truck - purple
    5: (0, 0, 255),        # other-vehicle - red
    6: (255, 30, 30),      # person - bright red
    7: (255, 40, 200),     # bicyclist - pink
    8: (150, 30, 90),      # motorcyclist - dark pink
    9: (255, 0, 255),      # road - magenta
    10: (255, 150, 255),   # parking - light magenta
    11: (75, 0, 75),       # sidewalk - dark purple
    12: (175, 0, 75),      # other-ground - maroon
    13: (255, 200, 0),     # building - yellow
    14: (255, 120, 50),    # fence - orange
    15: (0, 175, 0),       # vegetation - green
    16: (135, 60, 0),      # trunk - brown
    17: (150, 240, 80),    # terrain - light green
    18: (255, 240, 150),   # pole - light yellow
    19: (255, 0, 0),       # traffic-sign - red
}

# Alternative colormap: more visually distinct
DISTINCT_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),          # unlabeled
    1: (0, 0, 230),        # car
    2: (219, 142, 0),      # bicycle
    3: (50, 50, 200),      # motorcycle
    4: (150, 0, 200),      # truck
    5: (200, 50, 200),     # other-vehicle
    6: (255, 0, 0),        # person
    7: (255, 100, 170),    # bicyclist
    8: (180, 60, 100),     # motorcyclist
    9: (128, 0, 128),      # road
    10: (200, 128, 200),   # parking
    11: (80, 0, 80),       # sidewalk
    12: (150, 50, 80),     # other-ground
    13: (255, 255, 0),     # building
    14: (255, 165, 0),     # fence
    15: (0, 200, 0),       # vegetation
    16: (139, 69, 19),     # trunk
    17: (144, 238, 144),   # terrain
    18: (255, 255, 200),   # pole
    19: (255, 50, 50),     # traffic-sign
}

# SemanticKITTI class names
CLASS_NAMES = {
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

# Learning map for raw labels
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4,
    20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11,
    49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16,
    72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6,
    255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}


# =============================================================================
# Data Loading
# =============================================================================

def load_point_cloud(bin_path: str) -> np.ndarray:
    """Load point cloud from .bin file.

    Args:
        bin_path: Path to .bin file.

    Returns:
        Points as (N, 4) array [x, y, z, intensity].
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_labels(label_path: str, remap: bool = True) -> np.ndarray:
    """Load labels from .label file.

    Args:
        label_path: Path to .label file.
        remap: Whether to remap raw labels using LEARNING_MAP.

    Returns:
        Labels as (N,) array.
    """
    raw = np.fromfile(label_path, dtype=np.uint32)
    semantic = raw & 0xFFFF

    if remap:
        mapped = np.zeros_like(semantic)
        for raw_id, mapped_id in LEARNING_MAP.items():
            mapped[semantic == raw_id] = mapped_id
        return mapped

    return semantic


def labels_to_colors(
    labels: np.ndarray, colormap: str = "semantickitti"
) -> np.ndarray:
    """Convert label indices to RGB colors.

    Args:
        labels: (N,) array of class indices.
        colormap: Which colormap to use ('semantickitti' or 'distinct').

    Returns:
        (N, 3) array of RGB colors in [0, 1].
    """
    cmap = SEMANTICKITTI_COLOR_MAP if colormap == "semantickitti" else DISTINCT_COLOR_MAP

    colors = np.zeros((len(labels), 3), dtype=np.float64)
    for cls_id, rgb in cmap.items():
        mask = labels == cls_id
        colors[mask] = np.array(rgb) / 255.0

    return colors


# =============================================================================
# Visualization with Open3D
# =============================================================================

def visualize_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    window_name: str = "Cylinder3D Visualization",
    point_size: float = 1.0,
) -> None:
    """Visualize colored point cloud using Open3D.

    Args:
        points: (N, 3) xyz coordinates.
        colors: (N, 3) RGB colors in [0, 1].
        window_name: Window title.
        point_size: Point rendering size.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(pcd)

    # Set rendering options
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([0.1, 0.1, 0.1])

    # Set initial view
    view_ctrl = vis.get_view_control()
    view_ctrl.set_front([0.0, -0.5, -0.8])
    view_ctrl.set_up([0.0, -1.0, 0.0])
    view_ctrl.set_lookat([0.0, 0.0, 20.0])
    view_ctrl.set_zoom(0.3)

    vis.run()
    vis.destroy_window()


def visualize_side_by_side_open3d(
    points: np.ndarray,
    gt_colors: np.ndarray,
    pred_colors: np.ndarray,
    offset: float = 50.0,
    window_name: str = "GT (left) vs Prediction (right)",
    point_size: float = 1.0,
) -> None:
    """Visualize GT and prediction side by side using Open3D.

    Args:
        points: (N, 3) xyz coordinates.
        gt_colors: (N, 3) RGB colors for ground truth.
        pred_colors: (N, 3) RGB colors for predictions.
        offset: X-offset between GT and prediction point clouds.
        window_name: Window title.
        point_size: Point rendering size.
    """
    import open3d as o3d

    # Ground truth (left)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)

    # Prediction (right, offset)
    points_pred = points[:, :3].copy()
    points_pred[:, 0] += offset
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    pcd_pred.colors = o3d.utility.Vector3dVector(pred_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1920, height=720)
    vis.add_geometry(pcd_gt)
    vis.add_geometry(pcd_pred)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([0.1, 0.1, 0.1])

    view_ctrl = vis.get_view_control()
    view_ctrl.set_front([0.0, -0.5, -0.8])
    view_ctrl.set_up([0.0, -1.0, 0.0])
    view_ctrl.set_lookat([offset / 2, 0.0, 20.0])
    view_ctrl.set_zoom(0.2)

    vis.run()
    vis.destroy_window()


# =============================================================================
# Visualization with Matplotlib (fallback)
# =============================================================================

def visualize_matplotlib(
    points: np.ndarray,
    colors: np.ndarray,
    title: str = "Point Cloud",
    max_points: int = 50000,
    elevation: float = 30.0,
    azimuth: float = -60.0,
    save_path: Optional[str] = None,
) -> None:
    """Visualize point cloud using matplotlib 3D scatter (fallback).

    Args:
        points: (N, 3+) array with xyz.
        colors: (N, 3) RGB colors in [0, 1].
        title: Plot title.
        max_points: Maximum points to render (downsample for performance).
        elevation: Camera elevation angle.
        azimuth: Camera azimuth angle.
        save_path: If provided, save figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Downsample for matplotlib performance
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors,
        s=0.5,
        alpha=0.8,
        edgecolors="none",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elevation, azim=azimuth)

    # Set axis limits for better framing
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-3, 10])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_side_by_side_matplotlib(
    points: np.ndarray,
    gt_colors: np.ndarray,
    pred_colors: np.ndarray,
    title: str = "GT vs Prediction",
    max_points: int = 30000,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side visualization using matplotlib.

    Args:
        points: (N, 3+) array.
        gt_colors: (N, 3) RGB for ground truth.
        pred_colors: (N, 3) RGB for prediction.
        title: Plot title.
        max_points: Maximum points per subplot.
        save_path: Path to save figure.
    """
    import matplotlib.pyplot as plt

    # Downsample
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        gt_colors = gt_colors[indices]
        pred_colors = pred_colors[indices]

    fig = plt.figure(figsize=(20, 8))

    # Ground truth
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=gt_colors, s=0.3, alpha=0.8, edgecolors="none",
    )
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=30, azim=-60)

    # Prediction
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=pred_colors, s=0.3, alpha=0.8, edgecolors="none",
    )
    ax2.set_title("Prediction")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=30, azim=-60)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# PLY Export
# =============================================================================

def save_to_ply(
    points: np.ndarray, colors: np.ndarray, output_path: str
) -> None:
    """Save colored point cloud to PLY file.

    Args:
        points: (N, 3+) array with xyz coordinates.
        colors: (N, 3) RGB colors in [0, 1].
        output_path: Path to save .ply file.
    """
    n_points = len(points)
    colors_uint8 = (colors * 255).astype(np.uint8)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write(header)
        for i in range(n_points):
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n"
            )

    print(f"  Saved PLY ({n_points:,} points): {output_path}")


def save_to_ply_binary(
    points: np.ndarray, colors: np.ndarray, output_path: str
) -> None:
    """Save colored point cloud to binary PLY file (faster for large clouds).

    Args:
        points: (N, 3+) array with xyz coordinates.
        colors: (N, 3) RGB colors in [0, 1].
        output_path: Path to save .ply file.
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
        print(f"  Saved PLY binary ({len(points):,} points): {output_path}")
    except ImportError:
        # Fallback to ASCII PLY
        save_to_ply(points, colors, output_path)


# =============================================================================
# Batch Visualization
# =============================================================================

def batch_visualize(
    scan_dir: str,
    pred_dir: str,
    gt_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    colormap: str = "semantickitti",
    max_scans: int = 50,
    use_open3d: bool = True,
    save_ply: bool = False,
) -> None:
    """Batch visualize multiple scans.

    Args:
        scan_dir: Directory containing .bin scan files.
        pred_dir: Directory containing .label prediction files.
        gt_dir: Optional directory containing .label ground truth files.
        output_dir: Directory to save visualizations.
        colormap: Colormap to use.
        max_scans: Maximum number of scans to process.
        use_open3d: Whether to use Open3D (vs matplotlib).
        save_ply: Whether to save PLY files.
    """
    scan_files = sorted(Path(scan_dir).glob("*.bin"))[:max_scans]

    if not scan_files:
        print(f"No .bin files found in {scan_dir}")
        return

    print(f"Processing {len(scan_files)} scans...")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, scan_file in enumerate(scan_files):
        scan_name = scan_file.stem
        print(f"\n[{i+1}/{len(scan_files)}] Processing {scan_name}...")

        # Load point cloud
        points = load_point_cloud(str(scan_file))

        # Load prediction
        pred_file = Path(pred_dir) / f"{scan_name}.label"
        if not pred_file.exists():
            print(f"  WARNING: Prediction not found: {pred_file}")
            continue
        pred_labels = load_labels(str(pred_file), remap=False)
        pred_colors = labels_to_colors(pred_labels, colormap)

        # Load GT if available
        gt_colors = None
        if gt_dir:
            gt_file = Path(gt_dir) / f"{scan_name}.label"
            if gt_file.exists():
                gt_labels = load_labels(str(gt_file), remap=True)
                gt_colors = labels_to_colors(gt_labels, colormap)

        # Save PLY
        if save_ply and output_dir:
            ply_path = os.path.join(output_dir, f"{scan_name}_pred.ply")
            save_to_ply_binary(points, pred_colors, ply_path)
            if gt_colors is not None:
                ply_gt_path = os.path.join(output_dir, f"{scan_name}_gt.ply")
                save_to_ply_binary(points, gt_colors, ply_gt_path)

        # Visualize
        if not output_dir or not save_ply:  # Interactive mode
            if gt_colors is not None:
                if use_open3d:
                    try:
                        visualize_side_by_side_open3d(
                            points, gt_colors, pred_colors,
                            window_name=f"{scan_name} - GT (left) vs Pred (right)",
                        )
                    except ImportError:
                        visualize_side_by_side_matplotlib(
                            points, gt_colors, pred_colors,
                            title=f"{scan_name} - GT vs Pred",
                        )
            else:
                if use_open3d:
                    try:
                        visualize_open3d(
                            points, pred_colors,
                            window_name=f"{scan_name} - Prediction",
                        )
                    except ImportError:
                        visualize_matplotlib(
                            points, pred_colors,
                            title=f"{scan_name} - Prediction",
                        )
        elif output_dir:
            # Save as image
            img_path = os.path.join(output_dir, f"{scan_name}.png")
            if gt_colors is not None:
                visualize_side_by_side_matplotlib(
                    points, gt_colors, pred_colors,
                    title=f"{scan_name} - GT vs Pred",
                    save_path=img_path,
                )
            else:
                visualize_matplotlib(
                    points, pred_colors,
                    title=f"{scan_name} - Prediction",
                    save_path=img_path,
                )

    print(f"\nBatch visualization complete. Processed {len(scan_files)} scans.")


# =============================================================================
# Legend Generation
# =============================================================================

def create_legend_image(
    output_path: Optional[str] = None, colormap: str = "semantickitti"
) -> None:
    """Create a color legend image for the semantic classes.

    Args:
        output_path: Path to save legend image. If None, display interactively.
        colormap: Which colormap to show.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cmap = SEMANTICKITTI_COLOR_MAP if colormap == "semantickitti" else DISTINCT_COLOR_MAP

    fig, ax = plt.subplots(1, 1, figsize=(4, 8))

    patches = []
    for cls_id in sorted(cmap.keys()):
        color = np.array(cmap[cls_id]) / 255.0
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        patch = mpatches.Patch(color=color, label=f"{cls_id}: {name}")
        patches.append(patch)

    ax.legend(handles=patches, loc="center", fontsize=10, frameon=False)
    ax.axis("off")
    ax.set_title("SemanticKITTI Class Legend", fontsize=12, fontweight="bold")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Legend saved to: {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize Cylinder3D semantic segmentation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single prediction
  python visualize_results.py --scan 000000.bin --prediction 000000.label

  # Side-by-side GT vs prediction
  python visualize_results.py --scan 000000.bin --prediction pred.label --ground_truth gt.label

  # Save as PLY
  python visualize_results.py --scan 000000.bin --prediction 000000.label --save_ply output.ply

  # Batch mode with output images
  python visualize_results.py --scan_dir velodyne/ --pred_dir predictions/ --output_dir viz/

  # Generate color legend
  python visualize_results.py --legend --output_dir ./

  # Use matplotlib fallback (no Open3D required)
  python visualize_results.py --scan 000000.bin --prediction 000000.label --backend matplotlib
        """,
    )

    # Input options
    input_group = parser.add_argument_group("Input options")
    input_group.add_argument(
        "--scan", type=str, help="Path to a single .bin scan file"
    )
    input_group.add_argument(
        "--prediction", type=str, help="Path to prediction .label file"
    )
    input_group.add_argument(
        "--ground_truth", type=str, help="Path to ground truth .label file"
    )

    # Batch mode options
    batch_group = parser.add_argument_group("Batch mode options")
    batch_group.add_argument(
        "--scan_dir", type=str, help="Directory of .bin scan files (batch mode)"
    )
    batch_group.add_argument(
        "--pred_dir", type=str, help="Directory of prediction .label files (batch mode)"
    )
    batch_group.add_argument(
        "--gt_dir", type=str, help="Directory of GT .label files (batch mode)"
    )
    batch_group.add_argument(
        "--max_scans", type=int, default=50,
        help="Max scans in batch mode (default: 50)",
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--output_dir", type=str, help="Output directory for saved visualizations"
    )
    output_group.add_argument(
        "--save_ply", type=str, help="Save visualization as .ply file"
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization options")
    viz_group.add_argument(
        "--colormap",
        type=str,
        default="semantickitti",
        choices=["semantickitti", "distinct"],
        help="Color map to use (default: semantickitti)",
    )
    viz_group.add_argument(
        "--backend",
        type=str,
        default="open3d",
        choices=["open3d", "matplotlib"],
        help="Visualization backend (default: open3d)",
    )
    viz_group.add_argument(
        "--point_size",
        type=float,
        default=1.5,
        help="Point size for rendering (default: 1.5)",
    )
    viz_group.add_argument(
        "--max_points",
        type=int,
        default=100000,
        help="Max points for matplotlib backend (default: 100000)",
    )
    viz_group.add_argument(
        "--no_remap",
        action="store_true",
        help="Don't remap labels (already mapped to 0-19)",
    )

    # Utility options
    util_group = parser.add_argument_group("Utility options")
    util_group.add_argument(
        "--legend",
        action="store_true",
        help="Generate and display/save color legend",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Generate legend
    if args.legend:
        legend_path = None
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            legend_path = os.path.join(args.output_dir, "legend.png")
        create_legend_image(output_path=legend_path, colormap=args.colormap)
        if not args.scan and not args.scan_dir:
            return

    # Batch mode
    if args.scan_dir and args.pred_dir:
        batch_visualize(
            scan_dir=args.scan_dir,
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            colormap=args.colormap,
            max_scans=args.max_scans,
            use_open3d=(args.backend == "open3d"),
            save_ply=bool(args.output_dir),
        )
        return

    # Single scan mode
    if not args.scan:
        print("ERROR: Please provide --scan for single scan mode or")
        print("       --scan_dir and --pred_dir for batch mode.")
        print("       Use --help for usage information.")
        sys.exit(1)

    if not args.prediction:
        print("ERROR: Please provide --prediction for the label file.")
        sys.exit(1)

    # Load data
    print(f"Loading scan: {args.scan}")
    points = load_point_cloud(args.scan)
    print(f"  Loaded {len(points):,} points")

    print(f"Loading prediction: {args.prediction}")
    pred_labels = load_labels(args.prediction, remap=not args.no_remap)
    pred_colors = labels_to_colors(pred_labels, args.colormap)
    print(f"  Loaded {len(pred_labels):,} labels")

    # Verify point/label count match
    if len(points) != len(pred_labels):
        print(
            f"  WARNING: Point count ({len(points)}) != label count "
            f"({len(pred_labels)}). Truncating to minimum."
        )
        min_len = min(len(points), len(pred_labels))
        points = points[:min_len]
        pred_colors = pred_colors[:min_len]

    # Load GT if provided
    gt_colors = None
    if args.ground_truth:
        print(f"Loading ground truth: {args.ground_truth}")
        gt_labels = load_labels(args.ground_truth, remap=True)
        gt_colors = labels_to_colors(gt_labels, args.colormap)
        print(f"  Loaded {len(gt_labels):,} GT labels")

    # Print class distribution
    unique, counts = np.unique(pred_labels, return_counts=True)
    print("\n  Prediction class distribution:")
    for cls_id, count in zip(unique, counts):
        pct = count / len(pred_labels) * 100
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        print(f"    {cls_id:>2}: {name:<16} {count:>8,} ({pct:.1f}%)")

    # Save PLY if requested
    if args.save_ply:
        save_to_ply_binary(points, pred_colors, args.save_ply)
        if gt_colors is not None:
            gt_ply = args.save_ply.replace(".ply", "_gt.ply")
            save_to_ply_binary(points, gt_colors, gt_ply)

    # Visualize
    use_open3d = args.backend == "open3d"

    if gt_colors is not None:
        # Side-by-side mode
        if use_open3d:
            try:
                visualize_side_by_side_open3d(
                    points, gt_colors, pred_colors,
                    point_size=args.point_size,
                )
            except ImportError:
                print("  Open3D not available, falling back to matplotlib...")
                visualize_side_by_side_matplotlib(
                    points, gt_colors, pred_colors,
                    max_points=args.max_points,
                    save_path=os.path.join(args.output_dir, "comparison.png")
                    if args.output_dir else None,
                )
        else:
            visualize_side_by_side_matplotlib(
                points, gt_colors, pred_colors,
                max_points=args.max_points,
                save_path=os.path.join(args.output_dir, "comparison.png")
                if args.output_dir else None,
            )
    else:
        # Single prediction mode
        if use_open3d:
            try:
                visualize_open3d(
                    points, pred_colors,
                    point_size=args.point_size,
                )
            except ImportError:
                print("  Open3D not available, falling back to matplotlib...")
                visualize_matplotlib(
                    points, pred_colors,
                    max_points=args.max_points,
                    save_path=os.path.join(args.output_dir, "prediction.png")
                    if args.output_dir else None,
                )
        else:
            visualize_matplotlib(
                points, pred_colors,
                max_points=args.max_points,
                save_path=os.path.join(args.output_dir, "prediction.png")
                if args.output_dir else None,
            )

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
