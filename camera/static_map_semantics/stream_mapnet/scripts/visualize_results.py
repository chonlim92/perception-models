#!/usr/bin/env python3
"""
StreamMapNet - Visualization Script

Visualizes StreamMapNet predictions alongside ground truth in BEV (top-down)
view and optionally shows corresponding camera images.

Features:
  - BEV plot with lane dividers (blue), road boundaries (red),
    pedestrian crossings (green)
  - GT shown as solid lines, predictions as dashed lines
  - Camera image grid alongside BEV
  - Save individual figures or generate video for sequences

Usage:
    # Visualize single sample
    python visualize_results.py --predictions results/preds.pkl --gt data/nuscenes/map_gt/val_map_gt.pkl --sample-idx 0

    # Visualize full sequence and save video
    python visualize_results.py --predictions results/preds.pkl --gt data/nuscenes/map_gt/val_map_gt.pkl --video --output-dir vis_output/

    # Visualize without camera images
    python visualize_results.py --predictions results/preds.pkl --gt data/nuscenes/map_gt/val_map_gt.pkl --no-cameras
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    from matplotlib.gridspec import GridSpec
except ImportError:
    pass


# =============================================================================
# Configuration
# =============================================================================

# Colors for map element categories
COLORS = {
    "lane_divider": "#2196F3",       # Blue
    "road_boundary": "#F44336",       # Red
    "pedestrian_crossing": "#4CAF50",  # Green
}

# Display names for legend
DISPLAY_NAMES = {
    "lane_divider": "Lane Dividers",
    "road_boundary": "Road Boundaries",
    "pedestrian_crossing": "Ped. Crossings",
}

# Camera names in nuScenes (display order)
CAMERA_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

# Short camera labels for display
CAMERA_LABELS = {
    "CAM_FRONT_LEFT": "Front Left",
    "CAM_FRONT": "Front",
    "CAM_FRONT_RIGHT": "Front Right",
    "CAM_BACK_LEFT": "Back Left",
    "CAM_BACK": "Back",
    "CAM_BACK_RIGHT": "Back Right",
}


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_polylines(
    ax: plt.Axes,
    polylines: List[np.ndarray],
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 0.8,
    label: Optional[str] = None,
) -> None:
    """
    Plot a list of polylines on a matplotlib axis.

    Args:
        ax: Matplotlib axis
        polylines: List of (K, 2) arrays
        color: Line color
        linestyle: Line style ('-' solid, '--' dashed)
        linewidth: Line width
        alpha: Transparency
        label: Legend label (only applied to first polyline)
    """
    for i, polyline in enumerate(polylines):
        if polyline is None or len(polyline) < 2:
            continue
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if i == 0 else None,
        )


def plot_bev(
    ax: plt.Axes,
    gt_data: Dict,
    pred_data: Optional[Dict] = None,
    bev_range: float = 60.0,
    title: str = "BEV Map View",
) -> None:
    """
    Plot BEV (bird's eye view) map with GT and predictions.

    Args:
        ax: Matplotlib axis
        gt_data: Ground truth dictionary with map elements
        pred_data: Optional predictions dictionary
        bev_range: BEV range in meters for axis limits
        title: Plot title
    """
    # Plot ground truth (solid lines)
    if "lane_dividers" in gt_data:
        plot_polylines(
            ax,
            gt_data["lane_dividers"],
            color=COLORS["lane_divider"],
            linestyle="-",
            linewidth=2.0,
            label="Lane Div. (GT)",
        )

    if "road_boundaries" in gt_data:
        plot_polylines(
            ax,
            gt_data["road_boundaries"],
            color=COLORS["road_boundary"],
            linestyle="-",
            linewidth=2.0,
            label="Road Bnd. (GT)",
        )

    if "pedestrian_crossings" in gt_data:
        plot_polylines(
            ax,
            gt_data["pedestrian_crossings"],
            color=COLORS["pedestrian_crossing"],
            linestyle="-",
            linewidth=2.0,
            label="Ped. Cross. (GT)",
        )

    # Plot predictions (dashed lines)
    if pred_data is not None:
        if "lane_dividers" in pred_data:
            plot_polylines(
                ax,
                pred_data["lane_dividers"],
                color=COLORS["lane_divider"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Lane Div. (Pred)",
            )

        if "road_boundaries" in pred_data:
            plot_polylines(
                ax,
                pred_data["road_boundaries"],
                color=COLORS["road_boundary"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Road Bnd. (Pred)",
            )

        if "pedestrian_crossings" in pred_data:
            plot_polylines(
                ax,
                pred_data["pedestrian_crossings"],
                color=COLORS["pedestrian_crossing"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Ped. Cross. (Pred)",
            )

    # Draw ego vehicle
    ego_rect = mpatches.FancyBboxPatch(
        (-1.0, -2.0),
        2.0,
        4.0,
        boxstyle="round,pad=0.1",
        facecolor="gray",
        edgecolor="black",
        alpha=0.6,
    )
    ax.add_patch(ego_rect)
    ax.annotate(
        "EGO",
        (0, 0),
        ha="center",
        va="center",
        fontsize=6,
        fontweight="bold",
        color="white",
    )

    # Draw direction arrow
    ax.annotate(
        "",
        xy=(0, 5),
        xytext=(0, 2.5),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    # Configure axis
    ax.set_xlim(-bev_range, bev_range)
    ax.set_ylim(-bev_range, bev_range)
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=7, ncol=1)

    # Draw BEV range boundary
    range_rect = mpatches.Rectangle(
        (-bev_range, -bev_range),
        2 * bev_range,
        2 * bev_range,
        fill=False,
        edgecolor="gray",
        linestyle=":",
        linewidth=0.5,
    )
    ax.add_patch(range_rect)


def load_camera_images(
    dataroot: str,
    nusc,
    sample_token: str,
) -> Dict[str, np.ndarray]:
    """
    Load all camera images for a given sample.

    Args:
        dataroot: Path to nuScenes data root
        nusc: NuScenes instance
        sample_token: Sample token

    Returns:
        Dictionary mapping camera names to image arrays
    """
    images = {}
    sample = nusc.get("sample", sample_token)

    for cam_name in CAMERA_NAMES:
        if cam_name in sample["data"]:
            cam_data = nusc.get("sample_data", sample["data"][cam_name])
            img_path = os.path.join(dataroot, cam_data["filename"])
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                images[cam_name] = img

    return images


def plot_camera_grid(
    axes: List[plt.Axes],
    images: Dict[str, np.ndarray],
) -> None:
    """
    Plot camera images in a grid layout.

    Args:
        axes: List of 6 matplotlib axes (2 rows x 3 cols)
        images: Dictionary mapping camera names to image arrays
    """
    for i, cam_name in enumerate(CAMERA_NAMES):
        ax = axes[i]
        if cam_name in images:
            ax.imshow(images[cam_name])
        else:
            ax.text(
                0.5, 0.5, "No Image",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray"
            )
        ax.set_title(CAMERA_LABELS.get(cam_name, cam_name), fontsize=8)
        ax.axis("off")


def visualize_sample(
    gt_data: Dict,
    pred_data: Optional[Dict] = None,
    images: Optional[Dict[str, np.ndarray]] = None,
    bev_range: float = 60.0,
    save_path: Optional[str] = None,
    show: bool = True,
    sample_idx: int = 0,
) -> None:
    """
    Visualize a single sample with BEV map and camera images.

    Args:
        gt_data: Ground truth dictionary
        pred_data: Optional predictions dictionary
        images: Optional camera images dictionary
        bev_range: BEV range for plot limits
        save_path: Path to save figure (None to skip saving)
        show: Whether to display the figure
        sample_idx: Sample index for title
    """
    if images:
        # Create figure with BEV + camera grid
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 4, figure=fig, width_ratios=[2, 1, 1, 1])

        # BEV plot (left, spanning both rows)
        ax_bev = fig.add_subplot(gs[:, 0])

        # Camera images (right side, 2x3 grid)
        cam_axes = []
        for row in range(2):
            for col in range(1, 4):
                ax = fig.add_subplot(gs[row, col])
                cam_axes.append(ax)

        plot_bev(
            ax_bev, gt_data, pred_data, bev_range,
            title=f"BEV Map (Sample #{sample_idx})"
        )
        plot_camera_grid(cam_axes, images)

    else:
        # BEV only
        fig, ax_bev = plt.subplots(1, 1, figsize=(10, 10))
        plot_bev(
            ax_bev, gt_data, pred_data, bev_range,
            title=f"BEV Map (Sample #{sample_idx})"
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def create_video(
    output_path: str,
    frame_dir: str,
    fps: int = 2,
) -> None:
    """
    Create video from saved frames using matplotlib animation or ffmpeg.

    Args:
        output_path: Path to output video file
        frame_dir: Directory containing frame images
        fps: Frames per second
    """
    try:
        import subprocess

        # Get sorted frame files
        frames = sorted(Path(frame_dir).glob("frame_*.png"))
        if not frames:
            print("  No frames found to create video.")
            return

        # Try ffmpeg first
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir) + "/frame_%06d.png",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            output_path,
        ]

        result = subprocess.run(
            ffmpeg_cmd, capture_output=True, text=True
        )

        if result.returncode == 0:
            print(f"  Video saved: {output_path}")
        else:
            print(f"  ffmpeg failed, trying imageio...")
            _create_video_imageio(output_path, frames, fps)

    except FileNotFoundError:
        print("  ffmpeg not found, trying imageio...")
        frames = sorted(Path(frame_dir).glob("frame_*.png"))
        _create_video_imageio(output_path, frames, fps)


def _create_video_imageio(
    output_path: str, frames: List[Path], fps: int
) -> None:
    """Fallback video creation using imageio."""
    try:
        import imageio

        writer = imageio.get_writer(output_path, fps=fps)
        for frame_path in frames:
            frame = imageio.imread(str(frame_path))
            writer.append_data(frame)
        writer.close()
        print(f"  Video saved: {output_path}")
    except ImportError:
        print("  Warning: Neither ffmpeg nor imageio available for video creation.")
        print("  Install imageio: pip install imageio imageio-ffmpeg")


# =============================================================================
# Data Loading
# =============================================================================


def load_predictions(pred_path: str) -> List[Dict]:
    """
    Load prediction results from pickle file.

    Expected format: list of dicts, each with:
        - 'sample_token': str
        - 'lane_dividers': list of (K, 2) arrays
        - 'road_boundaries': list of (K, 2) arrays
        - 'pedestrian_crossings': list of (K, 2) arrays
        - 'scores': optional confidence scores

    Args:
        pred_path: Path to predictions pickle file

    Returns:
        List of prediction dictionaries
    """
    with open(pred_path, "rb") as f:
        predictions = pickle.load(f)

    if isinstance(predictions, dict):
        # Single prediction - wrap in list
        predictions = [predictions]

    return predictions


def load_ground_truth(gt_path: str) -> List[Dict]:
    """
    Load ground truth from pickle file.

    Args:
        gt_path: Path to ground truth pickle file

    Returns:
        List of ground truth dictionaries
    """
    with open(gt_path, "rb") as f:
        gt_data = pickle.load(f)

    if isinstance(gt_data, dict):
        gt_data = [gt_data]

    return gt_data


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize StreamMapNet predictions and ground truth"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions pickle file",
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to ground truth pickle file",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
        help="nuScenes data root (for loading camera images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vis_output",
        help="Output directory for visualizations (default: ./vis_output)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Visualize a specific sample index (default: all samples)",
    )
    parser.add_argument(
        "--bev-range",
        type=float,
        default=60.0,
        help="BEV range in meters (default: 60.0)",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Skip camera image visualization",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Create video from sequential visualizations",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Video frames per second (default: 2)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (default: save only)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to visualize",
    )
    args = parser.parse_args()

    # Use non-interactive backend if not showing
    if not args.show:
        matplotlib.use("Agg")

    # Load data
    print("Loading predictions...")
    predictions = load_predictions(args.predictions)
    print(f"  Loaded {len(predictions)} predictions")

    print("Loading ground truth...")
    gt_data = load_ground_truth(args.gt)
    print(f"  Loaded {len(gt_data)} ground truth samples")

    # Load nuScenes for camera images if needed
    nusc = None
    if not args.no_cameras and args.dataroot:
        try:
            from nuscenes.nuscenes import NuScenes

            print("Loading nuScenes for camera images...")
            # Try to detect version
            if os.path.exists(os.path.join(args.dataroot, "v1.0-mini")):
                version = "v1.0-mini"
            else:
                version = "v1.0-trainval"
            nusc = NuScenes(version=version, dataroot=args.dataroot, verbose=False)
        except Exception as e:
            print(f"  Warning: Could not load nuScenes: {e}")
            print("  Continuing without camera images.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

    # Determine samples to visualize
    if args.sample_idx is not None:
        indices = [args.sample_idx]
    else:
        indices = list(range(len(gt_data)))
        if args.max_samples:
            indices = indices[: args.max_samples]

    # Match predictions to GT by sample token
    pred_by_token = {}
    for pred in predictions:
        if "sample_token" in pred:
            pred_by_token[pred["sample_token"]] = pred

    # Visualize
    print(f"\nVisualizing {len(indices)} samples...")
    print(f"  Output: {output_dir}")

    for frame_num, idx in enumerate(indices):
        if idx >= len(gt_data):
            print(f"  Warning: Index {idx} out of range, skipping.")
            continue

        gt_sample = gt_data[idx]
        sample_token = gt_sample.get("sample_token", f"sample_{idx}")

        # Find matching prediction
        pred_sample = pred_by_token.get(sample_token)
        if pred_sample is None and idx < len(predictions):
            # Fall back to index-based matching
            pred_sample = predictions[idx]

        # Load camera images if available
        images = None
        if nusc and not args.no_cameras:
            try:
                images = load_camera_images(args.dataroot, nusc, sample_token)
            except Exception:
                pass

        # Generate visualization
        if args.video:
            save_path = str(frame_dir / f"frame_{frame_num:06d}.png")
        else:
            save_path = str(output_dir / f"vis_{sample_token[:16]}.png")

        visualize_sample(
            gt_data=gt_sample,
            pred_data=pred_sample,
            images=images,
            bev_range=args.bev_range,
            save_path=save_path,
            show=args.show and args.sample_idx is not None,
            sample_idx=idx,
        )

    # Create video if requested
    if args.video:
        video_path = str(output_dir / "sequence.mp4")
        print(f"\nCreating video ({len(indices)} frames at {args.fps} fps)...")
        create_video(video_path, str(frame_dir), fps=args.fps)

    print(f"\nVisualization complete! Output: {output_dir}")


if __name__ == "__main__":
    main()
