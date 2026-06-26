"""
MapTR Results Visualization Script

Compares predicted vectorized map elements against ground truth by rendering
them on a BEV (Bird's Eye View) canvas. Supports overlay mode, side-by-side
comparison, per-element Chamfer distance annotation, and sequential frame export.

Usage:
    python scripts/visualize_results.py \
        --pred_file results/predictions.pkl \
        --gt_file data/processed/maptr_val.pkl \
        --output_dir visualizations/ \
        --sample_indices 0 1 2 3 4

    python scripts/visualize_results.py \
        --pred_file results/predictions.pkl \
        --gt_file data/processed/maptr_val.pkl \
        --output_dir visualizations/ \
        --mode side_by_side \
        --export_video
"""

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.collections import LineCollection
    import matplotlib.animation as animation
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

CATEGORY_NAMES = ["ped_crossing", "divider", "boundary"]

CATEGORY_COLORS = {
    "ped_crossing": {"pred": "#4A90D9", "gt": "#A8D4FF"},  # Blue
    "divider": {"pred": "#E67E22", "gt": "#F5CBA7"},       # Orange
    "boundary": {"pred": "#27AE60", "gt": "#A9DFBF"},      # Green
}

CATEGORY_LABELS = {
    "ped_crossing": "Pedestrian Crossing",
    "divider": "Lane/Road Divider",
    "boundary": "Road Boundary",
}


# ============================================================================
# Chamfer Distance
# ============================================================================

def chamfer_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    """
    Compute Chamfer distance between two point sets.

    Args:
        pred_pts: (N, 2) predicted points
        gt_pts: (M, 2) ground truth points

    Returns:
        Chamfer distance (average of both directions)
    """
    # pred -> gt direction
    diff_p2g = pred_pts[:, None, :] - gt_pts[None, :, :]  # (N, M, 2)
    dist_p2g = np.linalg.norm(diff_p2g, axis=-1)  # (N, M)
    min_p2g = np.min(dist_p2g, axis=1)  # (N,)

    # gt -> pred direction
    min_g2p = np.min(dist_p2g, axis=0)  # (M,)

    # Symmetric Chamfer
    chamfer = 0.5 * (np.mean(min_p2g) + np.mean(min_g2p))
    return float(chamfer)


# ============================================================================
# Visualization
# ============================================================================

class MapVisualizer:
    """Visualization engine for vectorized map predictions and ground truth."""

    def __init__(self, perception_range: Tuple[float, float, float, float] = (-30, -15, 30, 15),
                 canvas_size: Tuple[int, int] = (1200, 800),
                 dpi: int = 150):
        """
        Args:
            perception_range: (x_min, y_min, x_max, y_max) in meters
            canvas_size: (width, height) in pixels
            dpi: Output DPI
        """
        self.x_min, self.y_min, self.x_max, self.y_max = perception_range
        self.canvas_width, self.canvas_height = canvas_size
        self.dpi = dpi

    def denormalize_points(self, points: np.ndarray) -> np.ndarray:
        """Convert normalized [0,1] coordinates back to metric BEV coordinates."""
        pts = points.copy()
        pts[:, 0] = pts[:, 0] * (self.x_max - self.x_min) + self.x_min
        pts[:, 1] = pts[:, 1] * (self.y_max - self.y_min) + self.y_min
        return pts

    def _setup_bev_axes(self, ax: plt.Axes, title: str = ""):
        """Configure axes for BEV visualization."""
        ax.set_xlim(self.y_min, self.y_max)
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.grid(True, alpha=0.2, color="white", linewidth=0.5)
        ax.set_xlabel("Lateral (m)", fontsize=9, color="white")
        ax.set_ylabel("Longitudinal (m)", fontsize=9, color="white")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.5)
        if title:
            ax.set_title(title, fontsize=11, color="white", pad=10)

        # Draw ego vehicle
        ego_rect = plt.Rectangle((-1, -2), 2, 4, linewidth=1.5,
                                  edgecolor="white", facecolor="#333333", zorder=10)
        ax.add_patch(ego_rect)
        ax.plot(0, 2.5, "^", color="white", markersize=6, zorder=11)

    def _draw_elements(self, ax: plt.Axes, elements: List[Dict],
                       is_gt: bool = False, show_points: bool = True,
                       show_direction: bool = True, alpha: float = 1.0):
        """Draw map elements on axes."""
        for elem in elements:
            category = elem.get("category", CATEGORY_NAMES[elem.get("label", 0)])
            points = elem["points"]

            if isinstance(points, list):
                points = np.array(points)

            # Check if points are normalized (all in [0,1])
            if points.max() <= 1.0 and points.min() >= 0.0:
                points = self.denormalize_points(points)

            # Select color
            color_key = "gt" if is_gt else "pred"
            color = CATEGORY_COLORS.get(category, {"pred": "white", "gt": "gray"})[color_key]

            # Line style
            linestyle = "--" if is_gt else "-"
            linewidth = 1.5 if is_gt else 2.5

            # Draw polyline (note: x=lateral, y=longitudinal in BEV view)
            ax.plot(points[:, 1], points[:, 0],
                    color=color, linewidth=linewidth,
                    linestyle=linestyle, alpha=alpha, zorder=5)

            # Draw points
            if show_points:
                marker_size = 2 if is_gt else 3
                ax.scatter(points[:, 1], points[:, 0],
                           c=color, s=marker_size, zorder=6, alpha=alpha)

            # Draw direction arrow
            if show_direction and len(points) >= 2:
                mid_idx = len(points) // 2
                dx = points[mid_idx, 1] - points[mid_idx - 1, 1]
                dy = points[mid_idx, 0] - points[mid_idx - 1, 0]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0.1:
                    ax.annotate("", xy=(points[mid_idx, 1], points[mid_idx, 0]),
                                xytext=(points[mid_idx - 1, 1], points[mid_idx - 1, 0]),
                                arrowprops=dict(arrowstyle="->", color=color,
                                                lw=1.5 * alpha),
                                zorder=7)

            # Show confidence if available
            confidence = elem.get("confidence", None)
            if confidence is not None and not is_gt:
                center = points.mean(axis=0)
                ax.text(center[1], center[0], f"{confidence:.2f}",
                        fontsize=6, color=color, ha="center", va="bottom",
                        zorder=8)

    def visualize_overlay(self, predictions: List[Dict], ground_truth: List[Dict],
                          title: str = "", save_path: Optional[str] = None,
                          show_chamfer: bool = True) -> plt.Figure:
        """
        Create overlay visualization with predictions and GT on same canvas.

        Args:
            predictions: List of predicted elements
            ground_truth: List of GT elements
            title: Plot title
            save_path: Path to save figure
            show_chamfer: Whether to annotate Chamfer distances
        """
        fig, ax = plt.subplots(1, 1, figsize=(
            self.canvas_width / self.dpi,
            self.canvas_height / self.dpi
        ), dpi=self.dpi)
        fig.patch.set_facecolor("#0d1117")

        self._setup_bev_axes(ax, title or "Predictions (solid) vs GT (dashed)")

        # Draw GT first (underneath)
        self._draw_elements(ax, ground_truth, is_gt=True, alpha=0.7)

        # Draw predictions on top
        self._draw_elements(ax, predictions, is_gt=False, alpha=0.9)

        # Annotate Chamfer distances for matched pairs
        if show_chamfer and predictions and ground_truth:
            self._annotate_chamfer(ax, predictions, ground_truth)

        # Legend
        legend_elements = []
        for cat in CATEGORY_NAMES:
            pred_color = CATEGORY_COLORS[cat]["pred"]
            gt_color = CATEGORY_COLORS[cat]["gt"]
            legend_elements.append(plt.Line2D([0], [0], color=pred_color,
                                              linewidth=2, label=f"{CATEGORY_LABELS[cat]} (pred)"))
            legend_elements.append(plt.Line2D([0], [0], color=gt_color,
                                              linewidth=1.5, linestyle="--",
                                              label=f"{CATEGORY_LABELS[cat]} (GT)"))
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7,
                  facecolor="#2d2d44", edgecolor="white", labelcolor="white")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Saved: {save_path}")

        return fig

    def visualize_side_by_side(self, predictions: List[Dict], ground_truth: List[Dict],
                                title: str = "", save_path: Optional[str] = None) -> plt.Figure:
        """Create side-by-side comparison of predictions and GT."""
        fig, (ax_pred, ax_gt) = plt.subplots(1, 2, figsize=(
            self.canvas_width * 2 / self.dpi,
            self.canvas_height / self.dpi
        ), dpi=self.dpi)
        fig.patch.set_facecolor("#0d1117")

        self._setup_bev_axes(ax_pred, "Predictions")
        self._setup_bev_axes(ax_gt, "Ground Truth")

        self._draw_elements(ax_pred, predictions, is_gt=False)
        self._draw_elements(ax_gt, ground_truth, is_gt=True)

        if title:
            fig.suptitle(title, fontsize=13, color="white", y=0.98)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Saved: {save_path}")

        return fig

    def _annotate_chamfer(self, ax: plt.Axes, predictions: List[Dict],
                          ground_truth: List[Dict]):
        """Annotate Chamfer distances between nearest pred-GT pairs."""
        # Simple greedy matching by category and proximity
        for pred_elem in predictions:
            pred_cat = pred_elem.get("category", CATEGORY_NAMES[pred_elem.get("label", 0)])
            pred_pts = np.array(pred_elem["points"])
            if pred_pts.max() <= 1.0 and pred_pts.min() >= 0.0:
                pred_pts = self.denormalize_points(pred_pts)

            # Find best matching GT
            best_dist = float("inf")
            for gt_elem in ground_truth:
                gt_cat = gt_elem.get("category", CATEGORY_NAMES[gt_elem.get("label", 0)])
                if gt_cat != pred_cat:
                    continue
                gt_pts = np.array(gt_elem["points"])
                if gt_pts.max() <= 1.0 and gt_pts.min() >= 0.0:
                    gt_pts = self.denormalize_points(gt_pts)

                dist = chamfer_distance(pred_pts, gt_pts)
                best_dist = min(best_dist, dist)

            if best_dist < float("inf"):
                center = pred_pts.mean(axis=0)
                color = "#ff6b6b" if best_dist > 1.5 else "#ffd93d" if best_dist > 0.5 else "#6bff6b"
                ax.text(center[1], center[0] + 1.0, f"CD:{best_dist:.2f}m",
                        fontsize=5, color=color, ha="center", va="bottom",
                        zorder=9, fontweight="bold")

    def create_animation(self, all_predictions: List[List[Dict]],
                         all_ground_truths: List[List[Dict]],
                         output_path: str, fps: int = 5):
        """
        Create an animation from sequential frames.

        Args:
            all_predictions: List of predictions per frame
            all_ground_truths: List of GT per frame
            output_path: Path to save video/gif
            fps: Frames per second
        """
        num_frames = min(len(all_predictions), len(all_ground_truths))
        if num_frames == 0:
            print("  No frames to animate")
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=100)
        fig.patch.set_facecolor("#0d1117")

        def update(frame_idx):
            ax.clear()
            self._setup_bev_axes(ax, f"Frame {frame_idx + 1}/{num_frames}")
            self._draw_elements(ax, all_ground_truths[frame_idx], is_gt=True, alpha=0.6)
            self._draw_elements(ax, all_predictions[frame_idx], is_gt=False, alpha=0.9)
            return ax,

        anim = animation.FuncAnimation(fig, update, frames=num_frames,
                                        interval=1000 // fps, blit=False)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if output_path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
        else:
            try:
                writer = animation.FFMpegWriter(fps=fps, codec="libx264")
                anim.save(output_path, writer=writer)
            except Exception:
                # Fallback to gif
                gif_path = output_path.rsplit(".", 1)[0] + ".gif"
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer)
                print(f"  FFmpeg unavailable, saved as GIF: {gif_path}")
                return

        print(f"  Saved animation: {output_path}")
        plt.close(fig)


# ============================================================================
# Data Loading
# ============================================================================

def load_data(filepath: str) -> List[Dict]:
    """Load predictions or ground truth from file."""
    if filepath.endswith(".pkl") or filepath.endswith(".pickle"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    # Normalize data format
    if isinstance(data, dict):
        # Could be a dict with sample tokens as keys
        samples = list(data.values())
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

    return samples


def extract_elements(sample: Dict) -> List[Dict]:
    """Extract map elements from a sample record."""
    # Handle different data formats
    if "map_elements" in sample:
        return sample["map_elements"]
    elif "predictions" in sample:
        return sample["predictions"]
    elif "elements" in sample:
        return sample["elements"]
    elif "results" in sample:
        return sample["results"]
    else:
        # Assume the sample itself is a list of elements
        if isinstance(sample, list):
            return sample
        return []


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MapTR predictions vs ground truth"
    )
    parser.add_argument(
        "--pred_file", type=str, required=True,
        help="Path to predictions file (pkl or json)"
    )
    parser.add_argument(
        "--gt_file", type=str, required=True,
        help="Path to ground truth file (pkl or json)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="visualizations",
        help="Output directory for visualization images"
    )
    parser.add_argument(
        "--sample_indices", type=int, nargs="+", default=None,
        help="Specific sample indices to visualize (default: first 10)"
    )
    parser.add_argument(
        "--mode", type=str, default="overlay",
        choices=["overlay", "side_by_side", "both"],
        help="Visualization mode"
    )
    parser.add_argument(
        "--export_video", action="store_true",
        help="Export sequential frames as video/gif"
    )
    parser.add_argument(
        "--video_fps", type=int, default=5,
        help="FPS for video export"
    )
    parser.add_argument(
        "--perception_range", type=float, nargs=4,
        default=[-30.0, -15.0, 30.0, 15.0],
        help="Perception range: x_min y_min x_max y_max (meters)"
    )
    parser.add_argument(
        "--show_chamfer", action="store_true", default=True,
        help="Annotate Chamfer distances on overlay"
    )
    parser.add_argument(
        "--no_chamfer", action="store_true",
        help="Disable Chamfer distance annotations"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MapTR Results Visualization")
    print("=" * 60)

    # Load data
    print(f"\nLoading predictions: {args.pred_file}")
    pred_samples = load_data(args.pred_file)
    print(f"  Loaded {len(pred_samples)} prediction samples")

    print(f"Loading ground truth: {args.gt_file}")
    gt_samples = load_data(args.gt_file)
    print(f"  Loaded {len(gt_samples)} ground truth samples")

    # Determine samples to visualize
    num_samples = min(len(pred_samples), len(gt_samples))
    if args.sample_indices is not None:
        indices = [i for i in args.sample_indices if i < num_samples]
    else:
        indices = list(range(min(10, num_samples)))

    print(f"\nVisualizing {len(indices)} samples")

    # Create visualizer
    perception_range = tuple(args.perception_range)
    visualizer = MapVisualizer(
        perception_range=perception_range,
        dpi=args.dpi,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    show_chamfer = args.show_chamfer and not args.no_chamfer

    # Visualize each sample
    all_pred_elements = []
    all_gt_elements = []

    for i, idx in enumerate(indices):
        pred_elements = extract_elements(pred_samples[idx])
        gt_elements = extract_elements(gt_samples[idx])

        all_pred_elements.append(pred_elements)
        all_gt_elements.append(gt_elements)

        sample_token = pred_samples[idx].get("token", f"sample_{idx:04d}")
        title = f"Sample {idx} ({sample_token[:8]}...)" if len(sample_token) > 8 else f"Sample {idx}"

        print(f"\n  [{i + 1}/{len(indices)}] Sample {idx}: "
              f"{len(pred_elements)} predictions, {len(gt_elements)} GT elements")

        # Overlay mode
        if args.mode in ("overlay", "both"):
            save_path = os.path.join(args.output_dir, f"overlay_{idx:04d}.png")
            fig = visualizer.visualize_overlay(
                pred_elements, gt_elements,
                title=title, save_path=save_path,
                show_chamfer=show_chamfer
            )
            plt.close(fig)

        # Side-by-side mode
        if args.mode in ("side_by_side", "both"):
            save_path = os.path.join(args.output_dir, f"comparison_{idx:04d}.png")
            fig = visualizer.visualize_side_by_side(
                pred_elements, gt_elements,
                title=title, save_path=save_path
            )
            plt.close(fig)

        # Compute overall metrics for this sample
        if pred_elements and gt_elements:
            # Compute per-category Chamfer distances
            for cat in CATEGORY_NAMES:
                cat_preds = [e for e in pred_elements
                             if e.get("category", CATEGORY_NAMES[e.get("label", 0)]) == cat]
                cat_gts = [e for e in gt_elements
                           if e.get("category", CATEGORY_NAMES[e.get("label", 0)]) == cat]
                if cat_preds and cat_gts:
                    dists = []
                    for p in cat_preds:
                        pts_p = np.array(p["points"])
                        min_d = min(chamfer_distance(pts_p, np.array(g["points"]))
                                    for g in cat_gts)
                        dists.append(min_d)
                    avg_cd = np.mean(dists)
                    print(f"    {cat}: avg Chamfer = {avg_cd:.3f}m ({len(cat_preds)} pred, {len(cat_gts)} GT)")

    # Export video/animation
    if args.export_video and len(all_pred_elements) > 1:
        print("\n  Creating animation...")
        video_path = os.path.join(args.output_dir, "sequence.gif")
        visualizer.create_animation(
            all_pred_elements, all_gt_elements,
            output_path=video_path, fps=args.video_fps
        )

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_preds = sum(len(e) for e in all_pred_elements)
    total_gts = sum(len(e) for e in all_gt_elements)
    print(f"  Total predictions visualized: {total_preds}")
    print(f"  Total GT elements visualized: {total_gts}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Files generated: {len(indices)} {'x 2' if args.mode == 'both' else ''}")
    print("=" * 60)


if __name__ == "__main__":
    main()
