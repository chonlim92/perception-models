# [IMPLEMENTED BY CLAUDE - was missing]
"""
Visualization script for radar occupancy prediction results.
Loads prediction results and displays BEV occupancy maps with color coding.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Semantic class colors mapping class indices to (name, color)
SEMANTIC_COLORS = {
    0: ("background", "#D3D3D3"),       # light gray
    1: ("vehicle", "#1E90FF"),          # blue
    2: ("pedestrian", "#DC143C"),       # red
    3: ("cyclist", "#FF8C00"),          # orange
    4: ("static_object", "#228B22"),    # green
}


def load_results(results_path):
    """Load prediction results from a .npz file.

    Args:
        results_path: Path to the .npz file containing predictions.

    Returns:
        Dictionary with keys:
            - 'occupancy_pred': Predicted occupancy map (H, W)
            - 'occupancy_gt': Ground truth occupancy map (H, W)
            - 'semantic_pred': Predicted semantic map (H, W) [optional]
            - 'semantic_gt': Ground truth semantic map (H, W) [optional]
            - 'radar_points': Radar point locations (N, 2) [optional]
    """
    data = np.load(results_path, allow_pickle=True)

    results = {}
    results["occupancy_pred"] = data["occupancy_pred"]
    results["occupancy_gt"] = data["occupancy_gt"]

    if "semantic_pred" in data:
        results["semantic_pred"] = data["semantic_pred"]
    else:
        results["semantic_pred"] = None

    if "semantic_gt" in data:
        results["semantic_gt"] = data["semantic_gt"]
    else:
        results["semantic_gt"] = None

    if "radar_points" in data:
        results["radar_points"] = data["radar_points"]
    else:
        results["radar_points"] = None

    return results


def create_occupancy_colormap():
    """Create a custom colormap for occupancy visualization.

    Color scheme:
        - 0 (free): green
        - 1 (occupied): red
        - 2 (unknown): gray

    Returns:
        Tuple of (colormap, norm) for use with matplotlib imshow.
    """
    colors = ["#2ECC40", "#FF4136", "#AAAAAA"]  # green, red, gray
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def visualize_occupancy_bev(occupancy_pred, occupancy_gt=None, config=None, ax=None):
    """Plot occupancy map in BEV with custom colormap.

    Args:
        occupancy_pred: Predicted occupancy map (H, W) with values 0=free, 1=occupied, 2=unknown.
        occupancy_gt: Optional ground truth occupancy map (H, W).
        config: Optional configuration dict with 'x_range', 'y_range', 'resolution' keys.
        ax: Optional matplotlib axes. If None, creates a new figure.

    Returns:
        The matplotlib figure (or None if ax was provided).
    """
    cmap, norm = create_occupancy_colormap()

    # Determine extent from config
    if config is not None:
        x_range = config.get("x_range", (-50.0, 50.0))
        y_range = config.get("y_range", (-50.0, 50.0))
        extent = [y_range[0], y_range[1], x_range[0], x_range[1]]
    else:
        h, w = occupancy_pred.shape
        extent = [0, w, 0, h]

    show_comparison = occupancy_gt is not None
    fig = None

    if ax is None:
        if show_comparison:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax_pred = axes[0]
            ax_gt = axes[1]
        else:
            fig, ax_pred = plt.subplots(1, 1, figsize=(7, 6))
            ax_gt = None
    else:
        ax_pred = ax
        ax_gt = None
        show_comparison = False

    # Plot predicted occupancy
    ax_pred.imshow(
        occupancy_pred,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax_pred.set_title("Predicted Occupancy (BEV)")
    ax_pred.set_xlabel("Y (m)")
    ax_pred.set_ylabel("X (m)")

    # Add legend
    legend_patches = [
        mpatches.Patch(color="#2ECC40", label="Free"),
        mpatches.Patch(color="#FF4136", label="Occupied"),
        mpatches.Patch(color="#AAAAAA", label="Unknown"),
    ]
    ax_pred.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # Plot GT if provided and we have side-by-side axes
    if show_comparison and ax_gt is not None:
        ax_gt.imshow(
            occupancy_gt,
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        ax_gt.set_title("Ground Truth Occupancy (BEV)")
        ax_gt.set_xlabel("Y (m)")
        ax_gt.set_ylabel("X (m)")
        ax_gt.legend(handles=legend_patches, loc="upper right", fontsize=8)

    if fig is not None:
        plt.tight_layout()

    return fig


def visualize_semantic_bev(semantic_pred, semantic_gt=None, class_names=None, ax=None):
    """Plot semantic segmentation map in BEV with class colors.

    Args:
        semantic_pred: Predicted semantic map (H, W) with integer class indices.
        semantic_gt: Optional ground truth semantic map (H, W).
        class_names: Optional dict mapping class index to name. Defaults to SEMANTIC_COLORS.
        ax: Optional matplotlib axes. If None, creates a new figure.

    Returns:
        The matplotlib figure (or None if ax was provided).
    """
    if class_names is None:
        class_names = {k: v[0] for k, v in SEMANTIC_COLORS.items()}

    # Build color array from semantic map
    num_classes = max(SEMANTIC_COLORS.keys()) + 1
    color_list = []
    for i in range(num_classes):
        if i in SEMANTIC_COLORS:
            color_list.append(mcolors.to_rgb(SEMANTIC_COLORS[i][1]))
        else:
            color_list.append((0.5, 0.5, 0.5))

    cmap = mcolors.ListedColormap(color_list)
    bounds = np.arange(-0.5, num_classes + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    show_comparison = semantic_gt is not None
    fig = None

    if ax is None:
        if show_comparison:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax_pred = axes[0]
            ax_gt = axes[1]
        else:
            fig, ax_pred = plt.subplots(1, 1, figsize=(7, 6))
            ax_gt = None
    else:
        ax_pred = ax
        ax_gt = None
        show_comparison = False

    # Plot predicted semantic map
    ax_pred.imshow(semantic_pred, cmap=cmap, norm=norm, origin="lower", aspect="equal")
    ax_pred.set_title("Predicted Semantics (BEV)")
    ax_pred.set_xlabel("Y (m)")
    ax_pred.set_ylabel("X (m)")

    # Add legend with class names
    legend_patches = []
    unique_classes = np.unique(semantic_pred)
    for cls_idx in sorted(unique_classes):
        cls_idx = int(cls_idx)
        if cls_idx in SEMANTIC_COLORS:
            name, color = SEMANTIC_COLORS[cls_idx]
            legend_patches.append(mpatches.Patch(color=color, label=name))
        else:
            legend_patches.append(
                mpatches.Patch(color="gray", label=f"class_{cls_idx}")
            )
    ax_pred.legend(handles=legend_patches, loc="upper right", fontsize=8)

    # Plot GT if provided
    if show_comparison and ax_gt is not None:
        ax_gt.imshow(semantic_gt, cmap=cmap, norm=norm, origin="lower", aspect="equal")
        ax_gt.set_title("Ground Truth Semantics (BEV)")
        ax_gt.set_xlabel("Y (m)")
        ax_gt.set_ylabel("X (m)")

        legend_patches_gt = []
        unique_gt = np.unique(semantic_gt)
        for cls_idx in sorted(unique_gt):
            cls_idx = int(cls_idx)
            if cls_idx in SEMANTIC_COLORS:
                name, color = SEMANTIC_COLORS[cls_idx]
                legend_patches_gt.append(mpatches.Patch(color=color, label=name))
            else:
                legend_patches_gt.append(
                    mpatches.Patch(color="gray", label=f"class_{cls_idx}")
                )
        ax_gt.legend(handles=legend_patches_gt, loc="upper right", fontsize=8)

    if fig is not None:
        plt.tight_layout()

    return fig


def visualize_radar_points_overlay(occupancy_map, radar_points, config=None, ax=None):
    """Show occupancy map with radar point locations overlaid as scatter.

    Args:
        occupancy_map: Occupancy map (H, W) with values 0=free, 1=occupied, 2=unknown.
        radar_points: Radar point locations (N, 2) in BEV coordinates.
        config: Optional configuration dict with 'x_range', 'y_range' keys.
        ax: Optional matplotlib axes. If None, creates a new figure.

    Returns:
        The matplotlib figure (or None if ax was provided).
    """
    cmap, norm = create_occupancy_colormap()

    if config is not None:
        x_range = config.get("x_range", (-50.0, 50.0))
        y_range = config.get("y_range", (-50.0, 50.0))
        extent = [y_range[0], y_range[1], x_range[0], x_range[1]]
    else:
        h, w = occupancy_map.shape
        extent = [0, w, 0, h]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Plot occupancy map as background
    ax.imshow(
        occupancy_map,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=extent,
        aspect="equal",
        alpha=0.7,
    )

    # Overlay radar points
    if radar_points is not None and len(radar_points) > 0:
        ax.scatter(
            radar_points[:, 1],  # y-coordinate
            radar_points[:, 0],  # x-coordinate
            c="cyan",
            s=10,
            marker="o",
            edgecolors="black",
            linewidths=0.3,
            label="Radar Points",
            zorder=5,
        )

    ax.set_title("Occupancy with Radar Points Overlay")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.legend(loc="upper right", fontsize=8)

    if fig is not None:
        plt.tight_layout()

    return fig


def visualize_comparison(results, config, output_path=None):
    """Create a multi-panel comparison figure.

    Creates a 2x3 or 2x2 figure with:
        - Panel 1: Predicted occupancy
        - Panel 2: GT occupancy
        - Panel 3: Predicted semantics (if available)
        - Panel 4: GT semantics (if available)
        - Panel 5: Error map (FP=red, FN=blue, TP=green)

    Args:
        results: Dictionary from load_results().
        config: Configuration dict with spatial extents and resolution.
        output_path: Optional path to save figure. If None, displays interactively.

    Returns:
        The matplotlib figure.
    """
    occupancy_pred = results["occupancy_pred"]
    occupancy_gt = results["occupancy_gt"]
    semantic_pred = results.get("semantic_pred")
    semantic_gt = results.get("semantic_gt")

    has_semantics = semantic_pred is not None and semantic_gt is not None

    if has_semantics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Setup colormap for occupancy
    occ_cmap, occ_norm = create_occupancy_colormap()

    if config is not None:
        x_range = config.get("x_range", (-50.0, 50.0))
        y_range = config.get("y_range", (-50.0, 50.0))
        extent = [y_range[0], y_range[1], x_range[0], x_range[1]]
    else:
        h, w = occupancy_pred.shape
        extent = [0, w, 0, h]

    # Panel 1: Predicted occupancy
    ax1 = axes[0, 0]
    ax1.imshow(
        occupancy_pred,
        cmap=occ_cmap,
        norm=occ_norm,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax1.set_title("Predicted Occupancy")
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("X (m)")
    occ_patches = [
        mpatches.Patch(color="#2ECC40", label="Free"),
        mpatches.Patch(color="#FF4136", label="Occupied"),
        mpatches.Patch(color="#AAAAAA", label="Unknown"),
    ]
    ax1.legend(handles=occ_patches, loc="upper right", fontsize=7)

    # Panel 2: GT occupancy
    ax2 = axes[0, 1]
    ax2.imshow(
        occupancy_gt,
        cmap=occ_cmap,
        norm=occ_norm,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax2.set_title("Ground Truth Occupancy")
    ax2.set_xlabel("Y (m)")
    ax2.set_ylabel("X (m)")
    ax2.legend(handles=occ_patches, loc="upper right", fontsize=7)

    # Panel 5: Error map
    # Compute error map: TP=1, FP=2, FN=3, TN=0
    # For binary: occupied=1 is positive
    pred_binary = (occupancy_pred == 1).astype(np.int32)
    gt_binary = (occupancy_gt == 1).astype(np.int32)

    tp = (pred_binary == 1) & (gt_binary == 1)
    fp = (pred_binary == 1) & (gt_binary == 0)
    fn = (pred_binary == 0) & (gt_binary == 1)
    tn = (pred_binary == 0) & (gt_binary == 0)

    error_map = np.zeros_like(occupancy_pred, dtype=np.int32)
    error_map[tn] = 0  # true negative -> white/light
    error_map[tp] = 1  # true positive -> green
    error_map[fp] = 2  # false positive -> red
    error_map[fn] = 3  # false negative -> blue

    error_colors = ["#F5F5F5", "#2ECC40", "#FF4136", "#0074D9"]  # TN, TP, FP, FN
    error_cmap = mcolors.ListedColormap(error_colors)
    error_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    error_norm = mcolors.BoundaryNorm(error_bounds, error_cmap.N)

    if has_semantics:
        ax_err = axes[1, 2]
    else:
        ax_err = axes[1, 1]

    ax_err.imshow(
        error_map,
        cmap=error_cmap,
        norm=error_norm,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax_err.set_title("Error Map")
    ax_err.set_xlabel("Y (m)")
    ax_err.set_ylabel("X (m)")
    err_patches = [
        mpatches.Patch(color="#2ECC40", label="TP (correct occupied)"),
        mpatches.Patch(color="#FF4136", label="FP (predicted occ, actually free)"),
        mpatches.Patch(color="#0074D9", label="FN (missed occupied)"),
        mpatches.Patch(color="#F5F5F5", label="TN (correct free)"),
    ]
    ax_err.legend(handles=err_patches, loc="upper right", fontsize=6)

    # Compute IoU
    intersection = tp.sum()
    union = tp.sum() + fp.sum() + fn.sum()
    iou = intersection / max(union, 1)

    # Panels 3 & 4: Semantics (if available)
    if has_semantics:
        num_classes = max(SEMANTIC_COLORS.keys()) + 1
        sem_color_list = []
        for i in range(num_classes):
            if i in SEMANTIC_COLORS:
                sem_color_list.append(mcolors.to_rgb(SEMANTIC_COLORS[i][1]))
            else:
                sem_color_list.append((0.5, 0.5, 0.5))
        sem_cmap = mcolors.ListedColormap(sem_color_list)
        sem_bounds = np.arange(-0.5, num_classes + 0.5, 1.0)
        sem_norm = mcolors.BoundaryNorm(sem_bounds, sem_cmap.N)

        # Panel 3: Predicted semantics
        ax3 = axes[0, 2]
        ax3.imshow(
            semantic_pred,
            cmap=sem_cmap,
            norm=sem_norm,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        ax3.set_title("Predicted Semantics")
        ax3.set_xlabel("Y (m)")
        ax3.set_ylabel("X (m)")
        sem_patches = [
            mpatches.Patch(color=SEMANTIC_COLORS[k][1], label=SEMANTIC_COLORS[k][0])
            for k in sorted(SEMANTIC_COLORS.keys())
        ]
        ax3.legend(handles=sem_patches, loc="upper right", fontsize=6)

        # Panel 4: GT semantics
        ax4 = axes[1, 0]
        ax4.imshow(
            semantic_gt,
            cmap=sem_cmap,
            norm=sem_norm,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        ax4.set_title("Ground Truth Semantics")
        ax4.set_xlabel("Y (m)")
        ax4.set_ylabel("X (m)")
        ax4.legend(handles=sem_patches, loc="upper right", fontsize=6)

        # Panel in row 1, col 1: radar overlay or leave for summary
        ax5 = axes[1, 1]
        if results.get("radar_points") is not None:
            ax5.imshow(
                occupancy_pred,
                cmap=occ_cmap,
                norm=occ_norm,
                origin="lower",
                extent=extent,
                aspect="equal",
                alpha=0.7,
            )
            radar_pts = results["radar_points"]
            ax5.scatter(
                radar_pts[:, 1],
                radar_pts[:, 0],
                c="cyan",
                s=8,
                marker="o",
                edgecolors="black",
                linewidths=0.2,
                label="Radar Points",
                zorder=5,
            )
            ax5.set_title("Occupancy + Radar Points")
            ax5.set_xlabel("Y (m)")
            ax5.set_ylabel("X (m)")
            ax5.legend(loc="upper right", fontsize=7)
        else:
            ax5.axis("off")
            ax5.text(
                0.5,
                0.5,
                f"Occupancy IoU: {iou:.4f}\n"
                f"TP: {tp.sum()}, FP: {fp.sum()}, FN: {fn.sum()}",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax5.transAxes,
            )
    else:
        # 2x2 layout: bottom-left for radar overlay or metrics
        ax_bl = axes[1, 0]
        if results.get("radar_points") is not None:
            ax_bl.imshow(
                occupancy_pred,
                cmap=occ_cmap,
                norm=occ_norm,
                origin="lower",
                extent=extent,
                aspect="equal",
                alpha=0.7,
            )
            radar_pts = results["radar_points"]
            ax_bl.scatter(
                radar_pts[:, 1],
                radar_pts[:, 0],
                c="cyan",
                s=8,
                marker="o",
                edgecolors="black",
                linewidths=0.2,
                label="Radar Points",
                zorder=5,
            )
            ax_bl.set_title("Occupancy + Radar Points")
            ax_bl.set_xlabel("Y (m)")
            ax_bl.set_ylabel("X (m)")
            ax_bl.legend(loc="upper right", fontsize=7)
        else:
            ax_bl.axis("off")
            ax_bl.text(
                0.5,
                0.5,
                f"Occupancy IoU: {iou:.4f}\n"
                f"TP: {tp.sum()}, FP: {fp.sum()}, FN: {fn.sum()}",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax_bl.transAxes,
            )

    fig.suptitle(f"Radar Occupancy Prediction | IoU: {iou:.4f}", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison figure to: {output_path}")

    return fig


def batch_visualize(results_dir, output_dir, config, max_samples=20):
    """Process multiple result files and save individual visualizations.

    Args:
        results_dir: Directory containing .npz result files.
        output_dir: Directory to save output visualizations.
        config: Configuration dict with spatial extents and resolution.
        max_samples: Maximum number of samples to visualize.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all .npz files
    npz_files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(".npz")]
    )

    if len(npz_files) == 0:
        print(f"No .npz files found in: {results_dir}")
        return

    num_to_process = min(len(npz_files), max_samples)
    print(f"Processing {num_to_process}/{len(npz_files)} result files...")

    montage_figs = []

    for idx, npz_file in enumerate(npz_files[:num_to_process]):
        results_path = os.path.join(results_dir, npz_file)
        sample_name = os.path.splitext(npz_file)[0]

        print(f"  [{idx + 1}/{num_to_process}] Visualizing: {sample_name}")

        try:
            results = load_results(results_path)
        except Exception as e:
            print(f"    Error loading {npz_file}: {e}")
            continue

        output_path = os.path.join(output_dir, f"{sample_name}_comparison.png")
        fig = visualize_comparison(results, config, output_path=output_path)
        montage_figs.append((sample_name, fig))
        plt.close(fig)

    # Create summary montage (grid of thumbnails)
    if len(montage_figs) > 0:
        _create_summary_montage(montage_figs, results_dir, output_dir, config, num_to_process, npz_files)

    print(f"Batch visualization complete. Output saved to: {output_dir}")


def _create_summary_montage(montage_figs, results_dir, output_dir, config, num_to_process, npz_files):
    """Create a summary montage of all processed samples.

    Args:
        montage_figs: List of (sample_name, figure) tuples.
        results_dir: Directory containing .npz result files.
        output_dir: Directory to save the montage.
        config: Configuration dict.
        num_to_process: Number of samples processed.
        npz_files: List of .npz filenames.
    """
    ncols = min(4, num_to_process)
    nrows = (num_to_process + ncols - 1) // ncols

    fig_montage, axes_montage = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows)
    )

    if nrows == 1 and ncols == 1:
        axes_montage = np.array([[axes_montage]])
    elif nrows == 1:
        axes_montage = axes_montage[np.newaxis, :]
    elif ncols == 1:
        axes_montage = axes_montage[:, np.newaxis]

    occ_cmap, occ_norm = create_occupancy_colormap()

    for idx in range(nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        ax = axes_montage[row, col]

        if idx < num_to_process:
            npz_file = npz_files[idx]
            sample_name = os.path.splitext(npz_file)[0]
            try:
                results = load_results(os.path.join(results_dir, npz_file))
                ax.imshow(
                    results["occupancy_pred"],
                    cmap=occ_cmap,
                    norm=occ_norm,
                    origin="lower",
                    aspect="equal",
                )
                ax.set_title(sample_name, fontsize=8)
            except Exception:
                ax.axis("off")
                ax.set_title(f"{sample_name} (error)", fontsize=8)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    fig_montage.suptitle("Summary Montage - Predicted Occupancy", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    montage_path = os.path.join(output_dir, "summary_montage.png")
    fig_montage.savefig(montage_path, dpi=100, bbox_inches="tight")
    plt.close(fig_montage)
    print(f"  Saved summary montage to: {montage_path}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Visualize radar occupancy prediction results in BEV."
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results file (.npz) or directory containing .npz files.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file with spatial extents and resolution.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./visualization_output",
        help="Output directory for saved visualizations.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Maximum number of samples to visualize (default: 20).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualizations interactively instead of only saving.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for saved images (default: 150).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png).",
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config is not None:
        try:
            import yaml

            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            print(f"Loaded config from: {args.config}")
        except ImportError:
            print("Warning: PyYAML not installed. Using default config.")
            config = {
                "x_range": (-50.0, 50.0),
                "y_range": (-50.0, 50.0),
                "resolution": 0.2,
            }
        except Exception as e:
            print(f"Warning: Could not load config ({e}). Using defaults.")
            config = {
                "x_range": (-50.0, 50.0),
                "y_range": (-50.0, 50.0),
                "resolution": 0.2,
            }
    else:
        config = {
            "x_range": (-50.0, 50.0),
            "y_range": (-50.0, 50.0),
            "resolution": 0.2,
        }

    # Update matplotlib settings
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["savefig.format"] = args.format

    results_path = args.results

    if os.path.isdir(results_path):
        # Batch mode: process directory of .npz files
        print(f"Batch mode: processing directory {results_path}")
        batch_visualize(
            results_dir=results_path,
            output_dir=args.output,
            config=config,
            max_samples=args.max_samples,
        )
    elif os.path.isfile(results_path) and results_path.endswith(".npz"):
        # Single file mode
        print(f"Single file mode: {results_path}")
        os.makedirs(args.output, exist_ok=True)

        results = load_results(results_path)

        sample_name = os.path.splitext(os.path.basename(results_path))[0]
        output_file = os.path.join(
            args.output, f"{sample_name}_comparison.{args.format}"
        )

        fig = visualize_comparison(results, config, output_path=output_file)

        if args.show:
            plt.show()
        else:
            plt.close(fig)

        print(f"Done. Output saved to: {output_file}")
    else:
        print(f"Error: Invalid results path: {results_path}")
        print("Provide a .npz file or directory containing .npz files.")
        return

    if args.show and os.path.isdir(results_path):
        print("Note: --show flag with batch mode only shows the last figure.")
        plt.show()


if __name__ == "__main__":
    main()
