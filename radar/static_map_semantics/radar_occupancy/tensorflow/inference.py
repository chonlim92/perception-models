# [IMPLEMENTED BY CLAUDE - was missing]
"""
TensorFlow 2 single-sample inference script for Radar Occupancy models.

Loads a trained PillarOccNet / TemporalPillarOccNet checkpoint, preprocesses
a raw radar point cloud into pillar format, runs inference, and visualizes
the resulting occupancy (and optional semantic) BEV map.

Usage:
    python inference.py \
        --config ../configs/radar_occupancy_nuscenes.yaml \
        --checkpoint ./output/best_checkpoint \
        --input sample_radar.npy \
        --output output_bev.png \
        --threshold 0.5
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from model import PillarOccNet, TemporalPillarOccNet, build_model


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_radar_points(points, config):
    """
    Convert raw radar points to pillar representation for model input.

    Args:
        points: numpy array of shape (N, 6) with columns
                [x, y, z, rcs, vr_comp, timestamp].
        config: dictionary loaded from YAML config file.

    Returns:
        pillar_features: np.ndarray of shape (1, max_pillars, max_points_per_pillar, 9)
        pillar_indices: np.ndarray of shape (1, max_pillars, 2) — grid row/col for each pillar
        num_pillars: np.ndarray of shape (1,) — number of non-empty pillars
    """
    # Extract grid parameters
    grid_cfg = config["grid"]
    x_range = grid_cfg["x_range"]  # [x_min, x_max]
    y_range = grid_cfg["y_range"]  # [y_min, y_max]
    z_range = grid_cfg["z_range"]  # [z_min, z_max]
    cell_size = grid_cfg["cell_size"]
    grid_size = grid_cfg["grid_size"]  # [H, W]
    H, W = grid_size

    # Extract pillar parameters
    pillar_cfg = config.get("model", {}).get("pillar", {})
    max_points_per_pillar = pillar_cfg.get("max_points_per_pillar", 20)
    max_pillars = pillar_cfg.get("max_pillars", 12000)

    # -------------------------------------------------------------------------
    # Step (a): Filter points within grid bounds
    # -------------------------------------------------------------------------
    mask_x = (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1])
    mask_y = (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1])
    mask_z = (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1])
    valid_mask = mask_x & mask_y & mask_z
    points = points[valid_mask]

    if len(points) == 0:
        # Return empty pillar tensors if no valid points
        pillar_features = np.zeros(
            (1, max_pillars, max_points_per_pillar, 9), dtype=np.float32
        )
        pillar_indices = np.zeros((1, max_pillars, 2), dtype=np.int32)
        num_pillars_out = np.array([0], dtype=np.int32)
        return pillar_features, pillar_indices, num_pillars_out

    # -------------------------------------------------------------------------
    # Step (a): Compute grid cell indices for each point
    # -------------------------------------------------------------------------
    col_indices = np.floor((points[:, 0] - x_range[0]) / cell_size).astype(np.int32)
    row_indices = np.floor((points[:, 1] - y_range[0]) / cell_size).astype(np.int32)

    # Clamp to grid bounds (safety)
    col_indices = np.clip(col_indices, 0, W - 1)
    row_indices = np.clip(row_indices, 0, H - 1)

    # -------------------------------------------------------------------------
    # Step (b): Group points into pillars (unique grid cells)
    # -------------------------------------------------------------------------
    # Create a unique key for each grid cell
    cell_keys = row_indices * W + col_indices

    # Find unique pillars
    unique_keys, inverse_indices = np.unique(cell_keys, return_inverse=True)
    n_pillars = min(len(unique_keys), max_pillars)

    # If more pillars than max_pillars, keep those with most points
    if len(unique_keys) > max_pillars:
        # Count points per pillar and keep top-k
        pillar_counts = np.bincount(inverse_indices, minlength=len(unique_keys))
        top_k_indices = np.argsort(pillar_counts)[::-1][:max_pillars]
        selected_keys = unique_keys[top_k_indices]
        # Rebuild with only selected pillars
        selected_set = set(selected_keys.tolist())
        keep_mask = np.array([cell_keys[i] in selected_set for i in range(len(cell_keys))])
        points = points[keep_mask]
        col_indices = col_indices[keep_mask]
        row_indices = row_indices[keep_mask]
        cell_keys = cell_keys[keep_mask]
        unique_keys, inverse_indices = np.unique(cell_keys, return_inverse=True)
        n_pillars = len(unique_keys)

    # -------------------------------------------------------------------------
    # Step (c): Build pillar feature tensors with augmented features
    # -------------------------------------------------------------------------
    # Output arrays
    pillar_features = np.zeros(
        (max_pillars, max_points_per_pillar, 9), dtype=np.float32
    )
    pillar_indices_arr = np.zeros((max_pillars, 2), dtype=np.int32)

    for pillar_idx, key in enumerate(unique_keys):
        if pillar_idx >= max_pillars:
            break

        # Get points belonging to this pillar
        point_mask = (inverse_indices == pillar_idx)
        pillar_points = points[point_mask]

        # Limit to max_points_per_pillar (random sample if too many)
        n_pts = len(pillar_points)
        if n_pts > max_points_per_pillar:
            chosen = np.random.choice(n_pts, max_points_per_pillar, replace=False)
            pillar_points = pillar_points[chosen]
            n_pts = max_points_per_pillar

        # Compute pillar center (mean x, y, z of points in this pillar)
        center_x = np.mean(pillar_points[:, 0])
        center_y = np.mean(pillar_points[:, 1])
        center_z = np.mean(pillar_points[:, 2])

        # Augmented features per point:
        # [x, y, z, rcs, vr_comp, timestamp, x - x_center, y - y_center, z - z_center]
        augmented = np.zeros((n_pts, 9), dtype=np.float32)
        augmented[:, 0] = pillar_points[:, 0]  # x
        augmented[:, 1] = pillar_points[:, 1]  # y
        augmented[:, 2] = pillar_points[:, 2]  # z
        augmented[:, 3] = pillar_points[:, 3]  # rcs
        augmented[:, 4] = pillar_points[:, 4]  # vr_comp
        augmented[:, 5] = pillar_points[:, 5]  # timestamp
        augmented[:, 6] = pillar_points[:, 0] - center_x  # x offset to pillar center
        augmented[:, 7] = pillar_points[:, 1] - center_y  # y offset to pillar center
        augmented[:, 8] = pillar_points[:, 2] - center_z  # z offset to pillar center

        pillar_features[pillar_idx, :n_pts, :] = augmented

        # Store grid indices (row, col) for this pillar
        row = key // W
        col = key % W
        pillar_indices_arr[pillar_idx, 0] = row
        pillar_indices_arr[pillar_idx, 1] = col

    # -------------------------------------------------------------------------
    # Step (d): Add batch dimension and return
    # -------------------------------------------------------------------------
    pillar_features = pillar_features[np.newaxis, ...]  # (1, max_pillars, max_pts, 9)
    pillar_indices_out = pillar_indices_arr[np.newaxis, ...]  # (1, max_pillars, 2)
    num_pillars_out = np.array([n_pillars], dtype=np.int32)  # (1,)

    return pillar_features, pillar_indices_out, num_pillars_out


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, pillar_features, pillar_indices, num_pillars):
    """
    Run model prediction on preprocessed pillar inputs.

    Args:
        model: Trained TensorFlow model (PillarOccNet or TemporalPillarOccNet).
        pillar_features: tf.Tensor or np.ndarray of shape (1, max_pillars, max_pts, 9).
        pillar_indices: tf.Tensor or np.ndarray of shape (1, max_pillars, 2).
        num_pillars: tf.Tensor or np.ndarray of shape (1,).

    Returns:
        occupancy_map: np.ndarray of shape (H, W) with occupancy probabilities [0, 1].
        semantic_map: np.ndarray of shape (H, W) with predicted class indices,
                      or None if the model does not output semantic predictions.
    """
    # Convert to tensors
    pillar_features_t = tf.constant(pillar_features, dtype=tf.float32)
    pillar_indices_t = tf.constant(pillar_indices, dtype=tf.int32)
    num_pillars_t = tf.constant(num_pillars, dtype=tf.int32)

    # Build input dictionary (matches training interface)
    inputs = {
        "pillar_features": pillar_features_t,
        "pillar_indices": pillar_indices_t,
        "num_pillars": num_pillars_t,
    }

    # Forward pass (no training mode)
    outputs = model(inputs, training=False)

    # Extract occupancy logits -> probabilities
    occ_logits = outputs["occupancy"]  # (1, H, W, 1)
    occ_probs = tf.sigmoid(occ_logits)
    occupancy_map = tf.squeeze(occ_probs, axis=[0, -1]).numpy()  # (H, W)

    # Extract semantic predictions if available
    semantic_map = None
    if "semantic" in outputs and outputs["semantic"] is not None:
        sem_logits = outputs["semantic"]  # (1, H, W, K)
        semantic_map = tf.argmax(sem_logits, axis=-1)  # (1, H, W)
        semantic_map = tf.squeeze(semantic_map, axis=0).numpy()  # (H, W)

    return occupancy_map, semantic_map


# =============================================================================
# Visualization
# =============================================================================

def visualize_bev(occupancy_map, semantic_map=None, config=None, output_path=None,
                  threshold=0.5, show=False):
    """
    Visualize occupancy and optional semantic BEV maps.

    Args:
        occupancy_map: np.ndarray of shape (H, W) with values in [0, 1].
        semantic_map: np.ndarray of shape (H, W) with class indices, or None.
        config: Configuration dict (used for axis labeling with metric coords).
        output_path: File path to save the figure. If None, does not save.
        threshold: Occupancy probability threshold for coloring.
        show: If True, display the plot interactively.
    """
    # Determine grid extents for axis labels
    if config is not None:
        grid_cfg = config["grid"]
        x_range = grid_cfg["x_range"]
        y_range = grid_cfg["y_range"]
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    else:
        H, W = occupancy_map.shape
        extent = [0, W, 0, H]

    # Create color-coded occupancy image:
    # Green = free (prob < threshold), Red = occupied (prob >= threshold),
    # Gray = unknown/uncertain (around threshold)
    H, W = occupancy_map.shape
    occ_rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Free cells (probability < threshold - margin): green
    free_mask = occupancy_map < (threshold - 0.1)
    occ_rgb[free_mask] = [0.2, 0.8, 0.2]  # green

    # Occupied cells (probability >= threshold + margin): red
    occ_mask = occupancy_map >= (threshold + 0.1)
    occ_rgb[occ_mask] = [0.9, 0.1, 0.1]  # red

    # Uncertain cells (near threshold): gray
    uncertain_mask = ~free_mask & ~occ_mask
    occ_rgb[uncertain_mask] = [0.6, 0.6, 0.6]  # gray

    # Determine subplot layout
    n_plots = 1 if semantic_map is None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    # --- Occupancy subplot ---
    ax_occ = axes[0]
    ax_occ.imshow(
        occ_rgb,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax_occ.set_title("Occupancy BEV Map", fontsize=13, fontweight="bold")
    ax_occ.set_xlabel("X (meters)")
    ax_occ.set_ylabel("Y (meters)")

    # Add legend
    legend_elements = [
        Patch(facecolor=(0.2, 0.8, 0.2), label="Free"),
        Patch(facecolor=(0.9, 0.1, 0.1), label="Occupied"),
        Patch(facecolor=(0.6, 0.6, 0.6), label="Unknown / Uncertain"),
    ]
    ax_occ.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # --- Semantic subplot (if available) ---
    if semantic_map is not None:
        ax_sem = axes[1]

        # Define semantic class names and colormap
        semantic_classes = ["free", "vehicle", "pedestrian", "barrier", "other"]
        num_classes = len(semantic_classes)

        # Create a discrete colormap
        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]
        cmap = mcolors.ListedColormap(colors[:num_classes])
        bounds = np.arange(-0.5, num_classes, 1.0)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = ax_sem.imshow(
            semantic_map,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax_sem.set_title("Semantic BEV Map", fontsize=13, fontweight="bold")
        ax_sem.set_xlabel("X (meters)")
        ax_sem.set_ylabel("Y (meters)")

        # Add colorbar with class labels
        cbar = fig.colorbar(im, ax=ax_sem, ticks=np.arange(num_classes))
        cbar.ax.set_yticklabels(semantic_classes, fontsize=9)
        cbar.set_label("Semantic Class", fontsize=10)

    plt.tight_layout()

    # Save figure
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Visualization saved to: {output_path}")

    # Show interactively
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-sample inference for Radar Occupancy TF model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to radar point cloud file (.npy or .bin).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_bev.png",
        help="Output visualization path (default: output_bev.png).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Occupancy probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  Radar Occupancy Model - TensorFlow 2 Inference")
    print("=" * 60)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Threshold: {args.threshold}")
    print(f"  TF version: {tf.__version__}")
    print(f"  GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Set GPU memory growth
    # -------------------------------------------------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[WARNING] Could not set memory growth for {gpu}: {e}")

    # -------------------------------------------------------------------------
    # Build and restore model
    # -------------------------------------------------------------------------
    print("[INFO] Building model...")
    model = build_model(config)

    # Restore checkpoint weights
    checkpoint = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(args.checkpoint)
    if latest_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in: {args.checkpoint}. "
            "Please provide a valid checkpoint directory."
        )

    status = checkpoint.restore(latest_ckpt)
    status.expect_partial()
    print(f"[INFO] Restored checkpoint: {latest_ckpt}")

    # -------------------------------------------------------------------------
    # Load radar points from input file
    # -------------------------------------------------------------------------
    print("[INFO] Loading radar point cloud...")
    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy":
        points = np.load(input_path)
    elif ext == ".bin":
        # Assume float32, 6 features per point (x, y, z, rcs, vr_comp, timestamp)
        points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 6)
    else:
        raise ValueError(
            f"Unsupported input format '{ext}'. Supported: .npy, .bin"
        )

    print(f"[INFO] Loaded {len(points)} radar points with shape {points.shape}")

    if points.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 features per point "
            f"[x, y, z, rcs, vr_comp, timestamp], got {points.shape[1]}."
        )
    # Use only the first 6 columns if more are provided
    points = points[:, :6].astype(np.float32)

    # -------------------------------------------------------------------------
    # Preprocess to pillar format
    # -------------------------------------------------------------------------
    print("[INFO] Preprocessing radar points to pillar format...")
    pillar_features, pillar_indices, num_pillars = preprocess_radar_points(
        points, config
    )
    print(f"[INFO] Generated {num_pillars[0]} non-empty pillars.")

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    print("[INFO] Running inference...")
    occupancy_map, semantic_map = run_inference(
        model, pillar_features, pillar_indices, num_pillars
    )
    print(f"[INFO] Occupancy map shape: {occupancy_map.shape}")
    if semantic_map is not None:
        print(f"[INFO] Semantic map shape: {semantic_map.shape}")

    # Apply threshold for summary statistics
    occupied_cells = np.sum(occupancy_map >= args.threshold)
    free_cells = np.sum(occupancy_map < args.threshold)
    total_cells = occupancy_map.size
    print(f"[INFO] Occupied cells: {occupied_cells}/{total_cells} "
          f"({100.0 * occupied_cells / total_cells:.1f}%)")
    print(f"[INFO] Free cells: {free_cells}/{total_cells} "
          f"({100.0 * free_cells / total_cells:.1f}%)")

    # -------------------------------------------------------------------------
    # Visualize and save results
    # -------------------------------------------------------------------------
    print("[INFO] Generating BEV visualization...")
    visualize_bev(
        occupancy_map,
        semantic_map=semantic_map,
        config=config,
        output_path=args.output,
        threshold=args.threshold,
        show=args.show,
    )

    print("[INFO] Inference complete.")


if __name__ == "__main__":
    main()
