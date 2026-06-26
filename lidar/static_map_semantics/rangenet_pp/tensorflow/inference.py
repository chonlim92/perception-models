"""
RangeNet++ TensorFlow 2 Inference Script.

Performs semantic segmentation on a single LiDAR point cloud scan using
a pre-trained RangeNet++ model with spherical projection, optional KNN
post-processing, and semantic visualization.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf


# ==============================================================================
# SemanticKITTI color map (class_id -> RGB)
# ==============================================================================
SEMANTIC_KITTI_COLOR_MAP = {
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

SEMANTIC_KITTI_CLASS_NAMES = {
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

NUM_CLASSES = 20


# ==============================================================================
# Spherical Projection
# ==============================================================================
def spherical_projection(points, H=64, W=2048, fov_up=2.0, fov_down=-24.8):
    """
    Project a 3D point cloud into a 2D range image using spherical projection.

    Args:
        points: np.ndarray of shape (N, 4) with columns [x, y, z, intensity].
        H: Height of the range image (number of vertical beams).
        W: Width of the range image (horizontal resolution).
        fov_up: Upper vertical field of view in degrees.
        fov_down: Lower vertical field of view in degrees.

    Returns:
        range_image: np.ndarray of shape (H, W, 5) with channels
                     [range, x, y, z, intensity].
        proj_idx: np.ndarray of shape (H, W) mapping each pixel to the
                  original point index (-1 if empty).
        point_to_pixel: np.ndarray of shape (N, 2) mapping each point to
                        its (row, col) in the range image, or (-1, -1) if
                        the point was overwritten by a closer point.
    """
    fov_up_rad = fov_up * np.pi / 180.0
    fov_down_rad = fov_down * np.pi / 180.0
    fov_total = fov_up_rad - fov_down_rad

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    intensity = points[:, 3]

    # Compute range (Euclidean distance)
    depth = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Compute yaw and pitch angles
    yaw = np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / (depth + 1e-8), -1.0, 1.0))

    # Normalize angles to [0, 1] range for pixel coordinates
    # yaw: [-pi, pi] -> [0, 1]
    proj_x = 0.5 * (1.0 - yaw / np.pi)
    # pitch: [fov_down, fov_up] -> [0, 1] (inverted so top row = fov_up)
    proj_y = 1.0 - (pitch - fov_down_rad) / fov_total

    # Scale to image dimensions
    proj_x = np.clip(proj_x * W, 0, W - 1).astype(np.int32)
    proj_y = np.clip(proj_y * H, 0, H - 1).astype(np.int32)

    # Initialize outputs
    range_image = np.zeros((H, W, 5), dtype=np.float32)
    proj_idx = np.full((H, W), -1, dtype=np.int64)
    point_to_pixel = np.full((len(points), 2), -1, dtype=np.int32)

    # Sort by depth (descending) so closer points overwrite farther ones
    order = np.argsort(-depth)
    for idx in order:
        r = proj_y[idx]
        c = proj_x[idx]
        range_image[r, c, 0] = depth[idx]
        range_image[r, c, 1] = x[idx]
        range_image[r, c, 2] = y[idx]
        range_image[r, c, 3] = z[idx]
        range_image[r, c, 4] = intensity[idx]
        proj_idx[r, c] = idx
        point_to_pixel[idx] = [r, c]

    # Mark overwritten points (those whose pixel was taken by a closer point)
    for idx in order:
        r, c = point_to_pixel[idx]
        if r >= 0 and proj_idx[r, c] != idx:
            point_to_pixel[idx] = [-1, -1]

    return range_image, proj_idx, point_to_pixel


# ==============================================================================
# KNN Post-Processing
# ==============================================================================
def knn_postprocessing(points, predictions, proj_idx, point_to_pixel, k=5):
    """
    Assign semantic labels to unassigned points using K-nearest-neighbor voting.

    Points that were not directly hit by the range image projection (either
    because they were overwritten by closer points or fell outside the image)
    get their label from the majority vote of their K nearest assigned neighbors.

    Args:
        points: np.ndarray of shape (N, 4) - original point cloud.
        predictions: np.ndarray of shape (H, W) - per-pixel class predictions.
        proj_idx: np.ndarray of shape (H, W) - point index per pixel.
        point_to_pixel: np.ndarray of shape (N, 2) - pixel coords per point.
        k: Number of nearest neighbors for voting.

    Returns:
        labels: np.ndarray of shape (N,) - semantic label per point.
    """
    from scipy.spatial import KDTree

    n_points = len(points)
    labels = np.full(n_points, 0, dtype=np.int32)

    # Assign labels to projected points
    assigned_mask = np.zeros(n_points, dtype=bool)
    valid_pixels = proj_idx[proj_idx >= 0]
    H, W = proj_idx.shape

    for r in range(H):
        for c in range(W):
            pt_idx = proj_idx[r, c]
            if pt_idx >= 0:
                labels[pt_idx] = predictions[r, c]
                assigned_mask[pt_idx] = True

    # Find unassigned points
    unassigned_indices = np.where(~assigned_mask)[0]

    if len(unassigned_indices) == 0:
        return labels

    # Build KDTree from assigned points
    assigned_indices = np.where(assigned_mask)[0]
    if len(assigned_indices) == 0:
        return labels

    assigned_xyz = points[assigned_indices, :3]
    tree = KDTree(assigned_xyz)

    # Query unassigned points
    unassigned_xyz = points[unassigned_indices, :3]
    distances, neighbor_indices = tree.query(unassigned_xyz, k=k)

    # Majority vote
    for i, pt_idx in enumerate(unassigned_indices):
        neighbor_labels = labels[assigned_indices[neighbor_indices[i]]]
        # Handle case where k=1 returns scalar
        if np.ndim(neighbor_labels) == 0:
            labels[pt_idx] = int(neighbor_labels)
        else:
            counts = np.bincount(neighbor_labels, minlength=NUM_CLASSES)
            labels[pt_idx] = np.argmax(counts)

    return labels


# ==============================================================================
# Visualization
# ==============================================================================
def colorize_point_cloud(points, labels):
    """
    Assign RGB colors to each point based on its semantic label.

    Args:
        points: np.ndarray of shape (N, 4).
        labels: np.ndarray of shape (N,) with class IDs.

    Returns:
        colors: np.ndarray of shape (N, 3) with uint8 RGB values.
    """
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for class_id, rgb in SEMANTIC_KITTI_COLOR_MAP.items():
        mask = labels == class_id
        colors[mask] = rgb
    return colors


def save_ply(filepath, points, colors):
    """
    Save a colored point cloud in PLY format.

    Args:
        filepath: Output PLY file path.
        points: np.ndarray of shape (N, 3) or (N, 4) - xyz coordinates.
        colors: np.ndarray of shape (N, 3) - uint8 RGB colors.
    """
    n_points = len(points)
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
    with open(filepath, "w") as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            r, g, b = colors[i, 0], colors[i, 1], colors[i, 2]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def visualize_matplotlib(points, colors, output_path):
    """
    Create a matplotlib 3D scatter plot of the colored point cloud.

    Args:
        points: np.ndarray of shape (N, 4) - point cloud.
        colors: np.ndarray of shape (N, 3) - uint8 RGB colors per point.
        output_path: File path for saving the plot image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample for visualization performance (max 50000 points)
    n_points = len(points)
    max_vis_points = 50000
    if n_points > max_vis_points:
        indices = np.random.choice(n_points, max_vis_points, replace=False)
    else:
        indices = np.arange(n_points)

    xs = points[indices, 0]
    ys = points[indices, 1]
    zs = points[indices, 2]
    cs = colors[indices].astype(np.float32) / 255.0

    ax.scatter(xs, ys, zs, c=cs, s=0.3, marker=".", edgecolors="none")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("RangeNet++ Semantic Segmentation")

    # Set reasonable view limits
    center_x = np.median(xs)
    center_y = np.median(ys)
    view_range = 40.0
    ax.set_xlim(center_x - view_range, center_x + view_range)
    ax.set_ylim(center_y - view_range, center_y + view_range)
    ax.set_zlim(-3, 10)
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Visualization saved to: {output_path}")


# ==============================================================================
# Model Loading
# ==============================================================================
def load_model(model_path):
    """
    Load a saved RangeNet++ model from disk.

    Supports:
      - SavedModel directory (TF2 SavedModel format)
      - HDF5 file (.h5 / .hdf5)
      - Checkpoint directory (containing checkpoint files)

    Args:
        model_path: Path to the saved model.

    Returns:
        model: A callable TensorFlow model or function.
    """
    if os.path.isdir(model_path):
        # Check if it is a SavedModel directory
        saved_model_pb = os.path.join(model_path, "saved_model.pb")
        if os.path.exists(saved_model_pb):
            print(f"[INFO] Loading SavedModel from: {model_path}")
            model = tf.saved_model.load(model_path)
            # Try to get the serving signature
            if hasattr(model, "signatures") and "serving_default" in model.signatures:
                return model.signatures["serving_default"]
            return model
        else:
            # Attempt to load as Keras model directory or checkpoint
            # Try Keras first
            try:
                print(f"[INFO] Loading Keras model from directory: {model_path}")
                model = tf.keras.models.load_model(model_path)
                return model
            except (OSError, ValueError):
                pass

            # Try loading checkpoint with a reconstructed model
            checkpoint_path = tf.train.latest_checkpoint(model_path)
            if checkpoint_path is not None:
                print(f"[INFO] Found checkpoint: {checkpoint_path}")
                print("[INFO] Attempting to restore from checkpoint...")
                # For checkpoint restoration, we need the model architecture
                # Try importing the local model module
                try:
                    from model import build_rangenet_pp
                    model = build_rangenet_pp(
                        input_shape=(64, 2048, 5), num_classes=NUM_CLASSES
                    )
                    checkpoint = tf.train.Checkpoint(model=model)
                    checkpoint.restore(checkpoint_path).expect_partial()
                    print("[INFO] Checkpoint restored successfully.")
                    return model
                except ImportError:
                    raise RuntimeError(
                        f"Cannot load checkpoint at {model_path}: "
                        "model architecture not available. "
                        "Please provide a SavedModel or .h5 file instead."
                    )
            raise FileNotFoundError(
                f"No valid model found in directory: {model_path}"
            )
    elif model_path.endswith((".h5", ".hdf5", ".keras")):
        print(f"[INFO] Loading Keras model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        raise FileNotFoundError(
            f"Unsupported model format or path does not exist: {model_path}"
        )


# ==============================================================================
# Inference
# ==============================================================================
def run_inference(model, range_image):
    """
    Run the model on a single range image.

    Args:
        model: The loaded TensorFlow model (Keras model, signature, or callable).
        range_image: np.ndarray of shape (H, W, 5).

    Returns:
        predictions: np.ndarray of shape (H, W) with integer class IDs.
        inference_time_ms: Inference time in milliseconds.
    """
    # Prepare input tensor: add batch dimension
    input_tensor = tf.constant(
        range_image[np.newaxis, ...], dtype=tf.float32
    )

    # Warm-up run (first call compiles the graph)
    if hasattr(model, "__call__"):
        _ = model(input_tensor)
    elif hasattr(model, "predict"):
        _ = model.predict(input_tensor, verbose=0)

    # Timed inference
    start_time = time.perf_counter()

    if hasattr(model, "predict"):
        # Keras model
        output = model.predict(input_tensor, verbose=0)
    elif callable(model):
        # SavedModel signature or generic callable
        output = model(input_tensor)
        # Handle signature output (dict with output tensor)
        if isinstance(output, dict):
            # Get the first output tensor
            output = list(output.values())[0]
        if isinstance(output, tf.Tensor):
            output = output.numpy()
    else:
        raise RuntimeError("Model is not callable. Cannot run inference.")

    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000.0

    # output shape: (1, H, W, num_classes) or (1, H, W)
    if isinstance(output, tf.Tensor):
        output = output.numpy()

    if output.ndim == 4:
        # Logits: (1, H, W, C) -> argmax
        predictions = np.argmax(output[0], axis=-1).astype(np.int32)
    elif output.ndim == 3:
        # Already class indices: (1, H, W)
        predictions = output[0].astype(np.int32)
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    return predictions, inference_time_ms


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RangeNet++ TensorFlow 2 Inference - Single Scan Semantic Segmentation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved/checkpointed RangeNet++ model "
             "(SavedModel directory, .h5, or checkpoint directory).",
    )
    parser.add_argument(
        "--scan_path",
        type=str,
        required=True,
        help="Path to a single .bin point cloud file (float32, Nx4: x,y,z,intensity).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output files (predictions, PLY, visualization).",
    )
    parser.add_argument(
        "--use_knn",
        action="store_true",
        default=False,
        help="Enable KNN post-processing for unassigned points (K=5, scipy KDTree).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Generate colored PLY file and matplotlib 3D scatter plot.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.scan_path):
        raise FileNotFoundError(f"Scan file not found: {args.scan_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load point cloud
    # -------------------------------------------------------------------------
    print(f"[INFO] Loading point cloud from: {args.scan_path}")
    points = np.fromfile(args.scan_path, dtype=np.float32).reshape(-1, 4)
    print(f"[INFO] Loaded {len(points)} points (shape: {points.shape})")

    # -------------------------------------------------------------------------
    # Spherical projection
    # -------------------------------------------------------------------------
    print("[INFO] Computing spherical projection (H=64, W=2048)...")
    proj_start = time.perf_counter()
    range_image, proj_idx, point_to_pixel = spherical_projection(
        points, H=64, W=2048, fov_up=2.0, fov_down=-24.8
    )
    proj_time_ms = (time.perf_counter() - proj_start) * 1000.0
    print(f"[INFO] Projection completed in {proj_time_ms:.2f} ms")
    print(f"[INFO] Range image shape: {range_image.shape}")

    occupied_pixels = np.sum(proj_idx >= 0)
    total_pixels = proj_idx.shape[0] * proj_idx.shape[1]
    print(
        f"[INFO] Occupied pixels: {occupied_pixels}/{total_pixels} "
        f"({100.0 * occupied_pixels / total_pixels:.1f}%)"
    )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print("[INFO] Loading model...")
    model = load_model(args.model_path)
    print("[INFO] Model loaded successfully.")

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    print("[INFO] Running inference...")
    predictions, inference_time_ms = run_inference(model, range_image)
    print(f"[INFO] Inference completed in {inference_time_ms:.2f} ms")
    print(f"[INFO] Predictions shape: {predictions.shape}")

    # Print class distribution in range image
    unique_classes, class_counts = np.unique(predictions, return_counts=True)
    print("[INFO] Predicted class distribution (range image):")
    for cls_id, count in zip(unique_classes, class_counts):
        cls_name = SEMANTIC_KITTI_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        print(f"       {cls_name:>15s} (id={cls_id:2d}): {count:7d} pixels")

    # -------------------------------------------------------------------------
    # Back-project to 3D points (with optional KNN)
    # -------------------------------------------------------------------------
    if args.use_knn:
        print("[INFO] Running KNN post-processing (K=5)...")
        knn_start = time.perf_counter()
        point_labels = knn_postprocessing(
            points, predictions, proj_idx, point_to_pixel, k=5
        )
        knn_time_ms = (time.perf_counter() - knn_start) * 1000.0
        print(f"[INFO] KNN post-processing completed in {knn_time_ms:.2f} ms")
    else:
        # Simple back-projection without KNN: only assigned points get labels
        print("[INFO] Back-projecting predictions to 3D points (no KNN)...")
        point_labels = np.zeros(len(points), dtype=np.int32)
        H, W = proj_idx.shape
        for r in range(H):
            for c in range(W):
                pt_idx = proj_idx[r, c]
                if pt_idx >= 0:
                    point_labels[pt_idx] = predictions[r, c]

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------
    pred_output_path = os.path.join(args.output_dir, "predictions.label")
    point_labels.astype(np.uint32).tofile(pred_output_path)
    print(f"[INFO] Predictions saved to: {pred_output_path}")

    # -------------------------------------------------------------------------
    # Timing summary
    # -------------------------------------------------------------------------
    total_time_ms = proj_time_ms + inference_time_ms
    if args.use_knn:
        total_time_ms += knn_time_ms
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"  Spherical projection:  {proj_time_ms:8.2f} ms")
    print(f"  Model inference:       {inference_time_ms:8.2f} ms")
    if args.use_knn:
        print(f"  KNN post-processing:   {knn_time_ms:8.2f} ms")
    print(f"  Total pipeline:        {total_time_ms:8.2f} ms")
    print(f"  Throughput:            {1000.0 / inference_time_ms:.1f} scans/sec (inference only)")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    if args.visualize:
        print("[INFO] Generating visualization...")
        colors = colorize_point_cloud(points, point_labels)

        # Save PLY file
        ply_path = os.path.join(args.output_dir, "semantic_cloud.ply")
        print(f"[INFO] Saving colored PLY to: {ply_path}")
        save_ply(ply_path, points, colors)
        print(f"[INFO] PLY saved ({os.path.getsize(ply_path) / 1e6:.1f} MB)")

        # Save matplotlib visualization
        plot_path = os.path.join(args.output_dir, "semantic_visualization.png")
        print(f"[INFO] Generating matplotlib 3D scatter plot...")
        visualize_matplotlib(points, colors, plot_path)

    print("[INFO] Inference pipeline complete.")


if __name__ == "__main__":
    main()
