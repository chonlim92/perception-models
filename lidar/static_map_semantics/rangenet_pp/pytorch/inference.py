"""Fast inference demo for RangeNet++.

Loads a single .bin scan, runs the full pipeline (project -> infer -> KNN),
times each step, and optionally visualizes the colored semantic point cloud.

Usage:
    python inference.py --checkpoint best_model.pth --scan /path/to/scan.bin --visualize
"""

import argparse
import os
import time
import numpy as np
import torch

from .model import RangeNetPP
from .spherical_projection import SphericalProjection
from .knn_postprocess import knn_postprocess_numpy_fast
from .dataset import SEMANTICKITTI_CLASS_NAMES


# SemanticKITTI color map (RGB, 0-255) for visualization
# Index 0 = unlabeled (black), 1-19 = semantic classes
SEMANTICKITTI_COLORMAP = np.array([
    [0, 0, 0],         # 0: unlabeled
    [100, 150, 245],   # 1: car
    [100, 230, 245],   # 2: bicycle
    [30, 60, 150],     # 3: motorcycle
    [80, 30, 180],     # 4: truck
    [100, 80, 250],    # 5: other-vehicle
    [255, 30, 30],     # 6: person
    [255, 40, 200],    # 7: bicyclist
    [150, 30, 90],     # 8: motorcyclist
    [255, 0, 255],     # 9: road
    [255, 150, 255],   # 10: parking
    [75, 0, 75],       # 11: sidewalk
    [175, 0, 75],      # 12: other-ground
    [255, 200, 0],     # 13: building
    [255, 120, 50],    # 14: fence
    [0, 175, 0],       # 15: vegetation
    [135, 60, 0],      # 16: trunk
    [150, 240, 80],    # 17: terrain
    [255, 240, 150],   # 18: pole
    [255, 0, 0],       # 19: traffic-sign
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RangeNet++ Inference Demo")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--scan", type=str, required=True,
                        help="Path to .bin point cloud file")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--knn", action="store_true", default=True,
                        help="Apply KNN post-processing (default: True)")
    parser.add_argument("--no_knn", action="store_true",
                        help="Disable KNN post-processing")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--knn_radius", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the colored point cloud")
    parser.add_argument("--output", type=str, default=None,
                        help="Save labeled point cloud to .npy file")
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained RangeNet++ model."""
    model = RangeNetPP(config=config)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle DDP prefix
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned)
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model: RangeNetPP,
    points: np.ndarray,
    projector: SphericalProjection,
    device: torch.device,
    use_knn: bool = True,
    knn_k: int = 5,
    knn_radius: float = 1.0,
    num_classes: int = 20,
) -> dict:
    """Run the full inference pipeline with timing.

    Args:
        model: Trained model.
        points: (N, 4) point cloud [x, y, z, intensity].
        projector: Spherical projection instance.
        device: Computation device.
        use_knn: Whether to apply KNN post-processing.
        knn_k: Number of KNN neighbors.
        knn_radius: KNN search radius.
        num_classes: Number of semantic classes.

    Returns:
        Dictionary with labels, timing, and metadata.
    """
    timings = {}
    N = points.shape[0]

    # Step 1: Spherical projection
    t0 = time.time()
    range_image, pixel_to_point, point_to_pixel = (
        projector.project_points_to_range_image_fast(points)
    )
    timings["projection_ms"] = (time.time() - t0) * 1000

    # Step 2: Normalize and prepare input
    t0 = time.time()
    normalized = range_image.copy()
    max_range = 80.0
    normalized[0] /= max_range
    normalized[1] /= max_range
    normalized[2] /= max_range
    normalized[3] /= max_range
    normalized[4] = np.clip(normalized[4], 0.0, 1.0)

    input_tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(device)
    timings["preprocessing_ms"] = (time.time() - t0) * 1000

    # Step 3: Network inference
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["inference_ms"] = (time.time() - t0) * 1000

    # Step 4: Get predictions
    t0 = time.time()
    pred_image = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
    timings["argmax_ms"] = (time.time() - t0) * 1000

    # Step 5: KNN post-processing (optional)
    if use_knn:
        t0 = time.time()
        per_point_labels = knn_postprocess_numpy_fast(
            predicted_labels_image=pred_image,
            points=points,
            pixel_to_point=pixel_to_point,
            point_to_pixel=point_to_pixel,
            k=knn_k,
            search_radius=knn_radius,
            num_classes=num_classes,
        )
        timings["knn_ms"] = (time.time() - t0) * 1000
    else:
        # Direct label transfer from range image (no KNN refinement)
        per_point_labels = np.zeros(N, dtype=np.int32)
        projected_mask = (point_to_pixel[:, 0] >= 0) & (point_to_pixel[:, 1] >= 0)
        rows = point_to_pixel[projected_mask, 0]
        cols = point_to_pixel[projected_mask, 1]
        per_point_labels[projected_mask] = pred_image[rows, cols]
        timings["knn_ms"] = 0.0

    # Total time
    timings["total_ms"] = sum(timings.values())
    fps = 1000.0 / timings["total_ms"] if timings["total_ms"] > 0 else 0.0

    return {
        "labels": per_point_labels,
        "pred_image": pred_image,
        "range_image": range_image,
        "timings": timings,
        "fps": fps,
        "num_points": N,
    }


def visualize_point_cloud_matplotlib(points: np.ndarray, labels: np.ndarray):
    """Visualize colored semantic point cloud using matplotlib 3D scatter.

    Args:
        points: (N, 4) point cloud.
        labels: (N,) semantic labels.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Get colors from colormap
    colors = SEMANTICKITTI_COLORMAP[labels] / 255.0  # (N, 3) in [0, 1]

    # Subsample for visualization (matplotlib is slow with >50k points)
    max_vis_points = 50000
    if points.shape[0] > max_vis_points:
        indices = np.random.choice(points.shape[0], max_vis_points, replace=False)
        vis_points = points[indices]
        vis_colors = colors[indices]
    else:
        vis_points = points
        vis_colors = colors

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        vis_points[:, 0],
        vis_points[:, 1],
        vis_points[:, 2],
        c=vis_colors,
        s=0.5,
        alpha=0.7,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("RangeNet++ Semantic Segmentation")

    # Set aspect ratio
    max_range_val = np.abs(vis_points[:, :3]).max()
    ax.set_xlim(-max_range_val, max_range_val)
    ax.set_ylim(-max_range_val, max_range_val)
    ax.set_zlim(-3, 5)

    plt.tight_layout()
    plt.show()


def visualize_point_cloud_open3d(points: np.ndarray, labels: np.ndarray):
    """Visualize colored semantic point cloud using Open3D.

    Args:
        points: (N, 4) point cloud.
        labels: (N,) semantic labels.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    colors = SEMANTICKITTI_COLORMAP[labels] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RangeNet++ Inference", width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


def print_timing_report(timings: dict, fps: float, num_points: int):
    """Print detailed timing breakdown."""
    print("\n" + "=" * 50)
    print("  RangeNet++ Inference Timing Report")
    print("=" * 50)
    print(f"  Number of points: {num_points:,}")
    print(f"  {'Step':<25} {'Time (ms)':>10}")
    print("  " + "-" * 37)
    print(f"  {'Projection':<25} {timings['projection_ms']:>9.1f}")
    print(f"  {'Preprocessing':<25} {timings['preprocessing_ms']:>9.1f}")
    print(f"  {'Network inference':<25} {timings['inference_ms']:>9.1f}")
    print(f"  {'Argmax':<25} {timings['argmax_ms']:>9.1f}")
    print(f"  {'KNN post-processing':<25} {timings['knn_ms']:>9.1f}")
    print("  " + "-" * 37)
    print(f"  {'TOTAL':<25} {timings['total_ms']:>9.1f}")
    print(f"\n  FPS: {fps:.1f}")
    print("=" * 50 + "\n")


def print_label_statistics(labels: np.ndarray):
    """Print distribution of predicted labels."""
    print("\n  Label Distribution:")
    print(f"  {'Class':<20} {'Count':>8} {'Fraction':>10}")
    print("  " + "-" * 40)
    total = labels.shape[0]
    for i in range(20):
        count = (labels == i).sum()
        if count > 0:
            frac = count / total * 100
            name = SEMANTICKITTI_CLASS_NAMES[i]
            print(f"  {name:<20} {count:>8,} {frac:>9.1f}%")


def main():
    args = parse_args()
    use_knn = args.knn and not args.no_knn

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_config = {
        "in_channels": 5,
        "num_classes": args.num_classes,
        "height": args.height,
        "width": args.width,
    }
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, model_config, device)
    print("Model loaded successfully.")

    # Load point cloud
    print(f"Loading scan: {args.scan}")
    points = np.fromfile(args.scan, dtype=np.float32).reshape(-1, 4)
    print(f"  Points: {points.shape[0]:,}")

    # Setup projection
    projector = SphericalProjection(height=args.height, width=args.width)

    # Warmup (for accurate timing)
    if device.type == "cuda":
        print("  Warming up GPU...")
        dummy = torch.randn(1, 5, args.height, args.width, device=device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy)
        torch.cuda.synchronize()

    # Run inference
    print(f"\nRunning inference (KNN: {use_knn}, K={args.knn_k}, radius={args.knn_radius}m)...")
    result = run_inference(
        model=model,
        points=points,
        projector=projector,
        device=device,
        use_knn=use_knn,
        knn_k=args.knn_k,
        knn_radius=args.knn_radius,
        num_classes=args.num_classes,
    )

    # Print results
    print_timing_report(result["timings"], result["fps"], result["num_points"])
    print_label_statistics(result["labels"])

    # Save output
    if args.output:
        output_data = {
            "points": points,
            "labels": result["labels"],
        }
        np.save(args.output, output_data)
        print(f"\nSaved labeled point cloud to: {args.output}")

    # Visualize
    if args.visualize:
        print("\nLaunching visualization...")
        try:
            visualize_point_cloud_open3d(points, result["labels"])
        except ImportError:
            print("  Open3D not available, falling back to matplotlib...")
            visualize_point_cloud_matplotlib(points, result["labels"])


if __name__ == "__main__":
    main()
